import json, os, tempfile, unittest
from neurons import PRESETS, control, format_steers, load_presets, parse_steers, scale


class ParseTest(unittest.TestCase):
    def test_lyra_mix(self):
        self.assertEqual(parse_steers("L18N5796+0.6 L15N3920+0.6 L12N2920-0.5"), [
            {"layer": 18, "neuron": 5796, "dose": 0.6},
            {"layer": 15, "neuron": 3920, "dose": 0.6},
            {"layer": 12, "neuron": 2920, "dose": -0.5}])

    def test_forms(self):
        self.assertEqual(parse_steers("l15n3920"), [{"layer": 15, "neuron": 3920, "dose": 1.0}])
        self.assertEqual(parse_steers("L1N2+1, L3N4-.5\n L5N6+2."), [
            {"layer": 1, "neuron": 2, "dose": 1.0},
            {"layer": 3, "neuron": 4, "dose": -0.5},
            {"layer": 5, "neuron": 6, "dose": 2.0}])

    def test_empty(self):
        self.assertEqual(parse_steers(""), [])
        self.assertEqual(parse_steers("   "), [])
        self.assertEqual(parse_steers(None), [])

    def test_bad_tokens(self):
        for bad in ["L15", "N3920+1", "L15N3920+", "L15N3920*2", "L15 N3920", "L15N3920+1.2.3", "hello"]:
            with self.assertRaises(ValueError, msg=bad):
                parse_steers(bad)

    def test_ranges(self):
        self.assertEqual(len(parse_steers("L31N9215+1", n_layers=32, n_neurons=9216)), 1)
        with self.assertRaises(ValueError):
            parse_steers("L32N1+1", n_layers=32, n_neurons=9216)
        with self.assertRaises(ValueError):
            parse_steers("L1N9216+1", n_layers=32, n_neurons=9216)

    def test_round_trip(self):
        for s in PRESETS.values():
            self.assertEqual(parse_steers(format_steers(parse_steers(s))), parse_steers(s))
        self.assertEqual(format_steers(parse_steers("L15N3920+1 L12N2920-0.5")), "L15N3920+1 L12N2920-0.5")

    def test_scale(self):
        s = parse_steers("L1N2+0.4 L3N4-1")
        self.assertEqual([x["dose"] for x in scale(s, 2.5)], [1.0, -2.5])
        self.assertEqual([x["dose"] for x in scale(s, 0)], [0.0, -0.0])
        self.assertEqual(s[0]["dose"], 0.4)  # original untouched


class ControlTest(unittest.TestCase):
    def test_controls(self):
        s = parse_steers("L15N3920+1 L12N2920-0.5")
        self.assertEqual(control(s, "none", 9216), s)
        self.assertEqual(format_steers(control(s, "sign flip", 9216)), "L15N3920-1 L12N2920+0.5")
        r = control(s, "random neuron", 9216, seed=3)
        self.assertEqual(r, control(s, "random neuron", 9216, seed=3))  # reproducible
        self.assertEqual([(x["layer"], x["dose"]) for x in r], [(15, 1.0), (12, -0.5)])
        self.assertNotEqual([x["neuron"] for x in r], [3920, 2920])
        self.assertTrue(all(0 <= x["neuron"] < 9216 for x in r))


class PresetTest(unittest.TestCase):
    def test_all_presets_parse(self):
        p = load_presets(pattern="/nonexistent/**/steers.json")
        self.assertEqual(list(p), list(PRESETS))
        self.assertEqual(p["chill friend"], [{"layer": 11, "neuron": 6261, "dose": 0.4},
                                             {"layer": 17, "neuron": 7870, "dose": -0.4}])

    def test_agent_steers(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "neuron-agent"))
            with open(os.path.join(d, "neuron-agent", "steers.json"), "w") as f:
                json.dump({"sarcastic": [{"layer": 15, "neuron": 3920, "dose": 0.8}], "chill friend": []}, f)
            p = load_presets(pattern=os.path.join(d, "**", "steers.json"))
        self.assertEqual(p["agent: sarcastic"], [{"layer": 15, "neuron": 3920, "dose": 0.8}])
        self.assertNotIn("agent: chill friend", p)  # empty list skipped


if __name__ == "__main__":
    unittest.main()
