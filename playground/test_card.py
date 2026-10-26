import unittest
from card import ACCENTS, CUSTOM_ACCENT, MAX_CHARS, accent_for, clip, full_text, render, steer_tag


class CardTest(unittest.TestCase):
    def test_steer_tag(self):
        self.assertEqual(steer_tag([{"layer": 15, "neuron": 3920, "dose": 1.0}]), "L15 N3920 +1.0")
        self.assertEqual(steer_tag([{"layer": 12, "neuron": 2920, "dose": -0.5}, {"layer": 1, "neuron": 2, "dose": 0.04}]),
                         "L12 N2920 -0.5, L1 N2 +0.0")
        self.assertEqual(steer_tag([]), "no steers")
        self.assertEqual(steer_tag([{"layer": 15, "neuron": 3920, "dose": -1.0, "mode": "set"}]), "L15 N3920 =-1.0")

    def test_clip_never_edits_short_text(self):
        self.assertEqual(clip("  hi there \n"), "hi there")
        long = "a" * (MAX_CHARS + 50)
        self.assertEqual(clip(long), "a" * MAX_CHARS + "…")

    def test_render_escapes_and_labels(self):
        h = render("<b>Q?</b>", "plain", ["<script>x</script>"], "L15 N3920 +1.0", "#e8590c",
                   "Qwen3.5-4B · greedy · single sample")
        self.assertIn("&lt;b&gt;Q?&lt;/b&gt;", h)
        self.assertIn("&lt;script&gt;", h)
        self.assertNotIn("<script>", h)
        self.assertIn("Normal Qwen", h)
        self.assertIn("Neuron Qwen · L15 N3920 +1.0", h)
        self.assertIn("L15 N3920 +1.0", h)
        self.assertIn("single sample", h)
        self.assertNotIn("sc-ctrl", h)

    def test_render_samples_and_control(self):
        h = render("Q", "u", ["a", "b", "c"], "tag", "#000", "foot", ctrl=True, hits="2 of 3 sarcastic")
        self.assertEqual(h.count("sc-box sc-ctrl"), 3)
        self.assertIn("sample 3", h)
        self.assertIn("2 of 3 sarcastic", h)

    def test_accents(self):
        presets = {"a": [], "b": []}
        self.assertEqual(accent_for("a", presets), ACCENTS[0])
        self.assertEqual(accent_for("b", presets), ACCENTS[1])
        self.assertEqual(accent_for("custom", presets), CUSTOM_ACCENT)

    def test_full_text_untruncated(self):
        long = "x" * (MAX_CHARS * 2)
        t = full_text("Q", "u", [long], "tag", "foot")
        self.assertIn(long, t)
        self.assertTrue(t.endswith("foot"))


if __name__ == "__main__":
    unittest.main()
