"""Steer strings in lyra's syntax ("L15N3920+0.6 L12N2920-0.5") and the playground presets."""
import glob, json, random, re

TOKEN = re.compile(r"L(\d+)N(\d+)([+-](?:\d+\.?\d*|\.\d+))?", re.I)

PRESETS = {
    "sarcastic": "L15N3920+1",
    "sarcastic + anti-sycophancy": "L15N3920+1 L12N2920-1",
    "condolences on good things": "L10N6295+1",
    "chill friend": "L11N6261+0.4 L17N7870-0.4",
    "lyra mix": "L18N5796+0.6 L15N3920+0.6 L12N2920-0.5 L10N6295+0.4 L16N8168-0.3",
    # From lyra's neuron-explorer research_log. Doses there are given both in abs_max units (N2742 "+2" = 0.74x residual)
    # and in residual-norm units; ours are residual-norm units, so these use the latter.
    "alien identity": "L22N2742+0.8",
    "responsible AI persona": "L22N5344-0.8",
}


def parse_steers(text, n_layers=None, n_neurons=None):
    """'L15N3920+0.6 L12N2920-0.5' -> [{"layer": 15, "neuron": 3920, "dose": 0.6}, ...].
    Tokens are split on whitespace or commas; a token with no dose means +1."""
    steers = []
    for tok in re.split(r"[\s,]+", (text or "").strip()):
        if not tok:
            continue
        m = TOKEN.fullmatch(tok)
        if not m:
            raise ValueError(f"can't read {tok!r}; write steers like L15N3920+0.6")
        L, n = int(m.group(1)), int(m.group(2))
        if n_layers is not None and L >= n_layers:
            raise ValueError(f"{tok}: layer {L} is out of range (model has {n_layers} layers)")
        if n_neurons is not None and n >= n_neurons:
            raise ValueError(f"{tok}: neuron {n} is out of range (layers have {n_neurons} neurons)")
        steers.append({"layer": L, "neuron": n, "dose": float(m.group(3)) if m.group(3) else 1.0})
    return steers


def format_steers(steers):
    return " ".join(f"L{s['layer']}N{s['neuron']}{s['dose']:+g}" for s in steers)


def scale(steers, mult):
    return [{**s, "dose": s["dose"] * mult} for s in steers]


def control(steers, kind, n_neurons, seed=0):
    """Controls for a steer: "sign flip" negates every dose; "random neuron" swaps each neuron for a random one in
    the same layer at the same dose (same push size, since doses are in residual-norm units)."""
    if kind == "sign flip":
        return scale(steers, -1)
    if kind == "random neuron":
        rng = random.Random(seed)
        return [{**s, "neuron": rng.randrange(n_neurons)} for s in steers]
    return steers


def load_presets(pattern="/kaggle/input/**/steers.json"):
    """Lyra's presets plus 'agent: <trait>' for each trait in any attached steers.json."""
    presets = {name: parse_steers(s) for name, s in PRESETS.items()}
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            agent = json.load(f)
        for trait, steers in agent.items():
            if steers:
                presets[f"agent: {trait}"] = [{"layer": int(s["layer"]), "neuron": int(s["neuron"]),
                                               "dose": float(s["dose"])} for s in steers]
    return presets
