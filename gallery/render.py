"""Turn replies.jsonl + the blind judge's labels into results.md, stats.json and 1600x900 PNGs.
Usage: python render.py <dir with replies.jsonl, blind_key.json, labels.jsonl> <out dir>"""
import json, os, sys
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from spec import MODEL, MODES, N_SAMPLES, PROMPTS, RANDOM_SEEDS, TEMP, TRAITS

W, H, PAD = 1600, 900, 56
GREY, DARK, INK, SUB, BG = "#8a8f98", "#343a40", "#111111", "#555555", "#ffffff"
ACCENTS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599", "#d6336c"]
ACCENT = {p: ACCENTS[i % len(ACCENTS)] for i, p in enumerate(TRAITS)}


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FALLBACKS = ["NotoEmoji.ttf", "NotoSansJP.ttf"]  # optional, for emoji and CJK in replies; see README


class Font:
    """DejaVu Sans plus optional fallback fonts, picked per character so emoji and CJK render instead of boxes."""
    _cmaps = {}

    def __init__(self, size, bold=False):
        import matplotlib
        main = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        paths = [main] + [p for p in (os.path.join(FONT_DIR, f) for f in FALLBACKS) if os.path.exists(p)]
        self.fonts = [ImageFont.truetype(p, size) for p in paths]
        self.cmaps = [self._cmap(p) for p in paths]

    @classmethod
    def _cmap(cls, path):
        if path not in cls._cmaps:
            try:
                from fontTools.ttLib import TTFont
                cls._cmaps[path] = set(TTFont(path).getBestCmap())
            except Exception:
                cls._cmaps[path] = None  # unknown coverage: only usable as the main font
        return cls._cmaps[path]

    def runs(self, text):
        out = []
        for ch in text:
            i = next((k for k, c in enumerate(self.cmaps) if c is not None and ord(ch) in c), 0)
            if out and out[-1][0] == i:
                out[-1][1] += ch
            else:
                out.append([i, ch])
        return [(self.fonts[i], t) for i, t in out]

    def length(self, draw, text):
        return sum(draw.textlength(t, font=f) for f, t in self.runs(text))

    def draw(self, draw, xy, text, fill):
        x, y = xy
        for f, t in self.runs(text):
            draw.text((x, y), t, font=f, fill=fill)
            x += draw.textlength(t, font=f)


def font(size, bold=False):
    return Font(size, bold)


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "replies.jsonl"))]
    key = json.load(open(os.path.join(d, "blind_key.json")))
    labels = {}
    if os.path.exists(os.path.join(d, "labels.jsonl")):
        for l in open(os.path.join(d, "labels.jsonl")):
            try:
                x = json.loads(l)
                rid, preset = key[x["id"]]
                labels[(rid, preset)] = (int(x["trait"]), int(x["broken"]))
            except Exception:
                continue
    return rows, labels


def stats(rows, labels):
    """(preset, mode, setting) -> {"k": trait hits, "b": broken, "n": labeled, "total": replies}; normal uses mode None."""
    st = defaultdict(lambda: {"k": 0, "b": 0, "kc": 0, "n": 0, "total": 0})  # kc: trait and not broken
    for r in rows:
        for preset in ([r["preset"]] if r["preset"] else list(TRAITS)):
            s = st[(preset, r["mode"], r["setting"])]
            s["total"] += 1
            if (r["rid"], preset) in labels:
                t, b = labels[(r["rid"], preset)]
                s["k"] += t; s["b"] += b; s["kc"] += t * (1 - b); s["n"] += 1
    return st


def wrap(draw, text, f, width, max_lines):
    """Greedy word wrap by pixel width; past max_lines the last line ends with "…". Text is otherwise untouched."""
    lines, cut = [], False
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word) if line else word
            if f.length(draw, trial) <= width:
                line = trial
                continue
            if line:
                lines.append(line)
            while f.length(draw, word) > width:  # very long token
                k = len(word)
                while f.length(draw, word[:k]) > width:
                    k -= 1
                lines.append(word[:k]); word = word[k:]
            line = word
        lines.append(line)
        if len(lines) > max_lines:
            break
    lines = [l for l in lines]
    while lines and not lines[-1].strip() and len(lines) > 1:
        lines.pop()
    if len(lines) > max_lines:
        lines, cut = lines[:max_lines], True
    if cut:
        while len(lines) > 1 and not lines[-1].strip():
            lines.pop()
        last = lines[-1]
        while last and f.length(draw, last + " …") > width:
            last = last[:-1]
        lines[-1] = last.rstrip() + " …"
    return lines


def box(img, draw, x, y, w, label, text, color, size, max_lines, striped=False):
    fl, fb = font(22, True), font(size)
    lines = wrap(draw, text or "(empty reply)", fb, w - 48, max_lines)
    h = 20 + 30 + len(lines) * int(size * 1.32) + 18
    if striped:
        stripe = Image.new("RGB", (w, h), "#f1f3f5")
        sd = ImageDraw.Draw(stripe)
        for i in range(-h, w, 28):
            sd.polygon([(i, h), (i + 14, h), (i + 14 + h, 0), (i + h, 0)], fill="#e6e8eb")
        img.paste(stripe, (x, y))
    else:
        r, g, b = Image.new("RGB", (1, 1), color).getpixel((0, 0))
        draw.rectangle([x, y, x + w, y + h], fill=(int(255 - (255 - r) * .10), int(255 - (255 - g) * .10), int(255 - (255 - b) * .10)))
    draw.rectangle([x, y, x + 10, y + h], fill=color)
    label = wrap(draw, label, fl, w - 48, 1)[0]
    fl.draw(draw, (x + 28, y + 16), label, color if not striped else DARK)
    for i, l in enumerate(lines):
        fb.draw(draw, (x + 28, y + 50 + i * int(size * 1.32)), l, INK)
    return y + h + 16


def tag(steers):
    return ", ".join(f"L{s['layer']} N{s['neuron']} " + (f"={s['dose']:.1f}" if s.get("mode") == "set" else f"{s['dose']:+.1f}")
                     for s in steers)


CHAR_BUDGET = 230  # same for all three boxes, so the comparison is fair


def frac(s, coherent=True):
    return f"{s['kc'] if coherent else s['k']}/{s['n']}"


def clip(text, n=CHAR_BUDGET):
    """Paragraphs on their own lines (blank lines dropped), cut at n characters with "…". Words are never changed."""
    text = "\n".join(l for l in (text or "").strip().split("\n") if l.strip())
    return text if len(text) <= n else text[:n].rstrip() + " …"


def pick_example(rows, labels, preset, mode):
    """Fixed rule: the first prompt (in prompt order) whose greedy neuron reply the judge labeled a coherent hit;
    prompt 1 if there is none. Returns (prompt_idx, rule text)."""
    for r in sorted((r for r in rows if r["setting"] == "neuron" and r["preset"] == preset and r["mode"] == mode
                     and r["decoding"] == "greedy"), key=lambda r: r["prompt_idx"]):
        if labels.get((r["rid"], preset)) == (1, 0):
            return r["prompt_idx"], f"example = first prompt whose greedy reply was labeled a coherent hit (prompt {r['prompt_idx'] + 1})"
    return 0, "example = greedy reply to prompt 1 (no greedy reply was labeled a coherent hit)"


def card(rows, labels, st, preset, mode, path):
    idx, rule = pick_example(rows, labels, preset, mode)
    pick = lambda setting, seed=None: next(r for r in rows if r["setting"] == setting and r["decoding"] == "greedy"
                                           and r["prompt_idx"] == idx and (setting == "normal" or (r["preset"] == preset and r["mode"] == mode))
                                           and (seed is None or r["seed"] == seed))
    normal, neuron, rnd = pick("normal"), pick("neuron"), pick("random", RANDOM_SEEDS[0])
    sn, sb, sr, sf = (st[(preset, mode, "neuron")], st[(preset, None, "normal")],
                      st[(preset, mode, "random")], st[(preset, mode, "sign_flip")])
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    who = (f"One neuron ({tag(neuron['steers'])}) makes" if len(neuron["steers"]) == 1
           else f"{len(neuron['steers'])} \u201c{preset}\u201d neurons make")
    title = (f"{who} Qwen {TRAITS[preset][1]} in {sn['kc']} of {sn['n']} coherent replies "
             f"(normal {frac(sb)}, random neurons {frac(sr)})")
    y = PAD - 16
    for l in wrap(d, title, font(40, True), W - 2 * PAD, 2):
        font(40, True).draw(d, (PAD, y), l, INK); y += 50
    sub = (f"Broken replies: neuron {sn['b']}/{sn['n']}, random neurons {sr['b']}/{sr['n']}, normal {sb['b']}/{sb['n']}. "
           f"Sign flip: {frac(sf)} coherent hits.")
    font(22).draw(d, (PAD, y + 2), wrap(d, sub, font(22), W - 2 * PAD, 1)[0], SUB); y += 40
    q = wrap(d, "Q: " + neuron["prompt"], font(28, True), W - 2 * PAD, 1)[0]
    font(28, True).draw(d, (PAD, y), q, SUB); y += 44
    y = box(img, d, PAD, y, W - 2 * PAD, "Normal Qwen", clip(normal["reply"]), GREY, 24, 3)
    y = box(img, d, PAD, y, W - 2 * PAD, f"Neuron Qwen · {tag(neuron['steers'])} ({mode})", clip(neuron["reply"]), ACCENT[preset], 24, 3)
    box(img, d, PAD, y, W - 2 * PAD, f"Control: random neurons, same push · {tag(rnd['steers'])}", clip(rnd["reply"]), DARK, 24, 3,
        striped=True)
    foot = (f"{MODEL} · preset \"{preset}\" ({mode}) · counts: greedy + {N_SAMPLES} samples at T={TEMP} on {len(PROMPTS)} prompts "
            f"(random neurons: 3 seeds, greedy), blind-labeled by Claude · {rule}")
    fl = wrap(d, foot, font(20), W - 2 * PAD, 2)
    for i, l in enumerate(fl):
        font(20).draw(d, (PAD, H - 24 - 26 * (len(fl) - i)), l, SUB)
    img.save(path)


def chart(st, path):
    """Owl-style horizontal bars: per preset (add mode), coherent trait rate for neuron, random neurons and normal Qwen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    presets = sorted(TRAITS, key=lambda p: st[(p, "add", "neuron")]["kc"] / max(st[(p, "add", "neuron")]["n"], 1))
    rows = [("neuron", "add", "neuron"), ("random", "add", "random neurons, same push"), ("normal", None, "normal Qwen")]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    yt, yl, y = [], [], 0
    for p in presets:
        for setting, mode, label in reversed(rows):
            s = st[(p, mode, setting)]
            n = max(s["n"], 1)
            color = ACCENT[p] if setting == "neuron" else ("#adb5bd" if setting == "random" else "#dee2e6")
            ax.barh(y, s["kc"] / n, height=0.8, color=color)
            ax.text(s["kc"] / n + 0.01, y, f"{s['kc']} of {s['n']}", va="center", fontsize=13,
                    fontweight="bold" if setting == "neuron" else "normal", color=INK)
            ax.text(1.17, y, f"broken {s['b']}/{s['n']}", va="center", ha="right", fontsize=11, color=GREY)
            yt.append(y); yl.append(label)
            y += 1
        ax.text(-0.01, y - 0.3, p, ha="right", va="bottom", fontsize=15, fontweight="bold", color=ACCENT[p],
                transform=ax.get_yaxis_transform())
        y += 1.2
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=11, color=SUB)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13)
    ax.set_xlabel("coherent replies that show the trait (blind Claude labels)", fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    fig.suptitle("Real neurons give Qwen a personality. Random neurons at the same push don't.",
                 fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.925, f"{MODEL}, add mode (set mode gave similar rates, within 3 replies; see results.md). {len(PROMPTS)} prompts.\n"
             f"Neuron and normal: greedy + {N_SAMPLES} samples at T={TEMP} (N=32). Random neurons: same layers and push, 3 seeds, greedy (N=24).",
             fontsize=13, color=SUB, va="top")
    fig.text(0.02, 0.012, "Sign flip (every dose negated): 0 coherent hits in every preset. Broken = gibberish, loops or "
             "off-topic, counted separately. Every reply was labeled blind by a fresh Claude session.", fontsize=12, color=SUB)
    fig.tight_layout(rect=(0.1, 0.03, 1, 0.87))
    fig.savefig(path)
    plt.close(fig)


def table(st):
    g = lambda s: f"{s['kc']}/{s['n']}"
    rows = ["| Preset | Mode | Neuron coherent hits | Neuron hits (any) | Neuron broken | Random neurons coherent hits | Random broken | "
            "Sign flip coherent hits | Normal coherent hits |", "|---|---|---|---|---|---|---|---|---|"]
    for p in TRAITS:
        for m in MODES:
            n, r, f, b = st[(p, m, "neuron")], st[(p, m, "random")], st[(p, m, "sign_flip")], st[(p, None, "normal")]
            rows.append(f"| {p} | {m} | {g(n)} | {n['k']}/{n['n']} | {n['b']}/{n['n']} | {g(r)} | {r['b']}/{r['n']} | {g(f)} | {g(b)} |")
    return "\n".join(["Blind Claude labels. Coherent hit = shows the preset's trait and isn't broken. Neuron and normal: greedy + "
                      "3 samples at T=0.7 on 8 prompts. Random neurons: 3 seeds, greedy. Sign flip: greedy.", ""] + rows) + "\n"


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows, labels = load(src)
    st = stats(rows, labels)
    json.dump({"|".join(map(str, k)): v for k, v in st.items()}, open(os.path.join(out, "stats.json"), "w"), indent=1)
    open(os.path.join(out, "results.md"), "w").write(table(st))
    for p in TRAITS:
        for m in MODES:
            card(rows, labels, st, p, m, os.path.join(out, f"card_{p.replace(' + ', '_').replace(' ', '_')}_{m}.png"))
    chart(st, os.path.join(out, "summary.png"))
    print(open(os.path.join(out, "results.md")).read())
