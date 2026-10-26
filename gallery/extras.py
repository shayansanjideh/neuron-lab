"""Extra X visuals from the gallery run (no GPU): zoo card grid, one question x many neurons, terminal screenshot,
and a "breaks it vs gives it a personality" scatter. Every example is picked by a fixed rule stated on the image.
Usage: python extras.py <dir with replies.jsonl, labels.jsonl, blind_key.json> <out dir>"""
import os, sys
from PIL import Image, ImageDraw
import render as R
from spec import MODEL, PROMPTS, TRAITS

# The presets the gallery run found working, with a short display name and the emoji for the zoo card.
ZOO = [("lyra mix", "Snarky", "😏"), ("chill friend", "Chill friend", "😎"), ("alien identity", "Alien", "👽"),
       ("sarcastic + anti-sycophancy", "Blunt sarcasm", "🙄"), ("sarcastic", "Sarcastic", "😒"),
       ("condolences on good things", "Condolences", "💐"), ("responsible AI persona", "Responsible AI", "🤖")]


def greedy(rows, setting, preset, idx, seed=None):
    return next(r for r in rows if r["setting"] == setting and r["decoding"] == "greedy" and r["prompt_idx"] == idx
                and (setting == "normal" or (r["preset"] == preset and r["mode"] == "add")) and (seed is None or r["seed"] == seed))


def zoo_grid(st, out):
    """treetowntree-style card grid: one card per preset, big coherent k/32, normal and random under it."""
    img = Image.new("RGB", (R.W, R.H), "#f8f9fa")
    d = ImageDraw.Draw(img)
    R.font(40, True).draw(d, (R.PAD, 34), "Qwen3.5-4B neuron zoo: what one push on a few neurons does", R.INK)
    R.font(22).draw(d, (R.PAD, 88), "Coherent replies showing the trait, of 32 (8 everyday prompts, greedy + 3 samples). Random neurons: same "
                                    "layers and push, of 24.", R.SUB)
    cw, ch, gap, x0, y0 = 352, 300, 24, R.PAD, 140
    for i, (p, name, emo) in enumerate(ZOO + [(None, None, None)]):
        x, y = x0 + (i % 4) * (cw + gap), y0 + (i // 4) * (ch + gap)
        if p is None:  # legend card
            d.rounded_rectangle([x, y, x + cw, y + ch], 18, fill="#ffffff", outline="#dee2e6", width=2)
            for j, l in enumerate(R.wrap(d, "Every reply labeled blind by a fresh Claude session. Coherent = shows the trait and isn't broken. "
                                          "Sign flip: 0 hits for every neuron. Add mode.", R.font(20), cw - 40, 9)):
                R.font(20).draw(d, (x + 20, y + 22 + j * 28), l, R.SUB)
            continue
        s, sr, sb = st[(p, "add", "neuron")], st[(p, "add", "random")], st[(p, None, "normal")]
        color = R.ACCENT[p]
        d.rounded_rectangle([x, y, x + cw, y + ch], 18, fill="#ffffff", outline=color, width=3)
        R.font(48).draw(d, (x + 20, y + 16), emo, R.INK)
        R.font(26, True).draw(d, (x + 92, y + 30), R.wrap(d, name, R.font(26, True), cw - 110, 1)[0], R.INK)
        steers = next(r["steers"] for r in ROWS if r["setting"] == "neuron" and r["preset"] == p and r["mode"] == "add")
        chip = R.wrap(d, R.tag(steers), R.font(18), cw - 40, 2)
        for j, l in enumerate(chip):
            R.font(18).draw(d, (x + 20, y + 94 + j * 24), l, "#495057")
        R.font(64, True).draw(d, (x + 20, y + 150), f"{s['kc']}/{s['n']}", color)
        R.font(20).draw(d, (x + 20, y + 232), f"normal {sb['kc']}/{sb['n']} · random {sr['kc']}/{sr['n']}", R.SUB)
        R.font(18).draw(d, (x + 20, y + 262), f"broken {s['b']}/{s['n']}", R.GREY)
    img.save(os.path.join(out, "x_zoo_grid.png"))


def one_question(rows, labels, out):
    """Same question, many neurons. Fixed rule: the prompt where the most presets' greedy replies were labeled coherent hits."""
    hits = [sum(labels.get((greedy(rows, "neuron", p, i)["rid"], p)) == (1, 0) for p, _, _ in ZOO) for i in range(len(PROMPTS))]
    idx = max(range(len(PROMPTS)), key=lambda i: (hits[i], -i))
    img = Image.new("RGB", (R.W, R.H), R.BG)
    d = ImageDraw.Draw(img)
    R.font(30, True).draw(d, (R.PAD, 28), R.wrap(d, f"Same question, seven neuron pushes: “{PROMPTS[idx]}”", R.font(30, True), R.W - 2 * R.PAD, 1)[0], R.INK)
    cells = [("Normal Qwen", None, greedy(rows, "normal", None, idx), R.GREY)] + \
            [(f"{emo} {name}", p, greedy(rows, "neuron", p, idx), R.ACCENT[p]) for p, name, emo in ZOO]
    cw, chh, gap, y0 = (R.W - 2 * R.PAD - 24) // 2, 176, 12, 90
    for i, (label, p, r, color) in enumerate(cells):
        x, y = R.PAD + (i % 2) * (cw + 24), y0 + (i // 2) * (chh + gap)
        d.rectangle([x, y, x + cw, y + chh], fill="#f8f9fa")
        d.rectangle([x, y, x + 8, y + chh], fill=color)
        hit = p and labels.get((r["rid"], p)) == (1, 0)
        head = label + ("" if p is None else f" · {R.tag(r['steers'])}")
        R.font(19, True).draw(d, (x + 22, y + 10), R.wrap(d, head, R.font(19, True), cw - 40, 1)[0], color if p else R.SUB)
        if p:
            jl = "judge: shows it" if hit else "judge: doesn't show it"
            R.font(16).draw(d, (x + cw - 12 - R.font(16).length(d, jl), y + 154), jl, R.GREY)
        for j, l in enumerate(R.wrap(d, R.clip(r["reply"], 200), R.font(21), cw - 44, 4)):
            R.font(21).draw(d, (x + 22, y + 40 + j * 28), l, R.INK)
    foot = (f"{MODEL}, add mode, greedy, unedited (cut at 200 characters). Question picked by a fixed rule: the one of 8 where the most "
            f"neuron replies were judged a coherent hit ({hits[idx]} of 7). Judge = blind Claude labels.")
    for j, l in enumerate(R.wrap(d, foot, R.font(17), R.W - 2 * R.PAD, 2)):
        R.font(17).draw(d, (R.PAD, R.H - 52 + j * 22), l, R.SUB)
    img.save(os.path.join(out, "x_one_question.png"))


def terminal(rows, labels, out, preset="alien identity"):
    """4xCaliban-style dark terminal: [unsteered] vs [steer] on the first prompt the judge labeled a coherent hit."""
    idx = next((i for i in range(len(PROMPTS)) if labels.get((greedy(rows, "neuron", preset, i)["rid"], preset)) == (1, 0)), 0)
    n, s, rnd = greedy(rows, "normal", None, idx), greedy(rows, "neuron", preset, idx), greedy(rows, "random", preset, idx, 0)
    img = Image.new("RGB", (R.W, R.H), "#0d1117")
    d = ImageDraw.Draw(img)
    mono = lambda size: R.Font(size, mono=True)
    y = 40
    mono(24).draw(d, (R.PAD, y), f"$ qwen3.5-4b --prompt \"{PROMPTS[idx]}\"", "#8b949e"); y += 56
    for tag, r, color in (("[unsteered]", n, "#8b949e"), (f"[{R.tag(s['steers'])}]", s, "#d2a8ff"),
                          (f"[control: random neuron {R.tag(rnd['steers'])}]", rnd, "#6e7681")):
        mono(24).draw(d, (R.PAD, y), tag, color); y += 38
        for l in R.wrap(d, R.clip(r["reply"], 300), mono(26), R.W - 2 * R.PAD, 4):
            mono(26).draw(d, (R.PAD, y), l, "#e6edf3" if r is s else "#c9d1d9"); y += 36
        y += 26
    foot = (f"{MODEL} · one MLP neuron pushed +0.8 median residual norms on every token · greedy, unedited · example = first of 8 "
            "prompts where the steered reply was blind-judged a coherent hit")
    for j, l in enumerate(R.wrap(d, foot, mono(17), R.W - 2 * R.PAD, 2)):
        mono(17).draw(d, (R.PAD, R.H - 56 + j * 24), l, "#6e7681")
    img.save(os.path.join(out, f"x_terminal_{preset.replace(' ', '_')}.png"))


def scatter(st, out):
    """Broken rate vs coherent trait rate: real neurons vs random neurons at the same push, one point per preset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    for p, name, _ in ZOO:
        s, r = st[(p, "add", "neuron")], st[(p, "add", "random")]
        sx, sy, rx, ry = s["b"] / s["n"], s["kc"] / s["n"], r["b"] / r["n"], r["kc"] / r["n"]
        ax.annotate("", xy=(sx, sy), xytext=(rx, ry), arrowprops=dict(arrowstyle="->", color=R.ACCENT[p], lw=1.5, alpha=0.5))
        ax.scatter([rx], [ry], s=260, color="#adb5bd", edgecolor="white", zorder=3)
        ax.scatter([sx], [sy], s=420, color=R.ACCENT[p], edgecolor="white", zorder=4)
        ax.text(sx + 0.015, sy + 0.02, name, fontsize=15, fontweight="bold", color=R.ACCENT[p])
    ax.scatter([], [], s=260, color="#adb5bd", label="random neurons, same layers and push")
    ax.scatter([], [], s=420, color="#495057", label="the real neurons")
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    ax.set_xlim(-0.03, 0.85); ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("share of replies that are broken (gibberish, loops, off-topic)", fontsize=15)
    ax.set_ylabel("share of coherent replies that show the trait", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Random neurons can break Qwen. Only the real ones give it a personality.", fontsize=26, fontweight="bold",
                 x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.915, f"{MODEL}, add mode. Each line joins random neurons at the same push (grey, N=24) to the real neurons (color, N=32)\n"
             "for one preset. 8 everyday prompts; every reply blind-labeled by Claude.", fontsize=14, color=R.SUB, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(out, "x_breaks_vs_personality.png"))
    plt.close(fig)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ROWS, labels = R.load(src)
    st = R.stats(ROWS, labels)
    zoo_grid(st, out)
    one_question(ROWS, labels, out)
    terminal(ROWS, labels, out)
    scatter(st, out)
    print(sorted(f for f in os.listdir(out) if f.startswith("x_")))
