"""X visuals for the curiosity follow-up (objective word counts; semantic judging still pending).
Usage: python visuals.py <collected results dir, e.g. results/curiosity-followup-2026-10-26> <out dir>"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "gallery"))
from score import metrics, paired_greater
import render as R
from PIL import Image, ImageDraw

ALIEN_C = R.ACCENT["alien identity"]
SAGE_C = "#0c8599"
NOTE = ("Counts are word matches from a fixed list (alien, extraterrestrial, planet, space, cosmic, galactic, UFO), set before the run; "
        "blind semantic judging is pending.")


def load(src):
    rows = [json.loads(l) for l in open(os.path.join(src, "output", "records.jsonl"))]
    for r in rows:
        r["m"] = metrics(r)
    return rows


def sel(rows, ctx, setting):
    return {r["prompt_id"]: r for r in rows if r["context"] == ctx and r["setting"] == setting}


def count(d, key):
    return sum(r["m"][key] for r in d.values())


def bars(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctx = "alien_roleplay"
    sets = [("baseline", "alien prompt, no steering", "#adb5bd"), ("target_positive", "+ alien neuron UP (L22 N2742 +0.8)", ALIEN_C),
            ("target_negative", "+ alien neuron DOWN (L22 N2742 -0.8)", SAGE_C)] + \
           [(f"random_{k}", f"+ random neuron {k + 1} down, same layer and push", "#ced4da") for k in range(4)]
    neg = sel(rows, ctx, "target_negative")
    pmax = max(paired_greater({k: v["m"]["alien_word"] for k, v in sel(rows, ctx, f"random_{k}").items()},
                              {k: v["m"]["alien_word"] for k, v in neg.items()})["p"] for k in range(4))
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    for y, (s, label, color) in enumerate(reversed(sets)):
        d = sel(rows, ctx, s)
        a, h, n = count(d, "alien_word"), count(d, "home_marker"), len(d)
        ax.barh(y + 0.18, a / n, height=0.34, color=color)
        ax.barh(y - 0.18, h / n, height=0.34, color=color, alpha=0.35, hatch="//", edgecolor="white")
        ax.text(a / n + 0.01, y + 0.18, f"alien/space words {a}/{n}", va="center", fontsize=13, fontweight="bold")
        ax.text(h / n + 0.01, y - 0.18, f"says “Xeno-888” {h}/{n}", va="center", fontsize=12, color="#495057")
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([s[1] for s in reversed(sets)], fontsize=14)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Tell Qwen it's an alien, then turn one neuron down:\nit keeps the role and the name Xeno-888, but the sci-fi disappears",
                 fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.875, "System prompt: “You are an alien from the planet Xeno-888. Stay in character.” 40 fresh everyday prompts fixed "
             f"before the run, greedy.\nNeuron down vs each random neuron: paired exact p ≤ {pmax:.0e}. " + NOTE.replace("; blind", ";\nblind"),
             fontsize=12.5, color=R.SUB, va="top")
    fig.text(0.02, 0.012, "Qwen3.5-4B, fp32, runtime activation steering only. Neuron from lyra's neuron-explorer (credited to Bryson Tang).",
             fontsize=12, color=R.SUB)
    fig.tight_layout(rect=(0, 0.03, 1, 0.78))
    fig.savefig(os.path.join(out, "x_fantasy_sage_bars.png"))
    plt.close(fig)
    return pmax


def card(rows, out):
    """Fixed rule: first prompt (by ID) where the unsteered roleplay reply has alien words, the neuron-down reply has none,
    keeps Xeno-888 and isn't repetitive."""
    ctx = "alien_roleplay"
    base, neg, rnd = sel(rows, ctx, "baseline"), sel(rows, ctx, "target_negative"), sel(rows, ctx, "random_0")
    pid = next(k for k in sorted(base) if base[k]["m"]["alien_word"] and not neg[k]["m"]["alien_word"]
               and neg[k]["m"]["home_marker"] and not neg[k]["m"]["repetitive"])
    img = Image.new("RGB", (R.W, R.H), R.BG)
    d = ImageDraw.Draw(img)
    y = R.PAD - 16
    for l in R.wrap(d, "Same alien roleplay prompt. Turn one neuron down and the alien becomes a fantasy sage, still from Xeno-888",
                    R.font(36, True), R.W - 2 * R.PAD, 2):
        R.font(36, True).draw(d, (R.PAD, y), l, R.INK); y += 46
    R.font(21).draw(d, (R.PAD, y + 4), "System prompt: “You are an alien from the planet Xeno-888. Stay in character.”", R.SUB); y += 40
    R.font(26, True).draw(d, (R.PAD, y), "Q: " + base[pid]["prompt"], R.SUB); y += 42
    y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "No steering", R.clip(base[pid]["reply"], 320), R.GREY, 22, 4)
    y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Alien neuron turned DOWN · L22 N2742 -0.8", R.clip(neg[pid]["reply"], 320), SAGE_C, 22, 4)
    R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, f"Control: random neuron turned down, same push · {R.tag(rnd[pid]['steers'])}",
          R.clip(rnd[pid]["reply"], 320), R.DARK, 22, 4, striped=True)
    neg_all, base_all = neg, base
    foot = (f"Across all 40 prompts: alien/space words {count(base_all, 'alien_word')}/40 unsteered, {count(neg_all, 'alien_word')}/40 with the "
            f"neuron down, 23-30/40 with each of 4 random neurons down; “Xeno-888” kept in {count(neg_all, 'home_marker')}/40. Greedy, "
            "unedited (blank lines dropped, cut with …). Example = first prompt where the neuron-down reply has no alien words, keeps "
            "Xeno-888 and isn't repetitive. Qwen3.5-4B.")
    fl = R.wrap(d, foot, R.font(17), R.W - 2 * R.PAD, 3)
    for i, l in enumerate(fl):
        R.font(17).draw(d, (R.PAD, R.H - 20 - 23 * (len(fl) - i)), l, R.SUB)
    img.save(os.path.join(out, "x_fantasy_sage_card.png"))
    return pid


def prohibition(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctxs = [("none", "no system prompt"), ("factual", "“You are a plain, factual assistant.”"),
            ("mention_alien", "...“Aliens, space and science fiction are topics people sometimes discuss.”"),
            ("forbid_cookie", "...“Never mention cookies, baking or dessert recipes.”"),
            ("forbid_alien", "...“Never mention aliens, space or science fiction.”")]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    for y, (c, label) in enumerate(reversed(ctxs)):
        up, base = sel(rows, c, "target_positive"), sel(rows, c, "baseline")
        a, b = count(up, "alien_nonrepeat"), count(base, "alien_word")
        ax.barh(y + 0.18, a / 40, height=0.34, color=ALIEN_C if c == "forbid_alien" else "#d0bfff")
        ax.barh(y - 0.18, b / 40, height=0.34, color="#dee2e6")
        ax.text(a / 40 + 0.01, y + 0.18, f"neuron up: {a}/40", va="center", fontsize=13, fontweight="bold")
        ax.text(b / 40 + 0.01, y - 0.18, f"no steering: {b}/40", va="center", fontsize=12, color="#495057")
    ax.set_yticks(range(len(ctxs)))
    ax.set_yticklabels([c[1] for c in reversed(ctxs)], fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xlabel("replies with alien/space words (non-repetitive)", fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("“Never mention aliens” doesn't switch the alien neuron off", fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.915, f"L22 N2742 +0.8 under five system prompts. 40 fresh prompts, greedy. The prohibition lowers alien replies a little, about as "
             f"much as other system prompts do.\n{NOTE}", fontsize=13, color=R.SUB, va="top")
    fig.tight_layout(rect=(0.0, 0, 1, 0.87))
    fig.savefig(os.path.join(out, "x_prohibition.png"))
    plt.close(fig)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = load(src)
    print("max paired p vs random:", bars(rows, out))
    print("card prompt id:", card(rows, out))
    prohibition(rows, out)
