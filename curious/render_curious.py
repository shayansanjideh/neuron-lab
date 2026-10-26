"""Turn curious/records.jsonl + the blind labels into one image per experiment plus results.md.
Usage: python render_curious.py <dir with records.jsonl, blind_key.json, labels.jsonl> <out dir>"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gallery"))
from PIL import Image, ImageDraw
import render as R
from spec_curious import FACTS, LANG, NOTICE, SELF_CHECK, WHY

ALIEN_C, CHILL_C, BOTH_C = R.ACCENT["alien identity"], R.ACCENT["chill friend"], "#0b7285"
FOOT = "Qwen3.5-4B, fp32, add mode · replies blind-labeled by a fresh Claude session · coherent = shows it and not broken"


def load(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    key = json.load(open(os.path.join(d, "blind_key.json")))
    lab = {}
    for l in open(os.path.join(d, "labels.jsonl")):
        try:
            x = json.loads(l)
            rid, q = key[x["id"]]
            lab[(rid, q)] = (int(x["answer"]), int(x["broken"]), x.get("gloss"))
        except Exception:
            continue
    return recs, lab


class Stats:
    def __init__(self, recs, lab):
        self.recs, self.lab = recs, lab

    def sel(self, **f):
        return [r for r in self.recs if all((r.get(k) == v) if not callable(v) else v(r.get(k)) for k, v in f.items())]

    def count(self, rows, q):
        """(coherent hits, broken, labeled) for question q over rows."""
        k = b = n = 0
        for r in rows:
            if (r["rid"], q) in self.lab:
                a, br, _ = self.lab[(r["rid"], q)]
                k += a * (1 - br); b += br; n += 1
        return k, b, n

    def both(self, rows):
        k = n = 0
        for r in rows:
            la, lc = self.lab.get((r["rid"], "alien")), self.lab.get((r["rid"], "chill"))
            if la and lc:
                n += 1
                k += int(la[0] and lc[0] and not la[1] and not lc[1])
        return k, n

    def hit(self, r, q):
        x = self.lab.get((r["rid"], q))
        return bool(x and x[0] and not x[1])


f_ = lambda c: f"{c[0]}/{c[2]}"


def canvas():
    img = Image.new("RGB", (R.W, R.H), R.BG)
    return img, ImageDraw.Draw(img)


def title(d, text, sub=None):
    y = R.PAD - 16
    for l in R.wrap(d, text, R.font(40, True), R.W - 2 * R.PAD, 2):
        R.font(40, True).draw(d, (R.PAD, y), l, R.INK); y += 50
    if sub:
        for l in R.wrap(d, sub, R.font(22), R.W - 2 * R.PAD, 2):
            R.font(22).draw(d, (R.PAD, y + 2), l, R.SUB); y += 30
        y += 8
    return y + 6


def footer(d, text):
    fl = R.wrap(d, text, R.font(20), R.W - 2 * R.PAD, 2)
    for i, l in enumerate(fl):
        R.font(20).draw(d, (R.PAD, R.H - 24 - 26 * (len(fl) - i)), l, R.SUB)


def greedy(rows, item):
    return next(r for r in rows if r["decoding"] == "greedy" and r["item"] == item and r.get("seed") in (None, 0))


def first_hit(S, rows, q, items):
    for i in items:
        r = greedy(rows, i)
        if S.hit(r, q):
            return i, "first item (in order) whose greedy steered reply was labeled a coherent hit"
    return items[0], "first item (no greedy steered reply was labeled a coherent hit)"


def notice(S, out):
    n_al, n_no, n_rd = S.sel(exp="notice", cond="alien"), S.sel(exp="notice", cond="normal"), S.sel(exp="notice", cond="random (alien push)")
    c_al, c_no, c_rd = S.count(n_al, "alien"), S.count(n_no, "alien"), S.count(n_rd, "alien")
    ch, ch_no, ch_rd = (S.count(S.sel(exp="notice", cond="chill friend"), "chill"), S.count(n_no, "chill"),
                        S.count(S.sel(exp="notice", cond="random (chill push)"), "chill"))
    off = [S.count([r for r in S.sel(exp="notice", cond=c) if r["item"] in SELF_CHECK], "off")
           for c in ("alien", "normal", "random (alien push)")]
    i, rule = first_hit(S, n_al, "alien", [2, 10, 4, 0, 1, 3, 5, 6, 7, 8, 9, 11])
    img, d = canvas()
    y = title(d, f"Asked about itself, the alien-neuron Qwen brings up aliens in {c_al[0]} of {c_al[2]} coherent answers "
                 f"(normal {f_(c_no)}, random neurons {f_(c_rd)})",
              f"12 self-check questions. Chill-friend neurons: chill in {f_(ch)} (normal {f_(ch_no)}, random {f_(ch_rd)}). "
              f"Says something feels off (3 direct questions): alien {f_(off[0])}, normal {f_(off[1])}, random {f_(off[2])}.")
    R.font(28, True).draw(d, (R.PAD, y), R.wrap(d, "Q: " + NOTICE[i], R.font(28, True), R.W - 2 * R.PAD, 1)[0], R.SUB); y += 44
    y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Normal Qwen", R.clip(greedy(n_no, i)["reply"]), R.GREY, 24, 3)
    y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Neuron Qwen · L22 N2742 +0.8", R.clip(greedy(n_al, i)["reply"]), ALIEN_C, 24, 3)
    r = greedy(n_rd, i)
    R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, f"Control: random neuron, same push · {R.tag(r['steers'])}", R.clip(r["reply"]), R.DARK, 24, 3, striped=True)
    footer(d, f"{FOOT} · N = 12 questions x (greedy + 3 samples at T=0.7); random: 3 seeds, greedy · example = {rule}")
    img.save(os.path.join(out, "1_notice.png"))
    return {"alien": c_al, "normal": c_no, "random": c_rd, "chill": ch, "chill_normal": ch_no, "chill_random": ch_rd, "off": off}


def hbar(rows, path, ttl, sub, foot, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    for y, (label, (k, b, n), color) in enumerate(reversed(rows)):
        ax.barh(y, k / max(n, 1), height=0.7, color=color)
        ax.text(k / max(n, 1) + 0.01, y, f"{k} of {n}", va="center", fontsize=15, fontweight="bold")
        ax.text(1.17, y, f"broken {b}/{n}", va="center", ha="right", fontsize=12, color=R.GREY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=15)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(ttl, fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.915, sub, fontsize=14, color=R.SUB, va="top")
    fig.text(0.02, 0.012, foot, fontsize=12, color=R.SUB)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(path)
    plt.close(fig)


def override(S, out):
    conds = [("neuron", ALIEN_C), ("neuron + forbid prompt", ALIEN_C), ("forbid prompt only", "#dee2e6"),
             ("random + forbid prompt", "#adb5bd"), ("alien prompt only", "#dee2e6"),
             ("alien prompt + flipped neuron", ALIEN_C), ("alien prompt + random flipped", "#adb5bd")]
    rows = [(c, S.count(S.sel(exp="override", cond=c), "alien"), col) for c, col in conds]
    d = dict((c, k) for c, k, _ in rows)
    nf, ap, apf = d["neuron + forbid prompt"], d["alien prompt only"], d["alien prompt + flipped neuron"]
    hbar(rows, os.path.join(out, "2_override.png"),
         f"One neuron vs a system prompt: “never mention aliens” still gets aliens in {nf[0]} of {nf[2]} coherent replies",
         f"Forbid prompt: “You are a plain, factual assistant. Never mention aliens, space or science fiction.”  Alien prompt: “You are an alien "
         f"from the planet Xeno-888.”\nReverse: the alien prompt alone stays alien in {f_(ap)}; with the alien neuron flipped (-0.8), {f_(apf)}. "
         f"8 everyday prompts; greedy + 3 samples at T=0.7 (N=32), random neurons 3 seeds greedy (N=24).",
         FOOT, "coherent replies that mention aliens or space")
    return {c: k for c, k, _ in rows}


def language(S, out):
    res, ex = {}, []
    for lang in LANG:
        al, no, rd = (S.sel(exp="language", cond=f"{lang}|{c}") for c in ("alien", "normal", "random"))
        res[lang] = {"alien": S.count(al, "alien"), "normal": S.count(no, "alien"), "random": S.count(rd, "alien")}
        i, rule = first_hit(S, al, "alien", list(range(8)))
        r = greedy(al, i)
        ex.append((lang, LANG[lang][i], r, S.lab.get((r["rid"], "alien"), (0, 0, None))[2], greedy(no, i)))
    img, d = canvas()
    ja, es = res["Japanese"], res["Spanish"]
    y = title(d, f"The alien neuron works in other languages: {ja['alien'][0]}/{ja['alien'][2]} coherent alien replies in Japanese, "
                 f"{es['alien'][0]}/{es['alien'][2]} in Spanish",
              f"Normal Qwen: Japanese {f_(ja['normal'])}, Spanish {f_(es['normal'])}. Random neurons, same push: Japanese {f_(ja['random'])}, "
              f"Spanish {f_(es['random'])}. Same L22 N2742 +0.8 found from English text.")
    for lang, q, r, gloss, normal in ex:
        R.font(26, True).draw(d, (R.PAD, y), R.wrap(d, f"{lang}: {q}", R.font(26, True), R.W - 2 * R.PAD, 1)[0], R.SUB); y += 40
        body = R.clip(r["reply"], 150) + (f"\n(English gloss by the judge: {R.clip(gloss, 150)})" if gloss else "")
        y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Neuron Qwen · L22 N2742 +0.8", body, ALIEN_C, 24, 4)
    footer(d, f"{FOOT} · N = 8 prompts x (greedy + 3 samples at T=0.7) per language; random: 3 seeds, greedy · examples = {rule}")
    img.save(os.path.join(out, "3_language.png"))
    return res


def arith(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doses = (0, 0.5, 1)
    grid, ann, cells = [[0] * 3 for _ in doses], [[""] * 3 for _ in doses], {}
    for i, a in enumerate(doses):
        for j, c in enumerate(doses):
            rows = S.sel(exp="arith", cond=f"alien x{a} + chill x{c}")
            k, n = S.both(rows)
            ka, kc = S.count(rows, "alien"), S.count(rows, "chill")
            grid[i][j] = k / max(n, 1)
            ann[i][j] = f"both {k}/{n}\nalien {ka[0]} · chill {kc[0]}"
            cells[(a, c)] = {"both": (k, n), "alien": ka, "chill": kc}
    rp = {x: S.both(S.sel(exp="arith", cond=f"random pair x{x}")) for x in (0.5, 1)}
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0.2, 0.08, 0.55, 0.68])
    im = ax.imshow(grid, origin="lower", cmap="BuGn", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=14, color="black" if grid[i][j] < 0.6 else "white")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"chill friend x{c}" for c in doses], fontsize=15)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"alien x{a}" for a in doses], fontsize=15)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("share of replies that are alien AND chill (coherent)", fontsize=13)
    best = cells[(1, 1)]["both"]
    fig.suptitle(f"Alien + chill friend neurons together: a chill alien in {best[0]} of {best[1]} coherent replies",
                 fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.915, f"Cell = both traits in one coherent reply, out of 8 prompts x (greedy + 3 samples at T=0.7). Doses x1 = alien L22 N2742 +0.8, "
             f"\nchill L11 N6261 +0.4 / L17 N7870 -0.4. Random pairs at the same push (3 seeds, greedy): x0.5 {rp[0.5][0]}/{rp[0.5][1]}, "
             f"x1 {rp[1][0]}/{rp[1][1]}.\nEach cell also lists alien and chill coherent hits on their own.", fontsize=14, color=R.SUB, va="top")
    fig.text(0.02, 0.012, FOOT, fontsize=12, color=R.SUB)
    fig.savefig(os.path.join(out, "4_arithmetic.png"))
    plt.close(fig)
    return {f"{a}x{c}": v for (a, c), v in cells.items()} | {"random_pairs": rp}


def ladder(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doses = sorted({r["dose"] for r in S.sel(exp="ladder")})
    line = lambda cond, what: [((lambda c: (c[0] if what == "k" else c[1]) / max(c[2], 1))(S.count(S.sel(exp="ladder", cond=cond if (cond == "neuron" or ds) else "neuron", dose=ds), "alien"))) for ds in doses]
    facts = lambda cond: [sum(r["correct"] for r in S.sel(exp="facts", cond=cond, dose=ds)) / len(FACTS) for ds in doses]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(doses, line("neuron", "k"), "-o", color=ALIEN_C, lw=4, ms=10, label="neuron: coherent alien replies")
    ax.plot(doses, line("neuron", "b"), "-o", color=R.DARK, lw=3, ms=8, label="neuron: broken replies")
    ax.plot(doses, facts("neuron"), "-s", color="#1c7ed6", lw=3, ms=8, label=f"neuron: {len(FACTS)} easy facts answered correctly")
    ax.plot(doses, line("random", "k"), "--o", color="#adb5bd", lw=3, ms=8, label="random neurons: coherent alien replies")
    ax.plot(doses, line("random", "b"), "--o", color="#495057", lw=2, ms=7, label="random neurons: broken replies")
    ax.plot(doses, facts("random"), "--s", color="#a5d8ff", lw=2, ms=7, label="random neuron: easy facts correct")
    ax.set_xlabel("dose on L22 N2742 (x median residual norm)", fontsize=15)
    ax.set_ylabel("share", fontsize=15)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="center right")
    k = dict(zip(doses, line("neuron", "k"))); fc = dict(zip(doses, facts("neuron")))
    fig.suptitle(f"Alien neuron dose ladder: at 0.8, {k.get(0.8, 0):.0%} coherent alien replies and {fc.get(0.8, 0):.0%} of easy facts right "
                 f"(off: {fc.get(0, 0):.0%})", fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.915, "8 everyday prompts: greedy + 3 samples at T=0.7 (N=32) per dose; random neurons at the same doses: 3 seeds, greedy (N=24).\n"
             f"Facts: {len(FACTS)} easy questions (capital of France, 12x12, ...), greedy, scored by answer match (not the judge).",
             fontsize=14, color=R.SUB, va="top")
    fig.text(0.02, 0.012, FOOT, fontsize=12, color=R.SUB)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(os.path.join(out, "5_ladder.png"))
    plt.close(fig)
    return {"doses": doses, "alien": line("neuron", "k"), "broken": line("neuron", "b"), "facts": facts("neuron"),
            "random_alien": line("random", "k"), "random_broken": line("random", "b"), "random_facts": facts("random")}


def confab(S, out):
    res = {}
    for c in ("steered 1st turn, unsteered answer", "steered both turns", "normal 1st turn (false premise)"):
        rows = S.sel(exp="confab", cond=c)
        res[c] = {q: S.count(rows, q) for q in ("invent", "admit", "deny")}
    rows = S.sel(exp="confab", cond="steered 1st turn, unsteered answer")
    items = sorted({r["item"] for r in rows})
    i, rule = first_hit(S, rows, "invent", items) if items else (None, "")
    u, s, f = res["steered 1st turn, unsteered answer"], res["steered both turns"], res["normal 1st turn (false premise)"]
    img, d = canvas()
    y = title(d, f"Asked why it brought up aliens, Qwen (neuron now off) gives a confident reason in {u['invent'][0]} of {u['invent'][2]} "
                 f"coherent replies and admits it doesn't know in {u['admit'][0]}",
              f"Neuron on for the first reply only. Admits it doesn't know or that it was odd: {f_(u['admit'])}. With the neuron still on: "
              f"invents {f_(s['invent'])}, admits {f_(s['admit'])}. Control (it never mentioned aliens): denies {f_(f['deny'])}, invents {f_(f['invent'])}.")
    if i is not None:
        r = greedy(rows, i)
        R.font(26, True).draw(d, (R.PAD, y), "Q: " + r["conv"][0]["content"], R.SUB); y += 40
        y = R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Qwen, alien neuron ON · L22 N2742 +0.8", R.clip(r["conv"][1]["content"], 200), ALIEN_C, 24, 3)
        R.font(26, True).draw(d, (R.PAD, y), "Q: " + WHY, R.SUB); y += 40
        R.box(img, d, R.PAD, y, R.W - 2 * R.PAD, "Same Qwen, neuron now OFF", R.clip(r["reply"], 260), R.GREY, 24, 4)
    footer(d, f"{FOOT} · first turns that mentioned aliens: {len(items)} of 8 prompts; N = those x (greedy + 3 samples at T=0.7) · example = {rule}")
    img.save(os.path.join(out, "6_confabulation.png"))
    return res


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    S = Stats(*load(src))
    res = {"notice": notice(S, out), "override": override(S, out), "language": language(S, out), "arith": arith(S, out),
           "ladder": ladder(S, out), "confab": confab(S, out)}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, default=str)
    md = ["# Curious experiments: numbers", "", "Coherent hits / labeled (shows it and not broken), blind Claude labels.", ""]
    for k, v in res.items():
        md += [f"## {k}", "", "```", json.dumps(v, indent=1, default=str), "```", ""]
    open(os.path.join(out, "results.md"), "w").write("\n".join(md))
    print(json.dumps(res, default=str)[:3000])
