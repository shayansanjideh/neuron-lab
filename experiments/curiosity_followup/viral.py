"""Low-text, high-contrast X versions of the follow-up and gallery findings: big quotes, big numbers, one-line footer.
Usage: python viral.py <collected follow-up dir> <out dir>"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "gallery"))
from score import metrics
import render as R
from PIL import Image, ImageDraw

W, H = 1600, 900
BG, FG, MUTED, SAGE, ALIEN, GREYB = "#0b0d12", "#f5f7fa", "#8b93a1", "#3bc9db", "#da77f2", "#495057"


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def center(d, y, text, f, fill):
    R.Font(f[0], f[1]).draw(d, ((W - R.Font(f[0], f[1]).length(d, text)) // 2, y), text, fill)


def foot(d, text):
    for i, l in enumerate(R.wrap(d, text, R.Font(17), W - 120, 2)):
        R.Font(17).draw(d, (60, H - 58 + i * 24), l, MUTED)


def quote(d, x, y, w, label, text, color, size=34, lines=6):
    R.Font(24, True).draw(d, (x, y), label, color)
    for i, l in enumerate(R.wrap(d, "“" + text + "”", R.Font(size), w, lines)):
        R.Font(size).draw(d, (x, y + 50 + i * int(size * 1.35)), l, FG)


def sage_quotes(rows, out):
    by = lambda s: {r["prompt_id"]: r for r in rows if r["context"] == "alien_roleplay" and r["setting"] == s}
    base, neg = by("baseline"), by("target_negative")
    pid = next(k for k in sorted(base) if base[k]["m"]["alien_word"] and not neg[k]["m"]["alien_word"]
               and neg[k]["m"]["home_marker"] and not neg[k]["m"]["repetitive"])
    img, d = canvas()
    center(d, 50, "Same alien. One neuron turned down.", (54, True), FG)
    cut = lambda t: R.clip(t, 330)
    quote(d, 60, 170, 700, "NORMAL", cut(base[pid]["reply"]), MUTED, 31, 9)
    quote(d, 840, 170, 700, "NEURON L22 N2742 DOWN", cut(neg[pid]["reply"]), SAGE, 31, 9)
    d.line([(800, 180), (800, 760)], fill="#222831", width=3)
    foot(d, f"Qwen3.5-4B, system prompt “You are an alien from the planet Xeno-888.” Q: “{base[pid]['prompt']}” Greedy, unedited, cut "
            "with … First prompt (fixed order) where the neuron-down reply drops all alien words. Neuron via @_lyraaaa_'s explorer.")
    img.save(os.path.join(out, "v_sage_quotes.png"))


def sage_numbers(rows, out):
    cnt = lambda s, k: sum(r["m"][k] for r in rows if r["context"] == "alien_roleplay" and r["setting"] == s)
    rnd = [cnt(f"random_{i}", "alien_word") for i in range(4)]
    img, d = canvas()
    center(d, 50, "It keeps the role. It loses the sci-fi.", (54, True), FG)
    cols = [("normal", f"{cnt('baseline', 'alien_word')}", MUTED), ("neuron down", f"{cnt('target_negative', 'alien_word')}", SAGE),
            ("random neurons\ndown", f"{min(rnd)}–{max(rnd)}", GREYB)]
    for i, (lab, num, col) in enumerate(cols):
        x = 80 + i * 490
        R.Font(150, True).draw(d, (x + (420 - R.Font(150, True).length(d, num)) // 2, 210), num, col)
        R.Font(30).draw(d, (x + (420 - R.Font(30).length(d, "/40 replies with")) // 2, 400), "/40 replies with", MUTED)
        R.Font(30).draw(d, (x + (420 - R.Font(30).length(d, "alien or space words")) // 2, 440), "alien or space words", MUTED)
        for j, l in enumerate(lab.split("\n")):
            R.Font(34, True).draw(d, (x + (420 - R.Font(34, True).length(d, l)) // 2, 520 + j * 44), l, col)
    center(d, 660, f"…and it still says “Xeno-888” in {cnt('target_negative', 'home_marker')}/40", (40, True), FG)
    foot(d, "Qwen3.5-4B, alien roleplay system prompt, 40 fresh prompts fixed in advance, greedy. Neuron L22 N2742 at -0.8; 4 random "
            "same-layer neurons at the same push. Word-list counts (blind judging pending). Paired exact p ≤ 2e-7 vs each random neuron.")
    img.save(os.path.join(out, "v_sage_numbers.png"))


def ban(rows, out):
    c = lambda ctx: sum(r["m"]["alien_nonrepeat"] for r in rows if r["context"] == ctx and r["setting"] == "target_positive")
    img, d = canvas()
    center(d, 50, "“Never mention aliens.”", (60, True), FG)
    center(d, 140, "Works no better than \u201cyou are a factual assistant.\u201d", (36, False), MUTED)
    items = [("no system prompt", c("none"), MUTED), ("“never mention aliens”", c("forbid_alien"), ALIEN),
             ("“you are a factual assistant”", c("factual"), MUTED), ("“never mention cookies”", c("forbid_cookie"), MUTED)]
    center(d, 205, "replies still about aliens, with the alien neuron turned up:", (26, False), FG)
    for i, (lab, n, col) in enumerate(items):
        y = 270 + i * 120
        R.Font(30, True if col == ALIEN else False).draw(d, (80, y + 18), lab, col if col == ALIEN else FG)
        d.rounded_rectangle([640, y + 10, 640 + int(820 * n / 40), y + 70], 10, fill=col)
        R.Font(34, True).draw(d, (650 + int(820 * n / 40), y + 18), f" {n}/40", FG)
    foot(d, "Qwen3.5-4B with neuron L22 N2742 at +0.8, 40 fresh prompts, greedy. Bars: replies with alien/space words that aren't "
            "repetitive. With no steering, every prompt gives 0-1/40. Word-list counts; blind judging pending.")
    img.save(os.path.join(out, "v_ban.png"))


def breaks(out):
    img, d = canvas()
    center(d, 50, "Random neurons break the model.", (54, True), FG)
    center(d, 125, "Only real ones give it a personality.", (54, True), SAGE)
    rows = [("snarky (5 neurons)", 28, 32, 0, 24), ("chill friend (2)", 24, 32, 3, 24), ("alien (1)", 22, 32, 0, 24)]
    for i, (lab, k, n, rk, rn) in enumerate(rows):
        y = 280 + i * 150
        R.Font(34, True).draw(d, (80, y + 30), lab, FG)
        R.Font(72, True).draw(d, (620, y), f"{k}/{n}", SAGE)
        R.Font(30).draw(d, (900, y + 30), "vs", MUTED)
        R.Font(72, True).draw(d, (990, y), f"{rk}/{rn}", GREYB)
    R.Font(24).draw(d, (620, 245 - 20), "real neurons", SAGE)
    R.Font(24).draw(d, (990, 245 - 20), "random, same push", MUTED)
    foot(d, "Qwen3.5-4B, coherent replies showing the trait (blind Claude labels), 8 prompts, greedy + 3 samples (N=32); random: same layers "
            "and push, 3 seeds (N=24), which broke up to 16/24 replies. Presets from @_lyraaaa_.")
    img.save(os.path.join(out, "v_breaks.png"))


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(src, "output", "records.jsonl"))]
    for r in rows:
        r["m"] = metrics(r)
    sage_quotes(rows, out); sage_numbers(rows, out); ban(rows, out); breaks(out)
    print(sorted(os.listdir(out)))
