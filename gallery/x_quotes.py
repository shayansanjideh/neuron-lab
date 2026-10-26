"""Quote-card X images (alien on every prompt, meaning of life four ways). Usage: python x_quotes.py <out dir>"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Font, wrap, load, stats, ACCENT, GREY, DARK, INK, SUB, BG, TRAITS, MODEL
from PIL import Image, ImageDraw
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
W, H, PAD = 1600, 900, 56
rows, labels = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
st = stats(rows, labels)
def g(preset, prompt, setting="neuron", mode="add"):
    return next(r for r in rows if r["prompt"] == prompt and r["decoding"] == "greedy" and r["setting"] == setting
                and (setting == "normal" or (r["preset"] == preset and r["mode"] == mode)))
def tidy(t, n):
    t = re.sub(r"\*\*|###? ?|</?think>", "", t)
    t = "".join(c for c in t if ord(c) < 0x2000 or 0x2010 <= ord(c) <= 0x2027)
    t = " ".join(l.strip() for l in t.split("\n") if l.strip())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"
def tint(c, a=.10):
    r, gg, b = Image.new("RGB", (1, 1), c).getpixel((0, 0))
    return tuple(int(255 - (255 - v) * a) for v in (r, gg, b))
def draw_hl(d, f, fb, xy, line, hl, color):
    x, y = xy
    for tok in re.split(r"(\s+)", line):
        bare = re.sub(r"[^\w-]", "", tok).lower()
        ff, cc = (fb, color) if hl and any(h in bare for h in hl) else (f, INK)
        ff.draw(d, (x, y), tok, cc); x += ff.length(d, tok)
def tile(img, d, x, y, w, h, label, text, color, size=22, hl=(), striped=False, max_lines=None):
    d.rectangle([x, y, x + w, y + h], fill=tint(color) if not striped else "#f1f3f5")
    d.rectangle([x, y, x + 8, y + h], fill=color)
    fl, f, fb = Font(20, True), Font(size), Font(size, True)
    fl.draw(d, (x + 24, y + 14), label, color)
    lh = int(size * 1.35)
    ml = max_lines or (h - 56) // lh
    for i, l in enumerate(wrap(d, text, f, w - 44, ml)):
        draw_hl(d, f, fb, (x + 24, y + 46 + i * lh), l, hl, color)
def title(d, t, sub, y=PAD - 16):
    for l in wrap(d, t, Font(40, True), W - 2 * PAD, 2):
        Font(40, True).draw(d, (PAD, y), l, INK); y += 50
    Font(22).draw(d, (PAD, y + 2), sub, SUB)
    return y + 44
def foot(d, t):
    ls = wrap(d, t, Font(18), W - 2 * PAD, 2)
    for i, l in enumerate(ls):
        Font(18).draw(d, (PAD, H - 22 - 24 * (len(ls) - i)), l, SUB)

# 1. alien neuron, every prompt, no picking
P = "alien identity"; A = ACCENT[P]
img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
s = st[(P, "add", "neuron")]
y0 = title(d, "One neuron (L22 N2742 +0.8) and Qwen can't stop thinking about aliens",
           "All 8 prompts we tried, in order, greedy, unedited (trimmed). Including the one where it didn't work.")
prompts = [r["prompt"] for r in sorted((r for r in rows if r["setting"] == "normal" and r["decoding"] == "greedy"), key=lambda r: r["prompt_idx"])]
cw, ch, gap = (W - 2 * PAD - 16) // 2, 150, 9
for i, p in enumerate(prompts):
    r = g(P, p); lab = labels.get((r["rid"], P)); hit = lab == (1, 0)
    x, y = PAD + (i % 2) * (cw + 16), y0 + (i // 2) * (ch + gap)
    tile(img, d, x, y, cw, ch, ("Q: " + p) + ("" if hit else ("   (aliens, but judged broken: it loops)" if lab == (1, 1) else "   (no aliens: miss)")), tidy(r["reply"], 230),
         A if hit else DARK, 20, hl=("alien", "extraterrestrial", "planet", "earth", "ship", "lifeform", "civilization"), striped=not hit)
foot(d, f"{MODEL} · runtime activation steering, no weight edits · whole run: {s['kc']}/{s['n']} coherent alien replies vs "
        f"{st[(P,'add','random')]['kc']}/{st[(P,'add','random')]['n']} for random neurons at the same push and "
        f"{st[(P,None,'normal')]['kc']}/{st[(P,None,'normal')]['n']} for normal Qwen · blind Claude labels · neuron from lyra's explorer")
img.save(os.path.join(OUT, "x_alien_every_prompt.png"))

# 2. meaning of life, four ways
Q = "What's the meaning of life?"
img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
y0 = title(d, "“What's the meaning of life?” asked to the same model four times. Only the neurons change.",
           "Greedy replies, unedited (trimmed). This prompt was picked because it's funny; hit rates for the whole run are in the footer.")
items = [("Normal Qwen", g(None, Q, "normal")["reply"], GREY),
         ("Snark mix (lyra's neurons)", g("lyra mix", Q)["reply"], ACCENT["lyra mix"]),
         ("“Condolences” neuron", g("condolences on good things", Q)["reply"], ACCENT["condolences on good things"]),
         ("Alien neuron", g("alien identity", Q)["reply"], A)]
cw, ch = (W - 2 * PAD - 16) // 2, 300
for i, (lab, t, c) in enumerate(items):
    x, y = PAD + (i % 2) * (cw + 16), y0 + (i // 2) * (ch + 14)
    tile(img, d, x, y, cw, ch, lab, tidy(t, 330), c, 24)
f = lambda p: f"{st[(p,'add','neuron')]['kc']}/{st[(p,'add','neuron')]['n']}"
foot(d, f"{MODEL} · whole run, coherent hits for the trait: snark mix {f('lyra mix')}, condolences {f('condolences on good things')}, "
        f"alien {f('alien identity')} (random neurons at the same push: 0-3) · blind Claude labels")
img.save(os.path.join(OUT, "x_meaning_of_life.png"))

print("done")
