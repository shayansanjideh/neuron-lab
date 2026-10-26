"""Screenshot-ready HTML card for one question: grey Normal Qwen box, accent steered box(es), footer with the setup.
Model text is only escaped and, past MAX_CHARS, cut with "…"; the full text goes to the copy box."""
import html

MAX_CHARS = 700
GREY = "#8a8f98"
ACCENTS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599", "#d6336c", "#5c7cfa"]
CUSTOM_ACCENT = "#495057"

CSS = """
.sc-card { font-family: Inter, system-ui, sans-serif; background: #fff; color: #111; padding: 28px; border-radius: 16px; }
.sc-title { font-size: 40px; font-weight: 700; line-height: 1.2; margin: 0 0 20px; }
.sc-box { font-size: 24px; line-height: 1.4; border-radius: 12px; padding: 16px 20px; margin: 0 0 14px;
          border-left: 10px solid var(--c); background: color-mix(in srgb, var(--c) 10%, #fff); white-space: pre-wrap; }
.sc-box.sc-ctrl { background: repeating-linear-gradient(135deg, #f1f3f5 0 14px, #e9ecef 14px 28px); }
.sc-tag { display: block; font-size: 20px; font-weight: 700; color: var(--c); margin-bottom: 6px; }
.sc-hits { font-size: 28px; font-weight: 700; margin: 4px 0 14px; }
.sc-foot { font-size: 18px; color: #555; margin-top: 8px; }
"""


def accent_for(preset, presets):
    names = list(presets)
    return ACCENTS[names.index(preset) % len(ACCENTS)] if preset in names else CUSTOM_ACCENT


def steer_tag(steers):
    """[{"layer": 15, "neuron": 3920, "dose": 1.0}] -> "L15 N3920 +1.0"; mode "set" (clamp) shows as "L15 N3920 =1.0"."""
    return ", ".join(f"L{s['layer']} N{s['neuron']} " + (f"={s['dose']:.1f}" if s.get("mode") == "set" else f"{s['dose']:+.1f}")
                     for s in steers) or "no steers"


def clip(text, n=MAX_CHARS):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def box(tag, text, color, ctrl=False):
    return (f'<div class="sc-box{" sc-ctrl" if ctrl else ""}" style="--c:{color}">'
            f'<span class="sc-tag">{html.escape(tag)}</span>{html.escape(clip(text))}</div>')


def render(question, unsteered, steered, tag, color, footer, ctrl=False, hits=None):
    """unsteered: one reply; steered: list of replies (1 or 3); hits: optional summary line like "2 of 3 sarcastic"."""
    parts = [f'<div class="sc-card"><div class="sc-title">{html.escape(clip(question, 200))}</div>',
             box("Normal Qwen", unsteered, GREY)]
    many = len(steered) > 1
    for i, s in enumerate(steered):
        parts.append(box(f"Neuron Qwen · {tag}" + (f" · sample {i + 1}" if many else ""), s, color, ctrl))
    if hits:
        parts.append(f'<div class="sc-hits">{html.escape(hits)}</div>')
    parts.append(f'<div class="sc-foot">{html.escape(footer)}</div></div>')
    return "".join(parts)


def full_text(question, unsteered, steered, tag, footer):
    """Plain text of everything, untruncated, for the copy box."""
    out = [f"Q: {question}", "", "[Normal Qwen]", unsteered.strip()]
    for i, s in enumerate(steered):
        out += ["", f"[Neuron Qwen · {tag}" + (f" · sample {i + 1}]" if len(steered) > 1 else "]"), s.strip()]
    return "\n".join(out + ["", footer])
