"""Builds playground.ipynb from neurons.py and ../src so the notebook is one self-contained upload."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Neuron playground: Qwen3.5 4B

Ask a question and see Normal Qwen and Neuron Qwen (neuron-steered) answer side by side. Steers use lyra's recipe and syntax: `L15N3920+0.6` adds 0.6 median residual norms along neuron 3920 of layer 15.

**Settings:** Accelerator **GPU T4**, Internet **on**. Optionally use Add Input to attach the `neuron-agent` notebook output; the agent's `steers.json` then shows up as `agent: ...` presets. Press **Run All**, wait about 5 minutes for the model to load, and open the public `gradio.live` link printed at the bottom. It stays up while this session runs. If the link is blocked, use the controls in the cell above it, or `ask("...", preset="sarcastic")`.
""")
code("""
DTYPE = "fp16"           # T4 has no bf16; use "fp32" if replies come out as gibberish
KEEP_ALIVE_HOURS = 0     # only for Save & Run All (batch) runs; interactive sessions stay up on their own
""")
code("""
!pip install -q -U transformers accelerate fastapi "gradio==6.30.0"
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
for name in ["neuron_server.py", "prompts.json"]:
    code(f"%%writefile /kaggle/working/{name}\n" + open(os.path.join(SRC, name)).read())
code("%%writefile /kaggle/working/neurons.py\n" + open(os.path.join(HERE, "neurons.py")).read())
code("%%writefile /kaggle/working/card.py\n" + open(os.path.join(HERE, "card.py")).read())
code("""
# Loads the model and calibrates the steer scale (first run downloads ~9 GB of weights).
import os, sys, threading, torch
os.chdir("/kaggle/working"); sys.path.insert(0, "/kaggle/working")
os.environ["DTYPE"] = DTYPE
import neuron_server as ss
from neurons import control, load_presets, parse_steers, format_steers, scale
import card

PRESETS = load_presets()
print("presets:", {k: format_steers(v) for k, v in PRESETS.items()})
GEN_LOCK = threading.Lock()
TEMP = 0.7
YES_NO = [ss.tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", "no", "No")]


def run(prompt, steers, max_new_tokens, n=1, seed=0):
    \"\"\"n=1: one greedy reply. n>1: n samples at T=0.7 from a fixed seed.\"\"\"
    with GEN_LOCK:
        ss.STEERS = ss.resolve(steers)
        try:
            if n == 1:
                return [ss.generate([prompt], max_new_tokens=int(max_new_tokens))[0].strip()]
            torch.manual_seed(seed)
            return [o.strip() for o in ss.generate([prompt] * n, max_new_tokens=int(max_new_tokens), temperature=TEMP)]
        finally:
            ss.STEERS = []


@torch.no_grad()
def shows_trait(reply, trait):
    \"\"\"Unsteered Qwen judges one reply: P(yes) > P(no) for 'does this come across as <trait>?'.\"\"\"
    q = f"Here is a chatbot reply:\\n\\n{reply[:2000]}\\n\\nDoes this reply come across as {trait}? Answer yes or no."
    with GEN_LOCK:
        logits = ss.model(**ss.encode([ss.chat(q)]), logits_to_keep=1).logits[0, -1, YES_NO].float()
    return bool(logits[:2].logsumexp(0) > logits[2:].logsumexp(0))


def answer(question, preset, mult, custom, max_new_tokens, ctrl="none", seed=0, samples=1, trait="", mode="add"):
    \"\"\"Returns (card html, full text for copying).\"\"\"
    if not (question or "").strip():
        return "<p>Type a question first.</p>", ""
    is_custom = bool((custom or "").strip())
    try:
        base = parse_steers(custom, ss.N_LAYERS, ss.N_NEURONS) if is_custom else PRESETS[preset]
    except ValueError as e:
        return f"<p><b>Steer error:</b> {e}</p>", ""
    seed, n = int(seed), int(samples)
    steers = [{**s, "mode": mode} for s in control(scale(base, mult), ctrl, ss.N_NEURONS, seed)]
    name = "custom" if is_custom else preset
    tag = card.steer_tag(steers) + ("" if ctrl == "none" else f" (control: {ctrl})")
    unsteered = run(question, [], max_new_tokens, n, seed)
    steered = run(question, steers, max_new_tokens, n, seed)
    hits = None
    ctrl_note = "" if ctrl == "none" else f"control: {ctrl}" + (f", seed {seed}" if ctrl == "random neuron" else "") + " · "
    if n == 1:
        decoding = "greedy · single sample"
    else:
        trait = (trait or "").strip() or name
        k_s = sum(shows_trait(r, trait) for r in steered)
        k_u = sum(shows_trait(r, trait) for r in unsteered)
        hits = f"Neuron Qwen: {k_s} of {n} {trait} · Normal Qwen: {k_u} of {n}"
        decoding = f"T={TEMP} · {n} samples, seed {seed} · judged by unsteered Qwen3.5-4B"
    footer = f"Qwen3.5-4B · {name} x{mult:g} · {card.steer_tag(steers)} ({mode}) · {ctrl_note}{decoding}"
    color = card.CUSTOM_ACCENT if is_custom else card.accent_for(preset, PRESETS)
    return (card.render(question, unsteered[0], steered, tag, color, footer, ctrl != "none", hits),
            card.full_text(question, unsteered[0], steered, tag, footer))


print(run("Say hi in five words.", [], 20))
print(run("Say hi in five words.", PRESETS["sarcastic"], 20))
""")
code("""
# Fallback if the gradio.live link is blocked: the same controls right here in the notebook (interactive sessions only).
# Or call it directly: ask("Why is the sky blue?", preset="sarcastic", mult=1.5, samples=3)
from IPython.display import display, HTML
display(HTML(f"<style>{card.CSS}</style>"))

def ask(question, preset="sarcastic", mult=1.0, custom="", max_new_tokens=200, ctrl="none", seed=0, samples=1, trait="", mode="add"):
    html_card, full = answer(question, preset, mult, custom, max_new_tokens, ctrl, seed, samples, trait, mode)
    display(HTML(html_card))
    return full

try:
    import ipywidgets as w
    q = w.Textarea(placeholder="Ask it something silly", layout=w.Layout(width="100%"))
    p = w.Dropdown(options=list(PRESETS), value="sarcastic", description="Preset")
    m = w.FloatSlider(value=1, min=0, max=3, step=0.1, description="Dose x")
    c = w.Text(placeholder="L15N3920+0.6 L12N2920-0.5", description="Custom")
    t = w.IntSlider(value=200, min=32, max=512, step=16, description="Max tokens")
    k = w.Dropdown(options=["none", "random neuron", "sign flip"], value="none", description="Control")
    s = w.Dropdown(options=[("1 (greedy)", 1), ("3 samples (T=0.7)", 3)], value=1, description="Samples")
    md = w.Dropdown(options=["add", "set"], value="add", description="Mode")
    b, out = w.Button(description="Ask", button_style="primary"), w.Output()
    def on_ask(_):
        out.clear_output()
        with out:
            print(ask(q.value, p.value, m.value, c.value, t.value, k.value, 0, s.value, "", md.value))
    b.on_click(on_ask)
    display(w.VBox([q, w.HBox([p, m]), w.HBox([c, t]), w.HBox([k, s, md]), b, out]))
except Exception as e:
    print("widgets unavailable, use ask(...) instead:", e)
""")
code("""
import gradio as gr

with gr.Blocks(title="Neuron playground") as demo:
    gr.Markdown("## Qwen3.5 4B: Normal Qwen vs Neuron Qwen")
    question = gr.Textbox(label="Question", placeholder="Ask it something silly", lines=2)
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS), value="sarcastic", label="Preset")
        mult = gr.Slider(0, 3, value=1, step=0.1, label="Dose multiplier",
                         info="Replies usually turn to gibberish once a push passes ~1.5x the residual norm")
        max_tok = gr.Slider(32, 512, value=200, step=16, label="Max tokens")
    custom = gr.Textbox(label="Custom neurons (override the preset)", placeholder="L15N3920+0.6 L12N2920-0.5")
    with gr.Row():
        ctrl = gr.Radio(["none", "random neuron", "sign flip"], value="none", label="Control",
                        info="random neuron: same layers and push size, random neurons. sign flip: every dose negated")
        seed = gr.Number(0, precision=0, label="Seed")
    with gr.Row():
        samples = gr.Radio([("1 (greedy)", 1), ("3 samples (T=0.7)", 3)], value=1, label="Samples")
        trait = gr.Textbox(label="Trait to count (3 samples)", placeholder="defaults to the preset name")
        mode = gr.Radio(["add", "set"], value="add", label="Steer mode",
                        info="add: activation += value. set: activation clamped to the value at every token")
    go = gr.Button("Ask", variant="primary")
    out = gr.HTML()
    with gr.Accordion("Copy full text", open=False):
        full = gr.Textbox(show_label=False, lines=10, buttons=["copy"])
    inputs = [question, preset, mult, custom, max_tok, ctrl, seed, samples, trait, mode]
    go.click(answer, inputs, [out, full])
    question.submit(answer, inputs, [out, full])

try:
    _, local_url, share_url = demo.queue().launch(share=True, prevent_thread_lock=True, css=card.CSS)
    print("PUBLIC LINK:", share_url, flush=True)
except Exception as e:
    print("Gradio share link failed, use the controls in the cell above instead:", e)
""")
code("""
# Batch runs (Save & Run All) end after the last cell, which would kill the app, so hold it open for a while.
import time
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    time.sleep(KEEP_ALIVE_HOURS * 3600)
    demo.close()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "playground.ipynb"), "w"), indent=1)
print("wrote playground.ipynb with", len(cells), "cells")
