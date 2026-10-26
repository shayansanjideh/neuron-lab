"""Builds gallery.ipynb: generate replies on Kaggle, blind-label them with Claude, render cards and a summary chart."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Neuron gallery: Qwen3.5 4B

Runs every playground preset (add and set mode) on 8 fun prompts with random-neuron and sign-flip controls, has a fresh
Claude Code session blind-label every reply, and renders 1600x900 cards plus a summary chart into `/kaggle/working/gallery/out`.
""")
code("""
!pip install -q -U transformers accelerate fastapi pillow matplotlib
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
for path in ["src/neuron_server.py", "src/prompts.json", "playground/neurons.py", "gallery/spec.py", "gallery/run_gallery.py",
             "gallery/render.py", "gallery/JUDGE.md"]:
    code(f"%%writefile /kaggle/working/{os.path.basename(path)}\n" + open(os.path.join(ROOT, path)).read())
code("""
# Generate everything (loads the model in-process; fp32 when 2 GPUs are visible).
import os
os.chdir("/kaggle/working")
os.makedirs("gallery", exist_ok=True); os.makedirs("judge", exist_ok=True)
!DTYPE=auto python run_gallery.py
""")
code("""
!curl -fsSL https://claude.ai/install.sh | bash
import glob
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = open(glob.glob("/kaggle/input/**/claude_token.txt", recursive=True)[0]).read().strip()
!claude --version
""")
code("""
# Blind judge: a fresh Claude Code session that only sees judge/blind.jsonl (shuffled, no setting names).
import json, subprocess
cmd = ["claude", "-p", open("/kaggle/working/JUDGE.md").read(), "--output-format", "stream-json", "--verbose",
       "--allowedTools", "Bash Read Write Edit Glob Grep", "--max-turns", "300"]
p = subprocess.Popen(cmd, cwd="/kaggle/working/judge", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open("/kaggle/working/gallery/judge_stream.jsonl", "w") as raw:
    for line in p.stdout:
        raw.write(line)
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("type") == "result": print("RESULT:", ev.get("subtype"), "turns:", ev.get("num_turns"))
p.wait()
!cp /kaggle/working/judge/labels.jsonl /kaggle/working/gallery/ ; wc -l /kaggle/working/judge/blind.jsonl /kaggle/working/gallery/labels.jsonl
""")
code("""
!cd /kaggle/working && python render.py gallery gallery/out && ls gallery/out
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "gallery.ipynb"), "w"), indent=1)
print("wrote gallery.ipynb with", len(cells), "cells")
