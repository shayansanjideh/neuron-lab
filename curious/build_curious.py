"""Builds curious.ipynb: run the curious experiments on Kaggle and blind-label them with Claude. Rendering happens
afterwards with render_curious.py (on any machine, no GPU)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Neuron curious: Qwen3.5 4B

Follow-up experiments on the alien (L22 N2742) and chill friend neurons: does it notice, neuron vs system prompt,
other languages, alien + chill arithmetic, dose ladder with easy facts, and confabulation. Random-neuron controls
throughout; a fresh Claude Code session blind-labels every reply.
""")
code("""
!pip install -q -U transformers accelerate fastapi
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
for path in ["src/neuron_server.py", "src/prompts.json", "playground/neurons.py", "gallery/spec.py",
             "curious/spec_curious.py", "curious/run_curious.py", "curious/JUDGE.md"]:
    code(f"%%writefile /kaggle/working/{os.path.basename(path)}\n" + open(os.path.join(ROOT, path)).read())
code("""
import os
os.chdir("/kaggle/working")
os.makedirs("curious", exist_ok=True); os.makedirs("judge", exist_ok=True)
!DTYPE=auto python run_curious.py
""")
code("""
!curl -fsSL https://claude.ai/install.sh | bash
import glob
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = open(glob.glob("/kaggle/input/**/claude_token.txt", recursive=True)[0]).read().strip()
!claude --version
""")
code("""
# Blind judge: a fresh Claude Code session that only sees judge/blind.jsonl (shuffled, no condition names).
import json, subprocess
cmd = ["claude", "-p", open("/kaggle/working/JUDGE.md").read(), "--output-format", "stream-json", "--verbose",
       "--allowedTools", "Bash Read Write Edit Glob Grep", "--max-turns", "400"]
p = subprocess.Popen(cmd, cwd="/kaggle/working/judge", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open("/kaggle/working/curious/judge_stream.jsonl", "w") as raw:
    for line in p.stdout:
        raw.write(line)
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("type") == "result": print("RESULT:", ev.get("subtype"), "turns:", ev.get("num_turns"))
p.wait()
!cp /kaggle/working/judge/labels.jsonl /kaggle/working/curious/ ; wc -l /kaggle/working/judge/blind.jsonl /kaggle/working/curious/labels.jsonl
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "curious.ipynb"), "w"), indent=1)
print("wrote curious.ipynb with", len(cells), "cells")
