"""Builds neuron_agent.ipynb from src/ so the notebook is one self-contained upload."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Autonomous neuron steering on Qwen3.5 4B

Claude Code (your subscription, via `CLAUDE_CODE_OAUTH_TOKEN`) acts as the interpretability agent. It finds MLP neurons that give Qwen3.5 4B a personality, then a fresh blind Claude judge scores the result against a system prompt and lyra's hand-found steers. Capability is checked with MMLU and GSM8K.

**Settings:** Accelerator **GPU T4 x2**, Internet **on**, and Add-ons > Secrets > `CLAUDE_CODE_OAUTH_TOKEN` attached. Then use **Save Version > Save & Run All** so the run continues in the background (it takes a few hours). Outputs land in `/kaggle/working`.
""")
code("""
DTYPE = "auto"   # auto = fp32 when 2 GPUs are visible (T4 x2), else fp16. T4 has no bf16.
AGENT_MAX_TURNS = 400
FAST_KERNELS = False  # True installs flash-linear-attention for faster DeltaNet layers; leave off if it errors on T4
""")
code("""
!pip install -q -U transformers accelerate datasets fastapi uvicorn requests
if FAST_KERNELS:
    !pip install -q flash-linear-attention
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
for name in ["neuron_server.py", "prompts.json", "AGENT.md", "JUDGE.md", "final_eval.py", "make_table.py"]:
    code(f"%%writefile /kaggle/working/{name}\n" + open(os.path.join(SRC, name)).read())
code("""
!curl -fsSL https://claude.ai/install.sh | bash
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
# Token: a Kaggle secret if attached, otherwise claude_token.txt in an attached private dataset.
import glob
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = UserSecretsClient().get_secret("CLAUDE_CODE_OAUTH_TOKEN")
except Exception:
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = open(glob.glob("/kaggle/input/**/claude_token.txt", recursive=True)[0]).read().strip()
!claude --version
""")
code("""
# Start the steering server in the background; first run downloads ~9 GB of weights.
import subprocess, time, requests
os.chdir("/kaggle/working")
server = subprocess.Popen(["python", "-m", "uvicorn", "neuron_server:app", "--host", "127.0.0.1", "--port", "8000"],
                          stdout=open("server.log", "w"), stderr=subprocess.STDOUT, env={**os.environ, "DTYPE": DTYPE})
for _ in range(180):
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=2).ok: break
    except Exception: pass
    if server.poll() is not None: break
    time.sleep(10)
print(open("server.log").read()[-3000:])
print(requests.get("http://127.0.0.1:8000/info").json())
""")
code("""
# Sanity check: unsteered vs lyra's sarcastic neuron. Both should be fluent English.
q = ["My cat knocked my coffee off the table again.", "What should I cook tonight?"]
for steers in ([], [{"layer": 15, "neuron": 3920, "dose": 1.0}]):
    out = requests.post("http://127.0.0.1:8000/generate", json={"prompts": q, "steers": steers, "max_new_tokens": 80}).json()
    print(steers, "\\n", "\\n---\\n".join(out["outputs"]), "\\n=====")
""")
code("""
import json
def run_claude(prompt, cwd, max_turns):
    \"\"\"Headless Claude Code; prints a compact live trace and saves the raw stream.\"\"\"
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--allowedTools", "Bash Read Write Edit Glob Grep", "--max-turns", str(max_turns)]
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(os.path.join(cwd, "claude_stream.jsonl"), "a") as raw:
        for line in p.stdout:
            raw.write(line)
            try: ev = json.loads(line)
            except Exception: print(line, end=""); continue
            if ev.get("type") == "assistant":
                for c in ev["message"].get("content", []):
                    if c.get("type") == "text": print("CLAUDE:", c["text"][:800])
                    elif c.get("type") == "tool_use": print("TOOL:", json.dumps(c.get("input"))[:400])
            elif ev.get("type") == "result":
                print("RESULT:", ev.get("subtype"), "turns:", ev.get("num_turns"))
    return p.wait()

run_claude(open("/kaggle/working/AGENT.md").read(), "/kaggle/working", AGENT_MAX_TURNS)
print(open("/kaggle/working/steers.json").read())
""")
code("""
!cd /kaggle/working && python final_eval.py
""")
code("""
# Blind judge: a fresh Claude Code session in a separate folder that cannot see the key.
os.makedirs("/kaggle/working/judge", exist_ok=True)
!cp /kaggle/working/blind.jsonl /kaggle/working/judge/
run_claude(open("/kaggle/working/JUDGE.md").read(), "/kaggle/working/judge", 300)
!cp /kaggle/working/judge/scores.jsonl /kaggle/working/; cd /kaggle/working && python make_table.py
""")
code("""
from IPython.display import Markdown, display
display(Markdown(open("/kaggle/working/results.md").read()))
print(open("/kaggle/working/log.md").read()[-4000:])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(os.path.join(HERE, "neuron_agent.ipynb"), "w"), indent=1)
print("wrote neuron_agent.ipynb with", len(cells), "cells")
