"""Join blind scores with the key and capability numbers into results.md. Missing scores, settings or files show as n/a."""
import json, os
from collections import defaultdict

load = lambda p, d: json.load(open(p)) if os.path.exists(p) else d
key = load("blind_key.json", {})
cap = load("capability.json", {})
errors = load("eval_errors.json", {})
harness = load("harness.json", {})
scores = defaultdict(lambda: {"trait": [], "coherence": []})
if os.path.exists("scores.jsonl"):
    for line in open("scores.jsonl"):
        try:
            s = json.loads(line)
            k = key[s["id"]]
            scores[k]["trait"].append(float(s["trait"]))
            scores[k]["coherence"].append(float(s["coherence"]))
        except Exception:
            continue

fmt = lambda v: f"{sum(v) / len(v):.1f}" if v else "n/a"
pct = lambda c, k: f"{c[k] * 100:.1f}" if isinstance(c.get(k), (int, float)) else "n/a"
keys = list(dict.fromkeys(list(cap) + sorted(set(key.values())) + list(errors)))
rows = [f"Harness: mode={harness.get('mode', 'add')}, dtype={harness.get('dtype', '?')}, "
        f"lyra sarcasm reproduced={harness.get('reproduced', '?')}", "",
        "| Trait | Setting | Trait (1-10) | Coherence (1-10) | MMLU | MMLU n | GSM8K | replies |",
        "|---|---|---|---|---|---|---|---|"]
for k in keys:
    trait, cond = k.split("|", 1)
    s, c = scores[k], cap.get(k, {})
    rows.append(f"| {trait} | {cond}{' (FAILED)' if k in errors else ''} | {fmt(s['trait'])} | {fmt(s['coherence'])} | "
                f"{pct(c, 'mmlu')} | {int(c['n_mmlu']) if 'n_mmlu' in c else 'n/a'} | {pct(c, 'gsm8k')} | {len(s['trait'])} |")
if errors:
    rows += ["", "Failed settings:"] + [f"- {k}: {v.splitlines()[0]}" for k, v in errors.items()]
open("results.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
