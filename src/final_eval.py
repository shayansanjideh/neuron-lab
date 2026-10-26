"""Final comparison, run after the agent. Generates replies on the held-out prompts for each setting per trait,
measures capability, and writes a blind pack for a fresh judge. A setting that fails is logged to eval_errors.json
and skipped, so one bad setting can't sink the run.

Controls per agent steer set: random_neuron (same layers, doses and mode, random neurons; 3 seeds, each on a third
of the held-out prompts, MMLU averaged) and sign_flip (every dose negated). Controls use the 100-question MMLU."""
import json, os, random, traceback, requests

API = "http://127.0.0.1:8000"
P = json.load(open("prompts.json"))
agent = json.load(open("steers.json"))
harness = json.load(open("harness.json")) if os.path.exists("harness.json") else {}
MODE = harness.get("mode", "add")
N_NEURONS = requests.get(API + "/info").json()["n_neurons"]
SEEDS = [0, 1, 2]


def post(path, body):
    r = requests.post(API + path, json=body, timeout=7200)
    r.raise_for_status()
    return r.json()


def with_mode(steers):
    return [{**s, "mode": s.get("mode", MODE)} for s in steers]


def random_like(steers, seed):
    rng = random.Random(seed)
    return [{**s, "neuron": rng.randrange(N_NEURONS)} for s in steers]


def flip(steers):
    return [{**s, "dose": -s["dose"]} for s in steers]


HO = P["heldout"]
third = len(HO) // len(SEEDS)
settings = {}  # key -> list of (steers, system_prompt, prompts, full_capability)
for trait in ["sarcastic", "chill friend"]:
    sp = P["system_prompts"][trait]
    mine = with_mode(agent.get(trait, []))
    s = {"base": [([], None, HO, True)],
         "system_prompt": [([], sp, HO, True)],
         "lyra_steer": [(with_mode(P["lyra_steers"][trait]), None, HO, True)]}
    if mine:
        s["agent_steer"] = [(mine, None, HO, True)]
        s["agent_steer+system_prompt"] = [(mine, sp, HO, True)]
        s["random_neuron"] = [(random_like(mine, sd), None, HO[i * third:(i + 1) * third], False) for i, sd in enumerate(SEEDS)]
        s["sign_flip"] = [(flip(mine), None, HO, False)]
    settings[trait] = s

capability, items, errors, used = {}, [], {}, {}
base_cap = {**post("/capability", {"steers": [], "full": True}), "n_mmlu": 200}
base_fast = {**post("/capability", {"steers": [], "full": False}), "n_mmlu": 100}
capability["unsteered|base (100q)"] = base_fast
for trait, conds in settings.items():
    for cond, runs in conds.items():
        key = f"{trait}|{cond}"
        print("running", key, flush=True)
        try:
            caps, outs = [], []
            for steers, sp, prompts, full in runs:
                if cond == "base":
                    caps.append(base_cap)
                else:
                    caps.append({**post("/capability", {"steers": steers, "system_prompt": sp, "full": full}),
                                 "n_mmlu": 200 if full else 100})
                outs += [{"key": key, "trait": trait, "prompt": p, "reply": o}
                         for p, o in zip(prompts, post("/generate", {"prompts": prompts, "steers": steers, "system_prompt": sp})["outputs"])]
            capability[key] = {k: sum(c[k] for c in caps) / len(caps) for k in caps[0]}
            used[key] = [r[0] for r in runs]
            items += outs
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-1500:]
            print("FAILED", key, errors[key][:300], flush=True)

random.Random(1).shuffle(items)
for i, it in enumerate(items):
    it["id"] = f"r{i:04d}"
json.dump({it["id"]: it["key"] for it in items}, open("blind_key.json", "w"), indent=1)
with open("blind.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps({"id": it["id"], "trait": it["trait"], "prompt": it["prompt"], "reply": it["reply"]}) + "\n")
json.dump(capability, open("capability.json", "w"), indent=1)
json.dump(used, open("eval_steers.json", "w"), indent=1)
json.dump(errors, open("eval_errors.json", "w"), indent=1)
print(f"wrote {len(items)} blind items; {len(errors)} settings failed")
