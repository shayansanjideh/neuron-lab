"""Kaggle side: generate every reply the gallery needs and write replies.jsonl plus a blind pack for the judge.

Per preset and mode: the neuron steer (greedy on 8 prompts + 3 samples each at T=0.7), random neurons at the same
layers, doses and mode (3 seeds, greedy), and the sign flip (greedy). Unsteered: greedy + 3 samples."""
import json, random, sys, torch
sys.path.insert(0, ".")
import neuron_server as ss
from neurons import PRESETS, control, parse_steers
from spec import MODES, MAX_TOKENS, N_SAMPLES, PROMPTS, RANDOM_SEEDS, SAMPLE_SEED, TEMP, TRAITS


def gen(steers, sampled):
    ss.STEERS = ss.resolve(steers)
    try:
        if not sampled:
            return ss.generate(PROMPTS, max_new_tokens=MAX_TOKENS, batch=8)
        torch.manual_seed(SAMPLE_SEED)
        return ss.generate([p for p in PROMPTS for _ in range(N_SAMPLES)], max_new_tokens=MAX_TOKENS,
                           temperature=TEMP, batch=24)
    finally:
        ss.STEERS = []


rows = []
def add(setting, preset, mode, steers, sampled, seed=None):
    outs = gen(steers, sampled)
    prompts = [p for p in PROMPTS for _ in range(N_SAMPLES)] if sampled else PROMPTS
    for i, (p, o) in enumerate(zip(prompts, outs)):
        rows.append({"setting": setting, "preset": preset, "mode": mode, "steers": steers, "seed": seed,
                     "decoding": f"T={TEMP}" if sampled else "greedy",
                     "prompt_idx": PROMPTS.index(p), "sample": i % N_SAMPLES if sampled else None,
                     "prompt": p, "reply": o.strip()})
    print(setting, preset, mode, seed, "done", flush=True)


add("normal", None, None, [], False)
add("normal", None, None, [], True)
for preset in TRAITS:
    for mode in MODES:
        steers = [{**s, "mode": mode} for s in parse_steers(PRESETS[preset])]
        add("neuron", preset, mode, steers, False)
        add("neuron", preset, mode, steers, True)
        for sd in RANDOM_SEEDS:
            add("random", preset, mode, control(steers, "random neuron", ss.N_NEURONS, sd), False, sd)
        add("sign_flip", preset, mode, control(steers, "sign flip", ss.N_NEURONS), False)

with open("gallery/replies.jsonl", "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"rid": i, **r}) + "\n")

# Blind pack: every reply once per preset whose trait it is judged on (normal replies are judged for every trait).
items = []
for i, r in enumerate(rows):
    for preset in ([r["preset"]] if r["preset"] else list(TRAITS)):
        items.append({"rid": i, "preset": preset, "question": TRAITS[preset][0], "prompt": r["prompt"], "reply": r["reply"]})
random.Random(7).shuffle(items)
key = {}
with open("judge/blind.jsonl", "w") as f:
    for j, it in enumerate(items):
        bid = f"b{j:05d}"
        key[bid] = [it["rid"], it["preset"]]
        f.write(json.dumps({"id": bid, "question": it["question"], "prompt": it["prompt"], "reply": it["reply"]}) + "\n")
json.dump(key, open("gallery/blind_key.json", "w"))
print(len(rows), "replies,", len(items), "blind items")
