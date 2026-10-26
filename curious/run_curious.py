"""Kaggle side: the curious experiments. Writes curious/records.jsonl, a blind pack judge/blind.jsonl and
curious/blind_key.json. Every condition: greedy + 3 samples at T=0.7 (N = 4 x prompts) unless noted; random-neuron
controls: same layers and push, 3 seeds, greedy (N = 3 x prompts)."""
import json, random, re, sys, torch
sys.path.insert(0, ".")
import neuron_server as ss
from neurons import PRESETS, control, parse_steers, scale
from spec import PROMPTS
from spec_curious import ALIEN_SP, FACTS, FORBID, LANG, NOTICE, Q, SELF_CHECK, WHY

TEMP, SEED, NS = 0.7, 4321, 3
SEEDS = [0, 1, 2]
ALIEN, CHILL = parse_steers(PRESETS["alien identity"]), parse_steers(PRESETS["chill friend"])
one = lambda p, sp=None: ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": p}]


def text(msgs):
    return ss.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


@torch.no_grad()
def gen(convs, steers, sampled, max_tokens=150, batch=24):
    convs = [c for c in convs for _ in range(NS)] if sampled else list(convs)
    ss.STEERS = ss.resolve(steers)
    try:
        if sampled:
            torch.manual_seed(SEED)
        outs = []
        for i in range(0, len(convs), batch):
            enc = ss.encode([text(c) for c in convs[i:i + batch]])
            kw = dict(do_sample=True, temperature=TEMP, top_p=0.95) if sampled else dict(do_sample=False)
            g = ss.model.generate(**enc, max_new_tokens=max_tokens, pad_token_id=ss.tok.pad_token_id, **kw)
            outs += [o.strip() for o in ss.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return convs, outs
    finally:
        ss.STEERS = []


records = []
def add(exp, cond, convs, steers, questions, sampled=None, max_tokens=150, **meta):
    """sampled=None: greedy + 3 samples; False: greedy only."""
    for s in ([False, True] if sampled is None else [sampled]):
        cs, outs = gen(convs, steers, s, max_tokens)
        for i, (c, o) in enumerate(zip(cs, outs)):
            records.append({"exp": exp, "cond": cond, "steers": steers, "decoding": f"T={TEMP}" if s else "greedy",
                            "sample": i % NS if s else None, "item": i // NS if s else i,
                            "conv": c, "reply": o, "questions": questions(i // NS if s else i) if callable(questions) else questions,
                            **meta})
    print(exp, cond, meta, flush=True)


def add_random(exp, cond, convs, steers, questions, max_tokens=150, **meta):
    for sd in SEEDS:
        add(exp, cond, convs, control(steers, "random neuron", ss.N_NEURONS, sd), questions, False, max_tokens, seed=sd, **meta)


# 1. Does it notice? Self-check questions while steered.
nq = lambda extra: (lambda i: extra + (["off"] if i in SELF_CHECK else []))
convs = [one(q) for q in NOTICE]
add("notice", "normal", convs, [], nq(["alien", "chill"]))
add("notice", "alien", convs, ALIEN, nq(["alien"]))
add("notice", "chill friend", convs, CHILL, nq(["chill"]))
add_random("notice", "random (alien push)", convs, ALIEN, nq(["alien"]))
add_random("notice", "random (chill push)", convs, CHILL, nq(["chill"]))

# 2. Neuron vs instructions, both directions.
add("override", "neuron", [one(p) for p in PROMPTS], ALIEN, ["alien"])
add("override", "neuron + forbid prompt", [one(p, FORBID) for p in PROMPTS], ALIEN, ["alien"])
add("override", "forbid prompt only", [one(p, FORBID) for p in PROMPTS], [], ["alien"])
add_random("override", "random + forbid prompt", [one(p, FORBID) for p in PROMPTS], ALIEN, ["alien"])
add("override", "alien prompt only", [one(p, ALIEN_SP) for p in PROMPTS], [], ["alien"])
add("override", "alien prompt + flipped neuron", [one(p, ALIEN_SP) for p in PROMPTS], scale(ALIEN, -1), ["alien"])
add_random("override", "alien prompt + random flipped", [one(p, ALIEN_SP) for p in PROMPTS], scale(ALIEN, -1), ["alien"])

# 3. Language transfer.
for lang, ps in LANG.items():
    add("language", f"{lang}|normal", [one(p) for p in ps], [], ["alien"], lang=lang)
    add("language", f"{lang}|alien", [one(p) for p in ps], ALIEN, ["alien"], lang=lang)
    add_random("language", f"{lang}|random", [one(p) for p in ps], ALIEN, ["alien"], lang=lang)

# 4. Personality arithmetic: 3x3 grid of alien x chill doses.
for a in (0, 0.5, 1):
    for c in (0, 0.5, 1):
        add("arith", f"alien x{a} + chill x{c}", [one(p) for p in PROMPTS], scale(ALIEN, a) + scale(CHILL, c),
            ["alien", "chill"], a=a, c=c)
for a, c in ((0.5, 0.5), (1, 1)):
    add_random("arith", f"random pair x{a}", [one(p) for p in PROMPTS], scale(ALIEN, a) + scale(CHILL, c), ["alien", "chill"], a=a, c=c)

# 5. Dose ladder: trait/broken on the 8 prompts, and 20 easy facts (scored by answer match, not the judge).
for m in (0, 0.25, 0.5, 1, 1.5, 2, 2.5):  # x0.8 base dose -> 0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0
    dose = round(0.8 * m, 2)
    add("ladder", "neuron", [one(p) for p in PROMPTS], scale(ALIEN, m), ["alien"], dose=dose)
    if m:
        add_random("ladder", "random", [one(p) for p in PROMPTS], scale(ALIEN, m), ["alien"], dose=dose)
    for cond, steers in (("neuron", scale(ALIEN, m)), ("random", scale(control(ALIEN, "random neuron", ss.N_NEURONS, 0), m))):
        cs, outs = gen([one(q) for q, _ in FACTS], steers, False, 60)
        for (q, rx), o in zip(FACTS, outs):
            records.append({"exp": "facts", "cond": cond, "dose": dose, "steers": steers, "decoding": "greedy", "conv": one(q),
                            "reply": o, "questions": [], "correct": bool(re.search(rx, o, re.I))})
        print("facts", cond, dose, flush=True)

# 6. Confabulation: steered first turn, then "why did you bring up aliens?"
_, first_alien = gen([one(p) for p in PROMPTS], ALIEN, False)
_, first_normal = gen([one(p) for p in PROMPTS], [], False)
idx = [i for i, r in enumerate(first_alien) if re.search(r"alien|extraterrestr|planet|UFO|outer space", r, re.I)]
why = lambda firsts, ids: [one(PROMPTS[i]) + [{"role": "assistant", "content": firsts[i]}, {"role": "user", "content": WHY}] for i in ids]
add("confab", "steered 1st turn, unsteered answer", why(first_alien, idx), [], ["invent", "admit", "deny"])
add("confab", "steered both turns", why(first_alien, idx), ALIEN, ["invent", "admit", "deny"])
add("confab", "normal 1st turn (false premise)", why(first_normal, range(len(PROMPTS))), [], ["invent", "admit", "deny"])

with open("curious/records.jsonl", "w") as f:
    for i, r in enumerate(records):
        f.write(json.dumps({"rid": i, **r}, ensure_ascii=False) + "\n")
items = [(i, q) for i, r in enumerate(records) for q in r["questions"]]
random.Random(11).shuffle(items)
key = {}
with open("judge/blind.jsonl", "w") as f:
    for j, (i, q) in enumerate(items):
        r = records[i]
        ctx = "\n".join(f"[{m['role']}] {m['content']}" for m in r["conv"] if m["role"] != "system")
        gloss = r["exp"] == "language" and r["decoding"] == "greedy" and q == "alien"
        key[f"c{j:05d}"] = [i, q]
        f.write(json.dumps({"id": f"c{j:05d}", "question": Q[q], "conversation": ctx, "reply": r["reply"],
                            **({"gloss": True} if gloss else {})}, ensure_ascii=False) + "\n")
json.dump(key, open("curious/blind_key.json", "w"))
print(len(records), "records,", len(items), "blind items; alien first turns mentioning aliens:", len(idx), "of", len(PROMPTS))
