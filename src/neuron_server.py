"""Neuron-steering server for Qwen3.5 4B on a Kaggle GPU.

Runs on 127.0.0.1:8000. Steer recipe (lyra): a neuron n in layer L gets
dose * median(||resid leaving L||, tokens 1:) / ||W_down[:, n]|| added to its
down_proj input at every position, so dose=1 writes one median residual norm.
"""
import os, re, json, random, threading
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(os.path.join(HERE, "prompts.json")))
LOCK = threading.Lock()
STEERS = []  # active steers, read by the hooks: [(layer, neuron, value, mode)]


def load_model(dtype):
    kw = dict(dtype=dtype, device_map="auto")
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    except Exception as e:
        print("CausalLM load failed, trying ImageTextToText:", e)
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)


tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_dt = os.environ.get("DTYPE", "fp16")
if _dt == "auto":  # fp32 needs both T4s (~18 GB); a single GPU gets fp16
    _dt = "fp32" if torch.cuda.device_count() >= 2 else "fp16"
DTYPE = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[_dt]
model = load_model(DTYPE).eval()

# Text decoder layers only (skip any vision tower).
DOWN, LAYER = {}, {}
for name, mod in model.named_modules():
    if "visual" in name or "vision" in name:
        continue
    m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
    if m:
        DOWN[int(m.group(1))] = mod
    m = re.search(r"layers\.(\d+)$", name)
    if m:
        LAYER[int(m.group(1))] = mod
N_LAYERS = len(DOWN)
N_NEURONS = DOWN[0].in_features
W_NORM = {L: DOWN[L].weight.float().norm(dim=0).cpu() for L in DOWN}  # ||W_down[:, n]||
print(f"{MODEL_ID}: {N_LAYERS} layers x {N_NEURONS} neurons, dtype {DTYPE}")


def chat(prompt, system_prompt=None):
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def encode(texts):
    return tok(texts, return_tensors="pt", padding=True).to(model.device)


# ---- steering hooks ---------------------------------------------------------
def make_hook(L):
    def hook(mod, args):
        active = [s for s in STEERS if s[0] == L]
        if not active:
            return None
        x = args[0].clone()
        for _, n, v, mode in active:
            if mode == "set":
                x[..., n] = v
            else:
                x[..., n] += v
        return (x,) + tuple(args[1:])
    return hook


for L in DOWN:
    DOWN[L].register_forward_pre_hook(make_hook(L))


# ---- calibration: median residual norm leaving each layer ---------------------
@torch.no_grad()
def calibrate():
    norms = {L: [] for L in LAYER}
    def grab(L):
        def h(mod, args, out):
            hs = out[0] if isinstance(out, tuple) else out
            norms[L].append(hs[0, 1:].float().norm(dim=-1).cpu())
        return h
    hs = [LAYER[L].register_forward_hook(grab(L)) for L in LAYER]
    try:
        for p in PROMPTS["calibration"]:
            model(**encode([chat(p)]))
    finally:
        for h in hs:
            h.remove()
    return {L: torch.cat(v).median().item() for L, v in norms.items()}


MEDIAN = calibrate()
print("median resid norms:", {L: round(v, 1) for L, v in MEDIAN.items()})


def resolve(steers):
    out = []
    for s in steers or []:
        L, n, d = int(s["layer"]), int(s["neuron"]), float(s["dose"])
        out.append((L, n, d * MEDIAN[L] / W_NORM[L][n].item(), s.get("mode", "add")))
    return out


# ---- core ops -----------------------------------------------------------------
@torch.no_grad()
def generate(prompts, system_prompt=None, max_new_tokens=200, temperature=0.0, batch=8):
    outs = []
    for i in range(0, len(prompts), batch):
        enc = encode([chat(p, system_prompt) for p in prompts[i:i + batch]])
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


_mmlu, _gsm = None, None


def load_evals():
    global _mmlu, _gsm
    if _mmlu is None:
        from datasets import load_dataset
        m = list(load_dataset("cais/mmlu", "all", split="test"))
        random.Random(0).shuffle(m)
        _mmlu = m[:200]
        g = list(load_dataset("openai/gsm8k", "main", split="test"))
        random.Random(0).shuffle(g)
        _gsm = g[:100]
    return _mmlu, _gsm


LETTERS = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]


@torch.no_grad()
def mmlu_acc(items, system_prompt=None, batch=8):
    right = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        texts = [chat(q["question"] + "\n" + "\n".join(f"{c}. {a}" for c, a in zip("ABCD", q["choices"]))
                      + "\nAnswer with the letter only.", system_prompt) for q in chunk]
        logits = model(**encode(texts), logits_to_keep=1).logits[:, -1, LETTERS]  # last token only; full logits OOM
        right += sum(int(p == q["answer"]) for p, q in zip(logits.argmax(-1).tolist(), chunk))
    return right / len(items)


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm_acc(items, system_prompt=None):
    outs = generate([q["question"] + "\nThink step by step, then give the final number on the last line." for q in items],
                    system_prompt, max_new_tokens=320, batch=8)
    gold = [q["answer"].split("####")[-1].strip().replace(",", "") for q in items]
    return sum(last_number(o) == g for o, g in zip(outs, gold)) / len(items)


# ---- API ------------------------------------------------------------------------
app = FastAPI()


class Steer(BaseModel):
    layer: int
    neuron: int
    dose: float
    mode: str = "add"


class GenReq(BaseModel):
    prompts: list[str]
    steers: list[Steer] = []
    system_prompt: str | None = None
    max_new_tokens: int = 200
    temperature: float = 0.0


class CapReq(BaseModel):
    steers: list[Steer] = []
    system_prompt: str | None = None
    full: bool = False


class ContrastReq(BaseModel):
    positive: list[str]
    negative: list[str]
    top_k: int = 40
    layers: list[int] | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/info")
def info():
    return {"model": MODEL_ID, "n_layers": N_LAYERS, "n_neurons": N_NEURONS, "dtype": str(DTYPE), "gpus": torch.cuda.device_count(),
            "median_resid_norm": MEDIAN}


@app.post("/generate")
def api_generate(r: GenReq):
    global STEERS
    with LOCK:
        STEERS = resolve([s.model_dump() for s in r.steers])
        try:
            return {"outputs": generate(r.prompts, r.system_prompt, r.max_new_tokens, r.temperature)}
        finally:
            STEERS = []


@app.post("/capability")
def api_capability(r: CapReq):
    """fast: 100 MMLU (logits only, ~1 min). full: 200 MMLU + 100 GSM8K (several minutes)."""
    global STEERS
    mmlu, gsm = load_evals()
    with LOCK:
        STEERS = resolve([s.model_dump() for s in r.steers])
        try:
            res = {"mmlu": mmlu_acc(mmlu if r.full else mmlu[:100], r.system_prompt)}
            if r.full:
                res["gsm8k"] = gsm_acc(gsm, r.system_prompt)
            return res
        finally:
            STEERS = []


@app.post("/contrast")
def api_contrast(r: ContrastReq):
    """Neurons whose mean down_proj input differs most between positive and negative texts
    (texts are fed raw, not chat-formatted). Score = mean difference / pooled std."""
    layers = r.layers or list(DOWN)
    acts = {L: {"pos": [], "neg": []} for L in layers}
    side = {"v": "pos"}
    def grab(L):
        def h(mod, args):
            acts[L][side["v"]].append(args[0][0, 1:].float().mean(0).cpu())
        return h
    with LOCK, torch.no_grad():
        hs = [DOWN[L].register_forward_pre_hook(grab(L)) for L in layers]
        try:
            for side["v"], texts in (("pos", r.positive), ("neg", r.negative)):
                for t in texts:
                    model(**encode([t]))
        finally:
            for h in hs:
                h.remove()
    found = []
    for L in layers:
        p, n = torch.stack(acts[L]["pos"]), torch.stack(acts[L]["neg"])
        diff = p.mean(0) - n.mean(0)
        score = diff / (torch.cat([p, n]).std(0) + 1e-6)
        for idx in score.abs().topk(r.top_k).indices.tolist():
            found.append({"layer": L, "neuron": idx, "score": round(score[idx].item(), 3),
                          "pos_mean": round(p.mean(0)[idx].item(), 4), "neg_mean": round(n.mean(0)[idx].item(), 4)})
    found.sort(key=lambda f: -abs(f["score"]))
    return {"neurons": found[:r.top_k]}
