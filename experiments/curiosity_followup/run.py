"""Prospective curiosity follow-up. Saves every batch; does not require Claude or a judge token."""
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())
PROMPTS = json.loads((HERE / 'prompts.json').read_text())
OUT = HERE / 'output'
OUT.mkdir(exist_ok=True)
assert len(PROMPTS) == P['n_prompts']
os.environ['DTYPE'] = 'auto'
import torch
import neuron_server as ss

target = P['target']
rng = random.Random(P['random_seed'])
random_ids = rng.sample([n for n in range(ss.N_NEURONS) if n != target['neuron']], P['n_random'])
runtime = dict(protocol_sha256=hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest(),
               prompts_sha256=hashlib.sha256((HERE / 'prompts.json').read_bytes()).hexdigest(),
               model_revision=os.environ['MODEL_REVISION'], random_neurons=random_ids,
               dtype=str(ss.DTYPE), calibration=ss.MEDIAN,
               cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               versions={n: importlib.metadata.version(n) for n in ['torch','transformers','accelerate']})
runtime_path = OUT / 'runtime.json'
if runtime_path.exists():
    previous = json.loads(runtime_path.read_text())
    assert previous == json.loads(json.dumps(runtime)), 'Runtime changed; do not mix results across environments.'
runtime_path.write_text(json.dumps(runtime, indent=2))
steer = lambda neuron, sign: [{'layer': target['layer'], 'neuron': neuron, 'dose': sign * P['absolute_dose'], 'mode': 'add'}]
done = {}
path = OUT / 'records.jsonl'
if path.exists():
    for line in path.read_text().splitlines():
        row = json.loads(line)
        done[(row['context'], row['setting'], row['prompt_id'])] = row

def generate(prompts, context, steers):
    texts = [ss.chat(p['text'], context) for p in prompts]
    ss.STEERS = ss.resolve(steers)
    try:
        with torch.inference_mode():
            enc = ss.encode(texts)
            gen = ss.model.generate(**enc, do_sample=False, max_new_tokens=P['max_new_tokens'],
                                    pad_token_id=ss.tok.pad_token_id)
            output_ids = gen[:, enc['input_ids'].shape[1]:].tolist()
            return ss.tok.batch_decode(output_ids, skip_special_tokens=True), output_ids
    finally:
        ss.STEERS = []

with path.open('a', buffering=1) as f:
    for context, system in P['contexts'].items():
        direction = -1 if context == 'alien_roleplay' else 1
        settings = [('baseline', []), ('target_positive', steer(target['neuron'], 1)),
                    ('target_negative', steer(target['neuron'], -1))]
        settings += [(f'random_{i}', steer(n, direction)) for i, n in enumerate(random_ids)]
        for setting, steers in settings:
            pending = [p for p in PROMPTS if (context, setting, p['id']) not in done]
            for start in range(0, len(pending), P['batch_size']):
                ps = pending[start:start + P['batch_size']]
                t = time.time()
                replies, token_ids = generate(ps, system, steers)
                for p, reply, ids in zip(ps, replies, token_ids):
                    row = dict(rid=f'{context}/{setting}/{p["id"]}', context=context, system_prompt=system,
                               setting=setting, prompt_id=p['id'], prompt=p['text'], steers=steers,
                               decoding='greedy', reply=reply.strip(), output_token_ids=ids,
                               answer_regex=p.get('answer_regex'))
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    done[context, setting, p['id']] = row
                f.flush(); os.fsync(f.fileno())
                print(context, setting, start + len(ps), '/', len(pending), 'seconds', round(time.time()-t, 1), flush=True)
            (OUT / 'progress.json').write_text(json.dumps(dict(context=context, setting=setting, records=len(done))))

expected = len(P['contexts']) * 7 * len(PROMPTS)
assert len(done) == expected, (len(done), expected)
# Semantic labels are a separate step. This pack exposes no setting, neuron, or system prompt.
order = list(done)
random.Random(46321).shuffle(order)
key = {}
with (OUT / 'blind.jsonl').open('w') as f:
    for i, lookup in enumerate(order):
        row = done[lookup]; bid = f'q{i:05d}'; key[bid] = row['rid']
        f.write(json.dumps(dict(id=bid, prompt=row['prompt'], reply=row['reply']), ensure_ascii=False) + '\n')
(OUT / 'blind_key.json').write_text(json.dumps(key))
(OUT / 'complete.json').write_text(json.dumps(dict(records=len(done), expected=expected)))
print('COMPLETE', len(done), flush=True)
