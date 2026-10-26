# Task: find personality neurons in Qwen3.5 4B

You are an interpretability researcher. A steering server for Qwen/Qwen3.5-4B (32 layers, 9,216 MLP neurons per layer) is running at http://127.0.0.1:8000. Use `curl` or short Python scripts against it. Do not load the model yourself, and do not edit the files in this directory other than the outputs listed below.

Goal: for each trait, **"sarcastic"** and **"chill friend"**, find a small set of neuron steers (1 to 6 neurons) that makes the model clearly show the trait on everyday prompts, while MMLU drops by at most 2 points versus the unsteered model and replies stay coherent.

## Endpoints

- `GET /info` returns layer count, neuron count, and median residual norms.
- `POST /contrast {"positive": [texts], "negative": [texts], "top_k": 40, "layers": optional}` returns the neurons whose activations best separate trait-heavy text from neutral text (raw text, not chat-formatted). This is your main discovery tool. Write your own contrast texts: 15 to 30 each, matched in topic and length, differing only in the trait.
- `POST /generate {"prompts": [...], "steers": [{"layer": L, "neuron": n, "dose": d, "mode": "add"}], "system_prompt": optional, "max_new_tokens": 200}` returns steered replies (greedy). The neuron gets value = dose x median residual norm / ||W_down[:, n]|| at every token. `"mode": "add"` (default) adds that value to the neuron's activation, so dose 1.0 writes one median residual norm along its output direction; `"mode": "set"` clamps the activation to that value instead. Useful doses are usually between -1.5 and 1.5. Negative doses suppress. Text usually breaks down past about 1.5x the residual norm.
- `POST /capability {"steers": [...], "full": false}` returns MMLU accuracy on 100 questions (about a minute). `"full": true` adds 200 MMLU plus 100 GSM8K and takes several minutes.

## Rules

1. **Harness check first.** Lyra's public recipe says L15 N3920 at dose +1 makes this model sarcastic, but last run it only looked upbeat, so the steering setup may not match hers. Check `GET /info` for the dtype. On 8 of your own everyday prompts, generate with no steer, then with `[{"layer":15,"neuron":3920,"dose":D,"mode":M}]` for M in add and set and D in 0.5, 1.0 and 1.5. Score each setting for sarcasm and coherence. Pick the mode that reproduces sarcasm best and use it for every steer after this. If neither reproduces it, say so plainly, use add, and continue with the hunt anyway. Write `harness.json`: `{"mode": "add" or "set", "dtype": "...", "reproduced": true or false, "notes": "one line"}`, and record the outputs and scores in log.md. Don't reuse that neuron as your answer unless your own search independently finds it.
2. Never use the `heldout` prompts in prompts.json for discovery or tuning. Write your own development prompts (about 10 per trait).
3. Budget per trait: at most 25 calls to `/capability` with `full:false` and at most 3 with `full:true`. Get the unsteered baseline once with `full:true` and reuse it.
4. Score your own dev outputs on trait strength (1 to 10) and coherence (1 to 10). A steer that produces gibberish or repetition fails, however strong the trait.
5. Work like a scientist: form a hypothesis about what a neuron does, test it, and keep notes on what didn't work.
6. Don't give up early. Last run stopped after 26 turns. Use at least 150 turns (tool calls) of real search before deciding a trait can't be done, and try several discovery strategies (different contrast texts, per-layer searches, pairs of neurons, both dose signs, both modes).
7. Every steer you put in steers.json must have been tested at exactly that layer, neuron, dose and mode, on your dev prompts and with `/capability`. No untested placeholders. If nothing works for a trait, write your best tested attempt and say so.
8. The final eval also runs controls on your steers: random neurons at the same layers and doses, and your steers with the sign flipped. A real finding should beat both.

## Outputs (write these in the current directory)

- `harness.json`: see rule 1.
- `steers.json`: `{"sarcastic": [{"layer":..,"neuron":..,"dose":..,"mode":..}, ...], "chill friend": [...]}`, with your single best tested steer set per trait.
- `log.md`: every steer tried, with your dev trait/coherence scores and its MMLU, plus a short summary of what you learned about which layers and neurons matter.

Stop when both traits have a final steer that meets the goal, or when you've used the budget. In that case, write your best attempt and say plainly in log.md that it fell short.
