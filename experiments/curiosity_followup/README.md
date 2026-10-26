# Follow-up: alien neuron vs system prompts (Qwen3.5-4B)

40 fresh everyday prompts, fixed before the run, under six system prompts (none, plain factual, "never mention aliens", "never mention cookies", a neutral mention of aliens, and an alien roleplay prompt). Each gets: no steering, the alien neuron L22 N2742 at +0.8 and -0.8, and 4 random same-layer neurons at the same push. Greedy, up to 120 tokens. 1,680 replies.

- `protocol.json` fixes the hypotheses and pass rules in advance; `score.py` computes the objective counts (word lists, repetition, easy-fact matches).
- `run.py` is the Kaggle runner (uses `src/neuron_server.py`).
- `visuals.py` and `viral.py` draw the charts from `results/curiosity-followup-2026-10-26/`.

Results: [objective_results.md](../../results/curiosity-followup-2026-10-26/output/objective_results.md). Both pre-registered hypotheses (a "spectral" persona, and the prohibition reducing repetition) failed their fixed tests. The descriptive result that held up: in the alien roleplay, turning the neuron down cuts alien/space words from 25/40 to 1/40 while the name Xeno-888 survives in 34/40; 4 random neurons at the same push keep 23-30/40. Counts are word matches; a blind semantic judge pass is still to come.
