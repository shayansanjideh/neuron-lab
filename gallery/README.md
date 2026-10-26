# Neuron gallery

Batch version of the playground: every preset (add and set mode) on 8 fun prompts, with random-neuron (3 seeds) and sign-flip controls, every reply blind-labeled by a fresh Claude Code session (trait yes/no, broken yes/no), then rendered as 1600x900 cards and a summary chart.

- Kaggle: copy `kernel-metadata.example.json` to `kernel-metadata.json` with your Kaggle username, then `kaggle kernels push -p gallery` (notebook `neuron-gallery`, built by `build_gallery.py`).
- Outputs in `out/`: `card_<preset>_<mode>.png`, `summary.png`, `results.md`, `stats.json`, raw `replies.jsonl`, the judge's `labels.jsonl` and `blind_key.json` (blind id -> reply id, preset).
- Re-render locally (no GPU): `python render.py <dir with replies.jsonl, labels.jsonl, blind_key.json> <out dir>` (needs pillow, matplotlib, fonttools). For emoji and Japanese text in replies, put `NotoEmoji.ttf` and `NotoSansJP.ttf` from Google Fonts in `gallery/fonts/` (not committed); without them those characters render as boxes.
- Example replies on cards follow a fixed rule, never hand-picked: the first prompt whose greedy neuron reply the judge labeled a coherent hit (prompt 1 if none). All three boxes show the same prompt with the same 230-character budget. Titles count coherent hits only (trait and not broken).
