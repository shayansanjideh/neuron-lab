# Task: blind-score chatbot replies

`blind.jsonl` holds chatbot replies, one JSON object per line: `id`, `trait`, `prompt`, `reply`. They come from several different setups mixed together, and you are not told which is which. Don't look for or open any other file in this directory.

For every line, score:

- `trait`, from 1 to 10: how clearly the reply shows the named trait. 1 means not at all and 10 means unmistakable throughout. "sarcastic" means dry, ironic, mocking wit. "chill friend" means casual, relaxed, and warm like a buddy.
- `coherence`, from 1 to 10: whether the reply is fluent, on-topic, and actually helpful. Repetition, gibberish, or ignoring the question score low.

Judge each reply on its own and use the full range. Go through the file in chunks, for example with a short Python script that prints 20 lines at a time. Write `scores.jsonl` with one line per id: `{"id": "...", "trait": n, "coherence": n}`. Every id must be scored exactly once.
