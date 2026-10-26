# Task: blind-label chatbot replies

`blind.jsonl` holds chatbot replies, one JSON object per line: `id`, `question`, `prompt`, `reply`. They come from several different setups mixed together, and you are not told which is which. Don't look for or open any other file in this directory.

For every line, answer two yes/no labels:

- `trait`: the answer to that line's `question` about the reply. 1 for yes, 0 for no. Say yes only if it's clearly true of the reply, not a faint hint.
- `broken`: 1 if the reply is broken (gibberish, heavy repetition or loops, wrong language for no reason, or it ignores the prompt entirely), otherwise 0.

Judge each line on its own, by reading it yourself. Do not label with keyword matching or any automatic heuristic. Go through the file in chunks, for example with a short Python script that prints 50 lines at a time, and append your labels as you go. Write `labels.jsonl` with one line per id: `{"id": "...", "trait": 0 or 1, "broken": 0 or 1}`. Every id must be labeled exactly once. At the end, check the count matches `blind.jsonl` and fill in anything missing.
