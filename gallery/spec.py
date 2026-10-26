"""What the gallery runs: prompts, presets, and how each preset's trait is judged and described."""
PROMPTS = [
    "My cat knocked my coffee off the table.",
    "I just got promoted!",
    "What's the meaning of life?",
    "Describe yourself in one sentence.",
    "Should I text my ex?",
    "Give me a pep talk for Monday.",
    "What's your favorite food?",
    "Explain gravity to a 5-year-old.",
]
# preset -> (judge question about the trait, phrase for titles: "makes Qwen <phrase>")
TRAITS = {
    "sarcastic": ("Is the reply sarcastic (dry, ironic or mocking)?", "sound sarcastic"),
    "sarcastic + anti-sycophancy": ("Is the reply sarcastic or bluntly unflattering (dry, ironic, mocking, or refusing to just agree)?",
                                    "sound sarcastic or blunt"),
    "chill friend": ("Does the reply sound casual, relaxed and warm, like a buddy talking?", "talk like a chill friend"),
    "alien identity": ("Does the reply talk as or about aliens or extraterrestrials?", "talk about aliens"),
    "responsible AI persona": ("Does the reply present itself as 'Responsible AI' or fixate on responsibility?",
                               "fixate on being Responsible AI"),
    "condolences on good things": ("Does the reply offer condolences or sympathy as if something bad happened?",
                                   "offer condolences"),
    "lyra mix": ("Is the reply sarcastic or snarky?", "sound snarky"),
}
MODES = ["add", "set"]
RANDOM_SEEDS = [0, 1, 2]
SAMPLE_SEED, N_SAMPLES, TEMP, MAX_TOKENS = 1234, 3, 0.7, 150
MODEL = "Qwen3.5-4B"
