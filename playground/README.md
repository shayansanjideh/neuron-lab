# Neuron playground

Chat with Qwen3.5 4B and see the Normal Qwen and Neuron Qwen (neuron-steered) replies side by side. It's for silly questions and screenshots.

## Start it (Kaggle, interactive)

1. Open the `neuron-playground` notebook on Kaggle and click **Edit**.
2. Settings: **Accelerator: GPU T4**, **Internet: on**.
3. Optional: attach a notebook or dataset that has a `steers.json` to get `agent: ...` presets. The first neuron-agent run's steers are placeholders (it found nothing), so it isn't attached by default.
4. **Run All**. The model loads in about 5 minutes; the first run also downloads ~9 GB.
5. Open the `PUBLIC LINK: https://....gradio.live` printed under the second-to-last cell. It works on your phone and stays up while the session runs (Kaggle stops idle sessions after a while, and the link expires after 1 week at most).
6. When you're done, stop the session (**Power off**) so it stops using your 30 GPU hours a week.

## Using it

- **Question**: anything. Press Enter or **Ask**. Every question runs twice, normal first and then steered (greedy, so the same question always gives the same answer).
- **Preset**: sarcastic, sarcastic + anti-sycophancy, condolences on good things, chill friend, lyra mix, alien identity, responsible AI persona, plus `agent: <trait>` if attached.
- **Dose multiplier** (0 to 3): scales every steer. 0 = normal model; 2 to 3 usually turns into word salad, which can also be funny.
- **Custom neurons**: lyra's syntax, overrides the preset. `L15N3920+0.6 L12N2920-0.5` means layer 15 neuron 3920 at +0.6, layer 12 neuron 2920 at -0.5. A token with no dose means +1.
- **Control**: `random neuron` swaps each steered neuron for a random one in the same layer at the same push size; `sign flip` negates every dose. Post a control next to the steered reply to answer "it's just pushing until it breaks". Change the seed for a different random pick.
- **Steer mode**: `add` (default) adds dose x median residual norm / ||W_down[:, n]|| to the neuron's activation at every token; `set` clamps the activation to that value instead. Lyra's recipe text is ambiguous between the two, so try both.
- **Max tokens**: reply length. Shorter is faster and screenshots better.
- The line under the answers shows exactly which steers ran. Keep it in the screenshot so people can reproduce it.

## Files

- `neurons.py`: neuron-string parser and presets (tests: `python3 -m unittest test_neurons test_card`).
- `build_playground.py`: rebuilds `playground.ipynb` from `neurons.py` and `../src/neuron_server.py`. Run it after editing either.
- `kernel-metadata.example.json`: copy to `kernel-metadata.json`, put in your Kaggle username, then `kaggle kernels push -p playground`.
