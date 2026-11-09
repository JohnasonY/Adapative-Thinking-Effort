# CSC 733 Final Project

## Adaptive Thinking Effort Based on Prompt Complexity

A Python command-line starter based on the [shared project structure](https://chatgpt.com/s/t_6af237afce708191bb14ddf330869ea2).
DistilBERT predicts five difficulty criteria; a separate weighted score selects low,
medium, or high thinking effort. This initial version selects effort only; connecting
a reasoning LLM and measuring answer quality, tokens, and latency are future work.

### Setup

Use Python 3.12 and run commands from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, use `uv venv --python 3.12` and `uv pip install -r requirements.txt`.
First-time training downloads the pretrained encoder and tokenizer from Hugging Face.
Prediction uses the saved checkpoint and local tokenizer. No API key is needed.

### Quick start

Try scoring without ML dependencies or a trained checkpoint:

```bash
python -m src.main "Compare TCP and UDP" --scores 3 3 3 1 1
```

This returns difficulty `11` and effort `medium`, explicitly marked as manual scores.
Train, predict, and evaluate with:

```bash
python -m src.train --epochs 3
python -m src.main "Explain how binary search works"
python -m src.evaluate
```

Training holds out 20% of training rows for validation and saves the checkpoint with
the lowest validation MSE. `data/test.csv` is reserved for final evaluation.
The seed defaults to 42; exact reproducibility can vary across hardware.
Prediction requires training first; no fabricated model weights are included.

### Structure

```text
data/
  train.csv              # Illustrative training data
  test.csv               # Illustrative held-out data
models/                  # Generated checkpoint and tokenizer (ignored by Git)
src/
  __init__.py
  dataset.py             # Validate CSV labels and tokenize batches
  model.py               # DistilBERT and five-output regression head
  train.py               # Train and save best validation checkpoint
  predict.py             # Load checkpoint and predict scores
  scoring.py             # Weighted sum of criterion scores
  effort_selector.py     # Threshold-based effort selection
  evaluate.py            # Overall/per-criterion MAE and MSE
  main.py                # CLI pipeline
requirements.txt
README.md
```

### Labels and scoring

CSV columns must be `prompt,reasoning,knowledge,steps,ambiguity,math`.
Each criterion is a finite number from 1 (easiest) to 5 (hardest).
The included tiny datasets have **illustrative, unvalidated labels** and are only
for exercising the pipeline. Replace them with a larger consistently annotated
dataset and ensure train/test prompts do not overlap before research evaluation.

Equal weights produce a sum in [5, 25]. Scores up to 10 select low, scores above
10 through 17 select medium, and scores above 17 select high. These are starting
thresholds, not validated research findings. The regression head bounds each output
between 1 and 5. Prompts are truncated at 256 tokens by default (training supports
`--max-length` up to 512).

`--weights` accepts five nonnegative weights in CSV criterion order. This is a raw
weighted sum: changing weights changes the score scale, so adjust `--low-threshold`
and `--medium-threshold` accordingly. Run any command with `--help` for options.

Implementation references: [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
and [PyTorch checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
