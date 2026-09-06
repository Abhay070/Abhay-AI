# scratch — a language model from zero

Your own transformer. You write it, you train it, the weights are yours.

```bash
pip install -r ../requirements-scratch.txt
python prepare_data.py
python train.py
python sample.py --prompt "To be, or" --stream
```

## Files

| File | What it is |
|---|---|
| `model.py` | The transformer. Attention, MLP, residuals, generation. ~200 lines. |
| `tokenizer.py` | Character-level text ↔ integers |
| `prepare_data.py` | Fetches a corpus (falls back to a bundled sample offline) |
| `train.py` | The training loop |
| `sample.py` | Generate text from a trained checkpoint |

## Presets

```bash
python train.py --preset tiny    # ~100k params, ~2 min CPU — proves it works
python train.py --preset small   # ~800k params, ~15 min CPU — the default
python train.py --preset big     # ~10M params, needs a GPU (free Colab T4)
```

## Reading the loss

Random guessing over 65 characters gives a loss of `ln(65) ≈ 4.17`. That is your
baseline; anything below it is learned structure.

| Loss | What the output looks like |
|---|---|
| 4.17 | pure noise |
| 3.0 | letter frequencies about right, no words |
| 2.5 | word-shaped clumps, correct punctuation habits |
| 2.0 | many real words, dialogue formatting, no grammar |
| 1.6 | plausible sentences, correct format, no meaning |
| < 1.5 | check for overfitting — is validation loss still falling too? |

If train loss keeps dropping while validation loss turns upward, the model has
started memorizing the corpus rather than learning its patterns. Stop there, or
get more data.

## Train on your own text

```bash
python prepare_data.py --input my_journal.txt
python train.py
```

Anything works — notes, exported chat logs, a book you have the rights to. Aim
for at least ~1 MB. Below that it memorizes instead of generalizing.

## What it will and will not do

**Will:** learn spelling, punctuation, capitalization, the formatting conventions
of your text, and a convincing surface rhythm.

**Will not:** answer questions, follow instructions, or mean anything. It is a
pretrained continuation engine, not an assistant — becoming an assistant is a
separate training stage (instruction tuning + RLHF) on a model thousands of times
larger.

That gap is the lesson. It is also exactly why stage 3 exists.
