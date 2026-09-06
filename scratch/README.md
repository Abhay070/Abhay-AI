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

### A real run, for calibration

`--preset small` on a 4-core CPU, 3000 iterations, ~16 minutes:

```
iter     0/3000  train 4.1847  val 4.1811     <- knows nothing
iter  1000/3000  train 1.8307  val 1.9540
iter  2000/3000  train 1.5631  val 1.7383
iter  3000/3000  train 1.4820  val 1.6815
```

Train and validation loss fell together the whole way, which is what healthy
training looks like — no overfitting. Output at 1.68, from `KING RICHARD:`:

```
KING RICHARD:
Gainst then else hath power'd; thereof I shall be:
Were thou ask in crept my bloody,
For are that I say that my should friat,
Will a dear than the murder of should, nare,
No, lord father. We prince be from approves:

HORTENSIO:
A with meet of your voices I carrance at despitions.
```

Look at what 795,904 parameters learned from raw text, with nobody telling it
any of it: English spelling, apostrophes, iambic-ish line lengths, the
`SPEAKER:` convention of a play, blank lines between speeches, and the names of
characters who appear in the corpus. It also produced "friat", "carrance" and
"despitions", and the sentences mean nothing.

That is the whole lesson in one screen. Form is learnable from very little.
Meaning is what costs a data center.

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
