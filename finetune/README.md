# finetune — make the weights yours

Take an open model and teach it your voice with LoRA. Needs a GPU; a **free
Google Colab T4** is the intended target.

```bash
pip install -r ../requirements-finetune.txt
python train_lora.py --data data/example.jsonl
python chat.py
```

## Why this is free

Full fine-tuning of a 7B model updates 7 billion numbers and needs ~60 GB of VRAM
for optimizer state alone. LoRA freezes the original weights and trains small
matrices beside them — under 1% of the parameters. Load the base model in 4-bit
and it fits in 16 GB.

You end up with a **~40 MB adapter**, not a new multi-gigabyte model. The base
model stays untouched, and you can train several adapters for different purposes
and swap between them.

## Data format

One JSON object per line in a `.jsonl` file:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`data/example.jsonl` has ten rows showing the shape. Replace it with yours.

**A few hundred careful examples beat ten thousand scraped ones.** This is the
most consistently underestimated fact in fine-tuning. The model is learning your
patterns — including your mistakes and inconsistencies, faithfully.

## Base models

| Model | Size | Fits a free T4? |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | comfortably — **the default** |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | yes, in 4-bit |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | yes, in 4-bit |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | yes, but gated — needs a HF account |

## Did it work?

```bash
python chat.py --base-only    # the untuned model
python chat.py                # yours
```

Ask both the same questions. If you cannot tell them apart, the fine-tune did
nothing — usually too little data, too few epochs, or a learning rate too low.

## When training goes wrong

| Symptom | Usual cause |
|---|---|
| Loss does not fall | Wrong chat template, or malformed data. Check these before hyperparameters. |
| Loss hits ~0 immediately | Too few examples — it memorized them. Add data. |
| Output is repetitive | Overtrained. Fewer epochs, or a lower learning rate. |
| Out of memory | Lower `--batch-size` to 1, raise `--grad-accum`, shorten `--max-len`. |
| Model forgot how to talk | Learning rate too high, or too many epochs. LoRA at 2e-4 for 3 epochs is a safe start. |

## Fine-tune or not?

Fine-tuning teaches **style and behaviour** well, and **facts** badly. If you
want the model to know things it does not, use RAG — retrieve documents at query
time instead. If you want it to *sound* a certain way or reliably produce a
certain format, fine-tune. See `../docs/ROADMAP.md`.
