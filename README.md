# Abhay-AI

Building my own AI — three stages, all of it free.

This repo answers the question "can I build my own Claude/ChatGPT/Gemini?" by
actually doing the parts that are possible and being straight about the part
that is not.

---

## The honest version first

There are three different things people mean by "make my own AI," and they cost
wildly different amounts:

| What you mean | What it costs | Can you? |
|---|---|---|
| An AI **product** with its own personality, memory and tools | ~free | Yes, this weekend |
| Your **own model weights**, trained by you, running on your machine | ~free | Yes |
| A model **as capable as Claude or GPT-4** | $100M–$1B+, thousands of GPUs, ~100 people, months | No — and neither can anyone else solo |

That last row is not pessimism, it is arithmetic. Frontier training runs burn
more electricity than a small town and require hardware you cannot rent casually.
Anyone claiming otherwise is selling a course.

The first two rows are completely within reach, and they are where all the actual
learning is. This repo does both.

---

## What's in here

```
scratch/     Build and train a transformer from zero. CPU is enough.
finetune/    LoRA-tune an open model so the weights are genuinely yours.
app/         A real chat app. Streaming UI, memory, personality, pluggable brain.
docs/        The roadmap: what to learn, in what order, and what to skip.
```

Each stage stands alone. Do them in any order. Suggested order is `scratch` →
`app` → `finetune`, because understanding the model makes everything else stop
feeling like magic.

---

## Stage 1 — Train your own model from scratch

You write the transformer, you train it, the weights are yours. Runs on a laptop
CPU in minutes.

```bash
pip install -r requirements-scratch.txt
python scratch/prepare_data.py          # gets a corpus (~1 MB of Shakespeare)
python scratch/train.py                 # ~10–20 min on CPU
python scratch/sample.py --prompt "To be, or" --stream
```

Watch the loss. It starts at about **4.17** — that is exactly the loss of a model
guessing uniformly among 65 characters, i.e. knowing nothing. Every point it
drops is real structure the model found in the text on its own. Nobody told it
what a word is, or that speakers in a play are followed by a colon.

**What you will get:** convincing-looking Shakespearean gibberish. It learns
spelling, punctuation, dialogue formatting, and the rhythm of the text — with no
grasp of meaning. That gap between *form* and *meaning*, and how it closes as
models scale, is the single most instructive thing in machine learning.

Train on your own text instead:

```bash
python scratch/prepare_data.py --input my_notes.txt
python scratch/train.py
```

The code is four short files and every non-obvious line is commented:

- `scratch/model.py` — the transformer. Attention, MLP, residuals, the lot.
- `scratch/tokenizer.py` — text ↔ integers
- `scratch/train.py` — the training loop
- `scratch/sample.py` — generation

**This is the same architecture as GPT-4, Claude and Gemini.** Not similar —
the same. They are this, with more layers, more data, and a data center.

---

## Stage 2 — A real chat app

The product layer: streaming responses, conversation memory, your own system
prompt, and a brain you can swap.

```bash
pip install -r requirements-app.txt
cp app/.env.example .env
python app/server.py                    # http://localhost:8000
```

Pick a backend by setting `PROVIDER` in `.env`. All four are free:

| Provider | What it is | Cost | Needs |
|---|---|---|---|
| `ollama` | Open models on your own machine — **best default** | free forever | [ollama.com](https://ollama.com), then `ollama pull llama3.2` |
| `groq` | Hosted open models, very fast | free tier | free key, no card |
| `gemini` | Google's models | free tier | free key |
| `scratch` | Your own model from stage 1 | free | stage 1 |

Two things in `app/server.py` are worth reading closely, because they are the
whole trick behind every chat assistant you have used:

1. **The model has no memory.** None. The illusion of conversation comes from
   resending the entire history on every single request. That is the mechanism.
2. **The personality is a string.** `SYSTEM_PROMPT` in your `.env`. Change it and
   you change who your AI is, more than any other single edit you can make.

Point `PROVIDER=scratch` at your stage-1 model for a genuinely useful lesson: the
app is identical, the model is 800k parameters instead of 70 billion, and the
output is nonsense. It shows you exactly how much of "an AI assistant" is
scaffolding and how much is the model.

---

## Stage 3 — Fine-tune an open model

Take Llama/Qwen/Mistral and teach it your voice. This is where you get weights
that are yours, run offline, and actually work.

```bash
pip install -r requirements-finetune.txt
python finetune/train_lora.py --data finetune/data/example.jsonl
python finetune/chat.py
```

Needs a GPU. A **free Google Colab T4** is enough — that is the intended target.

Why it is free: full fine-tuning of a 7B model needs ~60 GB of VRAM. LoRA freezes
the original weights and trains small matrices beside them — under 1% of the
parameters — and 4-bit quantization shrinks the rest. It fits in 16 GB. You get a
~40 MB adapter file, not a new multi-gigabyte model.

Your data goes in `finetune/data/*.jsonl`, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

A few hundred examples you wrote carefully beat ten thousand scraped ones. This
is the most consistently underestimated fact in fine-tuning.

---

## What to do first

If you only do one thing: **stage 1.** Run the training, watch the loss fall,
read `scratch/model.py` top to bottom. It is 200 lines and it is the whole idea.
Everything else in AI is a variation on it.

Then read [`docs/ROADMAP.md`](docs/ROADMAP.md) — what to learn next, in what
order, and which popular advice to ignore.

---

## Cost summary

| | Cost |
|---|---|
| Stage 1 — train from scratch | $0 (your CPU) |
| Stage 2 — chat app | $0 (Ollama local, or a free API tier) |
| Stage 3 — LoRA fine-tune | $0 (free Colab T4) |
| Matching Claude | $100,000,000+ |

Three out of four is a good ratio.
