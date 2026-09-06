# Roadmap

What to learn, in what order, and what to skip.

---

## Order of operations

**1. Run stage 1 and read `scratch/model.py`.**
Not skim — read. Every line. It is 200 lines and it contains the entire idea
behind every model you have used. If one part is opaque, put a `print(x.shape)`
next to it and run the training again. Shapes explain transformers better than
any diagram.

**2. Break things on purpose.** This is where understanding actually comes from:

- Remove the positional embedding (`wpe`) — why does quality collapse? (Attention
  has no inherent sense of order. Without position, "dog bites man" and "man
  bites dog" are literally the same input.)
- Remove the causal mask — why does loss drop to near zero while generation
  becomes garbage? (The model can see the answer. It learns to copy, not predict.)
- Set `n_layer=1` — what does it stop being able to do?
- Set temperature to 0.1, then 2.0 — watch the failure modes at each end.

**3. Scale one thing at a time.** Double `n_layer`. Then `n_embd`. Then the
dataset. Note the loss each time. You are re-deriving the scaling laws by hand,
and you will develop real intuition for where compute actually goes.

**4. Swap in a real tokenizer.** `pip install tiktoken`, set `vocab_size=50257`.
Feel the tradeoff: fewer tokens per sentence, much larger embedding table.

**5. Then fine-tune.** By this point stage 3 will read as obvious rather than
magical, because you already know what the weights are doing.

---

## Concepts, roughly in dependency order

**Foundations**
- Backpropagation — Karpathy's *micrograd* (~150 lines) is the clearest treatment
  that exists. Do not skip it because it looks small.
- Attention — "Attention Is All You Need" (2017). Short paper. Read it after
  stage 1, when you already have working code to map it onto.
- Tokenization — where a surprising number of real bugs live.

**Making models useful**
- **Pretraining vs. instruction tuning.** A pretrained model just continues text.
  The helpful assistant behaviour is a separate training stage on top. Your
  stage-1 model is pretrained-only, which is exactly why it will not answer
  questions.
- **RLHF / DPO** — how models learn to be helpful and harmless rather than merely
  fluent. This is most of what separates a raw model from Claude.
- **RAG** — giving a model your documents at query time instead of training them
  in. Cheaper and more current than fine-tuning for factual knowledge.
- **Tool use** — letting the model call functions. This is what turns a chat box
  into an agent, and it is mostly plumbing, not ML.

**Practical**
- Quantization (4-bit/8-bit) — how big models fit on small hardware
- KV caching — why generation speeds up after the first token
- Evaluation — how you know a change helped, instead of guessing

---

## Fine-tuning vs. RAG vs. prompting

People reach for fine-tuning when they want one of the other two. The rule:

| Goal | Use |
|---|---|
| Change the model's **voice, format, or behaviour** | fine-tuning |
| Give it **facts** it does not know | RAG |
| Adjust **tone or task** for one project | the system prompt |

Fine-tuning teaches *style and behaviour* well and *facts* badly. If your model
needs to know your company's Q3 numbers, do not fine-tune — retrieve.

---

## What to skip

- **Building a vector database from scratch.** Use one. It is not where the
  learning is.
- **Chasing every new model release.** The architecture has barely changed since
  2017. Understand one deeply; the rest is deltas.
- **Prompt-engineering courses.** Read the model provider's own docs, then
  experiment. The field moves faster than any course.
- **Training a large model from scratch.** Even with money, it is the wrong
  project. Fine-tuning gets you 95% of the value for 0.001% of the cost.

---

## Honest expectations

| Milestone | Realistic time |
|---|---|
| Stage 1 working, loss falling | one afternoon |
| Actually understanding `model.py` | a week of poking at it |
| A fine-tuned model that is useful to you | a weekend, once you have data |
| A polished AI product other people use | a few months |
| Frontier-competitive model | not on this path, and not solo |

The ceiling on this path is not "as good as Claude." It is: you understand how
these systems work, you can build real products with them, and you can make a
model that is genuinely yours. That is a substantial ceiling, and most people
who work on frontier models started exactly here.
