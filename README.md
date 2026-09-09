<div align="center">
  <img src="web/static/logo.svg" width="88" alt="Praxis">
  <h1>Praxis</h1>
  <p><strong>Think. Challenge. Build. Verify. Execute.</strong></p>
  <p>A personal AI that argues with your ideas instead of flattering them,<br>
  remembers what matters, uses real tools, and runs free on your own machine.</p>
</div>

---

## Get it running

```bash
git clone https://github.com/Abhay070/Abhay-AI.git
cd Abhay-AI
pip install -r requirements.txt
python server.py            # → http://localhost:8000
```

That works immediately on the demo backend — no key, no account, no model. Then
give it a real brain, free and offline:

```bash
curl -fsSL https://ollama.com/install.sh | sh   # Windows: installer at ollama.com
ollama pull llama3.2
echo "PROVIDER=ollama" > .env
python server.py
```

That's the whole setup. `llama3.2` is ~2 GB and runs on 8 GB of RAM.

---

## What it is

Most assistants are tuned to be agreeable — ask whether your idea is good and
you'll hear what's exciting about it. Praxis is built on a different
instruction: **respect the person, interrogate the idea.**

| | |
|---|---|
| **10 modes** | Founder interrogates. Teacher explains. Direct deletes every word that isn't the answer. One click, no cost. |
| **9 tools** | Exact arithmetic, web search, page reading, code execution, file analysis, memory. It computes instead of predicting. |
| **7 backends** | Ollama, Groq, Gemini, any OpenAI-compatible endpoint, Claude, a model you trained yourself — or the zero-setup demo. |
| **Real memory** | Durable facts across every conversation, each one visible, editable and deletable. |
| **Yours** | One SQLite file on your disk. No account, no telemetry, works with the network off. |

**[Read the field manual →](docs/PRAXIS.html)** — every mode, every tool, how a
turn works, and an honest section on what it isn't.

---

## Picking a backend

Set `PROVIDER` in `.env`. Everything except Claude has a free path.

| Backend | Cost | Offline | Setup |
|---|---|---|---|
| `demo` | free | yes | none — already running |
| `ollama` | free | yes | install, `ollama pull llama3.2` — **recommended** |
| `groq` | free tier | no | free key, no card, at [console.groq.com](https://console.groq.com) |
| `gemini` | free tier | no | free key at [aistudio.google.com](https://aistudio.google.com/apikey) |
| `openai` | varies | depends | LM Studio, vLLM, OpenRouter, Together… |
| `anthropic` | paid | no | strongest option, if you want one |
| `scratch` | free | yes | the model you train below |

Set `FALLBACK_CHAIN=ollama,groq,demo` and it health-checks them in order — if
your local model is down it moves on and says why, rather than failing.

---

## Where to run it

Praxis is a Python server, so something has to actually run it.

- **Your own machine** — free, private, offline. Start here.
- **Tailscale** — ten minutes, gets it on your phone, nothing exposed publicly.
- **GitHub Codespaces** — one click; this repo ships a `.devcontainer/`.
  Port forwarding is private by default, which is what makes that safe.
- **GitHub Pages** — hosts *this landing page* only. It serves static files and
  cannot run Python, so the assistant itself can't live there.

> **Before putting it on the public internet:** Praxis has no login and no
> per-user separation — it binds to localhost on purpose. Deployed publicly
> as-is, anyone with the URL reads your conversations, edits your memory, and
> spends your API quota. **[docs/HOSTING.md](docs/HOSTING.md)** lists what has
> to exist first.

---

## Also in here: how these models actually work

The parts most personal-AI projects skip. All free, all runnable.

**`scratch/` — a transformer from zero.** Not a simplified teaching version:
causal self-attention, pre-norm residual blocks, weight tying, temperature and
top-k sampling. The same architecture family as GPT-4, Claude and Gemini.

```bash
pip install -r requirements-scratch.txt
python scratch/prepare_data.py
python scratch/train.py              # ~15 min on a CPU
python scratch/sample.py --prompt "To be, or" --stream
```

A measured run: validation loss falls from **4.18** (a model guessing at random)
to **1.68** in 16 minutes on four CPU cores. At that loss it writes real English
words, correct punctuation, and the `SPEAKER:` convention of a play — having
been told none of it. It also means nothing at all. That gap between form and
meaning is the most useful thing in this repo.

**`finetune/` — LoRA on an open model.** Teach Llama or Qwen your voice. Fits a
free Colab T4 and produces a ~40 MB adapter rather than a new multi-gigabyte
model.

**[docs/ROADMAP.md](docs/ROADMAP.md)** — what to learn next, in what order, and
which popular advice to ignore.

---

## Layout

```
praxis/          the assistant — identity, modes, agent loop, tools, memory
server.py        the API — streaming chat, conversations, memory, uploads
web/             landing page and the app (no build step)
scratch/         a GPT you train yourself
finetune/        LoRA on an open model
docs/            field manual, hosting guide, learning roadmap
```

Configuration is one file. Copy `.env.example` to `.env`; it holds the backend,
your name, the default mode, which tools are enabled, and the limits. Rename the
whole product with `BRAND_NAME`.

---

## The honest part

Praxis is the scaffolding — identity, memory, tools, interface. The intelligence
is whatever backend you point it at. On a 3-billion-parameter local model it
will be noticeably less capable than ChatGPT, because it is. That's the trade
for free, private and offline, and it's a good trade — but it's a trade.

Matching a frontier model isn't on this path. That costs $100M+ in compute and a
large team. Everything else here is genuinely within reach.
