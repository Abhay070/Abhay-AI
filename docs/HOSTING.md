# Where to run Praxis

Praxis is a Python server. Something has to actually run `python server.py` —
the only question is whose computer. That rules some options in and out before
anything else.

## The three GitHub products

| Product | Runs Python? | Verdict |
|---|---|---|
| **GitHub Pages** | No — static files only | Can host `web/index.html` as a public landing page. Cannot host the app. |
| **GitHub Codespaces** | Yes — a full Linux container | **Works.** Private port forwarding means only you can reach it. |
| **GitHub Actions** | Only for the length of a job | No. It's a CI runner, not a server. |

## Codespaces

This repo ships a `.devcontainer/` config, so it is one click:

1. On the repo page: **Code → Codespaces → Create codespace**
2. Wait for the build (a minute or two the first time)
3. `server.py` starts automatically and the browser opens on the forwarded port

**Why it's safe by default.** Codespaces forwards ports *privately* — the URL
requires your GitHub login. That is what makes it acceptable to run an app with
no authentication of its own.

**If you set the port to Public in the Ports panel, stop.** You have just
published an assistant with no login: anyone with the link can read every
conversation, edit your memory, and spend your API quota. There is no warning
beyond this one.

**What to know before relying on it:**

- **It sleeps.** Codespaces stop after ~30 minutes idle. Fine for working
  sessions, wrong for an always-on assistant.
- **Your data lives in the codespace.** `data/praxis.db` holds your
  conversations and memory. Delete the codespace and it is gone. Copy it out
  before you do.
- **The free machine is small.** Ollama on 2 cores is painfully slow. In a
  Codespace, use Groq's free tier instead:
  ```
  PROVIDER=groq
  GROQ_API_KEY=your_key
  ```
- **The free allowance is finite** — GitHub gives personal accounts a monthly
  quota of core-hours and storage. Check the current figure on your own billing
  page; it has changed before.

## Pages, for the landing page only

`web/index.html` and `web/static/` are pure static files. You can publish those
as a public site — useful if you want to show the project to people — while the
app itself runs elsewhere. The **Open Praxis** button would need repointing at
wherever the app actually runs.

## The honest ranking

1. **Your own machine.** Free, private, offline, always yours. What it's built
   for, and where you should start.
2. **Tailscale**, when you want it on your phone. Ten minutes, free, nothing
   exposed to the public internet. Your laptop still does the work.
3. **Codespaces**, when you want it from a machine that isn't yours and don't
   mind it sleeping.
4. **A real host** (Railway, Render, Fly) — only after adding authentication.
   See below.

## Before ever making it public

Praxis binds to `127.0.0.1` on purpose. It is a single-user tool with no login,
no accounts, and no per-user data separation. Putting it on the open internet
as-is means handing strangers your conversations, your memory, and your API
quota.

What it would need first:

- A login — a password gate at minimum, OAuth if you want it to be pleasant
- Per-user rows in the database, so accounts cannot read each other
- Rate limiting, so one visitor cannot drain your API quota
- A `Dockerfile`, so the host has something to build
- `ENABLE_CODE_EXEC` left firmly off — it executes arbitrary Python

That is an afternoon of real work, not a config change. Do it deliberately or
not at all.
