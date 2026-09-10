"""
Check that Praxis can actually answer, and say exactly what to fix if it cannot.

    python scripts/doctor.py

Runs the whole path end to end: reads your .env, resolves the backend, asks
your account what models it really offers, sends one real question, and reports
what came back. Every check that fails prints the specific line to change.

This exists because the failures that hurt most all looked identical from the
outside — a reply that never arrived. A retired model name, a rejected key, a
free tier's per-minute allowance, and a thinking model that used its whole
output budget without writing an answer are four different problems with one
symptom, and guessing between them is miserable.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx                                                     # noqa: E402

from praxis import contracts as contracts_mod                    # noqa: E402
from praxis import modes as modes_mod                            # noqa: E402
from praxis import providers                                     # noqa: E402
from praxis import tools as toolkit                              # noqa: E402
from praxis.agent import CHARS_PER_TOKEN, Agent                  # noqa: E402
from praxis.config import settings                               # noqa: E402
from praxis.providers import EmptyAnswer, ProviderError, RateLimited  # noqa: E402
from praxis.store import Store                                   # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")
TICK, CROSS, WARN = f"{GREEN}✓{OFF}", f"{RED}✗{OFF}", f"{YELLOW}!{OFF}"

problems: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  {TICK} {label}" + (f"  {DIM}{detail}{OFF}" if detail else ""))


def bad(label: str, detail: str, fix: str) -> None:
    print(f"  {CROSS} {label}  {DIM}{detail}{OFF}")
    print(f"      {BOLD}fix:{OFF} {fix}")
    problems.append(label)


def warn(label: str, detail: str, note: str = "") -> None:
    print(f"  {WARN} {label}  {DIM}{detail}{OFF}")
    if note:
        print(f"      {DIM}{note}{OFF}")


async def main() -> int:
    print(f"\n{BOLD}Praxis doctor{OFF}\n")

    # --- 1. configuration --------------------------------------------------
    print(f"{BOLD}Configuration{OFF}")
    env = ROOT / ".env"
    if env.exists():
        ok(".env found", str(env))
    else:
        bad(".env is missing", "Praxis is running on defaults",
            "cp .env.example .env   then put your key in it")

    keys = {name: bool(os.getenv(name)) for name in
            ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    held = [k for k, v in keys.items() if v]
    if held:
        ok("API key loaded", ", ".join(held))
    elif settings.provider in ("demo", "ollama", "scratch"):
        ok("no API key needed", f"PROVIDER={settings.provider}")
    else:
        bad("no API key is set", f"PROVIDER={settings.provider} needs one",
            "get a free key at console.groq.com/keys and put it in .env — "
            "then restart")

    if settings.provider == "demo":
        warn("PROVIDER=demo", "the demo backend is a script, not a model",
             "It pattern-matches your message and reads back a prepared reply. "
             "Every mode will answer identically because nothing is reading the "
             "prompt. Set PROVIDER=groq (or ollama) in .env.")

    # --- 2. the backend ----------------------------------------------------
    print(f"\n{BOLD}Backend{OFF}")
    provider, notes = await providers.resolve(settings.provider,
                                              settings.fallback_chain)
    for note in notes:
        warn("skipped a backend", note)
    healthy, detail = await provider.health()
    if healthy:
        ok(f"{provider.label} is reachable", detail)
    else:
        bad(f"{provider.label} is not usable", detail,
            "run  python scripts/list_models.py  to see what your key can reach")

    model = getattr(provider, "model", "")
    if model and provider.name in ("groq", "openai"):
        base = getattr(provider, "base", "")
        try:
            async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
                r = await client.get(f"{base}/models", headers={
                    "Authorization": f"Bearer {getattr(provider, 'api_key', '')}"})
            catalogue = [m["id"] for m in r.json().get("data", [])]
        except Exception:
            catalogue = []
        if catalogue and model not in catalogue:
            bad(f"{model!r} is not in your account's catalogue",
                f"{len(catalogue)} models available",
                "run  python scripts/list_models.py  and paste the suggested "
                "line into .env — this is the cause of a 404 'model does not "
                "exist'")
        elif catalogue:
            ok("the configured model exists", model)

    thinks = bool(model and providers.REASONING_MODELS.search(model))
    if thinks:
        effort = getattr(provider, "reasoning_effort", "")
        ok(f"{model} thinks before answering",
           f"REASONING_EFFORT={effort or 'the model default'}")

    # --- 3. what a turn costs ---------------------------------------------
    print(f"\n{BOLD}Cost per turn{OFF}")
    store = Store(settings.db_path)
    agent = Agent(store, settings)
    cid = store.create_conversation("doctor")
    store.add_message(cid, "user", "What is the capital of Australia?")
    messages, _, tools = agent.compose(store.get_messages(cid), "standard")
    prompt_tokens = sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN
    system_tokens = len(messages[0]["content"]) // CHARS_PER_TOKEN
    ok("prompt size on a fresh chat",
       f"~{prompt_tokens} tokens ({system_tokens} of it the system prompt, "
       f"{len(tools)} tools)")
    ceiling = settings.max_prompt_tokens
    ok("prompt budget", f"MAX_PROMPT_TOKENS={ceiling} "
                        f"(~{ceiling * CHARS_PER_TOKEN:,} characters of history)")
    if provider.name == "groq" and prompt_tokens > 2600:
        warn("that is a large share of a free tier's minute",
             f"Groq's on-demand tier allows 8,000 tokens per minute",
             "Praxis waits out a 429 and drops older turns to fit, so this is "
             "survivable — but lowering MAX_PROMPT_TOKENS makes it rarer.")

    # --- 4. a real question -----------------------------------------------
    print(f"\n{BOLD}A real question{OFF}")
    question = [{"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "What is the capital of Australia?"}]
    started = time.time()
    answer, failure = "", ""
    try:
        async for chunk in provider.stream(question):
            answer += chunk
    except RateLimited as e:
        failure = f"rate limited ({e}); Praxis would wait {max(e.retry_after, 2):.0f}s"
    except EmptyAnswer as e:
        failure = str(e)
    except ProviderError as e:
        failure = str(e)
    elapsed = time.time() - started

    if answer.strip():
        ok(f"answered in {elapsed:.1f}s", " ".join(answer.split())[:90])
        if "canberra" not in answer.lower():
            warn("but it did not say Canberra",
                 "the machinery works; this is the model's knowledge",
                 "Try a larger model, or DEFAULT_STRATEGY=council to let "
                 "several models answer and take the best.")
    elif failure:
        bad("no answer came back", failure[:160],
            "the message above names what to change")
    else:
        bad("no answer and no error", "this should be impossible",
            "please report this — it is the exact bug the EmptyAnswer class "
            "exists to prevent")

    # --- 5. promises -------------------------------------------------------
    print(f"\n{BOLD}Features{OFF}")
    enforced = [k for k in modes_mod.MODES if contracts_mod.describe(k)]
    ok(f"{len(modes_mod.MODES)} modes",
       f"{len(enforced)} hold a mechanically enforced promise")
    ok(f"{len(toolkit.available(settings))} tools enabled",
       ", ".join(t.name for t in toolkit.available(settings)[:6]) + "…")
    for flag, label in (("enable_constraint_check", "constraint checking"),
                        ("enable_mode_contracts", "mode contracts"),
                        ("enable_memory", "memory"),
                        ("enable_web", "web search")):
        (ok if getattr(settings, flag) else warn)(
            label, "on" if getattr(settings, flag) else "off — set it in .env")

    store.delete_conversation(cid)

    print()
    if problems:
        print(f"  {RED}{len(problems)} problem(s):{OFF} " + "; ".join(problems))
        print(f"  {DIM}Each one printed the line to change above.{OFF}\n")
        return 1
    print(f"  {GREEN}Everything checks out.{OFF}")
    print(f"  {DIM}Next: python tests/benchmark.py   (100 questions, scored){OFF}")
    print(f"  {DIM}      python scripts/compare_modes.py \"your question\"{OFF}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
