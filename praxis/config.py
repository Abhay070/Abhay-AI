"""
Configuration. One place for every knob.

Environment variables win over defaults. A `.env` file in the project root is
loaded automatically, so nothing here needs a shell export to work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader. Avoids a dependency for twelve lines of parsing."""
    for candidate in (ROOT / ".env", ROOT / "app" / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Brand:
    """Rename the whole product from here. Every surface reads these."""

    name: str = env("BRAND_NAME", "Praxis")
    tagline: str = env("BRAND_TAGLINE", "Think. Challenge. Build. Verify. Execute.")
    owner: str = env("BRAND_OWNER", "Abhay")
    # praxis (n.) — the process of translating theory into action.
    meaning: str = "the process by which theory becomes action"


@dataclass(frozen=True)
class Settings:
    brand: Brand = field(default_factory=Brand)

    # Which backend answers. See praxis/providers.py.
    provider: str = env("PROVIDER", "demo")
    # Tried in order if the primary is unreachable. Comma-separated.
    fallback_chain: tuple[str, ...] = tuple(
        p.strip() for p in env("FALLBACK_CHAIN", "").split(",") if p.strip()
    )

    host: str = env("HOST", "127.0.0.1")
    port: int = env_int("PORT", 8000)

    db_path: Path = Path(env("DB_PATH", str(ROOT / "data" / "praxis.db")))
    upload_dir: Path = Path(env("UPLOAD_DIR", str(ROOT / "data" / "uploads")))

    # How many prior turns to resend. A ceiling on count, not on size.
    max_history: int = env_int("MAX_HISTORY", 24)
    # The real budget. Counting turns is a poor proxy for cost: twenty short
    # turns and twenty turns each carrying a PDF are the same number and wildly
    # different prompts. Free tiers notice the difference as an HTTP 429 —
    # Groq's on-demand tier allows 8,000 tokens per minute, prompt included —
    # so the default is sized to leave room for an answer inside that.
    max_prompt_tokens: int = env_int("MAX_PROMPT_TOKENS", 4000)
    # How much of an attached file rides along in the prompt. The rest stays on
    # disk and the model pulls it with read_file when it needs it.
    attachment_chars: int = env_int("ATTACHMENT_CHARS", 5000)
    # A rate limit is not a broken backend; it is a working one asking you to
    # wait. Wait, then, rather than failing the turn.
    max_rate_limit_retries: int = env_int("MAX_RATE_LIMIT_RETRIES", 3)

    # --- capacity ----------------------------------------------------------
    # An ordered pool of backends, best first. The order IS the quality order —
    # nothing reorders it by headroom, because a failover that quietly drops to
    # a weaker model while reporting success is the failure this refuses to
    # make. Uses the same spec grammar as the council: backend:model@base-url.
    # Empty falls back to PROVIDER + FALLBACK_CHAIN, so an existing .env keeps
    # working untouched.
    capacity_pool: str = env("CAPACITY_POOL", "")
    enable_capacity_router: bool = env_bool("CAPACITY_ROUTER", True)
    # Past this much already streamed, a mid-answer failover stops and keeps
    # what it has rather than re-spending the whole answer on another backend.
    capacity_restart_max_chars: int = env_int("CAPACITY_RESTART_MAX_CHARS", 1200)
    # Ceiling on tool-call rounds in a single turn, so a confused model cannot
    # spin forever burning tokens.
    max_tool_rounds: int = env_int("MAX_TOOL_ROUNDS", 6)

    enable_tools: bool = env_bool("ENABLE_TOOLS", True)
    # Check answers against constraints the request actually stated (letter
    # exclusions, word counts, JSON validity) and hand the model its specific
    # violations to repair. Models cannot count their own characters; code can.
    enable_constraint_check: bool = env_bool("ENABLE_CONSTRAINT_CHECK", True)
    max_constraint_retries: int = env_int("MAX_CONSTRAINT_RETRIES", 2)
    # Hold a mode to the promises its own prompt makes — Direct's word limit,
    # Brief's five headings, Reality Check's verdict. See praxis/contracts.py.
    enable_mode_contracts: bool = env_bool("ENABLE_MODE_CONTRACTS", True)
    # Catch an answer that stops at "I cannot determine this" and leaves the
    # user where they started, and make it give a way forward. Refusing to
    # help is a failure mode here, not a safe default. See praxis/momentum.py.
    enable_momentum: bool = env_bool("ENABLE_MOMENTUM", True)

    # --- the council -------------------------------------------------------
    # Several models answer the same question; the best answer wins. See
    # praxis/council.py. Members are "backend:model" specs, comma separated.
    # Unreachable members are dropped, so listing a local model alongside
    # hosted ones is safe: it simply sits out when the machine is off.
    council_members: str = env(
        "COUNCIL_MEMBERS",
        "groq:llama-3.3-70b-versatile,gemini:gemini-2.0-flash,ollama:llama3.2")
    # Who arbitrates. Empty falls back to a stated heuristic.
    council_judge: str = env("COUNCIL_JUDGE", "")
    # single | race | council | cascade
    default_strategy: str = env("DEFAULT_STRATEGY", "single")
    council_timeout: float = float(env("COUNCIL_TIMEOUT", "90"))
    # Cascade escalates to this when the first answer looks weak.
    cascade_strong: str = env("CASCADE_STRONG", "groq:llama-3.3-70b-versatile")
    enable_memory: bool = env_bool("ENABLE_MEMORY", True)
    enable_web: bool = env_bool("ENABLE_WEB", True)
    # Off by default: it executes model-written code on your machine.
    enable_code_exec: bool = env_bool("ENABLE_CODE_EXEC", False)
    code_exec_timeout: int = env_int("CODE_EXEC_TIMEOUT", 10)

    default_mode: str = env("DEFAULT_MODE", "standard")
    user_name: str = env("USER_NAME", "")

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
BRAND = settings.brand
