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

    # How many prior turns to resend. Context windows are finite; this is the
    # crudest possible answer and it is enough until it isn't.
    max_history: int = env_int("MAX_HISTORY", 24)
    # Ceiling on tool-call rounds in a single turn, so a confused model cannot
    # spin forever burning tokens.
    max_tool_rounds: int = env_int("MAX_TOOL_ROUNDS", 6)

    enable_tools: bool = env_bool("ENABLE_TOOLS", True)
    # Check answers against constraints the request actually stated (letter
    # exclusions, word counts, JSON validity) and hand the model its specific
    # violations to repair. Models cannot count their own characters; code can.
    enable_constraint_check: bool = env_bool("ENABLE_CONSTRAINT_CHECK", True)
    max_constraint_retries: int = env_int("MAX_CONSTRAINT_RETRIES", 2)
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
