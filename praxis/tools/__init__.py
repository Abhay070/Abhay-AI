"""
Tool registry.

A tool is a named function the model can call, with a JSON-schema-ish signature
it can read. The registry is the only thing the agent knows about; individual
tools are registered here and nowhere else needs to change.

Design constraint worth stating: tool calls are driven by a text protocol
(`<tool_call>{...}</tool_call>`) rather than a provider's native function-calling
API. That is deliberate. Native tool calling is better where it exists, but it
exists differently on every provider and not at all on many local models. The
text protocol works identically on Ollama, Groq, Gemini, and a model you trained
yourself last Tuesday — which is the whole point of this project.

Swapping to native calling per-provider is a clean upgrade later; the tool
implementations below do not change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


@dataclass
class ToolResult:
    ok: bool
    output: str
    # Anything structured the UI should render (a table, a source list, a chart).
    data: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """How the result is fed back to the model."""
        status = "OK" if self.ok else "ERROR"
        return f"[{status}] {self.output}"


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str]              # arg name -> human description
    run: Callable[..., ToolResult]
    icon: str = "▸"
    requires: str = ""                # a settings flag that must be on
    dangerous: bool = False

    def signature(self) -> str:
        if not self.args:
            return f"- {self.name}: {self.description} (no arguments)"
        arg_str = ", ".join(f"{k} ({v})" for k, v in self.args.items())
        return f"- {self.name}: {self.description} | args: {arg_str}"


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


def available(settings) -> list[Tool]:
    """Tools that are actually usable given the current configuration."""
    out = []
    for tool in REGISTRY.values():
        if tool.requires and not getattr(settings, tool.requires, False):
            continue
        out.append(tool)
    return out


def describe(tools: list[Tool]) -> str:
    return "\n".join(t.signature() for t in tools)


def parse_calls(text: str) -> list[dict]:
    """Extract tool calls from model output. Malformed JSON is skipped rather
    than raised — a model that emits a broken call should get a clear error
    back, not crash the request."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            calls.append({
                "name": str(payload["name"]),
                "args": payload.get("args") or payload.get("arguments") or {},
                "raw": match.group(0),
            })
    return calls


def strip_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def execute(name: str, args: dict, context: dict | None = None) -> ToolResult:
    tool = REGISTRY.get(name)
    if tool is None:
        known = ", ".join(REGISTRY) or "none"
        return ToolResult(False, f"No tool named '{name}'. Available: {known}")
    if not isinstance(args, dict):
        return ToolResult(False, f"Arguments for '{name}' must be an object.")
    try:
        # Tools that need conversation state declare a `context` parameter.
        import inspect
        if "context" in inspect.signature(tool.run).parameters:
            return tool.run(context=context or {}, **args)
        return tool.run(**args)
    except TypeError as e:
        return ToolResult(False, f"Bad arguments for '{name}': {e}")
    except Exception as e:  # a tool must never take down the request
        return ToolResult(False, f"{type(e).__name__}: {e}")


# Importing these registers them. Order here is the order shown to the model.
from . import calculate, timetool, memory, web, code, files  # noqa: E402,F401
