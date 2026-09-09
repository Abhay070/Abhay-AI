"""Current date and time.

Small, but it closes a real failure mode: a model asked "what day is it" will
confidently answer with something from its training data. Giving it a clock is
cheaper than teaching it humility."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import Tool, ToolResult, register


def current_time(timezone_offset: str = "") -> ToolResult:
    now = datetime.now(timezone.utc)
    label = "UTC"
    if timezone_offset:
        try:
            sign = -1 if timezone_offset.strip().startswith("-") else 1
            body = timezone_offset.strip().lstrip("+-")
            hours, _, minutes = body.partition(":")
            delta = timedelta(hours=int(hours), minutes=int(minutes or 0)) * sign
            now = now.astimezone(timezone(delta))
            label = f"UTC{timezone_offset}"
        except (ValueError, TypeError):
            return ToolResult(False, f"Could not parse offset '{timezone_offset}'. "
                                     "Use a form like '+05:30' or '-08:00'.")
    else:
        now = now.astimezone()
        label = now.tzname() or "local"

    return ToolResult(
        True,
        f"{now.strftime('%A, %d %B %Y, %H:%M:%S')} ({label}). "
        f"ISO: {now.isoformat(timespec='seconds')}. "
        f"Week {now.isocalendar().week}, day {now.timetuple().tm_yday} of the year.",
        {"iso": now.isoformat(timespec="seconds")},
    )


register(Tool(
    name="current_time",
    description="Get the current date and time. Use whenever the answer depends on "
                "today's date, or to compute how long until/since something",
    args={"timezone_offset": "optional, e.g. '+05:30'. Omit for server local time"},
    run=current_time,
    icon="◷",
))
