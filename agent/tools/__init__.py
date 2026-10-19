"""The tools Smart Scheduler was given in KIRA, and what they add to the prompt.

    find_free_slots, list_events, create_event, update_event, delete_event   Google Calendar (calendar.py)
    current_time                                                             built-in (builtin.py)
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from .builtin import builtin_tools
from .calendar import calendar_context, calendar_tools, describe_calendar_call

# The name the model sees → the tool it runs
BUILTINS = {"current_time": "current_time"}
CALENDAR = {
    "find_free_slots": "find_free_slots",
    "list_events": "list_events",
    "create_event": "create_event",
    "update_event": "update_event",
    "delete_event": "delete_event",
}


async def load_tools() -> list[BaseTool]:
    tools: list[BaseTool] = []
    tools += builtin_tools(BUILTINS)
    tools += calendar_tools(CALENDAR)
    return tools


def context() -> str:
    """Facts the tools add to the system prompt, worked out fresh on every turn."""
    return calendar_context()


def describe(name: str, args: dict[str, Any]) -> str:
    """One line on what a call that needs approval will do: shown on the approval card, and spoken."""
    if name in CALENDAR:
        return describe_calendar_call(CALENDAR[name], args)
    shown = ", ".join(f"{key}: {value}" for key, value in args.items())
    return f"Run {name} with {shown}." if shown else f"Run {name}."
