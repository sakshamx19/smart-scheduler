"""Google Calendar tools for LangChain. The calendar logic itself is in google_calendar.py and scheduling.py, the same
files KIRA ran while this agent was built and tested there; this module only wires them to LangChain and to a
Google service account."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ..config import env, require
from . import google_calendar as gcal
from .google_auth import ServiceAccount


@lru_cache
def settings() -> gcal.CalendarSettings:
    days = env("WORK_DAYS", "monday,tuesday,wednesday,thursday,friday")
    return gcal.CalendarSettings(
        calendar_id=require("GOOGLE_CALENDAR_ID"),
        timezone=env("CALENDAR_TIMEZONE", "UTC"),
        work_start=env("WORK_START", "09:00"),
        work_end=env("WORK_END", "18:00"),
        work_days=tuple(d.strip().lower() for d in days.split(",") if d.strip()),
    )


def _tool(calendar: gcal.GoogleCalendar, name: str, tool: str) -> BaseTool:
    description, parameters, needs_approval, _ = gcal.TOOLS[tool]

    async def run(**arguments: Any) -> str:
        return await gcal.run_tool(tool, arguments, calendar)  # a CalendarError reaches the model as the tool's error

    return StructuredTool.from_function(
        coroutine=run, name=name, description=description, args_schema=parameters, metadata={"needs_approval": needs_approval}
    )


def calendar_tools(names: dict[str, str]) -> list[BaseTool]:
    """``names``: the tool name the model sees → the calendar tool it runs."""
    calendar = gcal.GoogleCalendar(settings(), ServiceAccount(gcal.SCOPES).token)
    return [_tool(calendar, name, tool) for name, tool in names.items()]


def calendar_context() -> str:
    """Today, this week, next week, month ends and working hours, worked out fresh for every turn."""
    return gcal.date_context(settings())


def describe_calendar_call(tool: str, args: dict[str, Any]) -> str:
    return gcal.describe_call(tool, args, settings())
