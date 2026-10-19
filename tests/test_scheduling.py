"""Slot finding and the calendar tools against a fake calendar: no Google account needed.

The fake week (Asia/Kolkata), seen from Sunday Oct 18, 2026 at 10:00:
    Tue Oct 20   12:00–14:00 Design review, 15:00–16:30 1:1 with Sam
    Wed Oct 21   13:00–14:00 Lunch (marked free)
    Thu Oct 22   11:00–12:00 Project Alpha Kick-off
    Fri Oct 23   17:00–17:30 Weekly sync, 18:00–20:00 Flight to Delhi
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from agent.tools import google_calendar as gcal

TZ = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 18, 10, 0, tzinfo=TZ)


def event(title: str, start: str, end: str, busy: bool = True) -> gcal.Event:
    at = lambda text: datetime.fromisoformat(text).replace(tzinfo=TZ)  # noqa: E731
    return gcal.Event(id=title, title=title, start=at(start), end=at(end), all_day=False, busy=busy)


EVENTS = [
    event("Design review", "2026-10-20T12:00", "2026-10-20T14:00"),
    event("1:1 with Sam", "2026-10-20T15:00", "2026-10-20T16:30"),
    event("Lunch", "2026-10-21T13:00", "2026-10-21T14:00", busy=False),
    event("Project Alpha Kick-off", "2026-10-22T11:00", "2026-10-22T12:00"),
    event("Weekly sync", "2026-10-23T17:00", "2026-10-23T17:30"),
    event("Flight to Delhi", "2026-10-23T18:00", "2026-10-23T20:00"),
]


class FakeCalendar(gcal.GoogleCalendar):
    def __init__(self) -> None:
        async def token() -> str:
            return "unused"

        super().__init__(gcal.CalendarSettings("me@example.com", "Asia/Kolkata"), token)

    async def events(self, start: datetime, end: datetime, *, query: str | None = None, limit: int = 250) -> list[gcal.Event]:
        found = [e for e in EVENTS if e.start < end and e.end > start]
        return [e for e in found if not query or query.lower() in e.title.lower()][:limit]


def run(tool: str, **args: Any) -> dict[str, Any]:
    return json.loads(asyncio.run(gcal.run_tool(tool, args, FakeCalendar(), now=NOW)))


def starts(result: dict[str, Any], key: str = "slots") -> list[str]:
    return [s["start"] for s in result[key]]


def test_tuesday_afternoon_offers_the_gaps_between_meetings() -> None:
    result = run("find_free_slots", duration_minutes=60, start="2026-10-20", end="2026-10-20", earliest="12:00")
    assert starts(result) == ["2026-10-20T14:00+05:30", "2026-10-20T16:30+05:30"]
    assert [b["title"] for b in result["busy"]] == ["Design review", "1:1 with Sam"]


def test_a_full_window_offers_alternatives_instead_of_nothing() -> None:
    result = run("find_free_slots", duration_minutes=120, start="2026-10-20", end="2026-10-20", earliest="12:00")
    assert result["slots"] == []
    assert result["alternatives"], "a booked-up afternoon must still offer something"
    assert {a["why"] for a in result["alternatives"]} >= {"same days, other hours", "later days, same hours"}


def test_vague_and_negative_constraints() -> None:
    result = run("find_free_slots", duration_minutes=30, start="2026-10-19", end="2026-10-25", earliest="10:00", exclude_days=["wednesday"])
    days = {s[:10] for s in starts(result)}
    assert "2026-10-21" not in days, "Wednesday was excluded"
    assert all(s[11:16] >= "10:00" for s in starts(result)), "nothing before 10"


def test_a_buffer_after_the_last_meeting() -> None:
    result = run("find_free_slots", duration_minutes=60, start="2026-10-23", end="2026-10-23", earliest="19:00", latest="24:00", buffer_minutes=60)
    assert starts(result)[0] == "2026-10-23T21:00+05:30", "an hour after the flight lands at 20:00"


def test_events_marked_free_do_not_block() -> None:
    result = run("find_free_slots", duration_minutes=60, start="2026-10-21", end="2026-10-21", earliest="13:00", latest="14:00")
    assert starts(result) == ["2026-10-21T13:00+05:30"]


def test_a_weekend_window_says_why_it_is_empty() -> None:
    result = run("find_free_slots", duration_minutes=60, start="2026-10-24", end="2026-10-25")
    assert result["slots"] == [] and "include_weekends" in result["note"]


def test_finding_an_event_by_name() -> None:
    result = run("list_events", query="alpha")
    assert [e["title"] for e in result["events"]] == ["Project Alpha Kick-off"]


def test_booking_a_taken_time_is_refused() -> None:
    with pytest.raises(gcal.CalendarError, match="Not booked"):
        run("create_event", title="Sync", start="2026-10-20T12:30", duration_minutes=30)


def test_unclear_input_is_explained() -> None:
    with pytest.raises(gcal.CalendarError, match="not a date or time"):
        run("find_free_slots", duration_minutes=60, start="next tuesday", end="2026-10-20")


def test_date_context_works_out_the_calendar() -> None:
    context = gcal.date_context(gcal.CalendarSettings("me@example.com", "Asia/Kolkata"), NOW)
    assert "Next week: Mon Oct 19 – Sun Oct 25" in context
    assert "its last weekday is Friday, Oct 30" in context
