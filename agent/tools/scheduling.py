"""Finding meeting slots: busy intervals in, open slots out.

Pure functions with no I/O, so they are easy to test. KIRA's Google Calendar toolkit uses this module, and every
exported agent that uses Google Calendar gets this same file, so it imports nothing from KIRA.

All interval arithmetic happens in UTC; local time (``SlotQuery.tz``) is used only to decide which hours of which
days count, and to line slots up with the clock (2:00 and 2:30, not 2:07). That keeps durations exact across a
daylight-saving change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Iterable

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MIDNIGHT = time(0)


@dataclass(frozen=True)
class Interval:
    start: datetime  # timezone-aware
    end: datetime

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class SlotQuery:
    """What to look for. ``start`` and ``end`` are timezone-aware; ``earliest`` and ``latest`` are local times of
    day in ``tz`` (``latest`` = 00:00 means midnight at the end of the day)."""

    start: datetime
    end: datetime
    duration: timedelta
    tz: tzinfo
    earliest: time = time(9)
    latest: time = time(18)
    days: frozenset[int] = frozenset(range(5))  # weekdays allowed, 0 = Monday
    skip_dates: frozenset[date] = frozenset()
    buffer: timedelta = timedelta(0)  # kept free before and after every busy interval
    step: timedelta = timedelta(minutes=30)  # slots start on multiples of this on the local clock


def utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Sorted, with overlapping and touching intervals joined."""
    out: list[Interval] = []
    for iv in sorted(intervals, key=lambda i: i.start):
        if out and iv.start <= out[-1].end:
            if iv.end > out[-1].end:
                out[-1] = Interval(out[-1].start, utc(iv.end))
        else:
            out.append(Interval(utc(iv.start), utc(iv.end)))
    return out


def day_windows(q: SlotQuery) -> list[Interval]:
    """The parts of the search window that fall on allowed days, between ``earliest`` and ``latest``."""
    out: list[Interval] = []
    day = q.start.astimezone(q.tz).date()
    last = q.end.astimezone(q.tz).date()
    while day <= last:
        if day.weekday() in q.days and day not in q.skip_dates:
            lo = datetime.combine(day, q.earliest, q.tz)
            hi_day = day + timedelta(days=1) if q.latest == MIDNIGHT else day
            hi = datetime.combine(hi_day, q.latest, q.tz)
            lo, hi = max(utc(lo), utc(q.start)), min(utc(hi), utc(q.end))
            if hi > lo:
                out.append(Interval(lo, hi))
        day += timedelta(days=1)
    return out


def free_intervals(busy: Iterable[Interval], q: SlotQuery) -> list[Interval]:
    """Open stretches inside the allowed hours that are long enough for the meeting."""
    blocked = merge(Interval(b.start - q.buffer, b.end + q.buffer) for b in busy)
    out: list[Interval] = []
    for window in day_windows(q):
        cursor = window.start
        for b in blocked:
            if b.end <= cursor:
                continue
            if b.start >= window.end:
                break
            if b.start > cursor:
                out.append(Interval(cursor, b.start))
            cursor = max(cursor, b.end)
            if cursor >= window.end:
                break
        if cursor < window.end:
            out.append(Interval(cursor, window.end))
    return [f for f in out if f.end - f.start >= q.duration]


def align_up(moment: datetime, q: SlotQuery) -> datetime:
    """The first time at or after ``moment`` that sits on the step grid of the local clock."""
    local = moment.astimezone(q.tz)
    midnight = datetime.combine(local.date(), MIDNIGHT, q.tz)
    step = q.step.total_seconds()
    elapsed = (utc(moment) - utc(midnight)).total_seconds()
    steps = -(-elapsed // step)  # ceiling division
    return utc(midnight) + timedelta(seconds=steps * step)


def _starts(free: Interval, q: SlotQuery) -> list[datetime]:
    """Start times inside one open stretch: its first on-grid time, then one every hour (or every meeting length,
    if longer), so a long free afternoon offers 1:00, 2:00 and 3:00 rather than 1:00, 1:30 and 2:00."""
    gap = max(q.duration, timedelta(hours=1))
    out: list[datetime] = []
    start = align_up(free.start, q)
    while start + q.duration <= free.end:
        out.append(start)
        start = align_up(start + gap, q)
    return out


def find_slots(busy: Iterable[Interval], q: SlotQuery, limit: int = 5) -> list[Interval]:
    """Up to ``limit`` open slots, spread out: every day's first choice comes before any day's second choice, and
    within a day the first time of each open stretch comes before extra times in the same stretch."""
    by_day: dict[date, list[datetime]] = {}
    firsts: dict[date, list[datetime]] = {}
    for free in free_intervals(busy, q):
        starts = _starts(free, q)
        if not starts:
            continue
        day = starts[0].astimezone(q.tz).date()
        firsts.setdefault(day, []).append(starts[0])
        by_day.setdefault(day, []).extend(starts[1:])
    ranked = {day: firsts[day] + by_day.get(day, []) for day in firsts}

    picked: list[datetime] = []
    rank = 0
    while len(picked) < limit and any(rank < len(v) for v in ranked.values()):
        for day in sorted(ranked):
            if rank < len(ranked[day]) and len(picked) < limit:
                picked.append(ranked[day][rank])
        rank += 1
    return [Interval(s, s + q.duration) for s in sorted(picked)]


def conflicts(busy: Iterable[Interval], slot: Interval) -> list[Interval]:
    return [b for b in busy if b.overlaps(slot)]
