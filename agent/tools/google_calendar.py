"""Google Calendar for agents: a small async client for the Calendar REST API, and the tools agents call.

KIRA runs these tools for its Google Calendar toolkit, and copies this file (with scheduling.py) into every agent
export that uses one, so it imports nothing from KIRA. The caller supplies the access token.

Every tool answers JSON written for the model. Each time comes with a spoken-style label ("Tuesday, Oct 20,
2:00 PM – 3:00 PM") next to its ISO value, so the model can read times out without doing date arithmetic, and
``date_context`` gives the model the facts it would otherwise guess: today, this week, next week, month ends.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .scheduling import MIDNIGHT, WEEKDAYS, Interval, SlotQuery, day_windows, find_slots, utc

API = "https://www.googleapis.com/calendar/v3"
SCOPES = "https://www.googleapis.com/auth/calendar"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MAX_WINDOW_DAYS = 62  # the longest search window find_free_slots accepts
ALTERNATIVE_DAYS = 7  # how far past the window find_free_slots looks for alternatives
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

TokenProvider = Callable[[], Awaitable[str]]


class CalendarError(Exception):
    """A problem the model should hear about. The text says what went wrong and, where it can, what to do."""

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason  # Google's error reason, e.g. "forbiddenForServiceAccounts"


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise CalendarError(f"Unknown time zone {name!r}; use an IANA name such as Asia/Kolkata.") from None


@dataclass(frozen=True)
class CalendarSettings:
    calendar_id: str
    timezone: str = "UTC"
    work_start: str = "09:00"
    work_end: str = "18:00"
    work_days: tuple[str, ...] = WEEKDAYS[:5]

    @property
    def tz(self) -> ZoneInfo:
        return zone(self.timezone)


# ---------------------------------------------------------------------------------------------- labels
def clock(moment: datetime | time) -> str:
    """2:30 PM"""
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def day_label(day: date, this_year: int | None = None) -> str:
    """Tuesday, Oct 20 (with the year when it is not this year)"""
    text = f"{WEEKDAYS[day.weekday()].capitalize()}, {MONTHS[day.month - 1]} {day.day}"
    return text if this_year in (None, day.year) else f"{text}, {day.year}"


def short_day(day: date) -> str:
    """Tue Oct 20"""
    return f"{WEEKDAYS[day.weekday()][:3].capitalize()} {MONTHS[day.month - 1]} {day.day}"


def span_label(start: datetime, end: datetime, tz: ZoneInfo, *, all_day: bool = False) -> str:
    s, e = start.astimezone(tz), end.astimezone(tz)
    year = datetime.now(tz).year
    if all_day:
        last = (e - timedelta(days=1)).date()  # an all-day event's end date is exclusive
        if last <= s.date():
            return f"{day_label(s.date(), year)} (all day)"
        return f"{day_label(s.date(), year)} – {day_label(last, year)} (all day)"
    if s.date() == e.date() or (e.time() == MIDNIGHT and e.date() == s.date() + timedelta(days=1)):
        return f"{day_label(s.date(), year)}, {clock(s)} – {clock(e)}"
    return f"{day_label(s.date(), year)}, {clock(s)} – {day_label(e.date(), year)}, {clock(e)}"


def local_iso(moment: datetime, tz: ZoneInfo) -> str:
    return moment.astimezone(tz).isoformat(timespec="minutes")


def _days_text(days: tuple[str, ...] | list[str]) -> str:
    names = [d.capitalize() for d in WEEKDAYS if d in days]
    if names == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        return "Monday to Friday"
    return ", ".join(names) or "no days"


# ---------------------------------------------------------------------------------------------- arguments
def parse_when(value: Any, tz: ZoneInfo, *, field: str, end_of_day: bool = False) -> datetime:
    """A date (2026-10-20) or a date and time (2026-10-20T14:00, local to the calendar unless it has an offset).
    With ``end_of_day``, a date alone means the end of that day."""
    text = str(value or "").strip()
    if not text:
        raise CalendarError(f"{field} is required: a date (2026-10-20) or a date and time (2026-10-20T14:00).")
    if DATE_RE.fullmatch(text):
        try:
            day = date.fromisoformat(text)
        except ValueError:
            raise CalendarError(f"{field}: {text!r} is not a real date.") from None
        return datetime.combine(day + timedelta(days=1) if end_of_day else day, MIDNIGHT, tz)
    try:
        moment = datetime.fromisoformat(text.replace(" ", "T", 1))
    except ValueError:
        raise CalendarError(f"{field}: {text!r} is not a date or time. Use 2026-10-20 or 2026-10-20T14:00.") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=tz)


def parse_clock(value: Any, field: str) -> time:
    """14:30, 2:30 pm, 9am; 24:00 means midnight at the end of the day."""
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not match:
        raise CalendarError(f"{field}: {value!r} is not a time of day; use HH:MM, e.g. 13:30.")
    hour, minute, half = int(match[1]), int(match[2] or 0), match[3]
    if half:
        if not 1 <= hour <= 12:
            raise CalendarError(f"{field}: {value!r} is not a time of day.")
        hour = hour % 12 + (12 if half == "pm" else 0)
    if (hour, minute) == (24, 0):
        return MIDNIGHT
    if hour > 23 or minute > 59:
        raise CalendarError(f"{field}: {value!r} is not a time of day.")
    return time(hour, minute)


def parse_int(args: dict[str, Any], key: str, *, lo: int, hi: int, default: int | None = None) -> int:
    value = args.get(key)
    if value in (None, ""):
        if default is None:
            raise CalendarError(f"{key} is required.")
        return default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        raise CalendarError(f"{key} must be a number.") from None
    if not lo <= number <= hi:
        raise CalendarError(f"{key} must be between {lo} and {hi}.")
    return number


def _flag(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


def _text(args: dict[str, Any], key: str, limit: int) -> str:
    value = str(args.get(key) or "").strip()
    if len(value) > limit:
        raise CalendarError(f"{key} is longer than {limit} characters.")
    return value


def weekday_index(name: str) -> int:
    text = name.strip().lower()
    for i, day in enumerate(WEEKDAYS):
        if len(text) >= 3 and day.startswith(text):
            return i
    raise CalendarError(f"{name!r} is not a weekday name.")


def _exclusions(value: Any) -> tuple[set[int], set[date]]:
    """Weekday names and dates to skip."""
    items = re.split(r"[,\s]+", value) if isinstance(value, str) else list(value or [])
    days: set[int] = set()
    dates: set[date] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if DATE_RE.fullmatch(text):
            dates.add(parse_when(text, timezone.utc, field="exclude_days").date())
        else:
            days.add(weekday_index(text))
    return days, dates


def _emails(value: Any) -> list[str]:
    items = re.split(r"[,\s]+", value) if isinstance(value, str) else list(value or [])
    emails = [str(e).strip() for e in items if str(e).strip()]
    for email in emails:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise CalendarError(f"{email!r} is not an email address. Ask the user for their guests' email addresses.")
    return emails


# ---------------------------------------------------------------------------------------------- the client
@dataclass(frozen=True)
class Event:
    id: str
    title: str
    start: datetime
    end: datetime
    all_day: bool
    busy: bool  # counts against availability: not marked "free", not declined, not a working-location note
    location: str = ""
    attendees: tuple[str, ...] = ()
    link: str = ""

    def public(self, tz: ZoneInfo) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "when": span_label(self.start, self.end, tz, all_day=self.all_day),
            "start": self.start.astimezone(tz).date().isoformat() if self.all_day else local_iso(self.start, tz),
            "end": self.end.astimezone(tz).date().isoformat() if self.all_day else local_iso(self.end, tz),
            "minutes": int((self.end - self.start).total_seconds() // 60),
        }
        if self.all_day:
            out["all_day"] = True
        if not self.busy:
            out["shows_as_free"] = True
        if self.location:
            out["location"] = self.location
        if self.attendees:
            out["attendees"] = list(self.attendees)
        return out

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


def parse_event(item: dict[str, Any], tz: ZoneInfo) -> Event | None:
    if item.get("status") == "cancelled":
        return None
    start, end = item.get("start") or {}, item.get("end") or {}
    if "dateTime" in start:
        s = datetime.fromisoformat(start["dateTime"])
        e = datetime.fromisoformat(end.get("dateTime") or start["dateTime"])
        all_day = False
    elif "date" in start:
        s = datetime.combine(date.fromisoformat(start["date"]), MIDNIGHT, tz)
        e = datetime.combine(date.fromisoformat(end.get("date") or start["date"]), MIDNIGHT, tz)
        all_day = True
    else:
        return None
    people = item.get("attendees") or []
    declined = any(a.get("self") and a.get("responseStatus") == "declined" for a in people)
    busy = item.get("transparency") != "transparent" and not declined and item.get("eventType") != "workingLocation"
    guests = tuple(a.get("displayName") or a.get("email", "") for a in people if not a.get("self") and not a.get("resource"))
    return Event(
        id=item.get("id", ""),
        title=item.get("summary") or "(no title)",
        start=s,
        end=e,
        all_day=all_day,
        busy=busy,
        location=item.get("location") or "",
        attendees=tuple(g for g in guests if g),
        link=item.get("htmlLink") or "",
    )


class GoogleCalendar:
    """One calendar, reached with a bearer token from ``token`` (asked for before every request)."""

    def __init__(self, settings: CalendarSettings, token: TokenProvider, *, timeout: float = 20.0) -> None:
        self.settings = settings
        self.tz = settings.tz
        self._token = token
        self._timeout = timeout
        self._path = f"/calendars/{quote(settings.calendar_id, safe='')}"

    async def _call(self, method: str, path: str, *, params: dict[str, str] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {await self._token()}", "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(method, API + path, params=params, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise CalendarError(f"Could not reach Google Calendar ({exc.__class__.__name__}); try again.") from exc
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if resp.is_success:
            return data if isinstance(data, dict) else {}
        error = data.get("error") if isinstance(data, dict) else None
        error = error if isinstance(error, dict) else {}
        message = str(error.get("message") or resp.text[:300] or resp.reason_phrase)
        reason = str(((error.get("errors") or [{}])[0] or {}).get("reason") or "")
        raise CalendarError(self._explain(resp.status_code, reason, message), reason)

    def _explain(self, status: int, reason: str, message: str) -> str:
        calendar = self.settings.calendar_id
        if status == 401:
            return "Google did not accept the calendar credentials; check the service account key."
        if status == 404:
            return f"Not found: the event does not exist, or the calendar {calendar} is not shared with the service account."
        if status == 429 or reason in ("rateLimitExceeded", "userRateLimitExceeded"):
            return "Google Calendar is rate limiting; wait a moment and try again."
        if status == 403 and reason in ("forbidden", "insufficientPermissions", "requiredAccessLevel"):
            return f"No permission on the calendar {calendar} ({message}). Share it with the service account and allow it to make changes to events."
        return f"Google Calendar error {status}: {message}"

    async def info(self) -> dict[str, Any]:
        """The calendar's metadata (summary, timeZone); also a quick check that the credentials work."""
        return await self._call("GET", self._path)

    async def events(self, start: datetime, end: datetime, *, query: str | None = None, limit: int = 250) -> list[Event]:
        """Events overlapping [start, end), recurring ones expanded, in start order."""
        params = {
            "timeMin": utc(start).isoformat(),
            "timeMax": utc(end).isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": str(min(limit, 250)),
            "timeZone": self.settings.timezone,
        }
        if query:
            params["q"] = query
        out: list[Event] = []
        while True:
            data = await self._call("GET", f"{self._path}/events", params=params)
            out += [e for e in (parse_event(item, self.tz) for item in data.get("items") or []) if e]
            page = data.get("nextPageToken")
            if not page or len(out) >= limit:
                return out[:limit]
            params["pageToken"] = page

    async def busy(self, start: datetime, end: datetime) -> list[Event]:
        return [e for e in await self.events(start, end) if e.busy]

    async def get(self, event_id: str) -> Event:
        event = parse_event(await self._call("GET", f"{self._path}/events/{quote(event_id, safe='')}"), self.tz)
        if event is None:
            raise CalendarError(f"The event {event_id} was cancelled.")
        return event

    async def insert(self, body: dict[str, Any], *, notify: bool) -> Event:
        params = {"sendUpdates": "all" if notify else "none"}
        event = parse_event(await self._call("POST", f"{self._path}/events", params=params, body=body), self.tz)
        assert event is not None
        return event

    async def patch(self, event_id: str, body: dict[str, Any]) -> Event:
        params = {"sendUpdates": "all"}
        event = parse_event(await self._call("PATCH", f"{self._path}/events/{quote(event_id, safe='')}", params=params, body=body), self.tz)
        assert event is not None
        return event

    async def delete(self, event_id: str) -> None:
        await self._call("DELETE", f"{self._path}/events/{quote(event_id, safe='')}", params={"sendUpdates": "all"})


# ---------------------------------------------------------------------------------------------- the tools
def _slot(iv: Interval, tz: ZoneInfo) -> dict[str, str]:
    return {"start": local_iso(iv.start, tz), "end": local_iso(iv.end, tz), "label": span_label(iv.start, iv.end, tz)}


def _describe(q: SlotQuery, settings: CalendarSettings) -> str:
    tz = q.tz
    first, last = q.start.astimezone(tz), (q.end - timedelta(seconds=1)).astimezone(tz)
    year = datetime.now(tz).year
    days = day_label(first.date(), year) if first.date() == last.date() else f"{day_label(first.date(), year)} to {day_label(last.date(), year)}"
    hours = f"{clock(q.earliest)} – {'midnight' if q.latest == MIDNIGHT else clock(q.latest)}"
    allowed = _days_text([WEEKDAYS[i] for i in sorted(q.days)])
    text = f"{int(q.duration.total_seconds() // 60)}-minute slots, {days}, {hours}, on {allowed}"
    if q.skip_dates:
        text += ", skipping " + ", ".join(short_day(d) for d in sorted(q.skip_dates))
    if q.buffer:
        text += f", keeping {int(q.buffer.total_seconds() // 60)} minutes free around other events"
    return text


def _alternatives(busy: list[Interval], q: SlotQuery, settings: CalendarSettings, now: datetime) -> list[dict[str, str]]:
    """When nothing fits: the nearest openings outside what was asked, each with why it differs."""
    work_start, work_end = parse_clock(settings.work_start, "work_start"), parse_clock(settings.work_end, "work_end")
    wider_latest = MIDNIGHT if MIDNIGHT in (q.latest, work_end) else max(q.latest, work_end)
    tries: list[tuple[str, SlotQuery]] = []
    if (q.earliest, q.latest) != (min(q.earliest, work_start), wider_latest):
        tries.append(("same days, other hours", replace(q, earliest=min(q.earliest, work_start), latest=wider_latest)))
    tries.append(("later days, same hours", replace(q, start=q.end, end=q.end + timedelta(days=ALTERNATIVE_DAYS))))
    earlier_start = max(utc(now), q.start - timedelta(days=3))
    if earlier_start < q.start:
        tries.append(("earlier days, same hours", replace(q, start=earlier_start, end=q.start)))
    out: list[dict[str, str]] = []
    for why, alt in tries:
        alt = replace(alt, start=max(alt.start, utc(now)))
        if alt.start >= alt.end:
            continue
        for iv in find_slots(busy, alt, limit=2):
            out.append({**_slot(iv, q.tz), "why": why})
    return out[:5]


async def find_free_slots(cal: GoogleCalendar, args: dict[str, Any], now: datetime) -> dict[str, Any]:
    st, tz = cal.settings, cal.tz
    duration = parse_int(args, "duration_minutes", lo=5, hi=720)
    start = parse_when(args.get("start"), tz, field="start")
    end = parse_when(args.get("end"), tz, field="end", end_of_day=True)
    if end <= start:
        raise CalendarError("end must be after start (a date alone as end means the end of that day).")
    if end - start > timedelta(days=MAX_WINDOW_DAYS):
        raise CalendarError(f"Search at most {MAX_WINDOW_DAYS} days at a time.")
    earliest = parse_clock(args.get("earliest") or st.work_start, "earliest")
    latest = parse_clock(args.get("latest") or st.work_end, "latest")
    if latest != MIDNIGHT and latest <= earliest:
        raise CalendarError("latest must be later than earliest; overnight windows are not supported.")
    skip_days, skip_dates = _exclusions(args.get("exclude_days"))
    days = {weekday_index(d) for d in st.work_days} | ({5, 6} if _flag(args.get("include_weekends")) else set())
    buffer = timedelta(minutes=parse_int(args, "buffer_minutes", lo=0, hi=480, default=0))
    limit = parse_int(args, "max_results", lo=1, hi=10, default=5)
    q = SlotQuery(
        start=max(utc(start), utc(now)),
        end=utc(end),
        duration=timedelta(minutes=duration),
        tz=tz,
        earliest=earliest,
        latest=latest,
        days=frozenset(days - skip_days),
        skip_dates=frozenset(skip_dates),
        buffer=buffer,
        step=timedelta(minutes=30 if duration >= 30 else 15),
    )
    result: dict[str, Any] = {"time_zone": st.timezone, "searched": _describe(q, st) if q.start < q.end else "a window that is already over"}

    fetch_from = max(utc(now), q.start - timedelta(days=3)) - buffer
    events = await cal.busy(fetch_from, max(q.end, q.start) + timedelta(days=ALTERNATIVE_DAYS) + buffer)
    busy = [e.interval for e in events]
    slots = find_slots(busy, q, limit) if q.start < q.end else []
    result["slots"] = [_slot(s, tz) for s in slots]
    in_window = [e for e in events if e.interval.overlaps(Interval(q.start, q.end))]
    if in_window:
        result["busy"] = [{"title": e.title, "when": span_label(e.start, e.end, tz, all_day=e.all_day)} for e in in_window[:15]]
        if len(in_window) > 15:
            result["busy_more"] = len(in_window) - 15
    if slots:
        return result

    if q.start >= q.end:
        result["note"] = "That window is already over. Offer the alternatives, or ask for another time."
    elif not day_windows(q):
        result["note"] = (
            "No day in the window is allowed: the dates fall outside the working days "
            f"({_days_text(st.work_days)}) or were excluded. Set include_weekends for a weekend, or ask for another day."
        )
    else:
        result["note"] = "Nothing fits in the window. Say so plainly, then offer the alternatives (nearest openings outside it)."
    result["alternatives"] = _alternatives(busy, q, st, now)
    return result


async def list_events(cal: GoogleCalendar, args: dict[str, Any], now: datetime) -> dict[str, Any]:
    tz = cal.tz
    query = _text(args, "query", 200)
    today = datetime.combine(now.astimezone(tz).date(), MIDNIGHT, tz)
    start = parse_when(args["start"], tz, field="start") if args.get("start") else (now - timedelta(days=90) if query else today)
    end = parse_when(args["end"], tz, field="end", end_of_day=True) if args.get("end") else (now + timedelta(days=90) if query else today + timedelta(days=7))
    if end <= start:
        raise CalendarError("end must be after start.")
    limit = parse_int(args, "max_results", lo=1, hi=50, default=20)
    events = await cal.events(start, end, query=query or None, limit=limit + 1)
    year = now.astimezone(tz).year
    last = (end - timedelta(seconds=1)).astimezone(tz).date()
    result: dict[str, Any] = {
        "time_zone": cal.settings.timezone,
        "range": f"{day_label(start.astimezone(tz).date(), year)} to {day_label(last, year)}",
        "events": [e.public(tz) for e in events[:limit]],
    }
    if query:
        result["query"] = query
    if len(events) > limit:
        result["more"] = "There are more events; narrow the range or raise max_results."
    if not events:
        result["note"] = "No events found." + (" Google matches whole words: try one distinctive word, or a wider range." if query else "")
    return result


async def _check_free(cal: GoogleCalendar, slot: Interval, args: dict[str, Any], *, ignore: str = "") -> None:
    if _flag(args.get("allow_conflict")):
        return
    clash = [e for e in await cal.busy(slot.start, slot.end) if e.id != ignore and e.interval.overlaps(slot)]
    if clash:
        names = "; ".join(f"{e.title} ({span_label(e.start, e.end, cal.tz, all_day=e.all_day)})" for e in clash[:5])
        raise CalendarError(
            f"Not booked: {span_label(slot.start, slot.end, cal.tz)} overlaps {names}. Offer another time "
            "(find_free_slots), or set allow_conflict only if the user explicitly wants to double-book."
        )


def _when_body(start: datetime, end: datetime, settings: CalendarSettings) -> dict[str, Any]:
    tz = settings.tz
    rfc3339 = lambda moment: moment.astimezone(tz).isoformat(timespec="seconds")  # noqa: E731 - Google needs the seconds
    return {
        "start": {"dateTime": rfc3339(start), "timeZone": settings.timezone},
        "end": {"dateTime": rfc3339(end), "timeZone": settings.timezone},
    }


async def create_event(cal: GoogleCalendar, args: dict[str, Any], now: datetime) -> dict[str, Any]:
    tz = cal.tz
    title = _text(args, "title", 200)
    if not title:
        raise CalendarError("title is required.")
    if DATE_RE.fullmatch(str(args.get("start") or "").strip()):
        raise CalendarError("start needs a time as well as a date, e.g. 2026-10-20T14:00.")
    start = parse_when(args.get("start"), tz, field="start")
    end = utc(start) + timedelta(minutes=parse_int(args, "duration_minutes", lo=5, hi=720))
    if end <= utc(now):
        raise CalendarError("That time is already over; pick a time in the future.")
    slot = Interval(utc(start), end)
    await _check_free(cal, slot, args)

    attendees = _emails(args.get("attendees"))
    description = _text(args, "description", 4000)
    body: dict[str, Any] = {"summary": title, **_when_body(slot.start, slot.end, cal.settings)}
    if description:
        body["description"] = description
    if location := _text(args, "location", 300):
        body["location"] = location
    note = ""
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    try:
        event = await cal.insert(body, notify=bool(attendees))
    except CalendarError as exc:
        # A service account without domain-wide delegation may not invite guests: book it anyway, guests noted
        if not attendees or not (exc.reason == "forbiddenForServiceAccounts" or "Service accounts cannot invite" in str(exc)):
            raise
        body.pop("attendees")
        body["description"] = (description + "\n\n" if description else "") + "Guests: " + ", ".join(attendees)
        event = await cal.insert(body, notify=False)
        note = "Booked, but this calendar's account cannot send invitations, so the guests are listed in the description. Tell the user to invite them."
    result: dict[str, Any] = {"booked": True, "event": event.public(tz)}
    if event.link:
        result["link"] = event.link
    if note:
        result["note"] = note
    return result


async def update_event(cal: GoogleCalendar, args: dict[str, Any], now: datetime) -> dict[str, Any]:
    tz = cal.tz
    event_id = _text(args, "event_id", 300)
    if not event_id:
        raise CalendarError("event_id is required; find it with list_events.")
    before = await cal.get(event_id)
    body: dict[str, Any] = {}
    if title := _text(args, "title", 200):
        body["summary"] = title
    if description := _text(args, "description", 4000):
        body["description"] = description
    if args.get("start") or args.get("duration_minutes"):
        start = parse_when(args["start"], tz, field="start") if args.get("start") else before.start
        minutes = parse_int(args, "duration_minutes", lo=5, hi=720, default=int((before.end - before.start).total_seconds() // 60))
        slot = Interval(utc(start), utc(start) + timedelta(minutes=minutes))
        if slot.end <= utc(now):
            raise CalendarError("That time is already over; pick a time in the future.")
        await _check_free(cal, slot, args, ignore=before.id)
        body.update(_when_body(slot.start, slot.end, cal.settings))
    if not body:
        raise CalendarError("Nothing to change: give a new start, duration_minutes, title or description.")
    after = await cal.patch(event_id, body)
    return {"updated": True, "before": span_label(before.start, before.end, tz, all_day=before.all_day), "event": after.public(tz)}


async def delete_event(cal: GoogleCalendar, args: dict[str, Any], now: datetime) -> dict[str, Any]:
    event_id = _text(args, "event_id", 300)
    if not event_id:
        raise CalendarError("event_id is required; find it with list_events.")
    event = await cal.get(event_id)
    await cal.delete(event_id)
    return {"cancelled": True, "title": event.title, "when": span_label(event.start, event.end, cal.tz, all_day=event.all_day)}


Handler = Callable[[GoogleCalendar, dict[str, Any], datetime], Awaitable[dict[str, Any]]]

_WINDOW = {
    "start": {
        "type": "string",
        "description": "Start of the search window: a date (2026-10-20) or a local date and time (2026-10-20T13:00). Times already past are skipped.",
    },
    "end": {
        "type": "string",
        "description": "End of the search window. A date alone means the end of that day, so start=end=2026-10-20 searches that whole day.",
    },
}

# name → (description, JSON schema of the arguments, needs the user's approval, handler)
TOOLS: dict[str, tuple[str, dict[str, Any], bool, Handler]] = {
    "find_free_slots": (
        "Find open meeting times in the calendar. Give the meeting length and a search window; optionally limit the "
        "hours of the day, skip days, or keep a gap around existing events. Answers up to max_results slots spread "
        "across the window and the busy events in it. When nothing fits, it also answers the nearest alternatives "
        "outside the window (other hours, later or earlier days), each with why it differs. Call it again whenever "
        "the length, day or hours change.",
        {
            "type": "object",
            "properties": {
                "duration_minutes": {"type": "integer", "description": "Meeting length in minutes, e.g. 30 or 60."},
                **_WINDOW,
                "earliest": {
                    "type": "string",
                    "description": "Earliest start time each day, HH:MM (24-hour). Default: start of working hours. E.g. 12:00 for 'afternoon', 19:00 for 'after 7 pm', 10:00 for 'not too early'.",
                },
                "latest": {
                    "type": "string",
                    "description": "Latest end time each day, HH:MM. Default: end of working hours. E.g. 12:00 for 'morning', 24:00 for 'any time in the evening'.",
                },
                "exclude_days": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Days to skip: weekday names (wednesday) or dates (2026-10-21).",
                },
                "include_weekends": {"type": "boolean", "description": "Also search Saturday and Sunday. Default false."},
                "buffer_minutes": {
                    "type": "integer",
                    "description": "Minutes to keep free before and after every existing event, e.g. 60 for 'an hour to decompress after my last meeting'. Default 0.",
                },
                "max_results": {"type": "integer", "description": "How many slots to answer, 1 to 10. Default 5."},
            },
            "required": ["duration_minutes", "start", "end"],
        },
        False,
        find_free_slots,
    ),
    "list_events": (
        "List or search the calendar's events, past or future. Use it to find an event the user mentions ('my flight "
        "on Friday', 'the Project Alpha kick-off', 'my 5 PM meeting'), to see a day's schedule, or to learn the usual "
        "length of a recurring meeting from past ones. Answers each event's id, title, time, length and attendees. "
        "Without a range it searches the last and next 90 days when given a query, otherwise the next 7 days.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words to search for in titles, descriptions, locations and attendees."},
                "start": {"type": "string", "description": "From this date (2026-10-20) or local date and time."},
                "end": {"type": "string", "description": "Up to this date (inclusive) or local date and time."},
                "max_results": {"type": "integer", "description": "1 to 50. Default 20."},
            },
        },
        False,
        list_events,
    ),
    "create_event": (
        "Book a meeting in the calendar. Call it as soon as the user has chosen a slot: the app shows the user the "
        "details and asks them to confirm, so do not ask for a separate confirmation first. It refuses, without "
        "booking, if the time is no longer free.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short event title, e.g. 'Sync with Priya'. Use 'Meeting' if the user gave none."},
                "start": {"type": "string", "description": "Local date and time, e.g. 2026-10-20T14:00."},
                "duration_minutes": {"type": "integer", "description": "Length in minutes."},
                "description": {"type": "string", "description": "Optional agenda or notes."},
                "location": {"type": "string", "description": "Optional place or meeting link."},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "Optional guest email addresses."},
                "allow_conflict": {"type": "boolean", "description": "Book even if it overlaps another event. Only when the user explicitly asks."},
            },
            "required": ["title", "start", "duration_minutes"],
        },
        True,
        create_event,
    ),
    "update_event": (
        "Move, resize or rename an existing event (find its id with list_events). Only the fields given change; a new "
        "start keeps the event's length unless duration_minutes is given. The user is asked to confirm.",
        {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "start": {"type": "string", "description": "New local date and time, e.g. 2026-10-21T10:00."},
                "duration_minutes": {"type": "integer", "description": "New length in minutes."},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "allow_conflict": {"type": "boolean", "description": "Only when the user explicitly wants to double-book."},
            },
            "required": ["event_id"],
        },
        True,
        update_event,
    ),
    "delete_event": (
        "Cancel (delete) an event by id (find it with list_events). The user is asked to confirm.",
        {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]},
        True,
        delete_event,
    ),
}


def describe_call(name: str, args: dict[str, Any], settings: CalendarSettings) -> str:
    """One line on what a call will do, for an approval prompt (shown, and spoken in a voice conversation)."""
    tz = settings.tz
    try:
        if name == "create_event":
            start = parse_when(args.get("start"), tz, field="start")
            end = utc(start) + timedelta(minutes=parse_int(args, "duration_minutes", lo=5, hi=720))
            guests = _emails(args.get("attendees"))
            with_guests = f" with {', '.join(guests)}" if guests else ""
            return f"Book “{args.get('title') or 'Meeting'}” on {span_label(start, end, tz)}{with_guests}."
        if name == "update_event":
            changes = []
            if args.get("start"):
                start = parse_when(args["start"], tz, field="start").astimezone(tz)
                changes.append(f"move it to {day_label(start.date())}, {clock(start)}")
            if args.get("duration_minutes"):
                changes.append(f"make it {parse_int(args, 'duration_minutes', lo=5, hi=720)} minutes long")
            if args.get("title"):
                changes.append(f"rename it “{args['title']}”")
            return f"Change the event: {', '.join(changes)}." if changes else "Change the event."
        if name == "delete_event":
            return "Cancel the event (it is deleted from the calendar)."
    except (CalendarError, KeyError, TypeError, ValueError):
        pass
    return f"Run {name}."


async def run_tool(name: str, args: dict[str, Any], cal: GoogleCalendar, *, now: datetime | None = None) -> str:
    """Runs one calendar tool and answers its JSON text. Raises CalendarError for anything the model should hear."""
    if name not in TOOLS:
        raise CalendarError(f"There is no calendar tool named {name!r}.")
    handler = TOOLS[name][3]
    result = await handler(cal, args or {}, now or datetime.now(timezone.utc))
    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------------------------- prompt context
def date_context(settings: CalendarSettings, now: datetime | None = None) -> str:
    """Facts about dates for the system prompt, worked out here so the model does not have to."""
    tz = settings.tz
    local = (now or datetime.now(timezone.utc)).astimezone(tz)
    today = local.date()
    offset = local.strftime("%z")
    monday = today - timedelta(days=today.weekday())
    month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    last_weekday = month_end - timedelta(days=max(0, month_end.weekday() - 4))
    work = f"{_days_text(settings.work_days)}, {clock(parse_clock(settings.work_start, 'work_start'))} – {clock(parse_clock(settings.work_end, 'work_end'))}"
    weeks = []
    for w in range(4):
        days = []
        for d in range(7):
            day = monday + timedelta(days=7 * w + d)
            mark = " (today)" if day == today else " (tomorrow)" if day == today + timedelta(days=1) else ""
            days.append(short_day(day) + mark)
        weeks.append("  " + " · ".join(days))
    return "\n".join(
        [
            "Calendar facts (worked out by the app; trust them over your own date arithmetic):",
            f"- Now: {day_label(today)}, {today.year}, {clock(local)}, time zone {settings.timezone} (UTC{offset[:3]}:{offset[3:]}).",
            f"- Working hours: {work}.",
            f"- This week (Monday to Sunday): {short_day(monday)} – {short_day(monday + timedelta(days=6))}. "
            f"Next week: {short_day(monday + timedelta(days=7))} – {short_day(monday + timedelta(days=13))}.",
            f"- This month ends on {day_label(month_end)}; its last weekday is {day_label(last_weekday)}.",
            "- The next four weeks:",
            *weeks,
            "- Give the calendar tools local times in this time zone, e.g. 2026-10-20T14:00. When a relative date "
            "could mean two days (\"next Friday\"), say the date you chose so the user can correct it.",
        ]
    )
