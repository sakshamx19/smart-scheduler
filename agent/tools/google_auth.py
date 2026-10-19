"""Access tokens from a Google service account, for Google Calendar and any HTTP tool that calls Google's APIs.

The key comes from GOOGLE_SERVICE_ACCOUNT_JSON (the key file's contents, handy on hosting platforms) or
GOOGLE_SERVICE_ACCOUNT_FILE (a path to it). A service account has no calendar of its own: share the calendar with
the account's email address ("Make changes to events") in Google Calendar's settings.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2 import service_account

from ..config import ROOT, env


def _key_info() -> dict:
    raw = env("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw and (path := env("GOOGLE_SERVICE_ACCOUNT_FILE")):
        file = Path(path) if Path(path).is_absolute() else ROOT / path
        raw = file.read_text(encoding="utf-8")
    if not raw:
        raise RuntimeError("Set GOOGLE_SERVICE_ACCOUNT_JSON (or GOOGLE_SERVICE_ACCOUNT_FILE) to the service account's key.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON; paste the whole key file on one line.") from None


class ServiceAccount:
    """``await account.token()`` answers a valid access token, refreshing it when it is about to expire."""

    def __init__(self, scopes: str) -> None:
        self._credentials = service_account.Credentials.from_service_account_info(_key_info(), scopes=scopes.split())
        self._lock = asyncio.Lock()

    @property
    def email(self) -> str:
        return self._credentials.service_account_email

    async def token(self) -> str:
        async with self._lock:
            if not self._credentials.valid:  # google-auth refreshes a few minutes before expiry
                await asyncio.to_thread(self._credentials.refresh, Request())
            return self._credentials.token
