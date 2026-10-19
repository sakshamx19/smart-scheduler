"""Settings, read from the environment and from .env in the project folder (see .env.example for every one)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def require(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def env_float(name: str, default: float | None = None) -> float | None:
    value = env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, not {value!r}.") from None
