"""Built-in tools: the current time, a calculator and a URL reader. Only the ones this agent uses are loaded."""

from __future__ import annotations

import ast
import asyncio
import html
import ipaddress
import math
import operator
import re
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from langchain_core.tools import BaseTool, StructuredTool


async def current_time(timezone: str = "UTC") -> str:
    name = (timezone or "UTC").strip()
    try:
        tz = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone {name!r}; use an IANA name like Asia/Kolkata.") from None
    return datetime.now(UTC).astimezone(tz).strftime(f"%A, %d %B %Y, %H:%M:%S ({name}, UTC%z)")


_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_NAMES: dict[str, Any] = {
    "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round, "min": min, "max": max,
}


def _eval(node: ast.AST) -> Any:
    """Arithmetic only: no names, attributes or calls beyond the math functions above."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAMES:
        return _NAMES[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Only numbers, + - * / // % **, parentheses and math functions are allowed.")


async def calculator(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression or len(expression) > 500:
        raise ValueError("Give an expression of at most 500 characters.")
    return f"{expression} = {_eval(ast.parse(expression.replace('^', '**'), mode='eval'))}"


_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


async def _check_public(url: str) -> None:
    """Refuses private, loopback and link-local addresses, so the tool cannot be pointed at the server's network."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Give a full http(s) URL.")
    infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, None, type=socket.SOCK_STREAM)
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ValueError(f"{parts.hostname} is a private address; only public URLs can be read.")


async def fetch_url(url: str) -> str:
    url = (url or "").strip()
    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
        for _ in range(4):  # follow redirects by hand so every hop is checked
            await _check_public(url)
            resp = await client.get(url, headers={"User-Agent": "agent/1.0"})
            if resp.is_redirect and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                continue
            break
        else:
            raise ValueError("Too many redirects.")
    body = resp.text
    if "html" in resp.headers.get("content-type", ""):
        body = html.unescape(_TAG_RE.sub(" ", _SCRIPT_RE.sub(" ", body)))
    return f"HTTP {resp.status_code} {url}\n\n{_SPACE_RE.sub(' ', body).strip()}"


# name → (description, JSON schema, function). The function's parameter names match the schema's properties.
BUILTINS: dict[str, tuple[str, dict[str, Any], Callable[..., Any]]] = {
    "current_time": (
        "Get the current date and time, optionally in an IANA timezone such as Asia/Kolkata.",
        {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone; default UTC"}}},
        current_time,
    ),
    "calculator": (
        "Evaluate a math expression, e.g. (12.5 * 4) / 3 or sqrt(2) ** 3.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "The expression to evaluate"}}, "required": ["expression"]},
        calculator,
    ),
    "fetch_url": (
        "Read a public web page or API URL and return its text.",
        {"type": "object", "properties": {"url": {"type": "string", "description": "An http(s) URL"}}, "required": ["url"]},
        fetch_url,
    ),
}


def builtin_tools(names: dict[str, str]) -> list[BaseTool]:
    """``names``: the tool name the model sees → the built-in it runs."""
    tools = []
    for name, builtin in names.items():
        description, schema, run = BUILTINS[builtin]
        tools.append(StructuredTool.from_function(coroutine=run, name=name, description=description, args_schema=schema, metadata={"needs_approval": False}))
    return tools
