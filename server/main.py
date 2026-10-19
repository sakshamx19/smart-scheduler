"""The web server: the chat API (server-sent events) and, when exported with one, the chat portal in web/.

    GET  /api/config                               the agent's name and voice settings, for the portal
    POST /api/chat    {thread_id, message, voice}  one turn, streamed as server-sent events
    POST /api/resume  {thread_id, approve, voice}  the answer to an approval, streamed the same way
    GET  /api/threads/{thread_id}                  the conversation so far, and an approval still waiting

Events: ``token`` {text} · ``tool_start`` {id, name, args} · ``tool_end`` {id, name, ok, preview} ·
``approval`` {calls: [{id, name, args, summary}]} · ``done`` {ms} · ``error`` {message}.

A run executes inside the request that streams it. A pause for approval ends the stream; the paused state is in
the checkpointer, and /api/resume picks the run up from there. A new message while an approval waits counts as
declining it, with the message passed on to the model ("actually, make it 3 PM").

Set ACCESS_CODE to keep strangers off a public deployment: every /api call then needs it (X-Access-Code header).
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import importlib.util
import json
import logging
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import ROOT, env
from agent.graph import text_of
from agent.prompts import DESCRIPTION, NAME, STARTERS
from agent.runtime import open_agent

# server/voice.py is there when the project was exported with server-side speech
voice = importlib.import_module(".voice", __package__) if importlib.util.find_spec(".voice", __package__) else None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

WEB = ROOT / "web"
THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
OPEN_PATHS = ("/api/health", "/api/config")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with open_agent() as graph:
        app.state.graph = graph
        log.info("%s is ready", NAME)
        yield


app = FastAPI(title=NAME, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
_busy: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # one run at a time per conversation


@app.middleware("http")
async def isolation(request: Request, call_next: Any) -> Any:
    """Cross-origin isolation lets the on-device voice models (web/*.worker.js) run WebAssembly on several threads,
    several times faster. "credentialless" still lets the page load the models and libraries from their CDNs."""
    response = await call_next(request)
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
    return response


@app.middleware("http")
async def access_code(request: Request, call_next: Any) -> Any:
    code = env("ACCESS_CODE")
    path = request.url.path
    if code and path.startswith("/api/") and path not in OPEN_PATHS:
        given = request.headers.get("x-access-code") or request.query_params.get("code") or ""
        if not hmac.compare_digest(given.encode(), code.encode()):
            return JSONResponse({"detail": "Enter the access code."}, status_code=401)
    return await call_next(request)


class ChatIn(BaseModel):
    thread_id: str
    message: str = Field(min_length=1, max_length=8000)
    voice: bool = False  # the user spoke: the reply will be read aloud, so the agent keeps it short


class ResumeIn(BaseModel):
    thread_id: str
    approve: bool
    voice: bool = False


def _run_config(thread_id: str, spoken: bool) -> dict[str, Any]:
    if not THREAD_RE.match(thread_id):
        raise HTTPException(422, "thread_id must be 8 to 64 letters, digits, - or _.")
    return {"configurable": {"thread_id": thread_id, "voice": spoken}, "recursion_limit": 40}


def _error_text(exc: Exception) -> str:
    name = exc.__class__.__name__
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "The model provider rejected the API key; check .env."
    if name == "RateLimitError":
        return "The model provider is rate limiting; try again in a moment."
    if isinstance(exc, RuntimeError):  # configuration problems, written for people
        return str(exc)
    return f"Something went wrong ({name}); see the server log."


async def _events(payload: Any, config: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    started = time.perf_counter()
    graph = app.state.graph
    try:
        async for mode, chunk in graph.astream(payload, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                message, meta = chunk
                if meta.get("langgraph_node") == "agent" and isinstance(message, AIMessageChunk) and message.text:
                    yield {"type": "token", "text": message.text}
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    for item in update:
                        yield {"type": "approval", **item.value}
                elif node == "agent" and update:
                    for call in update["messages"][-1].tool_calls:
                        yield {"type": "tool_start", "id": call["id"], "name": call["name"], "args": call["args"]}
                elif node == "tools" and update:
                    for m in update["messages"]:
                        yield {"type": "tool_end", "id": m.tool_call_id, "name": m.name, "ok": m.status != "error", "preview": text_of(m.content)[:600]}
    except Exception as exc:
        log.exception("run failed")
        yield {"type": "error", "message": _error_text(exc)}
        return
    yield {"type": "done", "ms": round((time.perf_counter() - started) * 1000)}


def _stream(thread_id: str, payload: Any, config: dict[str, Any]) -> StreamingResponse:
    lock = _busy[thread_id]
    if lock.locked():
        raise HTTPException(409, "This conversation is still answering; wait for the reply.")

    async def body() -> AsyncIterator[str]:
        async with lock:
            async for event in _events(payload, config):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "name": NAME,
        "description": DESCRIPTION,
        "starters": STARTERS,
        "locked": bool(env("ACCESS_CODE")),
        "voice": voice.public_config() if voice else None,
    }


@app.post("/api/chat")
async def chat(body: ChatIn) -> StreamingResponse:
    config = _run_config(body.thread_id, body.voice)
    state = await app.state.graph.aget_state(config)
    if state.interrupts:  # answering an approval with words: decline it and pass the words on
        payload: Any = Command(resume={"approve": False, "note": body.message.strip()})
    else:
        payload = {"messages": [HumanMessage(body.message.strip())]}
    return _stream(body.thread_id, payload, config)


@app.post("/api/resume")
async def resume(body: ResumeIn) -> StreamingResponse:
    config = _run_config(body.thread_id, body.voice)
    state = await app.state.graph.aget_state(config)
    if not state.interrupts:
        raise HTTPException(409, "Nothing is waiting for approval in this conversation.")
    return _stream(body.thread_id, Command(resume={"approve": body.approve}), config)


@app.get("/api/threads/{thread_id}")
async def history(thread_id: str) -> dict[str, Any]:
    """The conversation as the portal draws it: user and assistant text, and each tool call with its result."""
    state = await app.state.graph.aget_state(_run_config(thread_id, False))
    items: list[dict[str, Any]] = []
    calls: dict[str, dict[str, Any]] = {}
    for m in (state.values or {}).get("messages", []):
        if isinstance(m, HumanMessage):
            items.append({"role": "user", "text": text_of(m.content)})
        elif isinstance(m, AIMessage):
            if m.text:
                items.append({"role": "assistant", "text": m.text})
            for call in m.tool_calls:
                calls[call["id"]] = {"role": "tool", "id": call["id"], "name": call["name"], "args": call["args"]}
                items.append(calls[call["id"]])
        elif isinstance(m, ToolMessage):
            if m.tool_call_id in calls:
                calls[m.tool_call_id].update(ok=m.status != "error", preview=text_of(m.content)[:600])
            if note := m.additional_kwargs.get("user_note"):
                items.append({"role": "user", "text": note})
    approval = state.interrupts[0].value if state.interrupts else None
    return {"messages": items, "approval": approval}


if voice:
    app.include_router(voice.router, prefix="/api")

if WEB.is_dir():  # last, so it never shadows the API
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
