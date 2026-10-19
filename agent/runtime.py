"""Puts the agent together from its parts (model, tools, prompt, checkpointer). The server and the CLI both use it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from .config import ROOT, env
from .graph import build_graph
from .llm import chat_model
from .prompts import system_prompt
from .tools import describe, load_tools


@asynccontextmanager
async def checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """SQLite when CHECKPOINT_DB names a file (conversations survive a restart), otherwise memory."""
    path = env("CHECKPOINT_DB")
    if not path:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    file = Path(path) if Path(path).is_absolute() else ROOT / path
    file.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(file)) as saver:
        yield saver


@asynccontextmanager
async def open_agent() -> AsyncIterator[CompiledStateGraph]:
    async with checkpointer() as saver:
        tools = await load_tools()
        yield build_graph(chat_model(), tools, system_prompt, checkpointer=saver, describe=describe)
