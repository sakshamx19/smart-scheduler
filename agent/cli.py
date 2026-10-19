"""Chat with the agent in a terminal: ``python -m agent.cli``. Replies stream as they are written, and actions that
need approval are asked about here (y/n)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from .prompts import NAME
from .runtime import open_agent


async def _run(graph: Any, payload: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Streams one stretch of the run to the terminal; answers the approvals it stopped for."""
    pending: list[dict[str, Any]] = []
    async for mode, chunk in graph.astream(payload, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            message, meta = chunk
            if meta.get("langgraph_node") == "agent" and isinstance(message, AIMessageChunk) and message.text:
                print(message.text, end="", flush=True)
        elif "__interrupt__" in chunk:
            pending += [i.value for i in chunk["__interrupt__"]]
        elif "agent" in chunk:
            for call in chunk["agent"]["messages"][-1].tool_calls:
                print(f"\n  · {call['name']} {call['args']}", flush=True)
    print()
    return pending


async def main() -> None:
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 40}
    async with open_agent() as graph:
        print(f"{NAME} — type a message, or Ctrl+C to quit.\n")
        while True:
            try:
                text = input("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not text:
                continue
            print(f"{NAME.lower()} › ", end="", flush=True)
            pending = await _run(graph, {"messages": [HumanMessage(text)]}, config)
            while pending:
                for call in pending[0]["calls"]:
                    print(f"  ! {call['summary']}")
                answer = input("  approve? [y/n] › ").strip().lower()
                pending = await _run(graph, Command(resume={"approve": answer.startswith("y")}), config)


if __name__ == "__main__":
    asyncio.run(main())
