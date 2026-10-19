"""The agent as a LangGraph state machine.

    START ─► agent ─► asked for tools? ─ no ─► END
               ▲            │ yes
               │            ▼
             tools ◄──── review

- **agent** calls the model with the system prompt, the whole conversation and the tools.
- **review** pauses the run with LangGraph's ``interrupt()`` when a call changes something (booking a meeting,
  say). The server shows the user what is about to happen and resumes the run with their answer. Nothing waits in
  memory: the checkpointer holds the paused state, so an approval survives a restart.
- **tools** runs the calls side by side. A declined call answers the model "the user declined", so it can ask what
  to do instead; a failing tool answers the error, so the model can recover rather than the run crashing.

The checkpointer keeps every conversation by thread id. That is the agent's memory: the meeting length said three
turns ago is still in the messages the model sees.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

TOOL_TIMEOUT_S = 45.0
MAX_RESULT_CHARS = 16_000
DECLINED = "The user declined this action. Do not retry it; ask the user how they want to proceed."
DECLINED_SAYING = "The user declined this action and said instead: “{note}”. Do what they asked now; do not retry the declined action unchanged."


class State(MessagesState):
    decisions: dict[str, bool]  # tool call id → approved, for the calls review asked about this round
    note: str  # what the user said instead of answering an approval ("actually, make it 3 PM")


def needs_approval(tool: BaseTool | None) -> bool:
    return bool(tool is not None and (tool.metadata or {}).get("needs_approval"))


def text_of(content: Any) -> str:
    """A tool result or message content as plain text (some providers and MCP tools answer content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(text_of(part.get("text", "") if isinstance(part, dict) else part) for part in content)
    return str(content)


def _approved(answer: Any, calls: list[dict[str, Any]]) -> set[str]:
    """The resume value is {"approve": true|false} for every call, or {"approved": [call ids]}; either may carry a
    "note", the user's words when they answered with something other than yes or no."""
    if isinstance(answer, dict) and isinstance(answer.get("approved"), list):
        return {str(i) for i in answer["approved"]}
    if isinstance(answer, dict) and answer.get("approve") is True:
        return {c["id"] for c in calls}
    return set()


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: Callable[[bool], str],
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    describe: Callable[[str, dict[str, Any]], str] | None = None,
) -> CompiledStateGraph:
    """``system_prompt(voice)`` is asked for on every turn, so facts in it (today's date) stay current.
    ``describe(name, args)`` writes the one-line summary of a call that needs approval."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools) if tools else model

    async def agent(state: State, config: RunnableConfig) -> dict[str, Any]:
        voice = bool(config.get("configurable", {}).get("voice"))
        messages = [SystemMessage(system_prompt(voice)), *state["messages"]]
        return {"messages": [await bound.ainvoke(messages, config)]}

    def route(state: State) -> Literal["review", "__end__"]:
        last = state["messages"][-1]
        return "review" if isinstance(last, AIMessage) and last.tool_calls else END

    def review(state: State) -> dict[str, Any]:
        calls = [c for c in state["messages"][-1].tool_calls if needs_approval(by_name.get(c["name"]))]
        if not calls:
            return {"decisions": {}, "note": ""}
        # Pauses here. On resume, LangGraph runs this node again and interrupt() returns the user's answer.
        answer = interrupt(
            {
                "calls": [
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "args": c["args"],
                        "summary": describe(c["name"], c["args"]) if describe else f"Run {c['name']}.",
                    }
                    for c in calls
                ]
            }
        )
        approved = _approved(answer, calls)
        note = str(answer.get("note") or "").strip() if isinstance(answer, dict) else ""
        return {"decisions": {c["id"]: c["id"] in approved for c in calls}, "note": note}

    async def run_call(call: dict[str, Any], decisions: dict[str, bool], note: str) -> ToolMessage:
        name, call_id = call["name"], call["id"]
        tool = by_name.get(name)
        if decisions.get(call_id) is False:
            if note:  # the note is kept on the message, so a reloaded conversation shows what the user said
                return ToolMessage(DECLINED_SAYING.format(note=note), tool_call_id=call_id, name=name, status="error", additional_kwargs={"user_note": note})
            return ToolMessage(DECLINED, tool_call_id=call_id, name=name, status="error")
        if tool is None:
            return ToolMessage(f"There is no tool named {name!r}.", tool_call_id=call_id, name=name, status="error")
        try:
            result = await asyncio.wait_for(tool.ainvoke(call["args"]), TOOL_TIMEOUT_S)
        except TimeoutError:
            return ToolMessage(f"The tool did not finish within {TOOL_TIMEOUT_S:.0f} seconds.", tool_call_id=call_id, name=name, status="error")
        except Exception as exc:  # a failing tool must not end the conversation
            return ToolMessage(f"Error: {exc}", tool_call_id=call_id, name=name, status="error")
        text = text_of(result)
        if len(text) > MAX_RESULT_CHARS:
            text = f"{text[:MAX_RESULT_CHARS]}\n…(truncated; {len(text)} characters in all)"
        return ToolMessage(text or "(no output)", tool_call_id=call_id, name=name)

    async def run_tools(state: State) -> dict[str, Any]:
        decisions, note = state.get("decisions") or {}, state.get("note") or ""
        calls = state["messages"][-1].tool_calls
        results = await asyncio.gather(*(run_call(c, decisions, note) for c in calls))
        return {"messages": list(results), "decisions": {}, "note": ""}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("review", review)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["review", END])
    graph.add_edge("review", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
