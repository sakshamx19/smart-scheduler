"""The graph's control flow, with a scripted model and fake tools: no API key and no network.

    python -m pytest -q
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import DECLINED, build_graph

CONFIG = {"configurable": {"thread_id": "test-thread"}}


class ScriptedModel(BaseChatModel):
    """Answers with the next message of its script, and remembers what it was shown."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


ran: list[tuple[str, dict[str, Any]]] = []


def fake_tool(name: str, needs_approval: bool, fails: bool = False) -> StructuredTool:
    async def run(**args: Any) -> str:
        if fails:
            raise RuntimeError("the calendar is unreachable")
        ran.append((name, args))
        return f"{name} done"

    schema = {"type": "object", "properties": {"when": {"type": "string"}}}
    return StructuredTool.from_function(coroutine=run, name=name, description=name, args_schema=schema, metadata={"needs_approval": needs_approval})


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}


def make(script: list[AIMessage]) -> tuple[ScriptedModel, Any]:
    ran.clear()
    model = ScriptedModel(script=script)
    tools = [fake_tool("find", False), fake_tool("book", True), fake_tool("broken", False, fails=True)]
    return model, build_graph(model, tools, lambda voice: "You are a test agent.", checkpointer=InMemorySaver())


def say(graph: Any, text: str) -> dict[str, Any]:
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage(text)]}, CONFIG))


def answer(graph: Any, value: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(graph.ainvoke(Command(resume=value), CONFIG))


def interrupts(graph: Any) -> tuple[Any, ...]:
    return asyncio.run(graph.aget_state(CONFIG)).interrupts


def test_read_tools_run_without_asking() -> None:
    _, graph = make([AIMessage("", tool_calls=[call("find", when="tuesday")]), AIMessage("Tuesday at 2 PM is free.")])
    state = say(graph, "Is Tuesday free?")
    assert ran == [("find", {"when": "tuesday"})]
    assert not interrupts(graph)
    assert state["messages"][-1].content == "Tuesday at 2 PM is free."


def test_changes_wait_for_approval_then_run() -> None:
    _, graph = make([AIMessage("", tool_calls=[call("book", when="tue 2pm")]), AIMessage("Booked for Tuesday at 2 PM.")])
    say(graph, "Book Tuesday at 2.")
    pending = interrupts(graph)
    assert ran == [] and pending, "booking must wait for the user"
    assert pending[0].value["calls"][0]["name"] == "book"
    state = answer(graph, {"approve": True})
    assert ran == [("book", {"when": "tue 2pm"})]
    assert state["messages"][-1].content == "Booked for Tuesday at 2 PM."


def test_a_declined_change_is_not_run() -> None:
    _, graph = make([AIMessage("", tool_calls=[call("book", when="tue 2pm")]), AIMessage("Okay, I won't book it.")])
    say(graph, "Book Tuesday at 2.")
    state = answer(graph, {"approve": False})
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert ran == [] and result.content == DECLINED and result.status == "error"


def test_words_instead_of_an_answer_reach_the_model() -> None:
    model, graph = make([AIMessage("", tool_calls=[call("book", when="tue 2pm")]), AIMessage("Sure, 3 PM then.")])
    say(graph, "Book Tuesday at 2.")
    answer(graph, {"approve": False, "note": "actually make it 3 PM"})
    shown = model.seen[-1][-1]
    assert isinstance(shown, ToolMessage) and "actually make it 3 PM" in shown.content
    assert shown.additional_kwargs["user_note"] == "actually make it 3 PM"


def test_a_failing_tool_answers_the_error_and_the_run_goes_on() -> None:
    model, graph = make([AIMessage("", tool_calls=[call("broken")]), AIMessage("The calendar is down; try again soon.")])
    state = say(graph, "What's free?")
    result = model.seen[-1][-1]
    assert isinstance(result, ToolMessage) and result.status == "error" and "unreachable" in result.content
    assert state["messages"][-1].content.startswith("The calendar is down")


def test_earlier_turns_are_remembered() -> None:
    model, graph = make([AIMessage("How long should it be?"), AIMessage("Got it, one hour.")])
    say(graph, "I need to schedule a meeting.")
    say(graph, "1 hour.")
    texts = [m.content for m in model.seen[-1]]
    assert "I need to schedule a meeting." in texts and "How long should it be?" in texts
