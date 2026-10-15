from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from ollama import ChatResponse, Message

from sample_agent_2.agent import Agent, AgentStepLimitError


def tool_call(name: str, **arguments: Any) -> Message.ToolCall:
    return Message.ToolCall(function=Message.ToolCall.Function(name=name, arguments=arguments))


class ScriptedLlm:
    """Returns pre-written assistant messages in order and records what it was sent."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.requests: list[list[Mapping[str, Any] | Message]] = []

    async def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ChatResponse:
        assert model == "test-model"
        assert tools == FakeTools.SCHEMAS
        self.requests.append(list(messages or []))
        return ChatResponse(model=model, message=self._replies.pop(0))


class FakeTools:
    SCHEMAS: ClassVar[list[dict[str, Any]]] = [{"type": "function", "function": {"name": "calculate"}}]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def ollama_tools(self) -> list[dict[str, Any]]:
        return self.SCHEMAS

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return f"{name} ok"


async def test_answers_directly_when_no_tool_is_needed() -> None:
    llm = ScriptedLlm(Message(role="assistant", content="Hello!"))
    agent = Agent(llm, FakeTools(), "test-model", system_prompt="sys")
    assert await agent.ask("hi") == "Hello!"
    assert llm.requests[0] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


async def test_runs_requested_tools_and_feeds_results_back() -> None:
    llm = ScriptedLlm(
        Message(role="assistant", tool_calls=[tool_call("calculate", expression="1+1"), tool_call("list_notes")]),
        Message(role="assistant", content="1+1 is 2 and you have notes."),
    )
    tools = FakeTools()
    seen: list[tuple[str, Mapping[str, Any], str]] = []
    agent = Agent(llm, tools, "test-model", on_tool_call=lambda *call: seen.append(call))

    assert await agent.ask("sum and notes?") == "1+1 is 2 and you have notes."
    assert tools.calls == [("calculate", {"expression": "1+1"}), ("list_notes", {})]
    assert [name for name, _, _ in seen] == ["calculate", "list_notes"]
    assert llm.requests[1][-2:] == [
        {"role": "tool", "content": "calculate ok", "tool_name": "calculate"},
        {"role": "tool", "content": "list_notes ok", "tool_name": "list_notes"},
    ]


async def test_keeps_history_until_reset() -> None:
    llm = ScriptedLlm(*(Message(role="assistant", content=text) for text in ("a", "b", "c")))
    agent = Agent(llm, FakeTools(), "test-model")
    await agent.ask("one")
    await agent.ask("two")
    assert len(llm.requests[1]) == 4  # system, user, assistant, user
    agent.reset()
    await agent.ask("three")
    assert len(llm.requests[2]) == 2


async def test_gives_up_after_max_steps() -> None:
    looping = Message(role="assistant", tool_calls=[tool_call("calculate", expression="1")])
    agent = Agent(ScriptedLlm(looping, looping), FakeTools(), "test-model", max_steps=2)
    with pytest.raises(AgentStepLimitError, match="2 tool-calling rounds"):
        await agent.ask("loop forever")
