"""The agent loop: ask the Ollama model, run any tools it requests over MCP, repeat."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ollama import ChatResponse, Message

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running fully on the user's machine. "
    "Use the provided tools whenever they help: the toolbox tools for arithmetic, unit "
    "conversion and the current time, and the notes tools to save, find and delete the "
    "user's notes. Never invent a tool result; if a tool returns an error, fix the input "
    "and try again or explain the problem. Keep answers short."
)
DEFAULT_MAX_STEPS = 8

ToolCallObserver = Callable[[str, Mapping[str, Any], str], None]


class ChatClient(Protocol):
    """The slice of `ollama.AsyncClient` the agent uses."""

    async def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send the conversation and return the model's next message."""
        ...


class ToolProvider(Protocol):
    """Anything that can describe tools to the model and execute them (see `McpHub`)."""

    def ollama_tools(self) -> list[dict[str, Any]]:
        """Tool schemas in Ollama's function-calling format."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return its result as text."""
        ...


class AgentStepLimitError(RuntimeError):
    """The model kept requesting tools without producing a final answer."""


class Agent:
    """A multi-turn chat agent that keeps its conversation history between calls to `ask`."""

    def __init__(
        self,
        llm: ChatClient,
        tools: ToolProvider,
        model: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_tool_call: ToolCallObserver | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._model = model
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._on_tool_call = on_tool_call
        self._messages: list[Mapping[str, Any] | Message] = []
        self.reset()

    def reset(self) -> None:
        """Forget the conversation, keeping only the system prompt."""
        self._messages = [{"role": "system", "content": self._system_prompt}]

    async def ask(self, prompt: str) -> str:
        """Answer `prompt`, calling tools as many times as the model needs (up to `max_steps`).

        Raises:
            AgentStepLimitError: if the model is still calling tools after `max_steps` rounds.
        """
        self._messages.append({"role": "user", "content": prompt})
        tool_schemas = self._tools.ollama_tools()
        for _ in range(self._max_steps):
            response = await self._llm.chat(self._model, self._messages, tools=tool_schemas)
            message = response.message
            self._messages.append(message)
            if not message.tool_calls:
                return message.content or ""
            for call in message.tool_calls:
                await self._run_tool(call.function.name, dict(call.function.arguments))
        raise AgentStepLimitError(f"no final answer after {self._max_steps} tool-calling rounds")

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> None:
        result = await self._tools.call_tool(name, arguments)
        if self._on_tool_call is not None:
            self._on_tool_call(name, arguments, result)
        self._messages.append({"role": "tool", "content": result, "tool_name": name})
