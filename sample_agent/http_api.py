"""HTTP API for the agent, so several agent instances can run side by side on different ports.

    GET    /health               model and tool names
    POST   /chat                 {"message": "...", "session_id": "optional"} -> reply + tool calls
    DELETE /chat/{session_id}    forget a conversation

Each session keeps its own history; requests within one session are serialized.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any

from ollama import ResponseError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sample_agent.agent import Agent, AgentStepLimitError, ToolCallObserver

DEFAULT_MAX_SESSIONS = 100


@dataclass(frozen=True)
class Runtime:
    """What the API needs from the running process: a way to build agents, plus info for /health."""

    model: str
    tool_names: list[str]
    new_agent: Callable[[ToolCallObserver], Agent]


@dataclass
class Session:
    """One conversation: its agent, a lock serializing its turns, and the current turn's tool calls."""

    agent: Agent
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class SessionStore:
    """Keeps at most `max_sessions` conversations, evicting the least recently used."""

    def __init__(self, new_agent: Callable[[ToolCallObserver], Agent], max_sessions: int = DEFAULT_MAX_SESSIONS):
        self._new_agent = new_agent
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get_or_create(self, session_id: str) -> Session:
        """Return the session for `session_id`, creating it if needed."""
        session = self._sessions.get(session_id)
        if session is None:
            session = self._create()
            self._sessions[session_id] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(session_id)
        return session

    def drop(self, session_id: str) -> bool:
        """Forget a session; return False if it did not exist."""
        return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)

    def _create(self) -> Session:
        calls: list[dict[str, Any]] = []

        def record(name: str, arguments: Mapping[str, Any], result: str) -> None:
            calls.append({"name": name, "arguments": dict(arguments), "result": result})

        return Session(agent=self._new_agent(record), tool_calls=calls)


def create_app(runtime: Runtime, max_sessions: int = DEFAULT_MAX_SESSIONS) -> Starlette:
    """Build the Starlette app serving one agent runtime."""
    sessions = SessionStore(runtime.new_agent, max_sessions)

    async def health(_: Request) -> Response:
        return JSONResponse(
            {"status": "ok", "model": runtime.model, "tools": runtime.tool_names, "sessions": len(sessions)}
        )

    async def chat(request: Request) -> Response:
        try:
            body = await request.json()
        except JSONDecodeError:
            return _error(400, "request body must be JSON")
        if not isinstance(body, dict):
            return _error(400, "request body must be a JSON object")
        message, session_id = body.get("message"), body.get("session_id") or uuid.uuid4().hex
        if not isinstance(message, str) or not message.strip():
            return _error(400, "'message' must be a non-empty string")
        if not isinstance(session_id, str):
            return _error(400, "'session_id' must be a string")

        session = sessions.get_or_create(session_id)
        async with session.lock:
            session.tool_calls.clear()
            try:
                reply = await session.agent.ask(message)
            except ResponseError as exc:
                return _error(502, f"model endpoint rejected the request ({exc.status_code}): {exc.error}")
            except ConnectionError as exc:
                return _error(502, f"cannot reach the model endpoint: {exc}")
            except AgentStepLimitError as exc:
                return _error(500, str(exc))
            return JSONResponse({"session_id": session_id, "reply": reply, "tool_calls": list(session.tool_calls)})

    async def reset(request: Request) -> Response:
        if not sessions.drop(request.path_params["session_id"]):
            return _error(404, "unknown session")
        return Response(status_code=204)

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/{session_id}", reset, methods=["DELETE"]),
        ]
    )


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)
