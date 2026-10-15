from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from ollama import ChatResponse, Message, ResponseError
from starlette.testclient import TestClient

from sample_agent_2.agent import Agent, AgentStepLimitError, ToolCallObserver
from sample_agent_2.http_api import Runtime, SessionStore, create_app


class EchoLlm:
    """Calls `calculate` when asked to, otherwise echoes how many user turns it has seen."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ChatResponse:
        if self._error:
            raise self._error
        history = list(messages or [])
        last = history[-1]
        if isinstance(last, dict) and last.get("role") == "tool":
            return ChatResponse(message=Message(role="assistant", content=f"result: {last['content']}"))
        if isinstance(last, dict) and "calc" in last["content"]:
            call = Message.ToolCall(
                function=Message.ToolCall.Function(name="calculate", arguments={"expression": "6*7"})
            )
            return ChatResponse(message=Message(role="assistant", tool_calls=[call]))
        turns = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")
        return ChatResponse(message=Message(role="assistant", content=f"turn {turns}"))


class FakeTools:
    def ollama_tools(self) -> list[dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "42"


def make_client(llm: EchoLlm | None = None, max_sessions: int = 100) -> TestClient:
    def new_agent(observer: ToolCallObserver) -> Agent:
        return Agent(llm or EchoLlm(), FakeTools(), "m", on_tool_call=observer)

    runtime = Runtime(model="m", tool_names=["calculate"], new_agent=new_agent)
    return TestClient(create_app(runtime, max_sessions=max_sessions))


def test_health_reports_model_and_tools() -> None:
    assert make_client().get("/health").json() == {"status": "ok", "model": "m", "tools": ["calculate"], "sessions": 0}


def test_chat_returns_reply_and_tool_calls() -> None:
    body = make_client().post("/chat", json={"message": "calc please", "session_id": "s1"}).json()
    assert body == {
        "session_id": "s1",
        "reply": "result: 42",
        "tool_calls": [{"name": "calculate", "arguments": {"expression": "6*7"}, "result": "42"}],
    }


def test_sessions_keep_separate_histories() -> None:
    client = make_client()
    assert client.post("/chat", json={"message": "hi", "session_id": "a"}).json()["reply"] == "turn 1"
    assert client.post("/chat", json={"message": "again", "session_id": "a"}).json()["reply"] == "turn 2"
    assert client.post("/chat", json={"message": "hi", "session_id": "b"}).json()["reply"] == "turn 1"


def test_missing_session_id_gets_a_fresh_one() -> None:
    client = make_client()
    first = client.post("/chat", json={"message": "hi"}).json()["session_id"]
    second = client.post("/chat", json={"message": "hi"}).json()["session_id"]
    assert first and second and first != second


def test_delete_resets_a_session() -> None:
    client = make_client()
    client.post("/chat", json={"message": "hi", "session_id": "a"})
    assert client.delete("/chat/a").status_code == 204
    assert client.delete("/chat/a").status_code == 404
    assert client.post("/chat", json={"message": "hi", "session_id": "a"}).json()["reply"] == "turn 1"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("not json", "must be JSON"),
        ("[1]", "JSON object"),
        ('{"message": ""}', "non-empty string"),
        ('{"message": "hi", "session_id": 5}', "'session_id' must be a string"),
    ],
)
def test_chat_rejects_bad_requests(body: str, error: str) -> None:
    response = make_client().post("/chat", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert error in response.json()["error"]


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ResponseError("Valid API key required", 401), 502),
        (ConnectionError("refused"), 502),
        (AgentStepLimitError("no final answer"), 500),
    ],
)
def test_chat_maps_agent_failures(exc: Exception, status: int) -> None:
    response = make_client(EchoLlm(error=exc)).post("/chat", json={"message": "hi"})
    assert response.status_code == status
    assert response.json()["error"]


def test_session_store_evicts_least_recently_used() -> None:
    store = SessionStore(lambda observer: Agent(EchoLlm(), FakeTools(), "m", on_tool_call=observer), max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("a")
    store.get_or_create("c")
    assert len(store) == 2
    assert store.drop("b") is False
    assert store.drop("a") is True
