from pathlib import Path

import pytest
from ollama import AsyncClient, ResponseError

from sample_agent_2.__main__ import DEFAULT_MODEL, DEFAULT_PORT, answer, open_llm, parse_args, print_tool_call, run
from sample_agent_2.agent import AgentStepLimitError
from sample_agent_2.openai_compat import OpenAIChatClient


class FailingAgent:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def ask(self, prompt: str) -> str:
        raise self._error


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    args = parse_args([])
    assert args.model == DEFAULT_MODEL
    assert args.config.name == "mcp_servers.json"
    assert args.prompt is None
    assert (args.serve, args.host, args.port) == (False, "127.0.0.1", DEFAULT_PORT)


def test_parse_args_reads_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert parse_args([]).model == "llama3.1:8b"
    assert parse_args(["-m", "qwen3:4b"]).model == "qwen3:4b"


def test_print_tool_call_truncates_long_results(capsys: pytest.CaptureFixture[str]) -> None:
    print_tool_call("search_notes", {"query": "x"}, "y" * 500)
    line = capsys.readouterr().out.strip()
    assert line.startswith('↳ search_notes({"query": "x"}) → yyy')
    assert line.endswith("...")
    assert len(line) < 220


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ResponseError("model 'nope' not found", 404), "ollama pull nope"),
        (ConnectionError("refused"), "is Ollama / the gateway running"),
        (AgentStepLimitError("no final answer"), "no final answer"),
    ],
)
async def test_answer_reports_failures(error: Exception, message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert await answer(FailingAgent(error), "hi", "nope") is False  # type: ignore[arg-type]
    assert message in capsys.readouterr().err


async def test_run_exits_2_on_bad_config(tmp_path: Path) -> None:
    assert await run(parse_args(["-c", str(tmp_path / "missing.json"), "-p", "hi"])) == 2


async def test_run_exits_1_when_servers_are_down(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "servers.json"
    config.write_text('{"mcpServers": {"toolbox": {"url": "http://127.0.0.1:9/mcp"}}}', encoding="utf-8")
    assert await run(parse_args(["-c", str(config), "-p", "hi"])) == 1
    assert "cannot connect to MCP server 'toolbox'" in capsys.readouterr().err


async def test_open_llm_picks_client_from_flags() -> None:
    async with open_llm(parse_args([])) as llm:
        assert isinstance(llm, AsyncClient)
    async with open_llm(parse_args(["--llm-base-url", "http://gw/ollama"])) as llm:
        assert isinstance(llm, OpenAIChatClient)
