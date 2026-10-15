from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from sample_agent.config import ServerConfig
from sample_agent.hub import McpConnectionError, McpHub, _hint, connect, format_tool_result
from servers import notes_server, toolbox_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def both_servers(tmp_path: Path) -> AsyncIterator[McpHub]:
    """A hub over in-memory sessions to both servers (a context manager, not a fixture,
    because anyio cancel scopes must be exited by the task that entered them)."""
    toolbox = toolbox_server.create_server()
    notes = notes_server.create_server(notes_server.NotesStore(tmp_path / "notes.db"))
    async with (
        create_connected_server_and_client_session(toolbox._mcp_server) as toolbox_session,
        create_connected_server_and_client_session(notes._mcp_server) as notes_session,
    ):
        hub = McpHub()
        await hub.add_server("toolbox", toolbox_session)
        await hub.add_server("notes", notes_session)
        yield hub


async def test_hub_merges_all_seven_tools(tmp_path: Path) -> None:
    async with both_servers(tmp_path) as hub:
        assert hub.tool_names == [
            "calculate",
            "convert_units",
            "get_current_time",
            "add_note",
            "list_notes",
            "search_notes",
            "delete_note",
        ]


async def test_ollama_tools_use_function_schema(tmp_path: Path) -> None:
    async with both_servers(tmp_path) as hub:
        calculate = hub.ollama_tools()[0]
        assert calculate["type"] == "function"
        assert calculate["function"]["name"] == "calculate"
        assert "arithmetic" in calculate["function"]["description"]
        assert calculate["function"]["parameters"]["required"] == ["expression"]


async def test_call_tool_routes_to_owning_server(tmp_path: Path) -> None:
    async with both_servers(tmp_path) as hub:
        assert await hub.call_tool("calculate", {"expression": "6 * 7"}) == "6 * 7 = 42"
        added = await hub.call_tool("add_note", {"title": "hi", "body": "there"})
        assert '"title": "hi"' in added


async def test_call_tool_reports_tool_errors_as_text(tmp_path: Path) -> None:
    async with both_servers(tmp_path) as hub:
        result = await hub.call_tool("delete_note", {"note_id": 99})
        assert result == "Error executing tool delete_note: no note with id 99"


async def test_call_tool_reports_unknown_tool(tmp_path: Path) -> None:
    async with both_servers(tmp_path) as hub:
        result = await hub.call_tool("launch_rocket", {})
        assert result.startswith("Error: unknown tool 'launch_rocket'")
        assert "calculate" in result


async def test_duplicate_tool_names_are_rejected() -> None:
    async with (
        create_connected_server_and_client_session(toolbox_server.create_server()._mcp_server) as first,
        create_connected_server_and_client_session(toolbox_server.create_server()._mcp_server) as second,
    ):
        hub = McpHub()
        await hub.add_server("a", first)
        with pytest.raises(ValueError, match="clashes"):
            await hub.add_server("b", second)


def test_format_tool_result_handles_non_text_blocks() -> None:
    result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="chart:"),
            types.ImageContent(type="image", data="", mimeType="image/png"),
        ]
    )
    assert format_tool_result(result) == "chart:\n[image content omitted]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("boom", "Error: boom"), ("Error executing tool x: boom", "Error executing tool x: boom")],
)
def test_format_tool_result_marks_errors_once(text: str, expected: str) -> None:
    result = types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)
    assert format_tool_result(result) == expected


async def test_connect_names_the_unreachable_server() -> None:
    configs = [ServerConfig(name="toolbox", url="http://127.0.0.1:9/mcp")]
    with pytest.raises(McpConnectionError, match="'toolbox'"):
        async with connect(configs):
            pytest.fail("should not connect")


@pytest.mark.parametrize(
    ("status", "hint"),
    [(401, "rejected the credentials"), (403, "rejected the credentials"), (404, "check the path"), (500, "running")],
)
def test_hint_matches_the_failure(status: int, hint: str) -> None:
    request = httpx.Request("POST", "http://gw/mcp")
    error = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(status, request=request))
    assert hint in _hint(error)
    assert "running" in _hint(httpx.ConnectError("refused"))


async def test_connect_over_stdio_negotiates_2025_06_18() -> None:
    configs = [
        ServerConfig(
            name="toolbox",
            command="python",
            args=("servers/toolbox_server.py", "--transport", "stdio"),
            cwd=PROJECT_ROOT,
        )
    ]
    async with connect(configs) as hub:
        assert hub.protocol_versions == {"toolbox": "2025-06-18"}
        assert await hub.call_tool("convert_units", {"value": 0, "from_unit": "c", "to_unit": "f"}) == "0 c = 32 f"
