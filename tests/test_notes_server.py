from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from servers.notes_server import NotesStore, _transport_security, create_server


@pytest.fixture
def store(tmp_path: Path) -> NotesStore:
    return NotesStore(tmp_path / "nested" / "notes.db")


def test_add_assigns_ids_and_normalises_tags(store: NotesStore) -> None:
    first = store.add("  Groceries ", "milk, eggs", ["Home", "todo", "home", " "])
    second = store.add("Standup", "demo the agent")
    assert (first.id, second.id) == (1, 2)
    assert first.title == "Groceries"
    assert first.tags == ["home", "todo"]
    assert second.tags == []


def test_add_rejects_blank_title(store: NotesStore) -> None:
    with pytest.raises(ValueError, match="title"):
        store.add("   ", "body")


def test_list_is_newest_first_and_filters_by_exact_tag(store: NotesStore) -> None:
    store.add("a", "", ["work"])
    store.add("b", "", ["homework"])
    store.add("c", "", ["work", "urgent"])
    assert [note.title for note in store.list()] == ["c", "b", "a"]
    assert [note.title for note in store.list(tag="WORK")] == ["c", "a"]
    assert [note.title for note in store.list(limit=1)] == ["c"]


def test_search_matches_title_or_body_case_insensitively(store: NotesStore) -> None:
    store.add("Recipe", "Add 100% butter")
    store.add("Meeting", "talk about the recipe book")
    store.add("Other", "nothing")
    assert [note.title for note in store.search("RECIPE")] == ["Meeting", "Recipe"]
    assert [note.title for note in store.search("100%")] == ["Recipe"]
    assert store.search("_") == []


def test_search_rejects_blank_query(store: NotesStore) -> None:
    with pytest.raises(ValueError, match="query"):
        store.search(" ")


def test_delete_reports_whether_note_existed(store: NotesStore) -> None:
    note = store.add("temp", "")
    assert store.delete(note.id) is True
    assert store.delete(note.id) is False
    assert store.list() == []


def test_notes_persist_across_store_instances(tmp_path: Path) -> None:
    NotesStore(tmp_path / "n.db").add("kept", "body")
    assert [note.title for note in NotesStore(tmp_path / "n.db").list()] == ["kept"]


async def test_server_exposes_four_tools(store: NotesStore) -> None:
    tools = await create_server(store).list_tools()
    assert sorted(tool.name for tool in tools) == ["add_note", "delete_note", "list_notes", "search_notes"]


async def test_server_round_trip(store: NotesStore) -> None:
    server = create_server(store)
    _, added = await server.call_tool("add_note", {"title": "Call Sam", "body": "re: gateway", "tags": ["work"]})
    assert added["id"] == 1
    _, listed = await server.call_tool("list_notes", {"tag": "work"})
    assert [note["title"] for note in listed["notes"]] == ["Call Sam"]
    _, found = await server.call_tool("search_notes", {"query": "gateway"})
    assert len(found["notes"]) == 1
    content, _ = await server.call_tool("delete_note", {"note_id": 1})
    assert content[0].text == "Deleted note 1."


async def test_server_delete_missing_note_is_a_tool_error(store: NotesStore) -> None:
    with pytest.raises(ToolError, match="no note with id 42"):
        await create_server(store).call_tool("delete_note", {"note_id": 42})


def test_transport_security_allows_extra_host_with_and_without_port() -> None:
    settings = _transport_security(8002, ["host.k3d.internal"])
    assert settings is not None
    assert {"host.k3d.internal", "host.k3d.internal:8002"} <= set(settings.allowed_hosts)
