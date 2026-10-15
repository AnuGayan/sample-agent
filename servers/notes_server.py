"""Notes MCP server: a small persistent notebook backed by SQLite.

Runs standalone over Streamable HTTP (the transport WSO2 AI gateways proxy) or stdio:

    python servers/notes_server.py                        # http://127.0.0.1:8002/mcp
    python servers/notes_server.py --transport stdio
    NOTES_DB_PATH=/tmp/notes.db python servers/notes_server.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

SERVER_NAME = "notes"
DEFAULT_PORT = 8002
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "notes.db"
MAX_RESULTS = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Note:
    """A stored note. Tags are kept lower-cased and de-duplicated."""

    id: int
    title: str
    body: str
    tags: list[str]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Note:
        """Build a Note from a `notes` table row."""
        tags = [tag for tag in row["tags"].split(",") if tag]
        return cls(row["id"], row["title"], row["body"], tags, row["created_at"])


@dataclass(frozen=True)
class NoteList:
    """Tool result wrapper: MCP structured content must be a JSON object, not a bare list."""

    notes: list[Note]


class NotesStore:
    """CRUD access to the notes table. Opens a short-lived connection per call."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, title: str, body: str, tags: list[str] | None = None) -> Note:
        """Insert a note and return it with its new id.

        Raises:
            ValueError: if the title is blank.
        """
        if not title.strip():
            raise ValueError("title must not be empty")
        clean_tags = sorted({tag.strip().lower() for tag in tags or [] if tag.strip()})
        created_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO notes (title, body, tags, created_at) VALUES (?, ?, ?, ?)",
                (title.strip(), body, ",".join(clean_tags), created_at),
            )
            note_id = cursor.lastrowid
        if note_id is None:
            raise RuntimeError("SQLite did not return an id for the inserted note")
        return Note(note_id, title.strip(), body, clean_tags, created_at)

    def list(self, tag: str | None = None, limit: int = 20) -> list[Note]:
        """Return the newest notes first, optionally only those carrying `tag`."""
        limit = max(1, min(limit, MAX_RESULTS))
        with closing(self._connect()) as conn:
            if tag:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE ',' || tags || ',' LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
                    (f"%,{_escape_like(tag.strip().lower())},%", limit),
                )
            else:
                rows = conn.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,))
            return [Note.from_row(row) for row in rows]

    def search(self, query: str, limit: int = 20) -> list[Note]:
        """Case-insensitive substring search over titles and bodies.

        Raises:
            ValueError: if the query is blank.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        pattern = f"%{_escape_like(query.strip())}%"
        limit = max(1, min(limit, MAX_RESULTS))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\' "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            return [Note.from_row(row) for row in rows]

    def delete(self, note_id: int) -> bool:
        """Delete a note; return False if it did not exist."""
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def create_server(
    store: NotesStore,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    allowed_hosts: list[str] | None = None,
) -> FastMCP:
    """Build the notes server with its four tools registered against `store`."""
    server = FastMCP(
        SERVER_NAME,
        instructions="A persistent notebook: add, list, search and delete notes.",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        log_level="WARNING",
        transport_security=_transport_security(port, allowed_hosts),
    )

    @server.tool()
    def add_note(title: str, body: str, tags: list[str] | None = None) -> Note:
        """Save a new note. `tags` is an optional list of short labels like ['work', 'todo']."""
        return store.add(title, body, tags)

    @server.tool()
    def list_notes(tag: str | None = None, limit: int = 20) -> NoteList:
        """List the most recent notes, newest first. Pass `tag` to only show notes with that label."""
        return NoteList(store.list(tag, limit))

    @server.tool()
    def search_notes(query: str, limit: int = 20) -> NoteList:
        """Find notes whose title or body contains `query` (case-insensitive)."""
        return NoteList(store.search(query, limit))

    @server.tool()
    def delete_note(note_id: int) -> str:
        """Permanently delete the note with the given id."""
        if not store.delete(note_id):
            raise ValueError(f"no note with id {note_id}")
        return f"Deleted note {note_id}."

    return server


def _transport_security(port: int, allowed_hosts: list[str] | None) -> TransportSecuritySettings | None:
    """Keep DNS-rebinding protection on, but let a gateway reach us under extra Host names."""
    if not allowed_hosts:
        return None  # FastMCP's default: protection on for loopback binds only.
    # Proxies that rewrite Host (e.g. WSO2 gateway `hostRewrite: auto`) may omit the port.
    extra = [host for name in allowed_hosts for host in (name, f"{name}:{port}")]
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *extra]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)


def main() -> None:
    """Parse CLI flags and run the server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="extra Host header name to accept, e.g. host.docker.internal for a gateway in Docker (repeatable)",
    )
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("NOTES_DB_PATH", DEFAULT_DB_PATH)))
    args = parser.parse_args()
    server = create_server(NotesStore(args.db), args.host, args.port, args.allowed_host)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
