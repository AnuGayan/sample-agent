import json
from pathlib import Path

import pytest

from sample_agent.config import ConfigError, ServerConfig, load_config


def write_config(tmp_path: Path, servers: object) -> Path:
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def test_loads_http_and_stdio_servers_with_env_expansion(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "toolbox": {
                "url": "https://${GW_HOST}/toolbox/mcp",
                "headers": {"Authorization": "Bearer ${GW_TOKEN}"},
            },
            "notes": {"command": "python", "args": ["servers/notes_server.py", "--transport", "stdio"]},
        },
    )
    configs = load_config(path, environ={"GW_HOST": "gw.local:8243", "GW_TOKEN": "abc"})
    assert configs == [
        ServerConfig(
            name="toolbox",
            url="https://gw.local:8243/toolbox/mcp",
            headers={"Authorization": "Bearer abc"},
        ),
        ServerConfig(
            name="notes",
            command="python",
            args=("servers/notes_server.py", "--transport", "stdio"),
            cwd=tmp_path,
        ),
    ]


def test_missing_env_var_is_reported_by_name(tmp_path: Path) -> None:
    path = write_config(tmp_path, {"t": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer ${NOPE}"}}})
    with pytest.raises(ConfigError, match=r"\$\{NOPE\}"):
        load_config(path, environ={})


@pytest.mark.parametrize(
    "servers",
    [
        {},
        {"t": {}},
        {"t": {"url": "http://x/mcp", "command": "python"}},
        {"t": "http://x/mcp"},
    ],
)
def test_invalid_entries_are_rejected(tmp_path: Path, servers: object) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, servers), environ={})


def test_unreadable_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(bad)


def test_shipped_config_points_at_both_local_servers() -> None:
    configs = load_config(Path(__file__).resolve().parent.parent / "mcp_servers.json", environ={})
    assert [(c.name, c.url) for c in configs] == [
        ("toolbox", "http://127.0.0.1:8001/mcp"),
        ("notes", "http://127.0.0.1:8002/mcp"),
    ]


def test_shipped_wso2_config_routes_both_servers_through_the_gateway() -> None:
    path = Path(__file__).resolve().parent.parent / "mcp_servers.wso2.json"
    configs = load_config(path, environ={"WSO2_GATEWAY_URL": "http://default-default.gateway.localhost:19080"})
    assert [(c.name, c.url) for c in configs] == [
        ("toolbox", "http://default-default.gateway.localhost:19080/toolbox/mcp"),
        ("notes", "http://default-default.gateway.localhost:19080/notes/mcp"),
    ]
