# sample-agent

A local AI agent in Python that runs an Ollama model on your machine and uses 7 tools served by two standalone MCP servers. The servers speak **MCP protocol 2025-06-18 over Streamable HTTP**, the combination WSO2 AI gateways can proxy.

```
                     ┌─────────── Streamable HTTP, MCP 2025-06-18 ───────────┐
 you ⇄ sample_agent ─┤  (direct, or through a WSO2 AI gateway)               │
        │            ├─▶ toolbox server :8001/mcp  calculate, convert_units, get_current_time
        ▼            └─▶ notes server   :8002/mcp  add_note, list_notes, search_notes, delete_note
   Ollama (local LLM, :11434)                       └─ SQLite: data/notes.db
```

| Server | Tool | What it does |
|---|---|---|
| toolbox | `calculate` | Safe arithmetic (an AST evaluator, not `eval`): `+ - * / // % **`, `sqrt`, `log`, `sin`, `pi`… |
| toolbox | `convert_units` | Converts length, mass, volume and temperature |
| toolbox | `get_current_time` | Current time in any IANA timezone |
| notes | `add_note` | Saves a note with optional tags |
| notes | `list_notes` | Lists the newest notes, optionally filtered by tag |
| notes | `search_notes` | Case-insensitive search over titles and bodies |
| notes | `delete_note` | Deletes a note by id |

## MCP version and WSO2 compatibility

WSO2's gateway MCP proxies declare which MCP spec versions they accept (`specVersions`). The newest AI Gateway docs list 2025-06-18, 2025-11-25 and 2026-07-28. Their example, and the oldest version all of them share, is **2025-06-18**. This project therefore pins **`mcp==1.23.0`**, the last Python SDK release whose native protocol is 2025-06-18. Release 1.23.1 moved to 2025-11-25, and the 2.x SDK defaults to 2026-07-28 and probes `server/discover`, which older gateways don't know.

The servers are also set up to work well behind a proxy:

- **Streamable HTTP** at `/mcp`. Gateways cannot proxy stdio. stdio is still available for local use.
- **`stateless_http=True`**: no `Mcp-Session-Id` affinity, so any gateway replica can forward any request.
- **`json_response=True`**: plain JSON replies instead of SSE streams.
- **DNS-rebinding protection stays on.** Use `--allowed-host` to accept the gateway's Host header.

Run `python -m sample_agent -v` to see the negotiated version for each server.

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ollama pull qwen3:8b        # default model; any Ollama model with tool calling works
```

Smaller or different models: `ollama pull qwen3:4b` / `llama3.1:8b`, then pass `-m <model>` or set `OLLAMA_MODEL`.

## Run the MCP servers

Each server is a single standalone file. Run each one in its own terminal:

```bash
python servers/toolbox_server.py          # http://127.0.0.1:8001/mcp
python servers/notes_server.py            # http://127.0.0.1:8002/mcp  (db: data/notes.db, or NOTES_DB_PATH / --db)
```

Flags shared by both: `--host`, `--port`, `--transport {streamable-http,stdio}`, `--allowed-host NAME` (repeatable).

You can inspect them with any MCP client, e.g. `npx @modelcontextprotocol/inspector`, using the URL `http://127.0.0.1:8001/mcp`.

## Run the agent

```bash
python -m sample_agent                                   # interactive chat
python -m sample_agent -p "What is 17% of 2340? Save the answer as a note tagged math"
python -m sample_agent -m llama3.1:8b -v                 # other model, show MCP connection details
```

In chat, `/tools` lists the tools, `/reset` clears the history, and `exit` quits. Each tool call is printed as it happens:

```
you> convert 10 miles to km and note it down
  ↳ convert_units({"value": 10, "from_unit": "mi", "to_unit": "km"}) → 10 mi = 16.09344 km
  ↳ add_note({"title": "10 miles in km", "body": "10 mi = 16.09344 km"}) → {"id": 1, ...}
agent> 10 miles is 16.09 km. I saved it as note #1.
```

Servers are listed in `mcp_servers.json`. It uses the same `mcpServers` shape as Claude Desktop, and an entry takes either `url` (+ optional `headers`) or `command` + `args` for stdio. `${VAR}` references are filled in from the environment.

## Run the agent as an HTTP API (two copies in parallel)

There are two full copies of the agent: `sample_agent` (default port **9001**) and `sample_agent_2` (default port **9002**). Each copy has its own code, so they can change independently. Both use the same MCP servers, model and config files. `--serve` runs a copy as an HTTP API instead of a terminal chat; `--port` overrides the port.

```bash
export WSO2_GATEWAY_URL=http://default-default.gateway.localhost:19080
export LLM_API_KEY=<consumer key from the console>
ARGS=(-m qwen3:8b -c mcp_servers.wso2.json --llm-base-url "$WSO2_GATEWAY_URL/ollama" --llm-api-key-header X-API-Key)

python -m sample_agent   --serve "${ARGS[@]}"     # http://127.0.0.1:9001
python -m sample_agent_2 --serve "${ARGS[@]}"     # http://127.0.0.1:9002   (second terminal)
```

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "model": ..., "tools": [...], "sessions": n}` |
| `POST` | `/chat` | `{"message": "...", "session_id": "optional"}` → `{"session_id", "reply", "tool_calls": [{name, arguments, result}]}` |
| `DELETE` | `/chat/{session_id}` | forget that conversation (204, or 404 if unknown) |

```bash
curl -s -X POST localhost:9001/chat -H 'Content-Type: application/json' \
     -d '{"message": "Convert 21 km to miles", "session_id": "demo"}'
```

Reuse a `session_id` to continue a conversation; leave it out to start a new one. Each copy keeps up to 100 sessions in memory, dropping the least recently used. Errors from the model endpoint return 502; an agent that runs out of tool steps returns 500.

## Going through a WSO2 AI gateway

Both the model and the tools can go through a WSO2 API Platform gateway. The values below match the gateway (1.2.1) bundled with a local Agent Manager (`amp-local` k3d cluster). Clients reach it at `http://default-default.gateway.localhost:19080`, and from inside the cluster this Mac is `host.k3d.internal`.

1. **Start the servers so the gateway may call them.** DNS-rebinding protection stays on; only that Host name is added:
   ```bash
   python servers/toolbox_server.py --allowed-host host.k3d.internal
   python servers/notes_server.py   --allowed-host host.k3d.internal
   ```
   Ollama needs nothing extra: the cluster reaches it at `host.k3d.internal:11434`.
2. **Create the gateway resources.** Full field values are in `wso2/gateway-resources.yaml`.
   - **Ollama: in the Agent Manager console** (http://console.amp.localhost:8080). Create an LLM provider with the `openai` template, context `/ollama`, upstream `http://host.k3d.internal:11434/v1`, and access allowed for `POST /chat/completions` and `GET /models`. The form requires an upstream API key; enter any value, since Ollama ignores it. Agent Manager adds a caller key check, so generate a consumer API key for the provider in the console. Callers send it in the `X-API-Key` header.
   - **MCP proxies: with kubectl.** The console can't create these: before saving, Agent Manager connects to the MCP server to list its tools, and its SSRF guard rejects private addresses such as `host.k3d.internal`. The WSO2 gateway operator deploys `Mcp` resources directly:
     ```bash
     k3d kubeconfig get amp-local > /tmp/amp-local.kubeconfig
     KUBECONFIG=/tmp/amp-local.kubeconfig kubectl -n default-default apply -f wso2/mcp-proxies.k8s.yaml
     # remove later: ... kubectl -n default-default delete mcp -l app.kubernetes.io/part-of=sample-agent
     ```
     The upstream is the full endpoint (`…:8001/mcp`), because the gateway forwards `/toolbox/mcp` to the upstream URL unchanged. With `hostRewrite: auto`, the server sees `Host: host.k3d.internal` without a port; `--allowed-host` accepts both forms. These proxies don't appear in the Agent Manager console.
3. **Point the agent at the gateway** for the tools, and for the model too once you have a consumer key:
   ```bash
   export WSO2_GATEWAY_URL=http://default-default.gateway.localhost:19080
   python -m sample_agent -v -m qwen3:1.7b -c mcp_servers.wso2.json          # tools via the gateway

   export LLM_API_KEY=<consumer key from the console>
   python -m sample_agent -v -m qwen3:1.7b -c mcp_servers.wso2.json \
       --llm-base-url "$WSO2_GATEWAY_URL/ollama" --llm-api-key-header X-API-Key   # model via the gateway too
   ```
   `--llm-base-url` switches the agent from Ollama's native API to the OpenAI-format `/chat/completions` that the gateway exposes. It also works without a gateway: `--llm-base-url http://127.0.0.1:11434/v1`.

## Tests and lint

```bash
pytest          # unit tests plus in-memory MCP round-trips through both servers; no Ollama needed
ruff check . && ruff format --check .
```

## Layout

```
servers/toolbox_server.py   standalone MCP server, 3 tools
servers/notes_server.py     standalone MCP server, 4 tools (SQLite)
sample_agent/config.py      mcpServers JSON loader with ${VAR} expansion
sample_agent/hub.py         connects to N MCP servers, merges tools, routes calls
sample_agent/agent.py       Ollama tool-calling loop
sample_agent/openai_compat.py  OpenAI-format chat client (for gateways / --llm-base-url)
sample_agent/__main__.py    CLI / REPL, or HTTP API with --serve
sample_agent/http_api.py    HTTP API (POST /chat, sessions)
sample_agent_2/             full copy of sample_agent (default port 9002)
mcp_servers.json            direct connection (default)
mcp_servers.wso2.json       connection through a WSO2 gateway
wso2/gateway-resources.yaml gateway LLM provider + MCP proxy field reference
wso2/mcp-proxies.k8s.yaml   MCP proxies to kubectl-apply (the console rejects private upstreams)
```
