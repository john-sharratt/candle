# zend

The Zen Code daemon: a persistent AI coding assistant server built on `candle-conversation`, exposing an OpenAI-compatible HTTP API plus a substrate/telemetry viewer web UI.

## What it does

`zend` is a single long-running binary that owns one `candle_conversation::ConversationEngine` (model weights, KV arenas, scheduler thread, persistence thread, summariser thread) and serves it over HTTP. Two kinds of client talk to it over the same `POST /v1/chat/completions` endpoint:

- **`zen-vscode`** — a Continue fork VS Code extension. It passes its own `tools` array; `zend` treats those as client-executed (emits `tool_calls` in the response and returns immediately, letting Continue post results back as `role: "tool"` messages).
- **The web chat** (`web/`, served from disk by the host process — no Node, no build step; saving a file is the deployment) — passes no tools, so `zend` injects its own server-registered tool catalog (`zend-tools`), executes any tool calls itself in a loop, and streams only the final assistant text.

`zend` runs as the one tenant of a **host process** (the `host` crate, `docs/shared_model_host.md`): the host owns logging, the self-heal watchdog, the model download, the HTTP server and the endpoints every tenant shares; zend brings its session and its own routes. `tokerad` is the same host with npcd beside zend, on one model.

On startup `zend` resolves a **workspace** directory (see `--working-dir` below), opens (or creates) `<workspace>/substrate/` — the mandatory redo-log persistence layer `candle-conversation` requires — replays it into a `Substrate`, loads the model, installs the tool catalog and any calibrated system-prompt sections, then starts **ingest** in the background: every branch origin holds, read from git rather than from any folder, populates the projection schema's turn-sink layers (`src/branch_ingest/`, design in `docs/zend_branch_ingest.md`). Origin watchers (`src/origin_watch/`) ask each repository's origin for its branch tips every couple of seconds and fetch when one moved, which wakes the next pass. Only once loading finishes does the daemon accept `/v1/chat/completions` traffic (`GET /v1/status` reports the loading-state machine's progress to the frontend in the meantime).

Every layer the projection schema declares is filled by convention rather than annotation — `src/ingest.rs` derives the load plan from the schema's shape, not from extra YAML metadata:

- `repo_map` — one conversation per folder a branch lists (`src/repo_scan/`), explored as ONE `code_read`-shaped tool round-trip (`file_list` the folder, listing its own entries and naming its subfolders without descending), which then **decodes** a one-sentence summary of what the folder is for. The folder is described from names and paths alone — nothing reads a file — and the tool response is produced by running the real tool at the commit the folder was found on, so a prefilled response cannot drift from the live one.
- `code_reading` — one conversation per file a branch holds (`src/code_read/`): a hidden tool-using conversation that reads the file with real `file_read` calls, at the commit it was found on, and answers with its summary. A file counts as read only once that conversation's chain finished; one that stopped part-way is queued again.
- Both are keyed by what they show — a file by its path and blob id, a folder by its listing and its manifest — so a unit on many branches is one conversation, a branch moving forward re-reads only what changed, and a conversation retrieves only the units its own base holds.
- Any other declared turn-sink layer reads raw ChatML records from a same-named folder — but only for a **mind** workspace (one carrying its own `<workspace>/projection.yaml`), never for an arbitrary coding-agent project directory.

**Ingest layers are append-only.** `repo_map` and `code_reading` are explicitly marked append-only cumulative content (`session.rs` calls `engine.mark_layer_append_only(layer_id)` before ingest runs) — a pass ingests new units and tombstones those no branch holds, but never rewrites history in place; this is also what the summariser and provenance self-locality logic key off of to exclude ingest content from certain live-dialogue-only behaviors.

## Key modules / layout

| Path | Role |
|---|---|
| `src/main.rs` | The binary: the host's flags beside zend's, then `host::run` with zend as its tenant |
| `src/cli.rs` | `ZendArgs` — zend's flags (and `tokerad.yaml`'s `zend:` section), resolved into a tenant |
| `src/tenant.rs` | `ZendTenant` / `ZendReports` — zend's start and its share of the host's shared endpoints |
| `src/lib.rs` | Crate module list (also built as a library for the test harnesses and `tokerad`) |
| `src/session.rs` | `ZendSession` / `InferenceState` — the daemon's central state: model load sequence, per-conversation state, tool/think-steering compilation, `submit`/`submit_with_sampling` streaming entry points |
| `src/api/` | zend's own routes: `chat.rs` (`/v1/chat/completions`), `models.rs`, `substrate.rs` (the viewer under `/v1/substrate/zend/`), `maintenance.rs`, `conversations.rs`, `files.rs`, `projections.rs` |
| `src/ingest.rs` | Structure-derived load-plan resolution — decides *how* each schema layer/collection gets populated |
| `src/branch_ingest/` | The ingest pass: every branch's tree, content keys, the plan (what to ingest, what to tombstone) |
| `src/origin_watch/` | Per-repository origin watchers: the `ls-refs` probe, cadence and backoff, fetch on change |
| `src/retrieval_scope.rs` | Which ingested units a conversation may retrieve — those its own base holds |
| `src/repo_scan/` | `repo_map` per-directory ingest: the unit's manifest hint, turn rendering, the folder tool loop, the directory pool |
| `src/code_read/` | `code_reading` per-file ingest: the hidden reading conversations, their pool, and which chains finished |
| `src/tools.rs`, `tool_def.rs`, `tool_summary.rs` | Tool catalog installation into the projection schema, tool-call extraction/execution loop, deterministic catalog summaries |
| `src/stencil` (in `candle-conversation`) | Constrained decoding backing the tool-call/think steering `zend` compiles at load |
| `src/config.rs` | `DaemonConfig` — workspace, disabled layers, ingest-dir overrides, the model (`host::MODEL`) and where the engine comes from |
| `src/conv_file_store.rs`, `conv_files.rs` | Per-conversation uploaded-file storage, independent of the inference engine |
| `web/` | The single-page frontend (chat UI, `substrate.html`, `perf.html`, `project.html`), served from disk by the host |
| `src/prompts/projection.yaml` | The bundled default projection schema |
| `src/prompts/tools/*.yaml` | Declarative tool definitions (schema, description, calibration examples) |

## Key types & entry points

- `main()` (`src/main.rs`) — parses the host's flags and zend's, resolves `ZendArgs` into a `ZendTenant`, and hands it to `host::run`, which starts it, serves, and drains it on Ctrl-C.
- `ZendSession::new` / `start_loading` / `submit` / `submit_with_sampling` — the daemon's façade over the engine; `submit` yields a `StreamItem` stream (`Status`, `Token`, `Projection`, `Tool`) consumed by the SSE handler.
- `api::routes(session, roles)` — zend's own routes, graded; the full table is in `src/api/mod.rs`. Status, identity, telemetry, the log stream, the tool catalog and the substrate overview are the host's shared endpoints (`host::endpoints`).
- `tenant::ZendTenant` — zend as a tenant of the `host` process: `zend` runs it alone, `tokerad` beside npcd.
- `ingest::{ingest_layers, section_sinks}` — derive the turn-sink / section-collection load plan from the active `Schema`.
- `tools::{install_tool_catalog, extract_tool_calls, run_tool_calls}` — bridge `zend_tools::registry` into the conversation's projected system prompt and the post-decode tool loop.

## HTTP API

zend's own routes (`src/api/mod.rs`, each graded by role — the table is pinned by a test there):

```
POST   /v1/chat/completions                          OpenAI-compatible chat endpoint (streaming SSE or single JSON body)
GET    /v1/models                                    OpenAI-shaped model list (Continue queries this on startup)
GET    /v1/repo_map                                  How complete the repo_map layer is
GET/DELETE /v1/substrate/zend/system-prompt          Current system-prompt section listing / retire its persisted copy   (admin)
GET    /v1/substrate/zend/layer/:name                One projection layer's conversations                                 (admin)
POST   /v1/substrate/zend/layer/:name/toggle         Enable/disable a layer                                               (admin)
GET/DELETE /v1/substrate/zend/timeline/:tl           One timeline's detail + summary forest / tombstone it                (admin)
GET    /v1/substrate/zend/timeline/:tl/selection     The timeline's last selection diagnostic                             (admin)
POST   /v1/substrate/zend/project                    Run a projection against the live substrate (search)                 (admin)
POST   /v1/substrate/zend/maintenance                Force a persistence maintenance pass                                 (admin)
GET    /v1/conversations                             Sidebar conversation list
GET/DELETE /v1/conversations/:id                     Conversation detail / delete (tombstone)
GET/POST /v1/conversations/:id/files                 Per-conversation file upload/list
GET/DELETE /v1/conversations/:id/files/:file_id      File content / delete
POST   /v1/conversations/:id/archive                 One-way archive (text-only distillation)
GET    /v1/conversations/:id/live                    The reply in flight, from its first frame
GET    /v1/conversations/:id/projections/:turn/:event  One projection point in full
POST   /v1/conversations/:id/projection-context      The context a projection assembled
```

The host serves the endpoints every tenant shares, once (`host::endpoints`): `GET /v1/status` (zend's line under `tenants.zend`), `GET /v1/me` (zend's tools modes under `tenants.zend`), `GET /v1/telemetry`, `/v1/memory`, `/v1/phases`, `/v1/promotes`, `/v1/profile`, `GET /v1/tools` (every catalog, each tool naming its tenant), `GET /v1/schema/layers`, `GET /v1/substrate`, and `WS /ws/logs`. It serves the web UI from `web/` for every path no route claims (`GET /`, `/perf`, `/substrate`, `/project`, resolved to their `.html` files), and the shared front-end at `/common/` and `/brand/`.

`POST /v1/chat/completions` accepts the standard OpenAI `messages`/`stream`/`max_tokens` fields plus `zend` extensions: **`conv_id` (required)**, `tools` (a `ToolMode` dial — `none`/`restricted`/`comprehensive`), `identity`, `effort`, `verbosity`, `think`, `assistant_prefill`, `force_high_resolution`, `lossless_kv`.

Every conversation is addressed by its own `conv_id`; there is no default conversation for unaddressed turns to accumulate on, so a request that omits it or sends it blank is refused with `400 missing_conv_id`. The one exception is `model: "passthrough"`, which runs the client's own context and never reaches the substrate.

### Tools modes and who may use them

| Mode | Tools | Grants |
|---|---|---|
| `none` | none | none |
| `restricted` | the safe subset, needing no grant and not high-risk: the file tools (reads, writes, edits, deletes) and the git readers | none |
| `comprehensive` | every tool: the network, credentials, the `code_*` JS sandbox, the git writers, SQLite, the command sandbox (`run_command`, `run_output`), and programs on this host (`ping_icmp`, `trace_route`, `sub_run`) | all |

In every mode the file tools work on the conversation's own overlay: a write, edit or delete is held in memory, recorded in the substrate as VFS events, and never reaches the workspace on disk. A conversation's changes reach a repository only through `git_commit`, which is Comprehensive's. So Restricted may write files and still runs nothing — no command line, no code, no network. Each mode projects, and summarises, exactly the tools its grants cover.

`run_command` runs one program (a test suite, a build) in a git repository's own folder, borrowed for the job: the folder is set aside, checked out on the conversation's branch with its uncommitted changes laid down, the program run, what it changed read back into the conversation's changes, and the folder put back exactly as it was. The programs it may start are listed in `src/sandbox_programs.rs`; job logs are kept in the workspace folder's `jobs/`. See `docs/zend_workspace_execution.md` §7.4.

`comprehensive` is for admins. The caller's role comes from the gateway's `x-tokera-*` identity headers, resolved against the estate's one role table, `host/roles.yaml` (compiled in). An admin defaults to `comprehensive`; everyone else — signed in or not — defaults to `restricted`, and a request asking for a mode above its role runs as `restricted` rather than failing (`src/access.rs`). The GUI asks `GET /v1/me` and offers only the allowed modes.

The identity headers are believed only from a trusted socket peer: loopback, the `--host` address (the gateway on this box connects from it), and each `--gateway <ip>`. From any other peer the server clears them before any handler sees the request, and the caller is anonymous, so a machine that reaches zend's port directly cannot claim to be an admin. `--local-signin <email>` stands in for a gateway that is not there: a loopback caller with no identity is signed in as that email (refused beside `--gateway`).

**What a mode may do is enforced below the prompt.** Each mode's tool round runs in a `ToolContext` carrying that mode's grants (`access::grants`), and `zend-tools` refuses a call twice over when the grant is missing: at dispatch, from the tool's declared capabilities, and again at the primitive — every socket, DNS lookup, HTTP client, subprocess, JS VM, SQLite connection, credential read and checkout run is reached only through a function that checks the grant. A call the model makes for a tool its mode never offered is answered `{"error":"not_permitted"}` and nothing is done. See `zend-tools/src/grants.rs`.

## Running it

zend serves a **workspace**: a folder holding a `workspace.yaml` that lists the repositories in scope, each a folder directly beside it. The daemon's `substrate/` lives in the workspace folder too, and it adds an `uploads` repository of its own. Every file tool requires a `repo` argument (constrained at decode time to these names; `file_list`, `file_search` and `file_grep` also take `*` for every repository) and a path relative to that repository. See `docs/zend_workspace_execution.md`.

```yaml
# <workspace>/workspace.yaml
repos:
  - name: candle
  - name: battle-cities
  - name: mind
```

```bash
zend                                # workspace = current directory, port 8080
zend /path/to/workspace             # explicit workspace path
zend --port 9090                    # custom port
zend --working-dir ../mind          # separate substrate + schema, cwd untouched
zend -v                             # DEBUG logging (-vv = TRACE)
```

The process's flags are the host's, the same in every binary (`host::HostArgs`, `host/src/cli.rs`):

| Flag | Effect |
|---|---|
| `--host <ip>` (default `127.0.0.1`) | Bind address. Binding non-loopback (e.g. `0.0.0.0`) logs a warning. Identity headers from this address (and loopback) are believed |
| `--port <u16>` (default `8080`) | TCP port |
| `--tls-port <u16>` | Also serve HTTPS (HTTP/3, HTTP/2, HTTP/1.1) on this port, with a certificate made at startup — the gateway's upstream |
| `--gateway <ip>` (repeatable) | Another peer whose `x-tokera-*` identity headers are believed — a gateway on a different machine. Every other peer is anonymous |
| `--local-signin <email>` | Sign a loopback caller that sends no identity in as this email. Off by default; refused beside `--gateway` |
| `-v` / `-vv` | DEBUG / TRACE logging |

zend's own flags (`src/cli.rs`, `ZendArgs`; the same settings are `tokerad.yaml`'s `zend:` section, less the one-shot ones):

| Flag | Effect |
|---|---|
| `workspace` (positional, default `.`) | The workspace folder — it must hold a `workspace.yaml` |
| `--working-dir <path>` | Overrides the workspace: where `substrate/` and an optional `projection.yaml` live, without `chdir`-ing the process. Takes precedence over the positional path. Use it to run a separate "mind" (its own substrate + tuned schema) alongside a normal coding workspace |
| `--ui <dir>` | Where the web UI is served from; by default `zend/web` in the source tree this binary was built from |
| `--disable-layer <NAME>` (repeatable) | Take a projection layer (or section collection) **out of service** by schema name: not ingested from the branches, excluded from the provenance gather, not normalization-warmed, not swept for crashed partials. Its turns stay in the substrate untouched — dropping the flag restores them — but while it is set they cannot be selected into any projection. An explicit upload into a disabled per-file layer is the one exception and still reads |
| `--ingest-dir <layer>=<path>` (repeatable) | Narrow a derived ingest layer to one workspace-relative folder (for the code layers, a folder inside one repository, e.g. `code_reading=candle/zend/src`) |
| `--max-depth <N>` | Bound how deep the `repo_map` and `code_reading` layers read, in path components below each repository's root or `--ingest-dir` folder (`1` = the root's own files, `2` = one folder down, like `find -maxdepth`), on every branch. Content ingested from deeper is not found by the walk, and is retired like anything else no branch holds |
| `--compact-substrate` | Force a whole-store redo-log compaction on load (one-shot) |
| `--self-check` / `--dry-run` | Ask every stored conversation four yes/no questions about its own integrity during load and tombstone each that answers no; with `--dry-run`, report and tombstone nothing (one-shot) |
| `--qsa-selection-budget <N>` | Run the QSA selection with this many positions in place of the checkpoint's own (a budget of the whole context is the dense control) |
| `--qsa-window <N>` / `--qsa-recent <N>` / `--qsa-recent-mode candidate\|forced` | The stratified selection's window and recent strata (defaults 131072 / 8192 / `candidate`; `0` and `0` is the checkpoint's own selection) |
| `--secrets <PATH>` | The secrets file holding the API keys and tokens zend presents to third-party services (`tavily_api_key`, `github_token`). Defaults to `~/.zend/secrets.yaml`; a named file that does not exist fails the launch |
| `--summarize` | Let conversations launch background tree summaries — of every eight turns, of accumulated segments, and at each UTC day boundary. **Off by default**: each summary re-reads its window as a fresh prefill on the same scheduler, ahead of the live turn, so a conversation's next tool round waits behind it |
| `--wipe-layer <NAME>` (repeatable) | **Destructive** — tombstone every conversation in a turn-sink layer before the first ingest pass, which rebuilds it from the branches (one-shot) |
| `--wipe-substrate` | **Destructive** — delete `<workspace>/substrate` before loading (one-shot) |

A one-shot flag acts once, at the start it is given to: a self-heal relaunch drops it, and `tokerad.yaml` cannot set it (`ZendArgs::ONE_SHOT`). There is no model flag: every daemon runs the prescribed model (`host::MODEL`), and a substrate written under another model refuses to open, naming both.

Continue (`zen-vscode`) configuration points at the daemon as an OpenAI provider:

```json
{ "provider": "openai", "apiBase": "http://localhost:8080", "model": "zen-code" }
```

A `projection.yaml` in the workspace (or `--working-dir`) overrides the bundled default schema (`src/prompts/projection.yaml`) entirely; its mere presence is also the "this is a mind, not a plain coding project" signal that gates raw ChatML turn-sinks and a workspace-local `tools/*.yaml` override.

## Related docs

- `docs/coding_assistant.md` — Zen Code product overview (daemon + `zen-vscode` + web chat); note several routes it documents (`/v1/zen/*`) are design-stage and not yet implemented — see `docs/zend_ui_redesign.md` for the ground-truth route table.
- `docs/zend_ui_redesign.md` — the authoritative frontend/API plan, closest to what is actually shipped.
- `docs/tool-system.md` — the full server-registered tool catalog (95 tools) and the Continue-vs-web-chat tool-execution split.
- `docs/sdlc_agent.md` — broader engineering-agent architecture vision this daemon is one instance of.
- `docs/web_search_design.md` — design of the `web_*` tool family (implemented in the sibling `zend-tools` crate).
- `docs/stencil_tree.md` — the constrained-decoding mechanism (tool-call shape, `<think>` steering) `zend` compiles at load from `candle_conversation::stencil`.
