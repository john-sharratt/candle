# zend

The Zen Code daemon: a persistent AI coding assistant server built on `candle-conversation`, exposing an OpenAI-compatible HTTP API plus a substrate/telemetry viewer web UI.

## What it does

`zend` is a single long-running binary that owns one `candle_conversation::ConversationEngine` (model weights, KV arenas, scheduler thread, persistence thread, summariser thread) and serves it over HTTP. Two kinds of client talk to it over the same `POST /v1/chat/completions` endpoint:

- **`zen-vscode`** — a Continue fork VS Code extension. It passes its own `tools` array; `zend` treats those as client-executed (emits `tool_calls` in the response and returns immediately, letting Continue post results back as `role: "tool"` messages).
- **The embedded web chat** (served from `web/`, no client tooling — no Node, no build step, assets embedded via `include_dir!`) — passes no tools, so `zend` injects its own server-registered tool catalog (`zend-tools`), executes any tool calls itself in a loop, and streams only the final assistant text.

On startup `zend` resolves a **workspace** directory (see `--working-dir` below), opens (or creates) `<workspace>/.substrate/` — the mandatory redo-log persistence layer `candle-conversation` requires — replays it into a `Substrate`, loads the model, installs the tool catalog and any calibrated system-prompt sections, then runs **ingest**: walking the workspace to populate the projection schema's turn-sink layers. A background filesystem watcher (`watcher.rs`) debounces edits into incremental re-ingest. Only once loading finishes does the daemon accept `/v1/chat/completions` traffic (`GET /v1/status` reports the loading-state machine's progress to the frontend in the meantime).

Every layer the projection schema declares is filled by convention rather than annotation — `src/ingest.rs` derives the load plan from the schema's shape, not from extra YAML metadata:

- `repo_map` — a per-directory folder scan (`src/repo_scan/`): walks the workspace and mints one conversation per directory, explored as two `code_read`-shaped tool round-trips (`file_list` the folder, then `file_read` its `README`/module-doc anchor), the last of which **decodes** a two-sentence summary of what the folder is for. Both tool responses are produced by running the real tools, so a prefilled response cannot drift from the live one.
- `code_reading` — a per-file ingest (`src/code_read/`): each file becomes one conversation, parsed into scope-aware parts via tree-sitter (Rust, Python, TypeScript, JavaScript, Go, C, C++, Java, Ruby, PHP, Bash, HTML, CSS, plus a structured-config and a generic fallback parser); each part contributes a prefilled `read_file` tool-call round-trip.
- Any other declared turn-sink layer reads raw ChatML records from a same-named folder — but only for a **mind** workspace (one carrying its own `<workspace>/projection.yaml`), never for an arbitrary coding-agent project directory.

**Ingest layers are append-only.** `repo_map` and `code_reading` are explicitly marked append-only cumulative content (`session.rs` calls `engine.mark_layer_append_only(layer_id)` before ingest runs) — refresh re-ingests changed files and tombstones deleted ones, but never rewrites history in place; this is also what the summariser and provenance self-locality logic key off of to exclude ingest content from certain live-dialogue-only behaviors.

## Key modules / layout

| Path | Role |
|---|---|
| `src/main.rs` | CLI parsing (`clap`), logging setup, GPU-poison watchdog, HTTP bind, graceful shutdown |
| `src/lib.rs` | Crate module list (also built as a library for the test harnesses) |
| `src/session.rs` | `ZendSession` / `InferenceState` — the daemon's central state: model load sequence, per-conversation state, tool/think-steering compilation, `submit`/`submit_with_sampling` streaming entry points |
| `src/api/` | The axum HTTP router: `chat.rs` (`/v1/chat/completions`), `models.rs`, `status.rs`, `substrate.rs` (read-only viewer), `telemetry.rs`, `conversations.rs`, `files.rs`, `ws_logs.rs` |
| `src/ingest.rs` | Structure-derived load-plan resolution — decides *how* each schema layer/collection gets populated |
| `src/repo_scan/` | `repo_map` folder-scan ingest: walk, per-directory units, anchor selection, turn rendering, binary sniffing |
| `src/code_read/` | `code_reading` per-file ingest: tree-sitter parsers, scope carving, header/summary generation |
| `src/tools.rs`, `tool_def.rs`, `tool_summary.rs` | Tool catalog installation into the projection schema, tool-call extraction/execution loop, deterministic catalog summaries |
| `src/stencil` (in `candle-conversation`) | Constrained decoding backing the tool-call/think steering `zend` compiles at load |
| `src/config.rs` | `DaemonConfig` — workspace path, port, disabled layers, ingest-dir overrides |
| `src/watcher.rs` | Filesystem watcher debouncing edits into `repo_map`/`code_reading` refresh |
| `src/conv_file_store.rs`, `conv_files.rs` | Per-conversation uploaded-file storage, independent of the inference engine |
| `src/model_choice.rs`, `download.rs` | VRAM-adaptive quant selection and first-run model download/cache resolution |
| `web/` | The embedded single-page frontend (chat UI, `substrate.html`, `perf.html`, `project.html`) |
| `src/prompts/projection.yaml` | The bundled default projection schema |
| `src/prompts/tools/*.yaml` | Declarative tool definitions (schema, description, calibration examples) |

## Key types & entry points

- `main()` (`src/main.rs`) — parses CLI, builds `DaemonConfig`, constructs `ZendSession`, builds the axum router, binds, serves with graceful shutdown.
- `ZendSession::new` / `start_loading` / `submit` / `submit_with_sampling` — the daemon's façade over the engine; `submit` yields a `StreamItem` stream (`Status`, `Token`, `Projection`, `Tool`) consumed by the SSE handler.
- `api::router(session)` — the axum `Router` builder; the full route table is in `src/api/mod.rs`.
- `ingest::{ingest_layers, section_sinks}` — derive the turn-sink / section-collection load plan from the active `Schema`.
- `tools::{install_tool_catalog, extract_tool_calls, run_tool_calls}` — bridge `zend_tools::registry` into the conversation's projected system prompt and the post-decode tool loop.

## HTTP API

All routes are served from one axum `Router` (`src/api/mod.rs`):

```
POST   /v1/chat/completions              OpenAI-compatible chat endpoint (streaming SSE or single JSON body)
GET    /v1/models                        OpenAI-shaped model list (Continue queries this on startup)
GET    /v1/me                            The caller's role, the tools modes it may choose, and its default
GET    /v1/status                        Loading-state snapshot for the frontend loading overlay
GET    /v1/telemetry                     Live perf-dashboard telemetry
GET    /v1/phases                        Per-wave phase-timing ring (scheduler wave breakdown)
GET    /v1/promotes                      Working-set promotion counters
GET    /v1/substrate                     Read-only substrate overview
GET    /v1/substrate/system-prompt       Current system-prompt section listing
GET    /v1/substrate/tools               Installed tool catalog
GET    /v1/substrate/layer/:name         One projection layer's conversations
POST   /v1/substrate/layer/:name/toggle  Enable/disable a layer
GET    /v1/substrate/timeline/:tl        One timeline's detail + summary forest
POST   /v1/substrate/project             Run a projection against the live substrate (search)
POST   /v1/debug/maintenance             Force a persistence maintenance pass
GET    /v1/conversations                 Sidebar conversation list
GET/DELETE /v1/conversations/:id         Conversation detail / delete (tombstone)
GET/POST /v1/conversations/:id/files     Per-conversation file upload/list
GET/DELETE /v1/conversations/:id/files/:file_id  File content / delete
POST   /v1/conversations/:id/archive     One-way archive (text-only distillation)
GET    /ws/logs                          WebSocket log tail (backlog replay + live broadcast)
```

Anything not matched falls back to the embedded `web/` frontend (`GET /`, `/perf`, `/substrate`, `/project`, resolved to their `.html` files).

`POST /v1/chat/completions` accepts the standard OpenAI `messages`/`stream`/`max_tokens` fields plus `zend` extensions: `conv_id`, `tools` (a `ToolMode` dial — `none`/`restricted`/`comprehensive`/`mutable`), `identity`, `effort`, `verbosity`, `think`, `assistant_prefill`, `force_high_resolution`, `lossless_kv`.

### Tools modes and who may use them

| Mode | Tools | File writes and deletes | Grants |
|---|---|---|---|
| `none` | none | — | none |
| `restricted` | the safe subset: not high-risk, needing no grant | the session's in-memory overlay | none |
| `comprehensive` | every tool | the session's in-memory overlay | network, exec, secrets |
| `mutable` | every tool | **the workspace on disk** | all, disk write included |

`comprehensive` and `mutable` are for admins. The caller's role comes from the gateway's `x-tokera-*` identity headers, resolved against `zend.roles.yaml` (embedded at build time; same shape as `npcd/npcd.web.yaml`'s `roles`). An admin defaults to `comprehensive`; everyone else — signed in or not — defaults to `restricted`, and a request asking for a mode above its role runs as `restricted` rather than failing (`src/access.rs`). The GUI asks `GET /v1/me` and offers only the allowed modes.

The identity headers are believed only from a trusted peer: loopback, the `--host` address (the gateway on this box connects from it), and each `--gateway <ip>`. From any other peer they are ignored and the caller is anonymous, so a machine that reaches zend's port directly cannot claim to be an admin.

**What a mode may do is enforced below the prompt.** Each mode's tool round runs in a `ToolContext` carrying that mode's grants (`access::grants`), and `zend-tools` refuses a call twice over when the grant is missing: at dispatch, from the tool's declared capabilities, and again at the primitive — every socket, DNS lookup, HTTP client, subprocess, JS VM, SQLite connection, credential read and disk-writing file store is reached only through a function that checks the grant. A call the model makes for a tool its mode never offered is answered `{"error":"not_permitted"}` and nothing is done. See `zend-tools/src/grants.rs`.

In `mutable` mode the file tools run against `VfsStore::direct`: writes go to disk (via a temporary file renamed over the target), deletes remove the file, and the `secrets/` refusal and `..` normalisation still apply.

## Running it

```bash
zend                                # workspace = current directory, port 8080
zend /path/to/project               # explicit workspace path
zend --port 9090                    # custom port
zend --working-dir ../mind          # separate substrate + schema, cwd untouched
zend -v                             # DEBUG logging (-vv = TRACE)
```

CLI flags (`src/main.rs`, `clap`-derived):

| Flag | Effect |
|---|---|
| `workspace` (positional, default `.`) | Root of the project to analyse |
| `--working-dir <path>` | Overrides the workspace: where `.substrate/` and an optional `projection.yaml` live, without `chdir`-ing the process. Takes precedence over the positional path. Use it to run a separate "mind" (its own substrate + tuned schema) alongside a normal coding workspace |
| `--port <u16>` (default `8080`) | TCP port |
| `--host <ip>` (default `127.0.0.1`) | Bind address; the daemon is **unauthenticated**, so binding non-loopback (e.g. `0.0.0.0`) logs a warning. Identity headers from this address (and loopback) are believed |
| `--gateway <ip>` (repeatable) | Another peer whose `x-tokera-*` identity headers are believed — a gateway on a different machine. Every other peer is anonymous |
| `-v` / `-vv` | DEBUG / TRACE logging |
| `--disable-layer <NAME>` (repeatable) | Take a projection layer (or section collection) **out of service** by schema name: not populated at boot, not refreshed by the watcher, excluded from the provenance gather, not normalization-warmed, not swept for crashed partials. Its turns stay in the substrate untouched — dropping the flag restores them — but while it is set they cannot be selected into any projection. An explicit upload into a disabled per-file layer is the one exception and still reads |
| `--skip-layer <NAME>` (repeatable) | Keep a turn-sink layer fully **in service** — gathered, warmed, and swept for crashed partials — but read nothing from disk for it this boot (no startup ingest, no watcher-driven refresh). The flag for "the corpus is built, stop re-reading the disk". A layer named by both flags is simply disabled |
| `--ingest-dir <layer>=<path>` (repeatable) | Override the content root a derived ingest layer reads from |
| `--max-depth <N>` | Bound how deep the `repo_map` and `code_reading` layers read, in path components below each layer's content root (`1` = the root's own files, `2` = one folder down, like `find -maxdepth`). Covers the startup ingest, the watcher-driven refresh and the watcher's event filter. Content already ingested from deeper is **frozen**: kept and retrievable, but never re-read and never retired by the deleted-file sweep. Changing or dropping the bound changes the listing of the root and of every folder with subfolders, so those folders are re-summarised once |
| `--compact-substrate` | Force a whole-store redo-log compaction on load |
| `--summarize` | Let conversations launch background tree summaries — of every eight turns, of accumulated segments, and at each UTC day boundary. **Off by default**: each summary re-reads its window as a fresh prefill on the same scheduler, ahead of the live turn, so a conversation's next tool round waits behind it |
| `--wipe-substrate` | **Destructive** — delete `<workspace>/.substrate` before loading |
| `--model <PRESET>` | Run this model preset, by its variant name (e.g. `Qwen35_0_8B_Q8`, `Qwen38_FlashNext_Q4KO`), instead of choosing one from the card's measured VRAM. A substrate holds one model's K/V, so pair a different model with its own `--working-dir` |

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
