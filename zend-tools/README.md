# zend-tools

Rust crate providing the 93-tool server-side execution engine for the **Zen Code** daemon (`zend`). Every tool in the system — from `web_search` and `datetime` through SSH sessions, cryptographic primitives, and sandboxed code execution — is implemented here as a statically-registered, JSON-in / JSON-out Rust function.

See [`docs/tool-system.md`](../docs/tool-system.md) for the full specification.

---

## Architecture overview

```
Orchestrator (zend daemon)
│
│  tool_call JSON from LLM
▼
runner::run(name, call_id, args, &ctx)
│
│  registry::find(name)  →  RegisteredTool
│    ├── schema()         JSON Schema for params
│    ├── run(ctx, args)   parse → validate → execute → serialize
│    └── confirmation()   confirmation prompt before execution
│
▼
tool_response JSON back to LLM
```

**Key invariant:** `runner::run` always returns valid JSON. A successful tool returns its typed response; a failure returns `{"error": "<code>", "detail": "..."}`. The LLM always gets something actionable.

---

## Crate layout

```
src/
├── lib.rs           — public re-exports, authoring guide
├── tool.rs          — Tool trait, ToolError trait, ConfirmationDetails
├── registry.rs      — RegisteredTool, ALL_TOOLS static table, find()
├── runner.rs        — run() and confirmation() dispatch entry points
├── context.rs       — ToolContext (shared state bundle)
├── state/           — in-memory stores used by tools
│   ├── credentials.rs   CredentialStore
│   ├── hash_state.rs    HashStateStore
│   ├── notes.rs         NotesStore
│   ├── sessions.rs      SessionRegistry (all protocol entries)
│   └── vfs.rs           VfsStore (in-memory virtual filesystem)
└── tools/           — 93 tool implementations
    ├── bytes/       — bytes_transcode, bytes_pack, bytes_unpack, bytes_xor
    ├── calculator.rs
    ├── code/        — code_run, code_session_{open,exec,list,close}
    ├── credentials/ — credential_{save,list,delete}
    ├── crypto/      — aead_{encrypt,decrypt}, hmac_compute, signature_{sign,verify},
    │                  kdf_derive, hkdf_{extract,expand_label}
    ├── datetime.rs
    ├── file/        — file_{write,read,edit,list,delete,present}
    ├── hash/        — hash_compute, hash_scan
    ├── hash_state/  — hash_state_{init,update,finalize}
    ├── http_session/— http_session_{open,request,list,close}
    ├── network_diag/— dns_lookup, ping_icmp, trace_route, port_scan, ip_scan, host_info
    ├── notes/       — notes_{write,read,search,list}
    ├── random.rs
    ├── remote_fs/   — remote_fs_session_{open,list_dir,stat,get,put,delete,mkdir,rename,list,close}
    ├── sql_session/ — sql_session_{open,query,list,close}
    ├── ssh/         — ssh_session_{open,exec,exec_async,poll,list,close}
    ├── subagent.rs
    ├── tcp_session/ — tcp_session_{open,send,recv,list,close}
    ├── telnet/      — telnet_session_{open,send,list,close}
    ├── tls_session/ — tls_session_{open,send,recv,list,close}
    ├── totp.rs
    ├── udp_session/ — udp_session_{open,send,recv,list,close}
    ├── unit_convert.rs
    ├── weather.rs
    ├── web_fetch.rs
    └── web_search.rs

tests/
├── harness/         — shared test helpers (invoke, expect_success, expect_error)
└── *.rs             — one integration test file per tool group
```

---

## Tool groups at a glance

| Group | Tools | Client | Notes |
|-------|-------|--------|-------|
| **Shared utilities** | `datetime`, `calculator`, `unit_convert`, `random`, `web_search`, `web_fetch`, `weather` | Both | No session state |
| **Virtual filesystem** | `file_{write,read,edit,list,delete,present}` | Web chat only | Per-session in-memory; 10 MiB cap |
| **Notes** | `notes_{write,read,search,list}` | Web chat only | Cross-conversation persistent KV store |
| **Credentials** | `credential_{save,list,delete}` | Web chat only | In-memory encrypted store |
| **SSH sessions** | `ssh_session_{open,exec,exec_async,poll,list,close}` | Web chat only | russh; TOFU host key; sentinel/nonce exec |
| **Telnet sessions** | `telnet_session_{open,send,list,close}` | Web chat only | Raw TCP + optional prompt-regex |
| **HTTP sessions** | `http_session_{open,request,list,close}` | Web chat only | reqwest; cookie jar; bearer/basic/header auth |
| **TCP sessions** | `tcp_session_{open,send,recv,list,close}` | Web chat only | Raw TCP; hex wire format for binary |
| **UDP sessions** | `udp_session_{open,send,recv,list,close}` | Web chat only | Bound UDP socket; hex wire format |
| **TLS sessions** | `tls_session_{open,send,recv,list,close}` | Web chat only | native-tls; for non-HTTP TLS services |
| **SQL sessions** | `sql_session_{open,query,list,close}` | Web chat only | SQLite via rusqlite |
| **Remote FS sessions** | `remote_fs_session_{open,list_dir,stat,get,put,delete,mkdir,rename,list,close}` | Web chat only | SFTP via ssh2; URI-addressed |
| **Network diagnostics** | `dns_lookup`, `ping_icmp`, `trace_route`, `port_scan`, `ip_scan`, `host_info` | Web chat only | Shell-subprocess implementations |
| **Security utilities** | `hash_compute`, `hash_scan`, `totp_generate` | Web chat only | SHA2/SHA3/BLAKE3/MD5; TOTP RFC 6238 |
| **Crypto primitives** | `aead_{encrypt,decrypt}`, `hmac_compute`, `signature_{sign,verify}`, `kdf_derive`, `hkdf_{extract,expand_label}` | Web chat only | AES-GCM, ChaCha20-Poly1305, Ed25519, P-256 |
| **Hash state** | `hash_state_{init,update,finalize}` | Web chat only | Streaming hash for large/chunked data |
| **Byte utilities** | `bytes_transcode`, `bytes_pack`, `bytes_unpack`, `bytes_xor` | Web chat only | struct.pack/unpack semantics; hex/base64 |
| **Code execution** | `code_run`, `code_session_{open,exec,list,close}` | Web chat only | Python/Node subprocess; persistent REPL |
| **Subagent** | `subagent_run` | Web chat only | Nested agent loop via SubagentRunner trait |

---

## The `Tool` trait

Every tool is a zero-state unit struct implementing `Tool`:

```rust
pub trait Tool: 'static {
    const NAME: &'static str;          // tool call name
    const DESCRIPTION: &'static str;  // 50–100 token trigger-rich blurb for the LLM
    type Request: DeserializeOwned + JsonSchema + Validate;
    type Response: Serialize;
    type Error: ToolError;
    fn run(ctx: &ToolContext, req: Self::Request) -> Result<Self::Response, Self::Error>;
    fn confirmation(_req: &Self::Request) -> Option<ConfirmationDetails> { None }
}
```

`Request` derives:
- `serde::Deserialize` — JSON argument parsing
- `schemars::JsonSchema` — auto-generated schema exposed to the LLM
- `validator::Validate` — field-level validation before `run` is called

`Response` derives `serde::Serialize`. `Error` implements `ToolError`:

```rust
pub trait ToolError: std::error::Error + Send + Sync + 'static {
    fn code(&self) -> &'static str;   // stable machine-readable error code
    fn detail(&self) -> String { self.to_string() }
}
```

Error responses are always `{"error": "<code>", "detail": "..."}`.

---

## Adding a tool

1. **Create `src/tools/<group>/<name>.rs`** (or add to an existing group module):

```rust
//! my_tool — one-line description.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::MyGroupError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct MyRequest {
    #[validate(length(min = 1))]
    pub query: String,
    pub max_results: Option<u32>,
}

#[derive(Serialize)]
pub struct MyResponse {
    pub results: Vec<String>,
    pub total: usize,
}

pub struct MyTool;

impl Tool for MyTool {
    const NAME: &'static str = "my_tool";
    const DESCRIPTION: &'static str =
        "50–100 token trigger-rich blurb. What it does, when to use it, \
         what triggers it ('search for', 'find me', 'look up'), what it returns, \
         and what to use INSTEAD when the request is slightly different. \
         See docs/tool-system.md § Tool Description Format.";
    type Request = MyRequest;
    type Response = MyResponse;
    type Error = MyGroupError;

    fn run(_ctx: &ToolContext, req: MyRequest) -> Result<MyResponse, MyGroupError> {
        Ok(MyResponse { results: vec![req.query], total: 1 })
    }
}

pub const MY_TOOL: RegisteredTool = RegisteredTool::new::<MyTool>();
```

2. **Export from the group's `mod.rs`**:
```rust
pub mod my_name;
pub use my_name::MY_TOOL;
```

3. **Register in `src/registry.rs`**:
```rust
use crate::tools::my_group::MY_TOOL;
// add MY_TOOL to the TOOLS static array in register_all()
```

4. **Write tests** in `tests/<group>.rs` — positive case + every error code.

---

## Error contract

The LLM should never see a panic or an opaque 500. Every failure path returns a stable JSON shape:

| Shape | Cause |
|-------|-------|
| `{"error": "invalid_arguments", "detail": "..."}` | Serde parse failure or `validator` constraint violation |
| `{"error": "unknown_tool", "detail": "..."}` | Tool name not in registry |
| `{"error": "internal_error", "detail": "..."}` | Response serialization failure (should never happen for well-formed types) |
| `{"error": "<tool-specific-code>", "detail": "..."}` | Tool returned `Err(e)` |

Error codes are **stable across releases** — the LLM may key off them for retry logic.

---

## Confirmation policy

Tools with remote side-effects return `Some(ConfirmationDetails)` from `confirmation()`. The orchestrator pauses the loop, shows a confirmation prompt to the user, and only calls `run()` on approval.

| Tool group | Confirmation policy |
|------------|---------------------|
| SSH `ssh_session_open` | Once per open (shows host + credential) |
| SSH `ssh_session_exec` / `exec_async` | Every call (shows exact command) |
| Telnet `telnet_session_send` | Every call |
| HTTP `http_session_request` | GET/HEAD/OPTIONS: never. POST/PUT/PATCH/DELETE: every call |
| TCP `tcp_session_send` | Every call |
| UDP `udp_session_send` | Every call |
| Credential `credential_save` | Every call (shows type + name) |
| List / recv / close tools | Never |

---

## Session lifecycle

Sessions are in-memory, scoped to a single conversation, and persist across model turns within that conversation.

- **IDs** — `sess_<uuid>` for sessions, `proc_<uuid>` for async SSH processes, `cred_<uuid>` for credentials
- **Idle timeout** — default 900 s, max 3600 s; configurable per-open
- **Resource cap** — 5 active sessions per user per protocol
- **Dead sessions** — connection error, remote disconnect, or idle timeout marks a session dead; the next tool call returns `{"error": "session_dead"}`
- **Conversation scoping** — sessions do not cross conversations

---

## Wire format for binary data

TCP and UDP use **hex** for non-text payloads (not base64). Hex stays readable when the model is reasoning about protocol bytes: `16 03 01` is a TLS record header the model recognises; `FgMB` in base64 is not. HTTP uses base64 (the established convention for HTTP-over-JSON tooling).

- `tcp_session_send` / `udp_session_send`: pass `data` (text) or `data_hex` (hex bytes); if both are given, `data_hex` wins
- `tcp_session_recv` / `udp_session_recv`: `format` parameter is `"auto"` | `"hex"` | `"text"`; auto returns `data` for valid printable UTF-8, `data_hex` otherwise
- Hex inputs accept whitespace and are case-insensitive: `"16 03 01"` == `"160301"`

---

## Credential types

| Type | Required fields | Used by |
|------|----------------|---------|
| `ssh_key` | `username`, `secret` (PEM/OpenSSH key), optional `passphrase` | `ssh_session_open`, `remote_fs_session_open` |
| `ssh_password` | `username`, `secret` | `ssh_session_open` |
| `telnet_password` | `username`, `secret` | `telnet_session_open` |
| `http_bearer` | `secret` (token) | `http_session_open` |
| `http_basic` | `username`, `secret` | `http_session_open` |
| `http_header` | `header_name`, `secret` | `http_session_open` |
| `totp_secret` | `secret` (base32 TOTP seed) | `totp_generate` |
| `sql_password` | `username`, `secret`, optional `default_database` | `sql_session_open` |
| `remote_fs_password` | `username`, `secret`, optional `domain` | `remote_fs_session_open` |
| `tls_client_cert` | `secret` (cert+key PEM bundle) | `tls_session_open` |
| `signing_key` | `secret` (PEM private key) | `signature_sign` |

Aliases `api_key` and `ed25519_key` are accepted for backward compatibility.

---

## Testing

Every tool group has a corresponding integration test file in `tests/`. Tests use the shared harness in `tests/harness/`:

```rust
// Invoke with a fresh default context:
let resp = harness::invoke("tool_name", json!({...}));

// Invoke with a shared context (for lifecycle tests):
let ctx = ToolContext::new();
let resp = harness::invoke_with_ctx("tool_name", json!({...}), &ctx);

// Assert success and get the response body:
let body = harness::expect_success(resp);

// Assert a specific error code:
harness::expect_error(&resp, "not_found");
```

Each tool group should have tests for:
- Happy path (every response field verified)
- Every distinct error code
- Validation failures (missing required field, out-of-range value)

Run all tests:
```bash
cargo test -p zend-tools
```
