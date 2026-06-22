# tests — integration test suite

One test file per tool group.  All tests exercise the same dispatch path the
production orchestrator uses: `runner::run(name, call_id, &args, &ctx)`.

## Test harness (`tests/harness/mod.rs`)

Four public functions:

```rust
// Invoke with a fresh default ToolContext:
harness::invoke("tool_name", json!({...})) -> Value

// Invoke with a caller-supplied context (for lifecycle tests):
harness::invoke_with_ctx("tool_name", json!({...}), &ctx) -> Value

// Assert the response is a success envelope; return the body:
harness::expect_success(resp) -> Value

// Assert the response has a specific error code; return the detail string:
harness::expect_error(&resp, "error_code") -> String
```

Schema testing:
```rust
// Retrieve the generated JSON Schema for a tool:
harness::schema("tool_name") -> Value

// Assert the generated schema matches a hand-written spec (normalised):
harness::assert_schema_matches_spec("tool_name", expected_json_str)
```

Confirmation testing:
```rust
// Get the confirmation details for a tool call, if any:
harness::confirmation("tool_name", args) -> Option<ConfirmationDetails>
```

## How to write good tests

Every tool group should have tests for:

1. **Happy path** — verify all expected response fields are present and correct
2. **Every error code** — one test per distinct `ToolError::code()` value
3. **Validation failures** — missing required field, out-of-range value
4. **Lifecycle tests** — for session tools: open → use → close in the same context

For lifecycle tests, share a `ToolContext` explicitly:

```rust
let ctx = ToolContext::new();
let saved = harness::expect_success(harness::invoke_with_ctx(
    "credential_save",
    json!({"name": "key", "type": "api_key", "secret": "s1"}),
    &ctx,
));
let list = harness::expect_success(harness::invoke_with_ctx(
    "cred_list", json!({}), &ctx,
));
```

For stateless tools, `harness::invoke` creates a fresh context per call — fine
for unit testing a single tool call in isolation.

## Test files

| File | Tool group tested |
|------|------------------|
| `bytes.rs` | `bytes_*` |
| `calculator.rs` | `calculator` |
| `code.rs` | `code_*` |
| `credentials.rs` | `credential_*` |
| `crypto.rs` | `aead_*`, `hmac_compute`, `signature_*`, `kdf_derive`, `hkdf_*` |
| `datetime.rs` | `datetime` |
| `file.rs` | `file_*` |
| `harness_smoke.rs` | Harness self-tests (unknown tool, schema) |
| `hash.rs` | `hash_compute`, `hash_scan` |
| `http_session.rs` | `http_session_*` |
| `network_diag.rs` | `dns_lookup`, `ping_icmp`, `trace_route`, `port_scan`, `ip_scan`, `host_info` |
| `notes.rs` | `notes_*` |
| `random.rs` | `random` |
| `remote_fs_session.rs` | `remote_fs_session_*` |
| `sql_session.rs` | `sql_session_*` |
| `ssh_session.rs` | `ssh_session_*` |
| `subagent.rs` | `sub_run` |
| `tcp_session.rs` | `tcp_session_*` |
| `telnet_session.rs` | `telnet_session_*` |
| `tls_session.rs` | `tls_session_*` |
| `udp_session.rs` | `udp_session_*` |
| `unit_convert.rs` | `unit_convert` |

## Running tests

```bash
# All tests
cargo test -p zend-tools

# Single group
cargo test -p zend-tools --test credentials

# Single test function
cargo test -p zend-tools credential_save_http_bearer
```
