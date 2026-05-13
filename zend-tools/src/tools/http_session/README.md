# http_session — http_session_{open,request,list,close}

Stateful HTTP client sessions with a persistent cookie jar and optional auth headers.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `http_session_open` | Create reqwest client with credential and base URL |
| `request.rs` | `http_session_request` | Send a request; handle binary bodies |
| `list.rs` | `http_session_list` | List open sessions |
| `close.rs` | `http_session_close` | Drop the client |
| `mod.rs` | — | `HttpSessionError` enum |

## When to use HTTP sessions vs web_fetch

| Situation | Right tool |
|-----------|-----------|
| One-off page fetch, no auth needed | `web_fetch` |
| REST API with Bearer/Basic auth | `http_session_open` + `http_session_request` |
| Multi-request flow needing cookie state | `http_session_open` + `http_session_request` |
| Custom headers (API key) across requests | `http_session_open` with `http_header` credential |

## Authentication

Credentials are set at open time and applied to every request:

| Credential type | Header sent |
|----------------|-------------|
| `http_bearer` | `Authorization: Bearer <token>` |
| `http_basic` | `Authorization: Basic <base64(user:pass)>` |
| `http_header` | `<header_name>: <secret>` (e.g. `X-API-Key: ...`) |

## Response body encoding

`http_session_request` returns at most one body field per response:
- `body` — present when the response has a text content-type (text/*, application/json, etc.)
  and the bytes are valid UTF-8
- `body_b64` — present for binary content types or non-UTF-8 bytes

HTTP uses base64 (not hex) — that's the established convention for HTTP-over-JSON
tooling, and response bodies are usually structured rather than byte-level protocol messages.

`body_truncated: true` is set when the body was cut at `max_response_bytes` (default 32 KiB,
max 1 MiB).

## Confirmation policy

| Method | Confirms |
|--------|----------|
| GET, HEAD, OPTIONS | Never (read-only by HTTP spec) |
| POST, PUT, PATCH, DELETE | Every call |

Open, list, and close do not confirm.

## Error codes

| Code | When |
|------|------|
| `session_not_found` | Session ID not in registry |
| `session_dead` | Underlying client invalid |
| `connection_failed` | Could not reach the host |
| `timeout` | Request exceeded timeout |
| `url_blocked` | Target resolves to private address (SSRF guard) |
| `invalid_credential_type` | Credential is not an `http_*` type |
| `session_limit_exceeded` | 5-session-per-user cap |
| `credential_not_found` | Named credential absent from store |
