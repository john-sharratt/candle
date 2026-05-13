# tcp_session — tcp_session_{open,send,recv,list,close}

Raw TCP sessions for byte-level protocol work.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `tcp_session_open` | Bind and connect; SSRF guard |
| `send.rs` | `tcp_session_send` | Write text or hex bytes |
| `recv.rs` | `tcp_session_recv` | Read by amount or timeout |
| `list.rs` | `tcp_session_list` | Enumerate open sessions |
| `close.rs` | `tcp_session_close` | Drop the connection |
| `mod.rs` | — | `TcpError` enum |

## When to use TCP vs TLS vs HTTP

| Use case | Right tool |
|----------|-----------|
| TLS-protected non-HTTP service (LDAPS, IMAPS…) | `tls_session_*` |
| HTTP API with auth / cookies | `http_session_*` |
| One-off page fetch | `web_fetch` |
| TLS handshake debugging / byte-level protocol work | `tcp_session_*` + crypto tools |
| Custom binary protocol on plain TCP | `tcp_session_*` |

## Wire format

TCP uses **hex** for non-text payloads.  Hex stays readable when the model
reasons about protocol bytes: `16 03 01` is a TLS record header the model
recognises; `FgMB` in base64 is not.

- **Send**: `data` (UTF-8 text) or `data_hex` (hex bytes).  If both are given,
  `data_hex` wins.  Hex input accepts whitespace and is case-insensitive.
- **Recv `format`**: `"auto"` (default) returns `data` for valid printable UTF-8,
  `data_hex` otherwise; `"hex"` always returns `data_hex`; `"text"` always
  returns `data` (with `had_invalid_bytes: true` if non-UTF-8 was received).

## Recv modes — exactly one required

| Parameter | Behaviour |
|-----------|-----------|
| `recv_amt` | Block until exactly this many bytes arrive (or EOF) |
| `recv_wait` | Collect whatever arrives within this many seconds |

Providing neither → `missing_recv_mode`.  Providing both → `conflicting_recv_modes`.

## Error codes

| Code | When |
|------|------|
| `connection_failed` | TCP connect error |
| `session_not_found` | Session ID not in registry |
| `url_blocked` | Target resolves to private/loopback address |
| `send_failed` | Write error |
| `recv_failed` | Read error |
| `missing_recv_mode` | Neither `recv_amt` nor `recv_wait` provided |
| `conflicting_recv_modes` | Both `recv_amt` and `recv_wait` provided |

## Confirmation

`tcp_session_send` confirms every call.  Open, recv, list, and close do not.
