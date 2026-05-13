# telnet — telnet_session_{open,send,list,close}

Raw TCP sessions for legacy network equipment that lacks SSH.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `telnet_session_open` | Connect; optional prompt pattern |
| `send.rs` | `telnet_session_send` | Write and optionally wait for a regex match |
| `list.rs` | `telnet_session_list` | List open sessions |
| `close.rs` | `telnet_session_close` | Drop the connection |
| `mod.rs` | — | `TelnetError` enum |

## When to use Telnet

Use `telnet_session_*` for legacy network gear (managed switches, routers,
serial console servers) that does not support SSH.  For any modern host with
SSH, use `ssh_session_*` instead.

## Send semantics

`telnet_session_send` parameters:
- `send` — text to write to the stream (e.g. `"show interfaces\r\n"`)
- `expect` — optional regex; the tool reads until the pattern matches
- `timeout_sec` — how long to wait for the expect match (default varies)

Response fields:
- `received` — everything read from the stream
- `matched` — whether `expect` matched
- `duration_ms` — elapsed time
- `received_truncated` — true if output was cut at 32 KiB

## Error codes

| Code | When |
|------|------|
| `connection_failed` | TCP connect error |
| `session_not_found` | Session ID not in registry |
| `send_failed` | Write to stream failed |
| `timeout` | Timeout elapsed before `expect` matched |

## Confirmation

`telnet_session_send` confirms every call.  Open, list, and close do not.
