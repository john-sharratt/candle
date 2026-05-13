# udp_session — udp_session_{open,send,recv,list,close}

Bound UDP socket sessions for datagram-based protocols.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `udp_session_open` | Bind local socket; set default peer |
| `send.rs` | `udp_session_send` | Sendto; per-send peer override; confirms |
| `recv.rs` | `udp_session_recv` | Recvfrom with timeout; returns source address |
| `list.rs` | `udp_session_list` | List open sessions |
| `close.rs` | `udp_session_close` | Release socket |
| `mod.rs` | — | `UdpError` enum |

## Wire format

Same as TCP sessions: use `data` for text, `data_hex` for binary.
See the TCP session README for the full wire format description.

## Per-send peer override

`udp_session_send` accepts an optional `peer` parameter that overrides the
session's default peer for that single send.  This is useful for protocols
(e.g. TFTP) that reply from a different port than they received on.

## Error codes

| Code | When |
|------|------|
| `bind_failed` | Could not bind local socket |
| `session_not_found` | Session ID not in registry |
| `send_failed` | Sendto error |
| `recv_failed` | Recvfrom error or timeout |
| `invalid_params` | Malformed peer address |

## Confirmation

`udp_session_send` confirms every call.  Open, recv, list, and close do not.
