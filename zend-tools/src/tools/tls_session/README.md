# tls_session — tls_session_{open,send,recv,list,close}

TLS-encrypted sessions via `native-tls` for non-HTTP application protocols.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `tls_session_open` | TLS handshake; optional mTLS via `tls_client_cert` credential |
| `send.rs` | `tls_session_send` | Write text or hex bytes |
| `recv.rs` | `tls_session_recv` | Read with timeout |
| `list.rs` | `tls_session_list` | List open sessions |
| `close.rs` | `tls_session_close` | TLS close_notify + TCP disconnect |
| `mod.rs` | — | `TlsError` enum |

## When to use TLS vs TCP

| Situation | Right tool |
|-----------|-----------|
| TLS-protected non-HTTP service (LDAPS, IMAPS, SMTPS, MQTTS…) | `tls_session_*` |
| HTTP/HTTPS APIs | `http_session_*` |
| Debugging TLS handshake at byte level | `tcp_session_*` + crypto tools |
| Custom binary protocol on plain TCP | `tcp_session_*` |

## Mutual TLS

Pass a `tls_client_cert` credential ID in `credential_id` to enable mTLS.
The credential's `secret` must be a PEM bundle containing both the certificate
chain and the private key (both objects concatenated in a single string).

## Wire format

Same as TCP sessions: `data` for text payloads, `data_hex` for binary.

## Error codes

| Code | When |
|------|------|
| `connection_failed` | TCP connect error |
| `handshake_failed` | TLS negotiation rejected |
| `session_not_found` | Session ID not in registry |
| `send_failed` | Write error |
| `recv_failed` | Read error or timeout |
| `invalid_params` | Malformed address or unrecognised parameter |
| `credential_not_found` | Named credential absent from store |
