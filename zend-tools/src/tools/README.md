# Tool implementations

Each subdirectory and file in this directory is one tool group. The table below
maps every tool name to its source file and summarises the key behaviours.

## Module conventions

Every tool module defines:

| Item | Type | Description |
|------|------|-------------|
| `FooRequest` | `struct` | Parameters — derives `Deserialize + JsonSchema + Validate` |
| `FooResponse` | `struct` | Success payload — derives `Serialize` |
| `FooError` (in `mod.rs`) | `enum` | Group error type — implements `ToolError` |
| `FooImpl` | unit `struct` | Implements `Tool` |
| `FOO_TOOL` | `pub const RegisteredTool` | Registration constant — added to `registry::register_all` |

The `DESCRIPTION` constant on each `Tool` impl is the **description tier** shown
to the LLM: 50–100 tokens, trigger-rich, with use-when / use-NOT-when cues and
cross-references. See `docs/tool-system.md § Tool Description Format`.

---

## Tool index

### Shared utilities (7 tools)

| Tool | File | Notes |
|------|------|-------|
| `datetime` | `datetime.rs` | chrono + chrono-tz; IANA timezone lookup |
| `calculator` | `calculator.rs` | evalexpr; no eval code path |
| `unit_convert` | `unit_convert.rs` | Static dimension table; affine temperature |
| `random` | `random.rs` | rand crate; integer/float/choice/shuffle/dice |
| `web_search` | `web_search.rs` | Tavily API; 1-hour cache |
| `web_fetch` | `web_fetch.rs` | reqwest + readability extractor; SSRF guard |
| `weather` | `weather.rs` | Open-Meteo geocoding + forecast APIs |

### Virtual filesystem (6 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `write` | `file/write.rs` | Create or overwrite; 10 MiB VFS cap |
| `file_read` | `file/read.rs` | Full content + line count |
| `file_edit` | `file/edit.rs` | Unique-substring replacement |
| `file_list` | `file/list.rs` | Path prefix filter; sorted |
| `file_delete` | `file/delete.rs` | Idempotent; returns `deleted` flag |
| `file_present` | `file/present.rs` | Foreground presentation gesture |

### Notes (4 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `notes_write` | `notes/write.rs` | Upsert; empty content = tombstone/delete |
| `notes_read` | `notes/read.rs` | Exact key lookup |
| `notes_search` | `notes/search.rs` | FTS5 query and/or tag filter |
| `notes_list` | `notes/list.rs` | Prefix/tag enumeration; metadata only |

### Credentials (3 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `credential_save` | `credentials/save.rs` | Type allowlist; PEM validation for ssh_key |
| `cred_list` | `credentials/list.rs` | Metadata only; never returns secrets |
| `credential_delete` | `credentials/delete.rs` | By name; active sessions unaffected |

### SSH sessions (6 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `ssh_open` | `ssh/open.rs` | TOFU host key; confirms |
| `ssh_session_exec` | `ssh/exec.rs` | Sentinel/nonce; 32 KiB cap; confirms |
| `ssh_session_exec_async` | `ssh/exec_async.rs` | Returns process_id; confirms |
| `ssh_session_poll` | `ssh/poll.rs` | Read chunks; optional signal |
| `ssh_session_list` | `ssh/list.rs` | Metadata only |
| `ssh_session_close` | `ssh/close.rs` | Graceful disconnect |

### Telnet sessions (4 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `telnet_session_open` | `telnet/open.rs` | Raw TCP; optional prompt pattern |
| `telnet_send` | `telnet/send.rs` | `send` + optional `expect` regex; confirms |
| `telnet_session_list` | `telnet/list.rs` | |
| `telnet_session_close` | `telnet/close.rs` | |

### HTTP sessions (4 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `http_session_open` | `http_session/open.rs` | reqwest client + cookie jar |
| `http_request` | `http_session/request.rs` | GET no-confirm; POST/PUT/etc confirm; body/body_b64 |
| `http_session_list` | `http_session/list.rs` | |
| `http_session_close` | `http_session/close.rs` | |

### TCP sessions (5 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `tcp_session_open` | `tcp_session/open.rs` | SSRF guard; confirms |
| `tcp_session_send` | `tcp_session/send.rs` | `data` or `data_hex`; confirms |
| `tcp_session_recv` | `tcp_session/recv.rs` | `recv_amt` XOR `recv_wait`; `auto/hex/text` format |
| `tcp_session_list` | `tcp_session/list.rs` | |
| `tcp_session_close` | `tcp_session/close.rs` | |

### UDP sessions (5 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `udp_session_open` | `udp_session/open.rs` | Binds local socket |
| `udp_session_send` | `udp_session/send.rs` | Per-send peer override; confirms |
| `udp_session_recv` | `udp_session/recv.rs` | Timeout-based; returns source addr |
| `udp_session_list` | `udp_session/list.rs` | |
| `udp_session_close` | `udp_session/close.rs` | |

### TLS sessions (5 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `tls_session_open` | `tls_session/open.rs` | native-tls; optional mTLS via `tls_client_cert` |
| `tls_session_send` | `tls_session/send.rs` | Same wire format as TCP |
| `tls_session_recv` | `tls_session/recv.rs` | |
| `tls_session_list` | `tls_session/list.rs` | |
| `tls_session_close` | `tls_session/close.rs` | |

### SQL sessions (4 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `sql_session_open` | `sql_session/open.rs` | SQLite via rusqlite |
| `sql_session_query` | `sql_session/query.rs` | Returns rows as JSON objects |
| `sql_session_list` | `sql_session/list.rs` | |
| `sql_session_close` | `sql_session/close.rs` | |

### Remote filesystem sessions (10 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `remote_fs_session_open` | `remote_fs/open.rs` | SFTP via ssh2; URI-addressed |
| `remote_fs_session_list_dir` | `remote_fs/list_dir.rs` | |
| `remote_fs_session_stat` | `remote_fs/stat.rs` | |
| `remote_fs_session_get` | `remote_fs/get.rs` | Downloads into VFS |
| `remote_fs_session_put` | `remote_fs/put.rs` | Uploads from VFS; confirms |
| `remote_fs_session_delete` | `remote_fs/delete.rs` | Confirms |
| `remote_fs_session_mkdir` | `remote_fs/mkdir.rs` | Confirms |
| `remote_fs_session_rename` | `remote_fs/rename.rs` | Confirms |
| `remote_fs_session_list` | `remote_fs/list.rs` | Active sessions metadata |
| `remote_fs_session_close` | `remote_fs/close.rs` | |

### Network diagnostics (6 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `dns_lookup` | `network_diag/dns_lookup.rs` | A/AAAA/MX/TXT/PTR records |
| `ping_icmp` | `network_diag/ping_icmp.rs` | Subprocess ping; RTT stats |
| `trace_route` | `network_diag/trace_route.rs` | Subprocess traceroute; hop list |
| `port_scan` | `network_diag/port_scan.rs` | TCP connect scan; open/closed/filtered |
| `ip_scan` | `network_diag/ip_scan.rs` | ARP/ping sweep of a CIDR range |
| `host_info` | `network_diag/host_info.rs` | Reverse DNS, PTR, OS hint |

### Security utilities (3 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `hash_compute` | `hash/compute.rs` | SHA256/512, SHA1, MD5, SHA3, BLAKE3 |
| `hash_scan` | `hash/scan.rs` | Identify algorithm from digest + pre-image |
| `totp` | `totp.rs` | RFC 6238 TOTP via `totp-rs` |

### Cryptographic primitives (8 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `aead_encrypt` | `crypto/aead_encrypt.rs` | AES-GCM, ChaCha20-Poly1305 |
| `aead_decrypt` | `crypto/aead_decrypt.rs` | same |
| `hmac_compute` | `crypto/hmac_compute.rs` | HMAC-SHA256/512 |
| `signature_sign` | `crypto/signature_sign.rs` | Ed25519, ECDSA-P256; inline or credential key |
| `signature_verify` | `crypto/signature_verify.rs` | same algorithms |
| `kdf_derive` | `crypto/kdf_derive.rs` | Argon2id, PBKDF2, scrypt |
| `hkdf_extract` | `crypto/hkdf_extract.rs` | HKDF-Extract (RFC 5869) |
| `hkdf_expand_label` | `crypto/hkdf_expand_label.rs` | HKDF-Expand-Label (TLS 1.3 §7.1) |

### Running hash state (3 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `hash_state_init` | `hash_state/init.rs` | Named context; fixed algorithm |
| `hash_state_update` | `hash_state/update.rs` | Feed chunks with any encoding |
| `hash_state_finalize` | `hash_state/finalize.rs` | Returns digest; discards context |

### Byte utilities (4 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `bytes_transcode` | `bytes/transcode.rs` | hex ↔ base64 ↔ base64url ↔ utf8 |
| `bytes_pack` | `bytes/pack.rs` | struct.pack semantics; big/little endian |
| `bytes_unpack` | `bytes/unpack.rs` | struct.unpack semantics |
| `bytes_xor` | `bytes/xor.rs` | XOR two byte sequences |

### Code execution (5 tools) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `code_run` | `code/run.rs` | One-shot; Python or Node subprocess |
| `code_session_open` | `code/session_open.rs` | Persistent REPL; Python or Node |
| `code_session_exec` | `code/session_exec.rs` | Execute in running REPL |
| `code_session_list` | `code/session_list.rs` | |
| `code_session_close` | `code/session_close.rs` | Kills subprocess |

### Subagent (1 tool) — web chat only

| Tool | File | Notes |
|------|------|-------|
| `sub_run` | `subagent.rs` | Nested agent loop via `SubagentRunner` trait |
