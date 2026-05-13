# state — in-memory stores

All shared mutable state used by tools lives here.  Each store is an `Arc`-
wrapped, lock-protected structure.  `ToolContext` holds one `Arc` of each store
and passes them by reference into every tool invocation.

Nothing in this directory touches the filesystem, network, or database — stores
are pure in-process memory structures that live for the lifetime of the daemon
process (or the test's `ToolContext`).

## Modules

| Module | Struct | Purpose |
|--------|--------|---------|
| `vfs.rs` | `VfsStore` | Per-session in-memory virtual filesystem |
| `credentials.rs` | `CredentialStore` | Named authentication material |
| `notes.rs` | `NotesStore` | Cross-conversation persistent key-value notes |
| `sessions.rs` | `SessionRegistry` | All open protocol sessions |
| `hash_state.rs` | `HashStateStore` | Running hash contexts for streaming digest tools |

---

## VfsStore (`vfs.rs`)

`HashMap<String, String>` — normalised path → UTF-8 content.

- **Cap**: 10 MiB total content per instance (enforced on `write`)
- **Normalisation**: leading `/` stripped; `.`/`..` resolved
- **Methods**: `write`, `read`, `list`, `delete`, `total_bytes`

---

## CredentialStore (`credentials.rs`)

`HashMap<String, Credential>` — name → credential record.

- Primary key is `name` (friendly label); `id` is a UUID for legacy lookups
- `delete` operates by name; active sessions using the credential are unaffected
- Secrets are stored as plain strings in this in-process store (production daemon
  would encrypt with chacha20poly1305)
- **Methods**: `save`, `list`, `delete`, `get_by_name`, `get_by_id`

---

## NotesStore (`notes.rs`)

`HashMap<String, Note>` — key → note record.

- Key max 256 bytes; content max 1 MiB
- Writing empty content tombstones (removes) the note — no separate delete
- `search` does substring/FTS scan over content and tags
- `list` returns metadata only (key, bytes, tags, updated_at); no content
- **Methods**: `write`, `read`, `search`, `list`

---

## SessionRegistry (`sessions.rs`)

Ten typed `RwLock<HashMap<String, Arc<Mutex<Entry>>>>` sub-maps, one per
protocol/resource class:

| Field | Entry type | Protocol |
|-------|-----------|---------|
| `ssh` | `SshEntry` | SSH (ssh2 + TcpStream) |
| `ssh_processes` | `SshProcess` | Async SSH commands |
| `telnet` | `TelnetEntry` | Telnet (raw TCP) |
| `http` | `HttpEntry` | HTTP/HTTPS (reqwest) |
| `tcp` | `TcpEntry` | Raw TCP |
| `udp` | `UdpEntry` | UDP socket |
| `tls` | `TlsEntry` | TLS (native-tls) |
| `sql` | `SqlEntry` | SQLite (rusqlite) |
| `remote_fs` | `RemoteFsEntry` | SFTP (ssh2) |
| `code` | `CodeEntry` | Python/Node subprocess |

Each sub-map exposes `insert_*`, `get_*`, `remove_*`, `list_*` methods.

`SshConn` and `SqlConn` use `unsafe impl Send` guarded by the outer `Mutex`
because their underlying types are not `Send` by default.

---

## HashStateStore (`hash_state.rs`)

`HashMap<String, HashContext>` — ID → running hash context.

- Created by `hash_state_init`; updated by `hash_state_update`; finalised and
  removed by `hash_state_finalize`
- Each context holds the algorithm name and accumulated digest state
- IDs must be unique per store; `init` with a duplicate ID returns `id_already_exists`
