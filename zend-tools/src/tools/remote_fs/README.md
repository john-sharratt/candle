# remote_fs — remote_fs_session_* (10 tools)

Protocol-agnostic remote filesystem operations addressed by URI scheme.

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `remote_fs_session_open` | Connect via SFTP; confirms |
| `list_dir.rs` | `remote_fs_session_list_dir` | List directory contents |
| `stat.rs` | `remote_fs_session_stat` | Stat a file or directory |
| `get.rs` | `remote_fs_session_get` | Download remote file into VFS |
| `put.rs` | `remote_fs_session_put` | Upload VFS file to remote; confirms |
| `delete.rs` | `remote_fs_session_delete` | Delete remote file; confirms |
| `mkdir.rs` | `remote_fs_session_mkdir` | Create directory; confirms |
| `rename.rs` | `remote_fs_session_rename` | Rename/move remote file; confirms |
| `list.rs` | `remote_fs_session_list` | List open remote FS sessions |
| `close.rs` | `remote_fs_session_close` | Close the connection |
| `mod.rs` | — | `RemoteFsError` enum; `now()` helper |

## URI format

`sftp://<host>:<port>/<path>`

The URI scheme determines the protocol.  Currently only `sftp://` is supported.
Other schemes return `not_supported`.

## VFS integration

`remote_fs_session_get` downloads the remote file and writes it into the session
VFS at the path you specify (or a default path derived from the remote filename).
You can then `file_read`, `file_edit`, and `remote_fs_session_put` the file back.

## Error codes

| Code | When |
|------|------|
| `session_not_found` | Session ID not in registry |
| `credential_not_found` | Named credential absent from store |
| `not_supported` | URI scheme is not `sftp` |
| `connection_failed` | TCP or SSH handshake error |
| `auth_failed` | Credential rejected by remote |
| `sftp_error` | SFTP protocol error (stat, read, write, etc.) |
| `not_found` | Remote path does not exist |
| `vfs_error` | Error writing downloaded content to local VFS |
| `session_limit_exceeded` | 5-session-per-user cap |

## Confirmation policy

| Tool | Confirms |
|------|----------|
| `remote_fs_session_open` | Once |
| `remote_fs_session_put` | Every call |
| `remote_fs_session_delete` | Every call |
| `remote_fs_session_mkdir` | Every call |
| `remote_fs_session_rename` | Every call |
| All read and list tools | Never |
