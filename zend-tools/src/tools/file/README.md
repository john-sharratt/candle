# file — file_{write,read,edit,list,delete,present}

In-memory virtual filesystem (VFS) tools for the web chat session.  Nothing
written here ever touches disk — the VFS is a `HashMap<String, String>` in the
[`VfsStore`] store, scoped to the session lifetime.

## Files

| File | Tool | Description |
|------|------|-------------|
| `write.rs` | `write` | Create or overwrite a file; enforces 10 MiB cap |
| `read.rs` | `file_read` | Return full content + line count |
| `edit.rs` | `file_edit` | Unique-substring replacement |
| `list.rs` | `file_list` | List all files, optionally filtered by path prefix |
| `delete.rs` | `file_delete` | Remove a file; returns `deleted` flag |
| `present.rs` | `file_present` | Foreground presentation gesture |
| `mod.rs` | — | `FileError` enum |

## Path normalisation

All paths are normalised before storage:
- Leading `/` stripped
- `.` and empty segments collapsed
- `..` pops a level

`/src/../main.rs`, `./main.rs`, and `main.rs` all map to the same entry `main.rs`.

## `file_edit` uniqueness requirement

`old_str` must appear exactly once in the file:
- 0 occurrences → `not_found`
- 1 occurrence → replacement applied
- 2+ occurrences → `ambiguous` error with count

This matches Claude Code's `str_replace` semantics and prevents accidental
multi-site edits from an insufficiently specific search string.

## `file_present` vs Files panel

The Files panel is driven by `vfs_update` SSE events emitted after each
`write` / `file_edit` / `file_delete`.  `file_present` is a separate,
explicit foreground gesture that emits a `file_present` SSE event — use it
to draw the user's attention to specific files as deliverables.

## VFS size cap

10 MiB total across all files in the session.  The cap is checked on every
`write`; if the new total would exceed 10 MiB the write is rejected with
`vfs_full`.

## Error codes

| Code | When |
|------|------|
| `not_found` | Path not in VFS (`read`, `edit`, `delete`) |
| `vfs_full` | Write would exceed 10 MiB cap |
| `ambiguous` | `old_str` appears more than once (`edit`) |
| `no_files_found` | All requested paths missing (`present`) |
