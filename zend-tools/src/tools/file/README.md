# file — file_{write,read,edit,list,delete,present}

Overlay-filesystem tools. [`VfsStore`] stacks an in-memory session layer over
the daemon's working directory: reads resolve session-first and fall through to
the real project, writes and edits land in memory. **Nothing here ever modifies a
file on disk.**

Editing a project file reads it from below and writes the result above, so the
copy-up happens only when the edit succeeds. Deleting one records a *whiteout* —
the path stops resolving and stops listing, the file on disk is untouched.

Without a configured workspace (`ToolContext::new` rather than
`ToolContext::with_workspace`) the store degenerates to the session layer alone,
which is what most unit tests use.

## Files

| File | Tool | Description |
|------|------|-------------|
| `write.rs` | `write` | Create or overwrite a file; enforces 10 MiB cap |
| `read.rs` | `file_read` | Return a line range as a numbered, fenced excerpt |
| `edit.rs` | `file_edit` | Unique-substring replacement |
| `list.rs` | `file_list` | Paged union listing of project + session files, optionally filtered by path prefix |
| `delete.rs` | `file_delete` | Drop a session file or whiteout a project one; returns `deleted` flag |
| `present.rs` | `file_present` | Foreground presentation gesture |
| `mod.rs` | — | `FileError` enum |

## Path normalisation

All paths are normalised to one canonical key, shared by both layers:
- Leading `/` stripped
- `.` and empty segments collapsed
- `..` pops a level (it can never escape the root — popping an empty stack is a no-op)
- A leading `workspace/` segment dropped, because `/workspace` is the mount point
  the tool definitions document for the working directory

`/workspace/src/main.rs`, `/src/main.rs`, `./src/../src/main.rs`, and
`src/main.rs` all map to the same entry `src/main.rs`. A project containing a
genuine top-level `workspace/` directory cannot address it through these tools.

## Workspace layer rules

The walk is `ignore`-driven (ripgrep's crate), so `.gitignore`, `.ignore`, the
global git ignore, and hidden-file rules all apply — `target/` never appears.
Hidden files are omitted from listings the way `ls` omits them but still read
fine by exact path, which is what `file_read`'s own `/workspace/.gitignore`
example depends on.

Project files above 4 MiB, or whose bytes are not valid UTF-8, list with their
true size but fail to read with `unreadable`.

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

10 MiB total across the session layer.  The cap is checked on every `write`; if
the new total would exceed 10 MiB the write is rejected with `vfs_full`. Reading
through to a project file costs nothing against the cap because nothing is
retained — only a write or a successful edit consumes budget.

## Error codes

| Code | When |
|------|------|
| `not_found` | Path resolves in neither layer (`read`, `edit`, `delete`) |
| `vfs_full` | Write would exceed 10 MiB cap |
| `ambiguous` | `old_str` appears more than once (`edit`) |
| `no_files_found` | All requested paths missing (`present`) |
| `unreadable` | Project file above the read limit or not UTF-8 text |
