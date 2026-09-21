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
| `read.rs` | `file_read` | Return a file, or a line range of it, as a numbered, fenced excerpt; only `path` is required |
| `edit.rs` | `file_edit` | Applies a unified diff; the engine is `patch.rs` |
| `list.rs` | `file_list` | Paged, one-level union listing of a directory's project + session files |
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

## `file_read` paging

`path` and `page` are both required — there is no whole-file read. `page` is
0-based and every page is [`PAGE_LINES`](../../state/vfs.rs) lines
(currently 200); a page past the end clamps to the last one rather than
failing. The header reads `(page P of N, lines a-b of total)`, which names
both the page just returned and the total page count, so the model reads it
straight to know whether to keep going.

## `file_edit` patches

`file_edit` takes a unified diff: one or more `@@` hunks of `' '` context, `-`
removed and `+` added lines. `patch.rs` is the engine, and its module docs are
the reference for the format.

Hunks are located by their **content**, never by the `@@` line numbers. A
hunk's pre-image (its context and removed lines) must match a run of whole
lines exactly — no fuzz, and never a substring of a line. The line numbers are
only a hint for choosing between equal matches:

- pre-image found once, or nearest the hint → applied
- pre-image found in two equally-distant places → `ambiguous`
- pre-image absent but post-image present → **already applied**, nothing written
- neither present → `not_found`, naming the hunk and its `@@` header
- not a readable diff → `invalid_arguments`

Already-applied detection is what makes the tool idempotent: sending the same
patch twice succeeds both times and the second call writes nothing. It is also
why `-retries = 3` / `+retries = 30` cannot produce `retries = 300` — matching
is by whole line, so `retries = 30` is not a second `retries = 3`.

Hunks apply in order, each searched from the end of the one before, so they
cannot overlap. Either every hunk lands or the file is left exactly as it was —
a partially patched file is never written.

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
| `not_found` | Path resolves in neither layer (`read`, `edit`, `delete`), or an `edit` hunk matches nothing |
| `vfs_full` | Write would exceed 10 MiB cap |
| `ambiguous` | An `edit` hunk matches in more than one place |
| `no_files_found` | All requested paths missing (`present`) |
| `unreadable` | Project file above the read limit or not UTF-8 text |
| `invalid_arguments` | A required argument is missing — e.g. `file_read` without `path` — or an `edit` patch is not a readable unified diff |
