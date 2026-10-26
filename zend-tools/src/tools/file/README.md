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
| `read.rs` | `file_read` | Return a page of a file — its parts under their anchors — in a fenced block; `repo`, `path` and `page` are required, `line` optional |
| `edit.rs` | `file_edit` | One anchored edit — replace, insert, delete, move — by the anchors `file_read` shows; the engine is `doc-units` |
| `grep.rs` | `file_grep` | Regex over file contents; each hit names its part's anchor and the page holding it |
| `render.rs` | — | The page format, shared with the `code_reading` ingest |
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

## Parts and anchors

A file is shown and edited in its parts (`docs/npcd_anchored_edits.md`): a
function, a type or a block of imports for code (carved by tree-sitter), a
paragraph for Markdown and text, a line for YAML. Each part carries a
four-character anchor hashed from its own text, so a part another hand
changed no longer has the anchor it was read under. `doc-units` is the engine.

## `file_read` paging

`repo`, `path` and `page` are required — there is no whole-file read. `page`
counts from 1 and every page holds whole parts, up to
[`PAGE_LINES`](render.rs) lines (200) of them; a page past the end clamps to
the last one rather than failing, and page 0 is refused. `line` reads the page
holding that line instead. The fence opens `page=P pages=N lines=L` and closes
`end of <file> page P of N — next: page=P+1`, or `— the last page`, so the
model reads straight off it whether and where to keep going.

## `file_edit` operations

`file_edit` names a part by its anchor (`at`) and says what to `do`:

- `replace` — the part with `text`; with `old`, just those words inside it,
  which must occur in it once (`ambiguous` / `not_found` otherwise)
- `insert_before` / `insert_after` — a new part `text` beside it, set apart
  by a blank line where the file sets its parts apart
- `delete` — the part, and the separator after it
- `move` — the part, `before`/`after` the part `dest`, text unchanged
- `to` — extends replace, delete and move over `at` through `to`

An anchor the file no longer has is `stale`, so a replace or delete re-sent
after it landed is refused rather than applied twice — `retries = 3` →
`retries = 30` cannot produce `retries = 300`. An insert leaves its anchor as
it was, so it is the one edit a resumed turn may not re-issue (`replay`). The
reply names what was done and shows the parts it wrote under their new
anchors, between their neighbours.

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
| `not_found` | Path resolves in neither layer (`read`, `edit`, `delete`), or an `edit`'s `old` is not in its part |
| `stale` | An `edit`'s anchor names no part of the file as it stands — read it again |
| `vfs_full` | Write would exceed 10 MiB cap |
| `ambiguous` | An `edit`'s `old` occurs in its part more than once |
| `no_files_found` | All requested paths missing (`present`) |
| `unreadable` | Project file above the read limit or not UTF-8 text |
| `invalid_arguments` | A required argument is missing — e.g. `file_read` without `path` — or an `edit` that names nothing to do: no `text`, a span run backwards, a move into itself |
