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
| `edit.rs` | `file_edit` | Replaces `old_text` with `new_text`; the engine is `zend-vfs/src/replace/` |
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
0-based and every page is [`PAGE_LINES`](../../../../zend-vfs/src/vfs.rs) lines
(currently 200); a page past the end clamps to the last one rather than
failing. The header reads `(page P of N, lines a-b of total)`, which names
both the page just returned and the total page count, so the model reads it
straight to know whether to keep going.

## `file_edit` replacements

`file_edit` takes `old_text` — the text to change, quoted from the file as it
stands — and `new_text`, what takes its place. `zend-vfs/src/replace/` is the
engine, and its module docs are the reference.

- `old_text` found exactly once → replaced (`"matched": "exact"`); an indented
  `old_text` must start a line
- found more than once (overlapping counted apart) → `ambiguous`, unless
  `replace_all` is set
- not found exactly, but found as whole lines with their indentation ignored →
  replaced, the new text re-indented to the file's (`"matched": "indentation"`)
- absent, with `new_text` standing in the file as whole lines that are not
  only punctuation → **already applied**, nothing written
- neither → `not_found`, saying where the first line of `old_text` is, if it is
- empty `old_text`, or `new_text` the same → `invalid_arguments`

Already-applied detection is what makes the tool idempotent: sending the same
edit twice succeeds both times and the second call writes nothing. An
occurrence of `old_text` inside an occurrence of `new_text` is the edit's own
result and is not counted, which is why `retries = 3` → `retries = 30` cannot
produce `retries = 300`.

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
| `not_found` | Path resolves in neither layer (`read`, `edit`, `delete`), or an `edit`'s `old_text` is not in the file |
| `vfs_full` | Write would exceed 10 MiB cap |
| `ambiguous` | An `edit`'s `old_text` occurs more than once and `replace_all` is not set |
| `no_files_found` | All requested paths missing (`present`) |
| `unreadable` | Project file above the read limit or not UTF-8 text |
| `invalid_arguments` | A required argument is missing — e.g. `file_read` without `path` — or an `edit` whose `old_text` is empty or the same as its `new_text` |
