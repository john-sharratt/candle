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
| `image.rs` | `file_image` | Resized copies of a picture: PNG, JPEG, WebP, multi-size ICO |
| `read.rs` | `file_read` | Read a `part`, a run of `lines` or a page of the `list` of a file's parts — exactly one — each part a fenced block |
| `edit.rs` | `file_edit` | One anchored edit — replace, insert, delete, move — by the anchors `file_read` shows; the engine is `doc-units` |
| `grep.rs` | `file_grep` | Regex over file contents; each hit names its part's anchor, its line in the part and the part's scope |
| `render.rs` | — | The read format, shared with the `code_reading` ingest |
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
paragraph for Markdown, YAML and text. Each part carries a four-character
anchor hashed from its own text, so a part another hand changed no longer has
the anchor it was read under. `doc-units` is the engine.

Each part is shown as a fenced block of its own, its lines numbered from `+1`:

````text
```rust file=candle/npcd/src/engine/act.rs part=1Uax lines=46-52 scope="struct Act"
 +1 /// One act a character took.
 +2 #[derive(Clone, Debug, PartialEq, Serialize)]
 +3 pub struct Act {
…
```
````

## What `file_read` reads

`repo` and `path`, and exactly one of — there is no whole-file read:

- `part` — one part, by its anchor;
- `lines` — `84` or `120-180`: the whole parts those lines touch, up to
  [`PAGE_LINES`](render.rs) lines (200) of them, from the part holding the
  first. A one-line read says which part and line of it that is
  (`line 84 is part=KehT +30`). The read closes `end of <file> lines A-B of N —
  next: lines=C-D`, or `— the end of the file`, so the model reads straight off
  it whether and where to keep going;
- `list` — a page of the file's parts, one line each
  (`part=KehT lines=55-101 impl Act > fn summary`); a file one read holds
  comes back whole instead.

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
