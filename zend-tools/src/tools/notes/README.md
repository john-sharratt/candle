# notes — notes_{write,read,search,list}

Cross-conversation persistent key-value store for agent memory.

## Files

| File | Tool | Description |
|------|------|-------------|
| `write.rs` | `notes_write` | Upsert; empty content tombstones the note |
| `read.rs` | `notes_read` | Exact key lookup |
| `search.rs` | `notes_search` | Full-text + tag search |
| `list.rs` | `notes_list` | Prefix / tag enumeration; metadata only |
| `mod.rs` | — | `NotesError`; `MAX_NOTE_BYTES` (1 MiB) |

## Notes vs VFS

| Aspect | Notes | VFS |
|--------|-------|-----|
| Lifetime | Cross-conversation (persistent) | Single session (in-memory) |
| Purpose | Agent memory (facts, context, decisions) | Working files (code, drafts) |
| Deletion | Write empty content to tombstone | `file_delete` |
| Content type | Structured facts, markdown prose | Any text |

## Key format

Free-form strings, max 256 bytes.  Hierarchical patterns like `infra/dns/internal`
are encouraged for organisation but not enforced.

## Tombstoning

`notes_write` with empty content removes the note.  There is no separate
`notes_delete` — a single write path avoids accidental data loss from a stray
delete call and gives the model one clear way to remove information.

## Search query syntax (FTS5)

`notes_search` uses SQLite FTS5 query syntax:
- `"exact phrase"` — phrase match
- `term1 AND term2` — both required
- `term1 OR term2` — either
- `NOT term` — exclusion
- `term*` — prefix wildcard

At least one of `query` or `tags` must be provided; omitting both returns
`no_search_criteria`.

## Size limits

- Individual note: 1 MiB (`MAX_NOTE_BYTES`)
- Key: 256 bytes

## Error codes

| Code | When |
|------|------|
| `not_found` | Key absent from store (`read`) |
| `note_too_large` | Content exceeds 1 MiB |
| `key_too_long` | Key exceeds 256 bytes |
| `no_search_criteria` | `search` called with neither `query` nor `tags` |
