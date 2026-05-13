# sql_session — sql_session_{open,query,list,close}

Database sessions via `rusqlite` (SQLite).

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `sql_session_open` | Open a SQLite file; optional `sql_password` credential |
| `query.rs` | `sql_session_query` | Execute SQL; return rows as JSON objects |
| `list.rs` | `sql_session_list` | List open sessions |
| `close.rs` | `sql_session_close` | Close the connection |
| `mod.rs` | — | `SqlError` enum |

## DSN format

Currently `sqlite` only.  Pass the file path as the DSN:
- `sqlite:///absolute/path/to/db.sqlite3`
- Or just the file path directly

The `SqlConn` enum is designed for future MySQL/PostgreSQL/MSSQL extension.

## Query results

`sql_session_query` returns rows as a JSON array of objects, keyed by column name.
SQLite type affinity is mapped to JSON:

| SQLite | JSON |
|--------|------|
| TEXT | string |
| INTEGER / REAL | number |
| BLOB | base64 string |
| NULL | null |

## Error codes

| Code | When |
|------|------|
| `session_not_found` | Session ID not in registry |
| `credential_not_found` | Named credential absent from store |
| `invalid_credential_type` | Credential is not `sql_password` |
| `not_supported` | DSN scheme is not `sqlite` |
| `connection_failed` | Cannot open or create the database file |
| `query_failed` | SQL execution error (syntax, constraint, missing table…) |
| `session_limit_exceeded` | 5-session-per-user cap |
