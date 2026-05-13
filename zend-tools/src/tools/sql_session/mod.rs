//! SQL session tools: `sql_session_{open,query,list,close}`.
//!
//! Database sessions backed by `rusqlite` (SQLite).  Each session holds an open
//! `rusqlite::Connection` and executes queries on demand.
//!
//! # Supported databases
//!
//! Currently SQLite only.  The DSN format is a file path
//! (e.g. `sqlite:///path/to/db.sqlite3` or just the path).  MySQL, PostgreSQL,
//! and MSSQL are future work — the `SqlConn` enum is designed for extension.
//!
//! # Query results
//!
//! `sql_session_query` returns rows as an array of JSON objects keyed by column
//! name.  All SQL values are coerced to JSON types (TEXT → string, INTEGER/REAL
//! → number, BLOB → base64 string, NULL → null).
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `session_not_found` | Session ID not in registry |
//! | `credential_not_found` | Named credential not in store |
//! | `invalid_credential_type` | Credential is not `sql_password` |
//! | `not_supported` | DSN scheme is not `sqlite` |
//! | `connection_failed` | Could not open the database file |
//! | `query_failed` | SQL execution error (syntax, constraint, etc.) |
//! | `session_limit_exceeded` | 5-session-per-user cap reached |

use thiserror::Error;
use crate::ToolError;

pub mod open;
pub mod query;
pub mod list;
pub mod close;

pub use open::SQL_SESSION_OPEN;
pub use query::SQL_SESSION_QUERY;
pub use list::SQL_SESSION_LIST;
pub use close::SQL_SESSION_CLOSE;

#[derive(Debug, Error)]
pub enum SqlError {
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
    #[error("invalid credential type: expected sql_password")]
    InvalidCredentialType,
    #[error("database driver not supported for DSN: {0}")]
    NotSupported(String),
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("query failed: {0}")]
    QueryFailed(String),
    #[error("session limit exceeded")]
    SessionLimitExceeded,
}

impl ToolError for SqlError {
    fn code(&self) -> &'static str {
        match self {
            SqlError::SessionNotFound(_) => "session_not_found",
            SqlError::CredentialNotFound(_) => "credential_not_found",
            SqlError::InvalidCredentialType => "invalid_credential_type",
            SqlError::NotSupported(_) => "not_supported",
            SqlError::ConnectionFailed(_) => "connection_failed",
            SqlError::QueryFailed(_) => "query_failed",
            SqlError::SessionLimitExceeded => "session_limit_exceeded",
        }
    }
}
