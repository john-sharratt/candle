//! Remote filesystem session tools: `remote_fs_session_*` (10 tools).
//!
//! Protocol-agnostic file operations accessed via URI scheme.  The model picks
//! an operation; the URI carries the protocol.  This design trades a small amount
//! of protocol-specific expressiveness (e.g. FTP transfer mode) for a much
//! smaller surface and uniform semantics across protocols.
//!
//! # Supported protocols
//!
//! Currently `sftp://` (via `ssh2`).  The `RemoteFsConn` struct and error codes
//! are designed for future FTP/SMB/NFS extension.
//!
//! # URI format
//!
//! `sftp://<host>:<port>/<path>`
//!
//! Files downloaded via `remote_fs_session_get` are written into the session VFS
//! so the model can then inspect them with `file_read` or edit them with
//! `file_edit` before uploading back.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `session_not_found` | Session ID not in registry |
//! | `credential_not_found` | Named credential not in store |
//! | `not_supported` | URI scheme is not `sftp` |
//! | `connection_failed` | TCP connect or SSH handshake error |
//! | `auth_failed` | Wrong credential for SFTP auth |
//! | `sftp_error` | SFTP protocol error (read, write, stat, etc.) |
//! | `not_found` | Remote file or directory does not exist |
//! | `vfs_error` | Error writing downloaded content to the local VFS |
//! | `session_limit_exceeded` | 5-session-per-user cap reached |
//!
//! # Confirmation policy
//!
//! `remote_fs_session_open` confirms once.  Write operations (`put`, `delete`,
//! `mkdir`, `rename`) confirm every call.  Read operations (`get`, `list_dir`,
//! `stat`) and management operations (`list`, `close`) do not confirm.

use crate::ToolError;
use thiserror::Error;

pub mod close;
pub mod delete;
pub mod get;
pub mod list;
pub mod list_dir;
pub mod mkdir;
pub mod open;
pub mod put;
pub mod rename;
pub mod stat;

pub use close::REMOTE_FS_SESSION_CLOSE;
pub use delete::REMOTE_FS_SESSION_DELETE;
pub use get::REMOTE_FS_SESSION_GET;
pub use list::REMOTE_FS_SESSION_LIST;
pub use list_dir::REMOTE_FS_SESSION_LIST_DIR;
pub use mkdir::REMOTE_FS_SESSION_MKDIR;
pub use open::REMOTE_FS_SESSION_OPEN;
pub use put::REMOTE_FS_SESSION_PUT;
pub use rename::REMOTE_FS_SESSION_RENAME;
pub use stat::REMOTE_FS_SESSION_STAT;

#[derive(Debug, Error)]
pub enum RemoteFsError {
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
    #[error("protocol not supported: {0} — use sftp://")]
    NotSupported(String),
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("auth failed: {0}")]
    AuthFailed(String),
    #[error("sftp error: {0}")]
    SftpError(String),
    #[error("file not found: {0}")]
    NotFound(String),
    #[error("vfs error: {0}")]
    VfsError(String),
    #[error("session limit exceeded")]
    SessionLimitExceeded,
}

impl ToolError for RemoteFsError {
    fn code(&self) -> &'static str {
        match self {
            RemoteFsError::SessionNotFound(_) => "session_not_found",
            RemoteFsError::CredentialNotFound(_) => "credential_not_found",
            RemoteFsError::NotSupported(_) => "not_supported",
            RemoteFsError::ConnectionFailed(_) => "connection_failed",
            RemoteFsError::AuthFailed(_) => "auth_failed",
            RemoteFsError::SftpError(_) => "sftp_error",
            RemoteFsError::NotFound(_) => "not_found",
            RemoteFsError::VfsError(_) => "vfs_error",
            RemoteFsError::SessionLimitExceeded => "session_limit_exceeded",
        }
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}
