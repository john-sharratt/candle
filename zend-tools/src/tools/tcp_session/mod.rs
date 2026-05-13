//! TCP session tools: `tcp_session_{open,send,recv,list,close}`.
//!
//! Raw TCP sessions for byte-level protocol work — TLS handshake debugging,
//! off-spec server behaviour, custom binary protocols, or any situation where
//! the model needs full control over the wire bytes.  For TLS-protected services
//! where the model talks to the application above the encryption, use
//! `tls_session_*` instead.  For HTTP, use `http_session_*`.
//!
//! # Wire format
//!
//! TCP uses hex for non-text payloads.  Hex stays readable when the model is
//! reasoning about protocol bytes: `16 03 01` is a TLS record header the model
//! recognises; the same bytes in base64 (`FgMB`) are not.
//!
//! - **Send**: pass `data` (text) or `data_hex` (hex bytes); if both are given,
//!   `data_hex` wins.  Hex accepts whitespace and is case-insensitive.
//! - **Recv**: `format` parameter is `"auto"` | `"hex"` | `"text"`.  In `auto`
//!   mode, valid printable UTF-8 returns `data`; anything else returns `data_hex`.
//!
//! # Recv modes
//!
//! Exactly one of `recv_amt` or `recv_wait` must be provided:
//! - `recv_amt`: read exactly this many bytes (blocks until satisfied or EOF)
//! - `recv_wait`: read whatever arrives within this many seconds
//!
//! Providing neither returns `missing_recv_mode`; providing both returns
//! `conflicting_recv_modes`.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `connection_failed` | TCP connect error |
//! | `session_not_found` | Session ID not in registry |
//! | `url_blocked` | Target resolves to private/loopback address (SSRF guard) |
//! | `send_failed` | Write to stream failed |
//! | `recv_failed` | Read from stream failed |
//! | `missing_recv_mode` | Neither `recv_amt` nor `recv_wait` was provided |
//! | `conflicting_recv_modes` | Both `recv_amt` and `recv_wait` were provided |
//!
//! # Confirmation policy
//!
//! `tcp_session_send` confirms every call.  Open, recv, list, and close do not.

use thiserror::Error;
use crate::ToolError;

pub mod open;
pub mod send;
pub mod recv;
pub mod list;
pub mod close;

pub use open::TCP_SESSION_OPEN;
pub use send::TCP_SESSION_SEND;
pub use recv::TCP_SESSION_RECV;
pub use list::TCP_SESSION_LIST;
pub use close::TCP_SESSION_CLOSE;

#[derive(Debug, Error)]
pub enum TcpError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("send failed: {0}")]
    SendFailed(String),
    #[error("recv failed: {0}")]
    RecvFailed(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("neither recv_amt nor recv_wait was specified; provide exactly one")]
    MissingRecvMode,
    #[error("both recv_amt and recv_wait were specified; provide exactly one")]
    ConflictingRecvModes,
}

impl ToolError for TcpError {
    fn code(&self) -> &'static str {
        match self {
            TcpError::ConnectionFailed(_) => "connection_failed",
            TcpError::SessionNotFound(_) => "session_not_found",
            TcpError::UrlBlocked(_) => "url_blocked",
            TcpError::SendFailed(_) => "send_failed",
            TcpError::RecvFailed(_) => "recv_failed",
            TcpError::InvalidParams(_) => "invalid_params",
            TcpError::MissingRecvMode => "missing_recv_mode",
            TcpError::ConflictingRecvModes => "conflicting_recv_modes",
        }
    }
}
