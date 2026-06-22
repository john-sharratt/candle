//! Telnet session tools: `telnet_session_{open,send,list,close}`.
//!
//! Telnet sessions are raw TCP streams with optional IAC negotiation.  They are
//! the right tool for legacy network equipment (switches, routers, serial
//! consoles) that lacks SSH.  Use `ssh_session_*` for any modern host that
//! supports SSH.
//!
//! # Send semantics
//!
//! `telnet_send` writes the `send` string to the stream and then reads
//! until the optional `expect` regex matches or `timeout_sec` elapses.  The
//! response includes the full received text, a `matched` flag, and a
//! `received_truncated` flag (cap: 32 KiB).
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `connection_failed` | TCP connect error |
//! | `session_not_found` | Session ID not in registry |
//! | `send_failed` | Write to stream failed |
//! | `timeout` | Timeout elapsed before `expect` matched |
//!
//! # Confirmation policy
//!
//! `telnet_send` confirms every call.  Open, list, and close do not.

use crate::ToolError;
use thiserror::Error;

pub mod close;
pub mod list;
pub mod open;
pub mod send;

pub use close::TELNET_SESSION_CLOSE;
pub use list::TELNET_SESSION_LIST;
pub use open::TELNET_SESSION_OPEN;
pub use send::TELNET_SESSION_SEND;

#[derive(Debug, Error)]
pub enum TelnetError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("send failed: {0}")]
    SendFailed(String),
    #[error("timeout")]
    Timeout,
}

impl ToolError for TelnetError {
    fn code(&self) -> &'static str {
        match self {
            TelnetError::ConnectionFailed(_) => "connection_failed",
            TelnetError::SessionNotFound(_) => "session_not_found",
            TelnetError::SendFailed(_) => "send_failed",
            TelnetError::Timeout => "timeout",
        }
    }
}
