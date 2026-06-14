//! TLS session tools: `tls_session_{open,send,recv,list,close}`.
//!
//! TLS-encrypted sessions for non-HTTP services: LDAPS, IMAPS, SMTPS, MQTTS,
//! database wire protocols over TLS, or any custom application protocol that
//! runs above TLS.  The model talks to the application layer — TLS is transparent.
//!
//! For byte-level control of the TLS handshake itself (debugging, off-spec
//! counterparty behaviour), use `tcp_session_*` with the crypto primitives
//! instead.
//!
//! # Mutual TLS
//!
//! Pass a `tls_client_cert` credential ID in `credential_id` to enable mTLS.
//! The secret must be a PEM bundle containing both the certificate chain and
//! the private key.
//!
//! # Wire format
//!
//! TLS sessions use the same send/recv wire format as TCP sessions (hex for
//! non-text payloads, `data` vs `data_hex`).
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `connection_failed` | TCP connect error |
//! | `handshake_failed` | TLS handshake rejected (cert mismatch, protocol error) |
//! | `session_not_found` | Session ID not in registry |
//! | `send_failed` | Write to TLS stream failed |
//! | `recv_failed` | Read from TLS stream failed |
//! | `invalid_params` | Malformed address or invalid credential type |
//! | `credential_not_found` | Named credential not in store |

use crate::ToolError;
use thiserror::Error;

pub mod close;
pub mod list;
pub mod open;
pub mod recv;
pub mod send;

pub use close::TLS_SESSION_CLOSE;
pub use list::TLS_SESSION_LIST;
pub use open::TLS_SESSION_OPEN;
pub use recv::TLS_SESSION_RECV;
pub use send::TLS_SESSION_SEND;

#[derive(Debug, Error)]
pub enum TlsError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("TLS handshake failed: {0}")]
    HandshakeFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("send failed: {0}")]
    SendFailed(String),
    #[error("recv failed: {0}")]
    RecvFailed(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
}

impl ToolError for TlsError {
    fn code(&self) -> &'static str {
        match self {
            TlsError::ConnectionFailed(_) => "connection_failed",
            TlsError::HandshakeFailed(_) => "handshake_failed",
            TlsError::SessionNotFound(_) => "session_not_found",
            TlsError::SendFailed(_) => "send_failed",
            TlsError::RecvFailed(_) => "recv_failed",
            TlsError::InvalidParams(_) => "invalid_params",
            TlsError::CredentialNotFound(_) => "credential_not_found",
        }
    }
}
