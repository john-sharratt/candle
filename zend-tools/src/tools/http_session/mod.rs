//! HTTP session tools: `http_session_{open,request,list,close}`.
//!
//! Stateful HTTP client sessions backed by `reqwest`.  Each session maintains a
//! cookie jar and optional auth headers across multiple requests — right for REST
//! APIs, web scraping with login, or any workflow that requires state across
//! calls.  For one-off page fetches, `web_fetch` is simpler.
//!
//! # Authentication
//!
//! Credentials are attached at `http_session_open` and applied to every
//! subsequent request in the session:
//! - `http_bearer` → `Authorization: Bearer <token>`
//! - `http_basic` → `Authorization: Basic <base64(user:pass)>`
//! - `http_header` → arbitrary header name/value pair
//!
//! # Binary response bodies
//!
//! `http_request` returns `body` (UTF-8 string) for text content types
//! and `body_b64` (base64) for binary.  At most one is present per response.
//! HTTP uses base64 (not hex) — that's the established convention for
//! HTTP-over-JSON tooling, and response bodies are usually structured rather than
//! byte-level protocol messages.
//!
//! # Confirmation policy
//!
//! `http_request` confirms for mutating methods (POST, PUT, PATCH, DELETE).
//! GET, HEAD, and OPTIONS do not confirm (read-only by HTTP spec).
//! Open, list, and close do not confirm.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `session_not_found` | Session ID not in registry |
//! | `session_dead` | Underlying client became invalid |
//! | `connection_failed` | Could not reach the host |
//! | `timeout` | Request exceeded configured timeout |
//! | `url_blocked` | Target resolves to private address (SSRF guard) |
//! | `invalid_credential_type` | Credential is not an `http_*` type |
//! | `session_limit_exceeded` | 5-session-per-user cap reached |
//! | `credential_not_found` | Named credential not in store |

use crate::ToolError;
use thiserror::Error;

pub mod close;
pub mod list;
pub mod open;
pub mod request;

pub use close::HTTP_SESSION_CLOSE;
pub use list::HTTP_SESSION_LIST;
pub use open::HTTP_SESSION_OPEN;
pub use request::HTTP_SESSION_REQUEST;

#[derive(Debug, Error)]
pub enum HttpSessionError {
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("session dead")]
    SessionDead,
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("timeout")]
    Timeout,
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("invalid credential type: {0}")]
    InvalidCredentialType(String),
    #[error("session limit exceeded")]
    SessionLimitExceeded,
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
}

impl ToolError for HttpSessionError {
    fn code(&self) -> &'static str {
        match self {
            HttpSessionError::SessionNotFound(_) => "session_not_found",
            HttpSessionError::SessionDead => "session_dead",
            HttpSessionError::ConnectionFailed(_) => "connection_failed",
            HttpSessionError::Timeout => "timeout",
            HttpSessionError::UrlBlocked(_) => "url_blocked",
            HttpSessionError::InvalidCredentialType(_) => "invalid_credential_type",
            HttpSessionError::SessionLimitExceeded => "session_limit_exceeded",
            HttpSessionError::CredentialNotFound(_) => "credential_not_found",
        }
    }
}
