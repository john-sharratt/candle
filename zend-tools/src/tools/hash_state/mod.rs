//! Running hash state tools: `hash_state_{init,update,finalize}`.
//!
//! Streaming hash computation for data too large to pass in a single tool call
//! (log files, firmware images, large database exports, chunked uploads).
//!
//! # Workflow
//!
//! 1. `hash_state_init` — create a named context with an algorithm; returns the ID
//! 2. `hash_state_update` — feed data chunks one at a time; repeatable
//! 3. `hash_state_finalize` — produce the final digest and discard the context
//!
//! Contexts are stored in [`crate::state::HashStateStore`] keyed by the ID
//! provided at init.  They live until finalized or the session ends.
//!
//! # Algorithms
//!
//! SHA-256, SHA-512, SHA-1, MD5, SHA3-256, SHA3-512, BLAKE3.
//! The algorithm is fixed at init time; update/finalize use whatever was set.
//!
//! # Data encoding
//!
//! Each `update` call accepts `data` plus an `encoding` parameter (`"text"`,
//! `"hex"`, `"base64"`).  Different chunks can use different encodings in the
//! same context, which is handy when mixing text headers with binary payloads.
//!
//! Shared helper: [`decode_data`]
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `unknown_algorithm` | Algorithm name not recognised |
//! | `id_already_exists` | `hash_state_init` called with a duplicate ID |
//! | `not_found` | Context ID not in store (`update`, `finalize`) |
//! | `invalid_data_encoding` | Bad hex or base64 chunk data |

use crate::ToolError;
use thiserror::Error;

pub mod finalize;
pub mod init;
pub mod update;

pub use finalize::HASH_STATE_FINALIZE;
pub use init::HASH_STATE_INIT;
pub use update::HASH_STATE_UPDATE;

#[derive(Debug, Error)]
pub enum HashStateError {
    #[error("unknown algorithm: {0}")]
    UnknownAlgorithm(String),
    #[error("id already exists: {0}")]
    IdAlreadyExists(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid data encoding: {0}")]
    InvalidDataEncoding(String),
}

impl ToolError for HashStateError {
    fn code(&self) -> &'static str {
        match self {
            HashStateError::UnknownAlgorithm(_) => "unknown_algorithm",
            HashStateError::IdAlreadyExists(_) => "id_already_exists",
            HashStateError::NotFound(_) => "not_found",
            HashStateError::InvalidDataEncoding(_) => "invalid_data_encoding",
        }
    }
}

pub fn decode_data(data: &str, encoding: &str) -> Result<Vec<u8>, HashStateError> {
    match encoding {
        "text" => Ok(data.as_bytes().to_vec()),
        "hex" => hex::decode(data).map_err(|e| HashStateError::InvalidDataEncoding(e.to_string())),
        "base64" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| HashStateError::InvalidDataEncoding(e.to_string()))
        }
        other => Err(HashStateError::InvalidDataEncoding(format!(
            "unknown encoding: {other}"
        ))),
    }
}
