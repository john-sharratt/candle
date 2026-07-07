//! Hash tools: `hash_compute` and `hash_scan`.
//!
//! # `hash_compute`
//!
//! Compute a cryptographic digest of a string or binary value.  Supports
//! SHA-256, SHA-512, SHA-1, MD5, SHA3-256, SHA3-512, and BLAKE3.  Input can be
//! encoded as `"text"`, `"hex"`, or `"base64"`; output can be `"hex"` or
//! `"base64"`.
//!
//! # `hash_scan`
//!
//! Given a hex or base64 digest, determine which hash algorithm produced it by
//! trying all known algorithms against the provided pre-image.  Useful for
//! identifying unknown digests in protocol captures or firmware headers.
//!
//! # Shared utilities
//!
//! - [`compute_hash`] — compute a digest for a given algorithm name
//! - [`decode_data`] — decode input bytes given an encoding name
//! - [`encode_output`] — encode digest bytes to hex or base64
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `unknown_algorithm` | Algorithm name not in the supported set |
//! | `invalid_data_encoding` | Bad hex or base64 input |
//! | `no_match` | No algorithm produced the given digest (`hash_scan`) |

use crate::ToolError;
use schemars::JsonSchema;
use serde::Serialize;
use thiserror::Error;

pub mod compute;
pub mod scan;

pub use compute::HASH_COMPUTE;
pub use scan::HASH_SCAN;

// Schema-only mirror of the digest names `compute_hash` accepts. Request fields
// stay `String`; referenced via `#[schemars(with = "…")]` so the JSON schema
// carries a real `"enum"` — a stronger signal than a doc line (which the model
// may override with a prior that, e.g., MD5 is unavailable).
/// Hash algorithm.
#[derive(JsonSchema, Serialize)]
pub enum HashAlgorithm {
    #[serde(rename = "sha256")]
    Sha256,
    #[serde(rename = "sha512")]
    Sha512,
    #[serde(rename = "sha1")]
    Sha1,
    #[serde(rename = "md5")]
    Md5,
    #[serde(rename = "sha3_256")]
    Sha3_256,
    #[serde(rename = "sha3_512")]
    Sha3_512,
    #[serde(rename = "blake3")]
    Blake3,
}

#[derive(Debug, Error)]
pub enum HashError {
    #[error("unknown algorithm: {0}")]
    UnknownAlgorithm(String),
    #[error("invalid data encoding: {0}")]
    InvalidDataEncoding(String),
    #[error("no matching algorithm found")]
    NoMatch,
}

impl ToolError for HashError {
    fn code(&self) -> &'static str {
        match self {
            HashError::UnknownAlgorithm(_) => "unknown_algorithm",
            HashError::InvalidDataEncoding(_) => "invalid_data_encoding",
            HashError::NoMatch => "no_match",
        }
    }
}

pub fn decode_data(data: &str, encoding: &str) -> Result<Vec<u8>, HashError> {
    match encoding {
        "text" => Ok(data.as_bytes().to_vec()),
        "hex" => hex::decode(data).map_err(|e| HashError::InvalidDataEncoding(e.to_string())),
        "base64" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| HashError::InvalidDataEncoding(e.to_string()))
        }
        other => Err(HashError::InvalidDataEncoding(format!(
            "unknown encoding: {other}"
        ))),
    }
}

pub fn encode_output(bytes: &[u8], encoding: &str) -> String {
    match encoding {
        "base64" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(bytes)
        }
        _ => hex::encode(bytes),
    }
}

pub fn compute_hash(data: &[u8], algo: &str) -> Result<Vec<u8>, HashError> {
    use digest::Digest;
    // Canonicalize: lowercase, dropping non-alphanumerics, so "MD5", "SHA-256",
    // and "SHA3-256" all match. `sha3_256`/`sha3_512` normalize to `sha3256`/`sha3512`.
    let norm: String = algo
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    Ok(match norm.as_str() {
        "sha256" => sha2::Sha256::digest(data).to_vec(),
        "sha512" => sha2::Sha512::digest(data).to_vec(),
        "sha1" => sha1::Sha1::digest(data).to_vec(),
        "md5" => md5::Md5::digest(data).to_vec(),
        "sha3256" => sha3::Sha3_256::digest(data).to_vec(),
        "sha3512" => sha3::Sha3_512::digest(data).to_vec(),
        "blake3" => blake3::hash(data).as_bytes().to_vec(),
        other => return Err(HashError::UnknownAlgorithm(other.to_string())),
    })
}
