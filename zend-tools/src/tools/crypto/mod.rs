//! Cryptographic primitive tools (8 tools).
//!
//! Pure-Rust cryptography via the RustCrypto family of crates.  These tools
//! give the model access to standard primitives for protocol implementation,
//! key derivation, signature verification, and authenticated encryption.
//!
//! # Tools
//!
//! | Tool | Primitive | Crates |
//! |------|-----------|--------|
//! | `aead_encrypt` | AES-128/256-GCM, ChaCha20-Poly1305 | `aes-gcm`, `chacha20poly1305` |
//! | `aead_decrypt` | same | same |
//! | `hmac_compute` | HMAC-SHA256/512 | `hmac`, `sha2` |
//! | `signature_sign` | Ed25519, ECDSA-P256 | `ed25519-dalek`, `p256` |
//! | `signature_verify` | Ed25519, ECDSA-P256 | same |
//! | `kdf_derive` | Argon2id, PBKDF2-SHA256, scrypt | `argon2`, `pbkdf2`, `scrypt` |
//! | `hkdf_extract` | HKDF-Extract (RFC 5869) | `hkdf`, `sha2` |
//! | `hkdf_expand_label` | HKDF-Expand-Label (TLS 1.3 RFC 8446 §7.1) | same |
//!
//! # Data encoding
//!
//! All binary inputs and outputs (keys, nonces, ciphertext, digests, signatures)
//! are encoded as strings.  The `encoding` parameter accepts `"hex"`, `"base64"`,
//! or `"text"` (UTF-8 bytes).  Default is `"hex"` for most tools.
//!
//! Shared helpers:
//! - [`decode_data`] — decode a string given an encoding name
//! - [`encode_output`] — encode bytes to hex (default) or base64
//!
//! # Key material and credentials
//!
//! `signature_sign` accepts either an inline PEM key or a `credential_id`
//! referencing a `signing_key` credential.  The algorithm is supplied per call
//! so one key can serve multiple signature schemes.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `invalid_algorithm` | Unsupported algorithm name |
//! | `invalid_key` | Key material cannot be parsed |
//! | `invalid_nonce` | Nonce has wrong length or encoding |
//! | `encryption_failed` | AEAD encrypt error |
//! | `decryption_failed` | Authentication tag mismatch or decrypt error |
//! | `unknown_algorithm` | Algorithm name not recognised |
//! | `invalid_data_encoding` | Bad hex/base64 input |
//! | `signing_failed` | Private key cannot produce a signature |
//! | `credential_not_found` | Named credential not in store |
//! | `invalid_credential_type` | Credential is not `signing_key` |
//! | `derivation_failed` | KDF parameter error |
//! | `expand_failed` | HKDF expand error |

use crate::ToolError;
use thiserror::Error;

pub mod aead_decrypt;
pub mod aead_encrypt;
pub mod hkdf_expand_label;
pub mod hkdf_extract;
pub mod hmac_compute;
pub mod kdf_derive;
pub mod signature_sign;
pub mod signature_verify;

pub use aead_decrypt::AEAD_DECRYPT;
pub use aead_encrypt::AEAD_ENCRYPT;
pub use hkdf_expand_label::HKDF_EXPAND_LABEL;
pub use hkdf_extract::HKDF_EXTRACT;
pub use hmac_compute::HMAC_COMPUTE;
pub use kdf_derive::KDF_DERIVE;
pub use signature_sign::SIGNATURE_SIGN;
pub use signature_verify::SIGNATURE_VERIFY;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("invalid algorithm: {0}")]
    InvalidAlgorithm(String),
    #[error("invalid key: {0}")]
    InvalidKey(String),
    #[error("invalid nonce: {0}")]
    InvalidNonce(String),
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),
    #[error("decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("unknown algorithm: {0}")]
    UnknownAlgorithm(String),
    #[error("invalid data encoding: {0}")]
    InvalidDataEncoding(String),
    #[error("signing failed: {0}")]
    SigningFailed(String),
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
    #[error("invalid credential type: {0}")]
    InvalidCredentialType(String),
    #[error("derivation failed: {0}")]
    DerivationFailed(String),
    #[error("expand failed: {0}")]
    ExpandFailed(String),
}

impl ToolError for CryptoError {
    fn code(&self) -> &'static str {
        match self {
            CryptoError::InvalidAlgorithm(_) => "invalid_algorithm",
            CryptoError::InvalidKey(_) => "invalid_key",
            CryptoError::InvalidNonce(_) => "invalid_nonce",
            CryptoError::EncryptionFailed(_) => "encryption_failed",
            CryptoError::DecryptionFailed(_) => "decryption_failed",
            CryptoError::UnknownAlgorithm(_) => "unknown_algorithm",
            CryptoError::InvalidDataEncoding(_) => "invalid_data_encoding",
            CryptoError::SigningFailed(_) => "signing_failed",
            CryptoError::CredentialNotFound(_) => "credential_not_found",
            CryptoError::InvalidCredentialType(_) => "invalid_credential_type",
            CryptoError::DerivationFailed(_) => "derivation_failed",
            CryptoError::ExpandFailed(_) => "expand_failed",
        }
    }
}

pub fn decode_data(data: &str, encoding: &str) -> Result<Vec<u8>, String> {
    match encoding {
        "text" => Ok(data.as_bytes().to_vec()),
        "hex" => hex::decode(data).map_err(|e| e.to_string()),
        "base64" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| e.to_string())
        }
        other => Err(format!("unknown encoding: {other}")),
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
