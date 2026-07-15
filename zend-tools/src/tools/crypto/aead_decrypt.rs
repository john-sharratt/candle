//! aead_decrypt tool.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes128Gcm, Aes256Gcm, Key, Nonce};
use chacha20poly1305::ChaCha20Poly1305;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct AeadDecryptRequest {
    /// Ciphertext-with-tag as hex, exactly as returned by aead_encrypt.
    #[validate(length(min = 1))]
    pub ciphertext_hex: String,
    /// Secret key as hex: 32 hex chars (16 bytes) for aes128gcm; 64 hex chars
    /// (32 bytes) for aes256gcm and chacha20poly1305.
    #[validate(length(min = 32, max = 64))]
    pub key_hex: String,
    /// The 12-byte nonce as 24 hex chars (the `nonce_hex` aead_encrypt returned).
    #[validate(length(min = 24, max = 24))]
    pub nonce_hex: String,
    /// AEAD cipher. One of: aes128gcm, aes256gcm, chacha20poly1305.
    #[schemars(with = "super::AeadAlgorithm")]
    #[validate(length(min = 1))]
    pub algorithm: String,
    /// Optional additional authenticated data (UTF-8 text) — must match what was used to encrypt.
    pub aad: Option<String>,
}

#[derive(Serialize)]
pub struct AeadDecryptResponse {
    pub plaintext: String,
    pub plaintext_encoding: String,
}

pub struct AeadDecrypt;

impl Tool for AeadDecrypt {
    const NAME: &'static str = "aead_decrypt";
    const DESCRIPTION: &'static str =
        "Decrypt and authenticate AEAD ciphertext — aes128gcm, aes256gcm, or \
         chacha20poly1305 — verifying the integrity tag and rejecting any \
         tampered input. Returns recovered plaintext as text when valid \
         UTF-8, otherwise hex.";

    type Request = AeadDecryptRequest;
    type Response = AeadDecryptResponse;
    type Error = CryptoError;

    fn run(
        _ctx: &ToolContext,
        req: AeadDecryptRequest,
    ) -> Result<AeadDecryptResponse, CryptoError> {
        let key_bytes =
            hex::decode(&req.key_hex).map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
        let nonce_bytes =
            hex::decode(&req.nonce_hex).map_err(|e| CryptoError::InvalidNonce(e.to_string()))?;
        let ct = hex::decode(&req.ciphertext_hex)
            .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?;
        let aad = req.aad.as_deref().unwrap_or("").as_bytes().to_vec();

        let plaintext = match normalize_algorithm(&req.algorithm).as_str() {
            "aes128gcm" => {
                if key_bytes.len() != 16 {
                    return Err(CryptoError::InvalidKey(
                        "AES-128-GCM key must be 16 bytes (32 hex chars)".to_string(),
                    ));
                }
                let key = Key::<Aes128Gcm>::from_slice(&key_bytes);
                let cipher = Aes128Gcm::new(key);
                let nonce = Nonce::from_slice(&nonce_bytes);
                let payload = aes_gcm::aead::Payload {
                    msg: &ct,
                    aad: &aad,
                };
                cipher
                    .decrypt(nonce, payload)
                    .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
            }
            "aes256gcm" => {
                if key_bytes.len() != 32 {
                    return Err(CryptoError::InvalidKey(
                        "AES-256-GCM key must be 32 bytes (64 hex chars)".to_string(),
                    ));
                }
                let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
                let cipher = Aes256Gcm::new(key);
                let nonce = Nonce::from_slice(&nonce_bytes);
                let payload = aes_gcm::aead::Payload {
                    msg: &ct,
                    aad: &aad,
                };
                cipher
                    .decrypt(nonce, payload)
                    .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
            }
            "chacha20poly1305" => {
                if key_bytes.len() != 32 {
                    return Err(CryptoError::InvalidKey(
                        "ChaCha20-Poly1305 key must be 32 bytes (64 hex chars)".to_string(),
                    ));
                }
                use chacha20poly1305::KeyInit as _;
                let key = chacha20poly1305::Key::from_slice(&key_bytes);
                let cipher = ChaCha20Poly1305::new(key);
                let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
                let payload = chacha20poly1305::aead::Payload {
                    msg: &ct,
                    aad: &aad,
                };
                cipher
                    .decrypt(nonce, payload)
                    .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
            }
            other => return Err(CryptoError::InvalidAlgorithm(other.to_string())),
        };

        let (text, encoding) = match std::str::from_utf8(&plaintext) {
            Ok(s) => (s.to_string(), "text"),
            Err(_) => (hex::encode(&plaintext), "hex"),
        };

        Ok(AeadDecryptResponse {
            plaintext: text,
            plaintext_encoding: encoding.to_string(),
        })
    }
}

pub const AEAD_DECRYPT: RegisteredTool = RegisteredTool::new::<AeadDecrypt>();
