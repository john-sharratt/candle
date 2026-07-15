//! aead_encrypt tool.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng as AeadOsRng};
use aes_gcm::{Aes128Gcm, Aes256Gcm, Key, Nonce};
use chacha20poly1305::ChaCha20Poly1305;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct AeadEncryptRequest {
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub data_encoding: Option<String>,
    /// Secret key as hex: 32 hex chars (16 bytes) for aes128gcm; 64 hex chars
    /// (32 bytes) for aes256gcm and chacha20poly1305.
    #[validate(length(min = 32, max = 64))]
    pub key_hex: String,
    /// AEAD cipher. One of: aes128gcm, aes256gcm, chacha20poly1305.
    #[schemars(with = "super::AeadAlgorithm")]
    #[validate(length(min = 1))]
    pub algorithm: String,
    /// Optional 12-byte nonce as 24 hex chars. A fresh random nonce is generated if omitted.
    pub nonce_hex: Option<String>,
    /// Optional additional authenticated data (UTF-8 text), bound to the tag but not encrypted.
    pub aad: Option<String>,
}

#[derive(Serialize)]
pub struct AeadEncryptResponse {
    pub ciphertext_hex: String,
    pub nonce_hex: String,
    pub algorithm: String,
}

pub struct AeadEncrypt;

impl Tool for AeadEncrypt {
    const NAME: &'static str = "aead_encrypt";
    const DESCRIPTION: &'static str =
        "Encrypt plaintext into authenticated ciphertext with an AEAD cipher — \
         aes128gcm, aes256gcm, or chacha20poly1305 — giving confidentiality and \
         tamper-detection in one step. Returns ciphertext plus a fresh nonce; \
         key as hex (16 bytes for aes128gcm, else 32 bytes).";

    type Request = AeadEncryptRequest;
    type Response = AeadEncryptResponse;
    type Error = CryptoError;

    fn run(
        _ctx: &ToolContext,
        req: AeadEncryptRequest,
    ) -> Result<AeadEncryptResponse, CryptoError> {
        let key_bytes =
            hex::decode(&req.key_hex).map_err(|e| CryptoError::InvalidKey(e.to_string()))?;

        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let plaintext = decode_data(&req.data, enc).map_err(CryptoError::InvalidDataEncoding)?;
        let aad = req.aad.as_deref().unwrap_or("").as_bytes().to_vec();

        match normalize_algorithm(&req.algorithm).as_str() {
            "aes128gcm" => {
                if key_bytes.len() != 16 {
                    return Err(CryptoError::InvalidKey(
                        "AES-128-GCM key must be 16 bytes (32 hex chars)".to_string(),
                    ));
                }
                let key = Key::<Aes128Gcm>::from_slice(&key_bytes);
                let cipher = Aes128Gcm::new(key);
                let nonce_bytes = if let Some(n) = &req.nonce_hex {
                    hex::decode(n).map_err(|e| CryptoError::InvalidNonce(e.to_string()))?
                } else {
                    Aes128Gcm::generate_nonce(&mut AeadOsRng).to_vec()
                };
                if nonce_bytes.len() != 12 {
                    return Err(CryptoError::InvalidNonce(
                        "nonce must be 12 bytes".to_string(),
                    ));
                }
                let nonce = Nonce::from_slice(&nonce_bytes);
                let payload = aes_gcm::aead::Payload {
                    msg: &plaintext,
                    aad: &aad,
                };
                let ct = cipher
                    .encrypt(nonce, payload)
                    .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;
                Ok(AeadEncryptResponse {
                    ciphertext_hex: hex::encode(&ct),
                    nonce_hex: hex::encode(&nonce_bytes),
                    algorithm: req.algorithm,
                })
            }
            "aes256gcm" => {
                if key_bytes.len() != 32 {
                    return Err(CryptoError::InvalidKey(
                        "AES-256-GCM key must be 32 bytes (64 hex chars)".to_string(),
                    ));
                }
                let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
                let cipher = Aes256Gcm::new(key);
                let nonce_bytes = if let Some(n) = &req.nonce_hex {
                    hex::decode(n).map_err(|e| CryptoError::InvalidNonce(e.to_string()))?
                } else {
                    Aes256Gcm::generate_nonce(&mut AeadOsRng).to_vec()
                };
                if nonce_bytes.len() != 12 {
                    return Err(CryptoError::InvalidNonce(
                        "nonce must be 12 bytes".to_string(),
                    ));
                }
                let nonce = Nonce::from_slice(&nonce_bytes);
                let payload = aes_gcm::aead::Payload {
                    msg: &plaintext,
                    aad: &aad,
                };
                let ct = cipher
                    .encrypt(nonce, payload)
                    .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;
                Ok(AeadEncryptResponse {
                    ciphertext_hex: hex::encode(&ct),
                    nonce_hex: hex::encode(&nonce_bytes),
                    algorithm: req.algorithm,
                })
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
                let nonce_bytes = if let Some(n) = &req.nonce_hex {
                    hex::decode(n).map_err(|e| CryptoError::InvalidNonce(e.to_string()))?
                } else {
                    ChaCha20Poly1305::generate_nonce(&mut AeadOsRng).to_vec()
                };
                if nonce_bytes.len() != 12 {
                    return Err(CryptoError::InvalidNonce(
                        "nonce must be 12 bytes".to_string(),
                    ));
                }
                let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
                let payload = chacha20poly1305::aead::Payload {
                    msg: &plaintext,
                    aad: &aad,
                };
                let ct = cipher
                    .encrypt(nonce, payload)
                    .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;
                Ok(AeadEncryptResponse {
                    ciphertext_hex: hex::encode(&ct),
                    nonce_hex: hex::encode(&nonce_bytes),
                    algorithm: req.algorithm,
                })
            }
            other => Err(CryptoError::InvalidAlgorithm(other.to_string())),
        }
    }
}

pub const AEAD_ENCRYPT: RegisteredTool = RegisteredTool::new::<AeadEncrypt>();
