//! aead_encrypt tool.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng as AeadOsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use chacha20poly1305::ChaCha20Poly1305;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct AeadEncryptRequest {
    pub data: String,
    pub data_encoding: Option<String>,
    #[validate(length(min = 64, max = 64))]
    pub key_hex: String,
    #[validate(length(min = 1))]
    pub algorithm: String,
    pub nonce_hex: Option<String>,
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
         AES-256-GCM or ChaCha20-Poly1305 — giving confidentiality and \
         tamper-detection in one step. Returns ciphertext plus a fresh nonce; \
         key provided as 32-byte hex.";

    type Request = AeadEncryptRequest;
    type Response = AeadEncryptResponse;
    type Error = CryptoError;

    fn run(
        _ctx: &ToolContext,
        req: AeadEncryptRequest,
    ) -> Result<AeadEncryptResponse, CryptoError> {
        let key_bytes =
            hex::decode(&req.key_hex).map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
        if key_bytes.len() != 32 {
            return Err(CryptoError::InvalidKey("key must be 32 bytes".to_string()));
        }

        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let plaintext = decode_data(&req.data, enc).map_err(CryptoError::InvalidDataEncoding)?;
        let aad = req.aad.as_deref().unwrap_or("").as_bytes().to_vec();

        match req.algorithm.as_str() {
            "aes256gcm" => {
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
