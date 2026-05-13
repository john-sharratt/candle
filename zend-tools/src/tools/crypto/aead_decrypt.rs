//! aead_decrypt tool.

use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit};
use chacha20poly1305::ChaCha20Poly1305;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::CryptoError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct AeadDecryptRequest {
    #[validate(length(min = 1))]
    pub ciphertext_hex: String,
    #[validate(length(min = 64, max = 64))]
    pub key_hex: String,
    #[validate(length(min = 24, max = 24))]
    pub nonce_hex: String,
    #[validate(length(min = 1))]
    pub algorithm: String,
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
        "Decrypt data with an AEAD cipher (AES-256-GCM or ChaCha20-Poly1305). \
         Returns plaintext as text if valid UTF-8, otherwise hex.";

    type Request = AeadDecryptRequest;
    type Response = AeadDecryptResponse;
    type Error = CryptoError;

    fn run(_ctx: &ToolContext, req: AeadDecryptRequest) -> Result<AeadDecryptResponse, CryptoError> {
        let key_bytes = hex::decode(&req.key_hex)
            .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
        let nonce_bytes = hex::decode(&req.nonce_hex)
            .map_err(|e| CryptoError::InvalidNonce(e.to_string()))?;
        let ct = hex::decode(&req.ciphertext_hex)
            .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?;
        let aad = req.aad.as_deref().unwrap_or("").as_bytes().to_vec();

        let plaintext = match req.algorithm.as_str() {
            "aes256gcm" => {
                let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
                let cipher = Aes256Gcm::new(key);
                let nonce = Nonce::from_slice(&nonce_bytes);
                let payload = aes_gcm::aead::Payload { msg: &ct, aad: &aad };
                cipher.decrypt(nonce, payload)
                    .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
            }
            "chacha20poly1305" => {
                use chacha20poly1305::KeyInit as _;
                let key = chacha20poly1305::Key::from_slice(&key_bytes);
                let cipher = ChaCha20Poly1305::new(key);
                let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
                let payload = chacha20poly1305::aead::Payload { msg: &ct, aad: &aad };
                cipher.decrypt(nonce, payload)
                    .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?
            }
            other => return Err(CryptoError::InvalidAlgorithm(other.to_string())),
        };

        let (text, encoding) = match std::str::from_utf8(&plaintext) {
            Ok(s) => (s.to_string(), "text"),
            Err(_) => (hex::encode(&plaintext), "hex"),
        };

        Ok(AeadDecryptResponse { plaintext: text, plaintext_encoding: encoding.to_string() })
    }
}

pub const AEAD_DECRYPT: RegisteredTool = RegisteredTool::new::<AeadDecrypt>();
