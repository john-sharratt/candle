//! signature_verify tool.

use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature as P256Signature, VerifyingKey as P256VerifyingKey};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SigVerifyRequest {
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub data_encoding: Option<String>,
    /// The signature to check, as hex.
    #[validate(length(min = 1))]
    pub signature_hex: String,
    /// The signer's public key in PEM form, including the
    /// `-----BEGIN PUBLIC KEY-----`/`-----END PUBLIC KEY-----` armor lines.
    #[validate(length(min = 1))]
    pub public_key_pem: String,
    /// Signature scheme. One of: ed25519, p256_sha256.
    #[schemars(with = "super::SignatureAlgorithm")]
    #[validate(length(min = 1))]
    pub algorithm: String,
}

#[derive(Serialize)]
pub struct SigVerifyResponse {
    pub valid: bool,
    pub algorithm: String,
}

pub struct SignatureVerify;

impl Tool for SignatureVerify {
    const NAME: &'static str = "signature_verify";
    const DESCRIPTION: &'static str =
        "Check whether a digital signature over a message was genuinely \
         produced by the holder of a given public key (ed25519 or \
         p256_sha256). Returns whether the signature is authentic.";

    type Request = SigVerifyRequest;
    type Response = SigVerifyResponse;
    type Error = CryptoError;

    fn run(_ctx: &ToolContext, req: SigVerifyRequest) -> Result<SigVerifyResponse, CryptoError> {
        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let data = decode_data(&req.data, enc).map_err(CryptoError::InvalidDataEncoding)?;
        let sig_bytes = hex::decode(&req.signature_hex)
            .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;

        let valid = match normalize_algorithm(&req.algorithm).as_str() {
            "ed25519" => {
                use ed25519_dalek::pkcs8::DecodePublicKey;
                let vk = ed25519_dalek::VerifyingKey::from_public_key_pem(&req.public_key_pem)
                    .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
                let sig = ed25519_dalek::Signature::from_slice(&sig_bytes)
                    .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;
                use ed25519_dalek::Verifier;
                vk.verify(&data, &sig).is_ok()
            }
            "p256sha256" => {
                use p256::pkcs8::DecodePublicKey;
                let vk = P256VerifyingKey::from_public_key_pem(&req.public_key_pem)
                    .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
                let sig = P256Signature::from_slice(&sig_bytes)
                    .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;
                vk.verify(&data, &sig).is_ok()
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        };

        Ok(SigVerifyResponse {
            valid,
            algorithm: req.algorithm,
        })
    }
}

pub const SIGNATURE_VERIFY: RegisteredTool = RegisteredTool::new::<SignatureVerify>();
