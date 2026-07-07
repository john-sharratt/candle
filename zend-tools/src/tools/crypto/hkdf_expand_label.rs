//! hkdf_expand_label tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HkdfExpandLabelRequest {
    /// Pseudorandom key as hex (the `prk_hex` from hkdf_extract).
    #[validate(length(min = 1))]
    pub prk_hex: String,
    /// The bare TLS 1.3 label, e.g. "c hs traffic" or "iv". The "tls13 " prefix
    /// is prepended automatically — do not include it.
    #[validate(length(min = 1))]
    pub label: String,
    /// Optional context (e.g. a transcript hash) as hex. Empty if omitted.
    pub context_hex: Option<String>,
    /// Output length in bytes (1–8160).
    #[validate(range(min = 1, max = 8160))]
    pub length: u32,
    /// HKDF hash function. One of: sha256, sha512.
    #[schemars(with = "super::HkdfHash")]
    #[validate(length(min = 1))]
    pub algorithm: String,
}

#[derive(Serialize)]
pub struct HkdfExpandLabelResponse {
    pub okm_hex: String,
    pub algorithm: String,
    pub length: u32,
}

pub struct HkdfExpandLabel;

impl Tool for HkdfExpandLabel {
    const NAME: &'static str = "hkdf_expand_label";
    const DESCRIPTION: &'static str = "HKDF-Expand-Label from the TLS 1.3 key schedule: expand a \
         pseudorandom key into named traffic secrets using a labeled, \
         versioned context. Use for TLS 1.3 key-schedule analysis.";

    type Request = HkdfExpandLabelRequest;
    type Response = HkdfExpandLabelResponse;
    type Error = CryptoError;

    fn run(
        _ctx: &ToolContext,
        req: HkdfExpandLabelRequest,
    ) -> Result<HkdfExpandLabelResponse, CryptoError> {
        let prk = hex::decode(&req.prk_hex).map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
        let ctx = if let Some(c) = &req.context_hex {
            hex::decode(c).map_err(|e| CryptoError::InvalidDataEncoding(e.to_string()))?
        } else {
            vec![]
        };

        let label_with_prefix = format!("tls13 {}", req.label);
        let label_bytes = label_with_prefix.as_bytes();

        let mut info = Vec::new();
        info.extend_from_slice(&(req.length as u16).to_be_bytes());
        info.push(label_bytes.len() as u8);
        info.extend_from_slice(label_bytes);
        info.push(ctx.len() as u8);
        info.extend_from_slice(&ctx);

        let mut okm = vec![0u8; req.length as usize];
        match normalize_algorithm(&req.algorithm).as_str() {
            "sha256" => {
                let hk = hkdf::Hkdf::<sha2::Sha256>::from_prk(&prk)
                    .map_err(|e| CryptoError::ExpandFailed(e.to_string()))?;
                hk.expand(&info, &mut okm)
                    .map_err(|e| CryptoError::ExpandFailed(e.to_string()))?;
            }
            "sha512" => {
                let hk = hkdf::Hkdf::<sha2::Sha512>::from_prk(&prk)
                    .map_err(|e| CryptoError::ExpandFailed(e.to_string()))?;
                hk.expand(&info, &mut okm)
                    .map_err(|e| CryptoError::ExpandFailed(e.to_string()))?;
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        }

        Ok(HkdfExpandLabelResponse {
            okm_hex: hex::encode(&okm),
            algorithm: req.algorithm,
            length: req.length,
        })
    }
}

pub const HKDF_EXPAND_LABEL: RegisteredTool = RegisteredTool::new::<HkdfExpandLabel>();
