//! hkdf_expand_label tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::CryptoError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HkdfExpandLabelRequest {
    #[validate(length(min = 1))]
    pub prk_hex: String,
    #[validate(length(min = 1))]
    pub label: String,
    pub context_hex: Option<String>,
    #[validate(range(min = 1, max = 8160))]
    pub length: u32,
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
    const DESCRIPTION: &'static str =
        "TLS 1.3 HKDF-Expand-Label. Derives keying material from a PRK using a labeled context. \
         Use for: TLS key schedule analysis, deriving traffic secrets.";

    type Request = HkdfExpandLabelRequest;
    type Response = HkdfExpandLabelResponse;
    type Error = CryptoError;

    fn run(_ctx: &ToolContext, req: HkdfExpandLabelRequest) -> Result<HkdfExpandLabelResponse, CryptoError> {
        let prk = hex::decode(&req.prk_hex)
            .map_err(|e| CryptoError::InvalidKey(e.to_string()))?;
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
        match req.algorithm.as_str() {
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
