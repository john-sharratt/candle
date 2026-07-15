//! hkdf_extract tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HkdfExtractRequest {
    #[validate(length(min = 1))]
    pub ikm: String,
    /// How `ikm` (input keying material) is encoded. One of: text, hex, base64. Defaults to hex.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub ikm_encoding: Option<String>,
    pub salt: Option<String>,
    /// How `salt` is encoded. One of: text, hex, base64. Defaults to hex.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub salt_encoding: Option<String>,
    /// HKDF hash function. One of: sha256, sha512.
    #[schemars(with = "super::HkdfHash")]
    #[validate(length(min = 1))]
    pub algorithm: String,
}

#[derive(Serialize)]
pub struct HkdfExtractResponse {
    pub prk_hex: String,
    pub algorithm: String,
}

pub struct HkdfExtract;

impl Tool for HkdfExtract {
    const NAME: &'static str = "hkdf_extract";
    const DESCRIPTION: &'static str =
        "HKDF-Extract (RFC 5869): condense raw, possibly non-uniform input \
         key material into one uniformly-random pseudorandom key — the first \
         stage of the HKDF derivation chain. Supports sha256 and sha512.";

    type Request = HkdfExtractRequest;
    type Response = HkdfExtractResponse;
    type Error = CryptoError;

    fn run(
        _ctx: &ToolContext,
        req: HkdfExtractRequest,
    ) -> Result<HkdfExtractResponse, CryptoError> {
        let ikm_enc = req.ikm_encoding.as_deref().unwrap_or("hex");
        let ikm = decode_data(&req.ikm, ikm_enc).map_err(CryptoError::InvalidDataEncoding)?;

        let salt_opt: Option<Vec<u8>> = if let Some(s) = &req.salt {
            let salt_enc = req.salt_encoding.as_deref().unwrap_or("hex");
            Some(decode_data(s, salt_enc).map_err(CryptoError::InvalidDataEncoding)?)
        } else {
            None
        };

        let prk_hex = match normalize_algorithm(&req.algorithm).as_str() {
            "sha256" => {
                let (prk, _) = hkdf::Hkdf::<sha2::Sha256>::extract(salt_opt.as_deref(), &ikm);
                hex::encode(prk)
            }
            "sha512" => {
                let (prk, _) = hkdf::Hkdf::<sha2::Sha512>::extract(salt_opt.as_deref(), &ikm);
                hex::encode(prk)
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        };

        Ok(HkdfExtractResponse {
            prk_hex,
            algorithm: req.algorithm,
        })
    }
}

pub const HKDF_EXTRACT: RegisteredTool = RegisteredTool::new::<HkdfExtract>();
