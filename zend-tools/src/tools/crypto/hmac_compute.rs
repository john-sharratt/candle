//! hmac_compute tool.

use hmac::Mac;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, encode_output, normalize_algorithm, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HmacRequest {
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub data_encoding: Option<String>,
    #[validate(length(min = 1))]
    pub key: String,
    /// How `key` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub key_encoding: Option<String>,
    /// HMAC hash function. One of: sha256, sha512, sha1.
    #[schemars(with = "super::HmacAlgorithm")]
    #[validate(length(min = 1))]
    pub algorithm: String,
    /// Encoding for the returned MAC. One of: text, hex, base64. Defaults to hex.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct HmacResponse {
    pub mac: String,
    pub algorithm: String,
}

pub struct HmacCompute;

impl Tool for HmacCompute {
    const NAME: &'static str = "hmac_compute";
    const DESCRIPTION: &'static str =
        "Compute a keyed HMAC authentication tag over a message (sha256, \
         sha512, or sha1). Use to sign API requests or to confirm a message \
         arrived unaltered from a party holding the shared secret key.";

    type Request = HmacRequest;
    type Response = HmacResponse;
    type Error = CryptoError;

    fn run(_ctx: &ToolContext, req: HmacRequest) -> Result<HmacResponse, CryptoError> {
        let data_enc = req.data_encoding.as_deref().unwrap_or("text");
        let key_enc = req.key_encoding.as_deref().unwrap_or("text");
        let out_enc = req.output_encoding.as_deref().unwrap_or("hex");

        let data = decode_data(&req.data, data_enc).map_err(CryptoError::InvalidDataEncoding)?;
        let key = decode_data(&req.key, key_enc).map_err(CryptoError::InvalidDataEncoding)?;

        let mac_bytes: Vec<u8> = match normalize_algorithm(&req.algorithm).as_str() {
            "sha256" => {
                type HmacSha256 = hmac::Hmac<sha2::Sha256>;
                let mut mac = <HmacSha256 as hmac::Mac>::new_from_slice(&key)
                    .map_err(|e| CryptoError::UnknownAlgorithm(e.to_string()))?;
                mac.update(&data);
                mac.finalize().into_bytes().to_vec()
            }
            "sha512" => {
                type HmacSha512 = hmac::Hmac<sha2::Sha512>;
                let mut mac = <HmacSha512 as hmac::Mac>::new_from_slice(&key)
                    .map_err(|e| CryptoError::UnknownAlgorithm(e.to_string()))?;
                mac.update(&data);
                mac.finalize().into_bytes().to_vec()
            }
            "sha1" => {
                type HmacSha1 = hmac::Hmac<sha1::Sha1>;
                let mut mac = <HmacSha1 as hmac::Mac>::new_from_slice(&key)
                    .map_err(|e| CryptoError::UnknownAlgorithm(e.to_string()))?;
                mac.update(&data);
                mac.finalize().into_bytes().to_vec()
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        };

        Ok(HmacResponse {
            mac: encode_output(&mac_bytes, out_enc),
            algorithm: req.algorithm,
        })
    }
}

pub const HMAC_COMPUTE: RegisteredTool = RegisteredTool::new::<HmacCompute>();
