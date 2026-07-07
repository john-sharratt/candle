//! hash_compute tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{compute_hash, decode_data, encode_output, HashError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ComputeRequest {
    /// Hash algorithm. One of: sha256, sha512, sha1, md5, sha3_256, sha3_512, blake3. Required.
    #[schemars(with = "super::HashAlgorithm")]
    #[validate(length(min = 1))]
    pub algorithm: String,
    /// The data to hash, interpreted according to `data_encoding`.
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub data_encoding: Option<String>,
    /// Encoding for the returned digest. One of: text, hex, base64. Defaults to hex.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct ComputeResponse {
    pub algorithm: String,
    pub digest: String,
    pub output_encoding: String,
}

pub struct HashCompute;

impl Tool for HashCompute {
    const NAME: &'static str = "hash_compute";
    const DESCRIPTION: &'static str =
        "Compute a one-shot cryptographic digest — a fixed-size fingerprint — \
         of data, using sha256, sha512, sha1, md5, sha3_256, sha3_512, or \
         blake3. Returns the digest as hex or base64 for content \
         fingerprinting and integrity checks.";

    type Request = ComputeRequest;
    type Response = ComputeResponse;
    type Error = HashError;

    fn run(_ctx: &ToolContext, req: ComputeRequest) -> Result<ComputeResponse, HashError> {
        let encoding = req.data_encoding.as_deref().unwrap_or("text");
        let output_enc = req.output_encoding.as_deref().unwrap_or("hex");
        let data = decode_data(&req.data, encoding)?;
        let digest_bytes = compute_hash(&data, &req.algorithm)?;
        let digest = encode_output(&digest_bytes, output_enc);
        Ok(ComputeResponse {
            algorithm: req.algorithm,
            digest,
            output_encoding: output_enc.to_string(),
        })
    }
}

pub const HASH_COMPUTE: RegisteredTool = RegisteredTool::new::<HashCompute>();
