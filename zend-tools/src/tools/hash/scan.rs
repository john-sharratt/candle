//! hash_scan tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{compute_hash, decode_data, HashError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ScanRequest {
    /// The candidate input, interpreted according to `data_encoding`.
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub data_encoding: Option<String>,
    /// The known digest to identify, interpreted according to `hash_encoding`.
    #[validate(length(min = 1))]
    pub known_hash: String,
    /// How `known_hash` is encoded. One of: text, hex, base64. Defaults to hex.
    #[schemars(with = "Option<super::super::DataEncoding>")]
    pub hash_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct ScanResponse {
    pub matches: bool,
    pub algorithm: String,
    pub digest: String,
}

pub struct HashScan;

impl Tool for HashScan {
    const NAME: &'static str = "hash_scan";
    const DESCRIPTION: &'static str =
        "Given a candidate input (`data`) and a digest (`known_hash`), identify \
         which algorithm produced the digest by recomputing the input's hash \
         under every supported algorithm and matching. Returns the algorithm \
         name when found. Always call this tool to identify the algorithm; \
         never deduce it from the digest yourself.";

    type Request = ScanRequest;
    type Response = ScanResponse;
    type Error = HashError;

    fn run(_ctx: &ToolContext, req: ScanRequest) -> Result<ScanResponse, HashError> {
        let data_enc = req.data_encoding.as_deref().unwrap_or("text");
        let hash_enc = req.hash_encoding.as_deref().unwrap_or("hex");
        let data = decode_data(&req.data, data_enc)?;
        let known = decode_data(&req.known_hash, hash_enc)?;

        let algos = [
            "sha256", "sha512", "sha1", "md5", "sha3_256", "sha3_512", "blake3",
        ];
        for algo in algos {
            if let Ok(digest) = compute_hash(&data, algo) {
                if digest == known {
                    return Ok(ScanResponse {
                        matches: true,
                        algorithm: algo.to_string(),
                        digest: hex::encode(&digest),
                    });
                }
            }
        }

        Err(HashError::NoMatch)
    }
}

pub const HASH_SCAN: RegisteredTool = RegisteredTool::new::<HashScan>();
