//! hash_scan tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{compute_hash, decode_data, HashError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ScanRequest {
    pub data: String,
    pub data_encoding: Option<String>,
    #[validate(length(min = 1))]
    pub known_hash: String,
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
        "Given a digest of unknown origin, identify which algorithm produced \
         it by recomputing the candidate input's hash under every supported \
         algorithm and matching. Returns the algorithm name when found.";

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
