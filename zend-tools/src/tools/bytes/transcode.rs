//! bytes_transcode tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_bytes, encode_bytes, BytesError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct TranscodeRequest {
    /// The input, interpreted according to `from`.
    #[validate(length(min = 1))]
    pub data: String,
    /// Encoding `data` is currently in. One of: hex, base64, base64url, utf8.
    #[schemars(with = "super::BytesEncoding")]
    #[validate(length(min = 1))]
    pub from: String,
    /// Encoding to convert to. One of: hex, base64, base64url, utf8.
    #[schemars(with = "super::BytesEncoding")]
    #[validate(length(min = 1))]
    pub to: String,
}

#[derive(Serialize)]
pub struct TranscodeResponse {
    pub data: String,
    pub from: String,
    pub to: String,
    pub bytes: usize,
}

pub struct BytesTranscode;

impl Tool for BytesTranscode {
    const NAME: &'static str = "bytes_transcode";
    const DESCRIPTION: &'static str = "Re-encode data between byte representations — hex, base64, \
         base64url, and raw utf8 — without changing the underlying bytes. \
         Use to normalize how a payload is written, not what it contains. \
         Always call this tool to re-encode; never convert the data yourself.";

    type Request = TranscodeRequest;
    type Response = TranscodeResponse;
    type Error = BytesError;

    fn run(_ctx: &ToolContext, req: TranscodeRequest) -> Result<TranscodeResponse, BytesError> {
        let bytes = decode_bytes(&req.data, &req.from)?;
        let encoded = encode_bytes(&bytes, &req.to)?;
        Ok(TranscodeResponse {
            data: encoded,
            from: req.from,
            to: req.to,
            bytes: bytes.len(),
        })
    }
}

pub const BYTES_TRANSCODE: RegisteredTool = RegisteredTool::new::<BytesTranscode>();
