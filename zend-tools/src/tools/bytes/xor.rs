//! bytes_xor tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_bytes, encode_bytes, BytesError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct XorRequest {
    /// First operand, interpreted according to `a_encoding`. Required.
    #[validate(length(min = 1))]
    pub a: String,
    /// How `a` is encoded. One of: hex, base64, base64url, utf8. Defaults to hex.
    #[schemars(with = "Option<super::BytesEncoding>")]
    pub a_encoding: Option<String>,
    /// Second operand, interpreted according to `b_encoding`. Required.
    #[validate(length(min = 1))]
    pub b: String,
    /// How `b` is encoded. One of: hex, base64, base64url, utf8. Defaults to hex.
    #[schemars(with = "Option<super::BytesEncoding>")]
    pub b_encoding: Option<String>,
    /// Encoding for the returned XOR result. One of: hex, base64, base64url, utf8. Defaults to hex.
    #[schemars(with = "Option<super::BytesEncoding>")]
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct XorResponse {
    pub result: String,
    pub bytes: usize,
}

pub struct BytesXor;

impl Tool for BytesXor {
    const NAME: &'static str = "bytes_xor";
    const DESCRIPTION: &'static str =
        "Combine two byte sequences with a bitwise XOR, zero-padding the \
         shorter one. Use for one-time-pad analysis, keystream masking, and \
         computing bitwise differences between binary blobs. Always call this \
         tool to perform the XOR; never compute the result yourself.";

    type Request = XorRequest;
    type Response = XorResponse;
    type Error = BytesError;

    fn run(_ctx: &ToolContext, req: XorRequest) -> Result<XorResponse, BytesError> {
        let a_enc = req.a_encoding.as_deref().unwrap_or("hex");
        let b_enc = req.b_encoding.as_deref().unwrap_or("hex");
        let out_enc = req.output_encoding.as_deref().unwrap_or("hex");

        let a = decode_bytes(&req.a, a_enc)?;
        let b = decode_bytes(&req.b, b_enc)?;

        let len = a.len().max(b.len());
        let mut result = vec![0u8; len];
        for (i, out) in result.iter_mut().enumerate() {
            let av = a.get(i).copied().unwrap_or(0);
            let bv = b.get(i).copied().unwrap_or(0);
            *out = av ^ bv;
        }

        let encoded = encode_bytes(&result, out_enc)?;
        Ok(XorResponse {
            result: encoded,
            bytes: result.len(),
        })
    }
}

pub const BYTES_XOR: RegisteredTool = RegisteredTool::new::<BytesXor>();
