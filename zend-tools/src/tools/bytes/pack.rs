//! bytes_pack tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{encode_bytes, pack_field, parse_format, BytesError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PackRequest {
    pub values: Vec<serde_json::Value>,
    #[validate(length(min = 1))]
    pub format: String,
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct PackResponse {
    pub data: String,
    pub bytes_packed: usize,
}

pub struct BytesPack;

impl Tool for BytesPack {
    const NAME: &'static str = "bytes_pack";
    const DESCRIPTION: &'static str =
        "Pack structured values into binary bytes using a format string (like Python struct). \
         Format: optional < (little-endian) or > (big-endian), then type chars: \
         B(u8), H(u16), I/L(u32), Q(u64), b(i8), h(i16), i/l(i32), q(i64), f(f32), d(f64), Ns(N-byte string).";

    type Request = PackRequest;
    type Response = PackResponse;
    type Error = BytesError;

    fn run(_ctx: &ToolContext, req: PackRequest) -> Result<PackResponse, BytesError> {
        let (big_endian, fields) = parse_format(&req.format)?;
        if fields.len() != req.values.len() {
            return Err(BytesError::PackFailed(format!(
                "format has {} fields but {} values provided",
                fields.len(), req.values.len()
            )));
        }

        let mut buf: Vec<u8> = Vec::new();
        for (field, val) in fields.iter().zip(req.values.iter()) {
            pack_field(&mut buf, field, val, big_endian)
                .map_err(BytesError::PackFailed)?;
        }

        let enc = req.output_encoding.as_deref().unwrap_or("hex");
        let encoded = encode_bytes(&buf, enc)?;
        Ok(PackResponse { data: encoded, bytes_packed: buf.len() })
    }
}

pub const BYTES_PACK: RegisteredTool = RegisteredTool::new::<BytesPack>();
