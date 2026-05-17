//! bytes_unpack tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{decode_bytes, parse_format, unpack_field, BytesError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct UnpackRequest {
    #[validate(length(min = 1))]
    pub data: String,
    pub data_encoding: Option<String>,
    #[validate(length(min = 1))]
    pub format: String,
}

#[derive(Serialize)]
pub struct UnpackResponse {
    pub values: Vec<serde_json::Value>,
    pub bytes_consumed: usize,
}

pub struct BytesUnpack;

impl Tool for BytesUnpack {
    const NAME: &'static str = "bytes_unpack";
    const DESCRIPTION: &'static str =
        "Decode a raw byte buffer back into structured values — fixed-width \
         integers, floats, and strings — using a struct-style format string. \
         The inverse of bytes_pack.";

    type Request = UnpackRequest;
    type Response = UnpackResponse;
    type Error = BytesError;

    fn run(_ctx: &ToolContext, req: UnpackRequest) -> Result<UnpackResponse, BytesError> {
        let enc = req.data_encoding.as_deref().unwrap_or("hex");
        let data = decode_bytes(&req.data, enc)?;
        let (big_endian, fields) = parse_format(&req.format)?;

        let mut cursor = std::io::Cursor::new(&data);
        let mut values = Vec::new();

        for field in &fields {
            let v = unpack_field(&mut cursor, field, big_endian)
                .map_err(BytesError::UnpackFailed)?;
            values.push(v);
        }

        let bytes_consumed = cursor.position() as usize;
        Ok(UnpackResponse { values, bytes_consumed })
    }
}

pub const BYTES_UNPACK: RegisteredTool = RegisteredTool::new::<BytesUnpack>();
