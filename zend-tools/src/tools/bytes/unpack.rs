//! bytes_unpack tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_bytes, parse_format, unpack_field, BytesError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct UnpackRequest {
    /// The byte buffer to decode, interpreted according to `data_encoding`. Required.
    #[validate(length(min = 1))]
    pub data: String,
    /// How `data` is encoded. One of: hex, base64, base64url, utf8. Defaults to hex.
    #[schemars(with = "Option<super::BytesEncoding>")]
    pub data_encoding: Option<String>,
    /// Python struct-style format string: optional endianness prefix (`<` little-endian, `>` big-endian)
    /// then type chars B/H/I/L/Q/b/h/i/l/q/f/d and `Ns` for an N-byte string. Required.
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
         The inverse of bytes_pack. Always call this tool to decode the bytes; \
         never work them out by hand.";

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
            let v =
                unpack_field(&mut cursor, field, big_endian).map_err(BytesError::UnpackFailed)?;
            values.push(v);
        }

        let bytes_consumed = cursor.position() as usize;
        Ok(UnpackResponse {
            values,
            bytes_consumed,
        })
    }
}

pub const BYTES_UNPACK: RegisteredTool = RegisteredTool::new::<BytesUnpack>();
