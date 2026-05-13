//! Byte encoding and manipulation tools: `bytes_{transcode,pack,unpack,xor}`.
//!
//! Low-level byte utilities for protocol archaeology, binary format parsing,
//! and working with the hex payloads produced by TCP/UDP session tools.
//!
//! # `bytes_transcode`
//!
//! Convert between hex, base64, base64url, and UTF-8 encodings.  E.g. take
//! `data_hex` from a TCP recv and re-encode it as base64 for an HTTP body.
//!
//! # `bytes_pack`
//!
//! Encode a list of typed values into a binary buffer, using a Python `struct`-
//! style format string.  Format: optional `>` (big-endian) or `<` (little-endian)
//! followed by field codes: `B`/`b` (u8/i8), `H`/`h` (u16/i16), `I`/`L`/`i`/`l`
//! (u32/i32), `Q`/`q` (u64/i64), `f`/`d` (f32/f64), `Ns` (N-byte string).
//! Useful for constructing custom binary protocol messages.
//!
//! # `bytes_unpack`
//!
//! Reverse of pack: parse a binary buffer into typed JSON values given the same
//! format string.  Useful for interpreting captured binary protocol frames.
//!
//! # `bytes_xor`
//!
//! XOR two byte sequences.  Useful for stream-cipher keystream application,
//! simple obfuscation analysis, or CRC pre/post-condition operations.
//!
//! # Shared utilities
//!
//! - [`decode_bytes`] / [`encode_bytes`] — hex/base64/base64url/utf8 codec
//! - [`parse_format`] — parse a struct format string into `(big_endian, Vec<FormatField>)`
//! - [`pack_field`] / [`unpack_field`] — pack/unpack a single typed field
//! - [`FormatField`] — the typed variant enum used during pack/unpack
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `invalid_encoding` | Unknown encoding name |
//! | `decode_failed` | Bad hex/base64 input |
//! | `encode_failed` | Bytes cannot be encoded (e.g. non-UTF8 for `utf8` target) |
//! | `invalid_format` | Unknown format character or malformed format string |
//! | `pack_failed` | Value does not fit the format field type |
//! | `unpack_failed` | Buffer too short for the format or read error |

use thiserror::Error;
use crate::ToolError;

pub mod transcode;
pub mod pack;
pub mod unpack;
pub mod xor;

pub use transcode::BYTES_TRANSCODE;
pub use pack::BYTES_PACK;
pub use unpack::BYTES_UNPACK;
pub use xor::BYTES_XOR;

#[derive(Debug, Error)]
pub enum BytesError {
    #[error("invalid encoding: {0}")]
    InvalidEncoding(String),
    #[error("decode failed: {0}")]
    DecodeFailed(String),
    #[error("encode failed: {0}")]
    EncodeFailed(String),
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("pack failed: {0}")]
    PackFailed(String),
    #[error("unpack failed: {0}")]
    UnpackFailed(String),
}

impl ToolError for BytesError {
    fn code(&self) -> &'static str {
        match self {
            BytesError::InvalidEncoding(_) => "invalid_encoding",
            BytesError::DecodeFailed(_) => "decode_failed",
            BytesError::EncodeFailed(_) => "encode_failed",
            BytesError::InvalidFormat(_) => "invalid_format",
            BytesError::PackFailed(_) => "pack_failed",
            BytesError::UnpackFailed(_) => "unpack_failed",
        }
    }
}

pub fn decode_bytes(data: &str, encoding: &str) -> Result<Vec<u8>, BytesError> {
    match encoding {
        "hex" => hex::decode(data).map_err(|e| BytesError::DecodeFailed(e.to_string())),
        "base64" => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.decode(data)
                .map_err(|e| BytesError::DecodeFailed(e.to_string()))
        }
        "base64url" => {
            use base64::Engine;
            base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data)
                .map_err(|e| BytesError::DecodeFailed(e.to_string()))
        }
        "utf8" => Ok(data.as_bytes().to_vec()),
        other => Err(BytesError::InvalidEncoding(format!("unknown encoding: {other}"))),
    }
}

pub fn encode_bytes(bytes: &[u8], encoding: &str) -> Result<String, BytesError> {
    match encoding {
        "hex" => Ok(hex::encode(bytes)),
        "base64" => {
            use base64::Engine;
            Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        "base64url" => {
            use base64::Engine;
            Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
        }
        "utf8" => std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|e| BytesError::EncodeFailed(e.to_string())),
        other => Err(BytesError::InvalidEncoding(format!("unknown encoding: {other}"))),
    }
}

// ── Format string parsing ──────────────────────────────────────────────────────

#[derive(Debug)]
pub enum FormatField {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
    Bytes(usize),
}

pub fn parse_format(fmt: &str) -> Result<(bool, Vec<FormatField>), BytesError> {
    let mut chars = fmt.chars().peekable();
    let big_endian = match chars.peek() {
        Some('>') => { chars.next(); true }
        Some('<') => { chars.next(); false }
        _ => true,
    };

    let mut fields = Vec::new();
    let mut num_str = String::new();

    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            num_str.push(c);
            continue;
        }
        let n: usize = if num_str.is_empty() { 1 } else {
            num_str.parse().map_err(|_| BytesError::InvalidFormat("bad number in format".to_string()))?
        };
        num_str.clear();

        match c {
            'B' => for _ in 0..n { fields.push(FormatField::U8) },
            'H' => for _ in 0..n { fields.push(FormatField::U16) },
            'I' | 'L' => for _ in 0..n { fields.push(FormatField::U32) },
            'Q' => for _ in 0..n { fields.push(FormatField::U64) },
            'b' => for _ in 0..n { fields.push(FormatField::I8) },
            'h' => for _ in 0..n { fields.push(FormatField::I16) },
            'i' | 'l' => for _ in 0..n { fields.push(FormatField::I32) },
            'q' => for _ in 0..n { fields.push(FormatField::I64) },
            'f' => for _ in 0..n { fields.push(FormatField::F32) },
            'd' => for _ in 0..n { fields.push(FormatField::F64) },
            's' => fields.push(FormatField::Bytes(n)),
            other => return Err(BytesError::InvalidFormat(format!("unknown format char: {other}"))),
        }
    }

    Ok((big_endian, fields))
}

pub fn pack_field(buf: &mut Vec<u8>, field: &FormatField, val: &serde_json::Value, big: bool) -> Result<(), String> {
    use byteorder::{BigEndian, LittleEndian, WriteBytesExt};

    macro_rules! get_int { ($t:ty) => { val.as_i64().ok_or_else(|| "expected integer".to_string())? as $t } }
    macro_rules! get_uint { ($t:ty) => { val.as_u64().ok_or_else(|| "expected integer".to_string())? as $t } }
    macro_rules! get_float { ($t:ty) => { val.as_f64().ok_or_else(|| "expected float".to_string())? as $t } }

    if big {
        match field {
            FormatField::U8  => buf.write_u8(get_uint!(u8)).unwrap(),
            FormatField::U16 => buf.write_u16::<BigEndian>(get_uint!(u16)).unwrap(),
            FormatField::U32 => buf.write_u32::<BigEndian>(get_uint!(u32)).unwrap(),
            FormatField::U64 => buf.write_u64::<BigEndian>(get_uint!(u64)).unwrap(),
            FormatField::I8  => buf.write_i8(get_int!(i8)).unwrap(),
            FormatField::I16 => buf.write_i16::<BigEndian>(get_int!(i16)).unwrap(),
            FormatField::I32 => buf.write_i32::<BigEndian>(get_int!(i32)).unwrap(),
            FormatField::I64 => buf.write_i64::<BigEndian>(get_int!(i64)).unwrap(),
            FormatField::F32 => buf.write_f32::<BigEndian>(get_float!(f32)).unwrap(),
            FormatField::F64 => buf.write_f64::<BigEndian>(get_float!(f64)).unwrap(),
            FormatField::Bytes(n) => {
                let s = val.as_str().ok_or_else(|| "expected string".to_string())?;
                let bytes = s.as_bytes();
                let mut out = vec![0u8; *n];
                let copy_len = bytes.len().min(*n);
                out[..copy_len].copy_from_slice(&bytes[..copy_len]);
                buf.extend_from_slice(&out);
            }
        }
    } else {
        match field {
            FormatField::U8  => buf.write_u8(get_uint!(u8)).unwrap(),
            FormatField::U16 => buf.write_u16::<LittleEndian>(get_uint!(u16)).unwrap(),
            FormatField::U32 => buf.write_u32::<LittleEndian>(get_uint!(u32)).unwrap(),
            FormatField::U64 => buf.write_u64::<LittleEndian>(get_uint!(u64)).unwrap(),
            FormatField::I8  => buf.write_i8(get_int!(i8)).unwrap(),
            FormatField::I16 => buf.write_i16::<LittleEndian>(get_int!(i16)).unwrap(),
            FormatField::I32 => buf.write_i32::<LittleEndian>(get_int!(i32)).unwrap(),
            FormatField::I64 => buf.write_i64::<LittleEndian>(get_int!(i64)).unwrap(),
            FormatField::F32 => buf.write_f32::<LittleEndian>(get_float!(f32)).unwrap(),
            FormatField::F64 => buf.write_f64::<LittleEndian>(get_float!(f64)).unwrap(),
            FormatField::Bytes(n) => {
                let s = val.as_str().ok_or_else(|| "expected string".to_string())?;
                let bytes = s.as_bytes();
                let copy_len = bytes.len().min(*n);
                let mut out = vec![0u8; *n];
                out[..copy_len].copy_from_slice(&bytes[..copy_len]);
                buf.extend_from_slice(&out);
            }
        }
    }
    Ok(())
}

pub fn unpack_field(
    cursor: &mut std::io::Cursor<&Vec<u8>>,
    field: &FormatField,
    big: bool,
) -> Result<serde_json::Value, String> {
    use byteorder::{BigEndian, LittleEndian, ReadBytesExt};

    let v = if big {
        match field {
            FormatField::U8  => serde_json::json!(cursor.read_u8().map_err(|e| e.to_string())?),
            FormatField::U16 => serde_json::json!(cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::U32 => serde_json::json!(cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::U64 => serde_json::json!(cursor.read_u64::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::I8  => serde_json::json!(cursor.read_i8().map_err(|e| e.to_string())?),
            FormatField::I16 => serde_json::json!(cursor.read_i16::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::I32 => serde_json::json!(cursor.read_i32::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::I64 => serde_json::json!(cursor.read_i64::<BigEndian>().map_err(|e| e.to_string())?),
            FormatField::F32 => serde_json::Number::from_f64(cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())? as f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            FormatField::F64 => serde_json::Number::from_f64(cursor.read_f64::<BigEndian>().map_err(|e| e.to_string())?).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            FormatField::Bytes(n) => {
                use std::io::Read;
                let mut buf = vec![0u8; *n];
                cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
                serde_json::Value::String(String::from_utf8_lossy(&buf).into_owned())
            }
        }
    } else {
        match field {
            FormatField::U8  => serde_json::json!(cursor.read_u8().map_err(|e| e.to_string())?),
            FormatField::U16 => serde_json::json!(cursor.read_u16::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::U32 => serde_json::json!(cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::U64 => serde_json::json!(cursor.read_u64::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::I8  => serde_json::json!(cursor.read_i8().map_err(|e| e.to_string())?),
            FormatField::I16 => serde_json::json!(cursor.read_i16::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::I32 => serde_json::json!(cursor.read_i32::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::I64 => serde_json::json!(cursor.read_i64::<LittleEndian>().map_err(|e| e.to_string())?),
            FormatField::F32 => serde_json::Number::from_f64(cursor.read_f32::<LittleEndian>().map_err(|e| e.to_string())? as f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            FormatField::F64 => serde_json::Number::from_f64(cursor.read_f64::<LittleEndian>().map_err(|e| e.to_string())?).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            FormatField::Bytes(n) => {
                use std::io::Read;
                let mut buf = vec![0u8; *n];
                cursor.read_exact(&mut buf).map_err(|e| e.to_string())?;
                serde_json::Value::String(String::from_utf8_lossy(&buf).into_owned())
            }
        }
    };
    Ok(v)
}
