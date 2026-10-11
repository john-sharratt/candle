//! The primitives every frame body is written in: little-endian integers,
//! `u32`-length blobs, `u32`-count lists and LEB128 varints, in the style of
//! the persistence codec. A reader refuses a short buffer, a trailing byte and a
//! value out of range; it never guesses.

mod reader;
mod varint;
mod writer;

pub use reader::Reader;
pub use varint::{read_varint, write_varint};
pub use writer::Writer;

use thiserror::Error;

/// Why bytes are not a frame.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WireError {
    #[error("{what}: the frame ends {need} bytes short")]
    Truncated { what: &'static str, need: usize },
    #[error("{what}: {extra} bytes past the end of the body")]
    Trailing { what: &'static str, extra: usize },
    #[error("unknown frame kind {0}")]
    UnknownKind(u8),
    #[error("{what}: invalid value {value}")]
    Invalid { what: &'static str, value: u64 },
    #[error("{what}: not UTF-8")]
    NotUtf8 { what: &'static str },
    #[error("{what}: {len} bytes is over the {max}-byte limit")]
    TooLong {
        what: &'static str,
        len: usize,
        max: usize,
    },
}
