//! Virtual-connection frames (§5.1): `OPEN`, `DATA`, `WINDOW`, `FIN` and
//! `RESET`. The stream id travels in the frame header, never in the body.

use crate::wire::{Reader, WireError, Writer};

/// The most bytes one `DATA` frame carries (§4.2).
pub const MAX_DATA: usize = 64 << 10;

/// Open a virtual connection towards `target`: empty when the gateway opens one
/// to a member, an app's name when a member opens one towards the gateway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Open {
    pub stream: u32,
    pub target: String,
}

impl Open {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.target);
    }

    pub fn read(r: &mut Reader, stream: u32) -> Result<Self, WireError> {
        Ok(Self {
            stream,
            target: r.str()?,
        })
    }
}

/// One stream's bytes; the body is the bytes, unprefixed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Data {
    pub stream: u32,
    pub bytes: Vec<u8>,
}

impl Data {
    pub fn read(r: &mut Reader, stream: u32) -> Result<Self, WireError> {
        let bytes = r.rest().to_vec();
        if bytes.is_empty() || bytes.len() > MAX_DATA {
            return Err(WireError::TooLong {
                what: r.what(),
                len: bytes.len(),
                max: MAX_DATA,
            });
        }
        Ok(Self { stream, bytes })
    }
}

/// The reader consumed `credit` more bytes: the writer may send that many more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    pub stream: u32,
    pub credit: u32,
}

impl Window {
    pub fn read(r: &mut Reader, stream: u32) -> Result<Self, WireError> {
        let credit = r.u32()?;
        if credit == 0 {
            return Err(WireError::Invalid {
                what: r.what(),
                value: 0,
            });
        }
        Ok(Self { stream, credit })
    }
}
