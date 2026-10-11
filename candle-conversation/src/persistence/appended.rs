//! Where an appended record landed, and the identity it was written with.

use super::identity::RecordIdentity;
use super::manifest::RecordLoc;
use super::segment::SegmentId;

/// One record placed in the active segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Appended {
    /// The segment that holds it.
    pub segment: SegmentId,
    /// Its byte offset in that segment.
    pub offset: u64,
    /// Its padded on-disk size, header and padding included.
    pub size: u64,
    /// The identity its header carries — minted for a record this substrate
    /// originated, carried for a copy.
    pub identity: RecordIdentity,
}

impl Appended {
    /// The record's location, for a payload of `payload_len` bytes.
    pub fn loc(&self, payload_len: u64) -> RecordLoc {
        RecordLoc {
            segment: self.segment,
            offset: self.offset,
            payload_len,
            record_size: self.size,
        }
    }
}
