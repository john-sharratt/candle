//! A segment's footprint in a turn's K/V grid, as a start offset and a length.
//!
//! A span lets a sub-range of a turn be windowed or projected without walking the
//! segments again. Its offset is also the running sum of the lengths before it, so
//! `validate_tiling` asserts the two agree, and the stored offset cannot drift
//! from the layout it describes.

use super::*;

/// A segment's footprint in the turn's K/V grid: `len` token positions starting
/// at absolute `offset`.  `offset` is recoverable from the running sum of prior
/// segment lengths — [`TurnLayout::validate_tiling`] asserts the two agree — and
/// is stored so a turn sub-range can be windowed/projected without re-walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSpan {
    pub offset: u32,
    pub len: u32,
}

impl KvSpan {
    pub fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }
    /// Exclusive end of the span (`offset + len`).
    pub fn end(&self) -> u32 {
        self.offset + self.len
    }
    /// The span as a `[start, end)` slice range, for windowing a token grid.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset as usize..self.end() as usize
    }
}
