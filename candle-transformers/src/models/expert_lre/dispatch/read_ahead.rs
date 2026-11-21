//! The layout of the read-ahead copy for each row: where each projection's bytes
//! sit inside a slot image.
//!
//! Each row records the offsets of its gate, up and down entries, and the size of
//! its image rounded up to the copy unit the workers move. Slots are cut to a stride
//! padded past that size, so a read-ahead copy always fits in the slot it targets.

use super::*;

/// Read-ahead's device buffers ([`Dispatch::ahead`]).
pub(super) struct AheadBuffers {
    pub(super) row_layout: CudaSlice<u64>,
    pub(super) items: CudaSlice<u64>,
    pub(super) done: CudaSlice<u32>,
}

/// Each row's read-ahead layout, `[rows][4]`: the gate, up and down entries'
/// offsets inside a slot image — what a published entry adds to the slot base —
/// and the bytes a read-ahead copies, the image rounded up to the 16-byte unit
/// the workers move. Every slot and warm slot holds at least that: both are cut
/// to a stride padded well past it.
pub(super) fn row_layout(table: &LiveTable, image_bytes: &[usize]) -> Vec<u64> {
    image_bytes
        .iter()
        .enumerate()
        .flat_map(|(row, &bytes)| {
            [
                table.offset(Proj::Gate, row),
                table.offset(Proj::Up, row),
                table.offset(Proj::Down, row),
                bytes.next_multiple_of(16) as u64,
            ]
        })
        .collect()
}
