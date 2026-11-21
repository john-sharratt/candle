//! Turns a buffer's shape and encoding into a byte count, in the one place that
//! does that arithmetic.
//!
//! A q8a128 buffer is packed into super-blocks of eight 128-element tiles, each
//! super-block 1152 bytes, and the int8 matmul's N tile sets the width a padded
//! projection emits. `BufferShape::bytes` rounds both axes up, so a partial tile
//! costs a whole tile and a partial group a whole super-block. Sizing is an
//! upper bound, and an upper bound rounds up.

use super::*;

/// Elements per q8a128 tile, and tiles per flat-grouped super-block.
///
/// The packing is eight 128-element tiles to a 1152-byte super-block
/// (`blocks.cuh`). Named rather than inlined because a reader checking this
/// arithmetic against the kernel should find both numbers in one place.
pub(super) const Q8A128_TILE_ELEMS: usize = 128;

/// The int8 matmul's N tile: a KO weight's row count is a multiple of it
/// (`ko_tileable`), so a projection padded to tile emits that many columns.
pub(super) const KO_N_TILE: usize = 32;

pub(super) const Q8A128_TILES_PER_BLOCK: usize = 8;

pub(super) const Q8A128_BLOCK_BYTES: usize = 1152;

/// How a buffer's elements are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Plain row-major elements of a candle dtype.
    Dense(DType),
    /// q8a128 flat-grouped super-blocks, as the int8 tensor-core path consumes.
    Q8a128,
}

/// A buffer's shape and encoding — everything needed to size it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferShape {
    pub rows: usize,
    pub cols: usize,
    pub encoding: Encoding,
}

impl BufferShape {
    /// The one place a shape becomes a byte count.
    ///
    /// `div_ceil` on both axes deliberately: a partial tile costs a whole tile
    /// and a partial group a whole super-block. Sizing is an upper bound, and an
    /// upper bound rounds up.
    pub fn bytes(&self) -> usize {
        match self.encoding {
            Encoding::Dense(dtype) => self.rows * self.cols * dtype.size_in_bytes(),
            Encoding::Q8a128 => {
                let tiles = self.rows * self.cols.div_ceil(Q8A128_TILE_ELEMS);
                tiles.div_ceil(Q8A128_TILES_PER_BLOCK) * Q8A128_BLOCK_BYTES
            }
        }
    }
}
