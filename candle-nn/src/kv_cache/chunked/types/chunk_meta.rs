//! The per-block metadata record that the paged attention kernels read for each
//! chunk of a sequence.
//!
//! Each record is 32 bytes and mirrors the C++ struct in `arena_table.cuh`, so
//! the kernel can address K and V data separately when they sit in different
//! arenas. Usage and offset share one packed word, and the absolute RoPE base
//! sits beside them, because the kernel applies rotation at read time. The
//! record is head-independent; per-head GIDs travel in their own tensor. The
//! flattened `u32` layout is what the `chunk_meta` tensor expects.

// ChunkMeta — packed per-block metadata for paged attention kernels

/// Per-block metadata entry in the `chunk_meta` AoS buffer passed to CUDA kernels.
///
/// 32 bytes, 32-byte aligned: `chunk_meta[batch * max_blocks + blk]`.
///
/// Carries separate K-side and V-side global chunk ids so the kernel can
/// address K and V data independently when they reside in different arenas
/// (e.g. after per-head adaptive quantization selects different formats for
/// K vs V).
///
/// Mirrors the C++ `ChunkMeta` struct in `arena_table.cuh`.  The buffer is built
/// by [`ChunkedKvBacking::chunk_meta_row`] on the host and transferred to the
/// device as a `U32` tensor (`max_blocks * 4` elements).
///
/// Use [`ChunkMeta::into_u32s`] to flatten a `Vec<ChunkMeta>` into the
/// `Vec<u32>` expected by [`candle::Tensor::from_vec`].
///
/// Per-head K/V GIDs are passed separately via the `head_gids` tensor;
/// ChunkMeta carries only the head-independent block metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C, align(16))]
pub struct ChunkMeta {
    /// Packed: `[15:0]` = usage (valid token count), `[31:16]` = offset (skip count).
    pub block_usage: u32,
    /// Absolute RoPE base position for the first **valid** token (`i32` bit-cast as `u32`).
    pub rope_pos_u32: u32,
    /// Reserved padding to reach 16-byte alignment.
    pub _pad0: u32,
    /// Reserved padding to reach 16-byte alignment.
    pub _pad1: u32,
}

impl ChunkMeta {
    /// Build a `ChunkMeta` entry from its components.
    ///
    /// * `block_usage` — valid token count in this block's window.
    /// * `rope_pos` — absolute RoPE base position for the first valid token.
    /// * `block_offset` — how many positions at the start of the physical chunk to skip.
    ///   Valid window is `[block_offset, block_offset + block_usage)`.
    #[inline]
    pub fn new(block_usage: u32, rope_pos: i32, block_offset: u16) -> Self {
        Self {
            block_usage: ((block_offset as u32) << 16) | (block_usage & 0xFFFF),
            rope_pos_u32: rope_pos as u32,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Valid token count in this block's window \[0..chunk\_size\] (low 16 bits).
    #[inline]
    pub fn usage(&self) -> u32 {
        self.block_usage & 0xFFFF
    }

    /// Skip-offset from start of physical chunk where valid data begins (high 16 bits).
    #[inline]
    pub fn offset(&self) -> u16 {
        (self.block_usage >> 16) as u16
    }

    /// Absolute RoPE base position for the first valid token in this block.
    #[inline]
    pub fn rope_base(&self) -> i32 {
        self.rope_pos_u32 as i32
    }

    /// Number of `u32` values per `ChunkMeta` entry in the flattened tensor.
    pub const U32S_PER_ENTRY: usize = 4;

    /// Flatten a `Vec<ChunkMeta>` into the raw `u32` layout expected by the
    /// `chunk_meta` tensor argument (`entries.len() * U32S_PER_ENTRY` values).
    pub fn into_u32s(entries: Vec<Self>) -> Vec<u32> {
        entries
            .into_iter()
            .flat_map(|e| [e.block_usage, e.rope_pos_u32, e._pad0, e._pad1])
            .collect()
    }
}
