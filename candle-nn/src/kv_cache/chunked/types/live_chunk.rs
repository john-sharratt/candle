//! The per-block chunk window that a live sequence holds, and the borrowed view
//! that a per-forward metadata build reads it through.
//!
//! A window pairs the RAII chunk IDs of one block with the valid-token span it
//! claims inside each physical chunk. RoPE is not stored: a block's position is
//! the positions the blocks before it take — each its lead (positions it leaves
//! empty in front of itself) and its tokens — plus its own lead, which is why
//! the same chunk bytes can be injected at any position. The borrowed view exists because
//! cloning every window on each layer call cost about half a millisecond at deep
//! prefixes. A window also carries the format tag of each band, and it takes
//! writes only while every band is in a format the paged stores write.

use super::*;

// SealedChunk / SealedSequence — immutable snapshot of sealed history

/// One committed (frozen) chunk window in a sequence's history.
///
/// Holds a `ChunkGid` for RAII ownership of the physical chunk.  Multiple
/// `SealedChunk` entries (from different turns) may reference the same
/// physical chunk with different `offset`/`token_count` windows — each
/// holds a clone of the same `ChunkGid` (which wraps an `Arc<GidInner>`).
///
/// # RoPE is *not* stored
///
/// Deliberately position-agnostic: every field describes either the
/// physical chunk's identity (`gids`), the byte window the record claims
/// (`offset`, `token_count`), or the dequantisation shape (`k_pal`,
/// `v_pal`, `k_scale`, `v_scale`).  **Nothing positional**.  K bytes
/// in the cache are stored *un-rotated* by the prefill kernel
/// ([paged_prefill_kernel.cuh `// KV writeback (un-rotated)`]).  RoPE
/// rotation is applied at the latest responsible moment — inside the
/// attention kernel at read time, against `slice_rope(...)` from the
/// per-chunk `ChunkMeta` buffer, which is rebuilt from cumulative
/// usage of preceding blocks in the *current slot*'s layout each
/// time the decode/prefill metadata is synced.
///
/// Consequence: a `SealedChunk` (and its underlying K/V bytes) can be
/// **injected at any absolute position** in any sequence and the
/// kernel will apply the correct RoPE for that new position.  No
/// re-rotation, no byte copy, no CoW required.
/// Borrowed view of one live chunk, for per-forward metadata builds
/// ([`super::ChunkedKvBacking::visit_live_chunks`]). The zero-clone
/// counterpart of [`SealedChunk`]: no `HeadGids` refcount traffic, no
/// `arena_byte_size` walk, no Arc bumps. The owned-snapshot path cost
/// ~0.5 ms of pure clone work per layer-call at deep prefixes — the
/// dominant host cost of a prefill call (measured, prefill_ab profiled
/// bench) — while the slice build itself is ~15 µs.
pub struct LiveChunkRef<'a> {
    /// Per-head RAII chunk IDs, indexed `head * 2 + is_value`.
    pub gids: &'a HeadGids,
    /// Start position within the physical chunk where this window begins.
    pub offset: u16,
    /// Number of valid tokens in this window (from `offset`).
    pub token_count: u16,
    /// Packed K/V palette maps (`n_kv_head × head_dim/4` bytes; empty =
    /// identity) and outer scales (`n_kv_head × N_PALETTE`; empty = 1.0).
    pub k_pal: &'a [u8],
    pub v_pal: &'a [u8],
    pub k_scale: &'a [f32],
    pub v_scale: &'a [f32],
    /// Per-`(head, palette)` K/V band format tags ([`ArenaFormatTag::as_u8`]),
    /// `n_kv_head × N_PALETTE` entries each. See [`SealedChunk::k_fmt`] for why
    /// the format travels with the chunk rather than with the arena.
    pub k_fmt: &'a [u8],
    pub v_fmt: &'a [u8],
    /// Device-resident KvHead record handle, when the chunk has one.
    pub meta: Option<&'a MetaGid>,
}

/// Per-block chunk window — combines physical chunk references (RAII `ChunkGid`s)
/// with the window geometry (usage count and offset within the physical chunk).
///
/// Each `ChunkGid` provides RAII ownership: dropping a `ChunkWindow` returns
/// the physical chunks to the pool (unless other clones exist).  Cloning a
/// `ChunkWindow` creates shared references (same Arc, bumped refcount).
///
/// `gids` always has length `2 * n_kv_head`, indexed as:
///
///   `slot = head * 2 + is_value`
///
/// where `is_value` is 0 for K and 1 for V.  When all heads share one arena,
/// every slot holds a clone of the same `ChunkGid` (cheap — just an Arc bump).
/// When per-head quantization is active, each slot may point to a different
/// arena / chunk.
///
/// RoPE positions are not stored — a block's first token sits at the positions
/// all preceding blocks take ([`Self::span`] each) plus its own [`Self::lead`].
#[derive(Debug, Clone)]
pub(crate) struct ChunkWindow {
    /// RAII chunk IDs, length `2 * n_kv_head`.
    /// Indexed as `head * 2 + is_value` (0 = K, 1 = V).
    pub(crate) gids: HeadGids,
    /// Valid token count in this chunk's window.
    pub(in super::super) usage: u32,
    /// Positions left empty in front of this window: a hole in the slot's
    /// position space that holds no K/V (`docs/placed_section_layout.md`
    /// §3.1). Zero for a window that follows the one before it directly.
    pub(in super::super) lead: u32,
    /// Skip-count from the start of the physical chunk where valid data begins.
    pub(in super::super) offset: u16,
    /// Per-chunk K-side palette map: `n_kv_head * (head_dim/4)` bytes.
    /// Always populated with the identity palette at construction time.
    pub(crate) k_pal: Arc<Vec<u8>>,
    /// Per-chunk V-side palette map: same layout as `k_pal`.
    pub(crate) v_pal: Arc<Vec<u8>>,
    /// Outer K scales: `n_kv_head × N_PALETTE` f32 values. Empty = all 1.0.
    /// Encoder multiplies values by this before quantizing; decoder divides
    /// dequantized values by this to recover original magnitude.
    pub(crate) k_scale: Arc<Vec<f32>>,
    /// Outer V scales: `n_kv_head × N_PALETTE` f32 values. Same convention as k_scale.
    pub(crate) v_scale: Arc<Vec<f32>>,
    /// Per-`(head, palette)` K-side storage format ([`ArenaFormatTag::as_u8`]),
    /// `n_kv_head × N_PALETTE` entries. See [`SealedChunk::k_fmt`] for why the
    /// format lives on the chunk rather than on the arena.
    ///
    /// [`ArenaFormatTag::as_u8`]: crate::kv_cache::ArenaFormatTag::as_u8
    pub(crate) k_fmt: Arc<Vec<u8>>,
    /// Per-`(head, palette)` V-side storage format. Same layout as `k_fmt`.
    pub(crate) v_fmt: Arc<Vec<u8>>,
    /// Co-resident KV-head metadata record handle, propagated from the
    /// `SealedChunk` this window was injected from (`Some`) or `None` for a
    /// freshly-allocated float writer chunk. Cloned with the window so every
    /// slot referencing the chunk shares one record. See [`super::meta_pool`].
    pub(crate) meta: Option<MetaGid>,
    /// The block's provenance `sign(Q)` bits once its R16 bands are gone — see
    /// [`QSign`]. Travels with the window as `meta` does.
    pub(crate) q_sign: Option<QSign>,
}

impl ChunkWindow {
    /// The positions this window takes in its slot: its lead, then its tokens.
    #[inline]
    pub(in super::super) fn span(&self) -> u32 {
        self.lead + self.usage
    }

    /// Whether every band is still native — in a format the paged stores
    /// write. A chunk sealed into a quantized format is read-only whatever room
    /// it has left.
    pub(crate) fn takes_writes(&self) -> bool {
        self.k_fmt
            .iter()
            .all(|&t| ArenaFormatTag::from_u8(t).takes_active_k_writes())
            && self
                .v_fmt
                .iter()
                .all(|&t| ArenaFormatTag::from_u8(t).takes_active_v_writes())
    }

    /// Iterate `(gid, format tag)` for every band slot, in the interleaved
    /// K,V order of [`HeadGids::as_slice`].
    ///
    /// The live-chunk twin of [`SealedChunk::bands`], sharing its indexing so
    /// a chunk answers the same way before and after it is sealed.
    pub(crate) fn bands(&self) -> impl Iterator<Item = (&ChunkGid, ArenaFormatTag)> + '_ {
        band_tags(&self.gids, &self.k_fmt, &self.v_fmt)
    }

    /// Total bytes this chunk's bands occupy, each from its own format tag.
    pub(crate) fn byte_size(&self, elems_per_chunk: usize) -> u64 {
        self.gids
            .arena_byte_size(&self.k_fmt, &self.v_fmt, elems_per_chunk)
    }
}
