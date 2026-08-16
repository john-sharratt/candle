//! Core types for chunked KV cache.
//!
//! This module contains the fundamental types used throughout the chunked
//! KV cache implementation:
//! - `ChunkMeta` - Packed per-block metadata passed to CUDA paged-attention kernels
//! - `ChunkGid` / `ChunkGidPool` - RAII global chunk IDs with automatic free-on-drop
//! - `SealedChunk` / `SealedSequence` - Immutable snapshot of sealed history
//! - `ChunkWindow` - Per-block chunk reference with window geometry
//! - `SequenceState` - Per-sequence allocation and block ownership state
//! - `BlockTableState` - Global chunk allocation state

// ============================================================================
// Constants
// ============================================================================

/// Tokens per physical arena chunk
pub const CHUNK_SIZE: usize = 32;

/// Target size for one arena allocation: 16 MiB.
pub const TARGET_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Re-exported so arena sizing and gid decode read the same constant. The
/// stride lives with the size-class ladder because that is what bounds it.
pub use super::size_class::GID_STRIDE;

use super::gid_pool::ChunkGid;
use super::gpu_chunks::GpuChunks;
use super::head_gids::{band_tags, HeadGids};
use super::meta_pool::MetaGid;
use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo};
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════════
// ChunkMeta — packed per-block metadata for paged attention kernels
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// SealedChunk / SealedSequence — immutable snapshot of sealed history
// ═══════════════════════════════════════════════════════════════════════════════

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

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SealedChunk {
    /// Per-head RAII chunk IDs, length `2 * n_kv_head`.
    /// Indexed as `head * 2 + is_value` (0 = K, 1 = V).
    pub gids: HeadGids,
    /// Start position within the physical chunk where this window begins.
    pub offset: u16,
    /// Number of valid tokens in this window (from `offset`).
    pub token_count: u16,
    /// Packed K palette maps, `n_kv_head × (head_dim/4)` bytes. Empty = identity palette.
    pub k_pal: Arc<Vec<u8>>,
    /// Packed V palette maps, `n_kv_head × (head_dim/4)` bytes. Empty = identity palette.
    pub v_pal: Arc<Vec<u8>>,
    /// Outer K scales, `n_kv_head × N_PALETTE` f32 values. Empty = all 1.0.
    /// Encoder multiplies values by this before quantizing; decoder divides
    /// dequantized values by this to recover original magnitude.
    pub k_scale: Arc<Vec<f32>>,
    /// Outer V scales, `n_kv_head × N_PALETTE` f32 values. Same convention as k_scale.
    pub v_scale: Arc<Vec<f32>>,
    /// Per-`(head, palette)` K-side storage format, as
    /// [`ArenaFormatTag::as_u8`], `n_kv_head × N_PALETTE` entries in
    /// `[h * N_PALETTE + p]` order. Empty ⇒ not yet recorded.
    ///
    /// **The format travels with the chunk, not with the arena.** Historically
    /// a band's format was recovered by looking up the arena its gid pointed
    /// into, which made "format ⇒ arena" a load-bearing invariant and forced
    /// one arena pool per format. Under size classes an arena holds whatever
    /// fits its stride, so the arena can no longer answer the question and the
    /// chunk must. See `docs/archived/arena_unification.md` §2 and principle 8.
    pub k_fmt: Arc<Vec<u8>>,
    /// Per-`(head, palette)` V-side storage format. Same layout as `k_fmt`.
    pub v_fmt: Arc<Vec<u8>>,
    /// Total bytes occupied by all unique physical chunks referenced by `gids`.
    /// Sum of `chunk_byte_stride` over each distinct arena index in the GID set.
    /// Preserved unchanged through CPU↔GPU migration (format is identical, only
    /// location differs).  Zero for diagnostic/test-only chunks.
    pub byte_size: u64,
    /// Co-resident KV-head metadata record handle. `Some` when this chunk's
    /// `KvHead[n_kv_head]` record is resident in a device meta-pool slab (built
    /// at quantize / GPU migration); the attention kernels read it via the
    /// slice's `kvheads_ptr`. `None` ⇒ no resident record (float/transient or a
    /// not-yet-promoted chunk); the host serializer builds per-forward scratch
    /// heads instead. See [`super::meta_pool`].
    pub meta: Option<MetaGid>,
}

impl SealedChunk {
    /// The chunk's per-`(head, palette)` K and V format tags, in the
    /// `[h * N_PALETTE + p]` order [`ChunkPayload`] persists and the cold-load
    /// path decodes.
    ///
    /// These bytes *are* the persisted format tags — [`crate::kv_cache::KvFormat::to_tag`]
    /// is defined as `ArenaFormatTag::from_kv_format(..).as_u8()`, the same
    /// encoding recorded here — so the persist path copies them rather than
    /// re-deriving formats from arena state.
    ///
    /// Errors when the tags were never recorded. A chunk that cannot say what
    /// format its bands are in cannot be persisted or migrated, and an empty
    /// format vector would produce an image the cold-load path silently
    /// reconstructs as zero bands.
    ///
    /// [`ChunkPayload`]: https://docs.rs/candle-conversation
    pub fn format_tags(&self) -> candle::Result<(&[u8], &[u8])> {
        let want = self.gids.n_kv_head() * crate::kv_cache::arena_table::N_PALETTE;
        if self.k_fmt.len() != want || self.v_fmt.len() != want {
            return Err(candle::Error::Msg(format!(
                "sealed chunk has no recorded format tags (k {}, v {}, expected {want}) — \
                 every construction site must propagate them",
                self.k_fmt.len(),
                self.v_fmt.len(),
            )));
        }
        Ok((self.k_fmt.as_slice(), self.v_fmt.as_slice()))
    }

    /// Iterate `(gid, format tag)` for every band slot, in the interleaved
    /// K,V order of [`HeadGids::as_slice`].
    ///
    /// The gid says *where* the band's bytes are; the tag says *how to read
    /// them*. Under size classes only the chunk can answer the second question
    /// (`docs/archived/arena_unification.md` principle 8), so predicates over a chunk's
    /// formats walk this rather than the arenas its gids point into.
    pub fn bands(&self) -> impl Iterator<Item = (&ChunkGid, ArenaFormatTag)> + '_ {
        band_tags(&self.gids, &self.k_fmt, &self.v_fmt)
    }

    /// Total bytes this chunk's bands occupy, each from its own format tag.
    pub fn byte_size(&self, elems_per_chunk: usize) -> u64 {
        self.gids
            .arena_byte_size(&self.k_fmt, &self.v_fmt, elems_per_chunk)
    }

    /// Construct a minimal `SealedChunk` for use in tests.
    ///
    /// Creates a single-element `gids` vec (n_kv_head=1), with `offset`
    /// set to 0 and empty palette fields.
    pub fn for_test(chunk_id: i64, token_count: u16) -> Self {
        SealedChunk {
            gids: HeadGids::uniform(ChunkGid::detached(chunk_id), 1),
            offset: 0,
            token_count,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(Vec::new()),
            v_fmt: Arc::new(Vec::new()),
            byte_size: 0,
            meta: None,
        }
    }
}

/// Immutable snapshot of a sequence's turn history.
///
/// Each `SealedChunk` entry may cover only a sub-window of its physical chunk
/// (with non-zero `offset`), allowing the same physical chunk to be referenced
/// by multiple turns.  Can be `Arc`-wrapped for sharing across forked sequences.
///
/// # Position-agnostic by construction
///
/// A `SealedSequence` carries no positional state: no absolute RoPE
/// base, no parent-sequence offset, no "where this came from" cursor.
/// K bytes in the underlying chunks are stored un-rotated (see
/// [`SealedChunk`]); RoPE is applied at the latest responsible moment
/// inside the attention kernel using a `slice_rope(...)` value that
/// the host recomputes from cumulative usage of preceding blocks in
/// the *current slot*'s layout (see
/// `gpu_chunks::GpuChunksGuard::rebuild_decode`).
///
/// Consequence: the same `SealedSequence` can be injected into any
/// destination slot at any tail position and the model will see
/// correctly-positioned K/V.  No re-RoPE pass, no byte migration, no
/// CoW — just `inject_sealed_at_tail`'s metadata clone.
///
/// # Partial trailing chunk
///
/// The last entry in `chunks` may be a partial chunk (`usage <
/// CHUNK_SIZE`).  `record_turn` deliberately seals it: dropping it
/// would silently lose up to `(sections - 1) * (CHUNK_SIZE - 1)`
/// tokens when sections are projected back-to-back via
/// `inject_sealed_at_tail` (each section's partial tail becomes a
/// gap in the destination's KV).
///
/// Sharing a partial chunk via `Arc<ChunkGid>` is safe because the
/// architecture guarantees **at most one writer per partial chunk**:
/// the unique sequence resumed on the chunk's owning
/// `(layer, group, instance)` target.  The resume-check rejects a
/// second attempt to bind a writer to the same target, so concurrent
/// extension can't happen.  All other holders (the substrate's
/// metadata pointer, BDP scans, sibling slots that projected the
/// section / turn purely as read context) are read-only and observe
/// the writer's extensions through the Arc — that mutability is
/// intended, not aliased corruption.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SealedSequence {
    /// Ordered list of chunk windows for this sequence's committed
    /// history, in token order.  The trailing entry may be partial
    /// (`token_count < CHUNK_SIZE`); see the type-level docs for why
    /// that's intentional and how single-writer-per-partial-chunk
    /// keeps it safe to share.
    pub chunks: Vec<SealedChunk>,
    /// Total number of valid tokens across all windows.
    pub token_count: usize,
    /// The chunk size used when recording (for consistency checks).
    pub chunk_size: usize,
    /// Coarse-grained tier tag for the sequence.  Every chunk's
    /// `ChunkGid` resolves to an arena at this same location — the
    /// field exists for fast O(1) tier classification without walking
    /// every chunk's `route_key`.  Set at construction by
    /// `record_turn` (or whatever produces the sequence) and never
    /// mutated; future tier-migration paths produce a *new*
    /// `SealedSequence` with a different `location`.
    pub location: super::ArenaLocation,
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
/// RoPE positions are not stored — they are recomputed as the cumulative token
/// count of all preceding blocks.
#[derive(Debug, Clone)]
pub(crate) struct ChunkWindow {
    /// RAII chunk IDs, length `2 * n_kv_head`.
    /// Indexed as `head * 2 + is_value` (0 = K, 1 = V).
    pub(crate) gids: HeadGids,
    /// Valid token count in this chunk's window.
    pub(super) usage: u32,
    /// Skip-count from the start of the physical chunk where valid data begins.
    pub(super) offset: u16,
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
}

impl ChunkWindow {
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

/// One layer's view of a sequence's block structure, reduced to the parts the
/// shared decode position map is built from.
///
/// Two layers whose `DecodeLayout` agree produce identical maps and can share
/// one; two that disagree cannot. Produced by
/// [`SequenceState::decode_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeLayout {
    /// Fold over every chunk's `(offset, usage)` window, in chunk order.
    pub(super) digest: u64,
    /// Number of allocated chunks — the slice count the map indexes into.
    pub(super) blocks: usize,
    /// Index of the chunk the write slot resolves to. This is the one field
    /// the map records for a token that does not exist yet, and the one the
    /// per-layer `write_slice` in the slot header must match.
    pub(super) writer: usize,
}

/// Opaque snapshot of a slot's writer-owned chunks, taken by
/// [`super::ChunkedKvBacking::split_off_writer_tail`] and restored by
/// [`super::ChunkedKvBacking::extend_writer_tail`].
///
/// Used by the stateless-slot rebuild path: the scheduler snapshots
/// the in-flight decode chunks before truncating the slot, re-injects
/// the projected prefix from substrate, and restores the tail in the
/// same call. RAII on the contained `ChunkGid`s keeps the underlying
/// arena chunks alive across the truncate — the bytes never move.
pub struct WriterTail {
    pub(super) chunks: Vec<ChunkWindow>,
}

impl WriterTail {
    /// `true` when the snapshot contains no chunks (the common
    /// turn-boundary case where decode hasn't started yet).
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Number of chunks in the snapshot.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeGpuChunksSyncKind {
    Empty,
    Rebuild,
    Reuse,
}

/// Number of front chunks fully outside the sliding window — the pure
/// decision behind [`SequenceState::evict_front_window`], factored out so the
/// off-by-one boundary is unit-testable without arenas.
///
/// `usages[i]` is chunk `i`'s valid-token count; `writer_start` bounds how far
/// the sweep may go (never evicts the writer or a shared prefix); `base_pos` is
/// the absolute position of the first resident token; the window ends at
/// `abs_pos` and spans `window_size` tokens. A chunk at absolute
/// `[start, start+usage)` is evictable iff `start + usage ≤ abs_pos −
/// window_size + 1` (its highest position is below the lowest in-window one).
pub(crate) fn front_evict_count(
    usages: &[u32],
    writer_start: usize,
    base_pos: u32,
    window_size: usize,
    abs_pos: usize,
) -> usize {
    if window_size == 0 {
        return 0;
    }
    // Lowest absolute position still inside the window.
    let lo = abs_pos.saturating_sub(window_size) + 1;
    let max_evict = writer_start.min(usages.len());
    let mut start = base_pos as usize;
    let mut drained = 0usize;
    for &usage in usages.iter().take(max_evict) {
        let end = start + usage as usize; // one past the chunk's last token
        if end <= lo {
            drained += 1;
            start = end;
        } else {
            break;
        }
    }
    drained
}

#[derive(Debug, Clone)]
pub(crate) struct SequenceState {
    /// All blocks in token order.  Shared (prefix) and owned chunks
    /// coexist in the same vec.  Partial tails are always copied at fork
    /// time, so `chunks.last()` is always uniquely owned and writable.
    chunks: Vec<ChunkWindow>,
    /// Cached pinned-host + GPU-side serialised slot-state for this sequence.
    gpu_chunks: GpuChunks,
    /// First chunk index that is writer-owned.  Chunks at index
    /// `[0, writer_start_idx)` are Arc-shared with substrate / parent
    /// and MUST NOT be extended by the writer.  Chunks at index
    /// `[writer_start_idx, chunks.len())` are uniquely owned and the
    /// kernel may write into them.
    ///
    /// Set by:
    /// - `truncate_chunks(0)`: → 0
    /// - `inject_sealed_at_tail`: advanced by N (injected chunks are
    ///   Arc-shared, writer starts past them)
    /// - `create_view_sequence` (CoW path): → view_chunk_count − 1 (the
    ///   CoW partial is uniquely-copied writer-owned)
    /// - `push_empty_writer_chunk`: no change (it pushes a chunk past
    ///   the current writer boundary; the chunk itself is writer-owned
    ///   by construction)
    writer_start_idx: usize,
    /// Absolute token position of the FIRST resident chunk's first token —
    /// i.e. the count of tokens evicted off the front by the sliding-window
    /// ring ([`Self::evict_front_window`]). Every chunk's serialised
    /// `rope_base` and every [`Self::rope_pos`] is offset by this, so window
    /// keys keep their ABSOLUTE positions after the front slides out. Zero for
    /// every non-windowed slot (dialogue/section KV never evicts its front), so
    /// those paths are byte-identical to a `base_pos == 0` derivation.
    base_pos: u32,
}

impl SequenceState {
    #[cfg(feature = "cuda")]
    pub(super) fn new(stream: Option<Arc<CudaStream>>) -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(stream),
            writer_start_idx: 0,
            base_pos: 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(),
            writer_start_idx: 0,
            base_pos: 0,
        }
    }

    #[inline]
    pub(crate) fn writer_start_idx(&self) -> usize {
        self.writer_start_idx
    }

    #[inline]
    pub(crate) fn set_writer_start_idx(&mut self, idx: usize) {
        self.writer_start_idx = idx;
    }

    /// Total allocated blocks.
    #[inline]
    pub(super) fn block_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total token count across all chunks.
    #[inline]
    pub(super) fn seq_len(&self) -> usize {
        self.chunks.iter().map(|c| c.usage as usize).sum()
    }

    /// The sliding-window ring's evicted-front count — the absolute position of
    /// the first resident token. Zero until the ring first slides.
    #[inline]
    pub(super) fn base_pos(&self) -> u32 {
        self.base_pos
    }

    /// Seed the evicted-front count directly (turn-seal ring restore): the
    /// remaining resident chunks then serialise ABSOLUTE `rope_base` positions
    /// from this base (`rope_pos = base_pos + Σ preceding usage`), so a resumed
    /// window continues the original absolute frame.
    #[inline]
    pub(super) fn set_base_pos(&mut self, v: u32) {
        self.base_pos = v;
    }

    /// Get the `ChunkWindow` for block `blk`.
    #[inline]
    pub(super) fn chunk_at(&self, blk: usize) -> Option<&ChunkWindow> {
        self.chunks.get(blk)
    }

    /// Replace the full per-head GID vector AND the per-block palette/scale
    /// metadata at block `blk`.
    ///
    /// `gids` must have length `GIDS_PER_HEAD * n_kv_head`.
    /// `k_pal` / `v_pal` must each be `n_kv_head × (head_dim/4)` bytes (or empty
    /// for identity routing).
    /// `k_scale` / `v_scale` must each be `n_kv_head × N_PALETTE` f32 values
    /// (or empty for unity scale = no outer scaling).
    ///
    /// Pal/scale **must** be passed explicitly — they describe how to interpret
    /// the bytes stored at `gids`, so they have to track every gid mutation.
    /// Callers that want to preserve existing semantics should read the chunk's
    /// current values first and pass them through.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn set_block_gids(
        &mut self,
        blk: usize,
        gids: HeadGids,
        k_pal: Arc<Vec<u8>>,
        v_pal: Arc<Vec<u8>>,
        k_scale: Arc<Vec<f32>>,
        v_scale: Arc<Vec<f32>>,
        k_fmt: Arc<Vec<u8>>,
        v_fmt: Arc<Vec<u8>>,
    ) {
        if blk < self.chunks.len() {
            let cw = &mut self.chunks[blk];
            cw.gids = gids;
            cw.k_pal = k_pal;
            cw.v_pal = v_pal;
            cw.k_scale = k_scale;
            cw.v_scale = v_scale;
            // The format tags travel with the gids and MUST be replaced with
            // them. A cold-load or elevate re-points this window at chunks in
            // whatever formats were persisted; leaving the window's previous
            // tags in place (the active R16/F16 a freshly-allocated writer
            // window carries) would make every reader decode those chunks as
            // raw floats. The pal/scale fields above have the same lifecycle
            // for the same reason.
            cw.k_fmt = k_fmt;
            cw.v_fmt = v_fmt;
            // Any GID mutation (defrag remap, cold-load reinjection) invalidates
            // the resident record's per-palette pointers. Drop it so the host
            // serializer falls back to per-forward scratch heads rather than
            // emitting a stale `kvheads_ptr`.
            cw.meta = None;
        }
    }

    /// Compute the ABSOLUTE RoPE base position for block `blk`: the count of
    /// tokens evicted off the front (`base_pos`) plus the cumulative token
    /// count of all preceding resident blocks. `base_pos` is zero until the
    /// sliding-window ring evicts, so a non-windowed slot reads exactly the
    /// cumulative-usage sum it always did.
    #[inline]
    pub(super) fn rope_pos(&self, blk: usize) -> i32 {
        let count = blk.min(self.chunks.len());
        self.base_pos as i32
            + self
                .chunks
                .iter()
                .take(count)
                .map(|c| c.usage as i32)
                .sum::<i32>()
    }

    /// Slide the sliding-window ring: drop every FRONT chunk that has fully
    /// exited the `window_size`-token window ending at absolute query position
    /// `abs_pos`, returning the new [`Self::base_pos`] (total evicted tokens).
    ///
    /// A query at `abs_pos` attends window keys with `key_pos > abs_pos −
    /// window_size` (the kernel's causal+window mask), so the lowest in-window
    /// absolute position is `abs_pos − window_size + 1`. A front chunk spanning
    /// absolute `[start, start+usage)` is fully out of window once
    /// `start + usage ≤ abs_pos − window_size + 1` — none of its tokens can be
    /// attended, so freeing it changes nothing the kernel reads while bounding
    /// the resident set (and the per-step tile walk) to `O(window_size)`.
    /// Older tokens are already folded into the compressed corpus (guaranteed
    /// by `window_size ≥ compress_ratio`), so no attended information is lost.
    ///
    /// Never touches the writer chunk or anything at/after `writer_start_idx`.
    /// Draining bumps `base_pos` (keeping remaining chunks' absolute positions
    /// intact via [`Self::rope_pos`] / the serialised `rope_base`) and shifts
    /// the writer boundary down by the number dropped.
    pub(crate) fn evict_front_window(&mut self, window_size: usize, abs_pos: usize) -> u32 {
        let usages: Vec<u32> = self.chunks.iter().map(|c| c.usage).collect();
        let drained = front_evict_count(
            &usages,
            self.writer_start_idx,
            self.base_pos,
            window_size,
            abs_pos,
        );
        if drained > 0 {
            let evicted: u32 = usages[..drained].iter().copied().sum();
            self.drain_front_chunks(drained); // RAII-frees GIDs + clears GPU buffer
            self.base_pos = self.base_pos.wrapping_add(evicted);
            self.writer_start_idx = self.writer_start_idx.saturating_sub(drained);
        }
        self.base_pos
    }

    // -----------------------------------------------------------------------
    // Read-only accessors
    // -----------------------------------------------------------------------

    /// `true` when no chunks have been allocated yet.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Borrow the last `ChunkWindow` (the active tail), if any.
    #[inline]
    pub(crate) fn last_chunk(&self) -> Option<&ChunkWindow> {
        self.chunks.last()
    }

    /// Mutably borrow the last `ChunkWindow` (the active tail), if any.
    #[inline]
    pub(crate) fn last_chunk_mut(&mut self) -> Option<&mut ChunkWindow> {
        self.chunks.last_mut()
    }

    /// Mutably borrow the `ChunkWindow` at block `blk`, if any.
    #[inline]
    pub(crate) fn chunk_at_mut(&mut self, blk: usize) -> Option<&mut ChunkWindow> {
        self.chunks.get_mut(blk)
    }

    /// Return all chunks as a slice for iteration / indexing.
    #[inline]
    /// Mutable view of all chunks (test helper / direct manipulation).
    #[allow(dead_code)]
    pub(crate) fn chunks_slice_mut(&mut self) -> &mut [ChunkWindow] {
        &mut self.chunks
    }

    pub(crate) fn chunks_slice(&self) -> &[ChunkWindow] {
        &self.chunks
    }

    /// Cheap host-side validation for decode.
    ///
    /// This mirrors the kernel's write-slice invariant and rejects obviously
    /// stale slot-state before the paged decode kernel is launched.
    pub(crate) fn validate_decode_state(
        &self,
        batch_idx: usize,
        seq_offset: usize,
    ) -> candle::Result<()> {
        let Some(tail) = self.last_chunk() else {
            candle::bail!(
                "chunked decode validation failed for batch_idx {}: sequence has no allocated tail chunk",
                batch_idx
            );
        };

        let host_n = self.block_count();
        let cached_n = self.gpu_chunks.n_chunks();
        if cached_n != 0 && cached_n != host_n {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: cached decode slot count {} does not match host chunk count {}",
                batch_idx,
                seq_offset,
                cached_n,
                host_n
            );
        }

        let last_rope_base: usize = if host_n > 1 {
            self.chunks[..host_n - 1]
                .iter()
                .map(|c| c.usage as usize)
                .sum()
        } else {
            0
        };
        let expected_write_len = seq_offset.saturating_sub(last_rope_base);
        if expected_write_len >= CHUNK_SIZE {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: computed write len {} is invalid for chunk_size {} (tail_offset={}, host_tail_usage={})",
                batch_idx,
                seq_offset,
                expected_write_len,
                CHUNK_SIZE,
                tail.offset,
                tail.usage
            );
        }

        let tail_fill = tail.offset as usize + expected_write_len;
        if tail_fill >= CHUNK_SIZE {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: writable tail is already full/stale (tail_offset={}, expected_write_len={}, host_tail_usage={}, chunk_size={})",
                batch_idx,
                seq_offset,
                tail.offset,
                expected_write_len,
                tail.usage,
                CHUNK_SIZE
            );
        }

        Ok(())
    }

    /// Clear the GPU slot-state buffer, forcing a full rebuild on the next decode step.
    ///
    /// Called after prefill to mark the buffer stale — the prefill kernel wrote
    /// KV data directly and may have changed chunk layout without going through
    /// the decode self-increment path.
    pub(crate) fn invalidate_gpu_chunks(&mut self) {
        self.gpu_chunks.as_mut().clear();
    }

    /// Re-serialise just the WRITER chunk's slice in the cached decode GPU
    /// buffer after a mid-decode prefill wrote tokens into that chunk in
    /// place (a stencil static run, a think-steer continuation).
    ///
    /// The prefill write path keeps host state authoritative: `set_len` tops
    /// the writer chunk's usage up to the sequence length, and a
    /// chunk-boundary append clears the whole buffer at the mutation site
    /// (`push_chunk`). A live buffer here therefore differs from host state
    /// only in this one slice, and patching it under the guard (async H→D on
    /// drop) is the O(1) alternative to dropping and re-uploading the entire
    /// per-layer table — which at depth costs megabytes of pinned realloc and
    /// a stream sync per layer. A missing buffer is left for the next decode
    /// sync's full rebuild; a shape mismatch (defensive) falls back to full
    /// invalidation.
    pub(crate) fn refresh_decode_writer_slice(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        let gpu_n = self.gpu_chunks.n_chunks();
        if gpu_n == 0 {
            return Ok(());
        }
        if gpu_n != self.chunks.len() {
            self.invalidate_gpu_chunks();
            return Ok(());
        }
        let wi = self.decode_write_chunk_idx();
        if wi >= gpu_n {
            self.invalidate_gpu_chunks();
            return Ok(());
        }
        self.update_gpu_chunk(wi, n_kv_head, head_dim, arena_info)
    }

    /// Re-serialise the GPU buffer slot at `blk` from the current host state
    /// (including updated GIDs), scheduling an async H→D copy on the guard drop.
    ///
    /// No-op if the GPU buffer has not been initialised yet (avoids errors for
    /// sequences that haven't gone through their first decode step).
    pub(super) fn update_gpu_chunk(
        &mut self,
        blk: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        if self.gpu_chunks.n_chunks() == 0 {
            return Ok(());
        }
        let rope_base: u32 =
            self.base_pos + self.chunks[..blk].iter().map(|c| c.usage).sum::<u32>();
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        gpu_chunks.as_mut().update_chunk(
            blk,
            &chunks[blk],
            n_kv_head,
            head_dim,
            rope_base,
            arena_info,
        )
    }

    /// Bulk variant of [`Self::update_gpu_chunk`] — serialises every
    /// block in `block_indices` under a **single** `GpuChunksGuard`,
    /// so the guard drop coalesces adjacent indices into one
    /// `memcpy_htod` per contiguous run (instead of one per block).
    ///
    /// Used by the cold-load `alloc_sealed_blocks_bulk` path to push
    /// the per-layer chunk metadata to the GPU as a single batched
    /// HtoD where possible.
    pub(super) fn update_gpu_chunks_bulk(
        &mut self,
        block_indices: &[usize],
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        if block_indices.is_empty() || self.gpu_chunks.n_chunks() == 0 {
            return Ok(());
        }
        // Seeded at `base_pos` so evicted-front slots keep absolute rope
        // positions; captured before the disjoint-field destructure below.
        let base_pos = self.base_pos;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        // Prefix-sum cumulative usage once so each block's rope_base
        // is an O(1) lookup. The previous `chunks[..blk].iter().sum()`
        // was O(blk) per block — quadratic over a layer's blocks.
        let mut rope_bases: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut acc: u32 = base_pos;
        for c in chunks.iter() {
            rope_bases.push(acc);
            acc = acc.wrapping_add(c.usage);
        }
        // One guard, scoped over the whole loop — its drop coalesces
        // every dirty chunk into the fewest contiguous memcpy_htod
        // runs the index set allows.
        let mut guard = gpu_chunks.as_mut();
        for &blk in block_indices {
            guard.update_chunk(
                blk,
                &chunks[blk],
                n_kv_head,
                head_dim,
                rope_bases[blk],
                arena_info,
            )?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Chunk-only mutation primitives (no GPU params required)
    //
    // Structural operations eagerly reset the GPU backing buffer because the
    // cached decode slot state is trusted on the hot path. Any change to the
    // chunk vector shape therefore must invalidate the cache immediately at the
    // mutation site rather than relying on decode-time mismatch detection.
    // -----------------------------------------------------------------------

    /// Append one `ChunkWindow` to the tail and invalidate the cached GPU slot
    /// buffer so the next decode rebuilds from the new structural state.
    #[inline]
    pub(crate) fn push_chunk(&mut self, cw: ChunkWindow) {
        self.chunks.push(cw);
        self.gpu_chunks.as_mut().clear();
    }

    /// Clear all chunks and clear the GPU buffer.
    pub(crate) fn clear_chunks(&mut self) {
        self.chunks.clear();
        self.gpu_chunks.as_mut().clear();
    }

    /// Drain the first `n` chunks (RAII-drops their GIDs) and clear the GPU buffer.
    pub(crate) fn drain_front_chunks(&mut self, n: usize) {
        let n = n.min(self.chunks.len());
        self.chunks.drain(..n).for_each(drop);
        if n > 0 {
            self.gpu_chunks.as_mut().clear();
        }
    }

    /// Prepend `prefix` before all existing chunks; GPU buffer is cleared.
    pub(crate) fn prepend_chunks(&mut self, mut prefix: Vec<ChunkWindow>) {
        prefix.extend(std::mem::take(&mut self.chunks));
        self.chunks = prefix;
        self.gpu_chunks.as_mut().clear();
    }

    /// Replace all chunks with `new_chunks` and clear the GPU buffer.
    pub(crate) fn replace_chunks(&mut self, new_chunks: Vec<ChunkWindow>) {
        self.chunks = new_chunks;
        self.gpu_chunks.as_mut().clear();
    }

    /// Drain all chunks from index `at` onward into a `Vec`.  GPU buffer is
    /// cleared because chunk count changes.
    pub(crate) fn split_off_chunks(&mut self, at: usize) -> Vec<ChunkWindow> {
        let tail = self.chunks.split_off(at);
        self.gpu_chunks.as_mut().clear();
        tail
    }

    /// Truncate to `n` chunks.  If the length actually decreases the GPU
    /// buffer is cleared.
    pub(crate) fn truncate_chunks(&mut self, n: usize) {
        if n < self.chunks.len() {
            self.chunks.truncate(n);
            self.gpu_chunks.as_mut().clear();
        }
    }

    /// Extend with additional `ChunkWindow`s and invalidate the cached GPU
    /// slot buffer because the chunk layout has changed.
    pub(crate) fn extend_chunks(&mut self, iter: impl IntoIterator<Item = ChunkWindow>) {
        self.chunks.extend(iter);
        self.gpu_chunks.as_mut().clear();
    }

    /// Rebuild the GPU slot-state buffer for decode using the true sequence length.
    ///
    /// Serialises all chunks into the pinned host buffer with per-chunk
    /// `rope_base` values derived from cumulative usage, then uploads to the
    /// device buffer asynchronously.
    ///
    /// Returns `(raw_device_ptr, n_chunks, write_chunk_idx)` where:
    /// - `raw_device_ptr` is the GPU base pointer for the decode kernel,
    /// - `n_chunks` is the number of serialised chunk entries,
    /// - `write_chunk_idx` is the index of the last (writable) chunk.
    ///
    /// `seq_offset` is the current sequence length used to derive the true
    /// token count for the write chunk (overrides the potentially-stale
    /// `chunk.usage` field).
    /// Index of the chunk the decode kernel writes into: the first non-full
    /// chunk at or after `writer_start_idx`. Chunks after it are trailing
    /// empties — e.g. a freshly-appended empty writer sitting past a partial
    /// sealed chunk — and must be skipped. This is the same selection rule
    /// `set_len` and the position_map use, so the K/V write, the rope base,
    /// and attention all agree on which chunk is the writer.
    pub(super) fn decode_write_chunk_idx(&self) -> usize {
        let n = self.chunks.len();
        if n == 0 {
            return 0;
        }
        let start = self.writer_start_idx().min(n - 1);
        for i in start..n {
            let c = &self.chunks[i];
            if (c.offset as usize + c.usage as usize) < CHUNK_SIZE {
                return i;
            }
        }
        n - 1
    }

    /// Everything the shared decode position map is derived from, reduced to a
    /// comparable value.
    ///
    /// The decode metadata builder constructs ONE position map per sequence — a
    /// `(slice_idx, in_blk)` entry per logical token, plus a final entry naming
    /// the write slot — from layer 0's block table, and hands it to all 48
    /// layers. That is only sound while every layer's block table agrees, and
    /// block structure is not unconditionally layer-uniform: a windowed creep
    /// prefill leaves resumed layers holding an empty writer chunk the layers
    /// still pending resume do not have. Comparing this value across layers is
    /// how the decode entry point establishes the invariant instead of assuming
    /// it. See [`super::ChunkedKvBacking::ensure_for_batch_entries_all`].
    pub(super) fn decode_layout(&self) -> DecodeLayout {
        // FNV-1a over each chunk's window. Order matters and the windows are the
        // whole of what the map encodes, so a fold over `(offset, usage)` in
        // chunk order captures exactly as much as the map does and no more.
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for c in &self.chunks {
            for field in [c.offset as u64, c.usage as u64] {
                digest ^= field;
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        DecodeLayout {
            digest,
            blocks: self.chunks.len(),
            writer: self.decode_write_chunk_idx(),
        }
    }

    /// Per-chunk `(offset, len, cum_before)` window using the SAME derivation the
    /// decode GPU buffer does ([`rebuild_decode`]): the writer chunk gets the
    /// `seq_offset`-derived length, every other chunk keeps its stored `usage`, and
    /// `offset` is the physical skip-count where valid data begins. This is exactly
    /// the real-token window attention reads, so a provenance / diagnostic gather can
    /// consult it to check only real slots and skip partial-chunk padding. Returned
    /// in chunk order; `cum_before` is the running real-token count preceding a chunk.
    pub(crate) fn provenance_chunk_layout(&self, seq_offset: usize) -> Vec<(u16, u16, usize)> {
        let n = self.chunks.len();
        if n == 0 {
            return Vec::new();
        }
        let wi = self.decode_write_chunk_idx();
        let before_wi: usize = self.chunks[..wi].iter().map(|c| c.usage as usize).sum();
        let write_len = seq_offset.saturating_sub(before_wi).min(CHUNK_SIZE);
        let mut out = Vec::with_capacity(n);
        let mut cum = 0usize;
        for (i, c) in self.chunks.iter().enumerate() {
            let len = if i == wi {
                write_len as u16
            } else {
                (c.usage as usize).min(CHUNK_SIZE) as u16
            };
            out.push((c.offset, len, cum));
            cum += len as usize;
        }
        out
    }

    pub(crate) fn rebuild_decode_gpu_chunks(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        seq_offset: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<(u64, u32, u32)> {
        let n = self.chunks.len();
        if n == 0 {
            return Ok((0, 0, 0));
        }
        // The writer is the first non-full chunk from `writer_start_idx`, NOT
        // `host_n - 1`: trailing empty chunks (a fresh writer past a sealed
        // partial chunk) must be skipped. Its rope base is the cum-token sum of
        // the chunks before it, and its GPU length is the seq_offset-derived
        // fill (overriding the possibly-stale stored usage).
        let wi = self.decode_write_chunk_idx();
        let before_wi: usize = self.chunks[..wi].iter().map(|c| c.usage as usize).sum();
        let write_len = seq_offset.saturating_sub(before_wi) as u16;

        // Borrow chunks and gpu_chunks as disjoint fields simultaneously.
        let base_pos = self.base_pos;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        gpu_chunks.as_mut().rebuild_decode(
            chunks, n_kv_head, head_dim, arena_info, write_len, wi, base_pos,
        )?;

        let ptr = self.gpu_chunks.raw_device_ptr();
        Ok((ptr, n as u32, wi as u32))
    }

    /// Synchronise the GPU slot-state buffer for decode.
    ///
    /// The decode hot path trusts the cached GPU slot buffer whenever it exists.
    /// Structural mutations must therefore invalidate it eagerly at the point of
    /// mutation rather than relying on decode-time mismatch heuristics.
    pub(crate) fn sync_decode_gpu_chunks(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        seq_offset: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<((u64, u32, u32), DecodeGpuChunksSyncKind)> {
        let host_n = self.chunks.len();
        if host_n == 0 {
            return Ok(((0, 0, 0), DecodeGpuChunksSyncKind::Empty));
        }

        if self.gpu_chunks.n_chunks() == 0 {
            let rebuilt =
                self.rebuild_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
            return Ok((rebuilt, DecodeGpuChunksSyncKind::Rebuild));
        }

        let wi = self.decode_write_chunk_idx();
        let ptr = self.gpu_chunks.raw_device_ptr();
        Ok((
            (ptr, host_n as u32, wi as u32),
            DecodeGpuChunksSyncKind::Reuse,
        ))
    }

    /// Snapshot the current slot-state into the stager `generation`, returning
    /// the immutable copy's device pointer. Call after `sync_decode_gpu_chunks`
    /// has brought `gpu_chunks` to the desired offset. See
    /// [`super::gpu_chunks::GpuChunks::snapshot_into_generation`].
    pub(crate) fn snapshot_gpu_chunks_into(
        &mut self,
        generation: &candle::quantized::pinned_staging::Generation,
        seq_offset: usize,
    ) -> candle::Result<u64> {
        // The write chunk's per-token length is not carried in the live buffer
        // between snapshots (see `snapshot_into_generation`); derive it from the
        // sequence offset (tokens already in the write chunk before this token).
        let wi = self.decode_write_chunk_idx();
        let rope_base: usize = self.chunks[..wi.min(self.chunks.len())]
            .iter()
            .map(|c| c.usage as usize)
            .sum();
        let write_len = seq_offset.saturating_sub(rope_base) as u16;
        self.gpu_chunks
            .snapshot_into_generation(generation, wi, write_len)
    }
}

#[cfg(test)]
mod band_iteration_tests {
    use super::*;
    use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    /// One head, all bands sharing a gid, with the given K/V tags.
    fn window(k_tag: u8, v_tag: u8) -> ChunkWindow {
        ChunkWindow {
            gids: HeadGids::uniform(ChunkGid::detached(0), 1),
            usage: 32,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(vec![k_tag; N_PALETTE]),
            v_fmt: Arc::new(vec![v_tag; N_PALETTE]),
            meta: None,
        }
    }

    /// **The K/V interleave.** `bands()` walks slots in `HeadGids` order —
    /// K,V alternating — and each slot must get the tag from its own side.
    /// Getting this backwards would decode every V band as a K format, which
    /// is exactly the class of bug the per-band tags exist to prevent.
    #[test]
    fn bands_alternate_k_and_v_tags() {
        let w = window(ArenaFormatTag::R16.as_u8(), ArenaFormatTag::F16.as_u8());
        let tags: Vec<ArenaFormatTag> = w.bands().map(|(_, t)| t).collect();

        assert_eq!(tags.len(), N_PALETTE * 2, "one slot per band, K and V");
        for (i, tag) in tags.iter().enumerate() {
            let want = if i % 2 == 0 {
                ArenaFormatTag::R16
            } else {
                ArenaFormatTag::F16
            };
            assert_eq!(*tag, want, "slot {i} took the wrong side's tag");
        }
    }

    /// A chunk that never recorded its tags reports `Invalid` rather than a
    /// plausible default. A guessed format would be decoded as real.
    #[test]
    fn unrecorded_tags_report_invalid() {
        let mut w = window(0, 0);
        w.k_fmt = Arc::new(Vec::new());
        w.v_fmt = Arc::new(Vec::new());

        assert!(
            w.bands().all(|(_, t)| t == ArenaFormatTag::Invalid),
            "absent tags must not decode to a real format"
        );
    }

    /// A live chunk and the sealed chunk it becomes must agree band-for-band —
    /// they share one indexing helper precisely so they cannot drift.
    #[test]
    fn a_window_and_its_sealed_form_agree() {
        let w = window(ArenaFormatTag::Q8_0.as_u8(), ArenaFormatTag::Q4_KS.as_u8());
        let sealed = SealedChunk {
            gids: w.gids.clone(),
            offset: w.offset,
            token_count: w.usage as u16,
            k_pal: w.k_pal.clone(),
            v_pal: w.v_pal.clone(),
            k_scale: w.k_scale.clone(),
            v_scale: w.v_scale.clone(),
            k_fmt: w.k_fmt.clone(),
            v_fmt: w.v_fmt.clone(),
            byte_size: 0,
            meta: None,
        };
        let from_window: Vec<ArenaFormatTag> = w.bands().map(|(_, t)| t).collect();
        let from_sealed: Vec<ArenaFormatTag> = sealed.bands().map(|(_, t)| t).collect();
        assert_eq!(from_window, from_sealed);
    }
}

/// Tests for the chunk-owned format tags as a *source of truth* — the accessors
/// every inverted reader now goes through.
#[cfg(test)]
mod band_tag_tests {
    use super::*;
    use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
    use crate::kv_cache::chunked::gid_pool::ChunkGid;
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    /// A two-head chunk whose every band carries a *distinct* tag, so any
    /// index arithmetic error in `bands()` shows up as a mismatched pair
    /// rather than an accidentally-correct uniform answer.
    fn distinct_band_chunk(n_kv_head: usize) -> SealedChunk {
        let n = n_kv_head * N_PALETTE;
        let mut sc = SealedChunk::for_test(0, 32);
        sc.gids = HeadGids::uniform(ChunkGid::detached(0), n_kv_head);
        // Tag values are arbitrary but unique per band, and the K and V ranges
        // are disjoint so a K/V swap is detectable.
        sc.k_fmt = Arc::new((0..n).map(|i| i as u8).collect());
        sc.v_fmt = Arc::new((0..n).map(|i| (100 + i) as u8).collect());
        sc
    }

    #[test]
    fn bands_pairs_every_slot_with_its_own_tag() {
        let n_kv_head = 2;
        let sc = distinct_band_chunk(n_kv_head);
        let got: Vec<u8> = sc.bands().map(|(_, tag)| tag.as_u8()).collect();

        // `HeadGids::as_slice` is K,V interleaved per (head, palette); tags are
        // indexed [h * N_PALETTE + p] per side. Rebuild that expectation by
        // hand rather than from the same helper being tested.
        let mut want: Vec<u8> = Vec::new();
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                want.push(ArenaFormatTag::from_u8((h * N_PALETTE + p) as u8).as_u8());
                want.push(ArenaFormatTag::from_u8((100 + h * N_PALETTE + p) as u8).as_u8());
            }
        }
        assert_eq!(
            got, want,
            "bands() must walk K,V interleaved per (head, palette)"
        );
        assert_eq!(got.len(), n_kv_head * N_PALETTE * 2);
    }

    #[test]
    fn bands_yields_the_gid_at_each_slot() {
        let sc = distinct_band_chunk(2);
        let gids: Vec<i64> = sc.bands().map(|(g, _)| g.raw()).collect();
        let want: Vec<i64> = sc.gids.as_slice().iter().map(|g| g.raw()).collect();
        assert_eq!(gids, want);
    }

    /// An unrecorded band reports `Invalid`, which fails every format
    /// predicate — a chunk that cannot say what it holds is never treated as
    /// eligible for a kernel that would misread it.
    #[test]
    fn unrecorded_bands_report_invalid() {
        let sc = SealedChunk::for_test(0, 32); // empty k_fmt / v_fmt
        assert!(sc.bands().all(|(_, t)| t == ArenaFormatTag::Invalid));
        assert!(
            sc.bands().all(|(_, t)| t.is_quantized()),
            "Invalid must not pass as a float"
        );
    }

    #[test]
    fn format_tags_rejects_an_unrecorded_chunk() {
        let sc = SealedChunk::for_test(0, 32);
        let err = sc.format_tags().unwrap_err().to_string();
        assert!(
            err.contains("no recorded format tags"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn format_tags_returns_both_sides_verbatim() {
        let sc = distinct_band_chunk(2);
        let (k, v) = sc.format_tags().unwrap();
        assert_eq!(k, sc.k_fmt.as_slice());
        assert_eq!(v, sc.v_fmt.as_slice());
    }

    /// **The load-bearing assertion for the persist path.** `ChunkPayload`'s
    /// `k_formats`/`v_formats` were built by mapping arena formats through
    /// [`KvFormat::to_tag`]; they are now the chunk's own tag bytes, copied.
    /// That substitution is byte-identical only because the two encodings are
    /// the same one — asserted here against raw byte values, per repo policy on
    /// serialization tests, rather than trusted from the type names.
    #[test]
    fn chunk_tag_bytes_are_the_persisted_format_encoding() {
        for (fmt, tag) in [
            (KvFormat::Float(DType::F32), ArenaFormatTag::F32),
            (KvFormat::Float(DType::F16), ArenaFormatTag::F16),
            (KvFormat::Float(DType::BF16), ArenaFormatTag::BF16),
            (KvFormat::Quantized(QuantFormat::R16), ArenaFormatTag::R16),
            (KvFormat::Quantized(QuantFormat::Q8_0), ArenaFormatTag::Q8_0),
            (
                KvFormat::Quantized(QuantFormat::Q4_KS),
                ArenaFormatTag::Q4_KS,
            ),
            (KvFormat::Quantized(QuantFormat::Q2_0), ArenaFormatTag::Q2_0),
            (KvFormat::Quantized(QuantFormat::Q0), ArenaFormatTag::Q0),
        ] {
            assert_eq!(
                fmt.to_tag(),
                tag.as_u8(),
                "persisted tag for {fmt:?} must equal the chunk's recorded byte"
            );
            assert_eq!(
                KvFormat::from_tag(tag.as_u8()),
                Some(fmt),
                "cold load must decode {tag:?} back to {fmt:?}"
            );
        }
    }

    /// Raw-byte goldens for the discriminants the substrate has already
    /// written to disk. These bytes are on-disk data: changing one silently
    /// re-interprets every persisted chunk.
    #[test]
    fn persisted_tag_discriminants_are_frozen() {
        assert_eq!(ArenaFormatTag::F32.as_u8(), 0);
        assert_eq!(ArenaFormatTag::F16.as_u8(), 1);
        assert_eq!(ArenaFormatTag::BF16.as_u8(), 2);
        assert_eq!(ArenaFormatTag::R16.as_u8(), 3);
        assert_eq!(ArenaFormatTag::Q8_0.as_u8(), 7);
        assert_eq!(ArenaFormatTag::Q8_KS.as_u8(), 10);
        assert_eq!(ArenaFormatTag::Q4_0.as_u8(), 15);
        assert_eq!(ArenaFormatTag::Q4_KS.as_u8(), 18);
        assert_eq!(ArenaFormatTag::Q2_0.as_u8(), 22);
        assert_eq!(ArenaFormatTag::Q0.as_u8(), 33);
        assert_eq!(ArenaFormatTag::Invalid.as_u8(), 255);
    }

    /// `from_u8` is the exact inverse of `as_u8` across the whole tag space,
    /// and unknown bytes degrade to `Invalid` rather than aliasing a format.
    #[test]
    fn tag_byte_round_trip_is_total() {
        for byte in 0u8..=255 {
            let tag = ArenaFormatTag::from_u8(byte);
            if tag == ArenaFormatTag::Invalid && byte != 255 {
                assert!(byte > 35, "byte {byte} should decode to a real tag");
            } else {
                assert_eq!(tag.as_u8(), byte, "round trip broken at byte {byte}");
            }
        }
    }
}

/// Global allocation state for all chunks.
///
/// This is the single source of truth for block/sequence state.
/// Per-block valid counts (`block_usage`) and RoPE positions are not stored
/// here; they are computed on-the-fly by the session-level code that builds
/// `DecodeMetadata` directly from `SequenceState::chunks`.
///
/// GID allocation is delegated to `ChunkGidPool`, which manages a bump counter
/// and free list with automatic return-on-drop semantics for `ChunkGid` values.
#[derive(Debug)]
pub(crate) struct BlockTableState {
    /// Layer index for this block table (for debugging / diagnostics).
    pub(crate) layer_idx: usize,
    /// Current width of the block table.
    pub(crate) max_blocks: usize,
    /// Per-slot state: `None` = free, `Some` = allocated.
    /// This is the single source of truth for slot allocation.
    /// The GPU block table is derived on demand from
    /// `sequences[b].chunk_at(blk).map(|cw| cw.gids[0].raw())`.
    pub(crate) sequences: Vec<Option<SequenceState>>,
}

impl BlockTableState {
    pub(crate) fn new(layer_idx: usize, max_blocks: usize, batch: usize) -> Self {
        Self {
            layer_idx,
            max_blocks,
            sequences: vec![None; batch],
        }
    }
}

#[cfg(test)]
mod front_evict_tests {
    use super::front_evict_count;
    use crate::CHUNK_SIZE;

    // Full 32-token chunks (the sliding-window ring's sealed chunks) plus a
    // partial writer tail — the exact shape the DeepSeek FP8 window holds.
    fn full_chunks(n: usize) -> Vec<u32> {
        vec![CHUNK_SIZE as u32; n]
    }

    #[test]
    fn window_zero_never_evicts() {
        // window_size 0 means "no ring bound" — must never drop a chunk.
        let usages = full_chunks(10);
        assert_eq!(front_evict_count(&usages, 10, 0, 0, 10_000), 0);
    }

    #[test]
    fn nothing_evicted_before_window_fills() {
        // 4 full chunks = 128 tokens, window 128, query at pos 127 (the 128th
        // token). Lowest in-window pos = 127 - 128 + 1 = 0, so chunk 0 (pos
        // [0,32)) is still partly in window → evict nothing.
        let usages = full_chunks(4);
        // writer_start = 3: chunks 0..3 sealed, chunk 3 is the writer tail.
        assert_eq!(front_evict_count(&usages, 3, 0, 128, 127), 0);
    }

    #[test]
    fn evicts_exactly_when_front_chunk_fully_exits() {
        // window 128; chunk 0 spans abs [0,32). Its last token (pos 31) leaves
        // the window when the lowest in-window pos exceeds 31, i.e.
        // abs_pos - 128 + 1 > 31 → abs_pos > 158 → abs_pos >= 159.
        let usages = full_chunks(8);
        let writer = 7;
        // At 158: lowest in-window = 158-128+1 = 31 ≤ 31 → chunk 0 still touches.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 158), 0);
        // At 159: lowest in-window = 32 > 31 → chunk 0 fully out, evict exactly 1.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 159), 1);
        // At 191: lowest in-window = 64; chunks 0 (end 32) and 1 (end 64) both
        // ≤ 64 → evict 2. Chunk 2 (end 96) > 64 stays.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 191), 2);
    }

    #[test]
    fn base_pos_shifts_the_absolute_frame() {
        // After earlier eviction (base_pos = 64), chunk 0 now spans abs
        // [64,96). Query at 223, window 128: lowest in-window = 96. Chunk 0
        // (end 96) ≤ 96 → evictable; chunk 1 (abs [96,128), end 128) > 96 stays.
        let usages = full_chunks(6);
        assert_eq!(front_evict_count(&usages, 5, 64, 128, 223), 1);
    }

    #[test]
    fn never_evicts_the_writer_or_beyond() {
        // Even with a huge query position, the writer chunk (and anything at or
        // past writer_start) is never dropped — the ring must keep a live tail.
        let usages = full_chunks(4);
        // writer_start = 1 → only chunk 0 is ever eligible.
        assert_eq!(front_evict_count(&usages, 1, 0, 32, 1_000_000), 1);
        // writer_start = 0 → nothing is eligible.
        assert_eq!(front_evict_count(&usages, 0, 0, 32, 1_000_000), 0);
    }

    #[test]
    fn resident_token_count_stays_bounded_past_window() {
        // Faithful ring simulation: decode 2000 tokens one at a time, sealing a
        // fresh 32-token writer chunk at each boundary and evicting fronts after
        // each step. The resident token span must stay bounded ABOVE by
        // `window + one partial chunk` (the ring is O(window), not O(N)), and
        // never drop BELOW the in-window token count (no attended key is
        // evicted). This is the property `window_bytes_flat_beyond_window_size`
        // asserts of the accounting model — here proven of the real eviction.
        let window = 128usize;
        let mut resident: Vec<u32> = Vec::new(); // usages of resident chunks
        let mut base = 0u32; // tokens evicted off the front
        for tok in 0..2000usize {
            if tok % CHUNK_SIZE == 0 {
                resident.push(0); // new writer chunk at each 32-token boundary
            }
            *resident.last_mut().unwrap() += 1; // the just-decoded token
            let live_before: usize = resident.iter().map(|&u| u as usize).sum();
            let pos = base as usize + live_before - 1; // absolute pos of this token == tok
            debug_assert_eq!(pos, tok);
            let writer = resident.len() - 1; // only sealed fronts are evictable
            let drained = front_evict_count(&resident, writer, base, window, pos);
            base += resident[..drained].iter().sum::<u32>();
            resident.drain(..drained);
            let live: usize = resident.iter().map(|&u| u as usize).sum();
            assert!(
                live <= window + CHUNK_SIZE,
                "ring unbounded: resident {live} > window+chunk at tok {tok}",
            );
            assert!(
                live >= window.min(tok + 1),
                "over-evicted an in-window key: resident {live} < {} at tok {tok}",
                window.min(tok + 1),
            );
        }
    }
}
