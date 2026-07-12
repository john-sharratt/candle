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

fn arena_bytes_per_chunk(format: crate::kv_cache::KvFormat) -> usize {
    let elems_per_chunk = CHUNK_SIZE * CHUNK_SIZE;
    match format {
        crate::kv_cache::KvFormat::Float(dtype) => elems_per_chunk * dtype.size_in_bytes(),
        crate::kv_cache::KvFormat::Quantized(qf) => {
            let ggml = qf.to_ggml_dtype();
            debug_assert_eq!(elems_per_chunk % ggml.block_size(), 0);
            (elems_per_chunk / ggml.block_size()) * ggml.type_size()
        }
    }
}

/// Compute the number of physical chunks that fit in a target-size arena for
/// the provided format.
///
/// The current palette4 CUDA layout stores one palette sub-band per chunk, and
/// that sub-band is `CHUNK_SIZE × CHUNK_SIZE` elements in the standard 128-dim
/// head layout. Integer division is used so the returned chunk count always
/// lands at the target size or just under it.
pub fn arena_chunks_for_format(format: crate::kv_cache::KvFormat) -> usize {
    (TARGET_ARENA_BYTES / arena_bytes_per_chunk(format)).max(1)
}

/// Global raw-GID stride used to map a raw chunk id to `(arena_idx, chunk_idx)`.
///
/// Physical arenas can now have format-specific capacities, but the raw GID
/// namespace still uses a single stride large enough to cover the densest arena
/// layout. This preserves stable routing while allowing per-format 16 MiB slabs.
pub fn arena_gid_stride() -> usize {
    use strum::IntoEnumIterator;

    let mut stride = arena_chunks_for_format(crate::kv_cache::KvFormat::Float(candle::DType::F16))
        .max(arena_chunks_for_format(crate::kv_cache::KvFormat::Float(
            candle::DType::BF16,
        )))
        .max(arena_chunks_for_format(crate::kv_cache::KvFormat::Float(
            candle::DType::F32,
        )));

    for qf in crate::kv_cache::QuantFormat::iter() {
        stride = stride.max(arena_chunks_for_format(
            crate::kv_cache::KvFormat::Quantized(qf),
        ));
    }

    stride
}

use super::gid_pool::ChunkGid;
use super::gpu_chunks::GpuChunks;
use super::head_gids::HeadGids;
use super::meta_pool::MetaGid;
use crate::kv_cache::arena_table::ResolvedArenaInfo;
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
    /// Co-resident KV-head metadata record handle, propagated from the
    /// `SealedChunk` this window was injected from (`Some`) or `None` for a
    /// freshly-allocated float writer chunk. Cloned with the window so every
    /// slot referencing the chunk shares one record. See [`super::meta_pool`].
    pub(crate) meta: Option<MetaGid>,
}

impl ChunkWindow {}

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
}

impl SequenceState {
    #[cfg(feature = "cuda")]
    pub(super) fn new(stream: Option<Arc<CudaStream>>) -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(stream),
            writer_start_idx: 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(),
            writer_start_idx: 0,
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
    pub(super) fn set_block_gids(
        &mut self,
        blk: usize,
        gids: HeadGids,
        k_pal: Arc<Vec<u8>>,
        v_pal: Arc<Vec<u8>>,
        k_scale: Arc<Vec<f32>>,
        v_scale: Arc<Vec<f32>>,
    ) {
        if blk < self.chunks.len() {
            let cw = &mut self.chunks[blk];
            cw.gids = gids;
            cw.k_pal = k_pal;
            cw.v_pal = v_pal;
            cw.k_scale = k_scale;
            cw.v_scale = v_scale;
            // Any GID mutation (defrag remap, cold-load reinjection) invalidates
            // the resident record's per-palette pointers. Drop it so the host
            // serializer falls back to per-forward scratch heads rather than
            // emitting a stale `kvheads_ptr`.
            cw.meta = None;
        }
    }

    /// Compute the RoPE base position for block `blk`.
    ///
    /// This is the cumulative token count of all preceding blocks.
    #[inline]
    pub(super) fn rope_pos(&self, blk: usize) -> i32 {
        let count = blk.min(self.chunks.len());
        self.chunks.iter().take(count).map(|c| c.usage as i32).sum()
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
        let rope_base: u32 = self.chunks[..blk].iter().map(|c| c.usage).sum();
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
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        // Prefix-sum cumulative usage once so each block's rope_base
        // is an O(1) lookup. The previous `chunks[..blk].iter().sum()`
        // was O(blk) per block — quadratic over a layer's blocks.
        let mut rope_bases: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut acc: u32 = 0;
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
    fn decode_write_chunk_idx(&self) -> usize {
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
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        gpu_chunks
            .as_mut()
            .rebuild_decode(chunks, n_kv_head, head_dim, arena_info, write_len, wi)?;

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
