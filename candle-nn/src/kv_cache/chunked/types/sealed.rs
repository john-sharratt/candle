//! The immutable record of a sequence's committed history, which can be shared
//! across forks and injected at any position.
//!
//! A sealed chunk names the bands of one committed window, with the palettes,
//! scales and per-band format tags needed to read them. The format travels with
//! the chunk because, under size classes, the arena no longer determines it.
//! Positions are not stored, so a sealed sequence can be injected anywhere with
//! no re-rotation and no copy. A trailing partial chunk may be shared safely,
//! since only one writer ever extends such a chunk. A chunk whose formats were
//! never recorded refuses to persist, because an empty format list would load
//! back as zero bands.

use super::*;

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
    /// The block's provenance `sign(Q)` bits, captured from its R16 bands by the
    /// gap quantize before its convert replaced them ([`QSign`]). `None` while
    /// the bands are R16 — Q is read from them — or when nothing captured it.
    pub q_sign: Option<QSign>,
}

/// A block's packed provenance `sign(Q)` bits, one layer's worth: `n_kv_head ×
/// N_PALETTE` warps of `CHUNK_SIZE` words, `[(head * N_PALETTE + palette) *
/// CHUNK_SIZE + slot]`, each word bit `d` set iff Q dim `d` of that sub-band is
/// non-negative — the sign-pack kernel's output for the block, as the
/// provenance gather would have computed it from the R16 bands.
///
/// **Q lives inside the R16 K band**, so a block converted to a quantized
/// format has no Q to read. The gap captures the bits first and the chunk
/// carries them — through snapshots, re-injection and compaction — to the seal
/// and the reprojection probe that read them.
pub type QSign = Arc<Vec<u64>>;

/// Where one block's provenance `sign(Q)` comes from, for one layer.
#[derive(Clone, Debug)]
pub enum ProvSource {
    /// The block's K is R16 and GPU-backed: the Q band base addresses,
    /// `[head][palette]`, for the sign-pack kernel to read.
    Q(Vec<i64>),
    /// The block was converted after the gap captured its signs.
    Signed(QSign),
}

impl SealedChunk {
    /// Whether every band is still native — the formats the paged stores
    /// write, R16 or float — so the chunk is one a gap quantizes. A chunk
    /// whose formats were never recorded is not.
    pub fn is_native(&self) -> bool {
        !self.k_fmt.is_empty()
            && self
                .k_fmt
                .iter()
                .all(|&t| ArenaFormatTag::from_u8(t).takes_active_k_writes())
            && self
                .v_fmt
                .iter()
                .all(|&t| ArenaFormatTag::from_u8(t).takes_active_v_writes())
    }

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
            q_sign: None,
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
    pub location: super::super::ArenaLocation,
}

impl SealedSequence {
    /// Drop the chunks at the tail that hold no token.
    ///
    /// A snapshot of a live slot takes every block, the empty writer and any
    /// pad pushed after it included. Sealed, an empty block is a chunk in a
    /// sealed format with room left in it; restored above a writer boundary it
    /// is the chunk the next write resolves to, and the paged stores write
    /// nothing into a sealed format. No position moves when they go — an
    /// empty chunk adds nothing to the cumulative count.
    pub fn drop_empty_tail(&mut self) {
        while self.chunks.last().is_some_and(|c| c.token_count == 0) {
            self.chunks.pop();
        }
    }
}
