//! The per-sequence block table of the chunked KV cache: the chunks a sequence
//! owns, the sealed history it shares, and the cached decode buffer that
//! describes both to the kernel.
//!
//! A sequence's chunks form a list in token order, with shared prefix history
//! below the writer boundary and writer-owned chunks above it. Every change to
//! the list invalidates or extends the cached decode buffer, and a count check
//! refuses any header that promises more entries than the buffer holds. Sliding
//! window eviction, compaction and the stateless rebuild all work on this list,
//! and the writer boundary tells each of them what may be moved.
//!
//! - `chunk_meta`: the per-block record the paged attention kernels read.
//! - `chunk_list`: structural edits to the chunk list, each keeping the decode buffer true.
//! - `decode_layout`: the layer summary that decides whether one position map serves all layers.
//! - `gpu_sync`: keeps the cached GPU slot-state buffer in step with the host list.
//! - `live_chunk`: the per-block chunk window and its borrowed view.
//! - `sealed`: the immutable, shareable record of committed history.
//! - `sequence_state`: sliding-window eviction and absolute-position bookkeeping.
//! - `writer_region`: the writer boundary and the compaction rewrite.
//! - `writer_tail`: the snapshot of writer-owned chunks kept across a rebuild.
//!
//! - `tests_bands`, `tests_front_evict`, `tests_sealed_tail`, `tests_slot_count`,
//!   `tests_writer_region`: tests for the band tags, eviction, trimming, the
//!   slot-count check and the writer boundary.

// Constants

/// Tokens per physical arena chunk
pub const CHUNK_SIZE: usize = 32;

/// Target size for one arena allocation: 16 MiB.
pub const TARGET_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Re-exported so arena sizing and gid decode read the same constant. The
/// stride lives with the size-class ladder because that is what bounds it.
pub use super::size_class::GID_STRIDE;

#[cfg(feature = "cuda")]
use super::compact_mint::RecordInputs;
use super::gap::{cursor_after_front_eviction, cursor_after_splice, cursor_after_truncate};
use super::gid_pool::ChunkGid;
use super::gpu_chunks::{ChunkPin, GpuChunks};
use super::head_gids::{band_tags, HeadGids};
use super::meta_pool::MetaGid;
use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo};
#[cfg(feature = "cuda")]
use candle::CudaDevice;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SequenceState {
    /// All blocks in token order.  Shared (prefix) and owned chunks
    /// coexist in the same vec.  Partial tails are always copied at fork
    /// time, so `chunks.last()` is always uniquely owned and writable.
    chunks: Vec<ChunkWindow>,
    /// Cached host + GPU-side serialised slot-state for this sequence.
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
    /// The first chunk not yet handed to a gap quantize
    /// (`docs/kv_quantize_in_the_gap.md` §3.1): the gap's table takes complete
    /// native chunks from here, so it never walks the sequence's history. Kept
    /// pointing at the same chunk by every change to the chunk list below it,
    /// and pulled back by a truncate that cuts it off.
    gap_cursor: usize,
    /// Absolute token position of the FIRST resident chunk's first token —
    /// i.e. the count of tokens evicted off the front by the sliding-window
    /// ring ([`Self::evict_front_window`]). Every chunk's serialised
    /// `rope_base` and every [`Self::rope_pos`] is offset by this, so window
    /// keys keep their ABSOLUTE positions after the front slides out. Zero for
    /// every non-windowed slot (dialogue/section KV never evicts its front), so
    /// those paths are byte-identical to a `base_pos == 0` derivation.
    base_pos: u32,
    /// Set by a commit made outside the decode kernel: the chunk index from
    /// which the cached decode slot buffer's lengths may be behind the host —
    /// the first chunk the earliest such commit since the buffer was last
    /// current filled ([`Self::fill_from`]), or, for a commit that did not say,
    /// the writer boundary as it stood then. The next
    /// [`Self::sync_decode_gpu_chunks`] — the only way a reader reaches the
    /// buffer — re-serialises from here to the writer before handing it out.
    /// It is fixed AT the commit, not read back at the sync, because a
    /// boundary that moves in between (`seal_writer_boundary`, a truncate's
    /// clamp) would otherwise shrink the region past chunks the commit filled.
    /// Left set over a buffer that has since been cleared it is harmless: the
    /// resync finds no buffer, and the rebuild serialises every chunk from the
    /// host state.
    decode_stale_from: Option<usize>,
    /// The first chunk a host-side length commit (`set_len`) changed since the
    /// last sync — every chunk before it was full and took nothing. The stale
    /// mark that follows the commit starts there instead of at the writer
    /// boundary, so a prefill chunk's sync re-serialises the chunks it filled
    /// rather than every chunk the sequence has written, which at depth is the
    /// whole history.
    fill_from: Option<usize>,
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

mod chunk_meta;
mod decode_layout;
mod live_chunk;
mod sealed;
mod sequence_state;
mod writer_tail;
#[cfg(test)]
use sequence_state::*;
mod chunk_list;
mod gpu_sync;
#[cfg(test)]
mod tests_bands;
#[cfg(test)]
mod tests_front_evict;
#[cfg(test)]
mod tests_sealed_tail;
#[cfg(test)]
mod tests_slot_count;
#[cfg(test)]
mod tests_writer_region;
mod writer_region;

pub use chunk_list::*;
pub use chunk_meta::*;
pub(super) use decode_layout::*;
pub(crate) use gpu_sync::*;
pub use live_chunk::*;
pub use sealed::*;
pub use writer_tail::*;
