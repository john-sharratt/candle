//! The injected pages a cache holds, and the state a rewind or a compaction
//! pass needs to undo or relocate them.
//!
//! A page is one injected prefix's index rows, placed at an absolute base in its
//! cache. It is resident once, so forks and other slots share its rows through
//! one `Arc`. Its rows are rotated to its base as the scorer reads them, which
//! means changing where a page sits changes only a number. A failed wave restores
//! the open block from a copy taken before it, and a compaction pass moves
//! buffers without changing what they hold.

use super::*;

/// One thing [`IndexCache::splice_pages`] puts in place.
#[derive(Debug, Clone)]
pub enum SplicePiece {
    /// An injected piece's page on this layer.
    Page(Arc<ResidentPage>),
    /// Tokens injected with no rows on this layer — a span nothing indexed.
    Gap(usize),
}

/// One injected page and where it sits.
#[derive(Debug, Clone)]
pub(super) struct PlacedPage {
    pub(super) page: Arc<ResidentPage>,
    /// Absolute position this page's first row occupies **in this cache**.
    ///
    /// The authority: the scorer rotates the page's row `j` at
    /// `base + j·ratio` as it reads it.
    pub(super) base: usize,
    /// Tokens the page covers, resolved at push where `ratio` is in hand — so
    /// nothing downstream needs a `ratio` to say how wide a page is.
    pub(super) tokens: usize,
}

/// What a wave must put back if it fails.
#[derive(Debug)]
pub struct IndexSnapshot {
    pub(super) n_blocks: usize,
    /// A COPY of the open block's rows, not a handle to them: the buffer they
    /// live in is written in place by the carry kernel, so a shared clone would
    /// be overwritten by the very wave this snapshot exists to undo. In one of
    /// the cache's own rewind buffers, which goes back when this drops; `None`
    /// when there were no open rows to keep.
    pub(super) raw: Option<SnapshotBuffer>,
    pub(super) n_open: usize,
}

/// The open block and its rewind copies, claimed together: `(raw, snapshot
/// buffers)`. `MAX_RATIO` rows each, so the open block never resizes — it holds
/// fewer than `ratio` rows and `ratio` is bounded by `MAX_RATIO`.
pub(super) fn open_block_buffers(
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, SnapshotBuffers)> {
    let mut bufs = alloc_buffers(device, MAX_RATIO, head_dim, 1 + SNAPSHOT_BUFFERS)?;
    let raw = bufs.pop().expect("claimed one more than the snapshots");
    Ok((raw, SnapshotBuffers::new(bufs)))
}

/// Compact the QSA-index arenas `caches` live in: for each of the tenant's two
/// strides — a key page, and an open block or its rewind copy — plan the
/// two-cursor pass and move every buffer whose slot is a source onto its
/// destination. The same walk and the same provisioning as the recurrent-state
/// pass (`compact_stores`); `max_moves` (zero for none) bounds each stride.
///
/// **Between forwards.** Every cache that could hold a source must be in `caches`:
/// a source left behind keeps its slot and its claimed destination goes back, which
/// is safe and shows as `moved` short of `planned`.
pub fn compact_index_caches(
    caches: &mut [&mut IndexCache],
    retired: &[Arc<ResidentPage>],
    head_dim: usize,
    device: &Device,
    max_moves: usize,
) -> Result<RecurrentCompaction> {
    if !matches!(device, Device::Cuda(_)) {
        return Ok(RecurrentCompaction::default());
    }
    let f32_bytes = DType::F32.size_in_bytes();
    let regions_before = arena_regions(device, SlotTenant::QsaIndex);
    let mut planned = 0usize;
    let mut moved = 0usize;
    for bytes in [key_page_bytes(head_dim), MAX_RATIO * head_dim * f32_bytes] {
        let moves = plan_slot_moves(device, SlotTenant::QsaIndex, bytes, max_moves)?;
        planned += moves.len();
        let mut by_source: HashMap<u64, ArenaSlot> =
            moves.into_iter().map(|m| (m.src, m.dst)).collect();
        for cache in caches.iter_mut() {
            moved += cache.relocate(&mut by_source)?;
        }
        // Pages a reset dropped are held for the rebuild by nothing a cache
        // reaches; left out, their slots kept every arena they sat in from emptying.
        for page in retired {
            moved += page.relocate(&mut by_source)?;
        }
        // Destinations whose source nothing here held go back to their arenas.
        drop(by_source);
    }
    Ok(RecurrentCompaction {
        planned,
        moved,
        regions_before,
        regions_after: arena_regions(device, SlotTenant::QsaIndex),
    })
}
