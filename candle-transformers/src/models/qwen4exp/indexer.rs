//! The QSA indexer, on the device.
//!
//! One of these per (sequence, full-attention layer). It carries the index
//! cache — the compressed keys the selection scores against — and turns a
//! wave's rows into the packed selection the paged attention kernels read.
//!
//! # What is cached, and what is not
//!
//! §3.1 of the design doc: one key per `ratio` tokens. The reference caches
//! the *raw* projected keys and pools, norms and ropes them at read time. The
//! pool and the norm are functions of the block alone — the mean of its
//! `ratio` raw keys and the indexer's `k_norm` — so a completed block is pooled
//! and normed once and stored, and only the `ratio − 1` raw rows of the block
//! still filling are held as raw rows.
//!
//! **The rotation is not stored.** A stored key is un-rotated, exactly as K is
//! stored without RoPE, and the scorer rotates each key at its block's first
//! position as it loads it (`qsa_score_paged.cu`) — the reference's own order.
//! So a stored row carries no position and no RoPE frequencies: a page moves
//! between projections, and a slot changes RoPE schedule, without a byte of the
//! index changing (`docs/progressive_yarn.md` §7).
//!
//! # Wave atomicity
//!
//! The cache is a carried state in the same class as the GDN recurrence and
//! the PLE conv tail (§6.3): [`IndexCache::snapshot`] before a wave,
//! [`IndexCache::restore`] if it fails. Restoring is cheap because appends
//! only ever advance `n_blocks` — the rows above it are dead, not deleted —
//! so the whole of a failed wave's work is undone by moving the count back
//! and putting the open block's raw rows back.
//!
//! # Capacity
//!
//! The live tail's keys are pages of the span's QSA-index arenas
//! ([`super::index_keys`]). [`IndexCache::ensure_capacity`] is called at wave
//! admission, never inside the layer loop: a page claim that needs a new arena
//! takes the arena window, which a forward holds shut (hot-path invariant 7).

use candle::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::capture_rows::CaptureRows;
use super::config::IndexerConfig;
/// Block keys per live-tail key page — see `index_keys`.
pub use super::index_keys::PAGE_BLOCKS;
use super::index_keys::{
    alloc_buffers, row_addr, write_host, KeyPages, SnapshotBuffer, SnapshotBuffers,
    SNAPSHOT_BUFFERS,
};
use super::qsa::IndexerWeights;
use super::qsa_select::{
    max_entries_for, max_gathered_for, max_keep, selected_width, Strata, MAX_RATIO,
};
use super::resident_page::ResidentPage;
use super::rows_matmul::rows_matmul_t;
use super::spec::SpecCapture;
use crate::models::delta_net::mix::SeqSpan;
use crate::models::delta_net::RecurrentCompaction;
use crate::models::operand_guard::{expect_dense, expect_dense_dtype};
use crate::models::qsa_selection::QsaSelection;
use crate::models::selection_strata::Recent;
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};
use candle::wave_provenance::WaveTicket;
#[cfg(feature = "cuda")]
use candle_kernels::simple::qsa_topk::{qsa_topk_split_parts, SPLIT_KEYS};
use candle_kernels::simple::qsa_topk::{MAX_ENTRIES, MAX_KEEP};
use candle_nn::kv_cache::{arena_regions, plan_slot_moves, relocate_tensor, ArenaSlot, SlotTenant};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ptr::null_mut;

use crate::models::rope_schedule::FactoredRope;

/// Rows of scores computed in one tile.
///
/// The score matrix is `[rows × heads, blocks]`, which at depth is the largest
/// thing in the selection path — 4 heads × 4 bytes × one block per 4 tokens,
/// per query. Tiling the query rows bounds it; the tile is chosen so a tile's
/// scores stay under this many bytes.
const SCORE_TILE_BYTES: usize = 128 << 20;

/// The live-tail cells — query rows × live-tail blocks — from which the tail is
/// scored by cuBLAS rather than by the paged scorer.
///
/// Below it a span's scores come from one paged-scorer launch over its pages
/// and its live tail, every key rotated as it loads. From it up — a prefill
/// chunk over a long unplaced turn — the live tail goes to cuBLAS instead,
/// rotated once into a scratch buffer that lives for the launch.
///
/// **A cell count, not a row count**, because that is what the paged route's
/// time follows: `tests/qsa_score_rot_harness.rs`'s sweep puts it at ~0.78 ms
/// per 2²⁵ cells whether they are 4096 rows over 8K tokens or 128 over 256K.
/// The GEMM has a fixed cost the paged route does not — the scratch and its
/// rotation, ~0.6 ms once the tail is wide. At 2²⁵ cells it is ahead by about
/// 0.1 ms in three of the four splits the sweep measured and behind by 0.08 in
/// the fourth; at twice that it wins by 1.4–1.8×. Below it the GEMM is never
/// more than 60 µs ahead.
pub const GEMM_TAIL_MIN_CELLS: usize = 1 << 25;

/// How a span's live tail is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailRoute {
    /// As one more page of the paged scorer, rotated on load.
    Paged,
    /// Rotated once into a scratch buffer, then one cuBLAS GEMM.
    Gemm,
}

impl TailRoute {
    /// The route for `rows` query rows over `live_cols` live-tail blocks —
    /// [`GEMM_TAIL_MIN_CELLS`].
    pub fn for_span(rows: usize, live_cols: usize) -> Self {
        if rows.saturating_mul(live_cols) < GEMM_TAIL_MIN_CELLS {
            Self::Paged
        } else {
            Self::Gemm
        }
    }
}

/// One sequence's index cache for one full-attention layer.
#[derive(Debug)]
pub struct IndexCache {
    /// Prepared block keys (pooled, normed, un-rotated), in pages of the span's
    /// QSA-index arenas. Only `[0, n_blocks)` is live.
    keys: KeyPages,
    n_blocks: usize,
    /// `[MAX_RATIO, head_dim]` F32 — the raw projected keys of the block still
    /// filling, contiguous in `[0, n_open)`. Fewer than `ratio` rows; empty
    /// when the cache ends on a block boundary.
    ///
    /// One buffer rather than the list of arrival pieces it used to be. The
    /// list existed to avoid a `cat` per step, but the batched append reads
    /// these rows from a KERNEL, which needs one base pointer and a row stride
    /// — and the carry that fills it is itself batched across the wave, so the
    /// per-step copy the list was avoiding does not come back.
    raw: Tensor,
    /// Rows of [`Self::raw`] that are live.
    n_open: usize,
    /// Rewind copies of [`Self::raw`], claimed with it — see [`Self::snapshot`].
    snaps: SnapshotBuffers,
    /// Index rows for positions this sequence holds but never forwarded —
    /// prefixes whose K/V arrived by injection.
    ///
    /// **The case that makes this necessary is section ingest.** A section is
    /// prefilled against an Arc-injected prefix: the prefix's K/V already
    /// exists, so copying it is free and the forward attends to real preceding
    /// context. The index cannot be copied that way — its keys come from hidden
    /// states, which injection never computes — so without these pages a query
    /// at position 22,020 asks for 5,505 blocks against a cache holding one.
    ///
    /// **They are pages and not more rows of `keys` because they are ragged.**
    /// A section advances the sequence by its real token count, not by a whole
    /// number of blocks, so the next section's pooling starts mid-block and its
    /// rows do not line up with `(pos + 1) / ratio`. Concatenating them into the
    /// uniform buffer would silently address the wrong block for every position
    /// after the first boundary. Each page therefore keeps its own width and
    /// [`Self::candidates_at`] walks them, exactly as
    /// [`PagedIndex`](super::paged_index::PagedIndex) does for a window
    /// reconstructed from turn records.
    ///
    /// Empty for a sequence that forwarded everything it holds, which is every
    /// sequence the gates run — there the walk reduces to the uniform formula
    /// and the scorer sees a single page.
    ///
    /// **Shared, not owned.** Each page is resident on the span once and held by
    /// `Arc`: a fork, a view carve and every other slot that injected the same
    /// piece read the same rows, exactly as they borrow the same K/V chunks.
    pages: Vec<PlacedPage>,
    /// Exclusive prefix sum of [`Self::pages`] row counts.
    ///
    /// Rows stay ordinal, and only rows: the scorer's candidate axis is a dense
    /// concatenation of every page's rows, so row `k` of page `p` is candidate
    /// `page_rows[p] + k` however the pages are positioned. Positions are the
    /// other thing entirely — see [`PlacedPage::base`].
    page_rows: Vec<usize>,
    /// Absolute position the live tail opens at.
    ///
    /// **Recorded, not accumulated.** It used to be the running sum of the page
    /// widths, which made every position downstream of an unindexed span wrong
    /// by that span's width; it is now whatever the last placement said, so a
    /// span nothing indexed is a hole and nothing else.
    tail_base: usize,
    /// Position the conversation's system prompt ends at — `0` when the slot
    /// holds none.
    ///
    /// Set by the projection that placed the prompt, and the reason it lives
    /// here rather than beside the sequence: a stratified selection ranks the
    /// prompt's blocks in every window, so it has to describe the same layout
    /// the pages do, and travel wherever they travel — a view carve, a move, a
    /// rebuild that keeps the prefix.
    prompt_end: usize,
}

/// One injected page and where it sits.
#[derive(Debug, Clone)]
struct PlacedPage {
    page: Arc<ResidentPage>,
    /// Absolute position this page's first row occupies **in this cache**.
    ///
    /// The authority: the scorer rotates the page's row `j` at
    /// `base + j·ratio` as it reads it.
    base: usize,
    /// Tokens the page covers, resolved at push where `ratio` is in hand — so
    /// nothing downstream needs a `ratio` to say how wide a page is.
    tokens: usize,
}

/// What a wave must put back if it fails.
#[derive(Debug)]
pub struct IndexSnapshot {
    n_blocks: usize,
    /// A COPY of the open block's rows, not a handle to them: the buffer they
    /// live in is written in place by the carry kernel, so a shared clone would
    /// be overwritten by the very wave this snapshot exists to undo. In one of
    /// the cache's own rewind buffers, which goes back when this drops; `None`
    /// when there were no open rows to keep.
    raw: Option<SnapshotBuffer>,
    n_open: usize,
}

/// The open block and its rewind copies, claimed together: `(raw, snapshot
/// buffers)`. `MAX_RATIO` rows each, so the open block never resizes — it holds
/// fewer than `ratio` rows and `ratio` is bounded by `MAX_RATIO`.
fn open_block_buffers(head_dim: usize, device: &Device) -> Result<(Tensor, SnapshotBuffers)> {
    let mut bufs = alloc_buffers(device, MAX_RATIO, head_dim, 1 + SNAPSHOT_BUFFERS)?;
    let raw = bufs.pop().expect("claimed one more than the snapshots");
    Ok((raw, SnapshotBuffers::new(bufs)))
}

impl IndexCache {
    pub fn new(head_dim: usize, device: &Device) -> Result<Self> {
        let (raw, snaps) = open_block_buffers(head_dim, device)?;
        Ok(Self {
            keys: KeyPages::new(head_dim, device),
            n_blocks: 0,
            raw,
            n_open: 0,
            snaps,
            pages: Vec::new(),
            page_rows: vec![0],
            tail_base: 0,
            prompt_end: 0,
        })
    }

    /// Record where the conversation's system prompt ends.
    pub fn set_prompt_end(&mut self, pos: usize) {
        self.prompt_end = pos;
    }

    /// Where the conversation's system prompt ends — `0` when none was set.
    pub fn prompt_end(&self) -> usize {
        self.prompt_end
    }

    /// The blocks wholly inside the system prompt — the span a stratified
    /// selection ranks in every window. Through the page layout, so a prompt
    /// placed as several sealed sections counts its short boundary blocks.
    pub fn prompt_blocks(&self, ratio: usize) -> usize {
        match self.prompt_end {
            0 => 0,
            end => self.candidates_at(end - 1, ratio),
        }
    }

    /// The indexer head width this cache stores.
    pub fn head_dim(&self) -> usize {
        self.keys.head_dim()
    }

    /// Blocks the key pages can currently address.
    pub fn capacity_blocks(&self) -> usize {
        self.keys.capacity_blocks()
    }

    /// Tokens this cache has consumed — `n_blocks · ratio + open`.
    /// Tokens this cache accounts for — injected pages plus the live tail.
    ///
    /// The pages count because the caller compares this against the sequence's
    /// own length, and the sequence holds their positions.
    pub fn len(&self, ratio: usize) -> usize {
        self.page_token_span() + self.n_blocks * ratio + self.n_open
    }

    pub fn is_empty(&self, ratio: usize) -> bool {
        self.len(ratio) == 0
    }

    /// A rewind point: the counters, and a copy of the open block's live rows in
    /// one of the cache's own rewind buffers — no allocation, which is what lets a
    /// forward take one after its span has opened.
    pub fn snapshot(&self) -> Result<IndexSnapshot> {
        let raw = if self.n_open > 0 {
            let buf = self.snaps.take()?;
            buf.tensor().slice_set(&self.open_rows()?, 0, 0)?;
            Some(buf)
        } else {
            None
        };
        Ok(IndexSnapshot {
            n_blocks: self.n_blocks,
            raw,
            n_open: self.n_open,
        })
    }

    /// The first position of `block`, through this cache's page layout — the
    /// inverse of [`Self::candidates_at`].
    ///
    /// Read off the page's own [`base`](PlacedPage::base), never accumulated
    /// from the widths before it, which is what makes an unindexed span harmless
    /// here: it contributes no rows and moves no page.
    pub fn block_start(&self, block: usize, ratio: usize) -> usize {
        let rows = self.page_row_span();
        if block >= rows {
            return self.tail_base + (block - rows) * ratio;
        }
        let p = match self.page_rows.binary_search(&block) {
            Ok(i) => return self.pages[i].base,
            Err(i) => i - 1,
        };
        self.pages[p].base + (block - self.page_rows[p]) * ratio
    }

    /// Cells of its own block the query at `pos` has, `1..=ratio`.
    ///
    /// What the selection kernel is handed instead of deriving `pos % ratio`:
    /// a block's width belongs to the page it is in, and at a page boundary the
    /// last block is short.
    pub fn tail_len(&self, pos: usize, ratio: usize) -> u32 {
        let block = self.candidates_at(pos, ratio);
        (pos + 1 - self.block_start(block, ratio)) as u32
    }

    /// This cache's page layout as `{tokens_before, blocks_before}` pairs,
    /// ascending, with a final entry for the live tail.
    ///
    /// The trailing entry is what makes the walk uniform: a position past every
    /// page resolves against `{page_token_span, page_row_span}` and lands in the
    /// live tail's own arithmetic, so the kernels need no special case for it.
    pub fn page_prefixes(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity((self.pages.len() + 1) * 2);
        for (i, p) in self.pages.iter().enumerate() {
            out.push(p.base as u32);
            out.push(self.page_rows[i] as u32);
        }
        out.push(self.tail_base as u32);
        out.push(self.page_row_span() as u32);
        out
    }

    /// Whether this cache holds any injected prefix at all.
    pub fn has_pages(&self) -> bool {
        !self.pages.is_empty()
    }

    /// The most bytes [`Self::score_rows`] can carve on its ticket's arena for
    /// `t` query rows whose deepest sees `cand_max` candidate blocks, after a
    /// wave appended `appended` more blocks to this cache — each carve rounded
    /// up to the bump's alignment, so the sum is a bound and never short.
    ///
    /// Mirrors the two routes: the paged scorer's page, offset and count
    /// tables, and — when the live tail goes to cuBLAS — the rotated tail, its
    /// page table and every tile's scoring product (a bump never rewinds within
    /// a phase, so the tiles add up).
    pub fn score_bound(
        &self,
        t: usize,
        cand_max: usize,
        appended: usize,
        heads: usize,
        d: usize,
        align: usize,
    ) -> Result<usize> {
        use candle_kernels::simple::qsa_score_paged::PAGE_WORDS;

        let a = |b: usize| b.div_ceil(align) * align;
        let page_cols = self.page_row_span();
        let live_cols = cand_max.saturating_sub(page_cols);
        let tail_in_paged = live_cols > 0 && TailRoute::for_span(t, live_cols) == TailRoute::Paged;
        let paged_cols = if tail_in_paged { cand_max } else { page_cols };
        let mut bytes = 0usize;
        if paged_cols > 0 {
            let mut chunks = 0usize;
            for placed in &self.pages {
                chunks += placed.page.descriptors()?.len();
            }
            if tail_in_paged {
                chunks += (self.n_blocks + appended).div_ceil(PAGE_BLOCKS);
            }
            bytes += a(chunks * PAGE_WORDS * 8) + a((chunks + 1) * 4) + a(t * 4);
        }
        if !tail_in_paged && live_cols > 0 {
            bytes += a(live_cols.div_ceil(PAGE_BLOCKS) * 8) + a(live_cols * d * 4);
            let per_tile = (SCORE_TILE_BYTES / (heads * live_cols * 4)).clamp(1, t);
            let (full, rem) = (t / per_tile, t % per_tile);
            bytes += full * a(per_tile * heads * live_cols * 4);
            if rem > 0 {
                bytes += a(rem * heads * live_cols * 4);
            }
        }
        Ok(bytes)
    }

    /// Blocks the live tail has completed.
    pub fn live_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Undo everything a failed wave appended. The keys above `n_blocks` are
    /// dead rows, so nothing has to be erased.
    ///
    /// Borrows the snapshot and copies its open rows back, because a rewind may
    /// restore the same entering state more than once — a partial accept
    /// restores, re-appends the accepted rows, and can be asked again on the
    /// next block. Handing the buffer over would leave the second restore with
    /// rows the first one's re-append had already overwritten.
    pub fn restore(&mut self, snap: &IndexSnapshot) -> Result<()> {
        if let Some(buf) = &snap.raw {
            self.raw
                .slice_set(&buf.tensor().narrow(0, 0, snap.n_open)?, 0, 0)?;
        }
        self.n_blocks = snap.n_blocks;
        self.n_open = snap.n_open;
        Ok(())
    }

    /// Close this cache on a block boundary, pooling the carried rows into one
    /// SHORT block. Returns that block's width in tokens, or `None` when the
    /// cache already ended on a boundary.
    ///
    /// **This is what makes a turn's index a self-contained page.** Without it a
    /// turn leaves `T mod ratio` rows carried, belonging to a block the next
    /// turn completes — so a window reconstructed from a subset of turns has a
    /// leading block pooled over tokens that are not in the window.
    ///
    /// The flushed block is a summary of fewer than `ratio` tokens and is
    /// therefore NOT what a continuous run would have produced for that span.
    /// That is the deliberate trade: the scorer carries each page's width and
    /// derives the candidate prefix from it, so a short block is expressible;
    /// a block pooled from another turn's tokens is not correctable at all.
    #[cfg(feature = "cuda")]
    pub fn flush_open_block(&mut self, w: &IndexerWeights, rms_eps: f64) -> Result<Option<usize>> {
        use candle_kernels::simple::qsa_index_append::{run_qsa_index_flush, FLUSH_WORDS};

        if self.n_open == 0 {
            return Ok(None);
        }
        let cells = self.n_open;
        let d = self.keys.head_dim();
        self.keys.ensure(self.n_blocks + 1)?;
        let dst = row_addr(&self.keys.page_ptrs()?, self.n_blocks, d);
        let src = self.raw_ptr()?;
        let jobs: Vec<i64> = vec![dst as i64, src as i64, cells as i64];

        let device = self.keys.device().clone();
        let candle::Device::Cuda(cuda) = &device else {
            candle::bail!("qsa index flush runs on CUDA");
        };
        debug_assert_eq!(jobs.len(), FLUSH_WORDS);
        let k_norm = tensor_ptr(&w.k_norm)?;
        // The one job through the device's staging scratch: a flush runs at a
        // page close between forwards as often as inside one, and allocates
        // nothing in either. The stream is taken inside, where the launch is:
        // the staged upload runs eagerly, so the launch belongs on the stream
        // it names there.
        cuda.with_staged_upload(&jobs, |jobs_ptr| {
            let stream = cuda.cuda_stream();
            candle::set_kernel_breadcrumb("run_qsa_index_flush", file!(), line!());
            unsafe {
                run_qsa_index_flush(
                    jobs_ptr as *const i64,
                    k_norm as *const f32,
                    d as i32,
                    rms_eps as f32,
                    1,
                    stream.cu_stream() as *mut c_void,
                );
            }
            Ok(())
        })?;
        self.n_blocks += 1;
        self.n_open = 0;
        Ok(Some(cells))
    }

    /// Close the live tail into a page, so what follows it starts a new one.
    ///
    /// **A page is the only place a ragged width can live.** The live tail's
    /// arithmetic is a uniform division in three places — [`Self::candidates_at`],
    /// [`Self::block_start`], and [`Self::indexed_tokens`] all read
    /// `n_blocks · ratio` — so a short block left *inside* it puts every later
    /// position on the wrong block and makes the cache over-report its own token
    /// count by `ratio − cells`, for the rest of the sequence, internally
    /// consistent and wrong against the K/V. [`Self::flush_open_block`] alone
    /// therefore cannot be used mid-sequence; lifting the flushed rows out into a
    /// page, whose `last_cells` records the short width, is what makes the cut
    /// expressible.
    ///
    /// Composed rather than reimplemented: the flush pools the carried rows into
    /// one short block, and [`Self::push_page`] does the prefix-sum bookkeeping.
    /// The reset between them *satisfies* `push_page`'s guard ("injected prefixes
    /// must precede anything this sequence forwarded") rather than bypassing it —
    /// once the tail has been lifted out there is nothing forwarded left to
    /// precede.
    ///
    /// **Nothing is copied.** The key pages holding the tail's rows become the
    /// page's own chunks as they stand, row-major, and the tail opens again on the
    /// pages above them.
    ///
    /// A no-op on an empty tail, so a caller may close unconditionally at a
    /// boundary without asking whether one is needed.
    ///
    /// Returns the tokens the new page covers — `0` when the tail was already
    /// empty. **The caller needs this to know the cut fired**: a boundary that
    /// silently closes nothing is indistinguishable from one placed after the
    /// tokens it was meant to separate, and that distinction cost several live
    /// runs to make by inference.
    #[cfg(feature = "cuda")]
    pub fn close_tail_into_page(
        &mut self,
        w: &IndexerWeights,
        ratio: usize,
        rms_eps: f64,
    ) -> Result<usize> {
        // Closing a page is index maintenance, not a forward: no phase is open.
        let cells = self.flush_open_block(w, rms_eps)?;
        if self.n_blocks == 0 {
            return Ok(0);
        }
        let last = cells.unwrap_or(ratio);
        let tokens = (self.n_blocks - 1) * ratio + last;
        let chunks = self.keys.detach(self.n_blocks)?;
        let page = ResidentPage::from_key_pages(chunks, self.n_blocks, last, self.keys.head_dim())?;
        // The page opens where the tail did. Its rows are un-rotated, so it
        // needs nothing but its base to be read where it already sits.
        let base = self.tail_base;
        self.n_blocks = 0;
        self.n_open = 0;
        self.push_page(page, base, ratio)?;
        Ok(tokens)
    }

    /// A cache holding `rows` as its live prefix and `open` as its carried,
    /// un-pooled tail — the resume path, and the exact inverse of
    /// [`Self::live_rows_host`] + [`Self::open_rows_host`].
    ///
    /// **`open` is not optional and not decoration.** The cache's arithmetic is
    /// `n_blocks · ratio + n_open == tokens`, and every consumer depends on it:
    /// [`Self::plan`] derives the next append from `n_open`, and the scorer
    /// addresses block `k` as tokens `[k · ratio, (k+1) · ratio)`. A restore
    /// that dropped the open rows would put the cache `tokens % ratio` behind
    /// its own K/V and keep it there — internally consistent, wrong against the
    /// sequence, and silent.
    ///
    /// `rows` and `open` are host rows, `[n, head_dim]` and `[n_open, head_dim]`
    /// row-major, written straight into the cache's arena slots.
    pub fn from_rows(rows: &[f32], open: &[f32], head_dim: usize, device: &Device) -> Result<Self> {
        if head_dim == 0
            || !rows.len().is_multiple_of(head_dim)
            || !open.len().is_multiple_of(head_dim)
        {
            candle::bail!(
                "index cache: {} row values and {} open values against head_dim {head_dim}",
                rows.len(),
                open.len()
            );
        }
        let n = rows.len() / head_dim;
        let n_open = open.len() / head_dim;
        if n_open > MAX_RATIO {
            candle::bail!(
                "index cache: {n_open} open rows exceeds the {MAX_RATIO}-row block — a full \
                 block would have been pooled into a row instead of carried"
            );
        }
        let (raw, snaps) = open_block_buffers(head_dim, device)?;
        if n_open > 0 {
            write_host(&raw, open)?;
        }
        let mut keys = KeyPages::new(head_dim, device);
        keys.write_host_rows(rows)?;
        Ok(Self {
            keys,
            n_blocks: n,
            raw,
            n_open,
            snaps,
            pages: Vec::new(),
            page_rows: vec![0],
            tail_base: 0,
            // A restored cache holds no projection; the next one sets it.
            prompt_end: 0,
        })
    }

    /// Append an injected prefix's index rows ahead of the live tail.
    ///
    /// Called when a sequence receives K/V it did not forward — today, a sealed
    /// section borrowed into a projection. The page keeps its own width because
    /// the piece it describes ended wherever its tokens ended; see
    /// [`Self::pages`].
    ///
    /// Refused once the sequence has forwarded anything, because a page landing
    /// after live rows would sit at the wrong positions: pages describe a
    /// prefix, and the live tail is what follows them.
    ///
    /// `base` is the absolute position the page's first row occupies here, and
    /// it is the whole of what makes a sealed page injectable: the rows arrive
    /// un-rotated, and the scorer rotates row `j` at `base + j·ratio`. A caller
    /// that knows only "after the last one" passes [`Self::next_base`].
    ///
    /// Recording only: the page is already resident, so pushing it moves no
    /// bytes, and the same page may sit in any number of caches.
    pub fn push_page(&mut self, page: Arc<ResidentPage>, base: usize, ratio: usize) -> Result<()> {
        if self.n_blocks != 0 || self.n_open != 0 {
            candle::bail!(
                "qsa index: a page arrived after {} live block(s) and {} carried row(s) — \
                 injected prefixes must precede anything this sequence forwarded",
                self.n_blocks,
                self.n_open,
            );
        }
        if base < self.tail_base {
            candle::bail!(
                "qsa index: a page placed at {base} would overlap the {} token(s) already \
                 placed — pages must ascend and may not overlap",
                self.tail_base,
            );
        }
        let rows = page.rows();
        let tokens = page.tokens(ratio);
        self.page_rows.push(self.page_rows.last().unwrap() + rows);
        self.pages.push(PlacedPage { page, base, tokens });
        self.tail_base = base + tokens;
        Ok(())
    }

    /// The position a page pushed now would sit at if it abuts the last one —
    /// what a caller injecting a contiguous run passes to [`Self::push_page`].
    pub fn next_base(&self) -> usize {
        self.tail_base
    }

    /// Move the live tail's opening position to `base` without pushing rows.
    ///
    /// For K/V injected with no index rows at all — a piece sealed before pages
    /// existed. Under the accumulating layout this had to be a zero-row *page*
    /// (`push_gap`), because a span that advanced the token sum without
    /// advancing it would have slid every later page's implied start earlier by
    /// its width. Positions are now recorded rather than summed, so the span is
    /// simply not indexed: no page moves, nothing needs to stand in for it, and
    /// all that is left to say is where the tail resumes.
    pub fn skip_to(&mut self, base: usize) -> Result<()> {
        if self.n_blocks != 0 || self.n_open != 0 {
            candle::bail!(
                "qsa index: the tail was moved to {base} after {} live block(s) and {} carried \
                 row(s) — an injected prefix must precede anything this sequence forwarded",
                self.n_blocks,
                self.n_open,
            );
        }
        if base < self.tail_base {
            candle::bail!(
                "qsa index: the tail cannot move back to {base} from {}",
                self.tail_base,
            );
        }
        self.tail_base = base;
        Ok(())
    }

    /// The position the live tail starts at — everything placed ahead of it.
    pub fn page_token_span(&self) -> usize {
        self.tail_base
    }

    /// Injected pages held, oldest first.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Page `i` and the tokens it covers.
    ///
    /// **Pages are atomic to a caller that wants to hand a span on.** A page's
    /// last row is ragged — it covers `last_cells` tokens, not `ratio` — so a
    /// span cannot start partway through one without re-pooling rows across a
    /// boundary the original piece ended at. A caller taking a trailing span
    /// therefore takes whole pages and stops when it has covered enough.
    pub fn page_at(&self, i: usize) -> Option<(&Arc<ResidentPage>, usize)> {
        let p = self.pages.get(i)?;
        Some((&p.page, p.tokens))
    }

    /// Page `i`'s absolute position in this cache.
    pub fn page_base(&self, i: usize) -> Option<usize> {
        self.pages.get(i).map(|p| p.base)
    }

    /// Rows held in injected pages.
    pub fn page_row_span(&self) -> usize {
        *self.page_rows.last().unwrap_or(&0)
    }

    /// Tokens this cache actually covers — injected pages plus the live tail.
    ///
    /// The quantity a query position is checked against, and the one that has
    /// to agree across every one of a sequence's caches: they index the same
    /// stream, so a cache holding fewer tokens than its siblings has missed
    /// appends the others took. Below the identity threshold nothing reads it,
    /// so a divergence here is silent until a select refuses at depth.
    pub fn indexed_tokens(&self, ratio: usize) -> usize {
        self.page_token_span() + self.n_blocks * ratio + self.n_open
    }

    /// Blocks wholly at or below `pos` — the ragged replacement for
    /// `(pos + 1) / ratio`.
    ///
    /// Walks the injected pages by their own widths, then counts uniformly
    /// inside the live tail. Identical to the uniform formula for a sequence
    /// with no pages, which is every sequence that forwarded its whole prefix.
    pub fn candidates_at(&self, pos: usize, ratio: usize) -> usize {
        let limit = pos + 1;
        if limit > self.tail_base {
            // Inside the live tail: the pages are wholly below, and the tail
            // pools uniformly from its own opening position.
            return self.page_row_span() + (limit - self.tail_base) / ratio;
        }
        // Inside the placed pages: the last one that starts at or before `pos`.
        // Searched by BASE rather than by a running width sum, so a position
        // that falls in a hole between two pages resolves to the page before it
        // with all of that page's rows below — which is the truth, and what the
        // width sum could not express.
        let p = match self.pages.binary_search_by_key(&limit, |p| p.base) {
            // A page opening exactly at `limit` opens after `pos`, so it and
            // everything above it are not below.
            Ok(i) => return self.page_rows[i],
            Err(0) => return 0,
            Err(i) => i - 1,
        };
        let placed = &self.pages[p];
        let rows = placed.page.rows();
        let inside = limit - placed.base;
        // The last row is short — it covers `last_cells`, not `ratio` — so it
        // only counts once the position has reached the page's full width.
        let whole = if inside >= placed.tokens {
            rows
        } else {
            (inside / ratio).min(rows.saturating_sub(1))
        };
        self.page_rows[p] + whole
    }

    /// The live prefix, `[n_blocks, head_dim]` row-major, read back to the host
    /// out of the key pages.
    ///
    /// What a seal writes. The rows above `n_blocks` are dead until an append
    /// writes them, so handing out the whole capacity would persist uninitialised
    /// memory. The scorer never comes here — it reads the pages in place.
    pub fn live_rows_host(&self) -> Result<Vec<f32>> {
        self.keys.host_rows(self.n_blocks)
    }

    /// The carried open block as a view — `[n_open, head_dim]`, no copy.
    ///
    /// The same rule as [`Self::live_rows_host`]: the rows above `n_open` are
    /// dead until the next append writes them.
    pub fn open_rows(&self) -> Result<Tensor> {
        self.raw.narrow(0, 0, self.n_open)
    }

    /// The carried open block, read back to the host — what a seal writes beside
    /// [`Self::live_rows_host`].
    pub fn open_rows_host(&self) -> Result<Vec<f32>> {
        self.open_rows()?.flatten_all()?.to_vec1::<f32>()
    }

    /// Rows the cache has completed, and the tokens still carried in the open
    /// block — the two numbers a turn seal needs to describe its page.
    pub fn seal_shape(&self) -> (usize, usize) {
        (self.n_blocks, self.n_open)
    }

    /// An independent copy of the whole cache — the view-carve fork.
    ///
    /// **Distinct from [`Self::snapshot`], which is a rewind point.** A snapshot
    /// keeps only what a failed wave must put back: the open block's rows and
    /// the two counters, because the keys above `n_blocks` are dead rows the
    /// re-append will overwrite. A fork is a different question — the child is
    /// about to append to keys the parent still owns, so the live prefix
    /// `[0, n_blocks)` has to come with it.
    ///
    /// Why the child needs it at all: a view borrows the parent's KV blocks
    /// zero-copy, so its sequence already contains the parent's tokens. An
    /// index that started empty there would leave selection scoring a handful
    /// of blocks against a KV holding the whole history — the mismatch is
    /// silent, and it reads as a retrieval that simply chose badly.
    ///
    /// The copy is `n_blocks × head_dim` floats per attention layer, into only
    /// the pages those keys reach, so it is proportional to depth rather than to
    /// the parent's capacity. That is the same shape of cost the recurrent store's
    /// `fork_from` already pays at every view carve.
    pub fn fork(&self) -> Result<Self> {
        let (raw, snaps) = open_block_buffers(self.keys.head_dim(), self.keys.device())?;
        if self.n_open > 0 {
            raw.slice_set(&self.open_rows()?, 0, 0)?;
        }
        Ok(Self {
            keys: self.keys.fork(self.n_blocks)?,
            n_blocks: self.n_blocks,
            raw,
            n_open: self.n_open,
            snaps,
            // The pages are shared, not copied: a page is a sealed prefix that
            // nothing appends to and is resident once, so parent and child read
            // the same rows through the same `Arc`. Only the live tail above them
            // is written, and that is copied.
            pages: self.pages.clone(),
            page_rows: self.page_rows.clone(),
            tail_base: self.tail_base,
            prompt_end: self.prompt_end,
        })
    }

    /// Move every buffer of this cache whose slot is the source of a planned move
    /// onto that move's destination — key pages, the open block, the free rewind
    /// buffers, and the chunks of the pages it holds. Answers how many moved.
    ///
    /// A page shared with other caches is moved by whichever reaches it first;
    /// the rest find its move already taken and read the new address through the
    /// same `Arc`.
    ///
    /// Between forwards only: a forward resolves page addresses into its tables
    /// as it runs, and between forwards nothing holds one.
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut moved = self.keys.relocate(moves)?
            + usize::from(relocate_tensor(&mut self.raw, moves)?)
            + self.snaps.relocate(moves)?;
        for p in &self.pages {
            moved += p.page.relocate(moves)?;
        }
        Ok(moved)
    }

    /// Room for the blocks a sequence at `tokens` tokens will have completed in
    /// its live tail.
    ///
    /// Called at wave admission. The tail opens at [`Self::next_base`], so the
    /// tokens injected pages already cover take no key pages here. Growth is a
    /// page at a time and moves nothing already written.
    pub fn ensure_capacity(&mut self, tokens: usize, ratio: usize) -> Result<()> {
        self.keys
            .ensure(tokens.saturating_sub(self.tail_base) / ratio + 1)
    }

    /// Start the sequence over — including its injected prefix. Returns the
    /// injected pages it held.
    ///
    /// The pages go too. They describe positions this slot held; a slot
    /// starting over holds none of them, and leaving them would put the next
    /// sequence's first token at the old prefix's end. The key pages go back to
    /// the span with them; admission claims what the new sequence needs.
    ///
    /// The pages are handed back rather than dropped because a slot that starts
    /// over to be rebuilt re-injects most of the same pieces: a caller that
    /// keeps them alive across the rebuild finds each one still resident
    /// instead of decoding and placing it again.
    pub fn reset(&mut self) -> Vec<Arc<ResidentPage>> {
        self.keys.clear();
        self.n_blocks = 0;
        self.n_open = 0;
        self.page_rows.truncate(1);
        self.tail_base = 0;
        self.prompt_end = 0;
        self.pages.drain(..).map(|p| p.page).collect()
    }

    /// The largest position at or before `pos` that [`Self::truncate_to`]
    /// accepts: the start of the page `pos` falls inside, or `pos` itself when
    /// no page straddles it.
    ///
    /// A page can span several pieces — a slot's whole injected prefix is
    /// closed into one page at a unit boundary — so a rebuild that wants to keep
    /// a prefix ending inside it must keep less, and re-inject from the page's
    /// start.
    pub fn cut_floor(&self, pos: usize) -> usize {
        self.pages
            .iter()
            .find(|p| p.base < pos && pos < p.base + p.tokens)
            .map_or(pos, |p| p.base)
    }

    /// Cut the cache back to position `pos`: keep the injected pages that end at
    /// or below it, drop the rest and the live tail, and resume the tail at
    /// `pos`. Returns the pages dropped.
    ///
    /// For a rebuild that keeps its slot's prefix up to a piece boundary and
    /// re-injects only what follows. A page is atomic — its last row is ragged —
    /// so a cut that falls inside one is refused rather than splitting it. The
    /// caller chooses its cut through [`Self::cut_floor`], so a cut anywhere
    /// else means the two have diverged.
    pub fn truncate_to(&mut self, pos: usize) -> Result<Vec<Arc<ResidentPage>>> {
        let keep = self
            .pages
            .iter()
            .take_while(|p| p.base + p.tokens <= pos)
            .count();
        if let Some(straddles) = self.pages.get(keep).filter(|p| p.base < pos) {
            candle::bail!(
                "qsa index: a cut at {pos} falls inside the page covering {}..{} — pages \
                 are atomic, so a cut must land on a page boundary",
                straddles.base,
                straddles.base + straddles.tokens,
            );
        }
        self.keys.clear();
        self.n_blocks = 0;
        self.n_open = 0;
        self.page_rows.truncate(keep + 1);
        self.tail_base = pos;
        // A prompt is a prefix: a cut below its end keeps only what it kept.
        self.prompt_end = self.prompt_end.min(pos);
        Ok(self.pages.drain(keep..).map(|p| p.page).collect())
    }

    /// Blocks this span would complete, and the rows it would leave open.
    fn plan(&self, rows: usize, ratio: usize) -> (usize, usize) {
        let have = self.n_open + rows;
        let n_new = have / ratio;
        (n_new, have - n_new * ratio)
    }

    /// Device address of this cache's open-block buffer.
    fn raw_ptr(&self) -> Result<u64> {
        tensor_ptr(&self.raw)
    }

    /// Score this segment's queries against the cache, into the wave's shared
    /// score buffer. Returns each row's candidate-block count.
    ///
    /// `q` is this sequence's rows of [`project_queries`], `[T, n_heads,
    /// head_dim]`, already normed and rotated. `qpos[i]` is row `i`'s absolute
    /// position. Rows land at `row_base`, `out_stride` apart. The stored keys
    /// are rotated from `rope` as they are read, at `rung` — this sequence's,
    /// the one its queries were rotated at.
    ///
    /// Scoring and SELECTING are split because they want different launch
    /// shapes. A span's scores must be computed against that span's own cache,
    /// so the matmul is per sequence; the top-k that follows is one block per
    /// ROW, and a span contributes about five of them — which on a 110-SM part
    /// is a kernel running on five SMs. `nsys` put that top-k at 46% of the
    /// engaged regime's GPU time. Writing every span into one buffer lets the
    /// whole wave's rows go in a single launch instead.
    #[allow(clippy::too_many_arguments)]
    pub fn score_rows(
        &self,
        q: &Tensor,
        qpos: &[usize],
        cfg: &IndexerConfig,
        ratio: usize,
        rope: &FactoredRope,
        rung: u32,
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
        ticket: Option<WaveTicket>,
    ) -> Result<Vec<u32>> {
        self.score_rows_routed(
            q,
            qpos,
            cfg,
            ratio,
            rope,
            rung,
            out,
            out_stride,
            row_base,
            TailRoute::for_span,
            ticket,
        )
    }

    /// [`Self::score_rows`] with the live tail's route picked by `route`, from
    /// the span's rows and live-tail blocks — what the harness measures both
    /// routes through, at the same shapes, to place [`GEMM_TAIL_MIN_CELLS`].
    #[allow(clippy::too_many_arguments)]
    pub fn score_rows_routed(
        &self,
        q: &Tensor,
        qpos: &[usize],
        cfg: &IndexerConfig,
        ratio: usize,
        rope: &FactoredRope,
        rung: u32,
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
        route: impl FnOnce(usize, usize) -> TailRoute,
        ticket: Option<WaveTicket>,
    ) -> Result<Vec<u32>> {
        let (t, qh, qd) = q.dims3()?;
        if t != qpos.len() {
            candle::bail!("qsa select: {t} rows against {} positions", qpos.len());
        }
        if t == 0 {
            return Ok(Vec::new());
        }
        let d = cfg.head_dim;
        let h = cfg.n_heads;
        if qh != h || qd != d {
            candle::bail!("qsa select: queries are [{t}, {qh}, {qd}] against [_, {h}, {d}]");
        }
        let q = q.reshape((t * h, d))?;

        // Per row, the candidate blocks are those wholly below its tail. With
        // injected pages ahead of the live tail this is a walk over their
        // widths rather than a division — see `candidates_at`.
        let cand: Vec<u32> = qpos
            .iter()
            .map(|&p| self.candidates_at(p, ratio) as u32)
            .collect();
        let cand_max = cand.iter().copied().max().unwrap_or(0) as usize;
        let held = self.page_row_span() + self.n_blocks;
        if cand_max > held {
            // Report the TOKEN accounting, not just the row counts. The rows say
            // the select cannot proceed; the tokens say why, and they are what
            // identifies the piece at fault: `unindexed` is exactly how many
            // tokens of this sequence's K/V no page and no append ever covered,
            // so it can be matched against the width of whatever the assembler
            // last put in the slot. Rows alone leave that invisible, because a
            // ragged page holds fewer tokens than `rows × ratio`.
            let pos = qpos.iter().max().copied().unwrap_or(0);
            let page_tok = self.page_token_span();
            let tail_tok = self.n_blocks * ratio + self.n_open;
            candle::bail!(
                "qsa select: a query at position {pos} needs {cand_max} blocks but the \
                 index cache holds {held} ({} in injected pages, {} forwarded) — the \
                 segment's keys were not appended first. Tokens: {} indexed \
                 ({page_tok} in pages + {tail_tok} in the live tail) against a query at \
                 {pos}, so {} token(s) of this sequence carry K/V that no page and no \
                 append ever covered",
                self.page_row_span(),
                self.n_blocks,
                page_tok + tail_tok,
                (pos + 1).saturating_sub(page_tok + tail_tok),
            );
        }

        // **Narrow spans: one launch over the pages AND the live tail.** The
        // paged scorer takes the tail as one more page, read row-major as the
        // append wrote it, and rotates every key — page or tail — as it loads
        // it. Column order is page rows then live rows, which is block order,
        // so the selection that follows indexes them exactly as it always has.
        //
        // **Wide spans over a wide tail: the live tail goes to cuBLAS.** From
        // [`GEMM_TAIL_MIN_CELLS`] up the paged scorer's grid re-reading every key
        // once per row tile costs more than a GEMM that stages a key tile once
        // and reuses it across its rows. The tail is rotated once into a
        // scratch buffer that lives for this call, and the pages stay on the
        // paged scorer.
        let page_cols = self.page_row_span();
        let live_cols = cand_max.saturating_sub(page_cols);
        let tail_in_paged = live_cols > 0 && route(t, live_cols) == TailRoute::Paged;
        let paged_cols = if tail_in_paged { cand_max } else { page_cols };
        if paged_cols > 0 {
            self.score_paged(
                &q,
                &cand,
                PagedScore {
                    t,
                    h,
                    d,
                    n_cand: paged_cols,
                    with_tail: tail_in_paged,
                    ratio,
                },
                rope,
                rung,
                out,
                out_stride,
                row_base,
                ticket,
            )?;
        }
        if tail_in_paged || live_cols == 0 {
            return Ok(cand);
        }

        // The scan's right operand is the rotated tail **transposed**, and that
        // is a view: cuBLAS takes the transpose as `OP_T` with
        // `lda = head_dim`. Live-tail columns only: the pages above already
        // covered theirs.
        // Read through the key pages' table, in place — the live tail is not one
        // dense block, and gathering it into one would be a copy per span per
        // layer per wave (hot-path invariant 2b).
        let page_ptrs = self.keys.page_ptrs()?;
        let rotated = rotate_rows(
            RowSource::Paged {
                pages: &page_ptrs,
                rows_per_page: PAGE_BLOCKS,
                rows: live_cols,
                d,
                device: self.keys.device(),
            },
            rope,
            1,
            RowPositions::Affine {
                base: self.tail_base,
                step: ratio,
            },
            RowRungs::Uniform(rung),
            RotSide::Key,
            ticket,
        )?;
        // Only narrowed when pages actually sit ahead of the tail. With none,
        // `page_cols` is 0 and the narrow is the whole buffer — a view that
        // costs a `Tensor` per span per layer per wave to describe what `out`
        // already is.
        let narrowed;
        let live_out = if page_cols == 0 {
            out
        } else {
            narrowed = out.narrow(1, page_cols, live_cols)?;
            &narrowed
        };
        let keys_t = rotated.t()?;
        let rows_per_tile = (SCORE_TILE_BYTES / (h * live_cols * 4)).clamp(1, t);
        let mut row = 0usize;
        while row < t {
            let rows = rows_per_tile.min(t - row);
            // [rows·h, d] × [d, cand_max], then ReLU and the fold over heads in
            // ONE pass through `indexer_score_reduce`, straight into this
            // span's rows of the WAVE's score buffer.
            //
            // Eagerly the fold was a `relu` plus `h − 1` strided adds, which
            // `nsys` over the engaged bench measured at 19.4% of all GPU time —
            // 1,270 `badd_f32` and 406 `urelu_f32` launches reading a
            // `[rows, h, cand]` intermediate `h` times to write a `[rows, cand]`
            // result. The fused kernel reads it once, and carries the
            // accumulation's low-order bits so the scores it hands the argsort
            // are strictly more faithful than the chain's (see its own notes).
            //
            // No per-head weight: this fold is a plain `Σ_h relu(·)`, which the
            // kernel spells as a null `w`.
            let raw = q.narrow(0, row * h, rows * h)?.matmul(&keys_t)?;
            fold_heads_into(
                &raw,
                rows,
                h,
                live_cols,
                live_out,
                out_stride,
                row_base + row,
            )?;
            row += rows;
        }
        Ok(cand)
    }

    /// Score the injected pages — and, with `with_tail`, the live tail as one
    /// more page — into columns `[0, n_cand)`.
    ///
    /// The pages are separately allocated and ragged, which is exactly the
    /// descriptor-table shape `qsa_score_paged` takes: one `{keys, strides,
    /// delta}` entry per page, and the widths folded into `cnt` on the host so
    /// the kernel never sees one (hot-path invariant 2b — nothing is
    /// concatenated). `delta` is what the kernel rotates by: page `p`'s global
    /// row `g` sits at `delta + g·ratio`.
    ///
    /// `cnt` is the FULL candidate count per row, page rows and live rows
    /// together, and the kernel masks anything past it — a row whose candidates
    /// run into the live tail simply has every page column visible, which is
    /// what "wholly below" means for a prefix.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn score_paged(
        &self,
        q: &Tensor,
        cand: &[u32],
        shape: PagedScore,
        rope: &FactoredRope,
        rung: u32,
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
        // The span the page/candidate tables below belong to — rebuilt per
        // scored layer and dead once the launch is issued.
        ticket: Option<WaveTicket>,
    ) -> Result<()> {
        use candle_kernels::simple::qsa_score_paged::{run_qsa_score_paged, PAGE_WORDS};

        let PagedScore {
            t,
            h,
            d,
            n_cand,
            with_tail,
            ratio,
        } = shape;
        let device = self.keys.device().clone();
        // One entry per chunk of every page, in row order. A page's rows are
        // consecutive global rows, so every chunk of it shares the page's
        // rotation offset; the entries' own row counts come from the chunks.
        let mut desc: Vec<i64> = Vec::with_capacity((self.pages.len() + 1) * PAGE_WORDS);
        let mut first: Vec<u32> = Vec::with_capacity(self.pages.len() + 2);
        first.push(0);
        let mut end = 0usize;
        for (i, placed) in self.pages.iter().enumerate() {
            let delta = placed.base as i64 - (self.page_rows[i] * ratio) as i64;
            for chunk in placed.page.descriptors()? {
                desc.push(chunk.ptr as i64);
                desc.push(chunk.group_stride);
                desc.push(chunk.row_stride);
                desc.push(delta);
                end += chunk.rows;
                first.push(end as u32);
            }
        }
        if end != self.page_row_span() {
            candle::bail!(
                "qsa index: the pages' chunks hold {end} rows against the {} the cache \
                 accounts for",
                self.page_row_span(),
            );
        }
        let mut n_pages = first.len() - 1;
        if with_tail {
            let span = self.page_row_span();
            // One entry per key page the live tail reaches, each row-major as the
            // append writes it: group `c` of row `j` at `c + j·(d/4)` float4s. The
            // tail's rows are consecutive global rows across its pages, so every
            // page shares the tail's `delta`.
            let delta = self.tail_base as i64 - (span * ratio) as i64;
            let mut end = span;
            for (ptr, used) in self.keys.tail(self.n_blocks)? {
                desc.push(ptr as i64);
                desc.push(1);
                desc.push((d / 4) as i64);
                desc.push(delta);
                end += used;
                first.push(end as u32);
                n_pages += 1;
            }
        }
        let n_desc = desc.len();
        let pages_tbl = wave_from_vec_ticketed(desc, (n_desc,), &device, ticket)?;
        let first_tbl = wave_from_vec_ticketed(first, (n_pages + 1,), &device, ticket)?;
        let cnt_t = wave_from_vec_ticketed(cand.to_vec(), (t,), &device, ticket)?;
        let table = rope.table(rung)?;

        let candle::Device::Cuda(cuda) = &device else {
            candle::bail!("qsa paged score runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        candle::set_kernel_breadcrumb("run_qsa_score_paged", file!(), line!());
        unsafe {
            run_qsa_score_paged(
                tensor_ptr(q)? as *const f32,
                i64_ptr(&pages_tbl)? as *const i64,
                u32_ptr(&first_tbl)? as *const u32,
                u32_ptr(&cnt_t)? as *const u32,
                tensor_ptr(&table)? as *const f32,
                tensor_ptr(rope.steps(rung, ratio)?)? as *const f32,
                tensor_ptr(out)? as *mut f32,
                t as i32,
                h as i32,
                d as i32,
                n_cand as i32,
                n_pages as i32,
                rope.pairs() as i32,
                ratio as i32,
                out_stride as i64,
                row_base as i64,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok(())
    }
}

/// The shape of one paged-scorer launch.
#[derive(Clone, Copy)]
struct PagedScore {
    /// Query rows.
    t: usize,
    /// Indexer heads per row.
    h: usize,
    /// Indexer head width.
    d: usize,
    /// Candidate columns the launch writes.
    n_cand: usize,
    /// Whether the live tail is scored as the last page.
    with_tail: bool,
    ratio: usize,
}

/// Where each group of rows sits, for [`rotate_rows`].
pub enum RowPositions<'a> {
    /// One position per group.
    PerGroup(&'a [usize]),
    /// Group `k` at `base + k·step`.
    Affine { base: usize, step: usize },
}

/// Which rung each group of rows rotates at, for [`rotate_rows`].
#[derive(Clone, Copy)]
pub enum RowRungs<'a> {
    /// One rung per group — a wave's queries, one sequence's rung per row.
    PerGroup(&'a [u32]),
    /// Every group at one rung — one sequence's keys.
    Uniform(u32),
}

/// What is being rotated, which decides the scale on the rotated channels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RotSide {
    /// Queries take the rung's `m²` (`docs/progressive_yarn.md` §4.2).
    Query,
    /// Keys take none.
    Key,
}

/// The rows [`rotate_rows`] reads.
pub enum RowSource<'a> {
    /// One dense `[n, d]` tensor.
    Dense(&'a Tensor),
    /// One dense `[n, d]` tensor RMS-normed over each row with `weight` and
    /// `eps` before it rotates — a wave's queries, normed and rotated in one
    /// launch with `qsa::rms_norm_last`'s arithmetic.
    Normed {
        rows: &'a Tensor,
        weight: &'a Tensor,
        eps: f64,
    },
    /// `rows` rows of width `d` in pages of `rows_per_page`, row `r` at
    /// `pages[r / rows_per_page] + (r % rows_per_page)·d` floats — a live tail's
    /// keys, read in place.
    Paged {
        pages: &'a [u64],
        rows_per_page: usize,
        rows: usize,
        d: usize,
        device: &'a Device,
    },
}

/// Rotate the rows of `src` at their positions and rungs from `rope`, into a new
/// dense `[n, d]` tensor.
///
/// Consecutive runs of `rows_per_pos` rows share a position and a rung — a
/// query's heads all sit at the query's position, on its sequence's rung.
#[cfg(feature = "cuda")]
pub fn rotate_rows(
    src: RowSource<'_>,
    rope: &FactoredRope,
    rows_per_pos: usize,
    positions: RowPositions<'_>,
    rungs: RowRungs<'_>,
    side: RotSide,
    // The open layer phase, for the position and rung tables below.
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    use candle_kernels::simple::qsa_rope_rows::{
        run_qsa_rope_rows, run_qsa_rope_rows_norm, QSA_ROPE_NORM_MAX_D,
    };

    let (n, d, device) = match &src {
        RowSource::Dense(t) => {
            expect_dense(t, "qsa rope rows")?;
            let (n, d) = t.dims2()?;
            (n, d, t.device())
        }
        RowSource::Normed { rows, weight, .. } => {
            expect_dense(rows, "qsa rope rows")?;
            expect_dense_dtype(weight, DType::F32, "qsa rope rows norm weight")?;
            let (n, d) = rows.dims2()?;
            if weight.dims() != [d] || d > QSA_ROPE_NORM_MAX_D {
                candle::bail!(
                    "qsa rope rows: a {:?} norm weight on {d}-wide rows (at most {})",
                    weight.dims(),
                    QSA_ROPE_NORM_MAX_D
                );
            }
            (n, d, rows.device())
        }
        RowSource::Paged {
            pages,
            rows_per_page,
            rows,
            d,
            device,
        } => {
            if *rows_per_page == 0 || rows.div_ceil(*rows_per_page) > pages.len() {
                candle::bail!(
                    "qsa rope rows: {rows} rows in pages of {rows_per_page} against a \
                     {}-page table",
                    pages.len()
                );
            }
            (*rows, *d, *device)
        }
    };
    if rope.rope_dim() > d {
        candle::bail!(
            "qsa rope rows: a {}-wide rotary width on {d}-wide rows",
            rope.rope_dim()
        );
    }
    // The kernel's own guard refuses to launch when `rows_per_pos == 0`
    // (`row / rows_per_pos` would divide by zero) and returns without writing
    // a single byte of `dst` — silently leaving it uninitialised rather than
    // erroring. Catch it here, before `dst` is even allocated.
    if rows_per_pos == 0 {
        candle::bail!("qsa rope rows: rows_per_pos must be at least 1");
    }
    let dst = wave_empty_ticketed((n, d), DType::F32, device, ticket)?;
    if n == 0 {
        return Ok(dst);
    }
    let groups = n / rows_per_pos;
    let (pos_t, base, step) = match positions {
        RowPositions::PerGroup(p) => {
            if p.len() * rows_per_pos != n {
                candle::bail!(
                    "qsa rope rows: {} positions for {n} rows in groups of {rows_per_pos}",
                    p.len()
                );
            }
            let v: Vec<u32> = p.iter().map(|&x| x as u32).collect();
            (
                Some(wave_from_vec_ticketed(v, (p.len(),), device, ticket)?),
                0,
                0,
            )
        }
        RowPositions::Affine { base, step } => (None, base, step),
    };
    let (rung_t, rung) = match rungs {
        RowRungs::PerGroup(r) => {
            if r.len() != groups {
                candle::bail!(
                    "qsa rope rows: {} rungs for {groups} groups of {rows_per_pos}",
                    r.len()
                );
            }
            (
                Some(wave_from_vec_ticketed(
                    r.to_vec(),
                    (r.len(),),
                    device,
                    ticket,
                )?),
                0,
            )
        }
        RowRungs::Uniform(r) => (None, r),
    };
    let n_rungs = rope.rungs().n_rungs() as u32;
    let top = match rungs {
        RowRungs::PerGroup(r) => r.iter().copied().max().unwrap_or(0),
        RowRungs::Uniform(r) => r,
    };
    if top >= n_rungs {
        candle::bail!("qsa rope rows: rung {top} of a {n_rungs}-rung set");
    }
    let pos_ptr = match &pos_t {
        Some(t) => u32_ptr(t)? as *const u32,
        None => std::ptr::null(),
    };
    let rung_ptr = match &rung_t {
        Some(t) => u32_ptr(t)? as *const u32,
        None => std::ptr::null(),
    };
    // The page table, uploaded for the launch; the dense source needs none.
    let (src_ptr, pages_t, rows_per_src_page) = match &src {
        RowSource::Dense(t) => (tensor_ptr(t)?, None, 0usize),
        RowSource::Normed { rows, .. } => (tensor_ptr(rows)?, None, 0usize),
        RowSource::Paged {
            pages,
            rows_per_page,
            rows,
            ..
        } => {
            let used: Vec<i64> = pages[..rows.div_ceil(*rows_per_page)]
                .iter()
                .map(|&p| p as i64)
                .collect();
            let len = used.len();
            (
                0,
                Some(wave_from_vec_ticketed(used, (len,), device, ticket)?),
                *rows_per_page,
            )
        }
    };
    let pages_ptr = match &pages_t {
        Some(t) => i64_ptr(t)? as *const i64,
        None => std::ptr::null(),
    };
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("qsa rope rows runs on CUDA");
    };
    let stream = cuda.cuda_stream();
    if let RowSource::Normed { weight, eps, .. } = &src {
        candle::set_kernel_breadcrumb("run_qsa_rope_rows_norm", file!(), line!());
        // SAFETY: `src_ptr` is `[n, d]` dense F32, `weight` `[d]` F32 and `dst`
        // `[n, d]`, checked above; the position and rung tables are sized to the
        // groups.
        unsafe {
            run_qsa_rope_rows_norm(
                src_ptr as *const f32,
                tensor_ptr(&dst)? as *mut f32,
                n as i32,
                d as i32,
                rows_per_pos as i32,
                pos_ptr,
                base as i64,
                step as i32,
                rope.rungs().ffi()?,
                rung_ptr,
                rung,
                i32::from(side == RotSide::Query),
                tensor_ptr(weight)? as *const f32,
                *eps as f32,
                stream.cu_stream() as *mut c_void,
            );
        }
        return Ok(dst);
    }
    candle::set_kernel_breadcrumb("run_qsa_rope_rows", file!(), line!());
    unsafe {
        run_qsa_rope_rows(
            src_ptr as *const f32,
            pages_ptr,
            rows_per_src_page as i32,
            tensor_ptr(&dst)? as *mut f32,
            n as i32,
            d as i32,
            rows_per_pos as i32,
            pos_ptr,
            base as i64,
            step as i32,
            rope.rungs().ffi()?,
            rung_ptr,
            rung,
            i32::from(side == RotSide::Query),
            stream.cu_stream() as *mut c_void,
        );
    }
    Ok(dst)
}

/// The row stride of a selection table whose deepest query sees `cand_max`
/// candidate blocks under `strata` — or why the selection kernel cannot run it.
///
/// Two ceilings, both the kernel's own and mirrored from it:
///
/// - **Survivors per window.** Each window streams its pool through one
///   buffer of `MAX_KEEP` survivors, so the budget must keep no more than that
///   ([`max_keep`]). Windows do not add up here — each starts over.
/// - **Entries gathered per row.** Every window's choice is gathered, sorted
///   and de-duplicated in one shared buffer of a power-of-two width, up to
///   `MAX_ENTRIES`, so the gathered count ([`max_gathered_for`], repeats
///   included) must round up inside it. More windows at a given depth is what
///   reaches it.
///
/// The stride itself is what survives the union, [`max_entries_for`].
pub fn selection_stride(
    ratio: usize,
    top_k: usize,
    cand_max: usize,
    strata: &Strata,
) -> Result<usize> {
    if ratio == 0 || ratio > MAX_RATIO {
        candle::bail!("qsa: compression ratio {ratio} outside 1..={MAX_RATIO}");
    }
    // `max_keep`, not a second copy of the arithmetic: this guard advertises
    // the selection kernel's survivor ceiling, so it has to be the same bound
    // the kernel actually reaches.
    let keep_max = max_keep(top_k, ratio);
    if keep_max > MAX_KEEP {
        candle::bail!(
            "qsa: top_k {top_k} at ratio {ratio} needs {keep_max} survivors, past the \
             selection kernel's {MAX_KEEP} — its streaming buffer could not absorb a chunk"
        );
    }
    let gathered = max_gathered_for(top_k, ratio, cand_max, strata);
    if gathered.next_power_of_two() > MAX_ENTRIES {
        candle::bail!(
            "qsa: {} window(s) of {} blocks at {cand_max} candidate blocks gather up to \
             {gathered} entries a row, past the selection kernel's {MAX_ENTRIES} — widen the \
             window",
            strata.windows(cand_max),
            strata.window_width(cand_max),
        );
    }
    Ok(max_entries_for(top_k, ratio, cand_max, strata))
}

/// The wave's selection table for one layer: one row per query, in the wave's
/// packed row order (decode slots first, then prefill rows).
///
/// Allocated once per layer and filled per sequence, so the attention kernels
/// take one table and index it exactly as they index their queries.
pub struct SelectionTable {
    entries: Tensor,
    cnt: Tensor,
    stride: usize,
    // The kernel's union buffer, in entries: the gathered bound rounded up to
    // the power of two its bitonic sort runs over.
    gather: usize,
    // The kernel's split scratch (`qsa_topk::SPLIT_KEYS` u64, as u32 pairs):
    // what a launch too narrow to fill the device ranks each row's slices into.
    // Uninitialised — every key the merge reads, the same launch wrote. Absent
    // when this table's launch fills the device and runs the single pass.
    split_keys: Option<Tensor>,
    ratio: usize,
    strata: Strata,
}

/// The split scratch a selection over `rows` rows needs — `None` when the
/// launch fills the device and runs the single pass, which reads none.
#[cfg(feature = "cuda")]
fn split_scratch(
    rows: usize,
    ratio: usize,
    top_k: usize,
    cand_max: usize,
    strata: &Strata,
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Option<Tensor>> {
    if !device.is_cuda() {
        return Ok(None);
    }
    let parts = unsafe {
        qsa_topk_split_parts(
            rows as i32,
            cand_max as i32,
            strata.window_blocks as i32,
            top_k as i32,
            ratio as i32,
        )
    };
    if parts == 0 {
        return Ok(None);
    }
    Ok(Some(wave_empty_ticketed(
        (SPLIT_KEYS * 2,),
        DType::U32,
        device,
        ticket,
    )?))
}

/// No selection kernel runs without CUDA, so nothing splits.
#[cfg(not(feature = "cuda"))]
fn split_scratch(
    _rows: usize,
    _ratio: usize,
    _top_k: usize,
    _cand_max: usize,
    _strata: &Strata,
    _device: &Device,
    _ticket: Option<WaveTicket>,
) -> Result<Option<Tensor>> {
    Ok(None)
}

impl SelectionTable {
    /// An uninitialised table for `rows` queries whose deepest sees `cand_max`
    /// candidate blocks, selecting under `strata` — on `ticket`'s arena when one
    /// is given, the layer phase the attention that reads it runs in.
    ///
    /// Uninitialised is correct, not sloppy: the kernels read `entries` only
    /// below each row's `cnt`, and every row's `cnt` is written by the
    /// selection kernel (hot-path invariant 6).
    pub fn new(
        rows: usize,
        ratio: usize,
        top_k: usize,
        cand_max: usize,
        strata: &Strata,
        device: &Device,
        ticket: Option<WaveTicket>,
    ) -> Result<Self> {
        let stride = selection_stride(ratio, top_k, cand_max, strata)?;
        Ok(Self {
            entries: wave_empty_ticketed((rows, stride), DType::U32, device, ticket)?,
            cnt: wave_empty_ticketed((rows,), DType::U32, device, ticket)?,
            stride,
            gather: max_gathered_for(top_k, ratio, cand_max, strata).next_power_of_two(),
            split_keys: split_scratch(rows, ratio, top_k, cand_max, strata, device, ticket)?,
            ratio,
            strata: *strata,
        })
    }

    /// The selection, as the attention kernels take it.
    pub fn into_selection(self) -> Result<QsaSelection> {
        QsaSelection::new(self.entries, self.cnt, self.ratio)
    }

    /// Run the selection kernel over one tile of scores, writing rows
    /// `[row_base, row_base + rows)`.
    ///
    /// Public because the scores are the kernel's whole input: a test that
    /// hands it scores of its own choosing is what pins the device selection
    /// against [`super::qsa_select::selection_entries`] without a model in the
    /// way.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn fill_rows(
        &mut self,
        scores: &Tensor,
        cand: &[u32],
        qpos: &[usize],
        // Cells of its own block each query has, `1..=ratio` — the one thing
        // the selection kernel cannot derive from a position once blocks stop
        // being uniformly `ratio` wide. See `IndexCache::tail_len`.
        tail: &[u32],
        // Blocks wholly inside each query's system prompt — ranked in every
        // window of a stratified selection. See `IndexCache::prompt_blocks`.
        prompt: &[u32],
        ratio: usize,
        top_k: usize,
        row_base: usize,
    ) -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle_kernels::simple::qsa_topk::run_qsa_topk_entries;

        let (rows, score_stride) = scores.dims2()?;
        if rows == 0 {
            return Ok(());
        }
        let candle::Device::Cuda(dev) = scores.device() else {
            candle::bail!("qsa selection runs on CUDA");
        };
        let stream = dev.cuda_stream();
        if tail.len() != rows || prompt.len() != rows {
            candle::bail!(
                "qsa selection: {} tail lengths and {} prompt spans against {rows} rows",
                tail.len(),
                prompt.len()
            );
        }
        // **One upload, not four.** Every one of these is a host→device copy
        // per layer per wave, and on WDDM a small transfer costs far more in
        // submission than in bytes — three of them measured ~5% of the whole
        // forward-batched ladder. They are the same length and the same dtype,
        // so they travel as one `[4, rows]` block and the kernel takes four
        // offsets into it.
        let mut packed: Vec<u32> = Vec::with_capacity(rows * 4);
        packed.extend_from_slice(cand);
        packed.extend(qpos.iter().map(|&p| p as u32));
        packed.extend_from_slice(tail);
        packed.extend_from_slice(prompt);
        // Beside the scores, on the layer's span, where the rest of the
        // selection lives.
        let packed_t = scores.from_vec_beside(packed, (4, rows))?;

        let (s_s, s_l) = scores.storage_and_layout();
        let s_slice = match &*s_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("qsa selection: scores must be CUDA"),
        }
        .slice(s_l.start_offset()..);
        let (c_s, c_l) = packed_t.storage_and_layout();
        let c_slice = match &*c_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: row metadata must be CUDA"),
        }
        .slice(c_l.start_offset()..);
        let (e_s, e_l) = self.entries.storage_and_layout();
        let e_slice = match &*e_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: entries must be CUDA"),
        }
        .slice(e_l.start_offset()..);
        let (n_s, n_l) = self.cnt.storage_and_layout();
        let n_slice = match &*n_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: cnt must be CUDA"),
        }
        .slice(n_l.start_offset()..);
        let k_store = self.split_keys.as_ref().map(|t| t.storage_and_layout());
        let k_slice = match &k_store {
            Some((k_s, k_l)) => Some(
                match &**k_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("qsa selection: split scratch must be CUDA"),
                }
                .slice(k_l.start_offset()..),
            ),
            None => None,
        };

        let (s_ptr, _sg) = s_slice.device_ptr(&stream);
        let (c_ptr, _cg) = c_slice.device_ptr(&stream);
        // The four rows of the packed block, in the order they were written.
        let p_ptr = (c_ptr as *const u32).wrapping_add(rows);
        let t_ptr = (c_ptr as *const u32).wrapping_add(rows * 2);
        let pr_ptr = (c_ptr as *const u32).wrapping_add(rows * 3);
        let (e_ptr, _eg) = e_slice.device_ptr(&stream);
        let (n_ptr, _ng) = n_slice.device_ptr(&stream);
        // Null when the table holds no scratch: the kernel then runs the single
        // pass.
        let k_dev = k_slice.as_ref().map(|k| k.device_ptr(&stream));
        let k_ptr = k_dev
            .as_ref()
            .map_or(null_mut(), |(p, _)| *p as *mut c_void);
        let cand_max = cand.iter().copied().max().unwrap_or(0);
        // The table is one allocation; a tile writes its own row window, so
        // the offsets go on the pointers rather than through a narrowed view.
        let e_ptr = (e_ptr as *mut u32).wrapping_add(row_base * self.stride);
        let n_ptr = (n_ptr as *mut u32).wrapping_add(row_base);
        let s = &self.strata;
        candle::set_kernel_breadcrumb("run_qsa_topk_entries", file!(), line!());
        unsafe {
            run_qsa_topk_entries(
                s_ptr as *const f32,
                score_stride as i32,
                c_ptr as *const u32,
                p_ptr,
                t_ptr,
                pr_ptr,
                e_ptr,
                self.stride as i32,
                n_ptr,
                ratio as i32,
                top_k as i32,
                s.window_blocks as i32,
                s.recent_blocks as i32,
                i32::from(s.recent == Recent::Forced),
                // Checked against the kernel's ceiling where the stride was
                // sized.
                self.gather as i32,
                cand_max as i32,
                k_ptr,
                rows as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok(())
    }
}

impl SelectionTable {
    /// Read one row's selection back to the host: `None` for a dense row.
    ///
    /// The readback is a test and diagnostic path — the forward hands the
    /// table straight to the attention kernels.
    pub fn row_to_host(&self, row: usize) -> Result<Option<Vec<u32>>> {
        let cnt = self.cnt.narrow(0, row, 1)?.to_vec1::<u32>()?[0];
        if cnt == super::qsa_select::DENSE_ROW {
            return Ok(None);
        }
        let ent = self
            .entries
            .narrow(0, row, 1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        Ok(Some(ent[..cnt as usize].to_vec()))
    }
}

/// The wave's raw index keys: one GEMM over every row, not one per sequence.
///
/// `h` is the layer's `[rows, hidden]` block input — the same rows the
/// attention projections read, which is what the reference projects its index
/// keys from. The per-sequence caches then take their own row window
/// ([`IndexCache::append`]); splitting the projection instead would put one
/// small GEMM per sequence per layer on the decode path, where launches are
/// the wall.
pub fn project_keys(h: &Tensor, w: &IndexerWeights) -> Result<Tensor> {
    rows_matmul_t(h, &w.k_proj)
}

/// The wave's indexer queries — projected, normed, and rotated at each row's
/// own absolute position, in one pass over every row.
///
/// The reference's order (`qsa_selection_mask`): project, RMS-norm with
/// `q_norm`, then rotate — from the same factored table the scorer rotates the
/// keys from, so the two sides of every dot product share one set of
/// frequencies. Each row rotates at its sequence's rung (`rungs`), and takes
/// that rung's `m²` as the attention's queries do (§12).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn project_queries(
    h: &Tensor,
    w: &IndexerWeights,
    cfg: &IndexerConfig,
    rope: &FactoredRope,
    positions: &[usize],
    rungs: RowRungs<'_>,
    rms_eps: f64,
    // The open layer phase, for the rotation's position and rung tables.
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let rows = h.dim(0)?;
    let q = rows_matmul_t(h, &w.q_proj)?.reshape((rows * cfg.n_heads, cfg.head_dim))?;
    rotate_rows(
        RowSource::Normed {
            rows: &q,
            weight: &w.q_norm,
            eps: rms_eps,
        },
        rope,
        cfg.n_heads,
        RowPositions::PerGroup(positions),
        rungs,
        RotSide::Query,
        ticket,
    )?
    .reshape((rows, cfg.n_heads, cfg.head_dim))
}

/// `out[row_base + r, j] = Σ_h relu(raw[r · h + i, j])`, in one launch.
///
/// `raw` is the scoring matmul's `[rows · h, m]` output, whose row axis carries
/// (row, head) in that order — so as a `[rows, h, m]` view its strides are
/// `(h · m, m, 1)`, which is exactly what the kernel reads through. `out` is the
/// wave's `[total_rows, out_stride]` score buffer; columns past this span's `m`
/// are left as allocated, and the top-k never reads them because each row's scan
/// is bounded by its own candidate count.
#[allow(clippy::too_many_arguments)]
fn fold_heads_into(
    raw: &Tensor,
    rows: usize,
    h: usize,
    m: usize,
    out: &Tensor,
    out_stride: usize,
    row_base: usize,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::simple::indexer_score::run_indexer_score_reduce;

    let candle::Device::Cuda(dev) = raw.device() else {
        candle::bail!("qsa selection runs on CUDA");
    };
    if m > out_stride {
        candle::bail!("qsa selection: {m} candidates into a {out_stride}-wide score row");
    }
    let stream = dev.cuda_stream();
    let (r_s, r_l) = raw.storage_and_layout();
    let r_slice = match &*r_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qsa selection: scores must be CUDA"),
    }
    .slice(r_l.start_offset()..);
    let (o_s, o_l) = out.storage_and_layout();
    let o_slice = match &*o_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qsa selection: fold output must be CUDA"),
    }
    .slice(o_l.start_offset() + row_base * out_stride..);
    let (r_ptr, _rg) = r_slice.device_ptr(&stream);
    let (o_ptr, _og) = o_slice.device_ptr(&stream);
    let raw_stream = stream.cu_stream() as *mut c_void;
    candle::set_kernel_breadcrumb("run_indexer_score_reduce", file!(), line!());
    unsafe {
        run_indexer_score_reduce(
            r_ptr as *const f32,
            std::ptr::null(),
            std::ptr::null(),
            o_ptr as *mut f32,
            rows as i32,
            h as i32,
            m as i32,
            (h * m) as i64,
            m as i64,
            1,
            0,
            0,
            0,
            out_stride as i64,
            raw_stream,
        );
    }
    Ok(())
}

/// A tensor's device address.
pub(super) fn tensor_ptr(t: &Tensor) -> Result<u64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let candle::Device::Cuda(dev) = t.device() else {
        candle::bail!("qsa index cache lives on CUDA");
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qsa index cache must be CUDA f32"),
    }
    .slice(l.start_offset()..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(ptr)
}

/// One span of the wave: whose cache it appends to, and which rows of the
/// wave's key projection are its own.
pub struct AppendSpan<'a> {
    pub cache: &'a mut IndexCache,
    /// First row of `k_all` belonging to this span.
    pub start: usize,
    pub rows: usize,
}

/// Append every span's index keys — pool, RMS-norm, store un-rotated — in ONE
/// pair of launches for the whole wave.
///
/// This replaces a per-sequence loop that cost ~30 launches a span, per layer,
/// per wave: `nsys` over `tests/qsa_index_bench.rs` measured ~21,500 launches
/// for 20.5 ms of GPU time against ~208 ms of wall clock, the GPU busy a tenth
/// of the time. The arithmetic was never the cost, so the fix is not faster
/// arithmetic but issuing it once (hot-path invariant 5). The kernel reads each
/// span's rows in place through a descriptor table rather than requiring one
/// dense block, which is what removes the copies as well (invariant 2b).
///
/// The two launches are ordered: the append reads the open-block rows the
/// PREVIOUS wave carried, and the carry then overwrites them with this wave's
/// trailing rows. Same stream, append first.
pub fn append_wave(
    work: &mut [AppendSpan<'_>],
    k_all: &Tensor,
    w: &IndexerWeights,
    ratio: usize,
    rms_eps: f64,
) -> Result<()> {
    use candle_kernels::simple::qsa_index_append::{
        run_qsa_index_append, run_qsa_index_carry, CARRY_WORDS, JOB_WORDS, MAX_D,
    };

    if work.is_empty() || work.iter().all(|s| s.rows == 0) {
        return Ok(());
    }
    if ratio == 0 || ratio > MAX_RATIO {
        candle::bail!("qsa append: ratio {ratio} outside 1..={MAX_RATIO}");
    }
    let d = work[0].cache.keys.head_dim();
    if d == 0 || d > MAX_D {
        candle::bail!(
            "qsa append: head_dim {d} outside 1..={MAX_D} — the kernel gives one thread to a \
             channel, so the channel count is the block width"
        );
    }
    let device = k_all.device().clone();
    let k_base = tensor_ptr(k_all)?;
    let elem = std::mem::size_of::<f32>() as u64;
    let row = d as u64 * elem;

    let mut jobs: Vec<i64> = Vec::new();
    let mut carries: Vec<i64> = Vec::new();
    // (n_blocks, n_open) each span ends at, applied only once the launches that
    // depend on the entering values have been issued.
    let mut commits: Vec<(usize, usize)> = Vec::with_capacity(work.len());

    for span in work.iter_mut() {
        let (n_new, left) = span.cache.plan(span.rows, ratio);
        let n_blocks = span.cache.n_blocks;
        let n_open = span.cache.n_open;
        // Admission sized the pages for this wave, so this claims nothing; a
        // page it did have to claim inside the forward is refused by the arena
        // window rather than carved out of ground the wave stands on.
        span.cache.keys.ensure(n_blocks + n_new)?;
        let pages = span.cache.keys.page_ptrs()?;
        let raw = span.cache.raw_ptr()?;

        for i in 0..n_new {
            // Only the first block of a span can straddle the carried rows;
            // every later one is contiguous in the wave's projection.
            let n0 = if i == 0 { n_open.min(ratio) } else { 0 };
            let src1 = span.start + (i * ratio).saturating_sub(n_open);
            jobs.push(row_addr(&pages, n_blocks + i, d) as i64);
            jobs.push(if n0 > 0 { raw as i64 } else { 0 });
            jobs.push(n0 as i64);
            jobs.push((k_base + src1 as u64 * row) as i64);
        }

        if left > 0 {
            // No block completed, so the carried rows stand and this span's
            // rows extend them; otherwise the block consumed them and the
            // trailing rows start the next one.
            let (dst, src, rows) = if n_new == 0 {
                (
                    raw + n_open as u64 * row,
                    k_base + span.start as u64 * row,
                    span.rows,
                )
            } else {
                (
                    raw,
                    k_base + (span.start + span.rows - left) as u64 * row,
                    left,
                )
            };
            carries.push(dst as i64);
            carries.push(src as i64);
            carries.push(rows as i64);
        }
        commits.push((n_blocks + n_new, left));
    }

    if !jobs.is_empty() || !carries.is_empty() {
        let candle::Device::Cuda(dev) = &device else {
            candle::bail!("qsa append runs on CUDA");
        };
        let n_jobs = jobs.len() / JOB_WORDS;
        let n_carry = carries.len() / CARRY_WORDS;
        // Both tables in one upload through the device's staging scratch — the
        // jobs, then the carries — so an append allocates nothing, whether it
        // runs inside a forward or in a rewind between two.
        let carries_at = jobs.len();
        let mut table = jobs;
        table.extend_from_slice(&carries);
        let k_norm = tensor_ptr(&w.k_norm)?;
        // The stream is taken inside, where the launches are — the staged
        // upload runs eagerly, so they belong on the stream it names there.
        dev.with_staged_upload(&table, |base| {
            let stream = dev.cuda_stream();
            let raw_stream = stream.cu_stream() as *mut c_void;
            if n_jobs > 0 {
                candle::set_kernel_breadcrumb("run_qsa_index_append", file!(), line!());
                unsafe {
                    run_qsa_index_append(
                        base as *const i64,
                        k_norm as *const f32,
                        d as i32,
                        ratio as i32,
                        rms_eps as f32,
                        n_jobs as i32,
                        raw_stream,
                    );
                }
            }
            if n_carry > 0 {
                candle::set_kernel_breadcrumb("run_qsa_index_carry", file!(), line!());
                unsafe {
                    run_qsa_index_carry(
                        (base as *const i64).add(carries_at),
                        d as i32,
                        n_carry as i32,
                        raw_stream,
                    );
                }
            }
            Ok(())
        })?;
    }

    for (span, (n_blocks, n_open)) in work.iter_mut().zip(commits) {
        span.cache.n_blocks = n_blocks;
        span.cache.n_open = n_open;
    }
    Ok(())
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
    for rows in [PAGE_BLOCKS, MAX_RATIO] {
        let moves = plan_slot_moves(
            device,
            SlotTenant::QsaIndex,
            rows * head_dim * f32_bytes,
            max_moves,
        )?;
        planned += moves.len();
        let mut by_source: HashMap<u64, ArenaSlot> =
            moves.into_iter().map(|m| (m.src, m.dst)).collect();
        for cache in caches.iter_mut() {
            moved += cache.relocate(&mut by_source)?;
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

/// Device address of an i64 descriptor table.
/// A U32 descriptor table's device address — prefix sums and per-row counts.
pub(super) fn u32_ptr(t: &Tensor) -> Result<u64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let candle::Device::Cuda(dev) = t.device() else {
        candle::bail!("qsa descriptor table must be CUDA");
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa descriptor table must be CUDA u32"),
    }
    .slice(l.start_offset()..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(ptr)
}

pub(super) fn i64_ptr(t: &Tensor) -> Result<u64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let candle::Device::Cuda(dev) = t.device() else {
        candle::bail!("qsa append table must be CUDA");
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<i64>()?,
        _ => candle::bail!("qsa append table must be CUDA i64"),
    }
    .slice(l.start_offset()..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(ptr)
}

/// The most candidate blocks any row of the wave sees on KV layer `kv` — at
/// least one. What the score buffer and the selection table are sized from;
/// [`select_layer`] and the admission pricing (`select_bytes`) both ask it here
/// so they cannot size the same carve two ways.
pub fn widest_candidates(
    spans: &[SeqSpan],
    offsets: &[usize],
    idx_map: &HashMap<usize, Vec<IndexCache>>,
    kv: usize,
    ratio: usize,
) -> usize {
    spans
        .iter()
        .zip(offsets)
        .map(|(span, &off)| {
            let last = off + span.len;
            match idx_map.get(&span.seq).and_then(|c| c.get(kv)) {
                Some(cache) => cache.candidates_at(last.saturating_sub(1), ratio),
                None => last.div_ceil(ratio),
            }
        })
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Whether a layer at this depth selects at all.
///
/// Below the budget every visible cell is attended, so the indexer would
/// compute a selection that is the identity — the reference skips it and so
/// does the engine (§12.5). This is the check that keeps a short context on
/// exactly the arithmetic it had before QSA existed.
pub fn selection_engages(max_visible: usize, cfg: &IndexerConfig, ratio: usize) -> bool {
    ratio > 0 && max_visible > selected_width(cfg.top_k, ratio)
}

/// One full-attention layer's QSA work: append every sequence's index keys,
/// then build the wave's selection table if any row is past the budget.
///
/// `kv` is the layer's index among the full-attention layers, which is also its
/// index into a sequence's caches. Returns `None` when the layer attends
/// densely — either the checkpoint declares it dense (`compress_ratio == 0`) or
/// no row has more visible cells than the budget, where the selection is the
/// identity and computing it would be arithmetic with no effect (§12.5).
///
/// The keys are appended for EVERY wave regardless, because the cache is what a
/// later, deeper wave scores against.
///
/// A free function rather than a method on the wave engine so the selection
/// path can be driven — and benchmarked — without a loaded model: everything it
/// needs is the indexer's own weights, the caches, and two config values.
#[allow(clippy::too_many_arguments)]
pub fn select_layer(
    kv: usize,
    compress_ratio: usize,
    indexer: &IndexerWeights,
    rope: &FactoredRope,
    h: &Tensor,
    spans: &[SeqSpan],
    offsets: &[usize],
    idx_map: &mut HashMap<usize, Vec<IndexCache>>,
    capture: Option<&mut SpecCapture>,
    total_rows: usize,
    idx_cfg: &IndexerConfig,
    eps: f64,
    device: &Device,
    qsa_rows: &AtomicU64,
    // The open LAYER phase's ticket, for the page, window, job and rotation
    // tables built below. Every one of them is rebuilt for each attention layer
    // and dead by the end of it, so a per-layer span is their lifetime — the
    // forward-scoped span, sized for the few kilobytes a wave builds once,
    // filled up partway through the sweep when they were put there instead.
    // `None` falls back to an ordinary upload.
    ticket: Option<WaveTicket>,
) -> Result<Option<QsaSelection>> {
    if compress_ratio == 0 {
        return Ok(None);
    }

    // Row `spans[i]` covers positions `offsets[i] ..< offsets[i] + len`.
    let engages = spans
        .iter()
        .zip(offsets)
        .any(|(span, &off)| selection_engages(off + span.len, idx_cfg, compress_ratio));

    // Absolute position of every row, in the wave's packed order, and each
    // span's rung — picked from its reach exactly as the attention's header
    // writers pick it, so a sequence's index and attention rotate alike.
    let mut positions: Vec<usize> = Vec::with_capacity(total_rows);
    let mut span_rungs: Vec<u32> = Vec::with_capacity(spans.len());
    let mut row_rungs: Vec<u32> = Vec::with_capacity(total_rows);
    for (span, &off) in spans.iter().zip(offsets) {
        positions.extend(off..off + span.len);
        let rung = rope.rung_for(off + span.len)?;
        span_rungs.push(rung);
        row_rungs.extend(std::iter::repeat_n(rung, span.len));
    }

    // Both projections run ONCE over the whole wave; the per-sequence caches
    // take their own row windows. One GEMM per sequence per layer would be
    // launch-bound on the decode path, where a wave is 16 rows.
    let k_all = project_keys(h, indexer)?;
    // The widest row in the wave sets the score buffer's stride, so every span
    // writes into one buffer and the top-k covers all of it in a single launch;
    // it also sizes the selection table's rows, which a stratified selection
    // widens with depth. Asked of the caches, not derived from the offsets: a
    // sequence holding injected pages has MORE rows than `tokens / ratio` — a
    // page ends wherever its piece did, so its last row is short, and a prefix
    // of several pieces carries one short row per boundary. Sizing from the
    // uniform formula under-allocates by exactly that many columns. Read before
    // this wave's append, which it does not depend on: a cache's candidates at
    // a position follow from its page layout and where its tail opens.
    let widest = if engages {
        widest_candidates(spans, offsets, idx_map, kv, compress_ratio)
    } else {
        0
    };
    let mut table = if engages {
        qsa_rows.fetch_add(total_rows as u64, Ordering::Relaxed);
        Some(SelectionTable::new(
            total_rows,
            compress_ratio,
            idx_cfg.top_k,
            widest,
            &idx_cfg.strata,
            device,
            ticket,
        )?)
    } else {
        None
    };
    let q_all = match table {
        Some(_) => {
            // One rung for the whole wave — every sequence inside the same
            // ceiling, which is every wave short of a trained window — is a
            // launch argument; only a wave that straddles one uploads a row map.
            let rungs = match span_rungs.split_first() {
                Some((&r, rest)) if rest.iter().all(|&x| x == r) => RowRungs::Uniform(r),
                _ => RowRungs::PerGroup(&row_rungs),
            };
            Some(project_queries(
                h, indexer, idx_cfg, rope, &positions, rungs, eps, ticket,
            )?)
        }
        None => None,
    };
    // A verifying span keeps this layer's raw keys, so a partial accept can
    // restore the entering cache and re-append exactly the accepted rows
    // (`super::spec`). `k_all` is on the layer's arena, which the phase reset
    // reclaims, and the rewind reads the keys after the wave — so they are
    // copied into the cohort's kept-row buffer for this layer, at the
    // sequence's stash row.
    if let Some(c) = capture {
        let SpecCapture { seqs, rows, .. } = c;
        let keys_buf = rows.qsa.get(kv).ok_or_else(|| {
            candle::Error::Msg(format!(
                "qsa capture: KV layer {kv} past the {} the capture was sized for",
                rows.qsa.len()
            ))
        })?;
        for span in spans {
            if let Some(s) = seqs.get_mut(&span.seq) {
                if s.qsa_keys.len() <= kv {
                    s.qsa_keys.resize_with(kv + 1, || None);
                }
                s.qsa_keys[kv] = Some(CaptureRows::keep(
                    keys_buf,
                    s.row,
                    &k_all.narrow(0, span.start, span.len)?,
                )?);
            }
        }
    }

    // Every span's append, in one pair of launches. The borrows are disjoint
    // because a wave carries at most one span per sequence, which is checked
    // rather than assumed — a duplicate would append a sequence's rows once and
    // silently drop the rest.
    let mut work: Vec<AppendSpan<'_>> = Vec::with_capacity(spans.len());
    for (seq, caches) in idx_map.iter_mut() {
        let mut mine = spans.iter().filter(|s| s.seq == *seq);
        let Some(span) = mine.next() else { continue };
        if mine.next().is_some() {
            candle::bail!("qsa append: sequence {seq} appears in more than one span of a wave");
        }
        let cache = caches
            .get_mut(kv)
            .ok_or_else(|| candle::Error::Msg(format!("no index cache for kv layer {kv}")))?;
        work.push(AppendSpan {
            cache,
            start: span.start,
            rows: span.len,
        });
    }
    if work.len() != spans.len() {
        candle::bail!(
            "qsa append: {} spans against {} sequences with caches — a span's sequence has none",
            spans.len(),
            work.len()
        );
    }
    append_wave(&mut work, &k_all, indexer, compress_ratio, eps)?;

    if let (Some(table), Some(q_all)) = (table.as_mut(), q_all.as_ref()) {
        // A row's own candidate count still bounds its scan, so the columns a
        // narrower span leaves untouched are never read — which is why the
        // buffer is allocated uninitialised (hot-path invariant 6).
        let scores = wave_empty_ticketed((total_rows, widest), DType::F32, device, ticket)?;
        let mut cand: Vec<u32> = vec![0; total_rows];
        let mut tail: Vec<u32> = vec![1; total_rows];
        let mut prompt: Vec<u32> = vec![0; total_rows];
        for (span, &rung) in spans.iter().zip(&span_rungs) {
            let cache = idx_map
                .get(&span.seq)
                .and_then(|c| c.get(kv))
                .ok_or_else(|| {
                    candle::Error::Msg(format!("seq {} has no index cache", span.seq))
                })?;
            // Name the cache, not just the shortfall. A sequence holds one cache
            // per KV layer and they are not interchangeable: the trunk's are
            // appended by every wave, while the MTP draft head's — the slot at
            // `n_attention_layers()` — is appended only by the draft walks it
            // runs. A shortfall reported without its layer reads as one bug in
            // the index when it is two different populations of the same array.
            let span_cand = cache
                .score_rows(
                    &q_all.narrow(0, span.start, span.len)?,
                    &positions[span.start..span.start + span.len],
                    idx_cfg,
                    compress_ratio,
                    rope,
                    rung,
                    &scores,
                    widest,
                    span.start,
                    ticket,
                )
                .map_err(|e| candle::Error::Msg(format!("kv layer {kv}, seq {}: {e}", span.seq)))?;
            cand[span.start..span.start + span.len].copy_from_slice(&span_cand);
            for (r, &p) in positions[span.start..span.start + span.len]
                .iter()
                .enumerate()
            {
                tail[span.start + r] = cache.tail_len(p, compress_ratio);
            }
            prompt[span.start..span.start + span.len]
                .fill(cache.prompt_blocks(compress_ratio) as u32);
        }
        expect_dense(&scores, "qsa selection scores")?;
        table.fill_rows(
            &scores,
            &cand,
            &positions,
            &tail,
            &prompt,
            compress_ratio,
            idx_cfg.top_k,
            0,
        )?;
    }
    // The page layout the attention kernels walk, built only when some sequence
    // in this wave holds an injected prefix. Every other wave passes null and
    // the kernels keep their inline `pos / ratio`.
    let sel = table.map(SelectionTable::into_selection).transpose()?;
    let Some(sel) = sel else { return Ok(None) };
    let needs_pages = spans.iter().any(|s| {
        idx_map
            .get(&s.seq)
            .and_then(|c| c.get(kv))
            .is_some_and(|c| c.has_pages())
    });
    if !needs_pages {
        return Ok(Some(sel));
    }
    let mut pages: Vec<u32> = Vec::new();
    let mut win: Vec<u32> = vec![0; total_rows * 2];
    for span in spans {
        let off = pages.len() / 2;
        let prefixes = match idx_map.get(&span.seq).and_then(|c| c.get(kv)) {
            Some(cache) => cache.page_prefixes(),
            // A sequence with no cache contributes the degenerate layout, which
            // is the uniform arithmetic.
            None => vec![0, 0],
        };
        let count = prefixes.len() / 2;
        pages.extend(prefixes);
        for r in span.start..span.start + span.len {
            win[r * 2] = off as u32;
            win[r * 2 + 1] = count as u32;
        }
    }
    let n_pages = pages.len() / 2;
    let pages_t = wave_from_vec_ticketed(pages, (n_pages, 2), device, ticket)?;
    let win_t = wave_from_vec_ticketed(win, (total_rows, 2), device, ticket)?;
    Ok(Some(sel.with_pages(pages_t, win_t)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::attention::RopeTables;
    use crate::models::qwen4exp::qsa::{qsa_selection_mask, IndexState};
    use crate::models::qwen4exp::qsa_select::{
        entry_block, entry_cells, selection_entries, RowSelection,
    };
    use crate::models::qwen4exp::select_bytes::{select_layer_bytes, CARVE_ALIGN};
    use crate::models::rope_schedule::plain_inv_freq;
    use candle_nn::kv_cache::{begin_wave, LayerPhase};

    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
    }

    /// **A compaction pass moves buffers, never their contents.** Three caches
    /// across several key pages, the middle one dropped to leave holes, plus a
    /// cache holding a resident page shared with a fork of it, then a pass: every
    /// surviving cache reads back exactly the keys, open rows and page rows it
    /// held.
    #[test]
    fn compaction_moves_index_buffers_without_changing_them() -> Result<()> {
        let Some(device) = cuda() else {
            return Ok(());
        };
        let d = 128usize;
        let build = |n: usize, seed: u64| -> Result<(IndexCache, Vec<f32>, Vec<f32>)> {
            let keys = lcg(n * d, seed, 1.0);
            let open = lcg(3 * d, seed ^ 0x55, 1.0);
            let cache = IndexCache::from_rows(&keys, &open, d, &device)?;
            Ok((cache, keys, open))
        };
        let (mut a, a_keys, a_open) = build(2 * PAGE_BLOCKS + 7, 1)?;
        let hole = build(3 * PAGE_BLOCKS, 2)?;
        let (mut c, c_keys, c_open) = build(PAGE_BLOCKS + 1, 3)?;
        let page_rows = lcg((PAGE_BLOCKS + 9) * d, 4, 1.0);
        let mut p = IndexCache::new(d, &device)?;
        p.push_page(placed_page(&page_rows, 2, d, &device)?, 0, 4)?;
        let mut p_fork = p.fork()?;
        drop(hole);

        let report =
            compact_index_caches(&mut [&mut a, &mut c, &mut p, &mut p_fork], d, &device, 0)?;
        assert!(report.moved <= report.planned);
        for (cache, keys, open) in [(&a, &a_keys, &a_open), (&c, &c_keys, &c_open)] {
            assert_eq!(&cache.live_rows_host()?, keys);
            assert_eq!(&cache.open_rows_host()?, open);
        }
        for cache in [&p, &p_fork] {
            let (page, _) = cache.page_at(0).expect("the page survives the pass");
            assert_eq!(page.host_rows()?, page_rows);
        }
        Ok(())
    }

    /// **A cut keeps the pages below it whole and drops the rest.** Three pages
    /// of 8 rows at ratio 4, the last row covering 2 tokens — 30 tokens each,
    /// abutting at 0, 30 and 60. A cut at 60 keeps two and hands back the third,
    /// and the tail resumes at the cut; a cut inside a page is refused.
    #[test]
    fn a_cut_keeps_whole_pages_below_it() -> Result<()> {
        let Some(device) = cuda() else {
            return Ok(());
        };
        let d = 128usize;
        let mut c = IndexCache::new(d, &device)?;
        for seed in 1..=3 {
            let page = placed_page(&lcg(8 * d, seed, 1.0), 2, d, &device)?;
            let base = c.next_base();
            c.push_page(page, base, 4)?;
        }
        assert_eq!(c.next_base(), 90);

        assert_eq!(c.cut_floor(0), 0);
        assert_eq!(c.cut_floor(3), 0, "3 is inside the page at 0..30");
        assert_eq!(c.cut_floor(30), 30);
        assert_eq!(c.cut_floor(45), 30, "45 is inside the page at 30..60");
        assert_eq!(c.cut_floor(90), 90);
        assert_eq!(c.cut_floor(95), 95, "past the pages, in the tail");

        let dropped = c.truncate_to(60)?;
        assert_eq!(dropped.len(), 1);
        assert_eq!(c.page_count(), 2);
        assert_eq!(c.page_row_span(), 16);
        assert_eq!(c.next_base(), 60);

        assert!(
            c.truncate_to(45).is_err(),
            "45 is inside the page at 30..60"
        );
        assert_eq!(c.page_count(), 2, "a refused cut changes nothing");

        let dropped = c.truncate_to(0)?;
        assert_eq!(dropped.len(), 2);
        assert_eq!(c.page_count(), 0);
        assert_eq!(c.next_base(), 0);
        Ok(())
    }

    /// A page of `host` rows (`[rows, d]` row-major) whose last row covers
    /// `last` tokens, placed on `device` the way an injected record is.
    fn placed_page(
        host: &[f32],
        last: usize,
        d: usize,
        device: &Device,
    ) -> Result<Arc<ResidentPage>> {
        let mut pages = ResidentPage::place_host(&[(host, last)], d, device)?;
        pages
            .pop()
            .ok_or_else(|| candle::Error::Msg("one page placed".into()))
    }

    /// A page of `rows` zero rows whose last covers `last` tokens, over
    /// slot-shaped chunks on `device` — the shape a closed tail hands over.
    fn zero_page(rows: usize, last: usize, d: usize, device: &Device) -> Arc<ResidentPage> {
        let chunks = (0..rows.div_ceil(PAGE_BLOCKS))
            .map(|_| Tensor::zeros((PAGE_BLOCKS, d), DType::F32, device).expect("chunk"))
            .collect();
        ResidentPage::from_key_pages(chunks, rows, last, d).expect("page")
    }

    /// An index cache holding `pages` (by token width) and `open` carried rows,
    /// built on the CPU.
    ///
    /// The page bookkeeping — prefix sums, the ragged walk, the token count — is
    /// pure arithmetic over widths, so it needs no device. Only the flush
    /// kernel does, which is why these tests can pin the contract the page cuts
    /// depend on without a GPU in the loop.
    fn caged(widths: &[usize], ratio: usize) -> IndexCache {
        let d = 4usize;
        let mut c = IndexCache::new(d, &Device::Cpu).expect("cpu cache");
        for &w in widths {
            // A page of `w` tokens is `ceil(w / ratio)` rows whose last one
            // covers the remainder — the shape a real seal produces.
            let rows = w.div_ceil(ratio);
            let last = w - (rows - 1) * ratio;
            let base = c.next_base();
            c.push_page(zero_page(rows, last, d, &Device::Cpu), base, ratio)
                .expect("push");
        }
        c
    }

    /// **An unindexed span moves nothing but where the next page opens.**
    ///
    /// This is the property the whole placement design buys. Positions used to
    /// be the running sum of the page widths, so a piece injected without its
    /// rows did not merely leave a span unindexed — it dragged every later page
    /// earlier by its own width, and every query past it resolved through a
    /// block describing different tokens. Measured live as 328 tokens of section
    /// K/V against zero indexed tokens on all thirteen layers.
    #[test]
    fn an_unindexed_span_leaves_later_pages_where_their_kv_is() {
        const RATIO: usize = 4;
        // 40 tokens of K/V with no rows, then a real 20-token page.
        let mut c = caged(&[], RATIO);
        c.skip_to(40).expect("skip");
        let rows = 20usize.div_ceil(RATIO);
        let page = zero_page(rows, 20 - (rows - 1) * RATIO, 4, &Device::Cpu);
        let base = c.next_base();
        c.push_page(page, base, RATIO).expect("page after the span");

        assert_eq!(
            c.indexed_tokens(RATIO),
            60,
            "the skipped 40 tokens are accounted for even though nothing indexed \
             them — without that the cache claims 20 while the K/V holds 60"
        );
        assert_eq!(
            c.block_start(0, RATIO),
            40,
            "the first REAL row begins where its K/V does, past the span"
        );
        // The span itself offers nothing: a position inside it resolves to the
        // row count before it, which is zero.
        assert_eq!(
            c.candidates_at(20, RATIO),
            0,
            "a query inside the span has no candidates — nothing indexed it"
        );
        // And one inside the real page resolves through THAT page.
        assert_eq!(
            c.candidates_at(43, RATIO),
            1,
            "a query four tokens into the page sees the page's first row"
        );
    }

    /// Moving the tail adds no rows and no candidates — only distance.
    #[test]
    fn skipping_adds_tokens_but_no_rows() {
        const RATIO: usize = 4;
        let mut c = caged(&[12], RATIO);
        let rows_before = c.page_row_span();
        let tokens_before = c.page_token_span();
        c.skip_to(tokens_before + 17).expect("skip");
        assert_eq!(
            c.page_row_span(),
            rows_before,
            "an unindexed span contributes no candidate rows"
        );
        assert_eq!(
            c.page_token_span(),
            tokens_before + 17,
            "but it does move the position the next page opens at"
        );
        // Standing still is a no-op, so a caller may declare unconditionally.
        c.skip_to(tokens_before + 17).expect("no-op skip");
        assert_eq!(c.page_token_span(), tokens_before + 17);
    }

    /// Moving the tail obeys the same ordering rule as a page: injected prefix
    /// state must precede anything the sequence forwarded, or the live tail's
    /// own rows would sit before content that came earlier.
    #[test]
    fn a_skip_is_refused_after_live_rows() {
        const RATIO: usize = 4;
        let mut c = caged(&[8], RATIO);
        c.n_blocks = 1;
        assert!(
            c.skip_to(c.next_base() + 4).is_err(),
            "moving the tail after live rows must be refused, exactly as a page is"
        );
    }

    /// **Pages ascend and may not overlap.** A hole is a span nobody indexed; an
    /// overlap is two rows claiming one token, which no placement can mean.
    #[test]
    fn a_page_may_not_overlap_the_one_before_it() {
        const RATIO: usize = 4;
        let mut c = caged(&[12], RATIO);
        let page = zero_page(2, RATIO, 4, &Device::Cpu);
        assert!(
            c.push_page(page, 4, RATIO).is_err(),
            "a page placed inside the previous page's span was accepted"
        );
    }

    /// **The ragged-page contract, which every thinking-span cut relies on.**
    ///
    /// A page records the tokens its last row covers, and the scorer walks those
    /// widths instead of dividing by `ratio`. These pin that a position resolves
    /// to the same block whether the pages are uniform, ragged, or a mix — the
    /// property that makes dropping a whole page renumber everything downstream
    /// by construction.
    #[test]
    fn ragged_pages_resolve_positions_by_walking_widths() {
        let ratio = 4;
        // Uniform pages: the walk must agree with the plain division.
        let c = caged(&[8, 8], ratio);
        assert_eq!(c.indexed_tokens(ratio), 16);
        for pos in 0..16 {
            assert_eq!(
                c.candidates_at(pos, ratio),
                (pos + 1) / ratio,
                "uniform pages must match the uniform formula at {pos}"
            );
        }

        // Ragged: 5 tokens (rows [4,1]) then 6 (rows [4,2]).
        let c = caged(&[5, 6], ratio);
        assert_eq!(c.indexed_tokens(ratio), 11);
        // Inside page 0: its short last row only counts once the position
        // reaches the page's full span.
        assert_eq!(c.candidates_at(0, ratio), 0);
        assert_eq!(c.candidates_at(3, ratio), 1);
        assert_eq!(
            c.candidates_at(4, ratio),
            2,
            "page 0 complete at its 5th token"
        );
        // Inside page 1, which starts at token 5.
        assert_eq!(c.candidates_at(8, ratio), 3);
        assert_eq!(c.candidates_at(10, ratio), 4, "page 1 complete at token 11");
    }

    /// A cut leaves a SHORT page, and the pages either side of it must still
    /// resolve exactly — this is the shape a thinking turn seals
    /// (`[prefill][reasoning][answer]`, none of them a multiple of `ratio`).
    #[test]
    fn a_short_page_between_two_others_keeps_the_walk_exact() {
        let ratio = 4;
        let c = caged(&[7, 3, 9], ratio);
        assert_eq!(
            c.indexed_tokens(ratio),
            19,
            "the cache spans its pages exactly"
        );

        // Every position must map to a block whose start is at or below it, and
        // the count must be monotone — the two properties the scorer needs.
        let mut prev = 0;
        for pos in 0..19 {
            let n = c.candidates_at(pos, ratio);
            assert!(n >= prev, "candidate count went backwards at {pos}");
            prev = n;
            if n > 0 {
                assert!(
                    c.block_start(n - 1, ratio) <= pos,
                    "block {} starts after the position that counts it",
                    n - 1
                );
            }
        }
        // A page boundary is exactly where the previous page's rows all count.
        assert_eq!(c.candidates_at(6, ratio), 2, "page 0 = rows [4,3]");
        assert_eq!(c.candidates_at(9, ratio), 3, "page 1 = one 3-token row");
    }

    /// **Dropping the reasoning page renumbers by construction.** The whole
    /// design rests on this: a cache built from `[pre][answer]` must resolve
    /// positions exactly as one built from `[pre][reasoning][answer]` does for
    /// the tokens that survive — because the projection compacts the K/V the
    /// same way.
    #[test]
    fn dropping_a_page_compacts_the_walk() {
        let ratio = 4;
        let with = caged(&[7, 3, 9], ratio);
        let without = caged(&[7, 9], ratio);

        assert_eq!(with.indexed_tokens(ratio), 19);
        assert_eq!(
            without.indexed_tokens(ratio),
            16,
            "the 3-token page is gone"
        );

        // The retained pages keep their own row counts, so every position in the
        // compacted cache sees exactly the rows of `[pre] ++ [answer]`.
        let pre_rows = with.candidates_at(6, ratio);
        assert_eq!(without.candidates_at(6, ratio), pre_rows);
        // The answer's last token: all rows of both retained pages.
        assert_eq!(without.candidates_at(15, ratio), without.page_row_span());
    }

    /// An empty cache, a single page, and a page narrower than `ratio` — the
    /// degenerate shapes a suppressed turn or a one-token answer produces.
    #[test]
    fn degenerate_page_shapes_are_expressible() {
        let ratio = 4;
        let empty = caged(&[], ratio);
        assert_eq!(empty.indexed_tokens(ratio), 0);
        assert_eq!(empty.candidates_at(0, ratio), 0);

        // A single 1-token page — a collapsed `<think></think>`.
        let one = caged(&[1], ratio);
        assert_eq!(one.indexed_tokens(ratio), 1);
        assert_eq!(one.candidates_at(0, ratio), 1, "its only row is complete");

        // Exactly one full block.
        let full = caged(&[4], ratio);
        assert_eq!(full.indexed_tokens(ratio), 4);
        assert_eq!(full.candidates_at(2, ratio), 0);
        assert_eq!(full.candidates_at(3, ratio), 1);
    }

    /// The push guard, from the CUT's side: `close_tail_into_page` resets the
    /// tail before pushing precisely because a page may not land after live
    /// rows. This pins both halves — refused with a tail, accepted without —
    /// so the reset in the cut cannot be dropped as redundant.
    #[test]
    fn a_page_is_refused_after_live_rows_and_accepted_once_the_tail_is_reset() {
        let ratio = 4;
        let d = 4usize;
        let rows = vec![0f32; 2 * d];

        let mut forwarded = IndexCache::from_rows(&rows, &[], d, &Device::Cpu).unwrap();
        assert_eq!(forwarded.live_blocks(), 2);
        let base = forwarded.next_base();
        assert!(
            forwarded
                .push_page(zero_page(1, 1, d, &Device::Cpu), base, ratio)
                .is_err(),
            "a page landing after live rows would sit at the wrong positions"
        );

        // A cache whose tail is empty — what the cut leaves behind — accepts it.
        let mut c = caged(&[8], ratio);
        let base = c.next_base();
        assert!(c
            .push_page(zero_page(1, 1, d, &Device::Cpu), base, ratio)
            .is_ok());
        assert_eq!(c.indexed_tokens(ratio), 9);
    }

    /// A snapshot keeps a COPY of the open rows: the wave after it writes the open
    /// block in place, and restoring — twice, as a partial accept can — puts back
    /// exactly the rows and counters it took.
    #[test]
    fn a_snapshot_restores_the_open_rows_it_copied() {
        let d = 4usize;
        let rows = vec![0f32; 2 * d];
        let open_v: Vec<f32> = (0..2 * d).map(|i| i as f32 + 0.5).collect();
        let mut cache = IndexCache::from_rows(&rows, &open_v, d, &Device::Cpu).unwrap();
        let snap = cache.snapshot().unwrap();

        let dirty = |c: &mut IndexCache| {
            c.raw
                .slice_set(
                    &Tensor::full(-1f32, (MAX_RATIO, d), &Device::Cpu).unwrap(),
                    0,
                    0,
                )
                .unwrap();
            c.n_blocks = 5;
            c.n_open = 3;
        };
        for _ in 0..2 {
            dirty(&mut cache);
            cache.restore(&snap).unwrap();
            assert_eq!(cache.seal_shape(), (2, 2));
            assert_eq!(cache.open_rows_host().unwrap(), open_v);
        }
    }

    /// Snapshots draw on the cache's own rewind buffers, which a dropped snapshot
    /// gives back; more outstanding at once than there are buffers is refused by
    /// name. A snapshot of an empty open block needs no buffer at all.
    #[test]
    fn snapshots_draw_on_a_fixed_set_of_rewind_buffers() {
        let d = 4usize;
        let rows = vec![0f32; d];
        let bare = IndexCache::from_rows(&rows, &[], d, &Device::Cpu).unwrap();
        let free: Vec<_> = (0..2 * SNAPSHOT_BUFFERS)
            .map(|_| bare.snapshot().unwrap())
            .collect();
        assert!(free.iter().all(|s| s.raw.is_none()));

        let open = vec![1f32; d];
        let cache = IndexCache::from_rows(&rows, &open, d, &Device::Cpu).unwrap();
        let mut held: Vec<_> = (0..SNAPSHOT_BUFFERS)
            .map(|_| cache.snapshot().unwrap())
            .collect();
        let err = cache.snapshot().unwrap_err().to_string();
        assert!(err.contains("outstanding"), "{err}");
        held.pop();
        assert!(
            cache.snapshot().is_ok(),
            "a dropped snapshot gives its buffer back"
        );
    }

    fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * scale
            })
            .collect()
    }

    /// The prompt end is a declaration about the slot, so it travels with the
    /// cache: a fork carries it, a truncate below it pulls it back to the cut,
    /// a truncate above it leaves it, and a reset forgets it. The blocks it
    /// covers are the whole indexed blocks below it, through the layout.
    #[test]
    fn the_prompt_end_travels_with_the_cache() -> Result<()> {
        let d = 2;
        // Four complete blocks and one open row at ratio 4: 17 positions.
        let mut c = IndexCache::from_rows(&[0.0; 8], &[0.0; 2], d, &Device::Cpu)?;
        assert_eq!((c.prompt_end(), c.prompt_blocks(4)), (0, 0));
        c.set_prompt_end(10);
        assert_eq!(c.prompt_blocks(4), 2);
        c.set_prompt_end(12);
        assert_eq!(c.prompt_blocks(4), 3);

        let mut fork = c.fork()?;
        assert_eq!(fork.prompt_end(), 12);
        fork.truncate_to(16)?;
        assert_eq!(fork.prompt_end(), 12, "a cut above the prompt keeps it");
        // A cut restarts the live tail at itself, so with no pages below it no
        // block is indexed under the clamped end until the rows are re-appended.
        c.truncate_to(8)?;
        assert_eq!((c.prompt_end(), c.prompt_blocks(4)), (8, 0));
        fork.reset();
        assert_eq!((fork.prompt_end(), fork.prompt_blocks(4)), (0, 0));
        Ok(())
    }

    /// The three ceilings the selection kernel imposes, each refused by name,
    /// and the stride a run that fits is given.
    #[test]
    fn selection_stride_refuses_what_the_kernel_cannot_run() {
        let err = |r: Result<usize>| r.unwrap_err().to_string();
        assert!(err(selection_stride(0, 2048, 1000, &Strata::WHOLE)).contains("ratio 0"));
        assert!(
            err(selection_stride(MAX_RATIO + 1, 2048, 1000, &Strata::WHOLE)).contains("outside")
        );
        // 4096 positions at ratio 4 keep 1025 blocks, past the 768 buffer.
        assert!(err(selection_stride(4, 4096, 100_000, &Strata::WHOLE)).contains("survivors"));
        // One-block windows over 20,000 candidates gather 20,000 · 513.
        let tiny = Strata {
            window_blocks: 1,
            recent_blocks: 0,
            recent: Recent::Candidate,
        };
        assert!(err(selection_stride(4, 2048, 20_000, &tiny)).contains("gather"));
        assert_eq!(
            selection_stride(4, 2048, 73_728, &Strata::WHOLE).unwrap(),
            514
        );
    }

    /// What one kernel-against-definition case runs: the queries, the geometry,
    /// the scores' grid, and the strata with its prompt span.
    struct KernelCase {
        qpos: Vec<usize>,
        top_k: usize,
        // `Some(step)` collapses the scores onto a grid of `step`.
        quantize: Option<f32>,
        seed: u64,
        strata: Strata,
        // Row r's prompt spans `prompt_blocks + r · prompt_spread` blocks.
        prompt_blocks: u32,
        prompt_spread: u32,
    }

    /// The selection kernel against the shared definition, row by row, on
    /// scores chosen by the test — no model, no GEMM, so any disagreement is
    /// the kernel's.
    ///
    /// `quantize` collapses the scores onto a coarse grid, which is how the
    /// tie-breaking rule gets exercised: at 8 distinct values over a few
    /// hundred blocks the cut lands inside a run of equal scores on almost
    /// every row, and the reference resolves those by ascending block.
    fn kernel_matches(case: &KernelCase) -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let ratio = 4usize;
        let rows = case.qpos.len();
        let cand: Vec<u32> = case
            .qpos
            .iter()
            .map(|&p| ((p + 1) / ratio) as u32)
            .collect();
        let blocks = *cand.iter().max().unwrap() as usize;

        let mut host = lcg(rows * blocks, case.seed, 4.0);
        if let Some(step) = case.quantize {
            for v in host.iter_mut() {
                *v = (*v / step).floor() * step;
            }
        }
        let scores = Tensor::from_vec(host.clone(), (rows, blocks), &device)?;
        let mut table =
            SelectionTable::new(rows, ratio, case.top_k, blocks, &case.strata, &device, None)?;
        // Uniform blocks here, so the tail is what the kernel used to derive.
        let tail: Vec<u32> = case
            .qpos
            .iter()
            .zip(&cand)
            .map(|(&p, &c)| (p + 1 - c as usize * ratio) as u32)
            .collect();
        let prompt: Vec<u32> = (0..rows as u32)
            .map(|r| case.prompt_blocks + r * case.prompt_spread)
            .collect();
        table.fill_rows(
            &scores, &cand, &case.qpos, &tail, &prompt, ratio, case.top_k, 0,
        )?;

        let mut want = Vec::new();
        for (r, &p) in case.qpos.iter().enumerate() {
            let row_scores = &host[r * blocks..r * blocks + blocks];
            let sel = selection_entries(
                row_scores,
                p,
                ratio,
                case.top_k,
                &case.strata,
                prompt[r] as usize,
                &mut want,
            );
            let got = table.row_to_host(r)?;
            match sel {
                RowSelection::Dense => assert!(got.is_none(), "row {r} should be dense"),
                RowSelection::Entries(_) => {
                    let got = got.unwrap_or_else(|| panic!("row {r} came back dense"));
                    assert_eq!(got, want, "row {r} (qpos {p}) selection differs");
                }
            }
        }
        Ok(())
    }

    /// Positions chosen so every row is past the budget and the phases of
    /// `qpos mod ratio` are all represented (the partial-block cut moves with
    /// the phase).
    fn shallow_rows() -> Vec<usize> {
        (0..16).map(|i| 40 + i * 7).collect()
    }

    /// Rows deep enough to cut into several small windows: 188 candidate
    /// blocks at the deepest.
    fn windowed_rows() -> Vec<usize> {
        (0..16).map(|i| 200 + i * 37).collect()
    }

    fn strata(window_blocks: usize, recent_blocks: usize, recent: Recent) -> Strata {
        Strata {
            window_blocks,
            recent_blocks,
            recent,
        }
    }

    #[test]
    fn selection_kernel_matches_the_definition() -> Result<()> {
        kernel_matches(&KernelCase {
            qpos: shallow_rows(),
            top_k: 8,
            quantize: None,
            seed: 0x51,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
        })
    }

    #[test]
    fn selection_kernel_breaks_ties_by_ascending_block() -> Result<()> {
        // A coarse grid forces long runs of equal scores through the cut: at 8
        // distinct values the cut lands inside a run on almost every row, and
        // the reference resolves those by ascending block.
        kernel_matches(&KernelCase {
            qpos: shallow_rows(),
            top_k: 8,
            quantize: Some(0.5),
            seed: 0x52,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
        })
    }

    #[test]
    fn selection_kernel_survives_more_blocks_than_the_buffer_holds() -> Result<()> {
        // Past one streaming trim: ~4000 candidate blocks against a 1024-slot
        // buffer, so the threshold path runs many times.
        kernel_matches(&KernelCase {
            qpos: (0..4).map(|i| 16000 + i * 3).collect(),
            top_k: 64,
            quantize: None,
            seed: 0x53,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
        })
    }

    /// Several windows, each spending the budget, with the prompt ranked in
    /// every one — and on a tie grid, so a prompt block two windows both choose
    /// arrives twice and must leave once, at its widest cut.
    #[test]
    fn stratified_selection_kernel_matches_the_definition() -> Result<()> {
        for (quantize, seed) in [(None, 0x61), (Some(0.5), 0x62)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 0, Recent::Candidate),
                prompt_blocks: 5,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    #[test]
    fn stratified_selection_kernel_ranks_a_recent_candidate_in_every_window() -> Result<()> {
        for (quantize, seed) in [(None, 0x63), (Some(0.5), 0x64)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 12, Recent::Candidate),
                prompt_blocks: 5,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    #[test]
    fn stratified_selection_kernel_attends_a_forced_span_whole() -> Result<()> {
        for (quantize, seed) in [(None, 0x65), (Some(0.5), 0x66)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 12, Recent::Forced),
                prompt_blocks: 5,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    /// Windows wider than the buffer, so every window streams through several
    /// trims of its own, with a forced span past one chunk's width.
    #[test]
    fn stratified_selection_kernel_survives_windows_wider_than_the_buffer() -> Result<()> {
        for recent in [Recent::Candidate, Recent::Forced] {
            kernel_matches(&KernelCase {
                qpos: (0..4).map(|i| 16000 + i * 3).collect(),
                top_k: 64,
                quantize: None,
                seed: 0x67,
                strata: strata(1024, 300, recent),
                prompt_blocks: 100,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    /// Which pass a launch runs: a launch of a few rows splits every row across
    /// blocks, and one of `SPLIT_BLOCKS` rows or more never splits — so the
    /// wide cases below are the single pass's, on any device.
    fn split_parts(rows: usize, cand_max: usize, strata: &Strata, top_k: usize) -> i32 {
        unsafe {
            qsa_topk_split_parts(
                rows as i32,
                cand_max as i32,
                strata.window_blocks as i32,
                top_k as i32,
                4,
            )
        }
    }

    #[test]
    fn a_narrow_launch_splits_and_a_wide_one_does_not() -> Result<()> {
        let Some(_device) = cuda() else { return Ok(()) };
        assert!(split_parts(4, 188, &strata(32, 12, Recent::Forced), 8) > 0);
        assert!(split_parts(4, 4000, &Strata::WHOLE, 64) > 1);
        assert_eq!(split_parts(512, 188, &strata(32, 12, Recent::Forced), 8), 0);
        assert_eq!(split_parts(512, 4000, &Strata::WHOLE, 64), 0);
        Ok(())
    }

    /// The prompt overlapping everything a pool is cut from. Row r's prompt is
    /// `3r` blocks against 15–30 candidates, so across the rows it ends inside
    /// a window, inside the recent span (the R∩W∩P term), at or past the forced
    /// span (R2's clip), and past the row itself (the clamp to the row).
    #[test]
    fn a_prompt_overlapping_the_recent_and_forced_spans_matches_the_definition() -> Result<()> {
        let narrow: Vec<usize> = (0..16).map(|i| 60 + i * 4).collect();
        // The same rows repeated past `SPLIT_BLOCKS`, so the single pass runs
        // them too.
        let wide: Vec<usize> = (0..512).map(|i| 60 + (i % 16) * 4).collect();
        for (qpos, spread) in [(narrow, 3), (wide, 0)] {
            for recent in [Recent::Candidate, Recent::Forced] {
                for (quantize, seed) in [(None, 0x81), (Some(0.5), 0x82)] {
                    kernel_matches(&KernelCase {
                        qpos: qpos.clone(),
                        top_k: 8,
                        quantize,
                        seed,
                        strata: strata(8, 12, recent),
                        prompt_blocks: if spread == 0 { 20 } else { 0 },
                        prompt_spread: spread,
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Many narrow windows on a few rows: each window runs as one block of its
    /// own, unsplit, and the merge only gathers them.
    #[test]
    fn windows_run_side_by_side_and_merge_to_the_definition() -> Result<()> {
        let few: Vec<usize> = (0..4).map(|i| 200 + i * 37).collect();
        let narrow = strata(4, 6, Recent::Candidate);
        let Some(_device) = cuda() else { return Ok(()) };
        // The deepest row, qpos 311, has 78 candidates.
        assert_eq!(split_parts(4, 78, &narrow, 8), 1);
        for (recent, quantize, seed) in [
            (Recent::Candidate, None, 0x77),
            (Recent::Candidate, Some(0.5), 0x78),
            (Recent::Forced, Some(0.5), 0x79),
        ] {
            kernel_matches(&KernelCase {
                qpos: few.clone(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(4, 6, recent),
                prompt_blocks: 5,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    #[test]
    fn the_single_pass_matches_the_definition_whole_and_stratified() -> Result<()> {
        let wide: Vec<usize> = (0..512).map(|i| 200 + i).collect();
        for (strata, quantize, seed) in [
            (Strata::WHOLE, None, 0x71),
            (Strata::WHOLE, Some(0.5), 0x72),
            (strata(32, 12, Recent::Candidate), None, 0x73),
            (strata(32, 12, Recent::Candidate), Some(0.5), 0x74),
            (strata(32, 12, Recent::Forced), Some(0.5), 0x75),
        ] {
            kernel_matches(&KernelCase {
                qpos: wide.clone(),
                top_k: 8,
                quantize,
                seed,
                strata,
                prompt_blocks: 5,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    /// The single pass at the released budget — 513 survivors a window, past
    /// half the buffer, so every chunk past the first few trims.
    #[test]
    fn the_single_pass_matches_at_the_released_budget() -> Result<()> {
        let wide: Vec<usize> = (0..512).map(|i| 65_536 + i * 3).collect();
        for strata in [Strata::WHOLE, strata(4096, 512, Recent::Forced)] {
            kernel_matches(&KernelCase {
                qpos: wide.clone(),
                top_k: 2048,
                quantize: None,
                seed: 0x76,
                strata,
                prompt_blocks: 500,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    /// The deployed geometry at the depth that motivated it: the released
    /// budget of 2048 positions at ratio 4, ~294K positions deep, cut into
    /// 128K-position windows with an 8K recent span and a 2K-block prompt.
    #[test]
    fn stratified_selection_kernel_matches_at_the_deployed_depth() -> Result<()> {
        for recent in [Recent::Candidate, Recent::Forced] {
            kernel_matches(&KernelCase {
                qpos: (0..4).map(|i| 294_000 + i * 5).collect(),
                top_k: 2048,
                quantize: None,
                seed: 0x68,
                strata: strata(32_768, 2048, recent),
                prompt_blocks: 2000,
                prompt_spread: 0,
            })?;
        }
        Ok(())
    }

    // —— Store and resume ————————————————————————————————————————————————
    //
    // A resumed sequence's index has to be the index it would have had if the
    // process had never stopped. These build a cache by ragged appends, put it
    // through the seal's export shape, rebuild it, and then require the rebuilt
    // one to behave identically — not merely to hold the same bytes.

    /// The test rig the store/resume tests share: weights, the rope table, and
    /// a hidden-state stream long enough to append in ragged waves.
    struct Rig {
        device: Device,
        cfg: IndexerConfig,
        w: IndexerWeights,
        rope: FactoredRope,
        keys: Tensor,
        ratio: usize,
        eps: f64,
    }

    impl Rig {
        fn new(device: Device, tokens: usize) -> Result<Self> {
            let cfg = IndexerConfig {
                n_heads: 2,
                head_dim: 16,
                top_k: 8,
                strata: Strata::WHOLE,
            };
            let (ratio, hidden, eps) = (4usize, 12usize, 1e-6);
            let w = IndexerWeights {
                q_proj: Tensor::from_vec(
                    lcg(cfg.n_heads * cfg.head_dim * hidden, 0x71, 0.6),
                    (cfg.n_heads * cfg.head_dim, hidden),
                    &device,
                )?,
                k_proj: Tensor::from_vec(
                    lcg(cfg.head_dim * hidden, 0x72, 0.6),
                    (cfg.head_dim, hidden),
                    &device,
                )?,
                q_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x73, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    &device,
                )?,
                k_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x74, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    &device,
                )?,
            };
            let x = Tensor::from_vec(lcg(tokens * hidden, 0x75, 1.0), (tokens, hidden), &device)?;
            let rope = FactoredRope::new(&plain_inv_freq(8, 1e6), &device)?;
            let keys = project_keys(&x, &w)?;
            Ok(Self {
                device,
                cfg,
                w,
                rope,
                keys,
                ratio,
                eps,
            })
        }

        /// Append `rows` tokens starting at `start`.
        fn append(&self, cache: &mut IndexCache, start: usize, rows: usize) -> Result<()> {
            cache.ensure_capacity(start + rows, self.ratio)?;
            let mut work = [AppendSpan { cache, start, rows }];
            append_wave(&mut work, &self.keys, &self.w, self.ratio, self.eps)
        }

        /// A cache holding the first `tokens` tokens, appended in waves whose
        /// lengths are deliberately not multiples of `ratio`.
        fn build(&self, tokens: usize) -> Result<IndexCache> {
            let mut cache = IndexCache::new(self.cfg.head_dim, &self.device)?;
            let mut at = 0usize;
            for step in [7usize, 5, 11, 3].iter().cycle() {
                if at >= tokens {
                    break;
                }
                let rows = (*step).min(tokens - at);
                self.append(&mut cache, at, rows)?;
                at += rows;
            }
            assert_eq!(cache.len(self.ratio), tokens);
            Ok(cache)
        }

        /// The seal's export shape, through the record bytes and back — the
        /// whole persistence path, not a direct field copy.
        fn round_trip(&self, cache: &IndexCache) -> Result<IndexCache> {
            use crate::models::qwen4exp::paged_index::{decode_page, encode_page, SealedIndex};
            let sealed = SealedIndex {
                rows: cache.live_rows_host()?,
                dim: self.cfg.head_dim,
                last_cells: self.ratio,
                open: cache.open_rows_host()?,
            };
            let back = decode_page(&encode_page(&sealed)?)?;
            IndexCache::from_rows(&back.rows, &back.open, self.cfg.head_dim, &self.device)
        }

        /// Every row's selection, as the model would compute it.
        fn select(&self, cache: &mut IndexCache, qpos: &[usize]) -> Result<Vec<Option<Vec<u32>>>> {
            let t = qpos.len();
            let widest = (cache.page_row_span() + cache.n_blocks).max(1);
            let x = Tensor::from_vec(
                lcg(t * self.w.q_proj.dim(1)?, 0x76, 1.0),
                (t, self.w.q_proj.dim(1)?),
                &self.device,
            )?;
            let q = project_queries(
                &x,
                &self.w,
                &self.cfg,
                &self.rope,
                qpos,
                RowRungs::Uniform(0),
                self.eps,
                None,
            )?;
            let scores = Tensor::empty((t, widest), DType::F32, &self.device)?;
            let cand = cache.score_rows(
                &q, qpos, &self.cfg, self.ratio, &self.rope, 0, &scores, widest, 0, None,
            )?;
            let mut table = SelectionTable::new(
                t,
                self.ratio,
                self.cfg.top_k,
                widest,
                &self.cfg.strata,
                &self.device,
                None,
            )?;
            let tail: Vec<u32> = qpos
                .iter()
                .map(|&p| cache.tail_len(p, self.ratio))
                .collect();
            let prompt = vec![cache.prompt_blocks(self.ratio) as u32; t];
            table.fill_rows(
                &scores,
                &cand,
                qpos,
                &tail,
                &prompt,
                self.ratio,
                self.cfg.top_k,
                0,
            )?;
            (0..t).map(|r| table.row_to_host(r)).collect()
        }
    }

    /// **The stated bound covers what a layer's selection actually carves.**
    ///
    /// The wave plan prices a layer's selection at
    /// [`super::super::select_bytes::select_layer_bytes`], and an open phase
    /// refuses anything past its price — so a bound short of the real carves
    /// fails the wave. Measured here by the cursor: a marker carve before
    /// `select_layer` and one after, at a depth where the selection engages,
    /// with the block input on the phase as the forward puts it.
    #[test]
    fn the_selection_bound_covers_what_a_layer_carves() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &device else {
            return Ok(());
        };
        let rig = Rig::new(device.clone(), 128)?;
        let (seq, past, rows) = (7usize, 96usize, 16usize);
        let mut cache = rig.build(past)?;
        cache.ensure_capacity(past + rows, rig.ratio)?;
        let mut idx_map: HashMap<usize, Vec<IndexCache>> = HashMap::new();
        idx_map.insert(seq, vec![cache]);
        let spans = [SeqSpan {
            seq,
            start: 0,
            len: rows,
        }];
        let offsets = [past];
        assert!(selection_engages(past + rows, &rig.cfg, rig.ratio));
        let bound = select_layer_bytes(0, rig.ratio, &spans, &offsets, &idx_map, &rig.cfg)?;

        let hidden = rig.w.q_proj.dim(1)?;
        let h = Tensor::from_vec(lcg(rows * hidden, 0x77, 1.0), (rows, hidden), &device)?;
        let wave = begin_wave(&cd.cuda_stream(), LayerPhase::Attention)?;
        let ticket = Some(wave.ticket());
        let h_wave = wave_empty_ticketed((rows, hidden), DType::F32, &device, ticket)?;
        h_wave.slice_set(&h, 0, 0)?;
        let before = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let sel = select_layer(
            0,
            rig.ratio,
            &rig.w,
            &rig.rope,
            &h_wave,
            &spans,
            &offsets,
            &mut idx_map,
            None,
            rows,
            &rig.cfg,
            rig.eps,
            &device,
            &AtomicU64::new(0),
            ticket,
        )?;
        assert!(sel.is_some(), "the selection engaged");
        drop(sel);
        let after = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let carved = (after - before) as usize - CARVE_ALIGN;
        assert!(
            carved <= bound,
            "select_layer carved {carved} B against a stated bound of {bound} B"
        );
        Ok(())
    }

    /// **The cache survives the record byte-for-byte, at every ragged width.**
    ///
    /// `T mod ratio` is uniformly distributed over `0..ratio` because a turn
    /// ends where its text ends, so all four remainders are exercised. Byte
    /// equality, not a tolerance: these are copies.
    #[test]
    fn a_cache_round_trips_through_the_record_at_every_ragged_width() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        for extra in 0..rig.ratio {
            let tokens = 10 * rig.ratio + extra;
            let cache = rig.build(tokens)?;
            assert_eq!(
                cache.seal_shape(),
                (tokens / rig.ratio, extra),
                "the built cache is not at the width the test intends"
            );

            let back = rig.round_trip(&cache)?;
            assert_eq!(
                back.seal_shape(),
                cache.seal_shape(),
                "the restored cache stands at a different shape ({extra} carried)"
            );
            assert_eq!(
                back.len(rig.ratio),
                tokens,
                "the restored cache reports {} tokens against {tokens} — it would \
                 index every later token against the wrong block",
                back.len(rig.ratio)
            );
            assert_eq!(
                back.live_rows_host()?,
                cache.live_rows_host()?,
                "the completed rows changed"
            );
            assert_eq!(
                back.open_rows_host()?,
                cache.open_rows_host()?,
                "the open block changed"
            );
        }
        Ok(())
    }

    /// **A resumed cache keeps decoding as though it never stopped.**
    ///
    /// The property the round trip alone cannot establish. Two caches reach the
    /// same 43 tokens — one continuously, one by resuming from a record sealed
    /// at a ragged boundary — and are then appended the same further tokens.
    /// Their rows must be identical.
    ///
    /// This is what fails if the open block is dropped: the restored cache pools
    /// its next row over the wrong tokens, every row after it is shifted, and
    /// both caches remain internally consistent while disagreeing about the
    /// sequence. Nothing downstream reports it.
    #[test]
    fn a_resumed_cache_appends_the_same_rows_as_one_that_never_stopped() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        // 43 is not a multiple of 4, so the seal lands mid-block — the case a
        // boundary-aligned fixture would miss entirely.
        let sealed_at = 43usize;
        let mut live = rig.build(sealed_at)?;
        let mut resumed = rig.round_trip(&live)?;

        for (start, rows) in [
            (sealed_at, 9usize),
            (sealed_at + 9, 17),
            (sealed_at + 26, 4),
        ] {
            rig.append(&mut live, start, rows)?;
            rig.append(&mut resumed, start, rows)?;
            assert_eq!(
                resumed.seal_shape(),
                live.seal_shape(),
                "after appending {rows} at {start} the two caches are at different shapes"
            );
            assert_eq!(
                resumed.live_rows_host()?,
                live.live_rows_host()?,
                "the resumed cache's rows diverged after appending {rows} at {start}"
            );
        }
        Ok(())
    }

    /// **And it SELECTS the same rows.**
    ///
    /// The end of the chain: identical keys are worth nothing if the selection
    /// built from them differs, and the selection is what the attention kernel
    /// actually consumes. Exact equality per row, for the same reason
    /// `device_selection_matches_the_cpu_oracle` requires it.
    #[test]
    fn a_resumed_cache_selects_exactly_what_the_live_one_selects() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        let sealed_at = 43usize;
        let mut live = rig.build(sealed_at)?;
        let mut resumed = rig.round_trip(&live)?;
        rig.append(&mut live, sealed_at, 21)?;
        rig.append(&mut resumed, sealed_at, 21)?;

        // Query positions spanning the seal boundary, so rows that see only
        // pre-seal blocks, only post-seal ones, and both are all represented.
        let qpos: Vec<usize> = (40..64).collect();
        assert_eq!(
            rig.select(&mut resumed, &qpos)?,
            rig.select(&mut live, &qpos)?,
            "the resumed cache selects different blocks than the live one — the \
             conversation would attend to different history after a restart"
        );
        Ok(())
    }

    /// **A cache populated only by INJECTION still forks.**
    ///
    /// The three carried classes are populated by different things: the
    /// recurrent store and the PLE state come from running a wave, the index
    /// also from a projection installing pages into a slot that has never
    /// decoded a token. `fork_recurrent` used to return early when the parent
    /// had no recurrent store — true reasoning for the recurrence, false for the
    /// index, and it took all three out together.
    ///
    /// That is exactly the base conversation's shape: an Arc-injected prefix of
    /// sections, never decoded. Every conversation forked from it got the base's
    /// K/V and none of the index describing it, which is silent until the prefix
    /// passes the QSA identity threshold and then fails every first turn.
    #[test]
    fn a_cache_holding_only_injected_pages_forks_them() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 128)?;
        let ratio = rig.ratio;

        // Injected only: pages pushed, nothing ever appended — no `n_blocks`,
        // no open rows, which is what "has not run a wave" looks like here.
        let mut injected = IndexCache::new(rig.cfg.head_dim, &device)?;
        let mut at = 0usize;
        for &w in &[16usize, 12, 20] {
            let rows = w.div_ceil(ratio);
            let page = zero_page(rows, w - (rows - 1) * ratio, rig.cfg.head_dim, &device);
            injected.push_page(page, at, ratio)?;
            at += w;
        }
        assert_eq!(injected.live_blocks(), 0, "the fixture forwarded something");
        assert_eq!(injected.indexed_tokens(ratio), 48);

        let child = injected.fork()?;
        assert_eq!(
            child.indexed_tokens(ratio),
            injected.indexed_tokens(ratio),
            "a fork of an injection-only cache lost its pages — the child would \
             hold the parent's borrowed K/V and no index describing it"
        );
        assert_eq!(child.page_count(), injected.page_count());
        Ok(())
    }

    /// **Placing pages together is placing them one by one.**
    ///
    /// A projection places every layer's page of a piece in one launch; a single
    /// page is placed alone. The rows each page holds — and so what it selects —
    /// must not depend on which pages shared its launch.
    #[test]
    fn pages_placed_together_select_the_same_as_pages_placed_alone() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let (ratio, d) = (rig.ratio, rig.cfg.head_dim);
        let widths = [12usize, 8, 20, 4];

        let mut made = Vec::new();
        for (i, &w) in widths.iter().enumerate() {
            let rows = w.div_ceil(ratio);
            made.push((lcg(rows * d, 0xA0 + i as u64, 1.0), w - (rows - 1) * ratio));
        }
        let host: Vec<(&[f32], usize)> = made.iter().map(|(r, l)| (r.as_slice(), *l)).collect();
        let together = ResidentPage::place_host(&host, d, &device)?;
        let alone = host
            .iter()
            .map(|&(r, l)| placed_page(r, l, d, &device))
            .collect::<Result<Vec<_>>>()?;

        let mut batched = IndexCache::new(d, &device)?;
        let mut single = IndexCache::new(d, &device)?;
        let mut at = 0usize;
        for ((a, b), &w) in together.into_iter().zip(alone).zip(widths.iter()) {
            batched.push_page(a, at, ratio)?;
            single.push_page(b, at, ratio)?;
            at += w;
        }
        let qpos: Vec<usize> = (0..at).collect();
        assert_eq!(
            rig.select(&mut batched, &qpos)?,
            rig.select(&mut single, &qpos)?,
            "the same pages selected differently depending on which launch placed them"
        );
        Ok(())
    }

    /// **A fork shares its parent's pages and selects what the parent selects.**
    ///
    /// Every view carve forks — a `repo_map` ingest carves one per directory —
    /// so a fork must score with no step of its own between the carve and the
    /// forward: it holds the parent's resident pages, not copies of them.
    #[test]
    fn a_forked_cache_shares_its_parents_pages_and_selects_the_same() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let ratio = rig.ratio;

        // A parent holding one closed page plus a live tail — the shape a view
        // is carved from mid-conversation.
        let mut parent = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut parent, 0, 40)?;
        parent.close_tail_into_page(&rig.w, ratio, rig.eps)?;
        rig.append(&mut parent, 40, 24)?;
        assert!(parent.has_pages(), "the fixture carved no page to inherit");

        let mut child = parent.fork()?;
        assert_eq!(child.page_count(), parent.page_count());
        for i in 0..parent.page_count() {
            let (a, _) = parent.page_at(i).expect("parent page");
            let (b, _) = child.page_at(i).expect("child page");
            assert!(
                Arc::ptr_eq(a, b),
                "page {i} of the fork is a copy — every carve would duplicate the \
                 parent's whole index"
            );
        }

        let qpos: Vec<usize> = (0..64).step_by(7).collect();
        assert_eq!(
            rig.select(&mut child, &qpos)?,
            rig.select(&mut parent, &qpos)?,
            "a fork selected differently than the parent it was carved from"
        );
        Ok(())
    }

    /// **The gate the placement design exists for: a page scores where it is
    /// PUT, not where it was made.**
    ///
    /// Forward the same tokens at two different offsets. Seal the first into a
    /// position-free page and inject it at the second's offset; it must select
    /// exactly what the sequence that actually forwarded those tokens there
    /// selects.
    ///
    /// A page scored at the wrong relative distance is invisible below the
    /// identity threshold (`selection_engages`) and, above it, a plausible
    /// score for the wrong block — so this is gated directly.
    #[test]
    fn a_sealed_page_selects_the_same_wherever_it_is_placed() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 512)?;
        let ratio = rig.ratio;
        // A whole number of blocks, so the pooling is identical on both sides —
        // a short trailing block is a different summary by design, and this
        // gate is about position, not about raggedness.
        let (made_at, placed_at, tokens) = (64usize, 320usize, 48usize);

        // The page, made at one offset. Its rows are un-rotated, so they carry
        // no trace of it.
        let mut origin = IndexCache::new(rig.cfg.head_dim, &device)?;
        origin.skip_to(made_at)?;
        rig.append(&mut origin, 0, tokens)?;
        let page = placed_page(&origin.live_rows_host()?, ratio, rig.cfg.head_dim, &device)?;

        // Injected at a different offset.
        let mut injected = IndexCache::new(rig.cfg.head_dim, &device)?;
        injected.skip_to(placed_at)?;
        injected.push_page(page, placed_at, ratio)?;

        // What that offset's own tokens would have produced.
        let mut forwarded = IndexCache::new(rig.cfg.head_dim, &device)?;
        forwarded.skip_to(placed_at)?;
        rig.append(&mut forwarded, 0, tokens)?;

        assert_eq!(
            injected.indexed_tokens(ratio),
            forwarded.indexed_tokens(ratio),
            "the two caches do not even cover the same span"
        );
        let qpos: Vec<usize> = (placed_at..placed_at + tokens).step_by(3).collect();
        assert_eq!(
            rig.select(&mut injected, &qpos)?,
            rig.select(&mut forwarded, &qpos)?,
            "an injected page selected differently than the same tokens forwarded at \
             that position — its rows are being read at the wrong positions"
        );
        Ok(())
    }

    /// **Closing a page mid-sequence changes nothing.** The live decode path: a
    /// unit boundary lifts the tail into a page that is placed exactly where it
    /// already sat, so every row keeps its position and the selection is
    /// untouched.
    ///
    /// Block-aligned deliberately — `close_tail_into_page` flushes a SHORT
    /// trailing block, which is a different summary on purpose, so a boundary
    /// off a block would be testing raggedness rather than position.
    #[test]
    fn closing_a_page_at_a_block_boundary_leaves_the_selection_alone() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 512)?;
        let ratio = rig.ratio;
        let (cut, total) = (40usize, 96usize);

        let mut whole = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut whole, 0, total)?;

        let mut split = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut split, 0, cut)?;
        let closed = split.close_tail_into_page(&rig.w, ratio, rig.eps)?;
        assert_eq!(closed, cut, "the cut did not close the tokens it was given");
        rig.append(&mut split, cut, total - cut)?;

        assert_eq!(split.indexed_tokens(ratio), whole.indexed_tokens(ratio));
        let qpos: Vec<usize> = (0..total).step_by(5).collect();
        assert_eq!(
            rig.select(&mut split, &qpos)?,
            rig.select(&mut whole, &qpos)?,
            "closing a page at a block boundary moved the selection"
        );
        Ok(())
    }

    /// **A page's rows are visible to exactly the queries they sit below.**
    ///
    /// The arithmetic that replaces `(pos + 1) / ratio` once a sequence holds a
    /// prefix it did not forward. Swept over three ragged pages and every
    /// position across them, because the widths only disagree with the uniform
    /// formula *after* the first boundary — a fixture with one page, or with
    /// block-aligned ones, agrees with it everywhere and proves nothing.
    #[test]
    fn candidates_walk_the_pages_widths() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let head_dim = 16usize;
        let ratio = 4usize;
        // Section lengths that are deliberately NOT multiples of the ratio, so
        // every boundary lands mid-block — which is what a section does, since
        // injection advances the sequence by its real token count.
        let sections = [13usize, 7, 26];
        let mut cache = IndexCache::new(head_dim, &device)?;
        let mut pos = 0usize;
        for &t in &sections {
            let rows = t.div_ceil(ratio);
            let page = zero_page(rows, t - (rows - 1) * ratio, head_dim, &device);
            cache.push_page(page, pos, ratio)?;
            pos += t;
        }
        let total: usize = sections.iter().sum();
        assert_eq!(cache.len(ratio), total, "pages do not span their tokens");
        assert_eq!(
            cache.page_row_span(),
            sections.iter().map(|t| t.div_ceil(ratio)).sum::<usize>()
        );

        // Monotone, never past what exists, and exactly complete at the end.
        let mut last = 0usize;
        for p in 0..total {
            let c = cache.candidates_at(p, ratio);
            assert!(c >= last, "candidates went backwards at {p}: {last} -> {c}");
            assert!(
                c <= cache.page_row_span(),
                "position {p} sees {c} rows but only {} exist",
                cache.page_row_span()
            );
            last = c;
        }
        assert_eq!(
            cache.candidates_at(total - 1, ratio),
            cache.page_row_span(),
            "the last token must see every page row — a page's short final row \
             is still wholly below it"
        );
        Ok(())
    }

    /// **The live tail counts from where the pages end, not from zero.**
    ///
    /// The join is the part that cannot be got right by accident: rows appended
    /// after an injected prefix pool from the tail's own first position, so
    /// their visibility is `page_rows + (pos − span) / ratio`. Off by one page
    /// and every forwarded token addresses a block belonging to the prefix.
    #[test]
    fn the_live_tail_counts_from_the_pages_end() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 128)?;
        let mut cache = IndexCache::new(rig.cfg.head_dim, &device)?;
        // One ragged page of 13 tokens => 4 rows, last covering 1.
        let page_tokens = 13usize;
        let page_rows = page_tokens.div_ceil(rig.ratio);
        cache.push_page(
            zero_page(
                page_rows,
                page_tokens - (page_rows - 1) * rig.ratio,
                rig.cfg.head_dim,
                &device,
            ),
            0,
            rig.ratio,
        )?;

        // Forward 9 more tokens on top: 2 whole rows and 1 carried.
        rig.append(&mut cache, 0, 9)?;
        assert_eq!(cache.len(rig.ratio), page_tokens + 9);
        assert_eq!(
            cache.candidates_at(page_tokens - 1, rig.ratio),
            page_rows,
            "the last page token must see the whole page and none of the tail"
        );
        assert_eq!(
            cache.candidates_at(page_tokens + 3, rig.ratio),
            page_rows + 1,
            "four tail tokens complete exactly one tail row"
        );
        assert_eq!(
            cache.candidates_at(page_tokens + 8, rig.ratio),
            page_rows + 2,
            "nine tail tokens complete two, with one carried"
        );
        Ok(())
    }

    /// **A page arriving after live rows is refused.**
    ///
    /// Pages describe a prefix. One pushed after a forward would claim
    /// positions the live tail already holds, and every later query would
    /// address the wrong block — silently, because the counts would still add
    /// up.
    #[test]
    fn a_page_after_live_rows_is_refused() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 32)?;
        let mut cache = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut cache, 0, 8)?;
        let page = zero_page(2, rig.ratio, rig.cfg.head_dim, &device);
        let base = cache.next_base();
        let err = cache.push_page(page, base, rig.ratio).unwrap_err();
        assert!(err.to_string().contains("must precede"), "{err}");
        Ok(())
    }

    /// **An open block wider than one can be is refused, not truncated.**
    ///
    /// `n_open` is always `< ratio ≤ MAX_RATIO` by construction, so a record
    /// declaring more is corrupt. Installing it would leave rows in the raw
    /// buffer that the next append neither pools nor overwrites.
    #[test]
    fn an_oversized_open_block_is_refused() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let head_dim = 16usize;
        let rows = vec![0f32; 4 * head_dim];
        let open = vec![0f32; (MAX_RATIO + 1) * head_dim];
        let err = IndexCache::from_rows(&rows, &open, head_dim, &device).unwrap_err();
        assert!(err.to_string().contains("open rows"), "{err}");
        Ok(())
    }

    /// **A stored row is the pooled, normed, UN-rotated block key.** The
    /// append kernel pools `ratio` projected rows and applies the indexer's
    /// `k_norm`, and nothing more: the rotation happens in the scorer, at
    /// whatever position the row is read at. Checked against the host's own
    /// arithmetic, at a nonzero `tail_base` so a stored row that picked up a
    /// rotation would show it.
    #[test]
    fn stored_rows_are_the_pooled_normed_unrotated_key() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 64)?;
        let (ratio, d) = (rig.ratio, rig.cfg.head_dim);
        let mut cache = IndexCache::new(d, &device)?;
        cache.skip_to(1000)?;
        rig.append(&mut cache, 0, 7)?;
        rig.append(&mut cache, 7, 13)?;
        let stored: Vec<Vec<f32>> = cache
            .live_rows_host()?
            .chunks_exact(d)
            .map(<[f32]>::to_vec)
            .collect();
        assert_eq!(stored.len(), 20 / ratio);

        let keys = rig.keys.to_vec2::<f32>()?;
        let k_norm = rig.w.k_norm.to_vec1::<f32>()?;
        for (b, row) in stored.iter().enumerate() {
            let mean: Vec<f64> = (0..d)
                .map(|c| {
                    (0..ratio)
                        .map(|r| keys[b * ratio + r][c] as f64)
                        .sum::<f64>()
                        / ratio as f64
                })
                .collect();
            let ms = mean.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let scale = 1.0 / (ms + rig.eps).sqrt();
            for c in 0..d {
                let want = mean[c] * scale * k_norm[c] as f64;
                assert!(
                    (row[c] as f64 - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "block {b} channel {c}: stored {} against the pooled normed key {want}",
                    row[c]
                );
            }
        }
        Ok(())
    }

    /// The whole device path — project, pool, norm, rope, score, select —
    /// against the CPU oracle's `qsa_selection_mask` on the same weights.
    ///
    /// The two compute their scores through different GEMMs, so in principle a
    /// rank could flip where two blocks' scores agree to within f32 rounding.
    /// The scores here are continuous and the gaps at the cut are orders of
    /// magnitude wider than that, so the selections are required to agree
    /// EXACTLY, row for row — if this ever becomes flaky the right answer is
    /// to say so, not to widen it into a test that no longer checks anything.
    #[test]
    fn device_selection_matches_the_cpu_oracle() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 16,
            top_k: 8,
            strata: Strata::WHOLE,
        };
        let (ratio, hidden, eps) = (4usize, 12usize, 1e-6);
        let t = 64usize;

        let w = |dev: &Device| -> Result<IndexerWeights> {
            Ok(IndexerWeights {
                q_proj: Tensor::from_vec(
                    lcg(cfg.n_heads * cfg.head_dim * hidden, 0x61, 0.6),
                    (cfg.n_heads * cfg.head_dim, hidden),
                    dev,
                )?,
                k_proj: Tensor::from_vec(
                    lcg(cfg.head_dim * hidden, 0x62, 0.6),
                    (cfg.head_dim, hidden),
                    dev,
                )?,
                q_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x63, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    dev,
                )?,
                k_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x64, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    dev,
                )?,
            })
        };
        let x_host = lcg(t * hidden, 0x65, 1.0);
        let x_gpu = Tensor::from_vec(x_host.clone(), (t, hidden), &device)?;
        let x_cpu = Tensor::from_vec(x_host, (t, hidden), &Device::Cpu)?;
        let rope_gpu = FactoredRope::new(&plain_inv_freq(8, 1e6), &device)?;
        let rope_cpu = RopeTables::new(8, 1e6, 512, &Device::Cpu)?;

        // Device: append the segment's keys, then select its rows.
        let mut cache = IndexCache::new(cfg.head_dim, &device)?;
        cache.ensure_capacity(t, ratio)?;
        let w_gpu = w(&device)?;
        let k_all = project_keys(&x_gpu, &w_gpu)?;
        // Appended in RAGGED waves, not one span. None of 7, 5, 20 is a
        // multiple of `ratio`, so every wave but the last leaves an open block
        // and the next wave's first key straddles the carried rows — which is
        // the case the batched append exists to handle and the case a
        // single-span test cannot reach. The oracle is unchanged: the same 64
        // tokens must produce the same selection however they arrived.
        let mut at = 0usize;
        for rows in [7usize, 5, 20, 32] {
            let mut work = [AppendSpan {
                cache: &mut cache,
                start: at,
                rows,
            }];
            append_wave(&mut work, &k_all, &w_gpu, ratio, eps)?;
            at += rows;
            assert_eq!(cache.len(ratio), at, "cache length after {at} tokens");
        }
        assert_eq!(at, t);
        let qpos: Vec<usize> = (0..t).collect();
        let q_all = project_queries(
            &x_gpu,
            &w_gpu,
            &cfg,
            &rope_gpu,
            &qpos,
            RowRungs::Uniform(0),
            eps,
            None,
        )?;
        let widest = t.div_ceil(ratio).max(1);
        let mut table =
            SelectionTable::new(t, ratio, cfg.top_k, widest, &cfg.strata, &device, None)?;
        let scores = Tensor::empty((t, widest), DType::F32, &device)?;
        let cand = cache.score_rows(
            &q_all, &qpos, &cfg, ratio, &rope_gpu, 0, &scores, widest, 0, None,
        )?;
        let tail: Vec<u32> = qpos.iter().map(|&p| cache.tail_len(p, ratio)).collect();
        let prompt = vec![0u32; t];
        table.fill_rows(&scores, &cand, &qpos, &tail, &prompt, ratio, cfg.top_k, 0)?;

        // Oracle: the additive mask, from the same weights on the CPU.
        let w_cpu = w(&Device::Cpu)?;
        let mut state = IndexState::empty();
        let mask = qsa_selection_mask(&x_cpu, &w_cpu, &mut state, &rope_cpu, ratio, &cfg, eps)?
            .expect("64 tokens is past the 11-cell budget");
        let mask: Vec<Vec<f32>> = mask.to_vec2()?;

        for (r, mrow) in mask.iter().enumerate() {
            let want: Vec<usize> = mrow
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 0.0)
                .map(|(j, _)| j)
                .collect();
            match table.row_to_host(r)? {
                None => {
                    assert_eq!(want, (0..=r).collect::<Vec<_>>(), "row {r} dense mismatch");
                }
                Some(entries) => {
                    let mut got: Vec<usize> = entries
                        .iter()
                        .flat_map(|&e| {
                            let b = entry_block(e);
                            (0..entry_cells(e)).map(move |c| b * ratio + c)
                        })
                        .collect();
                    got.sort_unstable();
                    assert_eq!(got, want, "row {r} disagrees with the oracle's selection");
                }
            }
        }
        Ok(())
    }

    /// **The device lookup is the host mirror, bit for bit.** A unit pair
    /// `(1, 0)` rotates to exactly `(cos, sin)` — every product is by 1 or 0 —
    /// so the rotated row is the kernel's lookup itself, compared against
    /// `table::lookup` at the positions where the factored split turns over:
    /// both ends of `LO`, every `HI` boundary sampled across the reach, and the
    /// last position. For both rotary widths the fleet runs.
    #[test]
    fn device_lookup_is_the_host_mirror_bit_for_bit() -> Result<()> {
        use crate::models::rope_schedule::ROPE_REACH;
        let Some(device) = cuda() else { return Ok(()) };
        let mut pos: Vec<usize> = vec![0, 1, 1023, 1024, 1025, ROPE_REACH - 1];
        for k in (1..2048).step_by(97) {
            pos.extend([(k << 10) - 1, k << 10, (k << 10) + 1]);
        }
        for (pairs, theta) in [(32usize, 1e7f32), (64, 1e6)] {
            let rope = FactoredRope::new(&plain_inv_freq(2 * pairs, theta), &device)?;
            let d = 2 * pairs;
            let mut host = vec![0f32; pos.len() * d];
            for r in 0..pos.len() {
                host[r * d..r * d + pairs].fill(1.0);
            }
            let src = Tensor::from_vec(host, (pos.len(), d), &device)?;
            let got = rotate_rows(
                RowSource::Dense(&src),
                &rope,
                1,
                RowPositions::PerGroup(&pos),
                RowRungs::Uniform(0),
                RotSide::Key,
                None,
            )?
            .to_vec2::<f32>()?;
            for (r, &p) in pos.iter().enumerate() {
                for k in 0..pairs {
                    let (c, s) = rope.rungs().cos_sin(0, p, k);
                    assert_eq!(
                        (got[r][k].to_bits(), got[r][k + pairs].to_bits()),
                        (c.to_bits(), s.to_bits()),
                        "P {pairs}, position {p}, pair {k}: device ({}, {}) against host ({c}, {s})",
                        got[r][k],
                        got[r][k + pairs]
                    );
                }
            }
        }
        Ok(())
    }

    /// **Normed rotation is the eager norm then the rotation, to rounding.** The
    /// queries' fused launch against `rms_norm_last` followed by the dense
    /// rotation, at the indexer's head width on two rungs; and the fused launch
    /// repeats bit for bit.
    #[test]
    fn normed_rotation_is_the_eager_norm_then_the_rotation() -> Result<()> {
        use super::super::qsa::rms_norm_last;
        use crate::models::rope_schedule::{RopeRungs, RopeSchedule, Rung};
        let Some(device) = cuda() else { return Ok(()) };
        let schedule = RopeSchedule::yarn(
            64,
            1e6,
            4096,
            vec![
                Rung {
                    ceiling: 4096,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 16384,
                    factor: 4.0,
                },
            ],
            true,
        )?;
        let rungs = RopeRungs::new(&schedule, &device)?;
        let rope = FactoredRope::over(&rungs, &device)?;
        let (heads, d) = (4usize, 128usize);
        let pos = [3usize, 900, 5000, 12_000, 17];
        let row_rung = [0u32, 0, 1, 1, 0];
        let n = pos.len() * heads;
        let src = Tensor::from_vec(lcg(n * d, 0x6B, 4.0), (n, d), &device)?;
        let weight = Tensor::from_vec(lcg(d, 0x6C, 1.0), (d,), &device)?;
        let eps = 1e-6;
        let normed = || {
            rotate_rows(
                RowSource::Normed {
                    rows: &src,
                    weight: &weight,
                    eps,
                },
                &rope,
                heads,
                RowPositions::PerGroup(&pos),
                RowRungs::PerGroup(&row_rung),
                RotSide::Query,
                None,
            )
        };
        let got = normed()?;
        let eager = rms_norm_last(&src, &weight, eps)?;
        let want = rotate_rows(
            RowSource::Dense(&eager),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::PerGroup(&row_rung),
            RotSide::Query,
            None,
        )?;
        let diff = (&got - &want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "fused norm+rotate off the eager chain by {diff} at scale {scale}"
        );
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            normed()?.flatten_all()?.to_vec1::<f32>()?,
            "the fused launch differs between two runs"
        );
        Ok(())
    }

    /// **Rows on different rungs share one rotation launch without bleed.** A
    /// wave's queries sit on their sequences' rungs; rotated together, each row
    /// is bit-for-bit what it is rotated alone at its own rung, and matches the
    /// host mirror of that rung's table with that rung's `m²` — while a key
    /// rotation at the same rung takes no scale.
    #[test]
    fn rows_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
        use crate::models::rope_schedule::{RopeRungs, RopeSchedule, Rung};
        let Some(device) = cuda() else { return Ok(()) };
        let schedule = RopeSchedule::yarn(
            8,
            1e6,
            64,
            vec![
                Rung {
                    ceiling: 64,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 128,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 256,
                    factor: 4.0,
                },
            ],
            true,
        )?;
        let rungs = RopeRungs::new(&schedule, &device)?;
        let rope = FactoredRope::over(&rungs, &device)?;
        let (heads, d) = (2usize, 16usize);
        let pos = [3usize, 90, 200, 17, 130, 250];
        let row_rung = [0u32, 1, 2, 0, 2, 1];
        let n = pos.len() * heads;
        let host = lcg(n * d, 0x5A, 1.0);
        let src = Tensor::from_vec(host.clone(), (n, d), &device)?;
        let mixed = rotate_rows(
            RowSource::Dense(&src),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::PerGroup(&row_rung),
            RotSide::Query,
            None,
        )?
        .to_vec2::<f32>()?;
        for (g, (&p, &r)) in pos.iter().zip(&row_rung).enumerate() {
            // Its own tensor: the launch takes a whole operand, never a view.
            let one = Tensor::from_vec(
                host[g * heads * d..(g + 1) * heads * d].to_vec(),
                (heads, d),
                &device,
            )?;
            let alone = rotate_rows(
                RowSource::Dense(&one),
                &rope,
                heads,
                RowPositions::PerGroup(&[p]),
                RowRungs::Uniform(r),
                RotSide::Query,
                None,
            )?
            .to_vec2::<f32>()?;
            let key = rotate_rows(
                RowSource::Dense(&one),
                &rope,
                heads,
                RowPositions::PerGroup(&[p]),
                RowRungs::Uniform(r),
                RotSide::Key,
                None,
            )?
            .to_vec2::<f32>()?;
            let m2 = rungs.q_scale(r);
            for h in 0..heads {
                let row = g * heads + h;
                assert_eq!(mixed[row], alone[h], "row {row} bled across rungs");
                let s = &host[row * d..(row + 1) * d];
                for k in 0..4 {
                    let (c, sn) = rungs.cos_sin(r, p, k);
                    let lo = s[k] * c - s[k + 4] * sn;
                    let hi = s[k + 4] * c + s[k] * sn;
                    for (got, want, what) in [
                        (mixed[row][k], lo * m2, "query lo"),
                        (mixed[row][k + 4], hi * m2, "query hi"),
                        (key[h][k], lo, "key lo"),
                        (key[h][k + 4], hi, "key hi"),
                    ] {
                        assert!(
                            (got - want).abs() <= 1e-5,
                            "row {row} pair {k} rung {r}: {what} {got} against {want}"
                        );
                    }
                }
                assert_eq!(
                    &mixed[row][8..],
                    &s[8..],
                    "row {row}: pass-through channels"
                );
            }
        }
        assert!(rotate_rows(
            RowSource::Dense(&src),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::Uniform(3),
            RotSide::Key,
            None,
        )
        .is_err());
        Ok(())
    }

    /// `rows_per_pos == 0` would divide by zero inside the kernel, so the
    /// guard there refuses to launch — but that guard also means `dst` never
    /// gets a single byte written to it. This needs no GPU: the check runs
    /// before `rotate_rows` ever asks whether `src`'s device is CUDA.
    #[test]
    fn rows_per_pos_zero_is_refused_before_dst_is_left_uninitialised() -> Result<()> {
        let device = Device::Cpu;
        let rope = FactoredRope::new(&plain_inv_freq(4, 1e6), &device)?;
        let src = Tensor::from_vec(vec![0f32; 8], (1, 8), &device)?;
        assert!(rotate_rows(
            RowSource::Dense(&src),
            &rope,
            0,
            RowPositions::Affine { base: 0, step: 1 },
            RowRungs::Uniform(0),
            RotSide::Key,
            None,
        )
        .is_err());
        Ok(())
    }

    /// The tail's route turns on the cell count at 2²⁵, however the cells
    /// split between rows and tail blocks.
    #[test]
    fn tail_route_turns_on_cells() {
        assert_eq!(GEMM_TAIL_MIN_CELLS, 33_554_432);
        assert_eq!(TailRoute::for_span(1, 0), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(4096, 8191), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(127, 262_144), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(4096, 8192), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(128, 262_144), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(2048, 32_768), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(usize::MAX, 2), TailRoute::Gemm);
    }
}
