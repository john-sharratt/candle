//! The QSA indexer, on the device.
//!
//! One of these per (sequence, full-attention layer). It carries the index
//! cache — the compressed keys the selection scores against — and turns a
//! wave's rows into the packed selection the paged attention kernels read.
//!
//! # What is cached, and why it is not the raw keys
//!
//! §3.1 of the design doc: one key per `ratio` tokens. The reference caches
//! the *raw* projected keys and pools/norms/ropes them at read time, but every
//! one of those steps is a function of the block alone — the mean of its
//! `ratio` raw keys, the indexer's `k_norm`, and a rotation at the block's
//! FIRST position, none of which change once the block is complete. So a
//! completed block is prepared once and stored ready to score, and only the
//! `ratio − 1` raw rows of the block still filling are held as raw rows.
//! That is the doc's "one BF16 key per four tokens plus the four-slot ring",
//! and it makes the scan a plain dot product.
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
//! [`IndexCache::ensure_capacity`] is called at wave admission, never inside
//! the layer loop: growing reallocates, and an allocation between a tier
//! placement and the forward that reads it is the arena-window hazard the
//! span rules exist to prevent (hot-path invariant 7).

use candle::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::paged_index::IndexPage;

use super::config::IndexerConfig;
use super::place::{PlacePage, Placement, PLACE_TILE_R};
use super::qsa::{rms_norm_last, IndexerWeights};
use super::qsa_select::{max_entries, max_keep, selected_width, MAX_RATIO};
use super::spec::SpecCapture;
use crate::models::delta_net::mix::SeqSpan;
use crate::models::operand_guard::expect_dense;
use crate::models::qsa_selection::QsaSelection;
use crate::models::qwen35::attention::RopeTables;

/// Rows of scores computed in one tile.
///
/// The score matrix is `[rows × heads, blocks]`, which at depth is the largest
/// thing in the selection path — 4 heads × 4 bytes × one block per 4 tokens,
/// per query. Tiling the query rows bounds it; the tile is chosen so a tile's
/// scores stay under this many bytes.
const SCORE_TILE_BYTES: usize = 128 << 20;

/// One sequence's index cache for one full-attention layer.
#[derive(Debug)]
pub struct IndexCache {
    /// `[capacity, head_dim]` F32 — prepared block keys (pooled, normed,
    /// roped). Only `[0, n_blocks)` is live.
    keys: Tensor,
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
    /// The placement staging every page in [`Self::pages`] is scored from —
    /// each page rotated into the frame of the position it sits at, channel-
    /// blocked, in one buffer.
    ///
    /// **One placement per batch of pages, appended — never rebuilt.** A cache
    /// gains pages a few at a time (a projection injects a run of them; a decode
    /// closes one at every break token), and re-planning the whole set on each
    /// arrival would allocate and abandon a staging arena per page: O(pages²)
    /// bytes through a pool that does not hand memory back. Placing only what is
    /// pending keeps it linear, and the batched arena still does its job on the
    /// path it was measured for — a projection placing every page at once is one
    /// plan and one launch.
    ///
    /// Covers `pages[..placed_pages]`, in order. [`Self::place_pending`] extends
    /// it; [`Self::score_pages`] refuses to score a cache it does not cover
    /// rather than extending it itself, because the rotation needs the rope
    /// tables and a scorer that reached for them would be hiding a caller that
    /// forgot to place.
    placed: Vec<Placement>,
    /// Pages [`Self::placed`] accounts for.
    placed_pages: usize,
}

/// One injected page and where it sits.
#[derive(Debug, Clone)]
struct PlacedPage {
    page: IndexPage,
    /// Absolute position this page's first row occupies **in this cache**.
    ///
    /// The authority. A page's rows carry the rotation of the frame they were
    /// prepared in, and this is the frame they are being read in; the difference
    /// is the rotation [`Placement`] applies once when the page is placed.
    base: usize,
    /// Tokens the page covers, resolved at push where `ratio` is in hand — so
    /// nothing downstream needs a `ratio` to say how wide a page is.
    tokens: usize,
}

/// What a wave must put back if it fails.
#[derive(Debug)]
pub struct IndexSnapshot {
    n_blocks: usize,
    /// An OWNED copy of the open block's rows, not a handle to them: the buffer
    /// they live in is written in place by the carry kernel, so a shared clone
    /// would be overwritten by the very wave this snapshot exists to undo.
    raw: Tensor,
    n_open: usize,
}

impl IndexCache {
    pub fn new(head_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            keys: Tensor::zeros((0, head_dim), DType::F32, device)?,
            n_blocks: 0,
            // `MAX_RATIO` rows so the buffer never resizes: the open block holds
            // fewer than `ratio` rows and `ratio` is bounded by `MAX_RATIO`.
            raw: Tensor::zeros((MAX_RATIO, head_dim), DType::F32, device)?,
            n_open: 0,
            pages: Vec::new(),
            page_rows: vec![0],
            tail_base: 0,
            placed: Vec::new(),
            placed_pages: 0,
        })
    }

    /// Blocks the key buffer can currently address — what a RoPE table built
    /// for this cache has to span, since `append` ropes each pooled block key
    /// at its own block position.
    pub fn capacity_blocks(&self) -> usize {
        self.keys.dim(0).unwrap_or(0)
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

    pub fn snapshot(&self) -> Result<IndexSnapshot> {
        Ok(IndexSnapshot {
            n_blocks: self.n_blocks,
            raw: self.raw.to_owned_tensor()?,
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

    /// Blocks the live tail has completed — the position the next pooled block
    /// key ropes at, and therefore the depth a RoPE table for it must span.
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
        self.n_blocks = snap.n_blocks;
        self.raw = snap.raw.to_owned_tensor()?;
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
    pub fn flush_open_block(
        &mut self,
        w: &IndexerWeights,
        rope: &RopeTables,
        ratio: usize,
        rms_eps: f64,
    ) -> Result<Option<usize>> {
        use candle_kernels::simple::qsa_index_append::{run_qsa_index_flush, FLUSH_WORDS};

        if self.n_open == 0 {
            return Ok(None);
        }
        let cells = self.n_open;
        let d = self.keys.dim(1)?;
        self.ensure_capacity((self.n_blocks + 1) * ratio, ratio)?;
        let dst = self.keys_ptr()? + (self.n_blocks as u64) * (d * 4) as u64;
        let src = self.raw_ptr()?;
        // The block's ABSOLUTE first position, not its ordinal in the tail. The
        // queries this key is scored against are roped at their absolute slot
        // position, so a tail that roped from zero put every one of its blocks
        // `tail_base` tokens away from the frame it is read in — invisible below
        // the identity threshold, and a silently wrong relative distance above
        // it.
        let pos = self.tail_base + self.n_blocks * ratio;
        let jobs: Vec<i64> = vec![dst as i64, src as i64, cells as i64, pos as i64];

        let device = self.keys.device().clone();
        let candle::Device::Cuda(cuda) = &device else {
            candle::bail!("qsa index flush runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        let jobs_t = Tensor::from_vec(jobs, (FLUSH_WORDS,), &device)?;
        candle::set_kernel_breadcrumb("run_qsa_index_flush", file!(), line!());
        let (cos, sin) = rope.table_ptrs()?;
        unsafe {
            run_qsa_index_flush(
                i64_ptr(&jobs_t)? as *const i64,
                tensor_ptr(&w.k_norm)? as *const f32,
                cos as *const f32,
                sin as *const f32,
                d as i32,
                rope.rope_dim() as i32,
                rms_eps as f32,
                1,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
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
        rope: &RopeTables,
        ratio: usize,
        rms_eps: f64,
    ) -> Result<usize> {
        let cells = self.flush_open_block(w, rope, ratio, rms_eps)?;
        if self.n_blocks == 0 {
            return Ok(0);
        }
        // Owned: the page outlives the buffer, which the next append overwrites
        // from row 0 once `n_blocks` is reset below.
        let rows = self.live_rows()?.to_owned_tensor()?;
        let last = cells.unwrap_or(ratio);
        let tokens = (self.n_blocks - 1) * ratio + last;
        // The tail was roped at its absolute positions, so the page it becomes
        // is already in the frame it is about to be read in: it records
        // `tail_base` as the frame it was roped in AND is placed there, so the
        // rotation is the identity. This is the live decode path, and it is what
        // makes a unit boundary free.
        let base = self.tail_base;
        self.n_blocks = 0;
        self.n_open = 0;
        self.push_page(IndexPage::at_frame(rows, base, last), base, ratio)?;
        self.place_pending(rope)?;
        Ok(tokens)
    }

    /// A cache holding `rows` as its live prefix and `open` as its carried,
    /// un-pooled tail — the resume path, and the exact inverse of
    /// [`Self::live_rows`] + [`Self::open_rows`].
    ///
    /// **`open` is not optional and not decoration.** The cache's arithmetic is
    /// `n_blocks · ratio + n_open == tokens`, and every consumer depends on it:
    /// [`Self::plan`] derives the next append from `n_open`, and the scorer
    /// addresses block `k` as tokens `[k · ratio, (k+1) · ratio)`. A restore
    /// that dropped the open rows would put the cache `tokens % ratio` behind
    /// its own K/V and keep it there — internally consistent, wrong against the
    /// sequence, and silent.
    pub fn from_rows(rows: &Tensor, open: &Tensor, head_dim: usize) -> Result<Self> {
        let (n, d) = rows.dims2()?;
        if d != head_dim {
            candle::bail!("index cache: rows are [{n}, {d}] against head_dim {head_dim}");
        }
        let (n_open, open_d) = open.dims2()?;
        if n_open > MAX_RATIO {
            candle::bail!(
                "index cache: {n_open} open rows exceeds the {MAX_RATIO}-row block — a full \
                 block would have been pooled into a row instead of carried"
            );
        }
        if n_open > 0 && open_d != head_dim {
            candle::bail!("index cache: open rows are [{n_open}, {open_d}] against {head_dim}");
        }
        let raw = Tensor::zeros((MAX_RATIO, head_dim), DType::F32, rows.device())?;
        if n_open > 0 {
            raw.slice_set(open, 0, 0)?;
        }
        Ok(Self {
            keys: rows.to_owned_tensor()?,
            n_blocks: n,
            raw,
            n_open,
            pages: Vec::new(),
            page_rows: vec![0],
            tail_base: 0,
            placed: Vec::new(),
            placed_pages: 0,
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
    /// carrying `page.roped_base`'s rotation, and the difference between the two
    /// is what [`Self::place_pending`] turns through. A caller that knows only
    /// "after the last one" passes [`Self::next_base`].
    ///
    /// Recording only — the staging is built by `place_pending`, so a caller
    /// pushing a page onto every layer pays one launch per layer rather than one
    /// per page per layer.
    pub fn push_page(&mut self, page: IndexPage, base: usize, ratio: usize) -> Result<()> {
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
        let rows = page.rows()?;
        let tokens = page.tokens(ratio)?;
        self.page_rows.push(self.page_rows.last().unwrap() + rows);
        self.pages.push(PlacedPage { page, base, tokens });
        self.tail_base = base + tokens;
        // The placement is not invalidated — it still covers the pages it
        // covered, and this one joins the pending set. See `Self::placed`.
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

    /// Rotate every page into the frame of the position it sits at and build the
    /// scorer's staging — **one launch for all of this cache's pages**.
    ///
    /// Idempotent and cheap when nothing changed: a cache whose pages have not
    /// moved since the last call returns without launching.
    #[cfg(feature = "cuda")]
    pub fn place_pending(&mut self, rope: &RopeTables) -> Result<()> {
        if self.placed_pages == self.pages.len() {
            return Ok(());
        }
        // Only what is pending. The pages already placed keep the staging they
        // were given — re-planning them would abandon a live arena per push.
        let jobs: Vec<PlacePage<'_>> = self.pages[self.placed_pages..]
            .iter()
            .map(|p| {
                // A sealed page is normalised to zero, so its delta is its base;
                // one closed in this cache sits where it was roped, so its delta
                // is zero and the placement is a pure transpose.
                PlacePage {
                    keys: &p.page.keys,
                    delta: p.base as isize - p.page.roped_base as isize,
                }
            })
            .collect();
        let placement = Placement::plan(&jobs)?;
        placement.run(rope, PLACE_TILE_R)?;
        self.placed.push(placement);
        self.placed_pages = self.pages.len();
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
    pub fn page_at(&self, i: usize) -> Option<(&IndexPage, usize)> {
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
        let rows = placed.page.rows().unwrap_or(0);
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

    /// The live prefix as a view — `[n_blocks, head_dim]`, no copy.
    ///
    /// What a seal writes and what a single-page window reads. The rows above
    /// `n_blocks` are dead until an append writes them, so handing out the whole
    /// buffer would persist uninitialised memory.
    pub fn live_rows(&self) -> Result<Tensor> {
        self.keys.narrow(0, 0, self.n_blocks)
    }

    /// The carried open block as a view — `[n_open, head_dim]`, no copy.
    ///
    /// The same rule as [`Self::live_rows`]: the rows above `n_open` are dead
    /// until the next append writes them.
    pub fn open_rows(&self) -> Result<Tensor> {
        self.raw.narrow(0, 0, self.n_open)
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
    /// The copy is `n_blocks × head_dim` floats per attention layer, so it is
    /// proportional to depth rather than to the turn. That is the same shape of
    /// cost the recurrent store's `fork_from` already pays at every view carve.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            keys: self.keys.to_owned_tensor()?,
            n_blocks: self.n_blocks,
            raw: self.raw.to_owned_tensor()?,
            n_open: self.n_open,
            // The pages are shared, not copied: a page is a sealed prefix that
            // nothing appends to, so parent and child read the same rows. Only
            // the live tail above them is written, and that is copied.
            pages: self.pages.clone(),
            page_rows: self.page_rows.clone(),
            tail_base: self.tail_base,
            // The staging is NOT shared. It is the child's to rebuild: the pages
            // sit at the same places, so the rebuild reproduces it exactly, and
            // sharing a buffer between two caches that each believe they own it
            // is the kind of aliasing this design exists to remove.
            placed: Vec::new(),
            placed_pages: 0,
        })
    }

    /// Room for the blocks a sequence at `tokens` tokens will have completed.
    ///
    /// Called at wave admission. Growth doubles, so a long sequence pays a
    /// logarithmic number of copies rather than one per wave.
    pub fn ensure_capacity(&mut self, tokens: usize, ratio: usize) -> Result<()> {
        let need = tokens / ratio + 1;
        let cap = self.keys.dim(0)?;
        if cap >= need {
            return Ok(());
        }
        let head_dim = self.keys.dim(1)?;
        let grown = (cap * 2).max(need).max(64);
        // Uninitialised: the live prefix is copied in below and everything
        // above `n_blocks` is dead until an append writes it, so zeroing is a
        // full-width memset of bytes nothing reads (hot-path invariant 6).
        let mut keys = Tensor::empty((grown, head_dim), DType::F32, self.keys.device())?;
        if self.n_blocks > 0 {
            keys = keys.slice_assign(
                &[0..self.n_blocks, 0..head_dim],
                &self.keys.narrow(0, 0, self.n_blocks)?,
            )?;
        }
        self.keys = keys;
        Ok(())
    }

    /// Reset to empty — a sequence starting over at offset 0.
    /// Start the sequence over — including its injected prefix.
    ///
    /// The pages go too. They describe positions this slot held; a slot
    /// starting over holds none of them, and leaving them would put the next
    /// sequence's first token at the old prefix's end.
    pub fn reset(&mut self) {
        self.n_blocks = 0;
        self.n_open = 0;
        self.pages.clear();
        self.page_rows.truncate(1);
        self.tail_base = 0;
        self.placed.clear();
        self.placed_pages = 0;
    }

    /// Blocks this span would complete, and the rows it would leave open.
    fn plan(&self, rows: usize, ratio: usize) -> (usize, usize) {
        let have = self.n_open + rows;
        let n_new = have / ratio;
        (n_new, have - n_new * ratio)
    }

    /// Device address of this cache's prepared-key buffer.
    fn keys_ptr(&self) -> Result<u64> {
        tensor_ptr(&self.keys)
    }

    /// Device address of this cache's open-block buffer.
    fn raw_ptr(&self) -> Result<u64> {
        tensor_ptr(&self.raw)
    }

    /// Score this segment's queries against the cache, into the wave's shared
    /// score buffer. Returns each row's candidate-block count.
    ///
    /// `q` is this sequence's rows of [`project_queries`], `[T, n_heads,
    /// head_dim]`, already normed and roped. `qpos[i]` is row `i`'s absolute
    /// position. Rows land at `row_base`, `out_stride` apart.
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
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
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

        // **Two layouts, two kernels, disjoint column ranges.**
        //
        // The injected pages are stored channel-blocked so a warp's key read is
        // contiguous (`paged_index`), while the live tail is the row-major
        // buffer the append kernel writes and cuBLAS reads transposed as a view.
        // Neither can read the other's layout, and converting either one would
        // cost a full copy of it per score — so each is scored by the kernel
        // built for it, into its own columns of the same output row.
        //
        // Column order is page rows then live rows, which is block order, so
        // the selection that follows indexes them exactly as it always has.
        let page_cols = self.page_row_span();
        if page_cols > 0 {
            self.score_pages(&q, &cand, t, h, d, out, out_stride, row_base)?;
        }

        // The scan's right operand is the cache **transposed**, and that is a
        // view, not a copy: `narrow` on dim 0 of a row-major cache is
        // contiguous, and cuBLAS takes the transpose as `OP_T` with
        // `lda = head_dim` (`gemm_config`'s second RHS case — minor stride
        // `k`, major stride 1). Materialising it copied the whole live cache
        // per sequence per layer per wave — 128 bytes a token, which at
        // conversational depth is the largest single copy in the selection
        // path and buys the GEMM nothing it could not already read.
        // Live-tail columns only: the pages above already covered theirs.
        let live_cols = cand_max.saturating_sub(page_cols);
        if live_cols == 0 {
            return Ok(cand);
        }
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
        let keys_t = self.keys.narrow(0, 0, live_cols)?.t()?;
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

    /// Score the injected pages into columns `[0, page_row_span())`.
    ///
    /// The pages are separately allocated and ragged, which is exactly the
    /// descriptor-table shape `qsa_score_paged` takes: one pointer and one row
    /// count per page, and the widths folded into `cnt` on the host so the
    /// kernel never sees one (hot-path invariant 2b — nothing is concatenated).
    ///
    /// `cnt` is the FULL candidate count per row, page rows and live rows
    /// together, and the kernel masks anything past its own column range on its
    /// own — a row whose candidates run into the live tail simply has every page
    /// column visible, which is what "wholly below" means for a prefix.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn score_pages(
        &self,
        q: &Tensor,
        cand: &[u32],
        t: usize,
        h: usize,
        d: usize,
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
    ) -> Result<()> {
        use candle_kernels::simple::qsa_score_paged::run_qsa_score_paged;

        let device = self.keys.device().clone();
        // The staging, not the pages: each page's rows rotated into the frame of
        // the position it sits at. A cache that has pages but no placement is a
        // caller that pushed and did not call `place_pending` — refused rather
        // than silently scored in the wrong frame, which is the failure this
        // whole design exists to remove.
        if self.placed_pages != self.pages.len() {
            candle::bail!(
                "qsa index: scoring {} page(s) of which only {} have been placed — \
                 `place_pending` must run after a push and before a score",
                self.pages.len(),
                self.placed_pages,
            );
        }
        // The placements in order, each covering the batch it was planned for,
        // so the flattened pointers are the pages' own order.
        let mut ptrs: Vec<i64> = Vec::with_capacity(self.pages.len());
        for placement in &self.placed {
            for t in placement.staged() {
                ptrs.push(tensor_ptr(t)? as i64);
            }
        }
        if ptrs.len() != self.pages.len() {
            candle::bail!(
                "qsa index: {} staging buffer(s) against {} page(s) — the placements do \
                 not tile the pages they claim to cover",
                ptrs.len(),
                self.pages.len(),
            );
        }
        let first: Vec<u32> = self.page_rows.iter().map(|&r| r as u32).collect();
        let keys_tbl = Tensor::from_vec(ptrs, (self.pages.len(),), &device)?;
        let first_tbl = Tensor::from_vec(first, (self.page_rows.len(),), &device)?;
        let cnt_t = Tensor::from_vec(cand.to_vec(), (t,), &device)?;

        let candle::Device::Cuda(cuda) = &device else {
            candle::bail!("qsa paged score runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        candle::set_kernel_breadcrumb("run_qsa_score_paged", file!(), line!());
        unsafe {
            run_qsa_score_paged(
                tensor_ptr(q)? as *const f32,
                i64_ptr(&keys_tbl)? as *const u64,
                u32_ptr(&first_tbl)? as *const u32,
                u32_ptr(&cnt_t)? as *const u32,
                tensor_ptr(out)? as *mut f32,
                t as i32,
                h as i32,
                d as i32,
                self.page_row_span() as i32,
                self.pages.len() as i32,
                out_stride as i64,
                row_base as i64,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }
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
    ratio: usize,
}

impl SelectionTable {
    /// An uninitialised table for `rows` queries.
    ///
    /// Uninitialised is correct, not sloppy: the kernels read `entries` only
    /// below each row's `cnt`, and every row's `cnt` is written by the
    /// selection kernel (hot-path invariant 6).
    pub fn new(rows: usize, ratio: usize, top_k: usize, device: &Device) -> Result<Self> {
        if ratio == 0 || ratio > MAX_RATIO {
            candle::bail!("qsa: compression ratio {ratio} outside 1..={MAX_RATIO}");
        }
        // `max_keep`, not a second copy of the arithmetic: this guard advertises
        // the selection kernel's survivor ceiling, so it has to be the same
        // bound the kernel actually reaches.
        let keep_max = max_keep(top_k, ratio);
        if keep_max > candle_kernels::simple::qsa_topk::MAX_KEEP {
            candle::bail!(
                "qsa: top_k {top_k} at ratio {ratio} needs {keep_max} survivors, past the \
                 selection kernel's {} — its streaming buffer could not absorb a chunk",
                candle_kernels::simple::qsa_topk::MAX_KEEP
            );
        }
        let stride = max_entries(top_k, ratio);
        Ok(Self {
            entries: Tensor::empty((rows, stride), DType::U32, device)?,
            cnt: Tensor::empty((rows,), DType::U32, device)?,
            stride,
            ratio,
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
        if tail.len() != rows {
            candle::bail!(
                "qsa selection: {} tail lengths against {rows} rows",
                tail.len()
            );
        }
        // **One upload, not three.** Every one of these is a host→device copy
        // per layer per wave, and on WDDM a small transfer costs far more in
        // submission than in bytes — three of them measured ~5% of the whole
        // forward-batched ladder. They are the same length and the same dtype,
        // so they travel as one `[3, rows]` block and the kernel takes three
        // offsets into it.
        let mut packed: Vec<u32> = Vec::with_capacity(rows * 3);
        packed.extend_from_slice(cand);
        packed.extend(qpos.iter().map(|&p| p as u32));
        packed.extend_from_slice(tail);
        let packed_t = Tensor::from_slice(&packed, (3, rows), scores.device())?;

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

        let (s_ptr, _sg) = s_slice.device_ptr(&stream);
        let (c_ptr, _cg) = c_slice.device_ptr(&stream);
        // The three rows of the packed block, in the order they were written.
        let p_ptr = (c_ptr as *const u32).wrapping_add(rows);
        let t_ptr = (c_ptr as *const u32).wrapping_add(rows * 2);
        let (e_ptr, _eg) = e_slice.device_ptr(&stream);
        let (n_ptr, _ng) = n_slice.device_ptr(&stream);
        // The table is one allocation; a tile writes its own row window, so
        // the offsets go on the pointers rather than through a narrowed view.
        let e_ptr = (e_ptr as *mut u32).wrapping_add(row_base * self.stride);
        let n_ptr = (n_ptr as *mut u32).wrapping_add(row_base);
        candle::set_kernel_breadcrumb("run_qsa_topk_entries", file!(), line!());
        unsafe {
            run_qsa_topk_entries(
                s_ptr as *const f32,
                score_stride as i32,
                c_ptr as *const u32,
                p_ptr,
                t_ptr,
                e_ptr,
                self.stride as i32,
                n_ptr,
                ratio as i32,
                top_k as i32,
                rows as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
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
    h.matmul(&w.k_proj.t()?)
}

/// The wave's indexer queries — projected, normed, and roped at each row's own
/// absolute position, in one pass over every row.
///
/// The reference's order (`qsa_selection_mask`): project, RMS-norm with
/// `q_norm`, then rotate.
pub fn project_queries(
    h: &Tensor,
    w: &IndexerWeights,
    cfg: &IndexerConfig,
    rope: &RopeTables,
    positions: &[usize],
    rms_eps: f64,
) -> Result<Tensor> {
    let rows = h.dim(0)?;
    let q = h
        .matmul(&w.q_proj.t()?)?
        .reshape((rows, cfg.n_heads, cfg.head_dim))?;
    let q = rms_norm_last(&q, &w.q_norm, rms_eps)?;
    rope.apply_at_positions(&q, positions)
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
    let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;
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

/// Append every span's index keys — pool, RMS-norm, RoPE, store — in ONE pair
/// of launches for the whole wave.
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
    rope: &RopeTables,
    ratio: usize,
    _rms_eps: f64,
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
    let d = work[0].cache.keys.dim(1)?;
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
        // Before the pointer is taken: growing reallocates the key buffer.
        span.cache
            .ensure_capacity((n_blocks + n_new) * ratio, ratio)?;
        let keys = span.cache.keys_ptr()?;
        let raw = span.cache.raw_ptr()?;

        for i in 0..n_new {
            // Only the first block of a span can straddle the carried rows;
            // every later one is contiguous in the wave's projection.
            let n0 = if i == 0 { n_open.min(ratio) } else { 0 };
            let src1 = span.start + (i * ratio).saturating_sub(n_open);
            jobs.push((keys + (n_blocks + i) as u64 * row) as i64);
            jobs.push(if n0 > 0 { raw as i64 } else { 0 });
            jobs.push(n0 as i64);
            jobs.push((k_base + src1 as u64 * row) as i64);
            // Absolute, not the tail's own ordinal — see `flush_open_block`.
            jobs.push((span.cache.tail_base + (n_blocks + i) * ratio) as i64);
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
        let stream = match &device {
            candle::Device::Cuda(dev) => dev.cuda_stream(),
            _ => candle::bail!("qsa append runs on CUDA"),
        };
        let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let n_jobs = jobs.len() / JOB_WORDS;
        let n_carry = carries.len() / CARRY_WORDS;
        // Kept alive until the launches are issued.
        let jobs_t = (!jobs.is_empty())
            .then(|| Tensor::from_vec(jobs, (n_jobs * JOB_WORDS,), &device))
            .transpose()?;
        let carries_t = (!carries.is_empty())
            .then(|| Tensor::from_vec(carries, (n_carry * CARRY_WORDS,), &device))
            .transpose()?;
        if let Some(t) = jobs_t.as_ref() {
            let k_norm = tensor_ptr(&w.k_norm)?;
            let (cos, sin) = rope.table_ptrs()?;
            candle::set_kernel_breadcrumb("run_qsa_index_append", file!(), line!());
            unsafe {
                run_qsa_index_append(
                    i64_ptr(t)? as *const i64,
                    k_norm as *const f32,
                    cos as *const f32,
                    sin as *const f32,
                    d as i32,
                    rope.rope_dim() as i32,
                    ratio as i32,
                    _rms_eps as f32,
                    n_jobs as i32,
                    raw_stream,
                );
            }
        }
        if let Some(t) = carries_t.as_ref() {
            candle::set_kernel_breadcrumb("run_qsa_index_carry", file!(), line!());
            unsafe {
                run_qsa_index_carry(
                    i64_ptr(t)? as *const i64,
                    d as i32,
                    n_carry as i32,
                    raw_stream,
                );
            }
        }
    }

    for (span, (n_blocks, n_open)) in work.iter_mut().zip(commits) {
        span.cache.n_blocks = n_blocks;
        span.cache.n_open = n_open;
    }
    Ok(())
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
    rope: &RopeTables,
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
) -> Result<Option<QsaSelection>> {
    if compress_ratio == 0 {
        return Ok(None);
    }

    // Row `spans[i]` covers positions `offsets[i] ..< offsets[i] + len`.
    let engages = spans
        .iter()
        .zip(offsets)
        .any(|(span, &off)| selection_engages(off + span.len, idx_cfg, compress_ratio));

    // Absolute position of every row, in the wave's packed order.
    let mut positions: Vec<usize> = Vec::with_capacity(total_rows);
    for (span, &off) in spans.iter().zip(offsets) {
        positions.extend(off..off + span.len);
    }

    // Both projections run ONCE over the whole wave; the per-sequence caches
    // take their own row windows. One GEMM per sequence per layer would be
    // launch-bound on the decode path, where a wave is 16 rows.
    let k_all = project_keys(h, indexer)?;
    let mut table = if engages {
        qsa_rows.fetch_add(total_rows as u64, Ordering::Relaxed);
        Some(SelectionTable::new(
            total_rows,
            compress_ratio,
            idx_cfg.top_k,
            device,
        )?)
    } else {
        None
    };
    let q_all = match table {
        Some(_) => Some(project_queries(h, indexer, idx_cfg, rope, &positions, eps)?),
        None => None,
    };
    // A verifying span keeps this layer's raw keys, so a partial accept can
    // restore the entering cache and re-append exactly the accepted rows
    // (`super::spec`). Owned: `k_all` is a wave-arena tensor the generation
    // reset reclaims, and `contiguous()` cannot leave the arena — it returns
    // `self.clone()` on an already-contiguous tensor and allocates with
    // `self.wave_ticket()` when it does copy. The rewind reads this after the
    // wave.
    if let Some(c) = capture {
        for span in spans {
            if let Some(s) = c.seqs.get_mut(&span.seq) {
                if s.qsa_keys.len() <= kv {
                    s.qsa_keys.resize_with(kv + 1, || None);
                }
                s.qsa_keys[kv] = Some(k_all.narrow(0, span.start, span.len)?.to_owned_tensor()?);
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
    append_wave(&mut work, &k_all, indexer, rope, compress_ratio, eps)?;

    // **Place any page that is carrying no staging, before anything scores.**
    //
    // A cache can hold pages that have never been through `place_pending`, and
    // the way in is not the push — `push_positional_state` places as it goes —
    // but [`IndexCache::fork`]. A fork shares the parent's pages and
    // deliberately does NOT share its staging (two caches each believing they
    // own one buffer is the aliasing this design removes), so the child arrives
    // with pages and nothing to score them from. Every view carve does this, and
    // a `repo_map` ingest carves one per directory: measured as 27 directories
    // failing with `scoring 17 page(s) that have not been placed`.
    //
    // Here rather than inside the scorer because this is where the rope tables
    // are: placing needs them, and a scorer that reached for them would be
    // taking a dependency it has no other use for. Idempotent and branch-cheap —
    // a cache whose pages are already placed returns without launching, which is
    // every wave after the first.
    for span in spans {
        if let Some(cache) = idx_map.get_mut(&span.seq).and_then(|c| c.get_mut(kv)) {
            cache.place_pending(rope)?;
        }
    }

    if let (Some(table), Some(q_all)) = (table.as_mut(), q_all.as_ref()) {
        // The widest row in the wave sets the score buffer's stride, so every
        // span writes into one buffer and the top-k covers all of it in a
        // single launch. A row's own candidate count still bounds its scan, so
        // the columns a narrower span leaves untouched are never read — which
        // is why the buffer is allocated uninitialised (hot-path invariant 6).
        // Asked of the caches, not derived from the offsets. A sequence holding
        // injected pages has MORE rows than `tokens / ratio`: a page ends
        // wherever its piece did, so its last row is short, and a prefix of
        // several pieces carries one short row per boundary. Sizing this from
        // the uniform formula under-allocates by exactly that many columns, and
        // the first span to write past the end fails inside the wave.
        let widest = spans
            .iter()
            .zip(offsets)
            .map(|(span, &off)| {
                let last = off + span.len;
                match idx_map.get(&span.seq).and_then(|c| c.get(kv)) {
                    Some(cache) => cache.candidates_at(last.saturating_sub(1), compress_ratio),
                    None => last.div_ceil(compress_ratio),
                }
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let scores = Tensor::empty((total_rows, widest), DType::F32, device)?;
        let mut cand: Vec<u32> = vec![0; total_rows];
        let mut tail: Vec<u32> = vec![1; total_rows];
        for span in spans {
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
                    &scores,
                    widest,
                    span.start,
                )
                .map_err(|e| candle::Error::Msg(format!("kv layer {kv}, seq {}: {e}", span.seq)))?;
            cand[span.start..span.start + span.len].copy_from_slice(&span_cand);
            for (r, &p) in positions[span.start..span.start + span.len]
                .iter()
                .enumerate()
            {
                tail[span.start + r] = cache.tail_len(p, compress_ratio);
            }
        }
        expect_dense(&scores, "qsa selection scores")?;
        table.fill_rows(
            &scores,
            &cand,
            &positions,
            &tail,
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
    let pages_t = Tensor::from_vec(pages, (n_pages, 2), device)?;
    let win_t = Tensor::from_vec(win, (total_rows, 2), device)?;
    Ok(Some(sel.with_pages(pages_t, win_t)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen4exp::qsa::{qsa_selection_mask, IndexState};
    use crate::models::qwen4exp::qsa_select::{
        entry_block, entry_cells, selection_entries, RowSelection,
    };

    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
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
            let keys = Tensor::zeros((rows, d), DType::F32, &Device::Cpu).unwrap();
            let base = c.next_base();
            c.push_page(IndexPage::new(keys, last), base, ratio)
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
        let keys = Tensor::zeros((rows, 4), DType::F32, &Device::Cpu).unwrap();
        let base = c.next_base();
        c.push_page(IndexPage::new(keys, 20 - (rows - 1) * RATIO), base, RATIO)
            .expect("page after the span");

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
        let keys = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(
            c.push_page(IndexPage::new(keys, RATIO), 4, RATIO).is_err(),
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
        let rows = Tensor::zeros((2, d), DType::F32, &Device::Cpu).unwrap();
        let open = Tensor::zeros((0, d), DType::F32, &Device::Cpu).unwrap();

        let mut forwarded = IndexCache::from_rows(&rows, &open, d).unwrap();
        assert_eq!(forwarded.live_blocks(), 2);
        let keys = Tensor::zeros((1, d), DType::F32, &Device::Cpu).unwrap();
        let base = forwarded.next_base();
        assert!(
            forwarded
                .push_page(IndexPage::new(keys, 1), base, ratio)
                .is_err(),
            "a page landing after live rows would sit at the wrong positions"
        );

        // A cache whose tail is empty — what the cut leaves behind — accepts it.
        let mut c = caged(&[8], ratio);
        let keys = Tensor::zeros((1, d), DType::F32, &Device::Cpu).unwrap();
        let base = c.next_base();
        assert!(c.push_page(IndexPage::new(keys, 1), base, ratio).is_ok());
        assert_eq!(c.indexed_tokens(ratio), 9);
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

    /// The selection kernel against the shared definition, on scores chosen by
    /// the test — no model, no GEMM, so any disagreement is the kernel's.
    ///
    /// `quantize` collapses the scores onto a coarse grid, which is how the
    /// tie-breaking rule gets exercised: at 8 distinct values over 300 blocks
    /// the cut lands inside a run of equal scores on almost every row, and the
    /// reference resolves those by ascending block.
    fn kernel_matches_definition(blocks: usize, quantize: Option<f32>, seed: u64) -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let (ratio, top_k) = (4usize, 8usize);
        let rows = 16usize;
        // Positions chosen so every row is past the budget and the phases of
        // `qpos mod ratio` are all represented (the partial-block cut moves
        // with the phase).
        let qpos: Vec<usize> = (0..rows).map(|i| 40 + i * 7).collect();
        let cand: Vec<u32> = qpos.iter().map(|&p| ((p + 1) / ratio) as u32).collect();
        let cand_max = *cand.iter().max().unwrap() as usize;
        assert!(cand_max <= blocks, "test needs {cand_max} candidate blocks");

        let mut host = lcg(rows * blocks, seed, 4.0);
        if let Some(step) = quantize {
            for v in host.iter_mut() {
                *v = (*v / step).floor() * step;
            }
        }
        let scores = Tensor::from_vec(host.clone(), (rows, blocks), &device)?;
        let mut table = SelectionTable::new(rows, ratio, top_k, &device)?;
        // Uniform blocks here, so the tail is what the kernel used to derive.
        let tail: Vec<u32> = qpos
            .iter()
            .zip(&cand)
            .map(|(&p, &c)| (p + 1 - c as usize * ratio) as u32)
            .collect();
        table.fill_rows(&scores, &cand, &qpos, &tail, ratio, top_k, 0)?;

        let mut want = Vec::new();
        for (r, &p) in qpos.iter().enumerate() {
            let row_scores = &host[r * blocks..r * blocks + blocks];
            let sel = selection_entries(row_scores, p, ratio, top_k, &mut want);
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

    #[test]
    fn selection_kernel_matches_the_definition() -> Result<()> {
        kernel_matches_definition(300, None, 0x51)
    }

    #[test]
    fn selection_kernel_breaks_ties_by_ascending_block() -> Result<()> {
        // A coarse grid forces long runs of equal scores through the cut.
        kernel_matches_definition(300, Some(0.5), 0x52)
    }

    #[test]
    fn selection_kernel_survives_more_blocks_than_the_buffer_holds() -> Result<()> {
        // Past one streaming trim: 4096 candidate blocks against a 1024-slot
        // buffer, so the threshold path runs many times.
        let Some(device) = cuda() else { return Ok(()) };
        let (ratio, top_k) = (4usize, 64usize);
        let rows = 4usize;
        let blocks = 4096usize;
        let qpos: Vec<usize> = (0..rows).map(|i| 16000 + i * 3).collect();
        let cand: Vec<u32> = qpos.iter().map(|&p| ((p + 1) / ratio) as u32).collect();
        let host = lcg(rows * blocks, 0x53, 4.0);
        let scores = Tensor::from_vec(host.clone(), (rows, blocks), &device)?;
        let mut table = SelectionTable::new(rows, ratio, top_k, &device)?;
        // Uniform blocks here, so the tail is what the kernel used to derive.
        let tail: Vec<u32> = qpos
            .iter()
            .zip(&cand)
            .map(|(&p, &c)| (p + 1 - c as usize * ratio) as u32)
            .collect();
        table.fill_rows(&scores, &cand, &qpos, &tail, ratio, top_k, 0)?;
        let mut want = Vec::new();
        for (r, &p) in qpos.iter().enumerate() {
            let row_scores = &host[r * blocks..r * blocks + blocks];
            selection_entries(row_scores, p, ratio, top_k, &mut want);
            assert_eq!(
                table.row_to_host(r)?.expect("selective row"),
                want,
                "row {r} differs past the buffer's first trim"
            );
        }
        Ok(())
    }

    // —— Store and resume ————————————————————————————————————————————————
    //
    // A resumed sequence's index has to be the index it would have had if the
    // process had never stopped. These build a cache by ragged appends, put it
    // through the seal's export shape, rebuild it, and then require the rebuilt
    // one to behave identically — not merely to hold the same bytes.

    /// The test rig the store/resume tests share: weights, rope tables, and a
    /// hidden-state stream long enough to append in ragged waves.
    struct Rig {
        device: Device,
        cfg: IndexerConfig,
        w: IndexerWeights,
        rope: RopeTables,
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
            let rope = RopeTables::new(8, 1e6, 4096, &device)?;
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
            append_wave(
                &mut work, &self.keys, &self.w, &self.rope, self.ratio, self.eps,
            )
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
            use crate::models::qwen4exp::paged_index::{decode_page, encode_page};
            let blob = encode_page(&cache.live_rows()?, self.ratio, &cache.open_rows()?)?;
            let back = decode_page(&blob, &self.device)?;
            IndexCache::from_rows(&back.page.keys, &back.open, self.cfg.head_dim)
        }

        /// Every row's selection, as the model would compute it.
        ///
        /// Takes `&mut` because a cache holding pages has to be placed before it
        /// can be scored — the scorer refuses a cache whose pages carry no
        /// staging rather than reading them in the frame they were sealed in.
        fn select(&self, cache: &mut IndexCache, qpos: &[usize]) -> Result<Vec<Option<Vec<u32>>>> {
            cache.place_pending(&self.rope)?;
            let t = qpos.len();
            let widest = (cache.page_row_span() + cache.n_blocks).max(1);
            let x = Tensor::from_vec(
                lcg(t * self.w.q_proj.dim(1)?, 0x76, 1.0),
                (t, self.w.q_proj.dim(1)?),
                &self.device,
            )?;
            let q = project_queries(&x, &self.w, &self.cfg, &self.rope, qpos, self.eps)?;
            let scores = Tensor::empty((t, widest), DType::F32, &self.device)?;
            let cand = cache.score_rows(&q, qpos, &self.cfg, self.ratio, &scores, widest, 0)?;
            let mut table = SelectionTable::new(t, self.ratio, self.cfg.top_k, &self.device)?;
            let tail: Vec<u32> = qpos
                .iter()
                .map(|&p| cache.tail_len(p, self.ratio))
                .collect();
            table.fill_rows(&scores, &cand, qpos, &tail, self.ratio, self.cfg.top_k, 0)?;
            (0..t).map(|r| table.row_to_host(r)).collect()
        }
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
                back.live_rows()?.flatten_all()?.to_vec1::<f32>()?,
                cache.live_rows()?.flatten_all()?.to_vec1::<f32>()?,
                "the completed rows changed"
            );
            assert_eq!(
                back.open_rows()?.flatten_all()?.to_vec1::<f32>()?,
                cache.open_rows()?.flatten_all()?.to_vec1::<f32>()?,
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
                resumed.live_rows()?.flatten_all()?.to_vec1::<f32>()?,
                live.live_rows()?.flatten_all()?.to_vec1::<f32>()?,
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
            let keys = Tensor::zeros((rows, rig.cfg.head_dim), DType::F32, &device)?;
            injected.push_page(IndexPage::new(keys, w - (rows - 1) * ratio), at, ratio)?;
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

    /// **Placing incrementally is placing once.**
    ///
    /// Pages arrive a few at a time — a decode closes one at every break token —
    /// and the placement extends rather than rebuilds, because re-planning the
    /// whole set per arrival abandons a staging arena per page: O(pages²) bytes
    /// through a pool that does not hand memory back. That is an allocation
    /// argument, and this is the correctness half of it: the pages must score
    /// the same either way, or the optimisation has changed the answer.
    #[test]
    fn placing_page_by_page_scores_the_same_as_placing_them_together() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let ratio = rig.ratio;
        let widths = [12usize, 8, 20, 4];

        // The pages themselves, built once and shared by both caches so the only
        // difference is WHEN each was placed.
        let mut made = Vec::new();
        let mut at = 0usize;
        for &w in &widths {
            let rows = w.div_ceil(ratio);
            let keys = Tensor::zeros((rows, rig.cfg.head_dim), DType::F32, &device)?;
            made.push((IndexPage::new(keys, w - (rows - 1) * ratio), at));
            at += w;
        }

        // One at a time, placed after each push — the decode shape.
        let mut incremental = IndexCache::new(rig.cfg.head_dim, &device)?;
        for (page, base) in &made {
            incremental.push_page(page.clone(), *base, ratio)?;
            incremental.place_pending(&rig.rope)?;
        }
        assert_eq!(
            incremental.placed.len(),
            widths.len(),
            "placing after every push should leave one placement per page — a \
             single placement means the set was re-planned each time"
        );

        // All at once, placed once — the projection shape.
        let mut batched = IndexCache::new(rig.cfg.head_dim, &device)?;
        for (page, base) in &made {
            batched.push_page(page.clone(), *base, ratio)?;
        }
        batched.place_pending(&rig.rope)?;
        assert_eq!(
            batched.placed.len(),
            1,
            "a projection placing every page at once must still be ONE plan and \
             one launch — that is the shape the batched arena was measured for"
        );

        let total: usize = widths.iter().sum();
        let qpos: Vec<usize> = (0..total).collect();
        assert_eq!(
            rig.select(&mut incremental, &qpos)?,
            rig.select(&mut batched, &qpos)?,
            "the same pages selected differently depending on when they were \
             placed — the incremental path is not equivalent to the batched one"
        );
        Ok(())
    }

    /// **A FORK arrives with pages and no staging, and must be placed before it
    /// scores.**
    ///
    /// The child shares the parent's pages but deliberately not its placement
    /// buffer — two caches each believing they own one buffer is the aliasing
    /// this design exists to remove — so a fork is the one way a cache reaches
    /// the scorer holding pages it cannot read. Nothing rebuilds it implicitly;
    /// `select_layer` does it explicitly, where the rope tables are.
    ///
    /// This went to production before it was gated: every view carve forks, a
    /// `repo_map` ingest carves one per directory, and 27 directories failed
    /// with `scoring 17 page(s) that have not been placed` on a fresh substrate.
    /// The refusal was right — the alternative is scoring pages in the frame
    /// they were sealed in — but nothing had asked a fork to score.
    #[test]
    fn a_forked_cache_must_be_placed_before_it_scores_and_then_matches_its_parent() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 256)?;
        let ratio = rig.ratio;

        // A parent holding one closed page plus a live tail — the shape a view
        // is carved from mid-conversation.
        let mut parent = IndexCache::new(rig.cfg.head_dim, &device)?;
        rig.append(&mut parent, 0, 40)?;
        parent.close_tail_into_page(&rig.w, &rig.rope, ratio, rig.eps)?;
        rig.append(&mut parent, 40, 24)?;
        assert!(parent.has_pages(), "the fixture carved no page to inherit");

        let mut child = parent.fork()?;
        assert!(
            child.has_pages(),
            "the fork lost the parent's pages — it would score against a prefix \
             its K/V still holds"
        );

        // Unplaced, the scorer must refuse rather than read the pages in the
        // frame they were sealed in.
        let qpos: Vec<usize> = (0..64).step_by(7).collect();
        let widest = (child.page_row_span() + child.live_blocks()).max(1);
        let scores = Tensor::empty((qpos.len(), widest), DType::F32, &device)?;
        let x = Tensor::from_vec(
            lcg(qpos.len() * rig.w.q_proj.dim(1)?, 0x9A, 1.0),
            (qpos.len(), rig.w.q_proj.dim(1)?),
            &device,
        )?;
        let q = project_queries(&x, &rig.w, &rig.cfg, &rig.rope, &qpos, rig.eps)?;
        let refused = child.score_rows(&q, &qpos, &rig.cfg, ratio, &scores, widest, 0);
        assert!(
            refused.is_err(),
            "an unplaced fork scored without complaint — its pages carry the \
             frame they were sealed in, so the scores are silently wrong"
        );

        // Placed, it must select exactly what the parent selects: same pages,
        // same positions, so the staging it builds is the parent's.
        child.place_pending(&rig.rope)?;
        assert_eq!(
            rig.select(&mut child, &qpos)?,
            rig.select(&mut parent, &qpos)?,
            "a placed fork selected differently than the parent it was carved \
             from — the child's rebuilt staging is not the parent's"
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
    /// This did not hold before. Index rows carried the rotation of the frame
    /// they were prepared in and nothing recorded what that frame was, so a page
    /// borrowed into a projection scored at the relative distance it had in the
    /// conversation it came from. It is invisible below the identity threshold
    /// (`selection_engages`) and, above it, a plausible score for the wrong
    /// block.
    #[test]
    fn a_sealed_page_selects_the_same_wherever_it_is_placed() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 512)?;
        let ratio = rig.ratio;
        // A whole number of blocks, so the pooling is identical on both sides —
        // a short trailing block is a different summary by design, and this
        // gate is about position, not about raggedness.
        let (made_at, placed_at, tokens) = (64usize, 320usize, 48usize);

        // The page, made at one offset and normalised to carry none.
        let mut origin = IndexCache::new(rig.cfg.head_dim, &device)?;
        origin.skip_to(made_at)?;
        rig.append(&mut origin, 0, tokens)?;
        let page = IndexPage::new(
            super::super::place::rotate_rows(&origin.live_rows()?, -(made_at as isize), &rig.rope)?,
            ratio,
        );

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
             that position — its rows are being read in the frame they were made in"
        );
        Ok(())
    }

    /// **Closing a page mid-sequence changes nothing.** The live decode path: a
    /// unit boundary lifts the tail into a page that is placed exactly where it
    /// already sat, so the placement rotation is the identity and the selection
    /// is untouched.
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
        let closed = split.close_tail_into_page(&rig.w, &rig.rope, ratio, rig.eps)?;
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
            let keys = Tensor::zeros((rows, head_dim), DType::F32, &device)?;
            cache.push_page(IndexPage::new(keys, t - (rows - 1) * ratio), pos, ratio)?;
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
            IndexPage::new(
                Tensor::zeros((page_rows, rig.cfg.head_dim), DType::F32, &device)?,
                page_tokens - (page_rows - 1) * rig.ratio,
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
        let page = IndexPage::new(
            Tensor::zeros((2, rig.cfg.head_dim), DType::F32, &device)?,
            rig.ratio,
        );
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
        let rows = Tensor::zeros((4, head_dim), DType::F32, &device)?;
        let open = Tensor::zeros((MAX_RATIO + 1, head_dim), DType::F32, &device)?;
        let err = IndexCache::from_rows(&rows, &open, head_dim).unwrap_err();
        assert!(err.to_string().contains("open rows"), "{err}");
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
        let rope_gpu = RopeTables::new(8, 1e6, 512, &device)?;
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
            append_wave(&mut work, &k_all, &w_gpu, &rope_gpu, ratio, eps)?;
            at += rows;
            assert_eq!(cache.len(ratio), at, "cache length after {at} tokens");
        }
        assert_eq!(at, t);
        let qpos: Vec<usize> = (0..t).collect();
        let q_all = project_queries(&x_gpu, &w_gpu, &cfg, &rope_gpu, &qpos, eps)?;
        let mut table = SelectionTable::new(t, ratio, cfg.top_k, &device)?;
        let widest = t.div_ceil(ratio).max(1);
        let scores = Tensor::empty((t, widest), DType::F32, &device)?;
        let cand = cache.score_rows(&q_all, &qpos, &cfg, ratio, &scores, widest, 0)?;
        let tail: Vec<u32> = qpos.iter().map(|&p| cache.tail_len(p, ratio)).collect();
        table.fill_rows(&scores, &cand, &qpos, &tail, ratio, cfg.top_k, 0)?;

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
}
