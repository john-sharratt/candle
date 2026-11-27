//! The index cache for one sequence on one full-attention layer: its keys, its
//! pages and its open rows.
//!
//! Positions are recorded rather than summed, so a span nothing indexed is a
//! hole that shifts nothing after it. A page keeps its own width, because its
//! last row can be short, and every candidate count walks the pages by their
//! widths rather than dividing by the ratio. Closing the live tail into a page
//! lets a turn's index stand on its own. A cache restored from a record has to
//! carry its open rows, or it sits behind its own K/V.

use super::*;

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
            pins: Vec::new(),
            layout_gen: 0,
            pin_cache: Mutex::new(None),
        })
    }

    /// Record where the conversation's system prompt ends.
    pub fn set_prompt_end(&mut self, pos: usize) {
        self.prompt_end = pos;
    }

    /// Record the position ranges every query attends whole, replacing any
    /// before. Sorted and merged here, so the cache always holds disjoint
    /// ranges.
    pub fn set_pins(&mut self, ranges: &[Range<usize>]) {
        self.pins = merge_ranges(ranges);
        self.layout_gen += 1;
    }

    /// The pinned position ranges, sorted and disjoint.
    pub fn pins(&self) -> &[Range<usize>] {
        &self.pins
    }

    /// The pinned ranges as blocks, through the page layout, as a row of at
    /// most `max_blocks` ([`PinRow::new`]): a range covers from the block its
    /// first position falls in to the last block wholly inside it. A pinned
    /// block a query cannot see yet is clipped by the kernel at the row's
    /// candidates.
    pub fn pin_row(&self, ratio: usize, max_blocks: usize) -> PinRow {
        if self.pins.is_empty() {
            return PinRow::EMPTY;
        }
        let mut cache = self
            .pin_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((g, r, m, row)) = *cache {
            if (g, r, m) == (self.layout_gen, ratio, max_blocks) {
                return row;
            }
        }
        let blocks: Vec<Range<usize>> = self
            .pins
            .iter()
            .map(|r| self.block_range(r, ratio))
            .collect();
        let row = PinRow::new(&blocks, max_blocks);
        *cache = Some((self.layout_gen, ratio, max_blocks, row));
        row
    }

    /// The blocks position range `r` covers through this cache's page layout
    /// ([`block_range`]) — the rule pins and hit pieces both go by.
    pub fn block_range(&self, r: &Range<usize>, ratio: usize) -> Range<usize> {
        block_range(r, |pos| self.candidates_at(pos, ratio))
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

    /// Whether the attention kernels need this cache's page table: it holds
    /// pages, or its live tail opens past position 0 — a span placed with no
    /// rows, or a hole, before it — so block `b` no longer starts at
    /// `b · ratio`.
    pub fn needs_page_table(&self) -> bool {
        self.has_pages() || self.tail_base > 0
    }

    /// Blocks the live tail has completed.
    pub fn live_blocks(&self) -> usize {
        self.n_blocks
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
        let (codes, scales) = key_addr(&self.keys.page_ptrs()?, self.n_blocks, d);
        let src = self.raw_ptr()?;
        let jobs: Vec<i64> = vec![codes as i64, src as i64, cells as i64, scales as i64];

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

    /// Leave `tokens` positions empty at the tail: a reserved range's hole
    /// (`docs/placed_section_layout.md` §3.1). Rows the sequence forwarded
    /// ahead of it — the system prompt's text up to a range, prefilled live —
    /// close into a page of their own where they sit ([`Self::close_tail_into_page`]),
    /// and the tail opens past the hole; a tail already empty only moves.
    #[cfg(feature = "cuda")]
    pub fn hole(
        &mut self,
        w: &IndexerWeights,
        ratio: usize,
        rms_eps: f64,
        tokens: usize,
    ) -> Result<()> {
        self.close_tail_into_page(w, ratio, rms_eps)?;
        let to = self.next_base() + tokens;
        self.skip_to(to)
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
    /// `rows` are the completed keys as they are stored and `open` the raw open
    /// rows, `[n_open, head_dim]` row-major, both written straight into the
    /// cache's arena slots.
    pub fn from_rows(
        rows: &Q8Keys,
        open: &[f32],
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        if head_dim == 0
            || (rows.dim != head_dim && !rows.is_empty())
            || !open.len().is_multiple_of(head_dim)
        {
            candle::bail!(
                "index cache: {} rows of {}-wide keys and {} open values against head_dim \
                 {head_dim}",
                rows.n_rows(),
                rows.dim,
                open.len()
            );
        }
        let n = rows.n_rows();
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
            pins: Vec::new(),
            layout_gen: 0,
            pin_cache: Mutex::new(None),
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
        self.layout_gen += 1;
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
        self.layout_gen += 1;
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

    /// The live prefix's keys as they are stored, read back to the host out of
    /// the key pages.
    ///
    /// What a seal writes. The rows above `n_blocks` are dead until an append
    /// writes them, so handing out the whole capacity would persist uninitialised
    /// memory. The scorer never comes here — it reads the pages in place.
    pub fn live_rows_host(&self) -> Result<Q8Keys> {
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
            pins: self.pins.clone(),
            layout_gen: self.layout_gen,
            pin_cache: Mutex::new(None),
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
        self.pins.clear();
        self.layout_gen += 1;
        self.pages.drain(..).map(|p| p.page).collect()
    }

    /// Device address of this cache's open-block buffer.
    pub(super) fn raw_ptr(&self) -> Result<u64> {
        tensor_ptr(&self.raw)
    }
}
