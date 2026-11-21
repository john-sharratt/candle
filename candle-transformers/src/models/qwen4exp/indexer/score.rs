//! Scores a segment's queries against its cache, into the wave's shared score
//! buffer.
//!
//! Scoring and selecting are separate because they want different launch
//! shapes. The matmul runs per sequence, while the top-k runs over the whole wave
//! in one launch. A narrow span is scored by one paged launch over its pages and
//! its live tail. A wide span over a wide tail sends the tail to cuBLAS, which
//! reads the rotated tail through a single scratch buffer. A prompt's scan runs
//! eagerly, since its launch count follows its length.

use super::*;

impl IndexCache {
    /// The most bytes [`Self::score_rows`] can carve on its ticket's arena for
    /// `t` query rows whose deepest sees `cand_max` candidate blocks, after a
    /// wave appended `appended` more blocks to this cache — each carve rounded
    /// up to the bump's alignment, so the sum is a bound and never short.
    ///
    /// Mirrors the two routes: the paged scorer's page, offset and count
    /// tables, and — when the live tail goes to cuBLAS — the rotated tail, its
    /// page table and the one tile's scoring product every tile is written into.
    pub fn score_bound(
        &self,
        t: usize,
        cand_max: usize,
        appended: usize,
        heads: usize,
        d: usize,
        align: usize,
    ) -> Result<usize> {
        score_bound_for(
            self.page_row_span(),
            || {
                let mut chunks = 0usize;
                for placed in &self.pages {
                    chunks += placed.page.descriptors()?.len();
                }
                Ok(chunks)
            },
            self.n_blocks + appended,
            ScoreRows {
                t,
                cand_max,
                heads,
                d,
            },
            align,
        )
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
        // **A prompt's scan runs eagerly, between the wave's recorded segments.**
        // It is one library GEMM per row tile, so its launch count follows the
        // prompt's length and the GEMM's own tile and split-K follow its rows:
        // recorded, every prompt gave its segment a node count no executable
        // held, and the segment was instantiated afresh
        // (`docs/decode_graphs.md` §2.2). A decode or verify wave is a tile or
        // two of a shape that recurs, and stays recorded.
        let _eager = if t > F32_ROWS_MATMUL_MAX_ROWS {
            Some(q.device().eager()?)
        } else {
            None
        };
        // **One tile's product, reused by every tile.** A fresh product per tile
        // is a fresh carve, and the bump does not give a carve back inside its
        // phase, so the tiles added up to the whole `[t·h, cand]` product the
        // tiling exists to bound: at 200K of history a 2.8K-row prefill chunk
        // carved ~2.2 GB of scores (BlackWorld, 2026-11-09), placed per layer
        // window and bought from the expert zone each time. Written into one
        // buffer instead (`matmul_f32_into`), a tile's fold has read it before
        // the next tile's product lands — one stream, in order.
        let tile = wave_empty_ticketed(
            (rows_per_tile * h, live_cols),
            DType::F32,
            q.device(),
            q.wave_ticket(),
        )?;
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
            let raw = tile.narrow(0, 0, rows * h)?;
            matmul_f32_into(&q.narrow(0, row * h, rows * h)?, &keys_t, &raw)?;
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
    pub(super) fn score_paged(
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
                desc.push(chunk.scales as i64);
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
            // append writes it: code group `c` of row `j` at `c·16 + j·d` bytes,
            // the scales behind the page's codes. The tail's rows are consecutive
            // global rows across its pages, so every page shares the tail's
            // `delta`.
            let delta = self.tail_base as i64 - (span * ratio) as i64;
            let mut end = span;
            for (ptr, used) in self.keys.tail(self.n_blocks)? {
                desc.push(ptr as i64);
                desc.push((ptr + scale_plane_offset(d) as u64) as i64);
                desc.push(CODE_GROUP as i64);
                desc.push(d as i64);
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
pub(super) struct PagedScore {
    /// Query rows.
    pub(super) t: usize,
    /// Indexer heads per row.
    pub(super) h: usize,
    /// Indexer head width.
    pub(super) d: usize,
    /// Candidate columns the launch writes.
    pub(super) n_cand: usize,
    /// Whether the live tail is scored as the last page.
    pub(super) with_tail: bool,
    pub(super) ratio: usize,
}
