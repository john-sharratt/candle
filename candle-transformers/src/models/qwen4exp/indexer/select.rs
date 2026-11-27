//! Builds the wave's selection table, which holds each query's chosen blocks.
//!
//! Selection runs only at depths past the budget. Below it the selection is the
//! identity, and computing it would have no effect. The row stride is checked
//! against the selection kernel's two ceilings, so a depth the kernel cannot run
//! is refused rather than corrupted. A row past the pinned-block capacity would
//! write into its neighbour, so the table is sized for the pins it can hold.
//! Every sequence's keys are appended on every wave, because a deeper later wave
//! scores against them.

use super::*;

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
    pub(super) entries: Tensor,
    pub(super) cnt: Tensor,
    /// `[rows, CUT_WORDS]` U32: each row's cut record — the lowest score its
    /// ranking kept, its position, candidates and tail cells — what the
    /// shadow scan keeps of a query (`docs/hit_driven_projection.md` §6.3).
    /// Every row is written by the selection kernel.
    pub(super) cuts: Tensor,
    pub(super) stride: usize,
    // The kernel's union buffer, in entries: the gathered bound rounded up to
    // the power of two its bitonic sort runs over.
    pub(super) gather: usize,
    // The kernel's split scratch (`qsa_topk::SPLIT_KEYS` u64, as u32 pairs):
    // what a launch too narrow to fill the device ranks each row's slices into.
    // Uninitialised — every key the merge reads, the same launch wrote. Absent
    // when this table's launch fills the device and runs the single pass.
    pub(super) split_keys: Option<Tensor>,
    pub(super) ratio: usize,
    pub(super) strata: Strata,
}

/// The split scratch a selection over `rows` rows needs — `None` when the
/// launch fills the device and runs the single pass, which reads none.
#[cfg(feature = "cuda")]
pub(super) fn split_scratch(
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
pub(super) fn split_scratch(
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
            cuts: wave_empty_ticketed((rows, CUT_WORDS), DType::U32, device, ticket)?,
            stride,
            gather: max_gathered_for(top_k, ratio, cand_max, strata).next_power_of_two(),
            split_keys: split_scratch(rows, ratio, top_k, cand_max, strata, device, ticket)?,
            ratio,
            strata: *strata,
        })
    }

    /// Read back `row` of the table and its scores for a diagnostic decode.
    /// Fences: only an armed decode comes here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_row(
        &self,
        kv: usize,
        row: usize,
        score_row: usize,
        candidates: usize,
        query_position: usize,
        scores: &Tensor,
        block_start: &dyn Fn(usize) -> usize,
    ) -> Result<()> {
        let count = self.cnt.narrow(0, row, 1)?.to_vec1::<u32>()?[0] as usize;
        let entries = self
            .entries
            .narrow(0, row, 1)?
            .narrow(1, 0, count)?
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|e| (entry_block(e), entry_cells(e)))
            .collect::<Vec<_>>();
        let scores = scores
            .narrow(0, score_row, 1)?
            .narrow(1, 0, candidates)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        decode_diag::record(&RawSelection {
            kv,
            query_position,
            scores: &scores,
            entries: &entries,
            block_start,
        });
        Ok(())
    }

    /// The selection, as the attention kernels take it, and the rows' cut
    /// records.
    pub fn into_selection(self) -> Result<(QsaSelection, Tensor)> {
        Ok((
            QsaSelection::new(self.entries, self.cnt, self.ratio)?,
            self.cuts,
        ))
    }

    /// Read `row`'s cut record back: the lowest score kept (`None` for a row
    /// that ranked nothing), and its position, candidates and tail cells. A
    /// test path.
    pub fn cut_to_host(&self, row: usize) -> Result<(Option<f32>, [u32; 3])> {
        let w = self
            .cuts
            .narrow(0, row, 1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let cut = (w[0] != NO_CUT).then(|| f32::from_bits(w[0]));
        Ok((cut, [w[1], w[2], w[3]]))
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
        // Each query's pinned block ranges — attended whole, outside every
        // window. See `IndexCache::pin_row`.
        pins: &[PinRow],
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
        if tail.len() != rows || prompt.len() != rows || pins.len() != rows {
            candle::bail!(
                "qsa selection: {} tail lengths, {} prompt spans and {} pin rows against \
                 {rows} rows",
                tail.len(),
                prompt.len(),
                pins.len()
            );
        }
        // The row stride and the union buffer were sized for `pin_blocks`
        // pinned blocks a row; a row past it would write into its neighbour.
        if let Some(p) = pins.iter().find(|p| p.blocks() > self.strata.pin_blocks) {
            candle::bail!(
                "qsa selection: a row pins {} blocks, past the {} its strata allow",
                p.blocks(),
                self.strata.pin_blocks
            );
        }
        // **One upload, not several.** Every one of these is a host→device copy
        // per layer per wave, and on WDDM a small transfer costs far more in
        // submission than in bytes — three of them measured ~5% of the whole
        // forward-batched ladder. They are the same length and the same dtype,
        // so they travel as one `[4 + 2·MAX_PINS, rows]` block and the kernel
        // takes offsets into it.
        let meta_rows = 4 + 2 * MAX_PINS;
        let mut packed: Vec<u32> = Vec::with_capacity(rows * meta_rows);
        packed.extend_from_slice(cand);
        packed.extend(qpos.iter().map(|&p| p as u32));
        packed.extend_from_slice(tail);
        packed.extend_from_slice(prompt);
        for k in 0..MAX_PINS {
            packed.extend(pins.iter().map(|p| p.0[k].0));
            packed.extend(pins.iter().map(|p| p.0[k].1));
        }
        // Beside the scores, on the layer's span, where the rest of the
        // selection lives.
        let packed_t = scores.from_vec_beside(packed, (meta_rows, rows))?;

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
        let (u_s, u_l) = self.cuts.storage_and_layout();
        let u_slice = match &*u_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: cuts must be CUDA"),
        }
        .slice(u_l.start_offset()..);
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
        let pin_ptr = (c_ptr as *const u32).wrapping_add(rows * 4);
        let (e_ptr, _eg) = e_slice.device_ptr(&stream);
        let (n_ptr, _ng) = n_slice.device_ptr(&stream);
        let (u_ptr, _ug) = u_slice.device_ptr(&stream);
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
        let u_ptr = (u_ptr as *mut u32).wrapping_add(row_base * CUT_WORDS);
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
                pin_ptr,
                e_ptr,
                self.stride as i32,
                n_ptr,
                u_ptr,
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
        if cnt == super::super::qsa_select::DENSE_ROW {
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

/// Whether a layer at this depth selects at all.
///
/// Below the budget every visible cell is attended, so the indexer would
/// compute a selection that is the identity — the reference skips it and so
/// does the engine (§12.5). This is the check that keeps a short context on
/// exactly the arithmetic it had before QSA existed.
pub fn selection_engages(max_visible: usize, cfg: &IndexerConfig, ratio: usize) -> bool {
    ratio > 0 && max_visible > selected_width(cfg.top_k, ratio)
}

/// What a layer's selection leaves of its query rows beside the table the
/// attention reads: their rotated queries and their cut records — what the
/// hit count keeps of a stride for the shadow scan
/// (`docs/hit_driven_projection.md` §6.3). Both live on the layer's span.
#[derive(Debug, Clone)]
pub struct SelectedRows {
    /// `[rows, n_heads, head_dim]` F32, rotated at each row's own position.
    pub q: Tensor,
    /// `[rows, CUT_WORDS]` U32 ([`SelectionTable::cut_to_host`]).
    pub cuts: Tensor,
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
///
/// `offsets` are the spans' first POSITIONS. `token_runs` has one entry per
/// span: a prefill span's slot position runs, which name its pages by token
/// for the prefill kernel (`token_pages`), and `None` for a decode span, whose
/// pages stay at their positions.
#[allow(clippy::too_many_arguments)]
pub fn select_layer(
    kv: usize,
    compress_ratio: usize,
    indexer: &IndexerWeights,
    rope: &FactoredRope,
    h: &Tensor,
    spans: &[SeqSpan],
    offsets: &[usize],
    token_runs: &[Option<Vec<[u32; 2]>>],
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
) -> Result<Option<(QsaSelection, SelectedRows)>> {
    if compress_ratio == 0 {
        return Ok(None);
    }
    if token_runs.len() != spans.len() {
        candle::bail!(
            "qsa select: {} spans against {} position-run entries",
            spans.len(),
            token_runs.len()
        );
    }

    // Row `spans[i]` covers positions `offsets[i] ..< offsets[i] + len`.
    // The dense control of a diagnostic decode: no selection, every cell read.
    let engages = !decode_diag::dense()
        && spans
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
    // sequence's capture row.
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
        // Scored and ranked a run at a time (`score_runs`), each run's buffer as
        // wide as its own deepest row. A row's own candidate count still bounds
        // its scan, so the columns a narrower span leaves untouched are never
        // read — which is why each buffer is allocated uninitialised (hot-path
        // invariant 6).
        for run in score_runs(spans) {
            let run_spans = &spans[run.clone()];
            let base = run_spans[0].start;
            let rows = run_spans.iter().map(|s| s.len).sum::<usize>();
            let run_widest = widest_candidates(
                run_spans,
                &offsets[run.clone()],
                idx_map,
                kv,
                compress_ratio,
            );
            let scores = wave_empty_ticketed((rows, run_widest), DType::F32, device, ticket)?;
            let mut cand: Vec<u32> = vec![0; rows];
            let mut tail: Vec<u32> = vec![1; rows];
            let mut prompt: Vec<u32> = vec![0; rows];
            let mut pins: Vec<PinRow> = vec![PinRow::EMPTY; rows];
            for (span, &rung) in run_spans.iter().zip(&span_rungs[run.clone()]) {
                let cache = idx_map
                    .get(&span.seq)
                    .and_then(|c| c.get(kv))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!("seq {} has no index cache", span.seq))
                    })?;
                let at = span.start - base;
                // Name the cache, not just the shortfall. A sequence holds one
                // cache per KV layer and they are not interchangeable: the
                // trunk's are appended by every wave, while the MTP draft head's
                // — the slot at `n_attention_layers()` — is appended only by the
                // draft walks it runs. A shortfall reported without its layer
                // reads as one bug in the index when it is two different
                // populations of the same array.
                let span_cand = cache
                    .score_rows(
                        &q_all.narrow(0, span.start, span.len)?,
                        &positions[span.start..span.start + span.len],
                        idx_cfg,
                        compress_ratio,
                        rope,
                        rung,
                        &scores,
                        run_widest,
                        at,
                        ticket,
                    )
                    .map_err(|e| {
                        candle::Error::Msg(format!("kv layer {kv}, seq {}: {e}", span.seq))
                    })?;
                cand[at..at + span.len].copy_from_slice(&span_cand);
                for (r, &p) in positions[span.start..span.start + span.len]
                    .iter()
                    .enumerate()
                {
                    tail[at + r] = cache.tail_len(p, compress_ratio);
                }
                prompt[at..at + span.len].fill(cache.prompt_blocks(compress_ratio) as u32);
                pins[at..at + span.len]
                    .fill(cache.pin_row(compress_ratio, idx_cfg.strata.pin_blocks));
            }
            expect_dense(&scores, "qsa selection scores")?;
            table.fill_rows(
                &scores,
                &cand,
                &positions[base..base + rows],
                &tail,
                &prompt,
                &pins,
                compress_ratio,
                idx_cfg.top_k,
                base,
            )?;
            if decode_diag::armed() {
                for span in run_spans {
                    if decode_diag::watching(span.seq) {
                        let row = span.start + span.len - 1;
                        let cache = idx_map.get(&span.seq).and_then(|c| c.get(kv));
                        if let Some(cache) = cache {
                            table.record_row(
                                kv,
                                row,
                                row - base,
                                cand[row - base] as usize,
                                positions[row],
                                &scores,
                                &|b| cache.block_start(b, compress_ratio),
                            )?;
                        }
                    }
                }
            }
        }
    }
    // The page layout the attention kernels walk, built only when some sequence
    // in this wave holds an injected prefix. Every other wave passes null and
    // the kernels keep their inline `pos / ratio`.
    let sel = table.map(SelectionTable::into_selection).transpose()?;
    let (Some((sel, cuts)), Some(q)) = (sel, q_all) else {
        return Ok(None);
    };
    let rows = SelectedRows { q, cuts };
    let needs_pages = spans.iter().any(|s| {
        idx_map
            .get(&s.seq)
            .and_then(|c| c.get(kv))
            .is_some_and(|c| c.needs_page_table())
    });
    if !needs_pages {
        log_bulk_selection(total_rows, widest, 0);
        return Ok(Some((sel, rows)));
    }
    let mut pages: Vec<u32> = Vec::new();
    let mut win: Vec<u32> = vec![0; total_rows * 2];
    for (span, runs) in spans.iter().zip(token_runs) {
        let off = pages.len() / 2;
        let mut prefixes = match idx_map.get(&span.seq).and_then(|c| c.get(kv)) {
            Some(cache) => cache.page_prefixes(),
            // A sequence with no cache contributes the degenerate layout, which
            // is the uniform arithmetic.
            None => vec![0, 0],
        };
        // A prefill span's rows are read by a kernel that walks tokens, so its
        // pages are named by token; a decode span's stay at their positions,
        // which its kernel resolves through the slices' RoPE bases.
        if let Some(runs) = runs {
            pages_to_tokens(&mut prefixes, runs);
        }
        let count = prefixes.len() / 2;
        pages.extend(prefixes);
        for r in span.start..span.start + span.len {
            win[r * 2] = off as u32;
            win[r * 2 + 1] = count as u32;
        }
    }
    let n_pages = pages.len() / 2;
    log_bulk_selection(total_rows, widest, n_pages);
    let pages_t = wave_from_vec_ticketed(pages, (n_pages, 2), device, ticket)?;
    let win_t = wave_from_vec_ticketed(win, (total_rows, 2), device, ticket)?;
    Ok(Some((sel.with_pages(pages_t, win_t)?, rows)))
}
