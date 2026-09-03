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

use super::config::IndexerConfig;
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
        })
    }

    /// Blocks the key buffer can currently address — what a RoPE table built
    /// for this cache has to span, since `append` ropes each pooled block key
    /// at its own block position.
    pub fn capacity_blocks(&self) -> usize {
        self.keys.dim(0).unwrap_or(0)
    }

    /// Tokens this cache has consumed — `n_blocks · ratio + open`.
    pub fn len(&self, ratio: usize) -> usize {
        self.n_blocks * ratio + self.n_open
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
    pub fn reset(&mut self) {
        self.n_blocks = 0;
        self.n_open = 0;
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

        // Per row, the candidate blocks are those wholly below its tail.
        let cand: Vec<u32> = qpos.iter().map(|&p| ((p + 1) / ratio) as u32).collect();
        let cand_max = cand.iter().copied().max().unwrap_or(0) as usize;
        if cand_max > self.n_blocks {
            candle::bail!(
                "qsa select: a query at position {} needs {cand_max} blocks but the \
                 index cache holds {} — the segment's keys were not appended first",
                qpos.iter().max().copied().unwrap_or(0),
                self.n_blocks
            );
        }

        // The scan's right operand is the cache **transposed**, and that is a
        // view, not a copy: `narrow` on dim 0 of a row-major cache is
        // contiguous, and cuBLAS takes the transpose as `OP_T` with
        // `lda = head_dim` (`gemm_config`'s second RHS case — minor stride
        // `k`, major stride 1). Materialising it copied the whole live cache
        // per sequence per layer per wave — 128 bytes a token, which at
        // conversational depth is the largest single copy in the selection
        // path and buys the GEMM nothing it could not already read.
        let keys_t = self.keys.narrow(0, 0, cand_max.max(1))?.t()?;
        let rows_per_tile = (SCORE_TILE_BYTES / (h * cand_max.max(1) * 4)).clamp(1, t);
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
                cand_max.max(1),
                out,
                out_stride,
                row_base + row,
            )?;
            row += rows;
        }
        Ok(cand)
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
    pub fn fill_rows(
        &mut self,
        scores: &Tensor,
        cand: &[u32],
        qpos: &[usize],
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
        let cand_t = Tensor::from_slice(cand, (rows,), scores.device())?;
        let qpos_t = Tensor::from_slice(
            &qpos.iter().map(|&p| p as u32).collect::<Vec<u32>>(),
            (rows,),
            scores.device(),
        )?;

        let (s_s, s_l) = scores.storage_and_layout();
        let s_slice = match &*s_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("qsa selection: scores must be CUDA"),
        }
        .slice(s_l.start_offset()..);
        let (c_s, c_l) = cand_t.storage_and_layout();
        let c_slice = match &*c_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: cand must be CUDA"),
        }
        .slice(c_l.start_offset()..);
        let (p_s, p_l) = qpos_t.storage_and_layout();
        let p_slice = match &*p_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: qpos must be CUDA"),
        }
        .slice(p_l.start_offset()..);
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
        let (p_ptr, _pg) = p_slice.device_ptr(&stream);
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
                p_ptr as *const u32,
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
fn tensor_ptr(t: &Tensor) -> Result<u64> {
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
            jobs.push(((n_blocks + i) * ratio) as i64);
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
fn i64_ptr(t: &Tensor) -> Result<u64> {
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

    if let (Some(table), Some(q_all)) = (table.as_mut(), q_all.as_ref()) {
        // The widest row in the wave sets the score buffer's stride, so every
        // span writes into one buffer and the top-k covers all of it in a
        // single launch. A row's own candidate count still bounds its scan, so
        // the columns a narrower span leaves untouched are never read — which
        // is why the buffer is allocated uninitialised (hot-path invariant 6).
        let widest = spans
            .iter()
            .zip(offsets)
            .map(|(span, &off)| (off + span.len).div_ceil(compress_ratio))
            .max()
            .unwrap_or(0)
            .max(1);
        let scores = Tensor::empty((total_rows, widest), DType::F32, device)?;
        let mut cand: Vec<u32> = vec![0; total_rows];
        for span in spans {
            let cache = idx_map
                .get(&span.seq)
                .and_then(|c| c.get(kv))
                .ok_or_else(|| {
                    candle::Error::Msg(format!("seq {} has no index cache", span.seq))
                })?;
            let span_cand = cache.score_rows(
                &q_all.narrow(0, span.start, span.len)?,
                &positions[span.start..span.start + span.len],
                idx_cfg,
                compress_ratio,
                &scores,
                widest,
                span.start,
            )?;
            cand[span.start..span.start + span.len].copy_from_slice(&span_cand);
        }
        expect_dense(&scores, "qsa selection scores")?;
        table.fill_rows(&scores, &cand, &positions, compress_ratio, idx_cfg.top_k, 0)?;
    }
    table.map(SelectionTable::into_selection).transpose()
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
        table.fill_rows(&scores, &cand, &qpos, ratio, top_k, 0)?;

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
        table.fill_rows(&scores, &cand, &qpos, ratio, top_k, 0)?;
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
        table.fill_rows(&scores, &cand, &qpos, ratio, cfg.top_k, 0)?;

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
