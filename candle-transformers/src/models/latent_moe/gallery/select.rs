//! The two-stage selection: sign-agreement recall shortlists the corpus, and a
//! float rescore picks the top-k.
//!
//! The Indexer's score for an entry is the head-weighted sum of rectified dot
//! products, computed only over the shortlist recall returns, so the float work per
//! query is bounded however deep the corpus is. The reduction over heads and the
//! padding mask run in one launch, with masked columns sent to a value that never
//! reaches the top-k. The reduction accumulates exactly in f32, so the kernel is
//! held to exact arithmetic rather than to the eager chain's own rounding. The
//! returned entries are the top-k set, and the reader does not depend on their
//! order.

use super::*;

/// `out[b, j] = Σ_h relu(scores[b, h, j])·w[b, h] + (j < counts[b] ? 0 : -1e30)`
/// — the two-stage selection's precision-stage reduction, for `scores`
/// `[B, H, M]` and per-head gate weights `w` `[B, H]`.
///
/// On CUDA this is ONE launch; eagerly it is eight (`relu`, `broadcast_mul`, a
/// `sum` over the head axis — a non-last dim, so candle's generic reduce — then
/// `arange`, `broadcast_lt`, `to_dtype`, `affine` to build the padding mask
/// on-device, and `broadcast_add` to apply it), and eight is a floor since that
/// `sum` may itself be more than one. Both batched selectors share it: the decode wave's
/// [`two_stage_select_batched`] and prefill's
/// [`FloatGallery::batched_causal_select_device`], which differ only in what
/// `counts` means — a per-session shortlist width in one, a per-token causal
/// visibility bound in the other.
///
/// `counts = None` leaves every column unmasked. The eager form below is the CPU
/// path and the reference `indexer_score_reduce_matches_eager` compares against.
///
/// Scope is deliberately narrow: `two_stage_select_batched` records a measured
/// −29% prefill from fusing this stage's MATMUL and ARGSORT, both of which want
/// the whole GPU. This replaces only the elementwise reduction between them.
pub(super) fn indexer_score_reduce(
    scores: &Tensor,
    w: &Tensor,
    counts: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, m) = scores.dims3()?;
    if w.dims2()? != (b, h) {
        candle::bail!(
            "indexer_score_reduce: scores {:?} and weights {:?} disagree",
            scores.dims(),
            w.dims()
        );
    }

    #[cfg(feature = "cuda")]
    if matches!(scores.device(), Device::Cuda(_)) {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        use candle_kernels::simple::indexer_score::run_indexer_score_reduce;

        let dev = match scores.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!("guarded by the matches! above"),
        };
        let stream = dev.cuda_stream();
        // The kernel writes every element, so allocate uninitialised rather than
        // memset bytes it is about to stamp (hot-path invariant 6).
        let out = Tensor::empty((b, m), DType::F32, scores.device())?;
        {
            let (s_sc, l_sc) = scores.storage_and_layout();
            let (s_w, l_w) = w.storage_and_layout();
            let (s_out, _) = out.storage_and_layout();
            let f32_ptr = |st: &Storage, off: usize| -> Result<u64> {
                match st {
                    Storage::Cuda(c) => {
                        Ok(c.as_cuda_slice::<f32>()?.slice(off..).device_ptr(&stream).0)
                    }
                    _ => candle::bail!("indexer_score_reduce: expected CUDA storage"),
                }
            };
            // `counts` is u32 and optional; a null pointer means "no padding".
            let cnt_guard = match counts {
                Some(c) => {
                    if c.dims1()? != b {
                        candle::bail!("indexer_score_reduce: counts {:?} must be [{b}]", c.dims());
                    }
                    Some(c.storage_and_layout())
                }
                None => None,
            };
            // Stride threaded like every other input: `counts` is the one a
            // caller is most likely to pass as a view of something wider, and
            // assuming stride 1 would read a neighbour as a visibility bound.
            let (p_cnt, cnt_s) = match &cnt_guard {
                Some((st, lay)) => match &**st {
                    Storage::Cuda(c) => (
                        c.as_cuda_slice::<u32>()?
                            .slice(lay.start_offset()..)
                            .device_ptr(&stream)
                            .0,
                        lay.stride()[0] as i64,
                    ),
                    _ => candle::bail!("indexer_score_reduce: expected CUDA counts"),
                },
                None => (0u64, 1i64),
            };
            let p_sc = f32_ptr(&s_sc, l_sc.start_offset())?;
            let p_w = f32_ptr(&s_w, l_w.start_offset())?;
            let p_out = f32_ptr(&s_out, 0)?;
            let (ss, ws) = (l_sc.stride(), l_w.stride());
            unsafe {
                run_indexer_score_reduce(
                    p_sc as *const f32,
                    p_w as *const f32,
                    p_cnt as *const u32,
                    p_out as *mut f32,
                    b as i32,
                    h as i32,
                    m as i32,
                    ss[0] as i64,
                    ss[1] as i64,
                    ss[2] as i64,
                    ws[0] as i64,
                    ws[1] as i64,
                    cnt_s,
                    // Packed: this caller's rows all have the same width.
                    m as i64,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        return Ok(out);
    }

    let weighted = scores.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
    match counts {
        None => Ok(weighted),
        Some(c) => {
            let col = Tensor::arange(0u32, m as u32, scores.device())?.reshape((1, m))?;
            let mask = col
                .broadcast_lt(&c.reshape((b, 1))?)?
                .to_dtype(DType::F32)?
                .affine(1e30, -1e30)?; // valid → 0, pad → −1e30
            weighted.broadcast_add(&mask)
        }
    }
}

/// Batched two-stage selection across concurrent decode sessions. Runs Stage 1
/// (BDP recall + top-M shortlist) for the WHOLE wave in one launch per kernel —
/// the per-session loop's dominant cost (`topm_select` = a single-warp serial
/// bin scan × sessions) collapses to a single grid — then does the bounded
/// Stage-2 float rescore per session (a few-column cuBLAS matmul + argsort over
/// ≤`top_m` keys). Each session's gallery is independent and, at decode, wholly
/// causal, so this is byte-identical per session to calling
/// [`FloatGallery::two_stage_select`] in a loop — the shortlist is the same
/// M-superset and the float rescore picks the same top-k. Returns per session
/// `(gids_ascending [k] u32, k)`; empty/degenerate sessions get `(zeros(1), 0)`.
#[cfg(feature = "cuda")]
pub fn two_stage_select_batched(
    galleries: &[&FloatGallery],
    queries: &[Tensor],
    weights: &[Tensor],
    top_m: usize,
    top_k: usize,
    generation: &Generation,
) -> Result<Vec<(Tensor, usize)>> {
    let n_sess = galleries.len();
    if n_sess == 0 {
        return Ok(Vec::new());
    }
    let dev = galleries[0].device.clone();
    let ihd = galleries[0].index_head_dim;
    let n_heads = queries[0].dim(0)?;
    let empty = || -> Result<(Tensor, usize)> { Ok((Tensor::zeros(1, DType::U32, &dev)?, 0)) };
    let max_m = top_m.clamp(1, 1024);

    // ── Partition: only DEEP sessions (`len > max_m`) need the recall stage.
    // Shallow sessions (`0 < len ≤ max_m`) take the whole gallery as the
    // shortlist — the per-session path's all-pass fast branch — so a batch of
    // short conversations pays no Stage-1 cost at all (the batched recall win is
    // realized exactly where it exists: unbounded-depth galleries). ──
    let deep_idx: Vec<usize> = (0..n_sess).filter(|&s| galleries[s].len > max_m).collect();

    // ── Stage 1 (batched) over the deep sessions only ──
    // `shortlist_deep[d]` (row d = deep_idx[d]) holds session-relative top-M ids.
    let shortlist_deep: Option<Tensor> = if deep_idx.is_empty() {
        None
    } else {
        let nd = deep_idx.len();
        let q_signs_cat = Tensor::cat(
            &deep_idx
                .iter()
                .map(|&s| sign_pack(&queries[s]))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?; // [nd*h, words]
            // Per-session resident sign indices, read in place via a device pointer table
            // inside `bdp_recall_batched` — no depth-scaling concatenation. The gallery owns
            // each `signs` buffer (never spilled), so these views stay valid for the launch.
        let mut sign_tensors: Vec<Tensor> = Vec::with_capacity(nd);
        let mut off = Vec::with_capacity(nd);
        let mut cnt = Vec::with_capacity(nd);
        let mut running = 0u32;
        let mut max_g = 0usize;
        for &s in &deep_idx {
            let len = galleries[s].len;
            off.push(running);
            cnt.push(len as u32);
            sign_tensors.push(galleries[s].packed_signs()?);
            running += len as u32;
            max_g = max_g.max(len);
        }
        // Staged ONCE and shared by the recall and the top-M below, rather than
        // each helper uploading its own copy of the same segment table.
        let off_t = desc::stage_slice(&off, generation)?;
        let cnt_t = desc::stage_slice(&cnt, generation)?;
        let bins = n_heads * ihd + 1;
        let counts = bdp_recall_batched(
            &q_signs_cat,
            &sign_tensors,
            off_t.ptr(),
            cnt_t.ptr(),
            nd,
            n_heads,
            max_g,
            ihd,
            generation,
        )?;
        Some(topm_select_batched(
            &counts,
            off_t.ptr(),
            cnt_t.ptr(),
            nd,
            max_g,
            max_m,
            bins,
        )?) // [nd, max_m]
    };

    // ── Stage 2 (batched): float rescore over the per-session shortlists ──
    // Phase A gathers each session's ≤`max_m` shortlist keys (tier-aware — from
    // CPU RAM when the gallery is spilled, in place otherwise). Phase B then
    // rescores the WHOLE batch in one padded `bmm` + one batched argsort + one
    // batched gather, replacing the per-session matmul/argsort launch loop. The
    // padding columns (`j ≥ mₛ`) are masked to −∞ so they never enter a top-k.
    // The selected gids are returned in descending-score order, not ascending:
    // the decode reader gathers them into a dense `comp_idx` block and attends
    // the SET (softmax is order-independent; each entry carries its own position
    // for RoPE-at-read), so order is immaterial — and the byte-for-byte per
    // session equivalence the tests hold is over the selected gid SET.
    let mut out_slots: Vec<Option<(Tensor, usize)>> = (0..n_sess).map(|_| None).collect();
    // Phase A — per-session shortlist + tier-aware key gather.
    struct Active {
        s: usize,
        sl: Tensor,   // [m_s] session-relative shortlist ids
        keys: Tensor, // [m_s, ihd] gathered scoring keys (GPU)
        m_s: usize,
        k: usize,
    }
    let mut active: Vec<Active> = Vec::new();
    let mut deep_cursor = 0usize;
    for (s, (g, _q)) in galleries.iter().zip(queries).enumerate() {
        let len = g.len;
        if len == 0 || top_k == 0 {
            out_slots[s] = Some(empty()?);
            continue;
        }
        let m_s = max_m.min(len);
        let sl = if len > max_m {
            let row = shortlist_deep
                .as_ref()
                .expect("deep sessions produced a shortlist")
                .narrow(0, deep_cursor, 1)?
                .reshape(max_m)?
                .narrow(0, 0, m_s)?;
            deep_cursor += 1;
            row
        } else {
            Tensor::arange(0u32, len as u32, &dev)?
        };
        let keys = g.gather_keys(&sl)?; // [m_s, ihd]
        active.push(Active {
            s,
            sl,
            keys,
            m_s,
            k: top_k.min(m_s),
        });
    }
    if active.is_empty() {
        return Ok(out_slots.into_iter().map(|o| o.expect("filled")).collect());
    }

    // Phase B — pad to the batch's widest shortlist and rescore all at once.
    let big = active.len();
    let mm = active.iter().map(|a| a.m_s).max().unwrap();
    let max_k = active.iter().map(|a| a.k).max().unwrap();
    let keys_pad: Vec<Tensor> = active
        .iter()
        .map(|a| {
            if a.m_s == mm {
                Ok(a.keys.clone())
            } else {
                let pad = Tensor::zeros((mm - a.m_s, ihd), DType::F32, &dev)?;
                Tensor::cat(&[&a.keys, &pad], 0)
            }
        })
        .collect::<Result<_>>()?;
    let keys_all = Tensor::stack(&keys_pad, 0)?; // [big, mm, ihd]
    let q_all = Tensor::stack(
        &active
            .iter()
            .map(|a| queries[a.s].clone())
            .collect::<Vec<_>>(),
        0,
    )?; // [big, h, ihd]
    let w_all = Tensor::stack(
        &active
            .iter()
            .map(|a| weights[a.s].clone())
            .collect::<Vec<_>>(),
        0,
    )?; // [big, h]

    // scores[b,h,j] = q·k ; weighted[b,j] = Σ_h relu(scores)·w
    //
    // The `contiguous()` on the transpose is a known invariant-2 violation, kept
    // for now because deleting it is NOT yet justified by measurement. candle's
    // matmul does accept the strided transpose — every gate stays green without
    // the copy, here and at the three sibling sites — but the model-level A/B
    // came back inside run-to-run variance (cfg8 bulk 790.5 without vs 802.8
    // with, against a 820.4 reading from an identical earlier build; cfg16
    // decode 54.2 vs 54.7 vs 56.6). A copy this size should show up, and it does
    // not, which points at cuBLAS picking a slower kernel for the strided
    // operand and roughly cancelling the saved traffic. Removing it is still the
    // right end state; it needs the GEMM work to land first, not just the
    // deletion, and it needs an nsys attribution of the ucopy_f32 population
    // rather than a whole-model sweep that cannot resolve it.
    // [big,h,mm]
    let scores = q_all.matmul(&keys_all.transpose(1, 2)?.contiguous()?)?;
    // relu, weight, sum over heads and the padding mask, in ONE launch. Padding
    // keys score 0, which could outrank a genuinely negative-scoring real entry,
    // so masked columns take −1e30 and stay out of the top-k. Only the tiny
    // [big] count vector is uploaded, never a [big·mm] host-built mask.
    let counts: Vec<u32> = active.iter().map(|a| a.m_s as u32).collect();
    let counts = Tensor::from_vec(counts, big, &dev)?;
    let weighted = indexer_score_reduce(&scores, &w_all, Some(&counts))?;
    // One batched argsort (descending) + one batched gather of the top-max_k
    // shortlist-relative ids through each session's `sl` → absolute entry ids.
    let order = weighted
        .arg_sort_last_dim(false)?
        .narrow(1, 0, max_k)?
        .contiguous()?; // [big,max_k]
    let sl_pad: Vec<Tensor> = active
        .iter()
        .map(|a| {
            if a.m_s == mm {
                Ok(a.sl.clone())
            } else {
                let pad = Tensor::zeros(mm - a.m_s, DType::U32, &dev)?;
                Tensor::cat(&[&a.sl, &pad], 0)
            }
        })
        .collect::<Result<_>>()?;
    let sl_all = Tensor::stack(&sl_pad, 0)?; // [big, mm]
    let gids_all = sl_all.gather(&order, 1)?; // [big, max_k] absolute ids, score order

    // Phase C — hand back each session's first `k` (varies with shortlist size).
    for (i, a) in active.iter().enumerate() {
        let gids = gids_all
            .narrow(0, i, 1)?
            .reshape(max_k)?
            .narrow(0, 0, a.k)?
            .contiguous()?;
        out_slots[a.s] = Some((gids, a.k));
    }
    Ok(out_slots.into_iter().map(|o| o.expect("filled")).collect())
}
