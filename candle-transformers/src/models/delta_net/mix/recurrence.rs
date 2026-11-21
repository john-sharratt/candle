//! The gated delta rule over a sequence, in the sequential form that serves as
//! the reference and the chunked form that serves prefill.
//!
//! The sequential step is the numerical reference, and the chunked form is tested
//! against it at every chunk width. The chunked form advances the state in place
//! in its stored orientation, so no transposed copy is made. Decay factors are
//! clamped before the exponential, because the discarded upper triangle of the
//! decay matrix would otherwise overflow to infinity, and the mask would turn
//! that into NaN and poison every later token. The triangular system is solved
//! rather than inverted, and substitution rather than powering keeps the logits
//! exact.

use super::*;

/// One token of the gated delta rule across all V heads.
///
/// `state` `[H, d_v, d_k]`; `q`/`k` `[H, d_k]` (already broadcast to V heads
/// and l2-normed); `v` `[H, d_v]`; `g_log`/`beta` `[H]`. Returns
/// `(o [H, d_v], new_state)`.
pub fn delta_step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_log: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (h, _d_v, _d_k) = state.dims3()?;
    let decay = g_log.exp()?.reshape((h, 1, 1))?;
    let decayed = state.broadcast_mul(&decay)?;
    // The value the decayed state currently predicts for k: S k → [H, d_v].
    let pred = decayed.matmul(&k.unsqueeze(2)?)?.squeeze(2)?;
    // β-scaled correction, written back along k.
    let err = v.sub(&pred)?.broadcast_mul(&beta.reshape((h, 1))?)?;
    let outer = err.unsqueeze(2)?.matmul(&k.unsqueeze(1)?)?;
    let new_state = decayed.add(&outer)?;
    // Read with the post-update state: the token attends to itself.
    let o = new_state.matmul(&q.unsqueeze(2)?)?.squeeze(2)?;
    Ok((o, new_state))
}

/// The recurrence over a whole segment.
///
/// `q`/`k` `[T, H, d_k]`, `v` `[T, H, d_v]`, `g_log`/`beta` `[T, H]`; the
/// entering state is consumed and the post-segment state returned, so calling
/// this over `[0..a)` then `[a..T)` is identical to one call over `[0..T)` —
/// the property turn sealing and resume rest on.
pub fn delta_recurrence(
    state: Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_log: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (t, h, d_v) = v.dims3()?;
    let mut s = state;
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let (o, s_next) = delta_step(
            &s,
            &q.get(i)?,
            &k.get(i)?,
            &v.get(i)?,
            &g_log.get(i)?,
            &beta.get(i)?,
        )?;
        outs.push(o.reshape((1, h, d_v))?);
        s = s_next;
    }
    Ok((Tensor::cat(&outs, 0)?, s))
}

/// The chunked (parallel-scan) form of [`delta_recurrence`] — the prefill
/// path. Mathematically identical to the sequential rule; the win is that a
/// chunk of `C` tokens costs a handful of `[C, d]`-shaped batched matmuls
/// plus ONE sequential state carry per chunk instead of `C` serial steps.
/// Ported from llama.cpp's `delta-net-base.cpp` chunking graph (the external
/// oracle); the sequential reference above is the in-repo oracle both are
/// tested against.
///
/// Per chunk, with entering state `S [d_v, d_k]` per head (`o = S q`
/// orientation) and within-chunk index `i`:
///
/// ```text
///   G[i]        = Σ_{j ≤ i} g[j]                       (log-decay cumsum)
///   D[i][j]     = exp(G[i] − G[j])   for j ≤ i         (decay mask)
///   A[i][j]     = (βk[i] · k[j]) D[i][j]  for j < i    (strictly lower)
///   T           = (I + A)^{-1}                          (unit lower-tri)
///   u           = T (β v)                               (pseudo-values)
///   kcd         = T (βk ⊙ e^G)
///   v_new       = u − kcd Sᵀ                            (chunk-local writes)
///   o[i]        = S (q[i] e^{G[i]})  +  Σ_{j ≤ i} (q[i]·k[j]) D[i][j] v_new[j]
///   S_next      = e^{G[C−1]} S + v_newᵀ (k ⊙ e^{G[C−1] − G})
/// ```
///
/// Shapes as in [`delta_recurrence`]; `chunk` is the chunk width (64 in the
/// production configuration, any positive value here — the tail chunk is
/// simply shorter, no padding).
pub fn delta_chunked<'w>(
    state: &mut Tensor,
    q: &LiveTensor<'w>,
    k: &LiveTensor<'w>,
    v: &LiveTensor<'w>,
    g_log: &LiveTensor<'w>,
    beta: &LiveTensor<'w>,
    chunk: usize,
) -> Result<LiveTensor<'w>> {
    let (t, h, d_k) = q.dims3()?;
    let d_v = v.dim(2)?;
    if chunk == 0 {
        candle::bail!("delta_chunked: chunk width must be positive");
    }
    let dev = q.device();

    // **The state is a buffer, not a value.** It is allocated once per sequence
    // and lives for that sequence's whole life, so every chunk writes *into* it
    // rather than producing a replacement. What that removes is not one copy
    // but four: a transpose into FLA orientation on the way in, two fresh
    // `[H, d, d]` allocations per chunk for the carry, and a transpose back on
    // the way out — ~16 MB of traffic per layer on the 35B, for memory that
    // never needed to move.
    //
    // The reads want `Sᵀ` and the store holds `S`, but a transpose is a *view*;
    // only materialising it cost anything. The carry runs in the stored
    // orientation instead — transposing `S ← e^g·S + Kᵀ V` gives
    // `S ← e^g·S + Vᵀ K`, which is two in-place ops on the buffer.
    let mut outs: Vec<LiveTensor<'w>> = Vec::with_capacity(t.div_ceil(chunk));

    // **Head-major once, not once per chunk.** The chunk loop wants `[H, c, d]`
    // and the inputs arrive `[T, H, d]`, so the transpose used to sit inside
    // the loop — five `transpose + contiguous` copies per chunk, eleven chunks
    // deep on a 649-token prefill, for a permutation that does not depend on
    // the chunk at all. Transposed once here, a chunk is a `narrow` on the
    // token axis of an already-contiguous buffer.
    let qh = q.transpose(0, 1)?.contiguous()?; // [H, T, d_k]
    let kh = k.transpose(0, 1)?.contiguous()?;
    let vh = v.transpose(0, 1)?.contiguous()?;
    let gh = g_log.transpose(0, 1)?.contiguous()?; // [H, T]
    let bh = beta.transpose(0, 1)?.contiguous()?;

    // The triangular masks depend only on the chunk width, and every chunk but
    // the last shares one. Built here and reused; the tail rebuilds its own.
    let mut mask_width = 0usize;
    let mut masks: Option<(Tensor, Tensor)> = None;

    let mut start = 0usize;
    while start < t {
        let c = chunk.min(t - start);
        // Chunk slices, head-major: [H, c, d] / [H, c].
        // Narrowing the token axis of a head-major buffer leaves a gap between
        // heads, and both `matmul` and `cumsum` (which lowers to one) require
        // contiguity — so the slice is materialised. That is the same single
        // copy the per-chunk `transpose` used to pay for, minus the transpose.
        let qc = qh.narrow(1, start, c)?.contiguous()?;
        let kc = kh.narrow(1, start, c)?.contiguous()?;
        let vc = vh.narrow(1, start, c)?.contiguous()?;
        let gc = gh.narrow(1, start, c)?.contiguous()?; // [H, c]
        let bc = bh.narrow(1, start, c)?.contiguous()?;
        if mask_width != c {
            masks = Some((
                lower_tri_mask(c, false, dev)?,
                lower_tri_mask(c, true, dev)?,
            ));
            mask_width = c;
        }
        let (mask_incl, mask_strict) = masks.as_ref().expect("built above");
        let rows = chunk_rows(state, &qc, &kc, &vc, &gc, &bc, mask_incl, mask_strict)?;
        outs.push(rows.o.transpose(0, 1)?.contiguous()?); // [c, H, d_v]

        // State carry, in place and in the stored orientation:
        //   S ← e^{G[c−1]}·S + V_newᵀ (K ⊙ e^{G[c−1]−G})
        let g_last = rows.g_cs.narrow(1, c - 1, 1)?; // [H, 1]
        let g_diff_exp = g_last.broadcast_sub(&rows.g_cs)?.exp()?; // e^{G_last − G[i]}
        let kg = kc.broadcast_mul(&g_diff_exp.unsqueeze(2)?)?; // [H, c, d_k]
        let decay_full = g_last
            .exp()?
            .unsqueeze(2)?
            .broadcast_as((h, d_v, d_k))?
            .contiguous()?;
        state.mul_mut(&decay_full)?;
        state.add_mut(&rows.v_new.transpose(1, 2)?.contiguous()?.matmul(&kg)?)?;

        start += c;
    }

    LiveTensor::cat(&outs, 0) // [T, H, d_v]
}

/// One chunk's row quantities against its entering state, which is read and
/// not advanced: the output, the inclusive log-decay scan `G` and the
/// chunk-local writes `v_new` — what [`delta_chunked`] carries the state with,
/// and what a short span keeps as its timeline instead.
pub(super) struct ChunkRows<'w> {
    /// `[H, c, d_v]`.
    pub(super) o: LiveTensor<'w>,
    /// `[H, c]`.
    pub(super) g_cs: LiveTensor<'w>,
    /// `[H, c, d_v]`.
    pub(super) v_new: LiveTensor<'w>,
}

/// [`ChunkRows`] for one head-major chunk: `qc`/`kc` `[H, c, d_k]`, `vc`
/// `[H, c, d_v]`, `gc`/`bc` `[H, c]`, against `state` `[H, d_v, d_k]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunk_rows<'w>(
    state: &Tensor,
    qc: &LiveTensor<'w>,
    kc: &LiveTensor<'w>,
    vc: &LiveTensor<'w>,
    gc: &LiveTensor<'w>,
    bc: &LiveTensor<'w>,
    mask_incl: &Tensor,
    mask_strict: &Tensor,
) -> Result<ChunkRows<'w>> {
    let d_k = qc.dim(2)?;
    let d_v = vc.dim(2)?;
    // Inclusive log-decay cumsum and its exp.
    let g_cs = gc.cumsum(1)?; // [H, c]
    let g_exp = g_cs.exp()?; // e^{G[i]}

    let bc3 = bc.unsqueeze(2)?; // [H, c, 1]
    let k_b = kc.broadcast_mul(&bc3)?;
    let v_b = vc.broadcast_mul(&bc3)?;

    // Decay mask D[i][j] = exp(G[i] − G[j]), lower incl. diagonal.
    //
    // **The clamp is load-bearing, not defensive.** `G` is a cumulative sum
    // of log-decays and every `g` is negative, so `G` decreases: for the
    // `j ≤ i` half this exponent is ≤ 0 and `exp` is a contraction. The
    // upper half is the mirror image — `G[i] − G[j]` grows *positive* with
    // the distance — and although the mask discards it, the mask is applied
    // to the result. `exp` runs first, overflows to `+inf` once the gap
    // passes ~88 in f32, and `inf × 0` is `NaN`, which then propagates
    // through `unit_lower_inverse` into every later token.
    //
    // It is depth- and content-dependent, so short probes miss it entirely:
    // random activations over 40 tokens stay finite, while real prompt
    // embeddings poisoned the row at token 23 of one prompt and 27 of
    // another. Clamping the exponent at 0 leaves the kept half untouched
    // (it is already ≤ 0) and makes the discarded half `exp(0) = 1` before
    // the mask zeroes it.
    let gi = g_cs.unsqueeze(2)?; // [H, c, 1]
    let gj = g_cs.unsqueeze(1)?; // [H, 1, c]
    let decay = gi
        .broadcast_sub(&gj)?
        .minimum(0f64)?
        .exp()?
        .broadcast_mul(&mask_incl.unsqueeze(0)?)?; // [H, c, c]

    // A[i][j] = (βk[i]·k[j]) D[i][j], strictly lower.
    let a = k_b
        .matmul(&kc.transpose(1, 2)?)?
        .mul(&decay)?
        .broadcast_mul(&mask_strict.unsqueeze(0)?)?;

    // Pseudo-values and the cumulative-decay key read: `u = T(βv)` and
    // `kcd = T(βk ⊙ e^G)` with `T = (I + A)⁻¹`.
    //
    // **Solved, not inverted.** Both share the left side `I + A`, so they
    // are one triangular solve with the two right-hand sides side by side —
    // and a solve never forms `T`. The alternative, and what the reference
    // below still does, is a recursive block inversion costing `log₂ c`
    // levels of a handful of launches each: about eighty serially dependent
    // ops per chunk, which measured as the single largest op-count item in
    // prefill. `cublasStrsmBatched` is one launch and is backward-stable.
    let kg_scaled = k_b.broadcast_mul(&g_exp.unsqueeze(2)?)?;
    let (u, kcd) = solve_pseudo_values(&a, &v_b, &kg_scaled, d_v, d_k)?;

    // Chunk-local writes: what each token adds beyond the entering state.
    // `Sᵀ` as a view of the buffer — free, because a transpose is a layout,
    // not a copy.
    let s_fla = state.transpose(1, 2)?.contiguous()?;
    let v_new = u.sub(&kcd.matmul(&s_fla)?)?; // [H, c, d_v]

    // Output: inter-chunk read of the entering state + intra-chunk reads
    // of this chunk's own writes (inclusive diagonal — the token sees its
    // own update, matching the sequential rule's post-update read).
    let q_g = qc.broadcast_mul(&g_exp.unsqueeze(2)?)?;
    let inter = q_g.matmul(&s_fla)?; // [H, c, d_v]
    let kq = qc.matmul(&kc.transpose(1, 2)?)?.mul(&decay)?; // q[i]·k[j] D[i][j], j ≤ i
    let intra = kq.matmul(&v_new)?;
    let o = inter.add(&intra)?; // [H, c, d_v]
    Ok(ChunkRows { o, g_cs, v_new })
}

/// A short span's mixer rows against `state`, which is read and not advanced:
/// its output `[T, H, d_v]`, and its `G` `[T, H]` and `v_new` `[T, H, d_v]` for
/// the timeline. One chunk — a short span is at most [`TIMELINE_ROWS`] rows.
/// Shapes as in [`delta_chunked`].
pub(super) fn delta_span_rows<'w>(
    state: &Tensor,
    q: &LiveTensor<'w>,
    k: &LiveTensor<'w>,
    v: &LiveTensor<'w>,
    g_log: &LiveTensor<'w>,
    beta: &LiveTensor<'w>,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>, LiveTensor<'w>)> {
    let t = q.dim(0)?;
    if t == 0 || t > TIMELINE_ROWS {
        candle::bail!("delta_span_rows: a short span is 1..={TIMELINE_ROWS} rows, got {t}");
    }
    let dev = q.device();
    let head_major = |x: &LiveTensor<'w>| x.transpose(0, 1)?.contiguous();
    let rows = chunk_rows(
        state,
        &head_major(q)?,
        &head_major(k)?,
        &head_major(v)?,
        &head_major(g_log)?,
        &head_major(beta)?,
        &lower_tri_mask(t, false, dev)?,
        &lower_tri_mask(t, true, dev)?,
    )?;
    Ok((
        rows.o.transpose(0, 1)?.contiguous()?,
        rows.g_cs.transpose(0, 1)?.contiguous()?,
        rows.v_new.transpose(0, 1)?.contiguous()?,
    ))
}

/// `u = T·v_b` and `kcd = T·kg` for `T = (I + A)⁻¹`, `A` strictly lower.
///
/// One triangular solve over the two right-hand sides concatenated, because
/// they share the left side. On CUDA in F32 that is a single
/// `cublasStrsmBatched`; elsewhere — CPU tests, the F32 reference, any other
/// dtype — it falls back to forming `T` with [`unit_lower_inverse`] and
/// multiplying, which is the same arithmetic and is what the solve is checked
/// against (`solve_matches_the_explicit_inverse`).
// `d_v`/`d_k` split the concatenated right-hand side that only the CUDA arm
// forms; the fallback multiplies `v_b` and `kg` separately and reads their own
// shapes.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(super) fn solve_pseudo_values<'w>(
    a: &LiveTensor<'w>,
    v_b: &LiveTensor<'w>,
    kg: &LiveTensor<'w>,
    d_v: usize,
    d_k: usize,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    #[cfg(feature = "cuda")]
    if a.device().is_cuda() && a.dtype() == DType::F32 {
        let rhs = LiveTensor::cat(&[v_b, kg], 2)?; // [H, c, d_v + d_k]
        let x = super::super::cuda::solve_unit_lower(a, rhs)?;
        // Both halves are strided views of one buffer — a narrow on the last
        // axis leaves a gap between rows. `kcd` is a matmul operand and must be
        // materialised; `u` is only ever the left side of a subtraction, which
        // reads strides, so it stays a view.
        return Ok((x.narrow(2, 0, d_v)?, x.narrow(2, d_v, d_k)?.contiguous()?));
    }
    let t_inv = unit_lower_inverse(a)?;
    Ok((t_inv.matmul(v_b)?, t_inv.matmul(kg)?))
}

/// [`lower_tri_mask`]'s cache, keyed by `(device location, chunk width, strict)`.
pub(super) type MaskCache = Mutex<HashMap<(DeviceLocation, usize, bool), Tensor>>;

/// `[c, c]` mask with 1 where `j ≤ i` (or `j < i` when `strict`), else 0.
///
/// **Built once per `(device, c, strict)` and kept.** The mask is a pure
/// function of its arguments — it depends on nothing the wave carries — so
/// rebuilding it is not a temporary, it is recomputing a constant.
///
/// The call site already avoided rebuilding it per *chunk*; what it could not
/// see is that it is rebuilt per layer, per wave. Measured by the
/// forbidden-allocation detector on the 3.6-35B gate: **120 calls, 41.3 MB per
/// config, the single largest source of device allocations made outside the
/// reservation.** Those allocations come from the free space the span
/// deliberately does not cover, which is the memory a load-time repack was
/// found competing for.
///
/// Keyed by device ordinal as well as shape: two devices need two masks, and a
/// mask handed to the wrong one is an illegal address rather than a wrong
/// answer. Bounded by construction — a model has one chunk width and two
/// strictness values, so the map holds a handful of entries a few KiB each for
/// the life of the process.
pub(super) fn lower_tri_mask(c: usize, strict: bool, dev: &Device) -> Result<Tensor> {
    static MASKS: OnceLock<MaskCache> = OnceLock::new();

    // Keyed by the device's own location, not by a CUDA ordinal with one
    // sentinel for "everything else" — that collapsed CPU and Metal onto a
    // single entry, so a CPU mask could be handed to a Metal caller.
    let key = (dev.location(), c, strict);
    let cache = MASKS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock() {
        if let Some(m) = map.get(&key) {
            return Ok(m.clone());
        }
    }
    // Built on the device from two index vectors rather than as a host `Vec`:
    // the host round trip costs more than the comparison it is avoiding.
    let rows = Tensor::arange(0u32, c as u32, dev)?.reshape((c, 1))?;
    let cols = Tensor::arange(0u32, c as u32, dev)?.reshape((1, c))?;
    let keep = if strict {
        cols.broadcast_lt(&rows)?
    } else {
        cols.broadcast_le(&rows)?
    };
    let mask = keep.to_dtype(DType::F32)?;
    if let Ok(mut map) = cache.lock() {
        map.insert(key, mask.clone());
    }
    Ok(mask)
}

/// Invert `I + A` for a strictly-lower-triangular `A [H, c, c]` by forward
/// substitution: row `i` of the inverse is `e_i − Σ_{j<i} A[i][j] · row j`.
/// `c` sequential steps of batched `[H, 1, c] × [H, c, c]` work — fine for a
/// reference; the CUDA kernel does this in shared memory.
/// [`unit_lower_inverse`], for the CUDA solve's parity test.
///
/// The inverse is the reference form the `trsm` path is checked against, and
/// that check lives beside the kernel in [`super::cuda`] rather than here.
pub fn unit_lower_inverse_for_test<'w>(a: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    unit_lower_inverse(a)
}

pub(super) fn unit_lower_inverse<'w>(a: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let (_h, c, _) = a.dims3()?;
    let dev = a.device();
    if c == 0 {
        return Ok(a.clone());
    }
    // Forward substitution, one row at a time: `T[i] = e_i − A[i, <i] · T[<i]`.
    //
    // **Not the Neumann series by repeated squaring.** `A` is nilpotent so
    // `Σ (−A)^k` is exact in exact arithmetic and reaches `c` terms in `log₂ c`
    // matmuls, which is tempting — it made prefill 3× faster here. But it forms
    // `B^{c/2}` explicitly, and a strictly lower triangular matrix with entries
    // near 1 grows about `C(c, c/2)` under powering: at `c = 64` the logits had
    // already lost three digits against this form (cosine 0.99 at 32 tokens,
    // 0.81 at 64) and at `c = 256` they came back non-finite. Substitution
    // forms no powers and stays exact.
    //
    // Done by halves rather than by rows. Writing `I + A` in 2×2 blocks,
    //
    //   ⎡P 0⎤⁻¹   ⎡  P⁻¹      0 ⎤
    //   ⎣C Q⎦   = ⎣−Q⁻¹ C P⁻¹  Q⁻¹⎦
    //
    // and `P`, `Q` are themselves unit lower triangular, so the same rule
    // applies to them. **The two sub-problems are independent**, so they are
    // stacked into the batch axis and solved by one recursive call: the depth
    // is `log₂ c` levels of a handful of launches each, against `c` rounds of
    // row substitution. Total element count halves every level, so the extra
    // batch costs nothing.
    //
    // Numerically this *is* substitution — every quantity is a product of
    // sub-inverses with original blocks, and no power of `A` is ever formed.
    if c == 1 {
        // A 1×1 unit lower triangular block is `[1]`, and so is its inverse.
        return Tensor::ones((_h, 1, 1), a.dtype(), dev);
    }
    let half = c / 2;
    if c % 2 != 0 {
        // Odd widths only arise from a ragged tail; peel one row the direct
        // way rather than carry a padding case through the recursion.
        let head = unit_lower_inverse(&a.narrow(1, 0, c - 1)?.narrow(2, 0, c - 1)?.contiguous()?)?;
        let a_row = a.narrow(1, c - 1, 1)?.narrow(2, 0, c - 1)?.contiguous()?;
        let last = a_row.matmul(&head)?.neg()?; // [H, 1, c-1]
        let one = Tensor::ones((_h, 1, 1), a.dtype(), dev)?;
        let zeros = Tensor::zeros((_h, c - 1, 1), a.dtype(), dev)?;
        return LiveTensor::cat(
            &[
                LiveTensor::cat(&[&head, &zeros], 2)?,
                LiveTensor::cat(&[&last, &one], 2)?,
            ],
            1,
        );
    }
    let p = a.narrow(1, 0, half)?.narrow(2, 0, half)?;
    let q = a.narrow(1, half, half)?.narrow(2, half, half)?;
    let cblk = a.narrow(1, half, half)?.narrow(2, 0, half)?.contiguous()?;
    // One call for both diagonal blocks.
    let stacked = LiveTensor::cat(&[&p.contiguous()?, &q.contiguous()?], 0)?;
    let t_stack = unit_lower_inverse(&stacked)?;
    let t_p = t_stack.narrow(0, 0, _h)?;
    let t_q = t_stack.narrow(0, _h, _h)?;
    let corner = t_q.matmul(&cblk.matmul(&t_p)?)?.neg()?;
    let zeros = Tensor::zeros((_h, half, half), a.dtype(), dev)?;
    LiveTensor::cat(
        &[
            LiveTensor::cat(&[&t_p, &zeros], 2)?,
            LiveTensor::cat(&[&corner, &t_q], 2)?,
        ],
        1,
    )
}
