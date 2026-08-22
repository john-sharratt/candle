//! KV `Compressor`: learned gated pooling of consecutive tokens into compressed KV
//! entries. Mirrors the `start_pos == 0` (prefill) branch of `Compressor.forward` in
//! `inference/model.py`.
//!
//! The reference model keeps per-session incremental state so decode can emit one
//! compressed entry every `ratio` tokens; this reference implementation instead
//! recomputes the compressed entries from the full prefix on every step, which is
//! numerically identical for the complete groups and avoids all cache-state handling.
//! Only complete groups produce entries (the trailing `seq % ratio` tokens are carried
//! by the sliding window, exactly as in the reference).
//!
//! With `ratio == 4` the compressor is **overlapping**: each entry pools `2·ratio`
//! rows — the current group of `ratio` (from the second half of the projection) plus the
//! previous group of `ratio` (from the first half) — jointly softmaxed. Larger ratios
//! (HCA's 128) are non-overlapping.
//!
//! FP8/FP4 fake-quantization of the entries (the QAT-matched storage precision) is the
//! P7 layer; this reference keeps entries in full precision, which is strictly within
//! the QAT tolerance (the same choice vLLM's BF16 fallback makes).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;

use super::linear::{shared_int8_pair, QLinear};
use super::rope::RotaryCache;

/// One compressor instance (attention-side or indexer-side). `head_dim` is the width of
/// the compressed entry (`d`); the projections emit `coff·d` where `coff = 2` for the
/// overlapping `ratio == 4` case and `1` otherwise.
#[derive(Debug, Clone)]
pub struct Compressor {
    wkv: QLinear,   // [coff*d, dim] — int8-KO on the engine path
    wgate: QLinear, // [coff*d, dim] — int8-KO on the engine path
    ape: Tensor,    // [ratio, coff*d] — additive positional bias (NOT matmul'd; stays dense)
    norm_w: Tensor, // [d]
    ratio: usize,
    head_dim: usize,
    rope_head_dim: usize,
    overlap: bool,
    eps: f64,
}

impl Compressor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wkv: impl Into<QLinear>,
        wgate: impl Into<QLinear>,
        ape: Tensor,
        norm_w: Tensor,
        ratio: usize,
        head_dim: usize,
        rope_head_dim: usize,
        eps: f64,
    ) -> Self {
        Self {
            wkv: wkv.into(),
            wgate: wgate.into(),
            ape,
            norm_w,
            ratio,
            head_dim,
            rope_head_dim,
            overlap: ratio == 4,
            eps,
        }
    }

    fn coff(&self) -> usize {
        if self.overlap {
            2
        } else {
            1
        }
    }

    /// Gated pooling of `x` `[b, s, dim]` into pre-norm compressed entries
    /// `[b, groups, d]` (`groups = s / ratio`). Returns `None` when `s < ratio`.
    pub fn pool(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let (b, s, _dim) = x.dims3()?;
        if s < self.ratio {
            return Ok(None);
        }
        let d = self.head_dim;
        let r = self.ratio;
        let cd = self.coff() * d;
        let groups = s / r;
        let cutoff = groups * r;

        let x = x.to_dtype(DType::F32)?;
        // wkv and wgate share the activation `x`; quantize it once for both projections.
        let (kv, score) = shared_int8_pair(&x, &self.wkv, &self.wgate)?;

        let kv = kv.narrow(1, 0, cutoff)?.reshape((b, groups, r, cd))?;
        let ape = self.ape.to_dtype(DType::F32)?.reshape((1, 1, r, cd))?;
        let score = score
            .narrow(1, 0, cutoff)?
            .reshape((b, groups, r, cd))?
            .broadcast_add(&ape)?;

        let (kv_p, score_p) = if self.overlap {
            (
                self.overlap_transform(&kv, b, groups, d, 0.0)?,
                self.overlap_transform(&score, b, groups, d, f32::NEG_INFINITY)?,
            )
        } else {
            (kv, score)
        };
        // softmax over the pooling axis (dim 2) then weighted sum.
        let w = softmax(&score_p, 2)?;
        let entry = kv_p.broadcast_mul(&w)?.sum(2)?; // [b, groups, d]
        Ok(Some(entry))
    }

    /// `overlap_transform`: reshape a `[b, groups, ratio, 2d]` projection into
    /// `[b, groups, 2·ratio, d]`, where the first `ratio` rows are the *previous* group's
    /// first-half dims (group 0 filled with `fill`) and the last `ratio` rows are the
    /// current group's second-half dims.
    fn overlap_transform(
        &self,
        t: &Tensor,
        b: usize,
        groups: usize,
        d: usize,
        fill: f32,
    ) -> Result<Tensor> {
        let dev = t.device();
        let curr = t.narrow(D::Minus1, d, d)?; // [b,groups,ratio,d]
        let prev_src = t.narrow(D::Minus1, 0, d)?; // [b,groups,ratio,d]
        let r = self.ratio;
        let pad = Tensor::full(fill, (b, 1, r, d), dev)?;
        // shift down one group: prev[g] = prev_src[g-1], prev[0] = pad
        let prev = if groups > 1 {
            Tensor::cat(&[&pad, &prev_src.narrow(1, 0, groups - 1)?], 1)?
        } else {
            pad
        };
        Tensor::cat(&[&prev, &curr], 2)
    }

    /// Full compressor forward: pool → RMSNorm → RoPE on the trailing `rope_head_dim`
    /// dims at group-start positions. Returns `None` when `seq < ratio`.
    pub fn forward(&self, x: &Tensor, rope: &RotaryCache) -> Result<Option<Tensor>> {
        let entry = match self.pool(x)? {
            None => return Ok(None),
            Some(e) => e,
        };
        let entry = self.rms_norm(&entry)?;
        let groups = entry.dim(1)?;
        let rd = self.rope_head_dim;
        let d = self.head_dim;
        // group-start positions: 0, ratio, 2*ratio, ...
        let positions: Vec<u32> = (0..groups).map(|g| (g * self.ratio) as u32).collect();

        let nope = entry.narrow(D::Minus1, 0, d - rd)?;
        let rope_part = entry.narrow(D::Minus1, d - rd, rd)?;
        let rope_part = rope.apply_positions(&rope_part, &positions, false)?;
        Ok(Some(Tensor::cat(&[&nope, &rope_part], D::Minus1)?))
    }

    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        // Single fused `rms_norm` launch (`x·rsqrt(mean(x²)+eps)·w`) instead of the
        // eager sqr/mean/add/sqrt/div/mul chain. `to_dtype(F32)` on an already-F32
        // `norm_w` is a no-op.
        let x = x.to_dtype(DType::F32)?;
        let w = self.norm_w.to_dtype(DType::F32)?;
        candle_nn::ops::rms_norm(&x, &w, self.eps as f32)
    }

    /// Pool the assembled group inputs into normed (and optionally roped) corpus
    /// entries — the stateless second half of the compressor emit, factored out so
    /// MANY sequences' [`GroupPool`]s can be concatenated on the group axis and
    /// pooled in ONE launch (the prefill wave batches the whole prompt fleet's
    /// pools this way). `pool_kv`/`pool_score` are `[G, P, d]` (`P = 2r` overlap /
    /// `r` non-overlap); `positions[g]` is group `g`'s start position for the RoPE.
    /// Bit-identical, per group, to pooling each sequence's groups separately (the
    /// softmax/weighted-sum/RMSNorm/RoPE are all per-group-independent).
    pub fn pool_and_norm(
        &self,
        pool_kv: &Tensor,
        pool_score: &Tensor,
        positions: &[u32],
        rope: Option<&RotaryCache>,
    ) -> Result<Tensor> {
        let g = pool_kv.dim(0)?;
        let d = pool_kv.dim(2)?;
        // Pool over the group axis (P), then RMSNorm over the entries.
        let entry = pool_group(pool_kv, pool_score)?; // [G, d]
        let entry = self.rms_norm(&entry.reshape((1, g, d))?)?; // [1, G, d]
        let entry = match rope {
            Some(rope) => {
                let rd = self.rope_head_dim;
                let nope = entry.narrow(D::Minus1, 0, d - rd)?;
                let rope_part = entry.narrow(D::Minus1, d - rd, rd)?;
                let rope_part = rope.apply_positions(&rope_part, positions, false)?;
                Tensor::cat(&[&nope, &rope_part], D::Minus1)?
            }
            None => entry,
        };
        entry.reshape((g, d))
    }

    /// The number of compressed entries produced for a prefix of length `seq`.
    pub fn num_entries(seq: usize, ratio: usize) -> usize {
        seq / ratio
    }

    pub fn device(&self) -> &Device {
        // `norm_w` stays a dense `Tensor`, so it yields a borrowed `&Device` (the KO `QLinear`
        // would only return an owned `Device`); all params are co-located anyway.
        self.norm_w.device()
    }

    /// The projected KV/score width `coff·d` (2·d for the overlapping `ratio == 4`
    /// compressor, `d` otherwise).
    fn cd(&self) -> usize {
        self.coff() * self.head_dim
    }

    /// Project one token `x` `[dim]` / `[1, dim]` / `[1, 1, dim]` into its raw (pre-`ape`,
    /// pre-pool) `kv` and `score` rows, each `[1, coff·d]` in F32 — the streaming form of the
    /// per-token `x·wkvᵀ` / `x·wgateᵀ` that `pool` computes for the whole prefix at once.
    fn project_row(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.reshape((1, self.wkv.in_dim()))?.to_dtype(DType::F32)?;
        let (kv, score) = shared_int8_pair(&x, &self.wkv, &self.wgate)?;
        Ok((kv, score))
    }

    /// Batched [`Self::project_row`]: project a whole prefix `xs` (`[n, dim]` /
    /// `[1, n, dim]`) into `(kv, score)`, each `[n, coff·d]` in F32, in ONE GEMM
    /// per projection instead of `n` per-row GEMVs. The rows are the exact
    /// per-token `project_row` outputs (`x·wkvᵀ` / `x·wgateᵀ`), so feeding row
    /// `t` into [`IncrementalCompressor::push_projected`] is bit-identical to
    /// `push_raw` on token `t` — this is the prefill fast path that hoists the
    /// per-token projection out of the token loop.
    pub fn project_rows(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = xs.elem_count() / self.wkv.in_dim();
        let xf = xs.reshape((n, self.wkv.in_dim()))?.to_dtype(DType::F32)?;
        let (kv, score) = shared_int8_pair(&xf, &self.wkv, &self.wgate)?;
        Ok((kv, score))
    }

    /// Build the incremental (decode) form of this compressor: a stateful streamer that
    /// accepts one token per `push` and emits ONE compressed entry every `ratio`-th token,
    /// bit-for-bit identical to the entry `forward`/`pool` produces for the same group over
    /// the full prefix. See [`IncrementalCompressor`].
    pub fn incremental(&self) -> IncrementalCompressor {
        IncrementalCompressor {
            c: self.clone(),
            kv_rows: Vec::with_capacity(self.ratio),
            score_rows: Vec::with_capacity(self.ratio),
            prev_kv_group: None,
            prev_score_group: None,
            group_idx: 0,
        }
    }
}

/// The compressor pool as an **online-softmax (LSE) monoid** — the per-channel
/// accumulator `(m, l, acc)` over a group's pooling rows (§C). It is the *same*
/// primitive as the attention split-KV combine: a group's compressed entry is
/// `acc / l` (pre-RMSNorm), and the fold is associative, so a group's rows may
/// be cut anywhere — across a turn seam — and re-merged exactly. The persisted
/// unit for a straddling group is one of these partials; at the seam the
/// boundary tokens' fresh rows fold in and the group finalizes, with no
/// re-prefill of the interior.
///
/// Per channel `c ∈ [0, head_dim)`:
/// `m_c = max_t s_t[c]`, `l_c = Σ_t e^{s_t[c]−m_c}`, `acc_c = Σ_t e^{s_t[c]−m_c}·kv_t[c]`.
#[derive(Clone)]
#[allow(dead_code)] // LSE-merge persistence monoid (docs/deepseek_batched_paged_attention_plan.md); wired in a later phase
pub struct GroupPartial {
    m: Tensor,   // [d] running per-channel max score
    l: Tensor,   // [d] running per-channel Σ exp
    acc: Tensor, // [d] running per-channel Σ exp·kv
}

#[allow(dead_code)]
impl GroupPartial {
    /// The monoid identity `(−∞, 0, 0)` — the empty fold.
    pub fn identity(d: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            m: Tensor::full(f32::NEG_INFINITY, d, device)?,
            l: Tensor::zeros(d, DType::F32, device)?,
            acc: Tensor::zeros(d, DType::F32, device)?,
        })
    }

    /// Fold `n` pooling rows into this partial. `scores`/`kvs` are `[n, d]`
    /// (already `score + ape`, and — for the overlapping compressor — already
    /// split to the `d`-wide pooling half). Order-independent.
    pub fn fold(&self, scores: &Tensor, kvs: &Tensor) -> Result<Self> {
        let (_n, d) = scores.dims2()?;
        // Local partial for the incoming rows, then LSE-merge with self.
        let m_local = scores.max(0)?; // [d]
        let shifted = scores.broadcast_sub(&m_local)?.exp()?; // [n, d]
        let l_local = shifted.sum(0)?; // [d]
        let acc_local = shifted.broadcast_mul(kvs)?.sum(0)?; // [d]
        let local = Self {
            m: m_local,
            l: l_local,
            acc: acc_local,
        };
        let _ = d;
        self.merge(&local)
    }

    /// LSE-merge two partials of the SAME group (associative + commutative).
    pub fn merge(&self, other: &Self) -> Result<Self> {
        let m = self.m.maximum(&other.m)?; // [d]
        let a = self.m.broadcast_sub(&m)?.exp()?; // e^{m_self − m}
        let b = other.m.broadcast_sub(&m)?.exp()?; // e^{m_other − m}
                                                   // NaN guard: −inf − −inf → NaN in the identity case; e^{−inf} is 0, but
                                                   // the subtraction NaN survives, so zero it where both maxes are −inf.
        let a = replace_nan(&a, 0.0)?;
        let b = replace_nan(&b, 0.0)?;
        let l = ((&self.l * &a)? + (&other.l * &b)?)?;
        let acc = ((self.acc.broadcast_mul(&a))? + (other.acc.broadcast_mul(&b))?)?;
        Ok(Self { m, l, acc })
    }

    /// Finalize the completed group: `acc / l` → the pre-RoPE, pre-RMSNorm
    /// pooled entry `[d]`.
    pub fn finalize(&self) -> Result<Tensor> {
        &self.acc / &self.l
    }
}

/// `out[g, k] = Σ_p softmax_p(score[g, :, k]) · kv[g, p, k]` — the pooling half
/// of [`Compressor::pool_and_norm`], where `pool_kv`/`pool_score` are
/// `[G, P, d]`.
///
/// On CUDA this is ONE launch. Eagerly it is seven, because candle's `softmax`
/// is unfused when the reduction is over a non-last dim (`max_keepdim`,
/// `broadcast_sub`, `exp`, `sum_keepdim`, `broadcast_div`) and the weighted sum
/// adds a `broadcast_mul` and a `sum` — and EVERY compressor emit in the model
/// runs it: the prefill fleet's pool, the decode wave's per-layer pool, the
/// streamed reference, and the seal `close`. The eager form below is the CPU
/// path.
///
/// The fused result is NOT bit-equal to the eager device chain, because the
/// eager chain is the less accurate of the two: under the archive's
/// `--use_fast_math` it pays `__expf` plus an approximate division across five
/// kernels, each rounding an intermediate out to memory, where the fused pass
/// keeps them in registers. `pool_kernel_matches_eager` measures all three
/// (kernel, eager, exact host f32) and holds the kernel to being at least as
/// faithful to exact arithmetic as what it replaces.
///
/// Each input is read through its OWN strides. Producers legitimately hand this
/// views — `assemble_groups_batched` slices per-sequence pools out of one
/// fleet-wide block, and the overlap split narrows the channel axis — and
/// flattening those with `contiguous()` would be an allocate-plus-copy of the
/// whole plane on a per-layer path. Reading the layout that exists is the
/// hot-path invariant; materialising the layout the kernel would prefer is the
/// violation of it.
fn pool_group(pool_kv: &Tensor, pool_score: &Tensor) -> Result<Tensor> {
    let (g, p, d) = pool_kv.dims3()?;
    if pool_score.dims3()? != (g, p, d) {
        candle::bail!(
            "pool_group: kv {:?} and score {:?} must have the same shape",
            pool_kv.dims(),
            pool_score.dims()
        );
    }
    if p == 0 {
        // A softmax over zero rows has no value to return, and the CUDA arm
        // allocates `out` UNINITIALISED for the kernel to fill — so an empty
        // pooling axis would hand back garbage rather than fail. Every producer
        // pools at least one row (`2r`, `r`, or `r + n` for a sealed partial),
        // so this is a guard on the invariant, not a reachable case.
        candle::bail!("pool_group: empty pooling axis (g={g}, d={d})");
    }

    #[cfg(feature = "cuda")]
    if matches!(pool_kv.device(), Device::Cuda(_)) {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        use candle_kernels::simple::compressor_pool::run_compressor_pool;

        let dev = match pool_kv.device() {
            Device::Cuda(dd) => dd.clone(),
            _ => unreachable!("guarded by the matches! above"),
        };
        let stream = dev.cuda_stream();
        // The kernel writes every element of `out`, so it is allocated
        // uninitialised — a `zeros` here would be a second full-width pass over
        // the exact bytes the kernel is about to stamp (hot-path invariant 6).
        let out = Tensor::empty((g, d), DType::F32, pool_kv.device())?;
        {
            let (s_kv, l_kv) = pool_kv.storage_and_layout();
            let (s_sc, l_sc) = pool_score.storage_and_layout();
            let (s_out, _) = out.storage_and_layout();
            let ptr = |st: &Storage, off: usize| -> Result<u64> {
                match st {
                    Storage::Cuda(c) => Ok(c.as_cuda_slice::<f32>()?.slice(off..).device_ptr(&stream).0),
                    _ => candle::bail!("pool_group: expected CUDA storage"),
                }
            };
            let p_kv = ptr(&s_kv, l_kv.start_offset())?;
            let p_sc = ptr(&s_sc, l_sc.start_offset())?;
            let p_out = ptr(&s_out, 0)?;
            // Each input is read through its OWN strides, so a caller's view is
            // consumed where it lies instead of being copied flat first.
            let (kv_s, sc_s) = (l_kv.stride(), l_sc.stride());
            unsafe {
                run_compressor_pool(
                    p_kv as *const f32,
                    p_sc as *const f32,
                    p_out as *mut f32,
                    g as i32,
                    p as i32,
                    d as i32,
                    kv_s[0] as i64,
                    kv_s[1] as i64,
                    kv_s[2] as i64,
                    sc_s[0] as i64,
                    sc_s[1] as i64,
                    sc_s[2] as i64,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        return Ok(out);
    }

    let w = softmax(pool_score, 1)?;
    pool_kv.broadcast_mul(&w)?.sum(1)
}

/// Replace NaNs with `fill` (element-wise): `where(x == x, x, fill)`.
#[allow(dead_code)]
fn replace_nan(x: &Tensor, fill: f64) -> Result<Tensor> {
    let is_nan = x.ne(x)?; // NaN != NaN → 1
    let fill_t = Tensor::full(fill as f32, x.shape(), x.device())?;
    is_nan.where_cond(&fill_t, x)
}

/// Streaming (decode-time) counterpart to [`Compressor`]. The prefill `Compressor::forward`
/// recomputes every compressed entry from the full prefix on each step; during incremental
/// decode we instead accumulate the current group's `ratio` token projections and emit one
/// entry the moment the group completes. For the overlapping (`ratio == 4`) compressor the
/// entry also pools the *previous* group's first-half projection rows, so the streamer
/// retains the last completed group's `(kv, score+ape)` to serve as the next group's "prev"
/// half — exactly the `overlap_transform` shift `prev[g] = prev_src[g-1]` done batch-wise in
/// prefill. The emitted entry is `pool → RMSNorm → RoPE(at group-start position g·ratio)`,
/// numerically equal to the prefill entry (proven by `incremental_matches_prefill`).
/// The assembled-but-not-yet-pooled output of [`IncrementalCompressor::assemble_groups`]:
/// the completed groups' pool inputs `[groups, P, d]` and their start positions.
/// Concatenating several sequences' `GroupPool`s on the group axis and running one
/// [`Compressor::pool_and_norm`] pools the whole prefill fleet in a single launch.
pub struct GroupPool {
    /// Pool input values `[groups, P, d]` (`P = 2r` overlap / `r` non-overlap).
    pub pool_kv: Tensor,
    /// Pool input scores `[groups, P, d]` (`ape` already added).
    pub pool_score: Tensor,
    /// Each group's start position `group_idx·ratio`, for the RoPE.
    pub positions: Vec<u32>,
}

pub struct IncrementalCompressor {
    c: Compressor,
    /// Current (incomplete) group's per-token projections, each `[1, cd]` F32.
    kv_rows: Vec<Tensor>,
    score_rows: Vec<Tensor>,
    /// Last completed group's `kv` (`[ratio, cd]`) and `score+ape` (`[ratio, cd]`), retained
    /// as the overlapping compressor's "prev" half. `None` before the first group completes.
    prev_kv_group: Option<Tensor>,
    prev_score_group: Option<Tensor>,
    /// Index of the next group to emit (its RoPE position is `group_idx · ratio`).
    group_idx: usize,
}

/// A point-in-time copy of an [`IncrementalCompressor`]'s streaming state — the
/// partial-group buffers, the overlap prev-group halves, and the group counter.
/// Tensors are immutable in candle, so the clones are `Arc` bumps: taking a
/// snapshot is O(buffered rows) pointer copies, no data movement. Used by the
/// speculative-decode verify path to roll the compressor back to the accepted
/// prefix after a partial accept (rejected draft tokens must not stay absorbed).
#[derive(Clone)]
pub struct CompressorState {
    kv_rows: Vec<Tensor>,
    score_rows: Vec<Tensor>,
    prev_kv_group: Option<Tensor>,
    prev_score_group: Option<Tensor>,
    group_idx: usize,
}

impl IncrementalCompressor {
    /// Snapshot the streaming state (see [`CompressorState`]). Cheap: `Arc`
    /// clones of the buffered rows + the counter.
    pub fn state_snapshot(&self) -> CompressorState {
        CompressorState {
            kv_rows: self.kv_rows.clone(),
            score_rows: self.score_rows.clone(),
            prev_kv_group: self.prev_kv_group.clone(),
            prev_score_group: self.prev_score_group.clone(),
            group_idx: self.group_idx,
        }
    }

    /// Restore a [`Self::state_snapshot`] — the compressor behaves exactly as it
    /// did at snapshot time (bit-identical emissions for identical subsequent
    /// rows; the snapshotted tensors are immutable).
    pub fn state_restore(&mut self, s: CompressorState) {
        self.kv_rows = s.kv_rows;
        self.score_rows = s.score_rows;
        self.prev_kv_group = s.prev_kv_group;
        self.prev_score_group = s.prev_score_group;
        self.group_idx = s.group_idx;
    }

    /// Feed one token's hidden state `x` (`[dim]` / `[1, dim]` / `[1, 1, dim]`) at the next
    /// sequence position and, when it completes a group of `ratio` tokens, return that group's
    /// compressed entry `[1, 1, d]` (post RMSNorm + RoPE). Returns `None` mid-group.
    ///
    /// `rope` must be the same `RotaryCache` the prefill path uses for this compressor.
    pub fn push(&mut self, x: &Tensor, rope: &RotaryCache) -> Result<Option<Tensor>> {
        let (kv, score) = self.c.project_row(x)?;
        self.push_projected_roped(&kv, &score, rope)
    }

    /// As [`Self::push`] but the caller supplies the already-projected `kv`/`score`
    /// rows (each `[1, coff·d]`, from [`Compressor::project_rows`]) instead of the
    /// raw hidden `x` — buffering + roped emit only, no per-row projection GEMV.
    /// Bit-identical to `push` on the same token (the projection is `push`'s only
    /// difference). The prefill fast path batches the projection once and streams
    /// the rows through here.
    pub fn push_projected_roped(
        &mut self,
        kv: &Tensor,
        score: &Tensor,
        rope: &RotaryCache,
    ) -> Result<Option<Tensor>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.emit_group(rope)).transpose()
    }

    /// As [`Self::push`] but emitting the **pre-RoPE** entry plus its
    /// group-start position — the paged-kernel path's form (the kernel applies
    /// RoPE at read time from the stored position; storage stays
    /// position-free).
    pub fn push_raw(&mut self, x: &Tensor) -> Result<Option<(Tensor, u32)>> {
        let (kv, score) = self.c.project_row(x)?;
        self.push_projected(&kv, &score)
    }

    /// As [`Self::push_raw`] but the caller supplies the already-projected
    /// `kv`/`score` rows (each `[1, coff·d]`, from [`Compressor::project_rows`])
    /// instead of the raw hidden `x` — buffering + pre-RoPE emit only, no per-row
    /// projection GEMV. Bit-identical to `push_raw` on the same token. Feeds the
    /// batched-projection prefill fast path.
    pub fn push_projected(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<(Tensor, u32)>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.emit_group_raw()).transpose()
    }

    /// Batched projection over a whole prefix `xs` (`[n, dim]` / `[1, n, dim]`)
    /// for this compressor's own weights — the source rows for
    /// [`Self::push_projected`] / [`Self::push_projected_roped`]. One GEMM pair
    /// instead of `n` per-row `project_row` GEMVs.
    pub fn project_rows(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        self.c.project_rows(xs)
    }

    /// Number of rows currently buffered in the incomplete group (`0..ratio`).
    /// The causal-visibility bound for a batched prefill: after consuming `t+1`
    /// rows this call, `(buffered_len() + t + 1) / ratio` groups are complete.
    pub fn buffered_len(&self) -> usize {
        self.kv_rows.len()
    }

    /// This compressor's group size (`ratio`).
    pub fn ratio(&self) -> usize {
        self.c.ratio
    }

    /// Snapshot this sequence's streaming state + its new rows as one
    /// [`SeqAssemble`] for [`assemble_groups_batched`]. Pure `Arc` clones —
    /// no device work. `kv`/`score` are this sequence's `[n, cd]` slices of
    /// the wave-batched projection.
    pub fn assemble_input(&self, kv: &Tensor, score: &Tensor) -> Result<SeqAssemble> {
        let n = kv.dim(0)?;
        let mut kv_pieces: Vec<Tensor> = Vec::with_capacity(self.kv_rows.len() + 1);
        let mut sc_pieces: Vec<Tensor> = Vec::with_capacity(self.score_rows.len() + 1);
        kv_pieces.extend(self.kv_rows.iter().cloned());
        sc_pieces.extend(self.score_rows.iter().cloned());
        kv_pieces.push(kv.clone());
        sc_pieces.push(score.clone());
        Ok(SeqAssemble {
            kv_pieces,
            sc_pieces,
            l0: self.kv_rows.len(),
            n,
            prev_kv: self.prev_kv_group.clone(),
            prev_sc: self.prev_score_group.clone(),
            group_idx: self.group_idx,
        })
    }

    /// Install the state advance [`assemble_groups_batched`] computed for this
    /// sequence — the batched twin of the state writes at the end of
    /// [`Self::assemble_groups`].
    pub fn assemble_apply(&mut self, upd: AssembleUpdate) {
        self.kv_rows = upd.kv_rows;
        self.score_rows = upd.score_rows;
        if let (Some(pk), Some(ps)) = (upd.prev_kv, upd.prev_sc) {
            self.prev_kv_group = Some(pk);
            self.prev_score_group = Some(ps);
        }
        self.group_idx += upd.group_add;
    }

    /// Batched, carried-state-aware group emission: consume `n` pre-projected
    /// rows (`kv`/`score` each `[n, cd]`, from [`Compressor::project_rows`]) and
    /// emit EVERY complete group they form in one batched pool — bit-identical to
    /// streaming the same rows one at a time through [`Self::push_projected`]
    /// (`rope = None`, pre-RoPE attn entries; matches `emit_group_raw`) or
    /// [`Self::push_projected_roped`] (`rope = Some`, roped indexer keys; matches
    /// `emit_group`). The carried partial-group buffer and the overlap prev-group
    /// seed the first emitted group, so a prefill that resumes mid-group
    /// (`base > 0`, non-empty buffer) is handled exactly. Returns
    /// `(entries [G, head_dim], group-start positions [G])`, or `None` when the
    /// rows complete no group (all buffered for the next call).
    pub fn emit_groups_projected(
        &mut self,
        kv: &Tensor,
        score: &Tensor,
        rope: Option<&RotaryCache>,
    ) -> Result<Option<(Tensor, Vec<u32>)>> {
        // Composition of the two halves below — behavior identical to the former
        // monolithic form (still gated by `emit_groups_batched_matches_streamed`).
        match self.assemble_groups(kv, score)? {
            None => Ok(None),
            Some(gp) => {
                let entry =
                    self.c
                        .pool_and_norm(&gp.pool_kv, &gp.pool_score, &gp.positions, rope)?;
                Ok(Some((entry, gp.positions)))
            }
        }
    }

    /// The batchable first half of [`Self::emit_groups_projected`]: combine the
    /// carried partial group with the new rows, and for every group that now
    /// completes produce its **pool inputs** `[groups, P, d]` (`P = 2r` overlap /
    /// `r` non-overlap) + group-start positions, WITHOUT pooling. Advances the
    /// per-seq streaming state (`prev_*_group`, partial buffer, `group_idx`).
    ///
    /// The state advance reads only the RAW group rows (`kv_g`/`score_g`/
    /// `kv_comb`), never the pooled/normed/roped entry, so several sequences'
    /// `GroupPool`s can be concatenated and pooled in ONE launch
    /// ([`Compressor::pool_and_norm`]) — the prefill wave batches the pool across
    /// all prompt sequences this way. `None` when no group completes (all buffered).
    pub fn assemble_groups(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<GroupPool>> {
        let r = self.c.ratio;
        let d = self.c.head_dim;
        let cd = self.c.cd();
        let dev = self.c.device().clone();
        let n = kv.dim(0)?;

        // Combined stream = carried partial-group rows ++ the new rows, so a
        // resume mid-group starts exactly where the per-token buffer left off.
        let l0 = self.kv_rows.len();
        let (kv_comb, score_comb) = if l0 == 0 {
            (kv.clone(), score.clone())
        } else {
            let ck = Tensor::cat(&self.kv_rows.iter().collect::<Vec<_>>(), 0)?;
            let cs = Tensor::cat(&self.score_rows.iter().collect::<Vec<_>>(), 0)?;
            (Tensor::cat(&[&ck, kv], 0)?, Tensor::cat(&[&cs, score], 0)?)
        };
        let n_tot = l0 + n;
        let groups = n_tot / r;
        if groups == 0 {
            // No group completes: buffer every combined row for the next call.
            // ONE forced copy of the whole (tiny, < ratio rows) combined buffer
            // breaks the pin on the batched-projection input `kv` (a dim-0 narrow,
            // which plain `contiguous` would NOT copy); the per-row entries are
            // then free views into that small owned buffer. Replaces the former
            // per-row `force_contiguous` storm (2·n_tot copies → 2).
            let kv_buf = kv_comb.force_contiguous()?;
            let sc_buf = score_comb.force_contiguous()?;
            self.kv_rows = (0..n_tot)
                .map(|i| kv_buf.narrow(0, i, 1))
                .collect::<Result<_>>()?;
            self.score_rows = (0..n_tot)
                .map(|i| sc_buf.narrow(0, i, 1))
                .collect::<Result<_>>()?;
            return Ok(None);
        }
        let cutoff = groups * r;

        // [groups, r, cd] group rows; `ape` is added to the score BEFORE the
        // overlap split (matches `pool` / `emit_group_raw`).
        let kv_g = kv_comb.narrow(0, 0, cutoff)?.reshape((groups, r, cd))?;
        let ape = self.c.ape.to_dtype(DType::F32)?.reshape((1, r, cd))?;
        let score_g = score_comb
            .narrow(0, 0, cutoff)?
            .reshape((groups, r, cd))?
            .broadcast_add(&ape)?;

        let (pool_kv, pool_score) = if self.c.overlap {
            let curr_kv = kv_g.narrow(D::Minus1, d, d)?; // [groups, r, d]
            let curr_score = score_g.narrow(D::Minus1, d, d)?;
            let prev_src_kv = kv_g.narrow(D::Minus1, 0, d)?; // [groups, r, d]
            let prev_src_score = score_g.narrow(D::Minus1, 0, d)?;
            // prev[0] = carried prev group's first-half (pad kv=0 / score=−inf for
            // a true group 0), prev[j>0] = group j−1's first-half — the batch-wise
            // form of `overlap_transform`'s `prev[g] = prev_src[g−1]` shift.
            let (p0_kv, p0_score) = match (&self.prev_kv_group, &self.prev_score_group) {
                (Some(pk), Some(ps)) => (
                    pk.narrow(D::Minus1, 0, d)?.reshape((1, r, d))?,
                    ps.narrow(D::Minus1, 0, d)?.reshape((1, r, d))?,
                ),
                _ => (
                    Tensor::zeros((1, r, d), DType::F32, &dev)?,
                    Tensor::full(f32::NEG_INFINITY, (1, r, d), &dev)?,
                ),
            };
            let prev_kv = if groups > 1 {
                Tensor::cat(&[&p0_kv, &prev_src_kv.narrow(0, 0, groups - 1)?], 0)?
            } else {
                p0_kv
            };
            let prev_score = if groups > 1 {
                Tensor::cat(&[&p0_score, &prev_src_score.narrow(0, 0, groups - 1)?], 0)?
            } else {
                p0_score
            };
            let pool_kv = Tensor::cat(&[&prev_kv, &curr_kv], 1)?; // [groups, 2r, d]
            let pool_score = Tensor::cat(&[&prev_score, &curr_score], 1)?;
            (pool_kv, pool_score)
        } else {
            // Non-overlap pools the group rows directly (cd == d).
            (kv_g.clone(), score_g.clone()) // [groups, r, cd]
        };

        let positions: Vec<u32> = (0..groups)
            .map(|j| ((self.group_idx + j) * r) as u32)
            .collect();

        // Retain state: the last complete group as the next overlap prev, the
        // trailing rows as the next partial buffer, and advance the group index.
        // Each retained tensor is materialised with a SINGLE forced copy out of
        // the combined-stream buffer (a dim-0 narrow that plain `contiguous`
        // would not copy — leaving the big batched-projection buffer pinned), so
        // the stream buffer frees while the retained state stays bounded to ≤2·r
        // rows. The `rem` trailing rows are held as free views into one small
        // owned tail buffer, replacing the former per-row `force_contiguous`
        // storm (2·rem copies → 2). Reads only the RAW group rows —
        // pooling-independent, so the pool can be deferred + batched across seqs.
        self.prev_kv_group = Some(
            kv_g.narrow(0, groups - 1, 1)?
                .reshape((r, cd))?
                .force_contiguous()?,
        );
        self.prev_score_group = Some(
            score_g
                .narrow(0, groups - 1, 1)?
                .reshape((r, cd))?
                .force_contiguous()?,
        );
        let rem = n_tot - cutoff;
        let kv_tail = kv_comb.narrow(0, cutoff, rem)?.force_contiguous()?;
        let sc_tail = score_comb.narrow(0, cutoff, rem)?.force_contiguous()?;
        self.kv_rows = (0..rem)
            .map(|i| kv_tail.narrow(0, i, 1))
            .collect::<Result<_>>()?;
        self.score_rows = (0..rem)
            .map(|i| sc_tail.narrow(0, i, 1))
            .collect::<Result<_>>()?;
        self.group_idx += groups;

        Ok(Some(GroupPool {
            pool_kv,
            pool_score,
            positions,
        }))
    }

    /// Absorb ONE pre-projected row (`[1, cd]`) and, when it completes a group,
    /// return that group's pool INPUTS without pooling — the deferred-pool twin
    /// of [`Self::push_projected`], and what the decode wave absorbs each step.
    /// The completed groups from every decode session then pool together in a
    /// single [`Compressor::pool_and_norm`].
    ///
    /// Distinct from [`Self::assemble_groups`], which combines a whole RUN of
    /// new rows with the carried buffer and is the prefill form: that rebuild
    /// costs a `cat` plus two `force_contiguous` copies EVERY call, which is the
    /// right trade when it amortises over a prompt but not when it runs per
    /// token. Here a step that completes no group costs nothing on the device —
    /// an `Arc` bump onto the row buffer — and at `ratio` 4 that is three decode
    /// steps in four.
    pub fn assemble_row(&mut self, kv: &Tensor, score: &Tensor) -> Result<Option<GroupPool>> {
        self.kv_rows.push(kv.clone());
        self.score_rows.push(score.clone());
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.assemble_group_raw()).transpose()
    }

    /// The buffered complete group's pool inputs `[1, P, d]` (`P = 2r` overlap /
    /// `r` non-overlap) + its start position, advancing the streaming state
    /// (`prev_*_group`, buffer, `group_idx`). The caller must have checked the
    /// group is full.
    ///
    /// Reads only the RAW group rows, never a pooled result, which is what lets
    /// the pool defer and batch across sessions.
    fn assemble_group_raw(&mut self) -> Result<GroupPool> {
        let d = self.c.head_dim;
        let r = self.c.ratio;
        let cd = self.c.cd();
        let dev = self.c.device().clone();

        let kv_rows: Vec<&Tensor> = self.kv_rows.iter().collect();
        let score_rows: Vec<&Tensor> = self.score_rows.iter().collect();
        let kv_group = Tensor::cat(&kv_rows, 0)?; // [r, cd]
                                                  // ape is added to the score BEFORE pooling / the overlap split (matches `pool`).
        let ape = self.c.ape.to_dtype(DType::F32)?.reshape((r, cd))?;
        let score_group = (Tensor::cat(&score_rows, 0)? + ape)?; // [r, cd]

        // Pool inputs over the group's rows (overlap: prev-half ‖ curr-half over 2·r rows).
        let (pool_kv, pool_score) = if self.c.overlap {
            let curr_kv = kv_group.narrow(D::Minus1, d, d)?; // [r, d] second-half dims
            let curr_score = score_group.narrow(D::Minus1, d, d)?;
            let (prev_kv, prev_score) = match (&self.prev_kv_group, &self.prev_score_group) {
                (Some(pk), Some(ps)) => (pk.narrow(D::Minus1, 0, d)?, ps.narrow(D::Minus1, 0, d)?),
                // Group 0 has no previous group: `pool`'s pad is kv=0, score=-inf (fully masked).
                _ => (
                    Tensor::zeros((r, d), DType::F32, &dev)?,
                    Tensor::full(f32::NEG_INFINITY, (r, d), &dev)?,
                ),
            };
            (
                Tensor::cat(&[&prev_kv, &curr_kv], 0)?.reshape((1, 2 * r, d))?,
                Tensor::cat(&[&prev_score, &curr_score], 0)?.reshape((1, 2 * r, d))?,
            )
        } else {
            // Non-overlap pools the group rows directly (cd == d).
            (
                kv_group.reshape((1, r, cd))?,
                score_group.reshape((1, r, cd))?,
            )
        };

        // Retain this group as the next group's "prev" half, then reset the current buffer.
        self.prev_kv_group = Some(kv_group);
        self.prev_score_group = Some(score_group);
        self.kv_rows.clear();
        self.score_rows.clear();

        let g = self.group_idx;
        self.group_idx += 1;
        Ok(GroupPool {
            pool_kv,
            pool_score,
            positions: vec![(g * r) as u32],
        })
    }

    /// Pool → RMSNorm (NO RoPE): the position-free entry `[1, 1, d]` and its
    /// group-start position. Assemble + pool, so the streamed reference and the
    /// deferred-pool decode path share ONE pool implementation
    /// ([`Compressor::pool_and_norm`]) rather than two that must be kept in
    /// step. RoPE is the caller's concern: `emit_group` applies it, while the
    /// kernel path stores pre-RoPE and rotates at read.
    fn emit_group_raw(&mut self) -> Result<(Tensor, u32)> {
        let d = self.c.head_dim;
        let gp = self.assemble_group_raw()?;
        let entry = self
            .c
            .pool_and_norm(&gp.pool_kv, &gp.pool_score, &gp.positions, None)?;
        Ok((entry.reshape((1, 1, d))?, gp.positions[0]))
    }

    /// Finalize the trailing partial group at a **turn seal**: pool the
    /// `< ratio` currently-buffered rows into a corpus entry and return it
    /// pre-RoPE (`[1, 1, d]`) alongside its group-start position, mirroring
    /// [`Self::emit_group_raw`] but over fewer than `ratio` rows. Returns `None`
    /// when nothing is buffered (the group boundary fell exactly on the seal).
    ///
    /// For the overlapping (`ratio == 4`) compressor the pool still includes the
    /// **previous complete group's first-half rows** (the retained `prev_*`
    /// halves) — the closed partial is a softmax-weighted latent exactly like a
    /// full group, so the attention kernel merges it with no special case
    /// (docs/deepseek_turn_seal_persistence.md Artifact B). `close` is terminal:
    /// it clears the buffer and advances `group_idx`; the compressor must not be
    /// pushed again after it.
    pub fn close(&mut self) -> Result<Option<(Tensor, u32)>> {
        let n = self.kv_rows.len();
        if n == 0 {
            return Ok(None);
        }
        let d = self.c.head_dim;
        let r = self.c.ratio;
        let dev = self.c.device().clone();

        let kv_rows: Vec<&Tensor> = self.kv_rows.iter().collect();
        let score_rows: Vec<&Tensor> = self.score_rows.iter().collect();
        let kv_group = Tensor::cat(&kv_rows, 0)?; // [n, cd]
                                                  // ape rows for the buffered within-group positions 0..n (added BEFORE
                                                  // the overlap split, matching `pool` / `emit_group_raw`).
        let ape = self
            .c
            .ape
            .to_dtype(DType::F32)?
            .reshape((r, self.c.cd()))?
            .narrow(0, 0, n)?;
        let score_group = (Tensor::cat(&score_rows, 0)? + ape)?; // [n, cd]

        let (pool_kv, pool_score) = if self.c.overlap {
            let curr_kv = kv_group.narrow(D::Minus1, d, d)?; // [n, d] second-half dims
            let curr_score = score_group.narrow(D::Minus1, d, d)?;
            let (prev_kv, prev_score) = match (&self.prev_kv_group, &self.prev_score_group) {
                (Some(pk), Some(ps)) => (pk.narrow(D::Minus1, 0, d)?, ps.narrow(D::Minus1, 0, d)?),
                // The partial group is itself group 0: no previous group, so the
                // `-inf` prev-half is fully masked (pools only the buffered rows).
                _ => (
                    Tensor::zeros((r, d), DType::F32, &dev)?,
                    Tensor::full(f32::NEG_INFINITY, (r, d), &dev)?,
                ),
            };
            (
                Tensor::cat(&[&prev_kv, &curr_kv], 0)?.reshape((1, r + n, d))?,
                Tensor::cat(&[&prev_score, &curr_score], 0)?.reshape((1, r + n, d))?,
            )
        } else {
            (
                kv_group.reshape((1, n, self.c.cd()))?, // cd == d
                score_group.reshape((1, n, self.c.cd()))?,
            )
        };

        // Terminal: consume the buffer, but do NOT retain a new `prev_*` (nothing
        // follows a seal-close in this compressor's lifetime).
        self.kv_rows.clear();
        self.score_rows.clear();

        let g = self.group_idx;
        self.group_idx += 1;
        let positions = vec![(g * r) as u32];
        // The same pool every other emit path runs — a closed partial is a
        // softmax-weighted latent exactly like a full group, just over fewer rows.
        let entry = self
            .c
            .pool_and_norm(&pool_kv, &pool_score, &positions, None)?;
        Ok(Some((entry.reshape((1, 1, d))?, positions[0])))
    }

    fn emit_group(&mut self, rope: &RotaryCache) -> Result<Tensor> {
        let d = self.c.head_dim;
        let gp = self.assemble_group_raw()?;
        self.c
            .pool_and_norm(&gp.pool_kv, &gp.pool_score, &gp.positions, Some(rope))?
            .reshape((1, 1, d))
    }

}

/// One sequence's input to [`assemble_groups_batched`]: its carried
/// partial-group rows + its new projected rows (all `Arc` clones), and the
/// scalar state the group math needs. Built by
/// [`IncrementalCompressor::assemble_input`].
pub struct SeqAssemble {
    kv_pieces: Vec<Tensor>,
    sc_pieces: Vec<Tensor>,
    l0: usize,
    n: usize,
    prev_kv: Option<Tensor>,
    prev_sc: Option<Tensor>,
    group_idx: usize,
}

/// One sequence's state advance out of [`assemble_groups_batched`], installed
/// by [`IncrementalCompressor::assemble_apply`]. `prev_*` are `None` when the
/// sequence completed no group (the carried prev is kept).
pub struct AssembleUpdate {
    kv_rows: Vec<Tensor>,
    score_rows: Vec<Tensor>,
    prev_kv: Option<Tensor>,
    prev_sc: Option<Tensor>,
    group_add: usize,
}

/// The fleet's assemble GEOMETRY — every host-side quantity and device index
/// tensor derivable from the per-seq `(carried, new, has_prev)` shape alone,
/// independent of the layer's actual data.
///
/// The same wave calls [`assemble_groups_batched`] once per LAYER, and the
/// geometry is layer-invariant by construction: every layer's compressor has
/// consumed the same token stream, so carried-tail lengths, group counts and
/// prev-presence match across the wave's layers. Building the indices per call
/// cost 4 host-vec builds + 4 tiny H2D uploads + 2 pad fills, × layers ×
/// compressor families; the cache pays that once per wave shape and every
/// other layer-call hits.
struct AssembleGeom {
    groups: Vec<usize>,
    cutoff: Vec<usize>,
    n_tot: Vec<usize>,
    g_off: Vec<usize>,
    total_rows: usize,
    total_groups: usize,
    tail_off: Vec<usize>,
    /// Combined-stream index of every remainder-tail row (`None`: no tails).
    tail_it: Option<Tensor>,
    /// Combined-stream index of every group row (`None`: no completed groups).
    gi_t: Option<Tensor>,
    /// Overlap only: each group's prev-source slot in `[pad | prevs | kv_g]`.
    pi_t: Option<Tensor>,
    /// Pool index of each group-completing seq's LAST group (its next prev).
    li_t: Option<Tensor>,
    /// Overlap only: the group-0 pad rows (kv 0 / score −inf), `[1, r, cd]`.
    pad_kv: Option<Tensor>,
    pad_sc: Option<Tensor>,
}

/// Cache key: the template's `(ratio, cd, overlap)` plus every seq's
/// `(carried, new, has_prev)` — equal keys imply identical geometry.
type AssembleGeomKey = (usize, usize, bool, Vec<(usize, usize, bool)>);

/// Process-wide geometry cache, bounded like `CudaDevice::info_table`
/// (cleared wholesale past the cap — entries are a handful of tiny device
/// tensors and rebuilding is one call's work). Single-accelerator assumption,
/// like `cached_sm_count` in candle-core: the cached index tensors live on
/// the one device this fork targets.
fn assemble_geom_cache() -> &'static Mutex<HashMap<AssembleGeomKey, Arc<AssembleGeom>>> {
    static CACHE: OnceLock<Mutex<HashMap<AssembleGeomKey, Arc<AssembleGeom>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn build_assemble_geom(
    r: usize,
    cd: usize,
    overlap: bool,
    seqs: &[(usize, usize, bool)],
    dev: &Device,
) -> Result<AssembleGeom> {
    let n_seq = seqs.len();
    let mut c_off = Vec::with_capacity(n_seq);
    let mut groups = Vec::with_capacity(n_seq);
    let mut cutoff = Vec::with_capacity(n_seq);
    let mut n_tot = Vec::with_capacity(n_seq);
    let mut g_off = Vec::with_capacity(n_seq);
    let mut off = 0usize;
    let mut goff = 0usize;
    for &(l0, n, _) in seqs {
        let nt = l0 + n;
        let g = nt / r;
        c_off.push(off);
        n_tot.push(nt);
        groups.push(g);
        cutoff.push(g * r);
        g_off.push(goff);
        off += nt;
        goff += g;
    }
    let total_rows = off;
    let total_groups = goff;

    let mut tail_idx: Vec<u32> = Vec::with_capacity(total_rows - total_groups * r);
    let mut tail_off = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        tail_off.push(tail_idx.len());
        for t in cutoff[i]..n_tot[i] {
            tail_idx.push((c_off[i] + t) as u32);
        }
    }
    let tail_it = if tail_idx.is_empty() {
        None
    } else {
        let n_tail = tail_idx.len();
        Some(Tensor::from_vec(tail_idx, n_tail, dev)?)
    };

    let (gi_t, pi_t, li_t, pad_kv, pad_sc) = if total_groups == 0 {
        (None, None, None, None, None)
    } else {
        let mut grp_idx: Vec<u32> = Vec::with_capacity(total_groups * r);
        for i in 0..n_seq {
            for t in 0..cutoff[i] {
                grp_idx.push((c_off[i] + t) as u32);
            }
        }
        let gi_t = Tensor::from_vec(grp_idx, total_groups * r, dev)?;

        let (pi_t, pad_kv, pad_sc) = if overlap {
            // Prev-source slot per group over `[pad | each seq's carried prev
            // | kv_g]`: group j>0 takes its group j−1; group 0 takes the seq's
            // carried prev, or slot 0 (the pad) when it has none.
            let mut prev_slot = vec![0u32; n_seq];
            let mut next = 1u32;
            for (i, &(_, _, has_prev)) in seqs.iter().enumerate() {
                if has_prev {
                    prev_slot[i] = next;
                    next += 1;
                }
            }
            let base = next;
            let mut pidx: Vec<u32> = Vec::with_capacity(total_groups);
            for i in 0..n_seq {
                for j in 0..groups[i] {
                    pidx.push(if j == 0 {
                        prev_slot[i]
                    } else {
                        base + (g_off[i] + j - 1) as u32
                    });
                }
            }
            let pi_t = Tensor::from_vec(pidx, total_groups, dev)?;
            let pad_kv = Tensor::zeros((1, r, cd), DType::F32, dev)?;
            let pad_sc = Tensor::full(f32::NEG_INFINITY, (1, r, cd), dev)?;
            (Some(pi_t), Some(pad_kv), Some(pad_sc))
        } else {
            (None, None, None)
        };

        let last_idx: Vec<u32> = (0..n_seq)
            .filter(|&i| groups[i] > 0)
            .map(|i| (g_off[i] + groups[i] - 1) as u32)
            .collect();
        let n_active = last_idx.len();
        let li_t = Tensor::from_vec(last_idx, n_active, dev)?;
        (Some(gi_t), pi_t, Some(li_t), pad_kv, pad_sc)
    };

    Ok(AssembleGeom {
        groups,
        cutoff,
        n_tot,
        g_off,
        total_rows,
        total_groups,
        tail_off,
        tail_it,
        gi_t,
        pi_t,
        li_t,
        pad_kv,
        pad_sc,
    })
}

/// The whole prompt fleet's compressor ASSEMBLE in ~a dozen device ops —
/// bit-identical per sequence to [`IncrementalCompressor::assemble_groups`]
/// called per seq, whose ~15 small launches × N sequences were HALF the
/// cfg20 prefill wall (15.7s of 29.8s — every enqueue in the per-seq loop
/// stalls under WDDM submission pressure, and the loop multiplies the stall
/// points by the fleet width).
///
/// Construction: one `cat` builds every sequence's `[carried ++ new]`
/// combined stream back-to-back; `index_select`s then pull the group rows,
/// the overlap-prev rows, the retained last-group rows and the remainder
/// tails for ALL sequences at once (pure row copies — byte-identical to the
/// per-seq narrows they replace); the one `broadcast_add(ape)` is
/// row-elementwise. The returned per-seq [`GroupPool`]s are dim-0 NARROW
/// VIEWS of one fleet-wide pool tensor, in sequence order — exactly the
/// concatenation `pool_prefill_across_seqs` builds from per-seq pools, so
/// everything downstream is unchanged. The index tensors + geometry come from
/// the per-shape [`AssembleGeom`] cache — layer-invariant within a wave, so
/// only the wave's first layer-call builds them.
///
/// `template` supplies the layer-shared [`Compressor`] (weights/ape/ratio —
/// identical across the fleet's per-seq states by construction).
pub fn assemble_groups_batched(
    template: &IncrementalCompressor,
    inputs: &[SeqAssemble],
) -> Result<Vec<(Option<GroupPool>, AssembleUpdate)>> {
    let c = &template.c;
    let r = c.ratio;
    let cd = c.cd();
    let d = c.head_dim;
    let dev = c.device().clone();
    let n_seq = inputs.len();

    // Geometry + index tensors from the per-shape cache (layer-invariant
    // within a wave — the key IS the verification that the shape matches).
    let key: AssembleGeomKey = (
        r,
        cd,
        c.overlap,
        inputs
            .iter()
            .map(|i| (i.l0, i.n, i.prev_kv.is_some()))
            .collect(),
    );
    let geom = {
        let mut cache = assemble_geom_cache().lock().unwrap();
        if let Some(g) = cache.get(&key) {
            g.clone()
        } else {
            if cache.len() >= 4096 {
                cache.clear();
            }
            let g = Arc::new(build_assemble_geom(r, cd, c.overlap, &key.3, &dev)?);
            cache.insert(key, g.clone());
            g
        }
    };
    let AssembleGeom {
        groups,
        n_tot,
        cutoff,
        g_off,
        total_rows,
        total_groups,
        tail_off,
        ..
    } = &*geom;
    let (total_rows, total_groups) = (*total_rows, *total_groups);

    // 1. Combined stream: every seq's [carried ++ new], back-to-back.
    let pieces_kv: Vec<&Tensor> = inputs.iter().flat_map(|i| i.kv_pieces.iter()).collect();
    let pieces_sc: Vec<&Tensor> = inputs.iter().flat_map(|i| i.sc_pieces.iter()).collect();
    let comb_kv = Tensor::cat(&pieces_kv, 0)?; // [total_rows, cd] owned
    let comb_sc = Tensor::cat(&pieces_sc, 0)?;
    debug_assert_eq!(comb_kv.dim(0)?, total_rows);

    // Remainder tails for ALL seqs (also the whole-stream buffer when a seq
    // completes no group): one select, per-seq row views into it.
    let (kv_tails, sc_tails) = match &geom.tail_it {
        None => (None, None),
        Some(it) => (
            Some(comb_kv.index_select(it, 0)?),
            Some(comb_sc.index_select(it, 0)?),
        ),
    };
    let tail_views = |src: &Option<Tensor>, i: usize| -> Result<Vec<Tensor>> {
        let rem = n_tot[i] - cutoff[i];
        let src = match src {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        (0..rem)
            .map(|t| src.narrow(0, tail_off[i] + t, 1))
            .collect()
    };

    if total_groups == 0 {
        // No sequence completes a group: everyone buffers its combined rows.
        let mut out = Vec::with_capacity(n_seq);
        for i in 0..n_seq {
            out.push((
                None,
                AssembleUpdate {
                    kv_rows: tail_views(&kv_tails, i)?,
                    score_rows: tail_views(&sc_tails, i)?,
                    prev_kv: None,
                    prev_sc: None,
                    group_add: 0,
                },
            ));
        }
        return Ok(out);
    }

    // 2. Group rows for the whole fleet: [total_groups, r, cd], `ape` added to
    // the scores before the overlap split (matches `assemble_groups`).
    let gi_t = geom
        .gi_t
        .as_ref()
        .expect("total_groups > 0 implies group indices");
    let kv_g = comb_kv
        .index_select(gi_t, 0)?
        .reshape((total_groups, r, cd))?;
    let ape = c.ape.to_dtype(DType::F32)?.reshape((1, r, cd))?;
    let sc_g = comb_sc
        .index_select(gi_t, 0)?
        .reshape((total_groups, r, cd))?
        .broadcast_add(&ape)?;

    // 3. Fleet-wide pool.
    let (pool_kv_all, pool_sc_all) = if c.overlap {
        // Prev source pool along dim 0: [pad | each seq's carried prev | kv_g].
        // Group j>0 of a seq takes its group j−1's rows; group 0 takes the
        // seq's carried prev, or the pad (kv 0 / score −inf) when it has none —
        // the batch-wise form of `assemble_groups`' prev shift. Full-cd rows
        // are selected and the halves split after (same bytes, one select).
        // The slot indices (`pi_t`) come from the geometry cache; the source
        // rows are per-call data, ordered to match the cached slot numbering.
        let pad_kv = geom.pad_kv.as_ref().expect("overlap implies pads");
        let pad_sc = geom.pad_sc.as_ref().expect("overlap implies pads");
        let mut src_kv: Vec<Tensor> = vec![pad_kv.clone()];
        let mut src_sc: Vec<Tensor> = vec![pad_sc.clone()];
        for inp in inputs.iter() {
            if let (Some(pk), Some(ps)) = (&inp.prev_kv, &inp.prev_sc) {
                src_kv.push(pk.reshape((1, r, cd))?);
                src_sc.push(ps.reshape((1, r, cd))?);
            }
        }
        let src_kv_refs: Vec<&Tensor> = src_kv.iter().chain(std::iter::once(&kv_g)).collect();
        let src_sc_refs: Vec<&Tensor> = src_sc.iter().chain(std::iter::once(&sc_g)).collect();
        let prev_src_kv = Tensor::cat(&src_kv_refs, 0)?;
        let prev_src_sc = Tensor::cat(&src_sc_refs, 0)?;
        let pi_t = geom.pi_t.as_ref().expect("overlap implies prev indices");
        let prev_kv = prev_src_kv.index_select(pi_t, 0)?.narrow(D::Minus1, 0, d)?;
        let prev_sc = prev_src_sc.index_select(pi_t, 0)?.narrow(D::Minus1, 0, d)?;
        let curr_kv = kv_g.narrow(D::Minus1, d, d)?;
        let curr_sc = sc_g.narrow(D::Minus1, d, d)?;
        (
            Tensor::cat(&[&prev_kv, &curr_kv], 1)?, // [total_groups, 2r, d]
            Tensor::cat(&[&prev_sc, &curr_sc], 1)?,
        )
    } else {
        (kv_g.clone(), sc_g.clone())
    };

    // 4. Retained last-group rows (the next call's overlap prev) for every
    // seq that completed a group: one select, per-seq views.
    let li_t = geom
        .li_t
        .as_ref()
        .expect("total_groups > 0 implies last-group indices");
    let last_kv = kv_g.index_select(li_t, 0)?; // [n_active, r, cd]
    let last_sc = sc_g.index_select(li_t, 0)?;

    // 5. Per-seq outputs: pool views + state advance.
    let mut out = Vec::with_capacity(n_seq);
    let mut a_slot = 0usize;
    for i in 0..n_seq {
        let gp = if groups[i] > 0 {
            let positions: Vec<u32> = (0..groups[i])
                .map(|j| ((inputs[i].group_idx + j) * r) as u32)
                .collect();
            Some(GroupPool {
                pool_kv: pool_kv_all.narrow(0, g_off[i], groups[i])?,
                pool_score: pool_sc_all.narrow(0, g_off[i], groups[i])?,
                positions,
            })
        } else {
            None
        };
        let (prev_kv, prev_sc) = if groups[i] > 0 {
            let pk = last_kv.narrow(0, a_slot, 1)?.reshape((r, cd))?;
            let ps = last_sc.narrow(0, a_slot, 1)?.reshape((r, cd))?;
            a_slot += 1;
            (Some(pk), Some(ps))
        } else {
            (None, None)
        };
        out.push((
            gp,
            AssembleUpdate {
                kv_rows: tail_views(&kv_tails, i)?,
                score_rows: tail_views(&sc_tails, i)?,
                prev_kv,
                prev_sc,
                group_add: groups[i],
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp};

    fn lin(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&w.t()?)
    }

    /// Non-overlapping pooling (`ratio != 4`) equals a scalar softmax-weighted average
    /// over each group of `ratio` consecutive tokens.
    #[test]
    fn nonoverlap_pool_matches_scalar() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio) = (6usize, 4usize, 3usize);
        let s = 7; // groups = 2, cutoff = 6, trailing token dropped
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;
        let wkv = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, d), &dev)?;
        let norm = Tensor::ones(d, DType::F32, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm,
            ratio,
            d,
            2,
            1e-6,
        );
        let got = c.pool(&x)?.unwrap(); // [1, 2, d]

        // Scalar reference.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [s, d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let got = got.i(0)?.to_vec2::<f32>()?;
        let groups = s / ratio;
        for g in 0..groups {
            for chan in 0..d {
                // softmax over the ratio rows of (score + ape) for this channel.
                let mut logits = vec![0f32; ratio];
                for t in 0..ratio {
                    logits[t] = sc[g * ratio + t][chan] + apev[t][chan];
                }
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f32 = exps.iter().sum();
                let mut acc = 0f32;
                for t in 0..ratio {
                    acc += exps[t] / z * kv[g * ratio + t][chan];
                }
                assert!(
                    (got[g][chan] - acc).abs() < 1e-4,
                    "g{g} c{chan}: {} vs {acc}",
                    got[g][chan]
                );
            }
        }
        Ok(())
    }

    /// §C/§E compression-seam monoid: a group's pooling rows folded WHOLE and
    /// folded in two fragments then LSE-merged both equal the single-shot
    /// softmax pool — the "two-turn seam-straddling group reconstructs ==
    /// single-shot forward of the concatenation" property. Also checks the
    /// monoid identity law. Pure `(m,l,acc)` arithmetic, model-independent.
    #[test]
    fn group_partial_seam_fold_matches_whole() -> Result<()> {
        let dev = Device::Cpu;
        let (r, d) = (6usize, 8usize);
        let scores = Tensor::randn(0f32, 1.0, (r, d), &dev)?;
        let kvs = Tensor::randn(0f32, 1.0, (r, d), &dev)?;

        // Single-shot reference: softmax over the r rows (per channel) · kv.
        let w = softmax(&scores, 0)?; // [r, d]
        let reference = w.broadcast_mul(&kvs)?.sum(0)?; // [d]
        let ref_v = reference.to_vec1::<f32>()?;

        let close = |a: &Tensor, msg: &str| -> Result<()> {
            let av = a.to_vec1::<f32>()?;
            for c in 0..d {
                assert!(
                    (av[c] - ref_v[c]).abs() < 1e-5,
                    "{msg} channel {c}: {} vs {}",
                    av[c],
                    ref_v[c]
                );
            }
            Ok(())
        };

        // Whole fold == reference.
        let whole = GroupPartial::identity(d, &dev)?
            .fold(&scores, &kvs)?
            .finalize()?;
        close(&whole, "whole fold")?;

        // Cut the group at every interior seam point; each split + merge ==
        // reference (order-independent, so both merge orders too).
        for cut in 1..r {
            let s1 = scores.narrow(0, 0, cut)?;
            let k1 = kvs.narrow(0, 0, cut)?;
            let s2 = scores.narrow(0, cut, r - cut)?;
            let k2 = kvs.narrow(0, cut, r - cut)?;
            let p1 = GroupPartial::identity(d, &dev)?.fold(&s1, &k1)?;
            let p2 = GroupPartial::identity(d, &dev)?.fold(&s2, &k2)?;
            close(
                &p1.merge(&p2)?.finalize()?,
                &format!("seam cut {cut} (p1⊕p2)"),
            )?;
            close(
                &p2.merge(&p1)?.finalize()?,
                &format!("seam cut {cut} (p2⊕p1)"),
            )?;
        }

        // Identity law: id ⊕ p == p ⊕ id == p.
        let p = GroupPartial::identity(d, &dev)?.fold(&scores, &kvs)?;
        let id = GroupPartial::identity(d, &dev)?;
        close(&id.merge(&p)?.finalize()?, "id ⊕ p")?;
        close(&p.merge(&id)?.finalize()?, "p ⊕ id")?;
        Ok(())
    }

    /// Overlapping pooling (`ratio == 4`): group 0 has no previous group, so its `-inf`
    /// prev-half is fully masked and the entry equals a pool over just the current 4 rows.
    #[test]
    fn overlap_group0_ignores_prev() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio) = (6usize, 4usize, 4usize);
        let s = 8; // groups = 2
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;
        let wkv = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?;
        let norm = Tensor::ones(d, DType::F32, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm,
            ratio,
            d,
            2,
            1e-6,
        );
        let got = c.pool(&x)?.unwrap().i((0, 0))?.to_vec1::<f32>()?; // group 0 entry

        // Scalar: group 0 pools the current 4 rows using the SECOND-half projection dims.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?;
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        for chan in 0..d {
            let mut logits = vec![0f32; ratio];
            for t in 0..ratio {
                // second-half dims are [d, 2d); ape added over the full 2d then sliced.
                logits[t] = sc[t][d + chan] + apev[t][d + chan];
            }
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let mut acc = 0f32;
            for t in 0..ratio {
                acc += exps[t] / z * kv[t][d + chan];
            }
            assert!(
                (got[chan] - acc).abs() < 1e-4,
                "c{chan}: {} vs {acc}",
                got[chan]
            );
        }
        Ok(())
    }

    /// The streaming (decode) compressor emits, group-by-group, entries numerically equal to
    /// the prefill `forward` over the full prefix — the mandatory prefill/decode equivalence
    /// (docs/deepseek_v4_flash.md §2.2). Exercises both the overlapping (`ratio == 4`) and
    /// non-overlapping compressors, across several complete groups plus trailing tokens.
    fn incremental_matches_prefill_case(ratio: usize, d: usize, rd: usize, s: usize) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;

        // Oracle: prefill over the whole prefix → [1, groups, d].
        let prefill = c.forward(&x, &rope)?.unwrap();
        let groups = s / ratio;
        assert_eq!(prefill.dim(1)?, groups);

        // Stream one token at a time; collect the emitted per-group entries.
        let mut inc = c.incremental();
        let mut emitted: Vec<Tensor> = Vec::new();
        for t in 0..s {
            let row = x.i((0, t))?; // [dim]
            if let Some(entry) = inc.push(&row, &rope)? {
                emitted.push(entry); // [1, 1, d]
            }
        }
        assert_eq!(emitted.len(), groups, "entry count (ratio={ratio})");
        let streamed = Tensor::cat(&emitted, 1)?; // [1, groups, d]

        let a = prefill.flatten_all()?.to_vec1::<f32>()?;
        let b = streamed.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "prefill vs streamed diverge (ratio={ratio}): max|Δ| = {max_abs}"
        );
        Ok(())
    }

    #[test]
    fn incremental_matches_prefill_overlap() -> Result<()> {
        // ratio 4 (overlapping): 3 complete groups + 2 trailing window tokens.
        incremental_matches_prefill_case(4, 6, 4, 14)
    }

    #[test]
    fn incremental_matches_prefill_nonoverlap() -> Result<()> {
        // ratio 3 (non-overlapping): 4 complete groups + 1 trailing token.
        incremental_matches_prefill_case(3, 5, 2, 13)
    }

    /// [`IncrementalCompressor::emit_groups_projected`] (batched, carried-state
    /// aware) must be bit-identical to streaming the same pre-projected rows one
    /// at a time through `push_projected` (raw) / `push_projected_roped` (roped) —
    /// including across a call boundary that splits a group (the resume-mid-group
    /// case a prefill at `base > 0` hits), for both the overlapping and
    /// non-overlapping compressors.
    fn emit_groups_batched_matches_streamed_case(
        ratio: usize,
        d: usize,
        rd: usize,
        s: usize,
        split: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let x = Tensor::randn(0f32, 1.0, (s, dim), &dev)?; // [s, dim]
        let (kv_all, score_all) = c.project_rows(&x)?; // [s, cd]

        let chunks = [(0usize, split), (split, s - split)];

        // Streamed vs batched, over both the raw (pre-RoPE) and roped forms.
        for roped in [false, true] {
            let mut inc_s = c.incremental();
            let mut ent_s: Vec<Tensor> = Vec::new();
            let mut pos_s: Vec<u32> = Vec::new();
            for t in 0..s {
                let kv = kv_all.narrow(0, t, 1)?;
                let sc = score_all.narrow(0, t, 1)?;
                if roped {
                    if let Some(e) = inc_s.push_projected_roped(&kv, &sc, &rope)? {
                        ent_s.push(e.reshape((1, d))?);
                    }
                } else if let Some((e, p)) = inc_s.push_projected(&kv, &sc)? {
                    ent_s.push(e.reshape((1, d))?);
                    pos_s.push(p);
                }
            }

            let mut inc_b = c.incremental();
            let mut ent_b: Vec<Tensor> = Vec::new();
            let mut pos_b: Vec<u32> = Vec::new();
            for &(a, len) in &chunks {
                if len == 0 {
                    continue;
                }
                let kv = kv_all.narrow(0, a, len)?;
                let sc = score_all.narrow(0, a, len)?;
                let rope_arg = if roped { Some(&rope) } else { None };
                if let Some((e, p)) = inc_b.emit_groups_projected(&kv, &sc, rope_arg)? {
                    ent_b.push(e);
                    pos_b.extend(p);
                }
            }

            if !roped {
                assert_eq!(pos_s, pos_b, "positions (ratio={ratio}, split={split})");
            }
            let a = Tensor::cat(&ent_s, 0)?.flatten_all()?.to_vec1::<f32>()?;
            let b = Tensor::cat(&ent_b, 0)?.flatten_all()?.to_vec1::<f32>()?;
            let max_abs = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-6,
                "batched vs streamed diverge (ratio={ratio}, split={split}, roped={roped}): max|Δ| = {max_abs}"
            );
        }
        Ok(())
    }

    #[test]
    fn emit_groups_batched_matches_streamed() -> Result<()> {
        // Overlapping (ratio 4): single-shot, split mid-group, split off-boundary,
        // and a first chunk that completes NO group (all buffered → None).
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 14)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 6)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 23, 7)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 2)?;
        // Non-overlapping (ratio 3).
        emit_groups_batched_matches_streamed_case(3, 5, 2, 13, 13)?;
        emit_groups_batched_matches_streamed_case(3, 5, 2, 13, 5)?;
        Ok(())
    }

    /// [`assemble_groups_batched`] over a staggered fleet is byte-identical,
    /// per sequence, to [`IncrementalCompressor::assemble_groups`] called per
    /// seq — pools, positions, AND the retained state (proven behaviorally: a
    /// second per-seq round through both paths must also match, so the carried
    /// buffer / overlap prev / group counter advanced identically).
    fn assemble_batched_matches_per_seq_case(ratio: usize, d: usize) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            2,
            1e-6,
        );
        // Staggered carried state per seq: 0 rows (fresh), a partial group, and
        // past a completed group (carried prev + partial buffer).
        let seeds = [0usize, ratio - 1, ratio + 2];
        // Round-1 row counts: complete ≥1 group, complete none, land mid-group.
        let lens1 = [2 * ratio + 1, 1, ratio];
        let lens2 = [ratio, 2 * ratio - 1, ratio + 1];

        let mut inc_a: Vec<IncrementalCompressor> = Vec::new(); // per-seq path
        let mut inc_b: Vec<IncrementalCompressor> = Vec::new(); // batched path
        for &p in &seeds {
            let mut i = c.incremental();
            if p > 0 {
                let x = Tensor::randn(0f32, 1.0, (p, dim), &dev)?;
                let (kv, sc) = c.project_rows(&x)?;
                i.emit_groups_projected(&kv, &sc, None)?;
            }
            inc_b.push(c.incremental());
            inc_b.last_mut().unwrap().state_restore(i.state_snapshot());
            inc_a.push(i);
        }

        let assert_pools = |a: &Option<GroupPool>, b: &Option<GroupPool>, tag: &str| match (a, b) {
            (None, None) => {}
            (Some(ga), Some(gb)) => {
                assert_eq!(ga.positions, gb.positions, "{tag}: positions");
                let ka = ga.pool_kv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let kb = gb.pool_kv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(ka, kb, "{tag}: pool_kv bytes");
                let sa = ga
                    .pool_score
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let sb = gb
                    .pool_score
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(sa, sb, "{tag}: pool_score bytes");
            }
            _ => panic!("{tag}: one path emitted a pool, the other did not"),
        };

        for (round, lens) in [(1, &lens1), (2, &lens2)] {
            let rows: Vec<(Tensor, Tensor)> = lens
                .iter()
                .map(|&n| {
                    let x = Tensor::randn(0f32, 1.0, (n, dim), &dev)?;
                    c.project_rows(&x)
                })
                .collect::<Result<_>>()?;
            // Per-seq reference.
            let gps_a: Vec<Option<GroupPool>> = inc_a
                .iter_mut()
                .zip(&rows)
                .map(|(i, (kv, sc))| i.assemble_groups(kv, sc))
                .collect::<Result<_>>()?;
            // Batched.
            let inputs: Vec<SeqAssemble> = inc_b
                .iter()
                .zip(&rows)
                .map(|(i, (kv, sc))| i.assemble_input(kv, sc))
                .collect::<Result<_>>()?;
            let outs = assemble_groups_batched(&inc_b[0], &inputs)?;
            for (i, (gp_b, upd)) in outs.into_iter().enumerate() {
                assert_pools(
                    &gps_a[i],
                    &gp_b,
                    &format!("ratio={ratio} round={round} seq={i}"),
                );
                inc_b[i].assemble_apply(upd);
            }
        }
        Ok(())
    }

    #[test]
    fn assemble_batched_matches_per_seq() -> Result<()> {
        assemble_batched_matches_per_seq_case(4, 6)?; // overlapping
        assemble_batched_matches_per_seq_case(3, 5)?; // non-overlapping
        Ok(())
    }

    /// The speculative-verify rollback contract: snapshot → absorb a whole verify
    /// block (accepted prefix + rejected draft tail, crossing a group boundary
    /// INSIDE the block — the corrupting case) → `state_restore` +
    /// `emit_groups_projected` over ONLY the accepted prefix must leave the
    /// compressor in exactly the state of never having seen the rejected tail:
    /// the replay's emissions AND every later emission (positions and bytes)
    /// match a reference stream fed the accepted prefix alone.
    fn snapshot_restore_replay_case(
        ratio: usize,
        d: usize,
        rd: usize,
        accepted: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        // Pre-rows: one full group behind (prev-group overlap populated) + a
        // partial buffer, so the 6-row block crosses the next boundary mid-block.
        let n_pre = 2 * ratio - 1;
        let block = 6usize;
        let tail = 2 * ratio;
        let pre_x = Tensor::randn(0f32, 1.0, (n_pre, dim), &dev)?;
        let blk_x = Tensor::randn(0f32, 1.0, (block, dim), &dev)?;
        let cont_x = Tensor::randn(0f32, 1.0, (tail, dim), &dev)?;
        let (kv_pre, sc_pre) = c.project_rows(&pre_x)?;
        let (kv_blk, sc_blk) = c.project_rows(&blk_x)?;
        let (kv_cont, sc_cont) = c.project_rows(&cont_x)?;

        for roped in [false, true] {
            let rope_arg = if roped { Some(&rope) } else { None };
            // Per-row feed helper (matches the live decode/prefill streaming path).
            let feed = |inc: &mut IncrementalCompressor,
                        kv: &Tensor,
                        sc: &Tensor,
                        range: std::ops::Range<usize>,
                        ent: &mut Vec<Tensor>,
                        pos: &mut Vec<u32>|
             -> Result<()> {
                for t in range {
                    let k = kv.narrow(0, t, 1)?;
                    let s = sc.narrow(0, t, 1)?;
                    if roped {
                        if let Some(e) = inc.push_projected_roped(&k, &s, &rope)? {
                            ent.push(e.reshape((1, d))?);
                        }
                    } else if let Some((e, p)) = inc.push_projected(&k, &s)? {
                        ent.push(e.reshape((1, d))?);
                        pos.push(p);
                    }
                }
                Ok(())
            };

            // Reference: pre + accepted prefix + continuation — the rejected tail
            // never existed. Emissions collected from the accepted prefix onward.
            let mut r = c.incremental();
            let (mut r_ent, mut r_pos) = (Vec::new(), Vec::new());
            feed(
                &mut r,
                &kv_pre,
                &sc_pre,
                0..n_pre,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            feed(
                &mut r,
                &kv_blk,
                &sc_blk,
                0..accepted,
                &mut r_ent,
                &mut r_pos,
            )?;
            feed(&mut r, &kv_cont, &sc_cont, 0..tail, &mut r_ent, &mut r_pos)?;

            // Test: pre → SNAPSHOT → the whole block (draft tail absorbed) →
            // RESTORE → replay the accepted prefix (the rollback's exact call) →
            // continuation.
            let mut t_ = c.incremental();
            let (mut t_ent, mut t_pos) = (Vec::new(), Vec::new());
            feed(
                &mut t_,
                &kv_pre,
                &sc_pre,
                0..n_pre,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            let snap = t_.state_snapshot();
            feed(
                &mut t_,
                &kv_blk,
                &sc_blk,
                0..block,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            t_.state_restore(snap);
            if accepted > 0 {
                if let Some((e, p)) = t_.emit_groups_projected(
                    &kv_blk.narrow(0, 0, accepted)?,
                    &sc_blk.narrow(0, 0, accepted)?,
                    rope_arg,
                )? {
                    let g = e.dim(0)?;
                    for gi in 0..g {
                        t_ent.push(e.narrow(0, gi, 1)?);
                    }
                    if !roped {
                        t_pos.extend(p);
                    }
                }
            }
            feed(&mut t_, &kv_cont, &sc_cont, 0..tail, &mut t_ent, &mut t_pos)?;

            assert_eq!(
                r_ent.len(),
                t_ent.len(),
                "emission count (ratio={ratio}, accepted={accepted}, roped={roped})"
            );
            if !roped {
                assert_eq!(
                    r_pos, t_pos,
                    "group positions (ratio={ratio}, accepted={accepted}) — a mismatch means \
                     group_idx was not rolled back"
                );
            }
            if !r_ent.is_empty() {
                let a = Tensor::cat(&r_ent, 0)?.flatten_all()?.to_vec1::<f32>()?;
                let b = Tensor::cat(&t_ent, 0)?.flatten_all()?.to_vec1::<f32>()?;
                let max_abs = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_abs < 1e-6,
                    "restore+replay diverges from clean absorb \
                     (ratio={ratio}, accepted={accepted}, roped={roped}): max|Δ| = {max_abs}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn snapshot_restore_replay_matches_clean_absorb() -> Result<()> {
        for &ratio in &[4usize, 3] {
            let (d, rd) = if ratio == 4 { (6, 4) } else { (5, 2) };
            // accepted = 0 (all drafts rejected), 1 (boundary completes during
            // replay for ratio 4), mid, and block-1 (max partial).
            for accepted in [0usize, 1, 3, 5] {
                snapshot_restore_replay_case(ratio, d, rd, accepted)?;
            }
        }
        Ok(())
    }

    /// The prefill wave's cross-sequence emit batch: assembling each sequence's
    /// groups independently, then concatenating their [`GroupPool`]s on the group
    /// axis and calling [`Compressor::pool_and_norm`] ONCE, is BIT-IDENTICAL to
    /// pooling each sequence separately (the pool is per-group-independent). Covers
    /// overlap + non-overlap, raw + roped, and ragged per-sequence lengths.
    #[test]
    fn pool_batched_across_seqs_matches_per_seq() -> Result<()> {
        let dev = Device::Cpu;
        let (d, rd, dim) = (6usize, 4usize, 8usize);
        for &ratio in &[4usize, 3usize] {
            let coff = if ratio == 4 { 2 } else { 1 };
            let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
            let c = Compressor::new(
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
                Tensor::randn(0f32, 1.0, d, &dev)?,
                ratio,
                d,
                rd,
                1e-6,
            );
            for roped in [false, true] {
                let rope_arg = if roped { Some(&rope) } else { None };
                // Ragged: two sequences of different lengths, each completing ≥1 group.
                let lens = [ratio * 3 + 2, ratio * 2 + 1];
                let mut per_seq: Vec<Tensor> = Vec::new();
                let mut pools: Vec<GroupPool> = Vec::new();
                for &len in &lens {
                    let x = Tensor::randn(0f32, 1.0, (len, dim), &dev)?;
                    let (kv, sc) = c.project_rows(&x)?;
                    let gp = c
                        .incremental()
                        .assemble_groups(&kv, &sc)?
                        .expect("completes a group");
                    per_seq.push(c.pool_and_norm(
                        &gp.pool_kv,
                        &gp.pool_score,
                        &gp.positions,
                        rope_arg,
                    )?);
                    pools.push(gp);
                }
                // Batched: concat on the group axis + one pool.
                let pool_kv =
                    Tensor::cat(&pools.iter().map(|g| &g.pool_kv).collect::<Vec<_>>(), 0)?;
                let pool_score =
                    Tensor::cat(&pools.iter().map(|g| &g.pool_score).collect::<Vec<_>>(), 0)?;
                let positions: Vec<u32> = pools.iter().flat_map(|g| g.positions.clone()).collect();
                let batched = c.pool_and_norm(&pool_kv, &pool_score, &positions, rope_arg)?;
                // Split back per sequence and compare bit-for-bit.
                let mut off = 0usize;
                for (i, gp) in pools.iter().enumerate() {
                    let g = gp.positions.len();
                    let got = batched.narrow(0, off, g)?.flatten_all()?.to_vec1::<f32>()?;
                    off += g;
                    let want = per_seq[i].flatten_all()?.to_vec1::<f32>()?;
                    for (j, (x, y)) in want.iter().zip(&got).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "seq {i} elem {j} (ratio={ratio}, roped={roped})"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// A group's pooled entry must not depend on how many OTHER groups were in
    /// the launch — pooling alone and pooling beside 31 others must give the
    /// same bytes.
    ///
    /// This is the property that keeps decode reproducible across session mixes:
    /// a wave pools however many sessions happened to close a group that step,
    /// so if batch width perturbed the result, the same session would get
    /// different corpus entries depending on who it was batched with, and MoE
    /// top-k would turn that into different tokens.
    /// `pool_batched_across_seqs_matches_per_seq` holds the same property for the
    /// eager path but runs on the CPU, so it never exercised the kernel; the
    /// kernel earns it structurally (one block per group per channel tile, no
    /// cross-group communication) and this pins that down.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn pool_kernel_is_batch_width_invariant() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(g, p, d) in &[(32usize, 8usize, 512usize), (7, 128, 512), (5, 8, 128)] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 2.0, (g, p, d), &dev)?;
            let batched = pool_group(&kv, &score)?;
            for i in 0..g {
                let alone = pool_group(&kv.narrow(0, i, 1)?, &score.narrow(0, i, 1)?)?;
                let a = batched.narrow(0, i, 1)?.flatten_all()?.to_vec1::<f32>()?;
                let b = alone.flatten_all()?.to_vec1::<f32>()?;
                for (j, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "group {i} elem {j} changed with batch width \
                         (g={g}, p={p}, d={d}): batched {x} vs alone {y}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Isolation bench for the fused pool, at the two shapes the model actually
    /// runs it at: the decode wave's per-layer pool (a handful of groups — one
    /// per session that closed a group this step) and a prefill fleet's pool
    /// (hundreds of groups at once). The decode shape is the interesting one,
    /// because a grid of one block per group leaves most of the GPU idle there,
    /// and that is invisible at the prefill shape.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     bench_compressor_pool -- --ignored --nocapture
    ///   ncu --set full --kernel-name compressor_pool_kernel --launch-count 4 \
    ///     -o pool_ncu target/release/deps/candle_transformers-<hash>.exe \
    ///     bench_compressor_pool --ignored --nocapture
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_compressor_pool() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(g, p, d, tag) in &[
            (1usize, 8usize, 512usize, "decode ×1"),
            (8, 8, 512, "decode ×8"),
            (16, 8, 512, "decode ×16"),
            (16, 8, 128, "decode ×16 idx"),
            (256, 8, 512, "prefill fleet"),
            (16, 128, 512, "hca"),
        ] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            // Warm the kernel + allocator, then time a run of launches so the
            // per-launch cost is not swamped by one-off setup.
            for _ in 0..8 {
                let _ = pool_group(&kv, &score)?;
            }
            dev.synchronize()?;
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = pool_group(&kv, &score)?;
            }
            dev.synchronize()?;
            let fused = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // The eager chain, for the ratio this is all about.
            for _ in 0..8 {
                let w = softmax(&score, 1)?;
                let _ = kv.broadcast_mul(&w)?.sum(1)?;
            }
            dev.synchronize()?;
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                let w = softmax(&score, 1)?;
                let _ = kv.broadcast_mul(&w)?.sum(1)?;
            }
            dev.synchronize()?;
            let eager = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

            println!(
                "[pool {tag:>14}] g={g:<4} p={p:<4} d={d:<4}  fused {fused:7.2} us  \
                 eager {eager:7.2} us  {:.2}x",
                eager / fused
            );
        }
        Ok(())
    }

    /// The fused pool kernel versus the eager chain it replaces
    /// (`softmax(score, 1)` → `broadcast_mul` → `sum(1)`), with the same
    /// expression in plain host f32 as the tiebreaker.
    ///
    /// Bit-equality with the eager chain is deliberately NOT the bar, because
    /// measurement says the eager chain is the inaccurate side: it tracks exact
    /// arithmetic ~3 orders of magnitude worse than the fused kernel does (the
    /// printed table shows it). Under `--use_fast_math` it pays `__expf` and an
    /// approximate division across five kernels, each rounding an intermediate
    /// out to memory and reading it back, while the fused pass keeps them in
    /// registers. So the assertion is that the kernel is at least as faithful to
    /// exact arithmetic as what it replaces — a strictly stronger property than
    /// reproducing the eager chain's drift, and the one that matters given MoE
    /// top-k turns small numerical differences into different tokens.
    ///
    /// Sweeps the shapes the model actually pools: `P` = 8 (overlapping CSA,
    /// 2·ratio), `P` = 128 (HCA, non-overlap at ratio 128) and `P` = 5 (a partial
    /// group closed at a turn seal), against both channel widths — 512 for the
    /// attention compressor, 128 for the indexer. Scores include a fully `-inf`
    /// row, the group-0 overlap pad, which must contribute exactly zero weight.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn pool_kernel_matches_eager() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(g, p, d) in &[
            (3usize, 8usize, 512usize),
            (1, 8, 512),
            (5, 128, 512),
            (4, 8, 128),
            (2, 5, 128),
            (17, 8, 512),
        ] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 2.0, (g, p, d), &dev)?;
            // Group 0's overlap pad: the first half of the pooling axis is
            // -inf, fully masked. `expf(-inf - m)` must land on exactly 0.
            let half = p / 2;
            let masked = Tensor::cat(
                &[
                    &Tensor::full(f32::NEG_INFINITY, (1, half, d), &dev)?,
                    &score.narrow(0, 0, 1)?.narrow(1, half, p - half)?,
                ],
                1,
            )?;
            let score = Tensor::cat(&[&masked, &score.narrow(0, 1, g - 1)?], 0)?;

            // Half the shapes go through as NON-CONTIGUOUS views — the channel
            // narrow the overlap split produces, and a dim-0 slice of a wider
            // block, which is what `assemble_groups_batched` hands over. The
            // first version of this kernel assumed packed inputs and the model
            // gate caught it, not this one; these views are that gap closed.
            let strided = d >= 128;
            let (kv, score) = if strided {
                let pad = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
                (
                    Tensor::cat(&[&kv, &pad], D::Minus1)?.narrow(D::Minus1, 0, d)?,
                    Tensor::cat(&[&score, &pad], D::Minus1)?.narrow(D::Minus1, 0, d)?,
                )
            } else {
                (kv, score)
            };
            assert_eq!(
                kv.is_contiguous(),
                !strided,
                "the strided cases must actually be strided (g={g}, p={p}, d={d})"
            );

            let got = pool_group(&kv, &score)?;
            let w = softmax(&score, 1)?;
            let want = kv.broadcast_mul(&w)?.sum(1)?;
            // The same expression on the HOST, in plain IEEE f32 with no
            // fast-math and no FMA contraction — the third opinion that says
            // which of the two device paths the seam belongs to.
            let cpu = {
                let kc = kv.to_device(&Device::Cpu)?;
                let sc = score.to_device(&Device::Cpu)?;
                let wc = softmax(&sc, 1)?;
                kc.broadcast_mul(&wc)?.sum(1)?
            };

            let a = want.flatten_all()?.to_vec1::<f32>()?;
            let b = got.flatten_all()?.to_vec1::<f32>()?;
            let c = cpu.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a.len(), b.len(), "shape (g={g}, p={p}, d={d})");
            let rel = |x: f32, y: f32| (x - y).abs() / x.abs().max(y.abs()).max(1e-30);
            let (mut k_vs_e, mut k_vs_c, mut e_vs_c) = (0f32, 0f32, 0f32);
            for ((x, y), z) in a.iter().zip(&b).zip(&c) {
                k_vs_e = k_vs_e.max(rel(*x, *y));
                k_vs_c = k_vs_c.max(rel(*y, *z));
                e_vs_c = e_vs_c.max(rel(*x, *z));
            }
            println!(
                "[pool g={g} p={p} d={d}] kernel-vs-eager {k_vs_e:.3e}  \
                 kernel-vs-cpu {k_vs_c:.3e}  eager-vs-cpu {e_vs_c:.3e}"
            );
            // The kernel must be at least as faithful to exact arithmetic as the
            // eager device chain it replaces — that is the property that matters,
            // and it is stronger than agreeing with the eager chain's own drift.
            assert!(
                k_vs_c <= e_vs_c.max(1e-6),
                "kernel drifts from exact arithmetic MORE than the eager chain it \
                 replaces (g={g}, p={p}, d={d}): kernel-vs-cpu {k_vs_c:.3e} > \
                 eager-vs-cpu {e_vs_c:.3e}"
            );
        }
        Ok(())
    }

    /// The DECODE wave's cross-session emit batch: every session absorbs ONE row
    /// per step through [`IncrementalCompressor::assemble_groups`] and DEFERS its
    /// pool, then the sessions that completed a group on that step are pooled
    /// TOGETHER in a single [`Compressor::pool_and_norm`] — reproducing, per
    /// session per step, the per-row `push_projected` / `push_projected_roped`
    /// emit it replaces.
    ///
    /// Sessions are staggered so their group boundaries land on different steps —
    /// the ragged case the batch must handle, where only some sessions contribute
    /// a pool on any given step (and, at ratio 4, some steps contribute none).
    /// Running many steps also proves the retained streaming state advanced
    /// identically: a divergence in the carried buffer, the overlap `prev_*` or
    /// `group_idx` would surface as a mismatched entry on a later step.
    ///
    /// Tolerance, not bit-exactness, and only for the reason
    /// `emit_groups_batched_matches_streamed` already records: the per-row emit
    /// softmaxes dim 0 of a `[P, d]` group while the pooled path softmaxes dim 1
    /// of `[G, P, d]`, so the two reductions may differ in the last ULP. That the
    /// BATCH WIDTH itself changes nothing — session `s`'s entry is identical
    /// whether it pools alone or beside 15 others, so a wave's session mix cannot
    /// perturb any session's corpus — is the bit-exact half, held by
    /// `pool_batched_across_seqs_matches_per_seq`.
    #[test]
    fn decode_pool_batched_across_sessions_matches_per_row() -> Result<()> {
        let dev = Device::Cpu;
        let (d, rd, dim) = (6usize, 4usize, 8usize);
        let n_sess = 3usize;
        for &ratio in &[4usize, 3usize] {
            let coff = if ratio == 4 { 2 } else { 1 };
            let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
            let c = Compressor::new(
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
                Tensor::randn(0f32, 1.0, d, &dev)?,
                ratio,
                d,
                rd,
                1e-6,
            );
            for roped in [false, true] {
                let rope_arg = if roped { Some(&rope) } else { None };
                // Stagger: session `s` enters the loop having already absorbed `s`
                // rows, so the sessions' group boundaries fall on different steps.
                let mut refs: Vec<IncrementalCompressor> = Vec::with_capacity(n_sess);
                let mut bats: Vec<IncrementalCompressor> = Vec::with_capacity(n_sess);
                for s in 0..n_sess {
                    let mut inc = c.incremental();
                    if s > 0 {
                        let x = Tensor::randn(0f32, 1.0, (s, dim), &dev)?;
                        let (kv, sc) = c.project_rows(&x)?;
                        inc.emit_groups_projected(&kv, &sc, rope_arg)?;
                    }
                    let mut b = c.incremental();
                    b.state_restore(inc.state_snapshot());
                    bats.push(b);
                    refs.push(inc);
                }

                let mut want: Vec<Vec<Tensor>> = vec![Vec::new(); n_sess];
                let mut got: Vec<Vec<Tensor>> = vec![Vec::new(); n_sess];
                for _ in 0..(3 * ratio + 2) {
                    // One wave-batched projection, one row per session.
                    let x = Tensor::randn(0f32, 1.0, (n_sess, dim), &dev)?;
                    let (kv, sc) = c.project_rows(&x)?; // [n_sess, cd]

                    // Reference: each row pools inside its own push.
                    for (s, inc) in refs.iter_mut().enumerate() {
                        let (k, v) = (kv.narrow(0, s, 1)?, sc.narrow(0, s, 1)?);
                        if roped {
                            if let Some(e) = inc.push_projected_roped(&k, &v, &rope)? {
                                want[s].push(e.reshape((1, d))?);
                            }
                        } else if let Some((e, _)) = inc.push_projected(&k, &v)? {
                            want[s].push(e.reshape((1, d))?);
                        }
                    }

                    // Batched: assemble every session, then ONE pool over those
                    // that completed a group — the wave's pass 1a / pass 1b.
                    let mut pools: Vec<Option<GroupPool>> = Vec::with_capacity(n_sess);
                    for (s, inc) in bats.iter_mut().enumerate() {
                        pools.push(inc.assemble_row(&kv.narrow(0, s, 1)?, &sc.narrow(0, s, 1)?)?);
                    }
                    let live: Vec<(usize, &GroupPool)> = pools
                        .iter()
                        .enumerate()
                        .filter_map(|(s, g)| g.as_ref().map(|g| (s, g)))
                        .collect();
                    if live.is_empty() {
                        continue;
                    }
                    let pool_kv =
                        Tensor::cat(&live.iter().map(|(_, g)| &g.pool_kv).collect::<Vec<_>>(), 0)?;
                    let pool_score = Tensor::cat(
                        &live.iter().map(|(_, g)| &g.pool_score).collect::<Vec<_>>(),
                        0,
                    )?;
                    let positions: Vec<u32> =
                        live.iter().flat_map(|(_, g)| g.positions.clone()).collect();
                    let pooled = c.pool_and_norm(&pool_kv, &pool_score, &positions, rope_arg)?;
                    let mut off = 0usize;
                    for (s, g) in &live {
                        let n = g.positions.len();
                        got[*s].push(pooled.narrow(0, off, n)?);
                        off += n;
                    }
                }

                for s in 0..n_sess {
                    assert_eq!(
                        want[s].len(),
                        got[s].len(),
                        "session {s} emitted a different number of groups \
                         (ratio={ratio}, roped={roped})"
                    );
                    assert!(
                        !want[s].is_empty(),
                        "session {s} emitted nothing — the case proves nothing \
                         (ratio={ratio}, roped={roped})"
                    );
                    let a = Tensor::cat(&want[s], 0)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = Tensor::cat(&got[s], 0)?.flatten_all()?.to_vec1::<f32>()?;
                    let max_abs = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        max_abs < 1e-6,
                        "session {s}: batched decode pool vs per-row emit diverge \
                         (ratio={ratio}, roped={roped}): max|Δ| = {max_abs}"
                    );
                }
            }
        }
        Ok(())
    }

    /// `close` finalizes a trailing partial group of `n < ratio` buffered rows
    /// (non-overlapping compressor): the pooled entry is the softmax over
    /// exactly those `n` rows of `(score + ape[:n])` per channel, then RMSNorm —
    /// hand-computed scalar reference. Also checks that closing an empty buffer
    /// (right after a group emitted) yields `None`.
    #[test]
    fn close_pools_trailing_partial_nonoverlap() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (6usize, 4usize, 3usize, 2usize);
        let wkv = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, d), &dev)?;
        let norm = Tensor::randn(0f32, 1.0, d, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm.clone(),
            ratio,
            d,
            rd,
            1e-6,
        );

        // Exactly `ratio` rows: a group emits and the buffer empties, so a close
        // immediately after must return None (boundary fell on the seal).
        let x_full = Tensor::randn(0f32, 1.0, (1, ratio, dim), &dev)?;
        let mut inc0 = c.incremental();
        let mut emitted = 0;
        for t in 0..ratio {
            if inc0.push_raw(&x_full.i((0, t))?)?.is_some() {
                emitted += 1;
            }
        }
        assert_eq!(emitted, 1);
        assert!(inc0.close()?.is_none(), "empty buffer must close to None");

        // Two buffered rows (< ratio=3): no group emitted, then close.
        let n = 2usize;
        let x = Tensor::randn(0f32, 1.0, (1, n, dim), &dev)?;
        let mut inc = c.incremental();
        for t in 0..n {
            assert!(inc.push_raw(&x.i((0, t))?)?.is_none());
        }
        let (entry, pos) = inc.close()?.expect("partial rows must close to an entry");
        assert_eq!(pos, 0, "first group starts at position 0");
        let got = entry.reshape(d)?.to_vec1::<f32>()?;

        // Scalar reference: per-channel softmax over the n rows of (score+ape[:n]),
        // weighted sum of kv, then RMSNorm·norm_w.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [n, d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let normv = norm.to_vec1::<f32>()?;
        let mut pooled = vec![0f32; d];
        for (chan, p) in pooled.iter_mut().enumerate() {
            let logits: Vec<f32> = (0..n).map(|t| sc[t][chan] + apev[t][chan]).collect();
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            *p = (0..n).map(|t| exps[t] / z * kv[t][chan]).sum();
        }
        let ms: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (chan, &pv) in pooled.iter().enumerate() {
            let expect = pv * inv * normv[chan];
            assert!(
                (got[chan] - expect).abs() < 1e-4,
                "c{chan}: {} vs {expect}",
                got[chan]
            );
        }
        Ok(())
    }

    /// `close` on the overlapping (`ratio == 4`) compressor after one complete
    /// group: the partial pool includes the previous group's first-half rows
    /// (already `+ape`) alongside the buffered second-half rows — hand-computed
    /// scalar reference, then RMSNorm.
    #[test]
    fn close_pools_trailing_partial_overlap() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (6usize, 4usize, 4usize, 2usize);
        let wkv = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?;
        let norm = Tensor::randn(0f32, 1.0, d, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm.clone(),
            ratio,
            d,
            rd,
            1e-6,
        );

        // One complete group (4 rows) then a partial (2 rows), then close.
        let n = 2usize;
        let total = ratio + n; // 6
        let x = Tensor::randn(0f32, 1.0, (1, total, dim), &dev)?;
        let mut inc = c.incremental();
        let mut emitted = 0;
        for t in 0..total {
            if inc.push_raw(&x.i((0, t))?)?.is_some() {
                emitted += 1;
            }
        }
        assert_eq!(
            emitted, 1,
            "group 0 emits, the trailing 2 rows stay buffered"
        );
        let (entry, pos) = inc.close()?.expect("partial closes to an entry");
        assert_eq!(pos, ratio as u32, "group 1 starts at position ratio");
        let got = entry.reshape(d)?.to_vec1::<f32>()?;

        // Reference: pool = prev group-0 first-half rows (t=0..ratio, dims [0,d),
        // score+ape[t]) ‖ curr partial rows (t=ratio..ratio+n, dims [d,2d),
        // score+ape[within]). Softmax over the r+n rows per channel, then RMSNorm.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [total, 2d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let normv = norm.to_vec1::<f32>()?;
        let mut pooled = vec![0f32; d];
        for (chan, p) in pooled.iter_mut().enumerate() {
            let mut logits: Vec<f32> = Vec::new();
            let mut kvs: Vec<f32> = Vec::new();
            for t in 0..ratio {
                logits.push(sc[t][chan] + apev[t][chan]);
                kvs.push(kv[t][chan]);
            }
            for j in 0..n {
                let t = ratio + j;
                logits.push(sc[t][d + chan] + apev[j][d + chan]);
                kvs.push(kv[t][d + chan]);
            }
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            *p = exps.iter().zip(&kvs).map(|(&e, &k)| e / z * k).sum();
        }
        let ms: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (chan, &pv) in pooled.iter().enumerate() {
            let expect = pv * inv * normv[chan];
            assert!(
                (got[chan] - expect).abs() < 1e-4,
                "c{chan}: {} vs {expect}",
                got[chan]
            );
        }
        Ok(())
    }

    #[test]
    fn forward_shape_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (8usize, 6usize, 4usize, 4usize);
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (2, 20, dim), &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?,
            Tensor::ones(d, DType::F32, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let out = c.forward(&x, &rope)?.unwrap();
        assert_eq!(out.dims(), &[2, 5, d]); // 20/4 = 5 groups
        assert!(out
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        Ok(())
    }
}
