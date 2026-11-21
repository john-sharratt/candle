//! The softmax-weighted pool that turns a group's rows into one compressed entry.
//!
//! A group's entry is, per channel, a softmax over its score rows plus a positional
//! bias, weighting its value rows. The accumulator `(m, l, acc)` is an associative
//! monoid, so a group may be cut at a turn seam and merged exactly. On the GPU one
//! launch pools every group of every batch, reading the rows through a descriptor
//! table rather than a concatenation, and a group's result does not depend on how
//! many other groups share the launch. The kernel writes every element of its
//! output, so that output is allocated uninitialised.

use super::*;

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
#[allow(dead_code)] // LSE-merge persistence monoid (docs/deepseek/deepseek_batched_paged_attention_plan.md); wired in a later phase
pub struct GroupPartial {
    pub(super) m: Tensor,   // [d] running per-channel max score
    pub(super) l: Tensor,   // [d] running per-channel Σ exp
    pub(super) acc: Tensor, // [d] running per-channel Σ exp·kv
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

/// `out[g, k] = Σ_p softmax_p(score_g[p, k] + ape[p, k]) · kv_g[p, k]` — the
/// pooling half of [`Compressor::pool_and_norm`], over every group of every
/// [`GroupPool`] in `pools`, in order, into one `[ΣG, d]` result.
///
/// On CUDA this is ONE launch **for the whole batch**, because the kernel takes
/// the pooling rows as a descriptor table rather than a dense block: each group
/// contributes its segments' base addresses and row strides, and the rows are
/// read where they already lie. That is what lets several sessions — each with
/// its own base pointers — pool together without a `cat`, what lets the
/// overlapping compressor pool a previous group's first-half channels alongside
/// the current group's second-half channels without materialising the join, and
/// what lets the decode streamer hold a group as `ratio` independent one-row
/// tensors that are never copied at all.
///
/// Eagerly the pool is eight launches per call, because candle's `softmax` is
/// unfused when the reduction is over a non-last dim (`max_keepdim`,
/// `broadcast_sub`, `exp`, `sum_keepdim`, `broadcast_div`), the weighted sum
/// adds a `broadcast_mul` and a `sum`, and `ape` adds a `broadcast_add` — and
/// EVERY compressor emit in the model runs it: the prefill fleet's pool, the
/// decode wave's per-layer pool, the streamed reference, and the seal `close`.
/// The eager form below is the CPU path, one group at a time (`G` is small
/// wherever it runs — the CPU arm is the reference, never the wave).
///
/// The fused result is NOT bit-equal to the eager device chain, because the
/// eager chain is the less accurate of the two: under the archive's
/// `--use_fast_math` it pays `__expf` plus an approximate division across five
/// kernels, each rounding an intermediate out to memory, where the fused pass
/// keeps them in registers. `pool_kernel_matches_eager` measures all three
/// (kernel, eager, exact host f32) and holds the kernel to being at least as
/// faithful to exact arithmetic as what it replaces.
pub(super) fn pool_group(
    pools: &[&GroupPool],
    d: usize,
    generation: &Generation,
) -> Result<Tensor> {
    let g_total: usize = pools.iter().map(|p| p.groups()).sum();
    if g_total == 0 {
        candle::bail!("pool_group: no groups to pool");
    }
    for gp in pools {
        gp.validate(d)?;
    }

    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = pools
        .iter()
        .find(|p| p.groups() > 0)
        .expect("g_total > 0 implies a non-empty pool")
        .device()
    {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        use candle_kernels::simple::compressor_pool::{
            run_compressor_pool, POOL_GROUP_WORDS, POOL_SEG_WORDS,
        };

        let dev = dev.clone();
        let device = Device::Cuda(dev.clone());
        let stream = dev.cuda_stream();

        // Part 1 is `POOL_GROUP_WORDS` per group, struct-of-arrays with stride
        // `g_total`; part 2 is `POOL_SEG_WORDS` per segment, array-of-structs,
        // appended after it. Both parts ride in one upload.
        let n_segs: usize = pools
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.groups() * s.segs.len())
            .sum();
        let mut desc = vec![0i64; POOL_GROUP_WORDS * g_total + POOL_SEG_WORDS * n_segs];
        let (head, tail) = desc.split_at_mut(POOL_GROUP_WORDS * g_total);

        // A segment's address arithmetic: the group index is baked into the
        // base pointer, so the kernel walks only the row axis.
        let seg_ptr = |t: &Tensor, g: usize| -> Result<(i64, i64)> {
            let (st, lay) = t.storage_and_layout();
            let off = lay.start_offset() + g * lay.stride()[0];
            let p = match &*st {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.slice(off..).device_ptr(&stream).0,
                _ => candle::bail!("pool_group: expected CUDA storage"),
            };
            Ok((p as i64, lay.stride()[1] as i64))
        };

        let mut col = 0usize;
        let mut seg_idx = 0usize;
        for gp in pools {
            for span in &gp.spans {
                for g in 0..span.groups() {
                    head[col] = seg_idx as i64;
                    head[g_total + col] = span.segs.len() as i64;
                    for seg in &span.segs {
                        let w = &mut tail[seg_idx * POOL_SEG_WORDS..][..POOL_SEG_WORDS];
                        let (p_kv, s_kv) = seg_ptr(&seg.kv, g)?;
                        let (p_sc, s_sc) = seg_ptr(&seg.score, g)?;
                        w[0] = p_kv;
                        w[1] = s_kv;
                        w[2] = p_sc;
                        w[3] = s_sc;
                        if let Some(ape) = &seg.ape {
                            // `ape` is `[rows, d]` — a bias per row OF THE GROUP,
                            // shared by every group, so it has no group axis to
                            // index and its row stride is its outer stride.
                            let (st, lay) = ape.storage_and_layout();
                            w[4] = match &*st {
                                Storage::Cuda(c) => {
                                    c.as_cuda_slice::<f32>()?
                                        .slice(lay.start_offset()..)
                                        .device_ptr(&stream)
                                        .0
                                }
                                _ => candle::bail!("pool_group: expected CUDA storage"),
                            } as i64;
                            w[5] = lay.stride()[0] as i64;
                        }
                        w[6] = seg.rows()? as i64;
                        seg_idx += 1;
                    }
                    col += 1;
                }
            }
        }
        debug_assert_eq!(col, g_total);
        debug_assert_eq!(seg_idx, n_segs);

        // Staged into the wave's pinned arena and read in place, not uploaded —
        // a fresh `Tensor::from_vec` measured 7.6 us against a 5.2 us kernel
        // (see `desc.rs`). Held until the launch below is enqueued.
        let staged = desc::stage(&desc, generation)?;
        let p_desc = staged.ptr();

        // The kernel writes every element of `out`, so it is allocated
        // uninitialised — a `zeros` here would be a second full-width pass over
        // the exact bytes the kernel is about to stamp (hot-path invariant 6).
        let out = Tensor::empty((g_total, d), DType::F32, &device)?;
        {
            let (s_out, _) = out.storage_and_layout();
            let p_out = match &*s_out {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream).0,
                _ => candle::bail!("pool_group: expected CUDA storage"),
            };
            unsafe {
                run_compressor_pool(
                    p_desc as *const i64,
                    p_out as *mut f32,
                    g_total as i32,
                    d as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        return Ok(out);
    }

    // CPU reference: pool each group on its own, so ragged groups need no pad.
    let mut rows: Vec<Tensor> = Vec::with_capacity(g_total);
    for gp in pools {
        for span in &gp.spans {
            for g in 0..span.groups() {
                let mut kv_parts: Vec<Tensor> = Vec::with_capacity(span.segs.len());
                let mut sc_parts: Vec<Tensor> = Vec::with_capacity(span.segs.len());
                for seg in &span.segs {
                    kv_parts.push(seg.kv.narrow(0, g, 1)?.squeeze(0)?);
                    let sc = seg.score.narrow(0, g, 1)?.squeeze(0)?;
                    sc_parts.push(match &seg.ape {
                        Some(ape) => (sc + ape)?,
                        None => sc,
                    });
                }
                let kv = Tensor::cat(&kv_parts, 0)?; // [P_g, d]
                let sc = Tensor::cat(&sc_parts, 0)?;
                let w = softmax(&sc, 0)?;
                rows.push((kv * w)?.sum(0)?.reshape((1, d))?);
            }
        }
    }
    Tensor::cat(&rows, 0)
}

/// Split a `[rows, cd]` block into per-row views. Free — every row is a `narrow`
/// of the block, which stays alive behind them.
///
/// Rows rather than the block itself, because rows are what the pool descriptor
/// consumes: the streaming path's per-token rows and a bulk path's retained
/// group both arrive at the pool as one segment each, so neither needs the other
/// shape and neither pays to build it.
pub(super) fn row_views(block: &Tensor) -> Result<Vec<Tensor>> {
    (0..block.dim(0)?).map(|i| block.narrow(0, i, 1)).collect()
}

/// [`row_views`] of a window that must OUTLIVE the buffer it came from: ONE
/// forced copy of the small block first, breaking a pin that would otherwise
/// keep a whole batched projection or fleet-wide stream alive for as long as the
/// streaming state does.
pub(super) fn retain_rows(block: &Tensor) -> Result<Vec<Tensor>> {
    if block.dim(0)? == 0 {
        return Ok(Vec::new());
    }
    row_views(&block.force_contiguous()?)
}

/// Replace NaNs with `fill` (element-wise): `where(x == x, x, fill)`.
#[allow(dead_code)]
pub(super) fn replace_nan(x: &Tensor, fill: f64) -> Result<Tensor> {
    let is_nan = x.ne(x)?; // NaN != NaN → 1
    let fill_t = Tensor::full(fill as f32, x.shape(), x.device())?;
    is_nan.where_cond(&fill_t, x)
}

/// One run of pooling rows shared by every group of a [`PoolSpan`]: group `g`'s
/// contribution is row block `g` — a `[rows, d]` slice — of each tensor, read in
/// place. `kv`/`score` are `[groups, rows, d]` views whose channel stride is 1;
/// no other stride is constrained, which is what lets a producer hand over a
/// channel-axis narrow (the overlap split), a row-axis narrow (a sequence's
/// slice of a fleet-wide block), or a single loose row without flattening it.
pub struct PoolSegs {
    /// Pool input values `[groups, rows, d]`.
    pub kv: Tensor,
    /// Pool input scores `[groups, rows, d]`, RAW — `ape` is applied as the
    /// kernel reads them, so the producer never has to keep a biased copy.
    pub score: Tensor,
    /// `[rows, d]` absolute positional bias added to `score`, broadcast over
    /// groups. `None` where the rows take none (the overlap pad).
    pub ape: Option<Tensor>,
}

impl PoolSegs {
    /// Rows this segment contributes to each group's pooling axis.
    pub(super) fn rows(&self) -> Result<usize> {
        self.kv.dim(1)
    }
}

/// A run of consecutive groups described by one set of segments. Most producers
/// emit a single span; the streamed multi-group assemble emits two, because its
/// first group's previous half comes from carried state while the rest take the
/// group before them.
pub struct PoolSpan {
    /// The row runs the pool walks per group, in order.
    pub segs: Vec<PoolSegs>,
}

impl PoolSpan {
    /// Groups this span covers.
    pub(super) fn groups(&self) -> usize {
        self.segs
            .first()
            .and_then(|s| s.kv.dims3().ok())
            .map(|(g, _, _)| g)
            .unwrap_or(0)
    }
}

/// The assembled-but-not-yet-pooled output of [`IncrementalCompressor::assemble_groups`]:
/// the completed groups' pool inputs and their start positions.
///
/// The pool inputs are **segments, not a dense block**. A group's pooling axis is
/// its segments' rows in order: two wide segments for the overlapping compressor
/// assembled in bulk (the previous group's first-half channels, then this group's
/// second-half channels), one for the non-overlapping one, and `2·ratio` one-row
/// segments for the decode streamer, which holds a group as that many
/// independently-allocated rows. A group whose previous group does not exist
/// simply omits that segment — the all-`−∞` pad it would contribute is discarded
/// by the softmax either way.
///
/// Handing [`Compressor::pool_and_norm`] the segments instead of their
/// concatenation is what removes the per-group allocate-plus-copy, and passing it
/// SEVERAL `GroupPool`s is what pools the whole fleet — each session with its own
/// base addresses — in one launch (hot-path invariants 2 and 2b).
pub struct GroupPool {
    /// Group runs, in group order; together they cover `positions`.
    pub spans: Vec<PoolSpan>,
    /// Each group's start position `group_idx·ratio`, for the RoPE.
    pub positions: Vec<u32>,
}

impl GroupPool {
    /// Groups assembled here — one pooled corpus entry each.
    pub fn groups(&self) -> usize {
        self.positions.len()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn device(&self) -> &Device {
        self.spans
            .first()
            .and_then(|s| s.segs.first())
            .map(|s| s.kv.device())
            .expect("a non-empty GroupPool always carries a segment")
    }

    /// A pool over one already-dense `[groups, P, d]` pair — the trivial single
    /// span, single segment, no `ape` case. What a caller holding a materialised
    /// pooling block hands the pool; also what the pool's own gates are written
    /// against, since it exercises the descriptor path with the simplest
    /// possible table.
    pub fn dense(kv: Tensor, score: Tensor, positions: Vec<u32>) -> Self {
        Self {
            spans: vec![PoolSpan {
                segs: vec![PoolSegs {
                    kv,
                    score,
                    ape: None,
                }],
            }],
            positions,
        }
    }

    /// Each group's pooling rows as one dense `[P_g, d]` pair, `ape` applied —
    /// the block this pool WOULD have been concatenated into. Reference only:
    /// it is what lets a gate compare two assemble paths' pool inputs directly,
    /// and nothing on the hot path builds it. `P_g` varies between groups when
    /// one of them has no previous group.
    pub fn materialize(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out = Vec::with_capacity(self.groups());
        for span in &self.spans {
            for g in 0..span.groups() {
                let mut kv_parts = Vec::with_capacity(span.segs.len());
                let mut sc_parts = Vec::with_capacity(span.segs.len());
                for seg in &span.segs {
                    kv_parts.push(seg.kv.narrow(0, g, 1)?.squeeze(0)?);
                    let sc = seg.score.narrow(0, g, 1)?.squeeze(0)?;
                    sc_parts.push(match &seg.ape {
                        Some(ape) => (sc + ape)?,
                        None => sc,
                    });
                }
                out.push((Tensor::cat(&kv_parts, 0)?, Tensor::cat(&sc_parts, 0)?));
            }
        }
        Ok(out)
    }

    /// Check the invariants `pool_group`'s descriptor table depends on: spans
    /// that cover exactly `positions`, matching `[groups, rows, d]` shapes, and
    /// unit channel stride (the kernel adds a channel index straight to a row
    /// address).
    pub(super) fn validate(&self, d: usize) -> Result<()> {
        let mut covered = 0usize;
        for span in &self.spans {
            if span.segs.is_empty() {
                candle::bail!("GroupPool: span with no segments");
            }
            let g = span.groups();
            covered += g;
            for seg in &span.segs {
                let (gk, rk, dk) = seg.kv.dims3()?;
                if seg.score.dims3()? != (gk, rk, dk) {
                    candle::bail!(
                        "GroupPool: segment kv {:?} and score {:?} must have the same shape",
                        seg.kv.dims(),
                        seg.score.dims()
                    );
                }
                if gk != g || dk != d {
                    candle::bail!(
                        "GroupPool: segment {:?} does not match groups={g} d={d}",
                        seg.kv.dims()
                    );
                }
                if rk == 0 {
                    candle::bail!("GroupPool: empty segment (groups={g}, d={d})");
                }
                if let Some(ape) = &seg.ape {
                    if ape.dims2()? != (rk, dk) {
                        candle::bail!(
                            "GroupPool: ape {:?} does not match segment {:?}",
                            ape.dims(),
                            seg.kv.dims()
                        );
                    }
                    if ape.stride()[1] != 1 {
                        candle::bail!("GroupPool: ape channel stride {} != 1", ape.stride()[1]);
                    }
                }
                for t in [&seg.kv, &seg.score] {
                    if t.stride()[2] != 1 {
                        candle::bail!("GroupPool: segment channel stride {} != 1", t.stride()[2]);
                    }
                }
            }
        }
        if covered != self.groups() {
            candle::bail!(
                "GroupPool: spans cover {covered} groups, positions has {}",
                self.groups()
            );
        }
        Ok(())
    }
}
