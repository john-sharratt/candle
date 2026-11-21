//! The KV compressor: learned gated pooling of consecutive tokens into compressed
//! KV entries.
//!
//! A compressor pools each group of `ratio` tokens into one entry. A softmax over
//! the group's score rows, plus a positional bias, weights the group's values, and
//! RMSNorm and RoPE finish the entry. With `ratio` 4 the compressor overlaps, so
//! each entry pools the previous group's first half with the current group's
//! second half. The prefill path computes every entry of a prefix at once, and the
//! incremental path emits one entry per completed group during decode; the two
//! agree numerically. Assembly and pooling are batched across sequences, so a wave
//! pays a fixed number of device operations rather than one set per sequence.
//! Entries are kept at full precision here; quantising them to storage precision
//! happens elsewhere.
//!
//! - `pool`: the softmax-weighted group pool and its online-softmax monoid.
//! - `assemble`: the batched assemble across a fleet of sequences.
//! - `incremental`: the streaming compressor for decode, with snapshot and restore.
//! - `tests_assemble`, `tests_incremental`, `tests_pool`: tests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;

#[cfg(feature = "cuda")]
use super::desc;
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

    /// The absolute positional encoding as `[ratio, cd]` — one bias row per
    /// within-group position. A view: `ape` is stored F32 at load, so the
    /// `to_dtype` is a proven no-op and the reshape is free.
    fn ape_rows(&self) -> Result<Tensor> {
        self.ape
            .to_dtype(DType::F32)?
            .reshape((self.ratio, self.cd()))
    }

    /// The channel window a pooling row occupies: an overlapping compressor
    /// pools half a row at a time (a previous group's first half, then the
    /// current group's second half), a non-overlapping one pools the whole row.
    fn pool_width(&self) -> usize {
        if self.overlap {
            self.head_dim
        } else {
            self.cd()
        }
    }

    /// Pool segments for a run of loose per-token projection rows (`[1, cd]`
    /// each) starting at within-group position 0 — ONE segment per row, because
    /// each row arrived on its own decode step from its own batched projection
    /// and consecutive rows share no stride. `off` picks the channel window
    /// ([`Self::pool_width`] wide): 0 for a previous group's first half or for
    /// the non-overlapping compressor, `head_dim` for a current group's second
    /// half.
    ///
    /// Nothing here touches the device — a segment is a narrow of a row that
    /// already exists, and `ape` rides along as a bias the pool kernel applies
    /// as it reads. This is what makes a decode step that closes a group cost
    /// zero copies.
    fn row_segs(
        &self,
        kv_rows: &[Tensor],
        sc_rows: &[Tensor],
        off: usize,
    ) -> Result<Vec<PoolSegs>> {
        let w = self.pool_width();
        let ape = self.ape_rows()?;
        kv_rows
            .iter()
            .zip(sc_rows.iter())
            .enumerate()
            .map(|(i, (kv, sc))| {
                Ok(PoolSegs {
                    kv: kv.narrow(1, off, w)?.unsqueeze(0)?,
                    score: sc.narrow(1, off, w)?.unsqueeze(0)?,
                    ape: Some(ape.narrow(0, i, 1)?.narrow(1, off, w)?),
                })
            })
            .collect()
    }

    /// One wide pool segment over a `[groups, ratio, cd]` block of assembled
    /// group rows: `ng` groups starting at `g0`, taking the `off` channel
    /// window of each row. The bulk counterpart to [`Self::row_segs`] — the
    /// rows are already strided, so one segment covers all of them.
    fn block_seg(
        &self,
        kv_g: &Tensor,
        sc_g: &Tensor,
        g0: usize,
        ng: usize,
        off: usize,
    ) -> Result<PoolSegs> {
        let w = self.pool_width();
        Ok(PoolSegs {
            kv: kv_g.narrow(0, g0, ng)?.narrow(D::Minus1, off, w)?,
            score: sc_g.narrow(0, g0, ng)?.narrow(D::Minus1, off, w)?,
            ape: Some(self.ape_rows()?.narrow(1, off, w)?),
        })
    }

    /// Spans covering `ng` groups starting at `g0` of a `[G, ratio, cd]` block
    /// of assembled group rows, with `prev` the carried previous group's rows
    /// (`None` for a true group 0).
    ///
    /// An overlapping compressor needs TWO spans because group `g0`'s previous
    /// half comes from carried state while every later group's comes from the
    /// group before it — and "the group before it" is the same block read one
    /// group earlier, so the shift the reference does with a gather is just a
    /// different narrow offset here. A true group 0 omits the previous segment
    /// entirely: the all-`−∞` pad it stands for contributes nothing.
    fn group_spans(
        &self,
        kv_g: &Tensor,
        sc_g: &Tensor,
        g0: usize,
        ng: usize,
        prev: Option<(&[Tensor], &[Tensor])>,
    ) -> Result<Vec<PoolSpan>> {
        if !self.overlap {
            return Ok(vec![PoolSpan {
                segs: vec![self.block_seg(kv_g, sc_g, g0, ng, 0)?],
            }]);
        }
        let d = self.head_dim;
        let mut first: Vec<PoolSegs> = Vec::with_capacity(self.ratio + 1);
        if let Some((pk, ps)) = prev {
            first.extend(self.row_segs(pk, ps, 0)?);
        }
        first.push(self.block_seg(kv_g, sc_g, g0, 1, d)?);
        let mut spans = vec![PoolSpan { segs: first }];
        if ng > 1 {
            spans.push(PoolSpan {
                segs: vec![
                    self.block_seg(kv_g, sc_g, g0, ng - 1, 0)?,
                    self.block_seg(kv_g, sc_g, g0 + 1, ng - 1, d)?,
                ],
            });
        }
        Ok(spans)
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
    /// MANY sequences' [`GroupPool`]s can be pooled in ONE launch (the prefill
    /// wave batches the whole prompt fleet's pools this way, the decode wave every
    /// session that closed a group). The pools are consumed as descriptor
    /// segments, so batching them costs a wider table rather than a concatenation
    /// of their rows. Bit-identical, per group, to pooling each sequence's groups
    /// separately (the softmax/weighted-sum/RMSNorm/RoPE are all
    /// per-group-independent).
    ///
    /// Returns `[ΣG, d]` — every pool's groups in order, so a caller splits the
    /// result back per sequence by its group count.
    /// `generation` is the wave's pinned-arena guard, which the descriptor table
    /// is bump-allocated from and read in place; `None` outside a wave (the
    /// streamed reference, a turn-seal `close`, the gates) stages through a
    /// persistent buffer instead. See `desc.rs`.
    pub fn pool_and_norm(
        &self,
        pools: &[&GroupPool],
        rope: Option<&RotaryCache>,
        generation: &Generation,
    ) -> Result<Tensor> {
        let positions: Vec<u32> = pools
            .iter()
            .flat_map(|p| p.positions.iter().copied())
            .collect();
        let positions = positions.as_slice();
        let g: usize = pools.iter().map(|p| p.groups()).sum();
        let d = self.head_dim;
        // Pool over each group's rows, then RMSNorm over the entries.
        let entry = pool_group(pools, d, generation)?; // [ΣG, d]
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

mod assemble;
mod incremental;
mod pool;
#[cfg(test)]
mod tests_assemble;
#[cfg(test)]
mod tests_incremental;
#[cfg(test)]
mod tests_pool;

pub use assemble::*;
pub use incremental::*;
pub use pool::*;
