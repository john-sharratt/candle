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

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;

use super::linear::QLinear;
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
        let kv = self.wkv.forward(&x)?; // [b,s,cd]
        let score = self.wgate.forward(&x)?;

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
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(ms + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.norm_w.to_dtype(DType::F32)?)
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
        let kv = self.wkv.forward(&x)?; // [1, cd]
        let score = self.wgate.forward(&x)?; // [1, cd]
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
pub struct GroupPartial {
    m: Tensor,   // [d] running per-channel max score
    l: Tensor,   // [d] running per-channel Σ exp
    acc: Tensor, // [d] running per-channel Σ exp·kv
}

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

/// Replace NaNs with `fill` (element-wise): `where(x == x, x, fill)`.
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

impl IncrementalCompressor {
    /// Feed one token's hidden state `x` (`[dim]` / `[1, dim]` / `[1, 1, dim]`) at the next
    /// sequence position and, when it completes a group of `ratio` tokens, return that group's
    /// compressed entry `[1, 1, d]` (post RMSNorm + RoPE). Returns `None` mid-group.
    ///
    /// `rope` must be the same `RotaryCache` the prefill path uses for this compressor.
    pub fn push(&mut self, x: &Tensor, rope: &RotaryCache) -> Result<Option<Tensor>> {
        let (kv, score) = self.c.project_row(x)?;
        self.kv_rows.push(kv);
        self.score_rows.push(score);
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
        self.kv_rows.push(kv);
        self.score_rows.push(score);
        if self.kv_rows.len() < self.c.ratio {
            return Ok(None);
        }
        Some(self.emit_group_raw()).transpose()
    }

    /// Pool → RMSNorm (NO RoPE): the position-free entry `[1, 1, d]` and its
    /// group-start position.
    fn emit_group_raw(&mut self) -> Result<(Tensor, u32)> {
        let d = self.c.head_dim;
        let r = self.c.ratio;
        let dev = self.c.device().clone();

        let kv_rows: Vec<&Tensor> = self.kv_rows.iter().collect();
        let score_rows: Vec<&Tensor> = self.score_rows.iter().collect();
        let kv_group = Tensor::cat(&kv_rows, 0)?; // [r, cd]
                                                  // ape is added to the score BEFORE pooling / the overlap split (matches `pool`).
        let ape = self.c.ape.to_dtype(DType::F32)?.reshape((r, self.c.cd()))?;
        let score_group = (Tensor::cat(&score_rows, 0)? + ape)?; // [r, cd]

        // Pool over the group's rows (overlap: prev-half ‖ curr-half over 2·r rows).
        let entry = if self.c.overlap {
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
            let kv_pool = Tensor::cat(&[&prev_kv, &curr_kv], 0)?; // [2r, d]
            let score_pool = Tensor::cat(&[&prev_score, &curr_score], 0)?;
            let w = softmax(&score_pool, 0)?;
            kv_pool.broadcast_mul(&w)?.sum(0)? // [d]
        } else {
            let w = softmax(&score_group, 0)?;
            kv_group.broadcast_mul(&w)?.sum(0)? // cd == d
        };

        // Retain this group as the next group's "prev" half, then reset the current buffer.
        self.prev_kv_group = Some(kv_group);
        self.prev_score_group = Some(score_group);
        self.kv_rows.clear();
        self.score_rows.clear();

        // RMSNorm, position carried alongside (RoPE is the caller's concern:
        // `emit_group` applies it here on the reference path; the kernel path
        // stores pre-RoPE and rotates at read).
        let g = self.group_idx;
        self.group_idx += 1;
        let entry = entry.reshape((1, 1, d))?;
        let entry = self.rms_norm_entry(&entry)?;
        Ok((entry, (g * r) as u32))
    }

    fn emit_group(&mut self, rope: &RotaryCache) -> Result<Tensor> {
        let (entry, pos) = self.emit_group_raw()?;
        let d = self.c.head_dim;
        let rd = self.c.rope_head_dim;
        let nope = entry.narrow(D::Minus1, 0, d - rd)?;
        let rope_part = entry.narrow(D::Minus1, d - rd, rd)?;
        let rope_part = rope.apply_positions(&rope_part, &[pos], false)?;
        Tensor::cat(&[&nope, &rope_part], D::Minus1)
    }

    fn rms_norm_entry(&self, x: &Tensor) -> Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(ms + self.c.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.c.norm_w.to_dtype(DType::F32)?)
    }
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
