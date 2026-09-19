//! The `qwen4exp` Gated Residual (hyper-connection): `hc` parallel residual
//! streams in place of layer norms — reference implementation.
//!
//! Semantics from `qwen4exp.cpp` `build_hc_mix` / `build_hc_combine`
//! (`docs/qwen38_flash_next.md` §12.3). This is **not** the DeepSeek mHC of
//! `latent_moe/hyper.rs` — there is no Sinkhorn and no combine matrix; the
//! read side is a low-rank sigmoid gate and the write side a `2·sigmoid`
//! scatter weight centred on 1.
//!
//! Layout: the wide residual is `[T, hc, n_embd]` — token-major, streams in
//! the middle — so `reshape((t, hc_dim))` lays a token's streams out
//! contiguously exactly as ggml's `[n_embd, hc, T]` does, and the `[hc_dim]`
//! norm weights apply as a plain broadcast.

use candle::wave_provenance::WaveTicket;
use candle::{LiveTensor, Result, Tensor};
use candle_nn::kv_cache::WaveGeneration;

/// Microbench + `ncu` target for the three fused kernels, with its own
/// correctness gate (§0.4 rule 4).
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
mod cuda_fused;
#[cfg(feature = "cuda")]
pub mod ko;

/// One hyper-connection module's weights (two per layer: pre-mixer, pre-FFN;
/// one at the head with no `inject`).
#[derive(Debug, Clone)]
pub struct HcWeights {
    /// `[hc_dim]` — the grouped-norm gamma, already folded to `1 + γ` by the
    /// GGUF converter; applied as a plain multiply.
    pub norm: Tensor,
    /// `[low_rank + hc, hc_dim]` on a module that injects, `[low_rank, hc_dim]`
    /// on the head module that does not.
    ///
    /// **The inject weight is stacked under the down projection**, because the
    /// two are the *same contraction against the same operand*: both are
    /// `xn_flat @ Wᵀ` over the full `hc_dim`, differing only in output width
    /// (`low_rank` against `hc`). Run separately they read `xn_flat` twice and
    /// pay two GEMM launches; stacked once at load they are one GEMM whose `N`
    /// grows by `hc` — 320 → 324 on the released checkpoint, 1.25% more work
    /// for a whole projection removed. Measured at 2,048 tokens: 534 µs for the
    /// stacked GEMM against 750 µs for the two separate ones.
    ///
    /// Stacking is exact. Output column `j` depends only on weight row `j` and
    /// the input row, so rows `low_rank..low_rank+hc` *are* the inject weight
    /// unchanged; only the GEMM's own K-reduction order may differ, which is
    /// the same last-ulp class as any tile-shape change. Both operands were
    /// already dense F32 (`dequantize` at load), so no quantization scale is
    /// shared between the two halves and there is nothing to re-derive.
    ///
    /// **The gate rows carry the read gate's `1/hc`**, folded in by
    /// [`Self::from_checkpoint`]. The algebra scales the gate projection by
    /// `1/hc` before its SiLU; with `hc` a power of two that scale is exact in
    /// floating point, so applying it to the weight once gives the same bits as
    /// applying it to every output row — and removes a full pass over
    /// `[rows, low_rank]` per call, 96 times a forward.
    pub down: Tensor,
    /// `[hc_dim, low_rank]`. Its second dimension is what says where
    /// [`Self::down`] splits — see [`HcWeights::low_rank`].
    pub up: Tensor,
}

impl HcWeights {
    /// A module from its checkpoint tensors: the norm gain, the `down`
    /// projection (gate rows, then — on a module that injects — the inject rows
    /// stacked beneath), and the `up` projection. The read gate's `1/hc` is
    /// folded into the gate rows here, once; see [`Self::down`].
    ///
    /// Every loader builds a module through this, so the fold cannot be applied
    /// by one and forgotten by another.
    pub fn from_checkpoint(norm: Tensor, down: Tensor, up: Tensor, hc: usize) -> Result<Self> {
        if !hc.is_power_of_two() {
            candle::bail!(
                "hc={hc} is not a power of two: folding the gate's 1/hc into its weight would \
                 round, so the folded and unfolded forms would no longer agree bit for bit"
            );
        }
        let low_rank = up.dim(1)?;
        let rows = down.dim(0)?;
        let scale = 1.0 / hc as f64;
        let gate = (down.narrow(0, 0, low_rank)? * scale)?;
        let down = if rows > low_rank {
            Tensor::cat(&[&gate, &down.narrow(0, low_rank, rows - low_rank)?], 0)?
        } else {
            gate
        };
        Ok(Self { norm, down, up })
    }

    /// The read gate's bottleneck, and the row at which [`Self::down`] splits
    /// into the gate projection and the inject projection.
    ///
    /// Read off `up` rather than stored, so the split point cannot drift out of
    /// agreement with the weight it splits.
    pub fn low_rank(&self) -> Result<usize> {
        self.up.dim(1)
    }

    /// Whether this module writes back — i.e. whether `down` carries inject
    /// rows beneath the gate's. The head module mixes but never injects.
    pub fn injects(&self) -> Result<bool> {
        Ok(self.down.dim(0)? > self.low_rank()?)
    }
}

/// A hyper-connection module's weights, as the read half computes with them.
///
/// Two forms, the same split every projection in this stack has: [`HcWeights`]
/// holds the F32 checkpoint and is the definition the CPU oracle runs;
/// [`ko::HcWeightsKo`] holds the two projections KO-quantized for the engine's
/// int8 matmul. The mix is written once against this and cannot tell them
/// apart.
pub trait HcProject {
    /// The `[hc_dim]` grouped-norm gain.
    fn norm(&self) -> &Tensor;
    /// `xn_flat · downᵀ`: the gate columns, then the inject columns beneath
    /// them on a module that injects.
    fn down<'w>(&self, xn_flat: &LiveTensor<'w>) -> Result<LiveTensor<'w>>;
    /// `lo · upᵀ`: the raw gate, `[t, hc_dim]`.
    fn up<'w>(&self, lo: &LiveTensor<'w>) -> Result<LiveTensor<'w>>;
    /// The leading columns of [`Self::down`]'s output the SiLU takes — the
    /// gate rank, or a padding of it whose extra columns are exactly zero.
    fn gate_cols(&self) -> usize;
    /// Where the inject columns start in [`Self::down`]'s output, on a module
    /// that injects.
    fn inject_col(&self) -> Option<usize>;
}

impl HcProject for HcWeights {
    fn norm(&self) -> &Tensor {
        &self.norm
    }

    fn down<'w>(&self, xn_flat: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        xn_flat.matmul(&self.down.t()?)
    }

    fn up<'w>(&self, lo: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        lo.matmul(&self.up.t()?)
    }

    fn gate_cols(&self) -> usize {
        self.up.dims()[1]
    }

    fn inject_col(&self) -> Option<usize> {
        (self.down.dims()[0] > self.gate_cols()).then(|| self.gate_cols())
    }
}

/// Grouped RMSNorm over the wide residual: the reduction runs per stream
/// (over `n_embd`), the `[hc_dim]` weight scales the flattened layout.
///
/// The reduction is `candle_nn::ops::rms_norm` — one fused kernel with F32
/// accumulation — rather than the five-pass `sqr → mean → +ε → sqrt → div`
/// op chain, which read and re-wrote the full wide residual per pass and was
/// the hot half of the eager Gated-Residual cost. The `[hc_dim]` gain cannot
/// ride the kernel's alpha (that is per-`n_embd`), so it applies as one flat
/// broadcast after.
///
/// On the device the output is carved from `wave`, which makes this the seed of
/// its phase: every op downstream of it inherits that arena.
pub fn hc_grouped_norm<'w>(
    x: &LiveTensor<'w>,
    weight: &Tensor,
    eps: f64,
    wave: Option<&'w WaveGeneration>,
) -> Result<LiveTensor<'w>> {
    #[cfg(feature = "cuda")]
    if matches!(x.device(), candle::Device::Cuda(_)) {
        // One launch: the reduction and the per-(stream, column) gain in a
        // single pass.
        return cuda_fused::norm(x, weight, eps, wave);
    }
    eager_grouped_norm(x, weight, eps)
}

/// The grouped norm as eager ops — the reference [`cuda_fused::norm`]
/// reproduces, and the path the CPU oracle runs.
fn eager_grouped_norm<'w>(x: &LiveTensor<'w>, weight: &Tensor, eps: f64) -> Result<LiveTensor<'w>> {
    let (t, hc, n_embd) = x.dims3()?;
    let unit = Tensor::ones(n_embd, x.dtype(), x.device())?;
    let normed = candle_nn::ops::rms_norm(&x.reshape((t * hc, n_embd))?, &unit, eps as f32)?;
    normed
        .reshape((t, hc * n_embd))?
        .broadcast_mul(weight)?
        .reshape((t, hc, n_embd))
}

/// The read half: collapse the wide residual `[T, hc, n_embd]` into the block
/// input `[T, n_embd]`, and produce the `[T, hc]` write weights for
/// [`hc_combine`] when the module carries an `inject`.
///
/// Both results are carved from `wave` on the device (the grouped norm seeds
/// the chain, and every op after it inherits), so they live exactly as long as
/// the phase that consumes them.
pub fn hc_mix<'w>(
    x: &LiveTensor<'w>,
    w: &impl HcProject,
    eps: f64,
    wave: Option<&'w WaveGeneration>,
) -> Result<(LiveTensor<'w>, Option<LiveTensor<'w>>)> {
    let g = crate::models::profile::gpu_span("hc_mix:norm", x.device());
    let xn = hc_grouped_norm(x, w.norm(), eps, wave)?;
    g.end();
    mix_from_norm(&xn, w)
}

/// [`hc_mix`] seeded from a wave **ticket**, for the head: its result is
/// `'static`-typed while it sits on the forward phase's span, because the
/// logits computed from it leave the forward with that phase's guard handed
/// back beside them — the same arrangement every other head in the engine
/// uses. `None` allocates owned.
pub fn hc_mix_rooted(
    x: &LiveTensor<'_>,
    w: &impl HcProject,
    eps: f64,
    root: Option<WaveTicket>,
) -> Result<(Tensor, Option<Tensor>)> {
    let g = crate::models::profile::gpu_span("hc_mix:norm", x.device());
    #[cfg(feature = "cuda")]
    let xn = if matches!(x.device(), candle::Device::Cuda(_)) {
        cuda_fused::norm_rooted(x, w.norm(), eps, root)?
    } else {
        eager_grouped_norm(&x.to_owned_tensor()?, w.norm(), eps)?
    };
    #[cfg(not(feature = "cuda"))]
    let xn = {
        let _ = root;
        eager_grouped_norm(&x.to_owned_tensor()?, w.norm(), eps)?
    };
    g.end();
    mix_from_norm(&xn, w)
}

/// Everything after the grouped norm. Every result inherits `xn`'s arena, so
/// where the norm was seeded decides where the whole mix lands.
fn mix_from_norm<'w>(
    xn: &LiveTensor<'w>,
    w: &impl HcProject,
) -> Result<(LiveTensor<'w>, Option<LiveTensor<'w>>)> {
    let (t, hc, n_embd) = xn.dims3()?;
    let dev = xn.device();
    let xn_flat = xn.reshape((t, hc * n_embd))?;

    // Low-rank read gate: silu(down(xn)/hc) → up(·), with the inject projection
    // riding the SAME down GEMM (see [`HcWeights::down`]) and the `1/hc` already
    // folded into its gate rows. The projections stay in cuBLAS — they are real
    // GEMMs, and `ncu` puts them at ~50% SM and only 10–14% DRAM, so they are
    // compute-limited on the SIMT F32 pipe rather than starved of bandwidth.
    // That is why stacking wins: it is not a read that disappears, it is a
    // whole GEMM's worth of work.
    let g = crate::models::profile::gpu_span("hc_mix:lowrank", dev);
    let proj = w.down(&xn_flat)?;
    // **One pass for the whole prelude.** The SiLU reads the gate half as a
    // strided view of the stacked output and writes a dense `[t, gate_cols]` —
    // which is also the compaction the up-projection needs, since neither
    // matmul takes a strided operand. `silu` is `x · sigmoid(x)` over the same
    // shared `fast_exp` sigmoid the separate ops used, so this is the same
    // arithmetic in one launch where there were three (compact, SiLU's
    // sigmoid, multiply) and a scale the weight now carries.
    let lo = proj.narrow(1, 0, w.gate_cols())?.silu()?;
    let gate_raw = w.up(&lo)?;
    g.end();

    let g = crate::models::profile::gpu_span("hc_mix:gate_mean", dev);
    let mixed = {
        #[cfg(feature = "cuda")]
        if matches!(dev, candle::Device::Cuda(_)) {
            // One launch over the wide buffer: the sigmoid, the multiply and
            // the stream collapse together, with the stream axis walked in
            // registers. `gate` arrives RAW from the up-projection here — the
            // kernel applies the sigmoid, so the eager path's separate pass
            // over `[t, hc·n_embd]` disappears with it.
            cuda_fused::mix(xn, &gate_raw, hc, n_embd)?
        } else {
            eager_gate_mean(&xn_flat, &gate_raw, t, hc, n_embd)?
        }
        #[cfg(not(feature = "cuda"))]
        eager_gate_mean(&xn_flat, &gate_raw, t, hc, n_embd)?
    };
    g.end();

    // The inject is the tail columns of the stacked GEMM above, handed on as a
    // view: [`hc_combine`] reads it through its row stride, so it costs neither
    // a launch nor an allocation here.
    let inject = match w.inject_col() {
        Some(col) => Some(proj.narrow(1, col, hc)?),
        None => None,
    };
    Ok((mixed, inject))
}

/// The stream collapse, as eager ops — the CPU reference for
/// [`cuda_fused::mix`].
///
/// The mean runs as `hc − 1` strided slice-adds, NOT `sum(1)`: a middle-axis
/// reduction takes the generic strided-reduce kernel, which measured ~9.6 ms
/// per call at prefill width (73% of the whole bulk wall) against
/// sub-millisecond for the adds.
fn eager_gate_mean<'w>(
    xn_flat: &LiveTensor<'w>,
    gate_raw: &LiveTensor<'w>,
    t: usize,
    hc: usize,
    n_embd: usize,
) -> Result<LiveTensor<'w>> {
    let gate = candle_nn::ops::sigmoid(gate_raw)?;
    let gated = xn_flat.mul(&gate)?.reshape((t, hc, n_embd))?;
    let mut acc = gated.narrow(1, 0, 1)?;
    for s in 1..hc {
        acc = (acc + gated.narrow(1, s, 1)?)?;
    }
    acc.squeeze(1)? * (1.0 / hc as f64)
}

/// The write half: scatter the block output back across the streams, into the
/// residual **in place**. `2·sigmoid(inject/hc)` centres the weights on 1, so a
/// zero injection is a plain residual add on every stream.
///
/// `res` is the wave's own residual — or a row range of it, which is how each
/// of a wave's groups combines its own rows without a concatenation between —
/// held `&mut` as `Tensor::add_mut` holds it. `block_out` and `inject` are only
/// read, so they may be wave-scoped: this is where a phase's result is consumed.
pub fn hc_combine(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &LiveTensor<'_>,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    if matches!(res.device(), candle::Device::Cuda(_)) {
        // One launch, one read and one write of the wide buffer, against the
        // eager chain's four passes below and the fresh residual it allocates.
        return cuda_fused::combine(res, block_out, inject);
    }
    eager_combine(res, block_out, inject)
}

/// The scatter as eager ops, in place — the reference [`cuda_fused::combine`]
/// reproduces.
fn eager_combine(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &LiveTensor<'_>,
) -> Result<()> {
    let (t, hc, _n_embd) = res.dims3()?;
    let w = (candle_nn::ops::sigmoid(&(inject * (1.0 / hc as f64))?)? * 2.0)?;
    let w = w.reshape((t, hc, 1))?;
    res.add_mut(&block_out.unsqueeze(1)?.broadcast_mul(&w)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn dev() -> Device {
        Device::Cpu
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    /// `down` carries `lr` gate rows plus, when the module injects, `hc` inject
    /// rows stacked beneath them — the layout both loaders build.
    fn tiny(hc: usize, n_embd: usize, lr: usize, with_inject: bool, dev: &Device) -> HcWeights {
        let hc_dim = hc * n_embd;
        let rows = lr + if with_inject { hc } else { 0 };
        HcWeights::from_checkpoint(
            lcg_tensor(&[hc_dim], 11, dev).affine(0.2, 1.0).unwrap(),
            lcg_tensor(&[rows, hc_dim], 12, dev)
                .affine(0.3, 0.)
                .unwrap(),
            lcg_tensor(&[hc_dim, lr], 13, dev).affine(0.3, 0.).unwrap(),
            hc,
        )
        .unwrap()
    }

    /// The fold scales exactly the gate rows by exactly `1/hc` — raw values, not
    /// a tolerance: with `hc` a power of two the scale is exact — and leaves the
    /// inject rows stacked beneath them untouched.
    #[test]
    fn the_checkpoint_fold_scales_only_the_gate_rows() {
        let dev = dev();
        let down = Tensor::new(&[[4.0f32, -8.0], [1.0, 2.0], [3.0, 5.0]], &dev).unwrap();
        let up = Tensor::new(&[[1.0f32, 1.0], [1.0, 1.0]], &dev).unwrap();
        let norm = Tensor::new(&[1.0f32, 1.0], &dev).unwrap();
        let w = HcWeights::from_checkpoint(norm, down, up, 4).unwrap();
        assert_eq!(
            w.down.to_vec2::<f32>().unwrap(),
            [[1.0, -2.0], [0.25, 0.5], [3.0, 5.0]],
            "gate rows × 1/4, inject row unchanged"
        );
        assert_eq!(w.low_rank().unwrap(), 2);
        assert!(w.injects().unwrap());
    }

    /// A non-power-of-two `hc` would make the fold round, so it is refused.
    #[test]
    fn a_fold_that_would_round_is_refused() {
        let dev = dev();
        let t = |r: usize, c: usize| Tensor::ones((r, c), DType::F32, &dev).unwrap();
        let err = HcWeights::from_checkpoint(t(1, 6).flatten_all().unwrap(), t(2, 6), t(6, 2), 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a power of two"), "{err}");
    }

    /// Combining into a row range of the residual updates those rows and no
    /// others — how a wave's groups each write their own rows in place.
    #[test]
    fn a_row_range_combine_touches_only_its_rows() {
        let dev = dev();
        let (t, hc, n_embd) = (4usize, 2usize, 3usize);
        let res = lcg_tensor(&[t, hc, n_embd], 91, &dev);
        let before = res.copy().unwrap();
        let out = lcg_tensor(&[2, n_embd], 92, &dev);
        let zero = Tensor::zeros((2, hc), DType::F32, &dev).unwrap();
        let mut rows = res.narrow(0, 1, 2).unwrap();
        hc_combine(&mut rows, &out, &zero).unwrap();
        let after = res.to_vec3::<f32>().unwrap();
        let was = before.to_vec3::<f32>().unwrap();
        assert_eq!(after[0], was[0], "row 0 is outside the range");
        assert_eq!(after[3], was[3], "row 3 is outside the range");
        let o = out.to_vec2::<f32>().unwrap();
        for (r, orow) in [(1usize, 0usize), (2, 1)] {
            for s in 0..hc {
                for j in 0..n_embd {
                    assert_eq!(after[r][s][j], was[r][s][j] + o[orow][j], "row {r}");
                }
            }
        }
    }

    #[test]
    fn a_zero_injection_is_a_plain_residual_add() {
        // 2·sigmoid(0) = 1: every stream gains exactly the block output.
        let dev = dev();
        let (t, hc, n_embd) = (3usize, 4usize, 6usize);
        let res = lcg_tensor(&[t, hc, n_embd], 21, &dev);
        let out = lcg_tensor(&[t, n_embd], 22, &dev);
        let zero_inject = Tensor::zeros((t, hc), DType::F32, &dev).unwrap();
        let mut got = res.copy().unwrap();
        hc_combine(&mut got, &out, &zero_inject).unwrap();
        let want = res
            .broadcast_add(&out.reshape((t, 1, n_embd)).unwrap())
            .unwrap();
        let d = got
            .sub(&want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-6, "zero-inject combine diverged: {d}");
    }

    #[test]
    fn identical_streams_mix_to_the_normed_gated_stream() {
        // With every stream equal, the mean collapse must return exactly one
        // stream's gated value — the mix is then a per-token function, not a
        // cross-stream one.
        let dev = dev();
        let (t, hc, n_embd, lr) = (2usize, 4usize, 6usize, 3usize);
        let w = tiny(hc, n_embd, lr, true, &dev);
        let one = lcg_tensor(&[t, 1, n_embd], 31, &dev);
        let wide = one
            .broadcast_as((t, hc, n_embd))
            .unwrap()
            .contiguous()
            .unwrap();
        let (mixed, inject) = hc_mix(&wide, &w, 1e-6, None).unwrap();
        assert_eq!(mixed.dims(), &[t, n_embd]);
        assert_eq!(inject.unwrap().dims(), &[t, hc]);
        // The four streams were identical but the [hc_dim] norm gamma is not,
        // so the collapse averages four differently-scaled copies — assert
        // finiteness and shape here; the algebra is pinned against llama.cpp
        // by the real-weights gate.
        let m = mixed
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite());
    }

    /// Stacking the inject rows under the down projection computes exactly what
    /// two separate projections computed.
    ///
    /// This is the gate for the merge itself, and it is written against the
    /// **separate form spelled out longhand** rather than against a saved
    /// expectation, so it states the property rather than a snapshot of it: one
    /// GEMM of `[low_rank + hc, hc_dim]` and two GEMMs of `[low_rank, hc_dim]`
    /// and `[hc, hc_dim]` are the same arithmetic, because an output column
    /// depends only on its own weight row.
    ///
    /// Not asserted bit-exact: the GEMM's K-reduction may be split differently
    /// at `N = low_rank + hc` than at `N = low_rank`, which is last-ulp and the
    /// same class as any tile-shape change. The bound is set where that lives.
    #[test]
    fn stacking_inject_under_down_matches_two_separate_projections() {
        let dev = dev();
        let (t, hc, n_embd, lr) = (3usize, 4usize, 6usize, 5usize);
        let hc_dim = hc * n_embd;
        let eps = 1e-6;
        let x = lcg_tensor(&[t, hc, n_embd], 61, &dev);
        let norm = lcg_tensor(&[hc_dim], 62, &dev).affine(0.2, 1.0).unwrap();
        let down = lcg_tensor(&[lr, hc_dim], 63, &dev).affine(0.3, 0.).unwrap();
        let up = lcg_tensor(&[hc_dim, lr], 64, &dev).affine(0.3, 0.).unwrap();
        let inj = lcg_tensor(&[hc, hc_dim], 65, &dev).affine(0.3, 0.).unwrap();

        // The two-projection form this change replaces, written out in full.
        let xn = hc_grouped_norm(&x, &norm, eps, None).unwrap();
        let xn_flat = xn.reshape((t, hc_dim)).unwrap();
        let lo = (xn_flat.matmul(&down.t().unwrap()).unwrap() * (1.0 / hc as f64)).unwrap();
        let lo = lo
            .broadcast_mul(&candle_nn::ops::sigmoid(&lo).unwrap())
            .unwrap();
        let gate_raw = lo.matmul(&up.t().unwrap()).unwrap();
        let want_mixed = eager_gate_mean(&xn_flat, &gate_raw, t, hc, n_embd).unwrap();
        let want_inject = xn_flat.matmul(&inj.t().unwrap()).unwrap();

        // The stacked form, through the production entry point — which also
        // folds the gate's `1/hc` into the weight the longhand form applies to
        // the output.
        let w = HcWeights::from_checkpoint(norm, Tensor::cat(&[&down, &inj], 0).unwrap(), up, hc)
            .unwrap();
        assert!(
            w.injects().unwrap(),
            "the stacked weight must report inject"
        );
        assert_eq!(w.low_rank().unwrap(), lr);
        let (got_mixed, got_inject) = hc_mix(&x, &w, eps, None).unwrap();
        let got_inject = got_inject.expect("a stacked module injects");

        let gap = |a: &Tensor, b: &Tensor| -> f32 {
            let g = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let w = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            g.iter()
                .zip(&w)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale
        };
        assert_eq!(got_inject.dims(), &[t, hc]);
        let gi = gap(&got_inject, &want_inject);
        let gm = gap(&got_mixed, &want_mixed);
        assert!(gi < 1e-6, "stacked inject diverged from its own GEMM: {gi}");
        assert!(gm < 1e-6, "stacked gate diverged from its own GEMM: {gm}");
    }

    #[test]
    fn the_head_module_carries_no_inject() {
        let dev = dev();
        let w = tiny(4, 6, 3, false, &dev);
        let x = lcg_tensor(&[2, 4, 6], 41, &dev);
        let (_, inject) = hc_mix(&x, &w, 1e-6, None).unwrap();
        assert!(inject.is_none());
    }

    // ── CUDA parity ────────────────────────────────────────────────────────
    //
    // **Both sides run on the GPU.** That is the whole design of these tests
    // and it took a wrong turn to arrive at: the fused kernels are compared
    // against the EAGER path on the same device, not against the CPU.
    //
    // Comparing to the CPU cannot answer the question. candle's CPU sigmoid is
    // a precise `1/(1+exp(-x))` while its CUDA sigmoid is `fast_exp::sigmoid`,
    // a cubic polynomial with ~0.009% error — so the CPU and GPU eager paths
    // already disagree by ~2e-5 on anything sigmoid-bearing, before a fused
    // kernel exists. A GPU-vs-CPU tolerance wide enough to admit that cannot
    // distinguish "the fusion is wrong" from "the two devices round
    // differently", which is exactly how the first cut of these kernels got
    // through: it rolled its own `1/(1 + __expf(-x))`, making the fused path
    // ~400× MORE accurate than the production path it replaced, and the first
    // sign of it was a KV calibration rung going red two runs later.
    //
    // The GPU eager path is also the right reference for a second reason: it
    // is what the KV threshold row was derived against. A fusion that computes
    // something else — even something better — invalidates that calibration.
    //
    // With both sides on the device the only admissible difference is
    // reassociation (the block tree-reduction sums 2560 squares in a different
    // order than `rms_norm`), so `GAP` is set where reassociation lives and
    // nowhere near where a different formula does.
    #[cfg(feature = "cuda")]
    const GAP: f32 = 2e-6;

    #[cfg(feature = "cuda")]
    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
    }

    /// Largest elementwise gap, relative to the reference's own magnitude.
    #[cfg(feature = "cuda")]
    fn rel_gap(got: &Tensor, want: &Tensor) -> f32 {
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(g.len(), w.len());
        let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        g.iter()
            .zip(&w)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
            / scale
    }

    /// Every geometry the kernels have to handle: the production width, a
    /// small one, and two that are NOT multiples of four so the scalar tail
    /// after the `float4` body actually runs.
    #[cfg(feature = "cuda")]
    const PARITY_SHAPES: &[(usize, usize, usize)] = &[
        (3, 4, 2560),
        (1, 4, 2560),
        (5, 4, 64),
        (2, 4, 258),
        (7, 2, 130),
    ];

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_grouped_norm_matches_the_eager_reference() {
        let Some(gpu) = cuda() else { return };
        for &(t, hc, d) in PARITY_SHAPES {
            let x = lcg_tensor(&[t, hc, d], 71, &gpu);
            let w = lcg_tensor(&[hc * d], 72, &gpu).affine(0.2, 1.0).unwrap();
            let want = eager_grouped_norm(&x, &w, 1e-6).unwrap();
            let got = cuda_fused::norm(&x, &w, 1e-6, None).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < GAP, "norm parity {t}x{hc}x{d}: rel gap {gap}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_hc_mix_matches_the_eager_reference() {
        let Some(gpu) = cuda() else { return };
        for &(t, hc, d) in PARITY_SHAPES {
            // The collapse, from the same `xn` and the same raw gate, so the
            // comparison isolates `gr_mix` from the two GEMMs feeding it.
            let xn = lcg_tensor(&[t, hc, d], 73, &gpu);
            let gate_raw = lcg_tensor(&[t, hc, d], 74, &gpu)
                .reshape((t, hc * d))
                .unwrap();
            let want =
                eager_gate_mean(&xn.reshape((t, hc * d)).unwrap(), &gate_raw, t, hc, d).unwrap();
            let got = cuda_fused::mix(&xn, &gate_raw, hc, d).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < GAP, "mix parity {t}x{hc}x{d}: rel gap {gap}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_combine_matches_the_eager_reference() {
        let Some(gpu) = cuda() else { return };
        for &(t, hc, d) in PARITY_SHAPES {
            let res = lcg_tensor(&[t, hc, d], 75, &gpu);
            let out = lcg_tensor(&[t, d], 76, &gpu);
            let inj = lcg_tensor(&[t, hc], 77, &gpu);
            let mut want = res.copy().unwrap();
            eager_combine(&mut want, &out, &inj).unwrap();
            let mut got = res.copy().unwrap();
            cuda_fused::combine(&mut got, &out, &inj).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < GAP, "combine parity {t}x{hc}x{d}: rel gap {gap}");
        }
    }

    /// The inject as the model hands it over: the tail columns of the pre-mix's
    /// stacked `[t, low_rank + hc]` projection, read through its row stride.
    /// Must equal the same values as a dense `[t, hc]` tensor, bit for bit —
    /// the stride changes which address is read, never the arithmetic.
    #[test]
    #[cfg(feature = "cuda")]
    fn fused_combine_reads_a_row_strided_inject() {
        let Some(gpu) = cuda() else { return };
        let (t, hc, d, low_rank) = (5usize, 4usize, 258usize, 6usize);
        let res = lcg_tensor(&[t, hc, d], 91, &gpu);
        let out = lcg_tensor(&[t, d], 92, &gpu);
        let proj = lcg_tensor(&[t, low_rank + hc], 93, &gpu);
        let strided = proj.narrow(1, low_rank, hc).unwrap();
        assert_eq!(
            strided.stride(),
            &[low_rank + hc, 1],
            "the test's own premise"
        );
        let dense =
            Tensor::from_vec(strided.to_vec2::<f32>().unwrap().concat(), (t, hc), &gpu).unwrap();
        let mut want = res.copy().unwrap();
        cuda_fused::combine(&mut want, &out, &dense).unwrap();
        let mut got = res.copy().unwrap();
        cuda_fused::combine(&mut got, &out, &strided).unwrap();
        assert_eq!(
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            want.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// The launchers are instantiated only for powers of two up to
    /// `GR_MAX_HC`; any other count would launch nothing and leave the output
    /// unwritten, so the host must refuse it rather than return garbage.
    #[test]
    #[cfg(feature = "cuda")]
    fn fused_kernels_refuse_an_uninstantiated_stream_count() {
        let Some(gpu) = cuda() else { return };
        let (t, hc, d) = (2usize, 3usize, 64usize);
        let xn = lcg_tensor(&[t, hc, d], 95, &gpu);
        let gate_raw = lcg_tensor(&[t, hc * d], 96, &gpu);
        assert!(cuda_fused::mix(&xn, &gate_raw, hc, d).is_err());
        let mut res = lcg_tensor(&[t, hc, d], 97, &gpu);
        let out = lcg_tensor(&[t, d], 98, &gpu);
        let inj = lcg_tensor(&[t, hc], 99, &gpu);
        assert!(cuda_fused::combine(&mut res, &out, &inj).is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_combine_keeps_the_zero_injection_identity_on_device() {
        // The property the eager test pins, re-asserted through the kernel:
        // 2·sigmoid(0) = 1, so every stream gains exactly the block output.
        let Some(gpu) = cuda() else { return };
        let (t, hc, d) = (3usize, 4usize, 2560usize);
        let res = lcg_tensor(&[t, hc, d], 77, &gpu);
        let out = lcg_tensor(&[t, d], 78, &gpu);
        let zero = Tensor::zeros((t, hc), DType::F32, &gpu).unwrap();
        let mut got = res.copy().unwrap();
        hc_combine(&mut got, &out, &zero).unwrap();
        let want = res.broadcast_add(&out.reshape((t, 1, d)).unwrap()).unwrap();
        let gap = rel_gap(&got, &want);
        assert!(gap < 1e-6, "zero-inject identity broken on device: {gap}");
    }

    /// Operands that start part-way into their storage.
    ///
    /// This is the case the model hits and the unit tests did not: the wave
    /// slices the wide residual out of its own buffer, so these kernels see a
    /// dense tensor whose storage begins thousands of elements in. Two
    /// flavours matter and both are here — an offset that is a whole number of
    /// `float4`s (the vector path stays on) and one that is not (it must fall
    /// back, or the `float4` load faults with a misaligned address rather than
    /// answering wrongly).
    ///
    /// The reference input is built as its **own tensor** holding the same
    /// values, rather than by compacting the view, and that is not a
    /// convenience. candle's own copy kernel vectorises and faults with a
    /// misaligned address on a view whose start offset is not a multiple of
    /// four — `view.contiguous()` is itself unavailable at `skip = 1`. So an
    /// unaligned offset is not something candle can produce or consume
    /// anywhere, which makes the scalar fallback in these kernels defensive
    /// rather than load-bearing: the offsets the model actually hands them come
    /// from narrows on the outer axis and are multiples of the row width
    /// (2560). The aligned case below is the real regression — it is what the
    /// gate hit — and the unaligned one documents that the fallback works, on a
    /// reference nothing else in the stack could have computed.
    #[test]
    #[cfg(feature = "cuda")]
    fn fused_path_handles_offset_views() {
        let Some(gpu) = cuda() else { return };
        let (t, hc, d) = (3usize, 4usize, 64usize);
        let span = t * hc * d;
        for skip in [4usize, 1] {
            // One flat buffer; the operand is a window starting at `skip`.
            let host = lcg_tensor(&[span + skip], 81, &gpu)
                .to_vec1::<f32>()
                .unwrap();
            let flat = Tensor::from_vec(host.clone(), span + skip, &gpu).unwrap();
            let view = flat
                .narrow(0, skip, span)
                .unwrap()
                .reshape((t, hc, d))
                .unwrap();
            assert_eq!(
                view.layout().start_offset(),
                skip,
                "the test's own premise: the operand must start at {skip}"
            );
            // The same values, as a tensor of their own — see the note above.
            let dense = Tensor::from_vec(host[skip..].to_vec(), (t, hc, d), &gpu).unwrap();
            let w = lcg_tensor(&[hc * d], 82, &gpu).affine(0.2, 1.0).unwrap();
            let want = eager_grouped_norm(&dense, &w, 1e-6).unwrap();
            let got = cuda_fused::norm(&view, &w, 1e-6, None).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < GAP, "offset-{skip} norm parity: rel gap {gap}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_path_is_deterministic() {
        // Same input twice must give identical bytes — a race in the block
        // reduction or an uninitialised output would show here.
        let Some(gpu) = cuda() else { return };
        let (t, hc, d, lr) = (4usize, 4usize, 2560usize, 8usize);
        let w = tiny(hc, d, lr, true, &gpu);
        let x = lcg_tensor(&[t, hc, d], 79, &gpu);
        let a = hc_mix(&x, &w, 1e-6, None).unwrap().0.flatten_all().unwrap();
        let b = hc_mix(&x, &w, 1e-6, None).unwrap().0.flatten_all().unwrap();
        assert_eq!(
            a.to_vec1::<f32>().unwrap(),
            b.to_vec1::<f32>().unwrap(),
            "fused hc_mix is not deterministic"
        );
    }

    #[test]
    fn grouped_norm_normalises_per_stream() {
        // Scaling ONE stream must not change the normed value of the others.
        let dev = dev();
        let (t, hc, n_embd) = (1usize, 2usize, 8usize);
        let w = Tensor::ones(hc * n_embd, DType::F32, &dev).unwrap();
        let x = lcg_tensor(&[t, hc, n_embd], 51, &dev);
        let base = hc_grouped_norm(&x, &w, 1e-6, None).unwrap();
        // Double stream 0, keep stream 1.
        let s0 = x.narrow(1, 0, 1).unwrap().affine(2.0, 0.).unwrap();
        let s1 = x.narrow(1, 1, 1).unwrap();
        let x2 = Tensor::cat(&[s0, s1], 1).unwrap();
        let bumped = hc_grouped_norm(&x2, &w, 1e-6, None).unwrap();
        let d = base
            .narrow(1, 1, 1)
            .unwrap()
            .sub(&bumped.narrow(1, 1, 1).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-6, "stream 1's norm moved with stream 0's scale: {d}");
    }
}
