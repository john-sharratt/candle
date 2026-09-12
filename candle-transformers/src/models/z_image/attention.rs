//! Z-Image's attention, and the reference a faster one has to match.
//!
//! Split out of [`super::quantized_model`] because it is the part worth
//! replacing and therefore the part that needs an oracle. Everything here
//! operates on `[batch, heads, seq, head_dim]` and computes plain
//! `softmax(q·kᵀ/√d)·v` — **no mask of any kind**.
//!
//! # Why there is no mask
//!
//! Not an omission. Z-Image's transformer is a bidirectional DiT: the sequence
//! is image patches followed by caption tokens, every patch attends to every
//! other patch *and* to the whole caption, and the caption attends back. There
//! is no order along it to be causal about. That single fact is why none of the
//! fork's existing attention kernels fit — `paged_prefill_int8` computes its
//! horizon as `prefix_len + token + 1`, which is causality built into the tile
//! loop rather than passed in.
//!
//! # The two implementations
//!
//! [`reference`] materialises the whole `[b, h, s, s]` score matrix. It is the
//! oracle: obvious, and at 1024×1024 it wants a gibibyte for the scores alone.
//!
//! [`banded`] is what the model runs. It is the *same arithmetic*, computed a
//! band of query rows at a time — with no mask and no cross-band term, a band of
//! queries against all keys is exactly those rows of the full product, and the
//! softmax normalises along the key axis, which every band holds in full. So
//! this is a tiling, not an online-softmax approximation, and needs no rescaling
//! between bands. What it buys is peak memory, not time.

use candle::{DType, Result, Tensor, D};

/// How many query rows one attention band covers.
///
/// 1,024 puts a band's scores at `30 × 1024 × seq` — 254 MiB at 1024×1024,
/// against 1 GiB for the whole matrix, so the peak drops by four with the two
/// matmuls still shaped like GEMMs rather than a stack of thin ones. Measured on
/// the 3090: identical wall time to computing the matrix whole (12.80 s against
/// 12.77 s over eight denoise steps), so the memory is free. Narrower bands are
/// not — 512 costs 2%, which is where the per-band launches start to show.
pub const ATTN_Q_BAND: usize = 1024;

/// `1/√head_dim`, the scale that belongs on `q`.
///
/// On `q` rather than on the scores, which is the same arithmetic three orders
/// of magnitude apart in cost: `q` is `[b, h, s, 128]` and the scores are
/// `[b, h, s, s]`, so at 1024×1024 that is 31 MiB against 1 GiB — and scaling
/// afterwards is a whole extra allocation plus a read and a write of the larger
/// one, per block, per step.
pub fn scale_of(head_dim: usize) -> f64 {
    1f64 / (head_dim as f64).sqrt()
}

/// The oracle: `softmax(q·kᵀ)·v` with the whole score matrix materialised.
///
/// `q` is expected pre-scaled (see [`scale_of`]), so this is exactly the
/// operation a fused kernel replaces and nothing else. Every input is
/// `[b, h, s, d]` and the result is too.
///
/// Correct rather than fast, and deliberately so: it exists to be *believed*,
/// and every shortcut a faster implementation takes is measured against it.
pub fn reference(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    // Softmaxed at the score matrix's own width. The bf16 kernel accumulates its
    // max and its exponent sum in f32 (`SOFTMAX_OP(__nv_bfloat16, float, …)` in
    // `reduce.cu`), so widening first would buy nothing and cost two full passes
    // over a tensor that is 1 GiB at 1024×1024.
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(v)
}

/// [`reference`], a band of query rows at a time.
///
/// `q` is expected pre-scaled. `kt` is `k` already transposed to `[b, h, d, s]`
/// and made contiguous — taken as an argument because it is loop-invariant and
/// transposing it inside would be a copy per band.
pub fn banded(q: &Tensor, kt: &Tensor, v: &Tensor, band: usize) -> Result<Tensor> {
    let seq = q.dim(2)?;
    if band == 0 {
        candle::bail!("attention band must be at least one row");
    }
    let mut bands = Vec::with_capacity(seq.div_ceil(band));
    for start in (0..seq).step_by(band) {
        let rows = band.min(seq - start);
        let scores = q.narrow(2, start, rows)?.matmul(kt)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        bands.push(probs.matmul(v)?);
    }
    if bands.len() == 1 {
        return Ok(bands.remove(0));
    }
    Tensor::cat(&bands, 2)
}

/// The attention this model runs.
///
/// int8 on a card that has the tensor-core MMA, [`banded`] bf16 otherwise —
/// dispatched on the same `Int8Mode` the weights are, so a deployment gets one
/// numeric mode throughout rather than int8 projections feeding a bf16
/// attention. `Off` is not a legacy path kept alive; it is what a card without
/// the instruction can do, exactly as it is for `QMatMul`.
///
/// `q` arrives **unscaled**: the int8 path folds `1/√d` in before it takes the
/// row amax, which is where it belongs, and this scales for the bf16 path so the
/// two have the same contract.
pub fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mode: candle::quantized::Int8Mode,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if mode.is_int8() {
        return super::attention_int8::attention(q, k, v);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = mode;
    let head_dim = q.dim(D::Minus1)?;
    let q = (q * scale_of(head_dim))?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    banded(&q, &kt, v, ATTN_Q_BAND)
}

/// The int8 grid an MMA-based attention would run, simulated in float.
///
/// **This exists to answer one question before a kernel is written**: what does
/// `paged_prefill_int8`'s quantization grid cost *this* model? Every step below
/// quantizes to int8 and immediately back, so the arithmetic still runs in bf16
/// and nothing here is fast — what it reproduces exactly is the *information*
/// an int8 kernel would have, which is the only part that decides the answer.
///
/// The grid, from that kernel's header:
///
/// | operand | scale |
/// |---|---|
/// | Q | per (row, 32-dim window) |
/// | K | per (token, 32-dim window), post-RoPE |
/// | P | per row, **fixed** 1/127 — the online softmax already puts it in (0,1] |
/// | V | per dim, over the tile |
///
/// The window is 32 because that is `mma_int8_m16n8k32`'s K: one window per MMA
/// is what lets each one take a fresh int32 accumulator and get its own scale
/// pair folded in f32.
#[derive(Clone, Copy, Debug)]
pub struct Int8Grid {
    /// Elements per Q/K scale along `head_dim`. 32 is the MMA's K; `head_dim`
    /// makes it one scale per row, which is coarser but lets an int32
    /// accumulator run the whole dot before a single fixup.
    pub window: usize,
    /// Subtract K's per-channel mean before quantizing.
    ///
    /// **Exactly free.** Replacing `k` with `k − μ` shifts every score in a row
    /// by the same `q·μ`, and softmax is invariant to a per-row shift — so there
    /// is no correction term to add back and the online max is unperturbed. It
    /// is the standard fix for K's channel-aligned outliers, which are what
    /// waste int8's range in attention.
    pub center_k: bool,
    /// Subtract V's per-channel mean before quantizing, and add it back after.
    ///
    /// **Exactly recoverable**, for a different reason: the softmax rows sum to
    /// one, so `P(V − μ) = PV − μ`. One add at the epilogue.
    ///
    /// Both centerings need the row to attend over every key — true here only
    /// because there is no mask. With one, `μ` would differ per row and neither
    /// identity would hold.
    pub center_v: bool,
}

impl Int8Grid {
    /// The paged kernel's own grid, with both centerings.
    pub fn windowed() -> Self {
        Self {
            window: 32,
            center_k: true,
            center_v: true,
        }
    }
}

/// Quantize `t`'s last axis to int8 on a `window`-sized grid, and back.
///
/// Symmetric, one f16-rounded scale per window, values clamped to ±127 — which
/// is Q8_0's encoding exactly, laid out for the MMA rather than for storage.
fn fake_int8_windowed(t: &Tensor, window: usize) -> Result<Tensor> {
    let dims = t.dims().to_vec();
    let last = *dims.last().expect("tensor has no axes");
    if !last.is_multiple_of(window) {
        candle::bail!("int8 window {window} does not divide a {last}-wide axis");
    }
    let mut grid = dims.clone();
    grid.pop();
    grid.push(last / window);
    grid.push(window);

    let f = t.to_dtype(DType::F32)?.reshape(grid)?;
    // `amax == 0` would divide by zero; the kernel takes the same branch by
    // storing a zero scale, which dequantizes the window back to zeros.
    let amax = f.abs()?.max_keepdim(D::Minus1)?;
    let scale = (amax / 127.0)?;
    let safe = scale.clamp(f32::MIN_POSITIVE, f32::INFINITY)?;
    let q = f
        .broadcast_div(&safe)?
        .round()?
        .clamp(-127.0, 127.0)?
        .broadcast_mul(&scale)?;
    q.reshape(dims)?.to_dtype(t.dtype())
}

/// Subtract the per-channel mean over tokens: `[b, h, s, d]` → mean over `s`.
fn center(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let mu = t.to_dtype(DType::F32)?.mean_keepdim(2)?;
    let centred = t.to_dtype(DType::F32)?.broadcast_sub(&mu)?;
    Ok((centred.to_dtype(t.dtype())?, mu))
}

/// [`reference`], with every operand carrying an int8 grid's information.
///
/// `q` is expected pre-scaled, as everywhere else here. The result is what an
/// int8 kernel on this grid would produce up to accumulation order.
pub fn reference_int8_sim(q: &Tensor, k: &Tensor, v: &Tensor, g: Int8Grid) -> Result<Tensor> {
    let (k, _) = if g.center_k {
        center(k)?
    } else {
        (k.clone(), k.clone())
    };
    let (v_c, v_mu) = if g.center_v {
        center(v)?
    } else {
        (v.clone(), v.clone())
    };

    let qq = fake_int8_windowed(q, g.window)?;
    let kq = fake_int8_windowed(&k, g.window)?;
    // V's scale is per channel over the whole run, which for `[b, h, s, d]` is
    // the last axis taken whole — one scale per `d`, not per window of it. The
    // transpose is how that axis becomes the one `fake_int8_windowed` grids.
    let vq = fake_int8_windowed(&v_c.transpose(2, 3)?.contiguous()?, v_c.dim(2)?)?
        .transpose(2, 3)?
        .contiguous()?;

    let scores = qq
        .matmul(&kq.transpose(2, 3)?.contiguous()?)?
        .to_dtype(DType::F32)?;

    // **The softmax is written out rather than called, because what gets
    // quantized is the un-normalised exponential.**
    //
    // `p = exp(score − rowmax)` has a row maximum of exactly 1.0 by
    // construction, which is what makes the fixed 1/127 scale well-conditioned
    // whether attention is peaked or diffuse. The *normalised* probability is
    // not: over 4,128 keys a diffuse row sits near `1/4128 = 0.00024`, and
    // `round(0.00024 × 127)` is zero — every probability in the row would
    // vanish. Quantizing the wrong one of these two is a 74% error against 0.5%,
    // and the kernel is explicit about which: `l_add[row] += p0` accumulates the
    // exact float while `rintf(p0 * 127.f)` quantizes only the MMA's operand.
    let m = scores.max_keepdim(D::Minus1)?;
    let p = scores.broadcast_sub(&m)?.exp()?;
    // The normaliser is the *exact* sum, not the sum of the quantized values —
    // the kernel divides by `l_run`, which it accumulated in f32 before
    // rounding anything.
    let l = p.sum_keepdim(D::Minus1)?;
    let pq = (&p * 127.0)?.round()?.clamp(0.0, 127.0)?;
    let pq = (pq / 127.0)?.to_dtype(q.dtype())?;

    let out = pq.matmul(&vq)?.to_dtype(DType::F32)?.broadcast_div(&l)?;
    if g.center_v {
        out.broadcast_add(&v_mu)?.to_dtype(q.dtype())
    } else {
        out.to_dtype(q.dtype())
    }
}

/// One attention call's shape, as the transformer actually runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl AttnShape {
    /// The shape Z-Image-Turbo attends over for a square image of `side` pixels.
    ///
    /// The sequence is the *padded* run the model builds: the image is patched
    /// to `(side/8/2)²` tokens and the caption is padded to a multiple of 32,
    /// with the two concatenated. `cap` is the caption's padded length — 32 for
    /// any prompt up to 32 tokens, which is every portrait prompt.
    pub fn turbo(side: usize, cap: usize) -> Self {
        let cfg = super::model::Config::turbo();
        let per_side = side / 8 / cfg.patch_size;
        let patches = per_side * per_side;
        Self {
            seq: patches + cap,
            heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        }
    }

    /// Multiply-accumulates in one call, counted as FLOPs: `q·kᵀ` and `p·v` are
    /// each `2·h·s²·d`.
    pub fn flops(&self) -> u64 {
        4 * self.heads as u64 * (self.seq as u64).pow(2) * self.head_dim as u64
    }

    /// Bytes the score matrix occupies at `dtype` — the number the whole design
    /// turns on.
    pub fn score_bytes(&self, dtype: DType) -> u64 {
        self.heads as u64 * (self.seq as u64).pow(2) * dtype.size_in_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    /// The two shapes the model is actually run at, and the caption padding both
    /// carry. These are the numbers the benchmark measures and the ones a fused
    /// kernel has to be correct at.
    const CAP: usize = 32;

    /// **The padded sequence lengths, pinned.** A fused kernel is written
    /// against a tile decomposition, and the sequence it divides is not the
    /// image's patch count — it is that plus a padded caption. Getting this
    /// wrong produces an attention that is correct for a square number of
    /// patches and wrong for every real prompt.
    #[test]
    fn the_shapes_are_the_ones_the_model_runs() {
        let small = AttnShape::turbo(512, CAP);
        assert_eq!(
            small.seq,
            32 * 32 + CAP,
            "512×512 is 1024 patches + caption"
        );
        assert_eq!(small.seq, 1056);
        let large = AttnShape::turbo(1024, CAP);
        assert_eq!(
            large.seq,
            64 * 64 + CAP,
            "1024×1024 is 4096 patches + caption"
        );
        assert_eq!(large.seq, 4128);
        for s in [small, large] {
            assert_eq!(s.heads, 30);
            assert_eq!(s.head_dim, 128);
        }
        // The claim the banding exists for: 30 × 4128² × 2 bytes, which is
        // 0.95 GiB of scores for one attention call out of thirty-four.
        assert_eq!(large.score_bytes(DType::BF16), 1_022_423_040);
    }

    /// Random `[1, h, s, d]` operands with a fixed seed, so a run is comparable
    /// with the one before it.
    fn operands(s: AttnShape, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = (1, s.heads, s.seq, s.head_dim);
        let mk = || Tensor::randn(0f32, 1f32, dims, dev)?.to_dtype(dtype);
        Ok((mk()?, mk()?, mk()?))
    }

    /// **The banded path is the reference, not an approximation of it.**
    ///
    /// With no mask, a band of query rows against all keys is exactly those rows
    /// of the full product, and each band's softmax sees the whole key axis — so
    /// the only difference between the two is which GEMM tiling cuBLAS picks for
    /// a 1,024-row operand versus a 4,128-row one. That is a rounding
    /// difference, and this pins how small.
    #[test]
    fn banding_changes_nothing_but_the_gemm_tiling() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            return Ok(());
        }
        // The small shape: the reference wants 67 MiB of scores here, against
        // 975 MiB at 1024×1024 — enough to prove the property, cheap enough to
        // hold both results at once.
        let s = AttnShape::turbo(512, CAP);
        let scale = scale_of(s.head_dim);
        let (q, k, v) = operands(s, DType::BF16, &dev)?;
        let q = (q * scale)?;
        let kt = k.transpose(2, 3)?.contiguous()?;

        let want = reference(&q, &k, &v)?.to_dtype(DType::F32)?.flatten_all()?;
        let got = banded(&q, &kt, &v, ATTN_Q_BAND)?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        let num = (&got - &want)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let den = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let rel = (num / den).sqrt();
        println!("banded vs whole-matrix reference: rel_l2 = {rel:.3e}");
        assert!(
            rel < 1e-3,
            "banding is not the same operation: rel_l2 {rel}"
        );
        Ok(())
    }

    /// What bf16 costs against an f32 evaluation of the same operands.
    ///
    /// The budget a fused kernel inherits: it may be no worse than this without
    /// the difference being a *choice* somebody made rather than the width the
    /// model already runs at.
    #[test]
    fn bf16_against_an_f32_evaluation() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            return Ok(());
        }
        let s = AttnShape::turbo(512, CAP);
        let scale = scale_of(s.head_dim);
        let (q, k, v) = operands(s, DType::F32, &dev)?;
        let qs = (q.clone() * scale)?;
        let want = reference(&qs, &k, &v)?.flatten_all()?;

        let qb = qs.to_dtype(DType::BF16)?;
        let kb = k.to_dtype(DType::BF16)?;
        let vb = v.to_dtype(DType::BF16)?;
        let got = reference(&qb, &kb, &vb)?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        let num = (&got - &want)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let den = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let rel = (num / den).sqrt();
        println!("bf16 attention vs f32: rel_l2 = {rel:.4}");
        assert!(
            rel < 0.02,
            "bf16 attention is further off f32 than expected: {rel}"
        );
        Ok(())
    }

    /// **What the int8 grid costs, before any of it is written in CUDA.**
    ///
    /// The number that decides whether an int8 attention kernel is worth
    /// building: every variant is measured against the same f32 oracle the bf16
    /// path is measured against, so the 0.53% bf16 already costs is the yardstick
    /// rather than an abstraction.
    ///
    /// Four rows, and the interesting ones are the middle two — centering K and
    /// V is exactly free (a per-row score shift softmax ignores, and a mean the
    /// epilogue adds back), so if it is worth anything at all it is worth taking.
    #[test]
    fn what_the_int8_grid_costs_against_f32() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            return Ok(());
        }
        let s = AttnShape::turbo(512, CAP);
        let scale = scale_of(s.head_dim);
        // f32 operands, so the oracle is the operation rather than a width.
        let (q, k, v) = operands(s, DType::F32, &dev)?;
        let q = (q * scale)?;
        let want = reference(&q, &k, &v)?.flatten_all()?;
        let rel = |got: Tensor| -> Result<f32> {
            let got = got.to_dtype(DType::F32)?.flatten_all()?;
            let num = (&got - &want)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let den = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
            Ok((num / den).sqrt())
        };

        // The yardstick: what the model already accepts.
        let qb = q.to_dtype(DType::BF16)?;
        let kb = k.to_dtype(DType::BF16)?;
        let vb = v.to_dtype(DType::BF16)?;
        println!(
            "  bf16 (what runs today)      rel_l2 = {:.4}",
            rel(reference(&qb, &kb, &vb)?)?
        );

        for (label, g) in [
            (
                "int8 w32, no centering    ",
                Int8Grid {
                    window: 32,
                    center_k: false,
                    center_v: false,
                },
            ),
            (
                "int8 w32, centre K        ",
                Int8Grid {
                    window: 32,
                    center_k: true,
                    center_v: false,
                },
            ),
            ("int8 w32, centre K and V  ", Int8Grid::windowed()),
            (
                "int8 per-row (w128), both ",
                Int8Grid {
                    window: 128,
                    center_k: true,
                    center_v: true,
                },
            ),
        ] {
            let got = reference_int8_sim(&qb, &kb, &vb, g)?;
            println!("  {label}  rel_l2 = {:.4}", rel(got)?);
        }
        Ok(())
    }

    /// **The baseline.** What Z-Image's attention costs today, per call and per
    /// denoise step, at both sizes the model is run at.
    ///
    /// Reported as achieved TFLOP/s and GB/s so the number means something on a
    /// card other than the one it was measured on. For context, the RTX 3090
    /// (GA102) ceilings are 35.6 TFLOP/s for bf16 with f32 accumulate — consumer
    /// Ampere halves tensor throughput for f32 accumulation, which is what cuBLAS
    /// uses — and 936 GB/s.
    ///
    /// Asserts nothing. A benchmark that fails on a slower machine is a test
    /// nobody can run; what this produces is a measurement to put beside the
    /// next one.
    #[test]
    #[ignore = "GPU benchmark: the z-image attention baseline; run with --ignored --nocapture"]
    fn attention_baseline_at_z_image_size() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() {
            println!("no CUDA device; nothing to measure");
            return Ok(());
        }
        const WARMUP: usize = 3;
        const ITERS: usize = 10;
        // **Thirty-two calls of this shape per step, to 0.1%.** The model runs
        // thirty-four: thirty full-width blocks over the whole `seq`, two noise
        // refiners over the image run alone, and two context refiners over the
        // padded caption. Attention costs `seq²`, so at 1024×1024 that is
        // `30·4128² + 2·4096² + 2·32² = 544.8M` against `32·4128² = 545.3M` —
        // the refiners over the image are all but a full-width block, and the
        // ones over a 32-token caption are nothing.
        const BLOCKS_AT_FULL_SEQ: usize = 32;

        println!(
            "{:<10} {:>6} {:>10} {:>9} {:>9} {:>10} {:>11}",
            "image", "seq", "scores", "ms/call", "TFLOP/s", "GB/s", "ms/step"
        );
        for side in [512usize, 1024] {
            let s = AttnShape::turbo(side, CAP);
            let scale = scale_of(s.head_dim);
            let (q, k, v) = operands(s, DType::BF16, &dev)?;
            let q = (q * scale)?;
            let kt = k.transpose(2, 3)?.contiguous()?;

            for _ in 0..WARMUP {
                let _ = banded(&q, &kt, &v, ATTN_Q_BAND)?;
            }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = banded(&q, &kt, &v, ATTN_Q_BAND)?;
            }
            dev.synchronize()?;
            let per_call = t0.elapsed().as_secs_f64() / ITERS as f64;

            // Traffic the score matrix alone costs: written by q·kᵀ, read and
            // written by the softmax, read by p·v. The operands are noise beside
            // it — q, k and v together are 94 MiB against 975 MiB of scores.
            let sb = s.score_bytes(DType::BF16) as f64;
            let traffic = 4.0 * sb;
            println!(
                "{:<10} {:>6} {:>9.0}M {:>9.2} {:>9.1} {:>10.0} {:>11.2}",
                format!("{side}×{side}"),
                s.seq,
                sb / (1 << 20) as f64,
                per_call * 1e3,
                s.flops() as f64 / per_call / 1e12,
                traffic / per_call / 1e9,
                per_call * BLOCKS_AT_FULL_SEQ as f64 * 1e3,
            );
        }
        Ok(())
    }
}
