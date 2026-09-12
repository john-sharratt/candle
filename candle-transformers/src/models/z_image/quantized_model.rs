//! The same transformer over GGUF weights.
//!
//! A separate implementation rather than a generic one, following candle's own
//! precedent in [`crate::models::flux`], because the two differ in more than
//! the type of a weight:
//!
//! | | safetensors ([`super::model`]) | GGUF (here) |
//! |---|---|---|
//! | Convention | diffusers | the original Z-Image release |
//! | Attention projections | `to_q` / `to_k` / `to_v`, separate | `qkv`, **fused** |
//! | Output projection | `to_out.0` | `out` |
//! | QK norms | `norm_q` / `norm_k` | `q_norm` / `k_norm` |
//! | Patch embed | `all_x_embedder.2-1` | `x_embedder` |
//! | Output head | `all_final_layer.2-1` | `final_layer` |
//!
//! The fused QKV is the substantive one and accounts for the tensor counts
//! exactly: 521 in the safetensors against 453 in the GGUF, a difference of two
//! per block across 34 blocks. It is also *better* — one matmul of three times
//! the width instead of three — so this path is not merely smaller weights.
//!
//! Everything about *where* things go is shared: [`super::model::Geometry`]
//! holds the patch layout, the padding and the 3-axis positions, and both
//! models use it. That is the part a mistake would silently corrupt, so there
//! is one copy of it.
//!
//! # The weights are int8, not merely quantised
//!
//! Every projection here is built through [`linear_b_mode`], so on a card with
//! the int8 MMA (Ampere and up) each one is repacked at load into its KO twin
//! and every matmul in the denoise runs q8a128 × KO on the tensor cores. Q6_K
//! and Q8_0 both sit at the top of `to_ko`'s ladder, so the twin is the same
//! width as the file — this buys the kernel without spending accuracy.
//!
//! One weight does not come along: `x_embedder` projects a 64-wide patch, and
//! the matmul tiles K in blocks of 128. `QMatMul` tests that per tensor and
//! leaves it on the standard path reporting `Off`; it is one projection of 453,
//! run once per step rather than per block.

use candle::quantized::{Int8Mode, SumScale};
use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{ops::silu, Module, RmsNorm};

use super::attention::attend;
use super::model::{layer_norm_affineless, Config, Geometry, ADALN_EMBED_DIM};
use crate::quantized_nn::{linear_b_mode, Linear};
use crate::quantized_var_builder::VarBuilder;

/// [`linear_b_mode`], with **this model's q8a128 sum convention**.
///
/// **Z-Image needs [`SumScale::ByAmax`] and a language model does not.** The
/// q8a128 block header stores its per-128 `Σx` in f16, and `|Σx|` can reach
/// `128 · amax`: an LLM's block sums stay under 10³ and never come near the
/// 65504 ceiling, but this model's SwiGLU intermediate reaches ≈2×10⁵. The raw
/// field then stores `+inf`, every dot product touching that block becomes NaN,
/// and the only symptom is a black image — no error, no warning, nothing in the
/// operands that reads as wrong. Normalising bounds the field at 128 whatever
/// the activation's magnitude and costs no precision (f16's relative precision
/// is scale-free).
///
/// Applied to every projection rather than only to `w2`, whose operand is the
/// intermediate that overflows: the two conventions are numerically equivalent,
/// so a model-wide choice costs nothing and removes the per-layer trap of
/// getting it right on one projection and wrong on the next.
fn zi_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    mode: Int8Mode,
    dtype: DType,
    vb: VarBuilder,
) -> Result<Linear> {
    Ok(linear_b_mode(in_dim, out_dim, bias, mode, dtype, vb)?.with_sum_scale(SumScale::ByAmax))
}

/// The width every activation between the two boundary casts runs at.
///
/// **BF16, and the attention score matrix is the whole reason.** At 1024×1024
/// the sequence is 4,128 and the scores are `[1, 30, 4128, 4128]` — 2 GiB in
/// f32, written and read three times per block, thirty-four blocks per step.
/// Halving that halves the dominant memory traffic in the model and puts the two
/// attention matmuls on the bf16 tensor cores instead of the f32 pipes.
///
/// Nothing here needs f32's mantissa. The two places that would — the affineless
/// LayerNorm's mean over 3,840 terms, and the softmax's exponent sum over 4,128 —
/// both accumulate in f32 internally regardless of what they are handed
/// (`layer_norm_affineless`, and `SOFTMAX_OP(__nv_bfloat16, float, …)` in
/// `reduce.cu`). And bf16 carries f32's *exponent*, which matters here more than
/// usual: this model's sandwich norms let intermediates reach 10⁵, where f16
/// would overflow.
const DTYPE: DType = DType::BF16;

/// An RMSNorm whose weight comes from a GGUF.
///
/// The quantized `VarBuilder` hands back a `QTensor`; a norm's weight is a
/// single vector that every token is multiplied by, so it is dequantised once
/// at load rather than per forward.
fn rms(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let w = vb.get(dim, "weight")?.dequantize(vb.device())?;
    Ok(RmsNorm::new(w.to_dtype(DTYPE)?, eps))
}

/// A plain tensor from a GGUF — the learned pad embeddings, which are added to
/// the sequence rather than multiplied by it.
fn plain(shape: (usize, usize), name: &str, vb: &VarBuilder) -> Result<Tensor> {
    vb.get(shape, name)?
        .dequantize(vb.device())?
        .to_dtype(DTYPE)
}

#[derive(Debug)]
struct TimestepEmbedder {
    l1: Linear,
    l2: Linear,
}

impl TimestepEmbedder {
    fn new(mid: usize, out: usize, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: zi_linear(
                super::model::FREQ_EMBED_SIZE,
                mid,
                true,
                mode,
                DTYPE,
                vb.pp("mlp.0"),
            )?,
            l2: zi_linear(mid, out, true, mode, DTYPE, vb.pp("mlp.2"))?,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // The frequency table is built in f32 — a cos/sin of `t · 10⁴` wants the
        // mantissa — and narrowed once, on 256 values, before it meets the model.
        let emb = super::model::timestep_frequencies(t)?.to_dtype(DTYPE)?;
        self.l2.forward(&silu(&self.l1.forward(&emb)?)?)
    }
}

#[derive(Debug)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    /// The widest reduction axis in the three projections, which is what sizes
    /// the token band.
    hidden: usize,
}

impl FeedForward {
    fn new(dim: usize, hidden: usize, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: zi_linear(dim, hidden, false, mode, DTYPE, vb.pp("w1"))?,
            w2: zi_linear(hidden, dim, false, mode, DTYPE, vb.pp("w2"))?,
            w3: zi_linear(dim, hidden, false, mode, DTYPE, vb.pp("w3"))?,
            hidden,
        })
    }
}

impl FeedForward {
    /// SwiGLU over a `[tokens, dim]` slab.
    fn dense(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.w1.forward(xs)?)?;
        self.w2.forward(&(gate * self.w3.forward(xs)?)?)
    }
}

impl Module for FeedForward {
    /// **Run a band of tokens at a time, because the matmul streams its
    /// activation once per output tile.**
    ///
    /// The int8 dense kernel tiles 32 tokens × 32 output rows and launches
    /// `grid(token_tiles, row_tiles)` token-tile-fastest, so a column's weight
    /// slice stays in L2 while the *activation* is re-read for every one of the
    /// `N/32` columns — 320 passes for `w1`. That is free while the activation
    /// fits in L2 and ruinous when it does not: measured on the 3090, `qkv` runs
    /// at 59.7 TOP/s while the activation is L2-resident and 46.9 at 4,128
    /// tokens, where it is 15.8 MiB against a 6 MiB cache.
    ///
    /// Splitting the tokens puts it back: the same weights, the same kernel, the
    /// same arithmetic — 40.4 TOP/s whole against 58.3 in bands, which is the
    /// resident ceiling recovered.
    ///
    /// **Exact, not an approximation.** A feed-forward is per token: a row's
    /// output depends on that row alone, so a band is those rows of the whole
    /// result. (The attention either side of it is not, which is why this stops
    /// here.) It also keeps the SwiGLU intermediate — 84.5 MiB at 1024×1024 —
    /// from ever existing in full.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3()?;
        // Flattened first: `narrow` on the token axis of a `[b, seq, dim]`
        // tensor is not contiguous, and the copy would cost more than the band
        // saves. On the 2-D view it is a plain offset.
        let flat = xs.reshape((b * seq, dim))?;
        let rows = b * seq;
        let band = token_band(self.hidden);
        if rows <= band {
            return self.dense(&flat)?.reshape((b, seq, dim));
        }
        let mut parts = Vec::with_capacity(rows.div_ceil(band));
        for start in (0..rows).step_by(band) {
            parts.push(self.dense(&flat.narrow(0, start, band.min(rows - start))?)?);
        }
        Tensor::cat(&parts, 0)?.reshape((b, seq, dim))
    }
}

/// Tokens per band, for a projection whose widest reduction axis is `k`.
///
/// Sized so the band's quantized activation — one byte per element — sits well
/// inside the smallest L2 in the fleet, leaving the rest for the weight column
/// streaming past it. Rounded to the kernel's own 32-token tile so no band ends
/// on a partial one.
///
/// Measured at 1024×1024 (`k = 10240`, so a 192-token band): the sweep over
/// 4128/2048/1024/512/256 gives 40.4/49.2/54.9/58.3/57.1 TOP/s, so anything from
/// a few hundred down is at the plateau and the exact figure is not delicate.
fn token_band(k: usize) -> usize {
    /// Half of the 3090's 6 MiB — the rest is the weight column in flight.
    const BUDGET: usize = 2 << 20;
    (BUDGET / k.max(1)).max(32) / 32 * 32
}

/// One projection, run a token band at a time — [`FeedForward::forward`]'s
/// reasoning applied to a single `Linear`.
///
/// Only worth it where `M` is the sequence *and* `N` is wide enough that the
/// activation is re-read many times. `adaLN` and the timestep MLP have one row;
/// the caption projection has thirty-two; the output head is 64 wide, so its
/// activation is read twice rather than hundreds of times. Those take the
/// early-out and pay nothing.
fn banded_proj(lin: &Linear, xs: &Tensor, k: usize) -> Result<Tensor> {
    let (b, seq, dim) = xs.dims3()?;
    let rows = b * seq;
    let band = token_band(k);
    let flat = xs.reshape((rows, dim))?;
    let reshape = |o: Tensor| -> Result<Tensor> {
        let n = o.dim(1)?;
        o.reshape((b, seq, n))
    };
    if rows <= band {
        return reshape(lin.forward(&flat)?);
    }
    let mut parts = Vec::with_capacity(rows.div_ceil(band));
    for start in (0..rows).step_by(band) {
        parts.push(lin.forward(&flat.narrow(0, start, band.min(rows - start))?)?);
    }
    reshape(Tensor::cat(&parts, 0)?)
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    head_dim: usize,
    /// The card's numeric mode, carried so the attention runs in the same one
    /// the projections do.
    mode: Int8Mode,
}

impl Attention {
    fn new(cfg: &Config, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        let (dim, head_dim) = (cfg.dim, cfg.head_dim());
        Ok(Self {
            // One projection of three times the width. `n_kv_heads == n_heads`
            // for this model, so the three parts are equal and the split below
            // is a plain thirds.
            qkv: zi_linear(dim, 3 * dim, false, mode, DTYPE, vb.pp("qkv"))?,
            out: zi_linear(dim, dim, false, mode, DTYPE, vb.pp("out"))?,
            q_norm: rms(head_dim, 1e-5, vb.pp("q_norm"))?,
            k_norm: rms(head_dim, 1e-5, vb.pp("k_norm"))?,
            n_heads: cfg.n_heads,
            head_dim,
            mode,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3()?;
        let qkv = banded_proj(&self.qkv, xs, dim)?;
        let split = |off: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, off * dim, dim)?
                .reshape((b, seq, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = self.q_norm.forward(&split(0)?)?;
        let k = self.k_norm.forward(&split(1)?)?;
        let v = split(2)?;

        let q = candle_nn::rotary_emb::rope_i(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k.contiguous()?, cos, sin)?;

        // The scale on `q` and the banding are both [`super::attention`]'s, so
        // the reference the module tests against is this code rather than a
        // second copy of it that could drift.
        let out = attend(&q, &k, &v, self.mode)?.transpose(1, 2)?.reshape((
            b,
            seq,
            self.n_heads * self.head_dim,
        ))?;
        banded_proj(&self.out, &out, self.n_heads * self.head_dim)
    }
}

#[derive(Debug)]
struct Block {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    ada_ln: Option<Linear>,
    dim: usize,
}

impl Block {
    fn new(cfg: &Config, modulation: bool, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden = (dim / 3) * 8;
        let ada_ln = if modulation {
            Some(zi_linear(
                ADALN_EMBED_DIM,
                4 * dim,
                true,
                mode,
                DTYPE,
                vb.pp("adaLN_modulation.0"),
            )?)
        } else {
            None
        };
        Ok(Self {
            attention: Attention::new(cfg, mode, vb.pp("attention"))?,
            feed_forward: FeedForward::new(dim, hidden, mode, vb.pp("feed_forward"))?,
            attention_norm1: rms(dim, cfg.norm_eps, vb.pp("attention_norm1"))?,
            attention_norm2: rms(dim, cfg.norm_eps, vb.pp("attention_norm2"))?,
            ffn_norm1: rms(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?,
            ffn_norm2: rms(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?,
            ada_ln,
            dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: Option<&Tensor>,
    ) -> Result<Tensor> {
        let Some((ada_ln, adaln)) = self.ada_ln.as_ref().zip(adaln) else {
            let attn = self
                .attention
                .forward(&self.attention_norm1.forward(xs)?, cos, sin)?;
            let xs = (xs + self.attention_norm2.forward(&attn)?)?;
            let ff = self.feed_forward.forward(&self.ffn_norm1.forward(&xs)?)?;
            return &xs + self.ffn_norm2.forward(&ff)?;
        };

        let m = ada_ln.forward(adaln)?;
        let d = self.dim;
        let scale_msa = (m.narrow(D::Minus1, 0, d)? + 1.0)?.unsqueeze(1)?;
        let gate_msa = m.narrow(D::Minus1, d, d)?.tanh()?.unsqueeze(1)?;
        let scale_mlp = (m.narrow(D::Minus1, 2 * d, d)? + 1.0)?.unsqueeze(1)?;
        let gate_mlp = m.narrow(D::Minus1, 3 * d, d)?.tanh()?.unsqueeze(1)?;

        let normed = self
            .attention_norm1
            .forward(xs)?
            .broadcast_mul(&scale_msa)?;
        let attn = self.attention.forward(&normed, cos, sin)?;
        let xs = (xs
            + self
                .attention_norm2
                .forward(&attn)?
                .broadcast_mul(&gate_msa)?)?;

        let normed = self.ffn_norm1.forward(&xs)?.broadcast_mul(&scale_mlp)?;
        let ff = self.feed_forward.forward(&normed)?;
        &xs + self.ffn_norm2.forward(&ff)?.broadcast_mul(&gate_mlp)?
    }
}

#[derive(Debug)]
struct FinalLayer {
    linear: Linear,
    ada_ln: Linear,
}

impl FinalLayer {
    fn new(cfg: &Config, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: zi_linear(cfg.dim, cfg.patch_dim(), true, mode, DTYPE, vb.pp("linear"))?,
            ada_ln: zi_linear(
                ADALN_EMBED_DIM,
                cfg.dim,
                true,
                mode,
                DTYPE,
                vb.pp("adaLN_modulation.1"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, adaln: &Tensor) -> Result<Tensor> {
        let scale = (self.ada_ln.forward(&silu(adaln)?)? + 1.0)?.unsqueeze(1)?;
        let xs = layer_norm_affineless(xs, 1e-6)?.broadcast_mul(&scale)?;
        self.linear.forward(&xs)
    }
}

pub struct ZImageTransformer {
    x_embedder: Linear,
    cap_norm: RmsNorm,
    cap_linear: Linear,
    t_embedder: TimestepEmbedder,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    layers: Vec<Block>,
    final_layer: FinalLayer,
    x_pad_token: Tensor,
    cap_pad_token: Tensor,
    geom: Geometry,
    cfg: Config,
}

impl ZImageTransformer {
    /// Build from a GGUF, with every projection repacked for `mode`.
    ///
    /// `mode` is the card's, from `Int8Mode::auto` — `Precision` on Ampere and
    /// up, `Off` on anything without the int8 MMA, where this is the standard
    /// dequantising path unchanged.
    pub fn new(cfg: Config, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();

        let x_embedder = zi_linear(
            cfg.patch_dim(),
            cfg.dim,
            true,
            mode,
            DTYPE,
            vb.pp("x_embedder"),
        )?;
        let final_layer = FinalLayer::new(&cfg, mode, vb.pp("final_layer"))?;

        let vb_cap = vb.pp("cap_embedder");
        let cap_norm = rms(cfg.cap_feat_dim, cfg.norm_eps, vb_cap.pp("0"))?;
        let cap_linear = zi_linear(cfg.cap_feat_dim, cfg.dim, true, mode, DTYPE, vb_cap.pp("1"))?;

        let t_embedder = TimestepEmbedder::new(1024, ADALN_EMBED_DIM, mode, vb.pp("t_embedder"))?;

        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let vb_n = vb.pp("noise_refiner");
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(Block::new(&cfg, true, mode, vb_n.pp(i.to_string()))?);
        }
        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let vb_c = vb.pp("context_refiner");
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(Block::new(&cfg, false, mode, vb_c.pp(i.to_string()))?);
        }
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.n_layers {
            layers.push(Block::new(&cfg, true, mode, vb_l.pp(i.to_string()))?);
        }

        let x_pad_token = plain((1, cfg.dim), "x_pad_token", &vb)?;
        let cap_pad_token = plain((1, cfg.dim), "cap_pad_token", &vb)?;
        // The RoPE tables at the activation width, so applying them is a rotate
        // rather than a rotate and two conversions.
        let geom = Geometry::new(&cfg, &dev, DTYPE)?;

        Ok(Self {
            x_embedder,
            cap_norm,
            cap_linear,
            t_embedder,
            noise_refiner,
            context_refiner,
            layers,
            final_layer,
            x_pad_token,
            cap_pad_token,
            geom,
            cfg,
        })
    }

    /// One denoising step. Same contract as [`super::model::ZImageTransformer::forward`].
    ///
    /// The latent crosses in and out at its own width and everything between runs
    /// at [`DTYPE`] — the two casts here and the one on the way out are the whole
    /// boundary, on the narrowest tensors in the pass.
    pub fn forward(&self, latent: &Tensor, cap: &Tensor, t: f64) -> Result<Tensor> {
        let dev = latent.device();
        let g = &self.geom;

        let t_in = Tensor::from_vec(vec![(t * self.cfg.t_scale) as f32], 1, dev)?;
        let adaln = self.t_embedder.forward(&t_in)?;

        let cap_ids = g.cap_ids(cap.dim(0)?);
        let cap_feats = self
            .cap_linear
            .forward(&self.cap_norm.forward(&cap.to_dtype(DTYPE)?)?)?;
        let cap_run = g.pad_run(&cap_feats, cap_ids, &self.cap_pad_token)?;
        let cap_padded = cap_run.feats.dim(0)?;

        let (patches, ht, wt) = g.patchify(latent)?;
        let x_feats = self.x_embedder.forward(&patches.to_dtype(DTYPE)?)?;
        let x_ids = g.img_ids(cap_padded, ht, wt);
        let x_run = g.pad_run(&x_feats, x_ids, &self.x_pad_token)?;

        // **A block at a time, when there is an arena to run them on.**
        //
        // [`candle_nn::kv_cache::guest_stage`] is the identity for every caller
        // that is not a co-resident guest, and it costs one copy of the hidden
        // state either side of a block for one that is. What that buys is the
        // arena holding a *block's* intermediates instead of the whole forward's
        // — measured, the sum of a 1024×1024 forward saturated seven gigabytes,
        // and once one carve fails nothing after it can inherit and the rest of
        // the draw falls back to the pool it was evicted from.
        let (x_cos, x_sin) = g.rope.gather(&x_run.ids, dev)?;
        let mut x = x_run.feats.unsqueeze(0)?;
        for blk in self.noise_refiner.iter() {
            x = candle_nn::kv_cache::guest_stage(&x, |x| {
                blk.forward(x, &x_cos, &x_sin, Some(&adaln))
            })?;
        }

        let (c_cos, c_sin) = g.rope.gather(&cap_run.ids, dev)?;
        let mut c = cap_run.feats.unsqueeze(0)?;
        for blk in self.context_refiner.iter() {
            c = candle_nn::kv_cache::guest_stage(&c, |c| blk.forward(c, &c_cos, &c_sin, None))?;
        }

        let mut h = Tensor::cat(&[&x, &c], 1)?;
        let cos = Tensor::cat(&[&x_cos, &c_cos], 0)?.contiguous()?;
        let sin = Tensor::cat(&[&x_sin, &c_sin], 0)?.contiguous()?;
        for blk in self.layers.iter() {
            h = candle_nn::kv_cache::guest_stage(&h, |h| blk.forward(h, &cos, &sin, Some(&adaln)))?;
        }

        let h = self.final_layer.forward(&h, &adaln)?;
        let h = h.i((0, ..ht * wt))?;
        g.unpatchify(&h, ht, wt)?.to_dtype(latent.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::ko_quant::ko_tileable;

    /// Every projection in the model, as the GGUF stores it: `[out, in]`.
    fn projections(cfg: &Config) -> Vec<(&'static str, usize, usize)> {
        let (d, h) = (cfg.dim, (cfg.dim / 3) * 8);
        vec![
            ("x_embedder", d, cfg.patch_dim()),
            ("cap_embedder.1", d, cfg.cap_feat_dim),
            (
                "t_embedder.mlp.0",
                1024,
                super::super::model::FREQ_EMBED_SIZE,
            ),
            ("t_embedder.mlp.2", ADALN_EMBED_DIM, 1024),
            ("attention.qkv", 3 * d, d),
            ("attention.out", d, d),
            ("feed_forward.w1", h, d),
            ("feed_forward.w2", d, h),
            ("feed_forward.w3", h, d),
            ("adaLN_modulation.0", 4 * d, ADALN_EMBED_DIM),
            ("final_layer.linear", cfg.patch_dim(), d),
            ("final_layer.adaLN_modulation.1", d, ADALN_EMBED_DIM),
        ]
    }

    /// **`x_embedder` is the only weight that stays off the int8 path, and the
    /// module header says so.** The KO matmul tiles rows by 32 and columns by
    /// 128; a 64-wide patch projection misses the second by half. Everything
    /// else tiles, and if a future geometry breaks one of them the header is
    /// wrong rather than merely out of date — `QMatMul` would drop that weight
    /// to the dequantising path silently, and only a throughput number would
    /// ever show it.
    #[test]
    fn one_projection_and_only_one_misses_the_int8_tiling() {
        let cfg = Config::turbo();
        let missed: Vec<_> = projections(&cfg)
            .into_iter()
            .filter(|(_, out, inp)| !ko_tileable(*out, *inp))
            .map(|(name, out, inp)| format!("{name} [{out}, {inp}]"))
            .collect();
        assert_eq!(missed, vec!["x_embedder [3840, 64]".to_string()]);
    }

    /// **What the per-call activation quantize costs, and what the projections
    /// achieve without it.**
    ///
    /// Every `Linear::forward` here goes `Module::forward` → `forward_live` →
    /// `to_dynamic` → `quantize_acts_q8a128`: a full read of the bf16 activation
    /// and a write of its int8 twin, *per matmul*. Five matmuls a block,
    /// thirty-two blocks a step. And `w1` and `w3` read the same activation, so
    /// that one is quantized twice for no reason at all.
    ///
    /// The engine's language models do not work this way. `quantized_qwen3`
    /// runs a *fused producer* — `ln1.forward_dynamic` emits q8a128 straight out
    /// of the norm — and hands one `DynamicActs` to three consumers; its MLP
    /// fuses gate and up into one weight so the post-norm activation is
    /// quantized once for both. All of that machinery is in-tree
    /// (`DynamicActs`, `QMatMul::forward_dynamic`, `rms_norm_q8a128`,
    /// `silu_mul_q8a128`) and this model uses none of it.
    ///
    /// So: measure one matmul both ways at the shapes this model runs. `forward`
    /// is what it does today; `forward_dynamic` over a once-quantized operand is
    /// what a fused producer would hand it. The gap is the whole question.
    ///
    /// Asserts nothing — a benchmark that fails on a slower card is a test
    /// nobody runs.
    #[test]
    #[ignore = "GPU benchmark: the z-image projection baseline; run with --ignored --nocapture"]
    fn projection_baseline_at_z_image_size() -> Result<()> {
        use candle::quantized::cuda::to_dynamic;
        use candle::quantized::{GgmlDType, QTensor};
        use candle::Device;

        let dev = Device::cuda_if_available(0)?;
        let Device::Cuda(cuda) = &dev else {
            println!("no CUDA device; nothing to measure");
            return Ok(());
        };
        const WARMUP: usize = 3;
        const ITERS: usize = 20;
        // 1024×1024: 4,096 image patches plus a 32-token padded caption.
        const SEQ: usize = 4128;
        let mode = Int8Mode::Precision;
        let cfg = Config::turbo();
        let (d, h) = (cfg.dim, (cfg.dim / 3) * 8);

        // **Throughput against token count, which is where the projections'
        // problem actually shows.**
        //
        // Measured on the 3090: 68.5 TOP/s at M=256, 62.2 at 1024, 46.9 at 4128,
        // 38.5 at 16384. It *falls* — and a kernel that is merely
        // under-amortised gets better with more work, not worse. What gets worse
        // with M is the activation's residency.
        //
        // The int8 dense kernel tiles `Bm = 32` tokens by `Bn = 32` rows
        // (`dispatcher.cu`: `row_tiles = (nrows_x + 31) / 32`, `batch_div = 32`),
        // and launches `grid(batch_tiles, row_tiles)` — so all of a column's
        // token tiles run before the next column, the weight slice stays in L2,
        // and the **activation is streamed once per N tile**. For `qkv` that is
        // 360 passes over `[M, 3840]`. At M=256 the activation is 1 MiB and L2
        // serves every pass; at M=4128 it is 15.8 MiB against a 6 MiB L2, so the
        // same 360 passes become 5.7 GiB of DRAM traffic — 6.1 ms at this card's
        // 936 GB/s, against 7.8 measured.
        //
        // That is a decode-shaped kernel meeting a diffusion-shaped problem: an
        // LLM step has a handful of tokens, where these tiles are right and the
        // activation is always resident. The fix is wider blocking, and half of
        // it already exists unused — `kernel.cuh`'s `N_SUB` template goes to 8
        // (`Bm = 128`, with its own single-buffered smem arm) and only 1 and 2
        // are ever instantiated.
        {
            let (out_dim, in_dim) = (3 * d, d);
            let w = Tensor::randn(0f32, 0.02f32, (out_dim, in_dim), &dev)?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            let mm = crate::models::quantized_matmul::QMatMul::from_qtensor_with_mode(qt, mode)?;
            print!("qkv TOP/s by token count:");
            for m in [256usize, 1024, 4128, 16384] {
                let xs = Tensor::randn(0f32, 1f32, (1, m, in_dim), &dev)?.to_dtype(DTYPE)?;
                for _ in 0..WARMUP {
                    mm.forward(&xs)?;
                }
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    mm.forward(&xs)?;
                }
                dev.synchronize()?;
                let per = t0.elapsed().as_secs_f64() / ITERS as f64;
                let flops = 2.0 * m as f64 * out_dim as f64 * in_dim as f64;
                print!("  M={m}: {:.1}", flops / per / 1e12);
            }
            println!();

            // **The same total work, split so the activation fits in L2.**
            //
            // If the 360 N-tile passes are what costs, then slicing M until
            // `chunk × K` sits inside the 6 MiB L2 should make every pass after
            // the first a cache hit — and cost nothing but a few extra launches.
            // Same arithmetic, same kernel, same weights; only the residency
            // changes. A flat result would mean the streaming theory is wrong.
            // 2-D, so `narrow` on the token axis is a contiguous view and the
            // chunk loop pays no copy — on a `[1, M, K]` tensor it would not be,
            // and the copy would be charged to chunking rather than to the
            // benchmark that asked for it.
            let xs = Tensor::randn(0f32, 1f32, (SEQ, in_dim), &dev)?.to_dtype(DTYPE)?;
            print!("qkv TOP/s by M-chunk (whole = {SEQ}):");
            for chunk in [SEQ, 2048, 1024, 512, 256] {
                let run = || -> Result<()> {
                    for s in (0..SEQ).step_by(chunk) {
                        let rows = chunk.min(SEQ - s);
                        mm.forward(&xs.narrow(0, s, rows)?)?;
                    }
                    Ok(())
                };
                for _ in 0..WARMUP {
                    run()?;
                }
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    run()?;
                }
                dev.synchronize()?;
                let per = t0.elapsed().as_secs_f64() / ITERS as f64;
                let flops = 2.0 * SEQ as f64 * out_dim as f64 * in_dim as f64;
                print!("  {chunk}: {:.1}", flops / per / 1e12);
            }
            println!();
        }

        println!(
            "{:<20} {:>13} {:>10} {:>10} {:>9} {:>10}",
            "projection", "shape", "fwd ms", "dyn ms", "quant ms", "dyn TOP/s"
        );
        // The four full-width projections, which are all but a rounding error of
        // the block's arithmetic. `w1` and `w3` are the same shape as each other.
        for (name, out_dim, in_dim) in [
            ("attention.qkv", 3 * d, d),
            ("attention.out", d, d),
            ("feed_forward.w1", h, d),
            ("feed_forward.w2", d, h),
        ] {
            // Random weights through the same route the loader takes: quantize
            // to the file's format, then let `QMatMul` build the KO twin.
            let w = Tensor::randn(0f32, 0.02f32, (out_dim, in_dim), &dev)?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            let mm = crate::models::quantized_matmul::QMatMul::from_qtensor_with_mode(qt, mode)?;
            let xs = Tensor::randn(0f32, 1f32, (1, SEQ, in_dim), &dev)?.to_dtype(DTYPE)?;

            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                for _ in 0..WARMUP {
                    f()?;
                }
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    f()?;
                }
                dev.synchronize()?;
                Ok(t0.elapsed().as_secs_f64() / ITERS as f64)
            };

            // What the model does now: quantize, then matmul.
            let fwd = time(&|| {
                mm.forward(&xs)?;
                Ok(())
            })?;
            // What a fused producer would hand it: the operand already int8.
            // The convention production uses here — see `zi_linear`. The
            // benchmark must quantize the way the model does or it measures a
            // path nothing runs.
            let acts = to_dynamic(&xs, mode, cuda, SumScale::ByAmax)?;
            let dynf = time(&|| {
                mm.forward_dynamic(acts.as_dynamic(), DTYPE)?;
                Ok(())
            })?;

            let flops = 2.0 * SEQ as f64 * out_dim as f64 * in_dim as f64;
            println!(
                "{:<20} {:>13} {:>10.3} {:>10.3} {:>9.3} {:>10.1}",
                name,
                format!("{out_dim}×{in_dim}"),
                fwd * 1e3,
                dynf * 1e3,
                (fwd - dynf) * 1e3,
                flops / dynf / 1e12,
            );
        }
        Ok(())
    }
}
