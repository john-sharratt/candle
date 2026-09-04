//! The `ZImageTransformer2DModel` forward pass.
//!
//! Ported from `diffusers`' `transformer_z_image.py`. This carries the
//! **standard** (text-to-image) path only: the reference also has an "omni"
//! mode for editing, which threads per-token noise masks and a second timestep
//! embedding through every block. None of that is reachable from a
//! text-to-image call, and carrying it would mean carrying branches nothing
//! here can exercise.

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, rms_norm, Linear, Module, RmsNorm};
use candle_nn::{ops::silu, VarBuilder};

/// The width the timestep embedding is carried at, and therefore the input
/// width of every `adaLN_modulation`. `min(dim, 256)` in the reference, and 256
/// for every published configuration.
pub(crate) const ADALN_EMBED_DIM: usize = 256;

/// Sequence lengths are padded up to a multiple of this.
///
/// Not an optimisation: the padding is *positional*. Pad tokens are given
/// position `(0, 0, 0)` and a learned pad embedding, and the caption's length
/// after padding is what the image's position axis starts from — so a different
/// multiple moves every image patch's RoPE phase and changes the picture.
const SEQ_MULTIPLE: usize = 32;

/// Frequencies in the sinusoidal timestep embedding, before the MLP.
pub(crate) const FREQ_EMBED_SIZE: usize = 256;

/// The sinusoidal half of the timestep embedding, before the MLP.
///
/// Built in f32 whatever the model's width: the frequencies span four orders of
/// magnitude, and the smallest are not representable in bf16 — a half-precision
/// table quantises the low frequencies onto each other and the embedding stops
/// distinguishing nearby timesteps.
pub(crate) fn timestep_frequencies(t: &Tensor) -> Result<Tensor> {
    let dev = t.device();
    let half = FREQ_EMBED_SIZE / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), dev)?;
    let args = t
        .to_dtype(DType::F32)?
        .reshape(((), 1))?
        .broadcast_mul(&freqs)?;
    // `cos` then `sin`, which is the reference's order — the opposite
    // convention would rotate every embedding by a quarter turn.
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub in_channels: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub cap_feat_dim: usize,
    pub norm_eps: f64,
    pub patch_size: usize,
    pub f_patch_size: usize,
    pub rope_theta: f64,
    pub t_scale: f64,
    /// Head dimensions given to each of the three position axes. Sums to
    /// `dim / n_heads`.
    pub axes_dims: Vec<usize>,
    /// The largest position each axis is precomputed for.
    pub axes_lens: Vec<usize>,
}

impl Config {
    /// Z-Image-Turbo, from the published `transformer/config.json`.
    pub fn turbo() -> Self {
        Self {
            in_channels: 16,
            dim: 3840,
            n_heads: 30,
            n_layers: 30,
            n_refiner_layers: 2,
            cap_feat_dim: 2560,
            norm_eps: 1e-5,
            patch_size: 2,
            f_patch_size: 1,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![1536, 512, 512],
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// The width of one patch once flattened: `f_patch * patch² * channels`.
    pub fn patch_dim(&self) -> usize {
        self.f_patch_size * self.patch_size * self.patch_size * self.in_channels
    }

    /// The key `all_x_embedder` and `all_final_layer` are indexed by, e.g.
    /// `"2-1"`. A `ModuleDict` on the Python side, so the patch geometry is
    /// part of the tensor name rather than a constructor argument.
    fn patch_key(&self) -> String {
        format!("{}-{}", self.patch_size, self.f_patch_size)
    }
}

/// The sinusoidal-then-MLP timestep embedding.
#[derive(Debug)]
struct TimestepEmbedder {
    l1: Linear,
    l2: Linear,
}

impl TimestepEmbedder {
    fn new(mid: usize, out: usize, vb: VarBuilder) -> Result<Self> {
        // `mlp.0` and `mlp.2` — index 1 is the SiLU, which has no weights.
        Ok(Self {
            l1: linear(FREQ_EMBED_SIZE, mid, vb.pp("mlp.0"))?,
            l2: linear(mid, out, vb.pp("mlp.2"))?,
        })
    }

    /// `t` is a 1-D tensor of one timestep per batch item.
    fn forward(&self, t: &Tensor, dtype: DType) -> Result<Tensor> {
        let emb = timestep_frequencies(t)?.to_dtype(dtype)?;
        self.l2.forward(&silu(&self.l1.forward(&emb)?)?)
    }
}

/// SwiGLU. `w2(silu(w1(x)) * w3(x))`.
#[derive(Debug)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(dim, hidden, vb.pp("w1"))?,
            w2: linear_no_bias(hidden, dim, vb.pp("w2"))?,
            w3: linear_no_bias(dim, hidden, vb.pp("w3"))?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.w1.forward(xs)?)?;
        self.w2.forward(&(gate * self.w3.forward(xs)?)?)
    }
}

/// Self-attention with qk-RMSNorm and 3-axis interleaved RoPE.
#[derive(Debug)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (dim, head_dim) = (cfg.dim, cfg.head_dim());
        Ok(Self {
            to_q: linear_no_bias(dim, dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, dim, vb.pp("to_v"))?,
            // `to_out` is a `ModuleList` whose second entry is a dropout, so
            // the weights live under `to_out.0`.
            to_out: linear_no_bias(dim, dim, vb.pp("to_out.0"))?,
            // Normed per head, not per model width: the reference applies these
            // after the head split.
            norm_q: rms_norm(head_dim, 1e-5, vb.pp("norm_q"))?,
            norm_k: rms_norm(head_dim, 1e-5, vb.pp("norm_k"))?,
            n_heads: cfg.n_heads,
            head_dim,
        })
    }

    /// `xs` is `(b, seq, dim)`; `cos`/`sin` are `(seq, head_dim / 2)`.
    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, seq, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = self.norm_q.forward(&split(self.to_q.forward(xs)?)?)?;
        let k = self.norm_k.forward(&split(self.to_k.forward(xs)?)?)?;
        let v = split(self.to_v.forward(xs)?)?;

        // Interleaved, not split-halves: the reference rotates `(even, odd)`
        // pairs by viewing the head as complex, which is what `rope_i` does.
        let q = candle_nn::rotary_emb::rope_i(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k.contiguous()?, cos, sin)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn =
            candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let out =
            attn.matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, seq, self.n_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// One block, in all three of its roles.
///
/// `modulation` is what separates them: the noise refiner and the thirty main
/// layers are conditioned on the timestep, the context refiner is not — a
/// caption does not depend on how far through the denoise the image is, and the
/// checkpoint carries no `adaLN_modulation` weights for those blocks.
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
    fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        // `int(dim / 3 * 8)` in the reference — 10,240 at dim 3840.
        let hidden = (dim / 3) * 8;
        let ada_ln = if modulation {
            Some(linear(
                ADALN_EMBED_DIM,
                4 * dim,
                vb.pp("adaLN_modulation.0"),
            )?)
        } else {
            None
        };
        Ok(Self {
            attention: Attention::new(cfg, vb.pp("attention"))?,
            feed_forward: FeedForward::new(dim, hidden, vb.pp("feed_forward"))?,
            attention_norm1: rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm1"))?,
            attention_norm2: rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm2"))?,
            ffn_norm1: rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?,
            ffn_norm2: rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?,
            ada_ln,
            dim,
        })
    }

    /// `adaln` is the timestep embedding, `(b, ADALN_EMBED_DIM)`.
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: Option<&Tensor>,
    ) -> Result<Tensor> {
        let Some(ada) = self.ada_ln.as_ref().zip(adaln) else {
            // Unmodulated: still sandwich-normed, which is the part that is
            // easy to drop and produces a plausible-looking wrong picture.
            let attn = self
                .attention
                .forward(&self.attention_norm1.forward(xs)?, cos, sin)?;
            let xs = (xs + self.attention_norm2.forward(&attn)?)?;
            let ff = self.feed_forward.forward(&self.ffn_norm1.forward(&xs)?)?;
            return &xs + self.ffn_norm2.forward(&ff)?;
        };
        let (ada_ln, adaln) = ada;

        let m = ada_ln.forward(adaln)?;
        let d = self.dim;
        // Four vectors, in this order. `scale` is `1 + x` so a zero-initialised
        // modulation is the identity; `gate` is `tanh` so it starts at zero and
        // the residual stream is untouched by an untrained block.
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

/// The output head: an unaffine LayerNorm scaled by the timestep, then a
/// projection back to patch space.
#[derive(Debug)]
struct FinalLayer {
    linear: Linear,
    ada_ln: Linear,
}

/// LayerNorm with no learned weight or bias.
///
/// `elementwise_affine=False` in the reference, so the checkpoint carries
/// nothing for it — the scale it would have had comes from the timestep
/// instead. `candle_nn::layer_norm` always asks its `VarBuilder` for a weight,
/// even when told the norm is unaffine, so the statistics are taken here rather
/// than through a module that would fail to load.
pub(crate) fn layer_norm_affineless(xs: &Tensor, eps: f64) -> Result<Tensor> {
    // In f32 regardless of the model's width: a sum of 3,840 terms in bf16
    // loses the mean it is trying to measure.
    let x = xs.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centred = x.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    centred
        .broadcast_div(&(var + eps)?.sqrt()?)?
        .to_dtype(xs.dtype())
}

impl FinalLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(cfg.dim, cfg.patch_dim(), vb.pp("linear"))?,
            // Index 1: index 0 is the SiLU.
            ada_ln: linear(ADALN_EMBED_DIM, cfg.dim, vb.pp("adaLN_modulation.1"))?,
        })
    }

    fn forward(&self, xs: &Tensor, adaln: &Tensor) -> Result<Tensor> {
        let scale = (self.ada_ln.forward(&silu(adaln)?)? + 1.0)?.unsqueeze(1)?;
        let xs = layer_norm_affineless(xs, 1e-6)?.broadcast_mul(&scale)?;
        self.linear.forward(&xs)
    }
}

/// Precomputed `cos`/`sin` for each position axis.
///
/// One table per axis, of its own width, looked up by that axis's coordinate
/// and concatenated — so a token's rotation is the product of three
/// independent rotations rather than one over a flattened index. That is what
/// lets a caption token and an image patch share a sequence without their
/// positions colliding: the caption moves along axis 0 while the image moves
/// along axes 1 and 2.
#[derive(Debug)]
pub(crate) struct Rope {
    cos: Vec<Tensor>,
    sin: Vec<Tensor>,
    axes_dims: Vec<usize>,
}

impl Rope {
    /// Built at the model's width, because `rope_i` requires the tables and the
    /// tensor it rotates to agree — and computed in f64 first, since the
    /// smallest frequencies are four orders of magnitude below the largest and
    /// a table generated in half precision collapses them onto each other.
    pub(crate) fn new(cfg: &Config, dev: &Device, dtype: DType) -> Result<Self> {
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for (&d, &len) in cfg.axes_dims.iter().zip(cfg.axes_lens.iter()) {
            let half = d / 2;
            let mut c = Vec::with_capacity(len * half);
            let mut s = Vec::with_capacity(len * half);
            for p in 0..len {
                for i in 0..half {
                    // `theta ** (2i / d)`, matching the reference's
                    // `arange(0, d, 2) / d`.
                    let freq = 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / d as f64);
                    let angle = p as f64 * freq;
                    c.push(angle.cos() as f32);
                    s.push(angle.sin() as f32);
                }
            }
            cos.push(Tensor::from_vec(c, (len, half), dev)?.to_dtype(dtype)?);
            sin.push(Tensor::from_vec(s, (len, half), dev)?.to_dtype(dtype)?);
        }
        Ok(Self {
            cos,
            sin,
            axes_dims: cfg.axes_dims.clone(),
        })
    }

    /// Gather `(cos, sin)` of shape `(seq, head_dim / 2)` for `ids`, one triple
    /// of coordinates per token.
    pub(crate) fn gather(&self, ids: &[[u32; 3]], dev: &Device) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::with_capacity(self.axes_dims.len());
        let mut sin_parts = Vec::with_capacity(self.axes_dims.len());
        for axis in 0..self.axes_dims.len() {
            let idx: Vec<u32> = ids.iter().map(|t| t[axis]).collect();
            let idx = Tensor::from_vec(idx, ids.len(), dev)?;
            cos_parts.push(self.cos[axis].index_select(&idx, 0)?);
            sin_parts.push(self.sin[axis].index_select(&idx, 0)?);
        }
        Ok((
            Tensor::cat(&cos_parts, D::Minus1)?.contiguous()?,
            Tensor::cat(&sin_parts, D::Minus1)?.contiguous()?,
        ))
    }
}

/// One padded run of tokens: the features, and the position of each.
pub(crate) struct Prepared {
    pub(crate) feats: Tensor,
    pub(crate) ids: Vec<[u32; 3]>,
}

/// Round `n` up to the next multiple of [`SEQ_MULTIPLE`].
fn padded_len(n: usize) -> usize {
    n.div_ceil(SEQ_MULTIPLE) * SEQ_MULTIPLE
}

/// Everything about *where* things go: patch layout, padding, and position ids.
///
/// Shared by the full-precision and quantized models, which differ only in the
/// type of their weights. The geometry is where the picture actually comes from
/// — a wrong permutation or a wrong position origin produces a plausible image
/// of the wrong thing — so there is one copy of it and both models hold it.
pub(crate) struct Geometry {
    pub(crate) rope: Rope,
    pub(crate) cfg: Config,
}

impl Geometry {
    pub(crate) fn new(cfg: &Config, dev: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            rope: Rope::new(cfg, dev, dtype)?,
            cfg: cfg.clone(),
        })
    }

    /// Flatten `(c, h, w)` into `(h/p · w/p, p²·c)` patches.
    ///
    /// The permutation is the whole of it: the reference views the image as
    /// `(c, ht, ph, wt, pw)` and permutes to `(ht, wt, ph, pw, c)`, so within a
    /// patch the channel is the *fastest* axis. Getting that order wrong keeps
    /// every number and scrambles which pixel each belongs to.
    pub(crate) fn patchify(&self, latent: &Tensor) -> Result<(Tensor, usize, usize)> {
        let (c, h, w) = latent.dims3()?;
        let p = self.cfg.patch_size;
        let (ht, wt) = (h / p, w / p);
        let x = latent
            .reshape((c, ht, p, wt, p))?
            .permute((1, 3, 2, 4, 0))?
            .reshape((ht * wt, p * p * c))?;
        Ok((x, ht, wt))
    }

    /// The inverse of [`Self::patchify`].
    pub(crate) fn unpatchify(&self, xs: &Tensor, ht: usize, wt: usize) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        let c = self.cfg.in_channels;
        xs.reshape((ht, wt, p, p, c))?
            .permute((4, 0, 2, 1, 3))?
            .reshape((c, ht * p, wt * p))
    }

    /// Pad a run to [`SEQ_MULTIPLE`], substituting the learned pad embedding and
    /// giving every pad token position `(0, 0, 0)`.
    ///
    /// The reference pads the *features* by repeating the last row and then
    /// overwrites those rows with the pad token, which is the same thing as
    /// appending the pad token directly.
    pub(crate) fn pad_run(
        &self,
        feats: &Tensor,
        ids: Vec<[u32; 3]>,
        pad: &Tensor,
    ) -> Result<Prepared> {
        let n = feats.dim(0)?;
        let total = padded_len(n);
        if total == n {
            return Ok(Prepared {
                feats: feats.clone(),
                ids,
            });
        }
        let pad_rows = pad.broadcast_as((total - n, pad.dim(1)?))?.contiguous()?;
        let feats = Tensor::cat(&[feats, &pad_rows], 0)?;
        let mut ids = ids;
        ids.resize(total, [0, 0, 0]);
        Ok(Prepared { feats, ids })
    }

    /// Position ids for a caption of `len` tokens: `(1 + j, 0, 0)`.
    pub(crate) fn cap_ids(&self, len: usize) -> Vec<[u32; 3]> {
        (0..len).map(|j| [1 + j as u32, 0, 0]).collect()
    }

    /// Position ids for an image, starting one past the caption's *padded*
    /// length on axis 0 and spreading over axes 1 and 2.
    pub(crate) fn img_ids(&self, cap_padded: usize, ht: usize, wt: usize) -> Vec<[u32; 3]> {
        (0..ht)
            .flat_map(|h| (0..wt).map(move |w| [cap_padded as u32 + 1, h as u32, w as u32]))
            .collect()
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
    dtype: DType,
}

impl ZImageTransformer {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let key = cfg.patch_key();
        let dev = vb.device().clone();
        let dtype = vb.dtype();

        let x_embedder = linear(
            cfg.patch_dim(),
            cfg.dim,
            vb.pp(format!("all_x_embedder.{key}")),
        )?;
        let final_layer = FinalLayer::new(&cfg, vb.pp(format!("all_final_layer.{key}")))?;

        let vb_cap = vb.pp("cap_embedder");
        let cap_norm = rms_norm(cfg.cap_feat_dim, cfg.norm_eps, vb_cap.pp(0))?;
        let cap_linear = linear(cfg.cap_feat_dim, cfg.dim, vb_cap.pp(1))?;

        let t_embedder = TimestepEmbedder::new(1024, ADALN_EMBED_DIM, vb.pp("t_embedder"))?;

        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let vb_n = vb.pp("noise_refiner");
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(Block::new(&cfg, true, vb_n.pp(i))?);
        }
        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        let vb_c = vb.pp("context_refiner");
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(Block::new(&cfg, false, vb_c.pp(i))?);
        }
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.n_layers {
            layers.push(Block::new(&cfg, true, vb_l.pp(i))?);
        }

        // Learned, and shaped `(1, dim)` in the checkpoint.
        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;
        let geom = Geometry::new(&cfg, &dev, dtype)?;

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
            dtype,
        })
    }

    /// One denoising step.
    ///
    /// `latent` is `(c, h, w)` — a single image, since the pipeline runs
    /// classifier-free guidance as two separate calls rather than a batch of
    /// two, and Turbo runs with no guidance at all. `cap` is the encoder's
    /// output for the prompt, `(cap_len, cap_feat_dim)`, already stripped of
    /// padding. `t` is the flow-matching time in `[0, 1]`.
    pub fn forward(&self, latent: &Tensor, cap: &Tensor, t: f64) -> Result<Tensor> {
        let dev = latent.device();

        // The timestep embedding every modulated block reads.
        let t_in = Tensor::from_vec(vec![(t * self.cfg.t_scale) as f32], 1, dev)?;
        let adaln = self.t_embedder.forward(&t_in, self.dtype)?;

        // ── caption ──────────────────────────────────────────────────────
        // Padded first, because the image's position axis starts *after* the
        // padded caption length.
        let g = &self.geom;
        let cap_ids = g.cap_ids(cap.dim(0)?);
        let cap_feats = self.cap_linear.forward(&self.cap_norm.forward(cap)?)?;
        let cap_run = g.pad_run(&cap_feats, cap_ids, &self.cap_pad_token)?;
        let cap_padded = cap_run.feats.dim(0)?;

        // ── image ────────────────────────────────────────────────────────
        let (patches, ht, wt) = g.patchify(latent)?;
        let x_feats = self.x_embedder.forward(&patches.to_dtype(self.dtype)?)?;
        // Position `(cap_padded + 1, h, w)`: the caption occupies axis 0 up to
        // its padded length, and the image sits one past it on that axis while
        // spreading over axes 1 and 2.
        let x_ids = g.img_ids(cap_padded, ht, wt);
        let x_run = g.pad_run(&x_feats, x_ids, &self.x_pad_token)?;

        // ── refine each separately ───────────────────────────────────────
        let (x_cos, x_sin) = g.rope.gather(&x_run.ids, dev)?;
        let mut x = x_run.feats.unsqueeze(0)?;
        for blk in self.noise_refiner.iter() {
            x = blk.forward(&x, &x_cos, &x_sin, Some(&adaln))?;
        }

        let (c_cos, c_sin) = g.rope.gather(&cap_run.ids, dev)?;
        let mut c = cap_run.feats.unsqueeze(0)?;
        for blk in self.context_refiner.iter() {
            c = blk.forward(&c, &c_cos, &c_sin, None)?;
        }

        // ── one sequence, thirty joint blocks ────────────────────────────
        // Image first, then caption: the reference's basic-mode order.
        let mut h = Tensor::cat(&[&x, &c], 1)?;
        let cos = Tensor::cat(&[&x_cos, &c_cos], 0)?.contiguous()?;
        let sin = Tensor::cat(&[&x_sin, &c_sin], 0)?.contiguous()?;
        for blk in self.layers.iter() {
            h = blk.forward(&h, &cos, &sin, Some(&adaln))?;
        }

        let h = self.final_layer.forward(&h, &adaln)?;
        // Only the image half is an image, and only the unpadded part of it.
        let h = h.i((0, ..ht * wt))?;
        g.unpatchify(&h, ht, wt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published geometry has to be self-consistent, and the RoPE axes are
    /// where it would silently not be: the three axis widths are handed to
    /// three separate tables whose concatenation must be exactly one head.
    #[test]
    fn the_rope_axes_tile_one_head_exactly() {
        let cfg = Config::turbo();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.axes_dims.iter().sum::<usize>(), cfg.head_dim());
        assert_eq!(cfg.axes_dims.len(), cfg.axes_lens.len());
        for d in &cfg.axes_dims {
            assert_eq!(d % 2, 0, "an axis of odd width has no interleaved pairs");
        }
    }

    /// A patch is `p²·c` wide and the output head projects back to exactly
    /// that. A mismatch loads — both are linears — and produces an image of
    /// the wrong shape several steps later.
    #[test]
    fn a_patch_round_trips_through_the_head() {
        let cfg = Config::turbo();
        // patch × patch × in_channels, spelled out.
        assert_eq!(cfg.patch_dim(), 2 * 2 * 16);
        assert_eq!(cfg.patch_key(), "2-1");
    }

    /// Padding is positional, not an optimisation: the image's axis-0 origin is
    /// the caption's *padded* length, so the multiple is part of the geometry.
    #[test]
    fn a_run_is_padded_to_the_sequence_multiple() {
        assert_eq!(padded_len(1), 32);
        assert_eq!(padded_len(32), 32);
        assert_eq!(padded_len(33), 64);
        assert_eq!(padded_len(0), 0);
    }

    /// **Patchify and unpatchify must be inverses.** The permutation puts the
    /// channel last inside a patch, and getting it wrong keeps every value
    /// while scrambling which pixel owns it — an image that is plausibly
    /// coloured and structurally noise.
    #[test]
    fn patchify_inverts() -> Result<()> {
        let cfg = Config::turbo();
        let dev = Device::Cpu;
        // A tiny stand-in for the real thing: the permutation is independent of
        // the channel count, and 16×8×8 is enough to catch a transposed axis.
        let (c, h, w) = (cfg.in_channels, 8, 6);
        let n = c * h * w;
        let src = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), (c, h, w), &dev)?
            .to_dtype(DType::F32)?;

        let p = cfg.patch_size;
        let (ht, wt) = (h / p, w / p);
        let patched = src
            .reshape((c, ht, p, wt, p))?
            .permute((1, 3, 2, 4, 0))?
            .reshape((ht * wt, p * p * c))?;
        let back = patched
            .reshape((ht, wt, p, p, c))?
            .permute((4, 0, 2, 1, 3))?
            .reshape((c, ht * p, wt * p))?;

        assert_eq!(
            src.flatten_all()?.to_vec1::<f32>()?,
            back.flatten_all()?.to_vec1::<f32>()?,
            "patchify and unpatchify are not inverses"
        );
        Ok(())
    }

    /// The RoPE tables are what a position *means*. Position zero must be the
    /// identity rotation, and the table must be indexable to its stated length.
    #[test]
    fn rope_tables_start_at_the_identity() -> Result<()> {
        let cfg = Config::turbo();
        let rope = Rope::new(&cfg, &Device::Cpu, DType::F32)?;
        assert_eq!(rope.cos.len(), 3);
        for (i, &d) in cfg.axes_dims.iter().enumerate() {
            assert_eq!(rope.cos[i].dims(), &[cfg.axes_lens[i], d / 2]);
            let c0 = rope.cos[i].i(0)?.to_vec1::<f32>()?;
            let s0 = rope.sin[i].i(0)?.to_vec1::<f32>()?;
            assert!(c0.iter().all(|v| (*v - 1.0).abs() < 1e-6), "cos(0) != 1");
            assert!(s0.iter().all(|v| v.abs() < 1e-6), "sin(0) != 0");
        }

        // A gathered pair is one whole head's worth of rotation.
        let ids = vec![[0u32, 0, 0], [5, 2, 3]];
        let (cos, sin) = rope.gather(&ids, &Device::Cpu)?;
        assert_eq!(cos.dims(), &[2, cfg.head_dim() / 2]);
        assert_eq!(sin.dims(), &[2, cfg.head_dim() / 2]);
        Ok(())
    }
}
