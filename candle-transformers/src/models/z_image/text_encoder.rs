//! The prompt encoder — Qwen3-4B over GGUF weights, read mid-stack.
//!
//! # Why this is not [`crate::models::quantized_qwen3`]
//!
//! Two reasons, and either alone would be enough.
//!
//! **The names.** The published Z-Image text-encoder GGUF is a conversion of the
//! release's `text_encoder/` safetensors and keeps *transformers* tensor names —
//! `model.layers.0.self_attn.q_proj.weight`, `model.norm.weight`.
//! [`crate::models::quantized_qwen3`] reads llama.cpp's (`blk.0.attn_q.weight`),
//! which is a different file, not a different spelling of this one.
//!
//! **The job.** That model is a decoder: KV caches, an `lm_head`, a batched wave
//! engine, one token out per step. This encodes a prompt — one forward over the
//! whole sequence, no cache, no sampling, and the answer is a hidden state
//! rather than a logit.
//!
//! # It stops early, on purpose
//!
//! Z-Image conditions on `hidden_states[-2]`: every layer but the last, with the
//! output norm **not** applied. So [`TextEncoder::new`] is told how many layers
//! to leave off the end and simply does not load them, and it never loads
//! `model.norm` at all — an encoder taking a mid-stack activation has nothing to
//! apply the LM's output norm to. The reasoning behind the choice is the same
//! one Stable Diffusion's CLIP makes: the final layer is specialised toward
//! predicting the next token, which is not what a conditioning signal wants.
//!
//! # int8
//!
//! Every projection is repacked at load into its KO twin, so the encode runs
//! q8a128 × KO on the tensor cores. The file is Q8_0 and `Int8Mode::Precision`
//! maps Q8_0 → Q8_KO, so nothing is given up for it. Every projection here tiles
//! — the narrowest is `q_norm`'s 128, and that is a norm, not a matmul.
//!
//! # The embedding table never reaches the card
//!
//! `model.embed_tokens` is 151,936 × 2,560 — 1.55 GiB dequantised, to look up
//! the couple of dozen rows a prompt actually names. So the lookup is a host
//! index and a 300 KiB transfer ([`EmbedTable`]), which is the shape hot-path
//! invariant 3 sanctions for exactly this: "token ids → host, CPU index_select,
//! transfer in, a pure index + transfer that keeps the embed table off VRAM".
//! [`TextEncoder`] therefore takes embeddings rather than ids, and does not load
//! that tensor at all — which for a co-resident guest is the difference between
//! the encoder fitting beside the transformer and not.

use std::sync::Arc;

use candle::quantized::Int8Mode;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::RmsNorm;

use crate::models::operand_guard::expect_dtype;
use crate::models::qwen3::{Config, Qwen3RotaryEmbedding};
use crate::quantized_nn::{linear_b_mode, Linear};
use crate::quantized_var_builder::VarBuilder;
use crate::utils::repeat_kv;

/// The activation width.
///
/// F32 throughout: the prompt is a few dozen tokens, so nothing here is large
/// enough for a narrower activation to buy anything, and the int8 matmul
/// quantizes its own input regardless of what it is handed.
const DTYPE: DType = DType::F32;

/// An RMSNorm whose weight comes from a GGUF — dequantised once at load, because
/// a norm is a vector every token is multiplied by rather than a matmul operand.
fn rms(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let w = vb.get(dim, "weight")?.dequantize(vb.device())?;
    Ok(RmsNorm::new(w.to_dtype(DTYPE)?, eps))
}

#[derive(Debug)]
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &Config, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: linear_b_mode(h, i, false, mode, DTYPE, vb.pp("gate_proj"))?,
            up: linear_b_mode(h, i, false, mode, DTYPE, vb.pp("up_proj"))?,
            down: linear_b_mode(i, h, false, mode, DTYPE, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        self.down.forward(&(gate * self.up.forward(xs)?)?)
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    rope: Arc<Qwen3RotaryEmbedding>,
}

impl Attention {
    fn new(
        cfg: &Config,
        rope: Arc<Qwen3RotaryEmbedding>,
        mode: Int8Mode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (h, d) = (cfg.hidden_size, cfg.head_dim);
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let b = cfg.attention_bias;
        Ok(Self {
            q_proj: linear_b_mode(h, nh * d, b, mode, DTYPE, vb.pp("q_proj"))?,
            k_proj: linear_b_mode(h, nkv * d, b, mode, DTYPE, vb.pp("k_proj"))?,
            v_proj: linear_b_mode(h, nkv * d, b, mode, DTYPE, vb.pp("v_proj"))?,
            o_proj: linear_b_mode(nh * d, h, b, mode, DTYPE, vb.pp("o_proj"))?,
            q_norm: rms(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads: nh,
            n_kv_heads: nkv,
            kv_groups: nh / nkv,
            head_dim: d,
            rope,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            t.reshape((b, l, n, self.head_dim))?.transpose(1, 2)
        };
        let q = heads(self.q_proj.forward(xs)?, self.n_heads)?;
        let k = heads(self.k_proj.forward(xs)?, self.n_kv_heads)?;
        let v = heads(self.v_proj.forward(xs)?, self.n_kv_heads)?.contiguous()?;

        // The QK norms are per *head*, over `head_dim`, so the heads are flattened
        // into the row axis rather than normed across the model width.
        let q =
            self.q_norm
                .forward(&q.flatten(0, 2)?)?
                .reshape((b, self.n_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.n_kv_heads,
            l,
            self.head_dim,
        ))?;

        let (q, k) = self.rope.apply(&q, &k, 0)?;
        let k = repeat_kv(k, self.kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
        let out =
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, l, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

#[derive(Debug)]
struct Layer {
    attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn new(
        cfg: &Config,
        rope: Arc<Qwen3RotaryEmbedding>,
        mode: Int8Mode,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, rope, mode, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, mode, vb.pp("mlp"))?,
            ln1: rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.ln1.forward(xs)?, mask)?)?;
        let h = self.mlp.forward(&self.ln2.forward(&xs)?)?;
        xs + h
    }
}

/// The embedding table, on the host.
///
/// Held dequantised at [`DTYPE`]'s width rather than the file's, because the
/// per-row cost of decoding Q8_0 on demand is paid on every prompt while this is
/// paid once — and 778 MiB of host RAM is not the resource under pressure here.
/// The card never sees it; [`Self::rows`] returns only the tokens asked for.
#[derive(Debug)]
pub struct EmbedTable {
    rows: Tensor,
    hidden: usize,
}

impl EmbedTable {
    /// Read `model.embed_tokens` out of a GGUF and dequantise it on the host.
    pub fn from_gguf(path: &std::path::Path, cfg: &Config) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        let q = content.tensor(&mut file, EMBED_TOKENS, &Device::Cpu)?;
        let rows = q.dequantize(&Device::Cpu)?.to_dtype(DTYPE)?;
        let (vocab, hidden) = rows.dims2()?;
        if hidden != cfg.hidden_size {
            candle::bail!(
                "{EMBED_TOKENS} is {vocab}×{hidden} and the config says the model is \
                 {}-wide — these are not the same checkpoint",
                cfg.hidden_size
            );
        }
        Ok(Self { rows, hidden })
    }

    /// The rows `ids` name, on `device`, as `[1, len, hidden]`.
    pub fn rows(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let idx = Tensor::new(ids, &Device::Cpu)?;
        self.rows
            .index_select(&idx, 0)?
            .reshape((1, ids.len(), self.hidden))?
            .to_device(device)
    }
}

/// The one tensor [`TextEncoder`] deliberately does not load.
pub const EMBED_TOKENS: &str = "model.embed_tokens.weight";

/// Qwen3-4B, loaded to a chosen depth.
#[derive(Debug)]
pub struct TextEncoder {
    layers: Vec<Layer>,
    device: Device,
}

impl TextEncoder {
    /// Load every layer but the last `skip_last`, with each projection repacked
    /// for `mode`.
    ///
    /// `skip_last` is a *load* argument rather than a forward one because the
    /// layers it names are never run: leaving it to the caller would download
    /// and place ~120 MB of weights per skipped layer that nothing reads. Z-Image
    /// passes 1, for `hidden_states[-2]`.
    pub fn new(cfg: &Config, skip_last: usize, mode: Int8Mode, vb: VarBuilder) -> Result<Self> {
        if skip_last >= cfg.num_hidden_layers {
            candle::bail!(
                "skip_last {skip_last} leaves nothing of a {}-layer encoder",
                cfg.num_hidden_layers
            );
        }
        if cfg.use_sliding_window {
            candle::bail!("sliding window is not supported");
        }
        let device = vb.device().clone();
        let rope = Arc::new(Qwen3RotaryEmbedding::new(DTYPE, cfg, &device)?);
        let vb_l = vb.pp("model.layers");
        let run = cfg.num_hidden_layers - skip_last;
        let mut layers = Vec::with_capacity(run);
        for i in 0..run {
            layers.push(Layer::new(cfg, rope.clone(), mode, vb_l.pp(i.to_string()))?);
        }
        Ok(Self { layers, device })
    }

    /// The device this encoder's weights are on, so a caller can put the
    /// embeddings there.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The hidden state after the loaded layers, `[len, hidden]`.
    ///
    /// `embeds` is `[1, len, hidden]` from [`EmbedTable::rows`] — the lookup is
    /// the caller's, so the table stays off the card.
    ///
    /// Causal, as the model was trained: it is a decoder being read rather than
    /// a bidirectional encoder, and the reference conditions on exactly these
    /// activations.
    pub fn encode(&self, embeds: &Tensor) -> Result<Tensor> {
        let (_, l, _) = embeds.dims3()?;
        // Validated rather than cast: the table is dequantised to this width on
        // the host, and a caller that hands over another one would otherwise
        // quietly run every layer at a precision nothing chose.
        expect_dtype(embeds, DTYPE, "z-image text encoder embedding")?;
        let mut h = embeds.clone();
        let mask = self.causal_mask(l)?;
        for layer in self.layers.iter() {
            h = layer.forward(&h, &mask)?;
        }
        h.i(0)
    }

    fn causal_mask(&self, l: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let m: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&m, (1, 1, l, l), &self.device)
    }
}

#[cfg(test)]
mod tests {
    use candle::quantized::ko_quant::ko_tileable;

    /// The released encoder's geometry, from `text_encoder/config.json`.
    const HIDDEN: usize = 2560;
    const INTERMEDIATE: usize = 9728;
    const HEAD_DIM: usize = 128;
    const N_HEADS: usize = 32;
    const N_KV_HEADS: usize = 8;

    /// **The header's claim that every projection here tiles.** Unlike the
    /// transformer, this model has no weight narrow enough to miss the KO
    /// matmul's 32×128 tiling — so if one ever does, it is the header that is
    /// wrong, and the symptom otherwise would be a silent drop to the
    /// dequantising path for that weight alone.
    #[test]
    fn every_projection_tiles_for_the_int8_matmul() {
        let q = N_HEADS * HEAD_DIM;
        let kv = N_KV_HEADS * HEAD_DIM;
        for (name, out, inp) in [
            ("q_proj", q, HIDDEN),
            ("k_proj", kv, HIDDEN),
            ("v_proj", kv, HIDDEN),
            ("o_proj", HIDDEN, q),
            ("gate_proj", INTERMEDIATE, HIDDEN),
            ("up_proj", INTERMEDIATE, HIDDEN),
            ("down_proj", HIDDEN, INTERMEDIATE),
        ] {
            assert!(ko_tileable(out, inp), "{name} [{out}, {inp}]");
        }
    }
}
