//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-8B) (architecture based on official implementations)
//!
// OLD: use super::batched_inference::{BatchedInferenceSession, ManagedBatchedModel as BatchableModel};
#[cfg(feature = "cuda")]
use super::batched_layer::{BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::BatchedModelCore;
use super::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
#[cfg(feature = "cuda")]
use candle::quantized::cuda::DynamicActs;
#[cfg(feature = "cuda")]
use candle::quantized::register_mmap_cuda;
use candle::quantized::{gguf_file, GgmlDType, Int8Mode, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::{Arc, RwLock};

/// Initial number of RoPE positions to precompute for quantized Qwen3 models.
///
/// Defaulting to 0 avoids up-front RoPE allocation; tables are extended on demand.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

type SharedCis = Arc<RwLock<CisPrecomputations>>;

/// Native context length for Qwen3 models (per model cards).
///
/// If a GGUF advertises a larger `context_length` but does not include an explicit
/// RoPE scaling factor, we infer a single-factor scaling as `context_length / native`.
const QWEN3_NATIVE_CONTEXT_LEN: usize = 32_768;

fn infer_rope_scaling_factor(context_length: usize, explicit: Option<f32>) -> Option<f32> {
    if let Some(f) = explicit {
        return Some(f);
    }
    if context_length > QWEN3_NATIVE_CONTEXT_LEN {
        let f = context_length as f32 / QWEN3_NATIVE_CONTEXT_LEN as f32;
        if f.is_finite() && f > 0.0 {
            return Some(f);
        }
    }
    None
}

fn qwen_inv_freq(head_dim: usize, rope_theta: f32, rope_scaling_factor: Option<f32>) -> Vec<f32> {
    // Apply a single RoPE scaling factor the same way as common GGUF exporters:
    // inv_freq = 1 / (factor * theta^(i/d))  (equivalently inv_freq /= factor).
    let factor = rope_scaling_factor.unwrap_or(1.0);
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / (factor * rope_theta.powf(i as f32 / head_dim as f32)))
        .collect()
}

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_up_proj: Option<QMatMul>,
    gate_proj: Option<QMatMul>,
    up_proj: Option<QMatMul>,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_w = gg.tensor(&format!("{prefix}.ffn_gate.weight"))?;
        let up_w = gg.tensor(&format!("{prefix}.ffn_up.weight"))?;

        let try_fuse = gg.device.is_cuda()
            && gate_w.dtype() == up_w.dtype()
            && !matches!(
                gate_w.dtype(),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            );

        let (gate_up_proj, gate_proj, up_proj) = if try_fuse {
            #[cfg(feature = "cuda")]
            {
                let (gate_n, gate_k) = gate_w.shape().dims2()?;
                let (up_n, up_k) = up_w.shape().dims2()?;
                if gate_n != up_n || gate_k != up_k {
                    candle::bail!(
                        "cannot fuse ffn_gate/ffn_up due to shape mismatch: gate=({}, {}) up=({}, {})",
                        gate_n,
                        gate_k,
                        up_n,
                        up_k
                    );
                }
                let fused = QTensor::concat_rows_cuda(&[&gate_w, &up_w])?;
                (Some(QMatMul::from_qtensor(fused)?), None, None)
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle::bail!("fused gate+up requires the cuda feature");
            }
        } else {
            (
                None,
                Some(QMatMul::from_qtensor(gate_w)?),
                Some(QMatMul::from_qtensor(up_w)?),
            )
        };

        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_up_proj,
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (gate, up) = if let Some(w) = &self.gate_up_proj {
            let gu = w.forward(x)?;
            let (_, _, out_dim) = gu.dims3()?;
            if out_dim % 2 != 0 {
                candle::bail!("unexpected fused gate+up output dim {out_dim} (not even)");
            }
            let half = out_dim / 2;
            let gate = gu.narrow(2, 0, half)?;
            let up = gu.narrow(2, half, half)?;
            (gate, up)
        } else {
            let gate_proj = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing gate_proj".into()))?;
            let up_proj = self
                .up_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing up_proj".into()))?;
            (gate_proj.forward(x)?, up_proj.forward(x)?)
        };
        let gated = (&gate.apply(&self.act_fn)? * &up)?;
        self.down_proj.forward(&gated)
    }

    /// B3 consumer: gate/up over a producer-prepared (fused ln2) activation, shared across both
    /// projections (no redundant ln2->q8a128). CUDA only.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(&self, acts: &DynamicActs, out_dtype: DType) -> Result<Tensor> {
        let (mut gate, mut up) = if let Some(w) = &self.gate_up_proj {
            let mut gu = w.forward_dynamic(acts.as_dynamic(), out_dtype)?;
            let (_, _, out_dim) = gu.dims3()?;
            if out_dim % 2 != 0 {
                candle::bail!("unexpected fused gate+up output dim {out_dim} (not even)");
            }
            // Coerce the fused output to out_dtype ONCE, in place, before splitting it into the
            // gate/up views: `gu` is owned + contiguous here so the cast is allocation-free, whereas
            // casting the two aliasing narrows separately forces two fallback allocations.
            gu.to_dtype_mut(out_dtype)?;
            let half = out_dim / 2;
            (gu.narrow(2, 0, half)?, gu.narrow(2, half, half)?)
        } else {
            let gate_proj = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing gate_proj".into()))?;
            let up_proj = self
                .up_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing up_proj".into()))?;
            (
                gate_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                up_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
            )
        };
        // Run silu/mul/down in out_dtype: the Float path returns the activation dtype (F16), but
        // MLP intermediates can exceed F16's ~65504 range, so compute in out_dtype (BF16). The fused
        // path already coerced `gu` above and the int8 path already returns out_dtype, so these are
        // no-ops except on the separate-weight Float path.
        gate.to_dtype_mut(out_dtype)?;
        up.to_dtype_mut(out_dtype)?;
        let gated = (&gate.apply(&self.act_fn)? * &up)?;
        let mut out = self.down_proj.forward(&gated)?;
        out.to_dtype_mut(out_dtype)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cis: SharedCis,
}

impl RotaryEmbedding {
    fn new(
        _dtype: DType,
        head_dim: usize,
        _max_position_embeddings: usize,
        rope_theta: f64,
        rope_scaling_factor: Option<f32>,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq = qwen_inv_freq(head_dim, rope_theta as f32, rope_scaling_factor);
        Ok(Self {
            cis: Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
                inv_freq,
                MAX_ROPE_SEQ_LEN,
                ROPE_EXTEND_CHUNK,
                dev,
            )?)),
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let (cos, sin) = {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.narrow_growable(0, offset, seq_len, q.dtype())?
        };
        let q = if q.is_contiguous() {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k = if k.is_contiguous() {
            k.clone()
        } else {
            k.contiguous()?
        };
        let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    // Separate q/k/v weights. On the int8 path one segmented kernel launch covers all three (no
    // weight stitching); the FP path runs them as three matmuls.
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_w = gg.tensor(&format!("{prefix}.attn_q.weight"))?;
        let k_w = gg.tensor(&format!("{prefix}.attn_k.weight"))?;
        let v_w = gg.tensor(&format!("{prefix}.attn_v.weight"))?;

        // q/k/v kept separate (no concat): int8 fuses them at launch via the segmented kernel.
        let q_proj = QMatMul::from_qtensor(q_w)?;
        let k_proj = QMatMul::from_qtensor(k_w)?;
        let v_proj = QMatMul::from_qtensor(v_w)?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            span_attn,
        })
    }

    #[inline]
    fn project_qkv_with_compute_type(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.q_proj.forward(x)?,
            self.k_proj.forward(x)?,
            self.v_proj.forward(x)?,
        ))
    }

    /// B1 consumer: q/k/v over a producer-prepared (fused ln1) activation.
    #[cfg(feature = "cuda")]
    fn project_qkv(
        &self,
        acts: &DynamicActs,
        out_dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = &self.q_proj;
        let k = &self.k_proj;
        let v = &self.v_proj;
        match acts {
            // int8: ONE segmented launch over the three KO weights (no concat) — float-identical
            // to three separate matmuls, with full GPU occupancy so the tiny k/v stop starving.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[q.inner(), k.inner(), v.inner()],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                Ok((
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                ))
            }
            DynamicActs::Float(_) => Ok((
                q.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                k.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                v.forward_dynamic(acts.as_dynamic(), out_dtype)?,
            )),
        }
    }

    fn forward(&self, cache: &mut KvCache, x: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let (q, k, v) = self.project_qkv_with_compute_type(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            cache.reset();
        }
        let (k, v) = cache.append(&k, &v)?;

        // KV cache already returns contiguous tensors, repeat_kv works with views
        // Removing redundant contiguous() saves 3 memory allocations per layer

        // Standard attention implementation - used as fallback or primary path
        let standard_attention = || -> Result<Tensor> {
            let k = repeat_kv(k.clone(), self.num_kv_groups)?;
            let v = repeat_kv(v.clone(), self.num_kv_groups)?;
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if l > 1 {
                // Generate causal mask on-demand for multi-token sequences
                let cache_len = offset + l;
                let mask: Vec<_> = (0..l)
                    .flat_map(|i| {
                        (0..cache_len).map(move |j| {
                            if j > offset + i {
                                f32::NEG_INFINITY
                            } else {
                                0.0f32
                            }
                        })
                    })
                    .collect();
                let mask_tensor = Tensor::from_vec(mask, (1, 1, l, cache_len), q.device())?
                    .to_dtype(scores.dtype())?;
                scores = scores.broadcast_add(&mask_tensor)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            probs.matmul(&v)
        };

        let ctx = if l > 1 {
            // Use Flash Attention for multi-token sequences
            // Flash Attention provides its own causal masking, so we ignore attn_mask
            #[cfg(feature = "flash-attn")]
            {
                // Match Llama: dtype conversion handled internally by flash-attn.
                let q_fa = q.transpose(1, 2)?;
                let k_fa = k.transpose(1, 2)?;
                let v_fa = v.transpose(1, 2)?;
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                match candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true) {
                    Ok(out) => {
                        let out = out.transpose(1, 2)?;
                        if out.dtype() != q.dtype() {
                            out.to_dtype(q.dtype())?
                        } else {
                            out
                        }
                    }
                    Err(_) => standard_attention()?,
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            standard_attention()?
        } else {
            standard_attention()?
        }; // (B, H, L, D)
        let reshaped_ctx =
            ctx.transpose(1, 2)?
                .contiguous()?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

/// Implement the `BatchedAttentionLayer` trait for Qwen3 layers.
///
/// This enables the use of generic batched layer processing from `batched_layer` module.
/// Qwen3 has Q/K normalization after QKV projection, which is applied inside `project_qkv()`.
#[cfg(feature = "cuda")]
impl BatchedAttentionLayer for LayerWeights {
    fn n_head(&self) -> usize {
        self.self_attn.num_heads
    }

    fn n_kv_head(&self) -> usize {
        self.self_attn.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.self_attn.head_dim
    }

    fn o_proj(&self) -> &QMatMul {
        &self.self_attn.o_proj
    }

    /// B1 producer: fuse ln1 -> q8a128 (int8) or FP rms_norm (Off) in one kernel.
    #[cfg(feature = "cuda")]
    fn attention_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs> {
        self.ln1.forward_dynamic(x, mode)
    }

    /// B1 consumer: q/k/v over the fused ln1 activation, then q/k/v RMSNorm + reshapes.
    #[cfg(feature = "cuda")]
    fn project_qkv(&self, acts: &DynamicActs, out_dtype: DType) -> Result<QkvProjection> {
        let (b_sz, seq_len) = match acts {
            DynamicActs::Float(t) => {
                let (b, s, _) = t.dims3()?;
                (b, s)
            }
            DynamicActs::Int8(op) => match op.lead.as_slice() {
                &[b, s] => (b, s),
                other => {
                    candle::bail!("project_qkv: expected [b, seq] lead, got {other:?}")
                }
            },
        };
        let (q, k, v) = self.self_attn.project_qkv(acts, out_dtype)?;
        let n_head = self.self_attn.num_heads;
        let n_kv_head = self.self_attn.num_kv_heads;
        let head_dim = self.self_attn.head_dim;
        let q = q
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let q_flat = self.self_attn.q_norm.forward(&q.flatten(0, 2)?)?;
        let q = q_flat
            .reshape((b_sz, n_head, seq_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, n_head * head_dim))?;
        let k = k
            .reshape((b_sz, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        let k_flat = self.self_attn.k_norm.forward(&k.flatten(0, 2)?)?;
        let k = k_flat
            .reshape((b_sz, n_kv_head, seq_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, n_kv_head * head_dim))?;
        Ok(QkvProjection { q, k, v })
    }

    /// B3 producer: fuse ln2 -> q8a128 (int8) or FP rms_norm (Off).
    #[cfg(feature = "cuda")]
    fn ffn_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs> {
        self.ln2.forward_dynamic(x, mode)
    }

    /// B3 consumer: dense MLP over the fused ln2 activation.
    #[cfg(feature = "cuda")]
    fn ffn_forward(&self, acts: DynamicActs, mlp_dtype: DType) -> Result<Tensor> {
        self.mlp.forward_dynamic(&acts, mlp_dtype)
    }
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&self, cache: &mut KvCache, x: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(cache, &h, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    span: tracing::Span,
    span_output: tracing::Span,
}

/// Implementation of `BatchedModelCore` for use with `BatchedInference` wrapper.
///
/// This is the new recommended way to use batched inference. The `BatchedInference`
/// wrapper handles RoPE caching at the model level, so this implementation is simpler.
#[cfg(feature = "cuda")]
impl BatchedModelCore for ModelWeights {
    type Layer = LayerWeights;

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_head(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.num_kv_heads)
            .unwrap_or(0)
    }

    fn head_dim(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.head_dim)
            .unwrap_or(0)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn embeddings(&self) -> &Embedding {
        &self.embeddings
    }

    fn layer(&self, idx: usize) -> &Self::Layer {
        &self.layers[idx]
    }

    fn final_norm(&self) -> &RmsNorm {
        &self.norm
    }

    fn output_proj(&self) -> &QMatMul {
        &self.lm_head
    }

    fn rope_interleaved(&self) -> bool {
        // Qwen3 uses standard (non-interleaved) RoPE format
        false
    }

    fn prune(&self) -> Result<()> {
        self.embeddings.compact();
        if let Some(layer) = self.layers.first() {
            if let Ok(mut cis) = layer.self_attn.rotary_emb.cis.write() {
                cis.compact();
            }
        }
        Ok(())
    }

    fn k_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.k_hi
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.k_low
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.v_hi
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.v_low
    }
}

impl ModelWeights {
    /// Load model from GGUF file with
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let md_opt_f32 = |k: &str| gg.metadata().get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| gg.metadata().get(k).and_then(|v| v.to_u32().ok());

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;

        // Some converters use different key names; accept a few common ones.
        let head_dim = md_opt_u32("qwen3.attention.key_length")
            .or_else(|| md_opt_u32("qwen3.attention.head_dim"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let max_position_embeddings = md_opt_u32("qwen3.context_length")
            .or_else(|| md_opt_u32("qwen3.max_position_embeddings"))
            .or_else(|| md_opt_u32("max_position_embeddings"))
            .unwrap_or_else(|| {
                // Keep existing behavior of failing fast if we can't determine context.
                // (We only hit this closure if all md_opt_u32 calls returned None.)
                0
            }) as usize;
        if max_position_embeddings == 0 {
            let _ = md_get("qwen3.context_length")?;
        }

        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_opt_f32("qwen3.rope.freq_base")
            .or_else(|| md_opt_f32("qwen3.rope.theta"))
            .or_else(|| md_opt_f32("rope.freq_base"))
            .or_else(|| md_opt_f32("rope.theta"))
            .unwrap_or(1_000_000f32) as f64;

        let rope_scaling_factor = md_opt_f32("qwen3.rope.scaling.factor")
            .or_else(|| md_opt_f32("qwen3.rope.scale_factor"))
            .or_else(|| md_opt_f32("rope.scaling.factor"))
            .or_else(|| md_opt_f32("rope.scale_factor"))
            .filter(|f| *f > 0.0);

        let rope_scaling_factor =
            infer_rope_scaling_factor(max_position_embeddings, rope_scaling_factor);

        // Extract model's native dtype from metadata
        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;

        let tok_embed = embed_tensor.dequantize(device)?;
        let embeddings = Embedding::new(tok_embed, hidden_size)?;

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            rope_scaling_factor,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            span,
            span_output,
        })
    }

    /// Load model from GGUF file using memory-mapped I/O for zero-copy tensor loading.
    ///
    /// This method eliminates intermediate RAM allocations and copies by using mmap:
    /// - Traditional: File → Vec<u8> → GPU (2 copies, 2x peak RAM)
    /// - This method: File (mmap) → GPU (1 copy, 1x peak RAM)
    ///
    /// Benefits:
    /// - **Eliminates RAM allocation** for tensor data
    /// - **Eliminates file→RAM copy** - only mmap→GPU remains
    /// - **Lower peak memory usage** - no temporary buffers
    /// - **OS page cache efficiency** - kernel optimizes page access
    ///
    /// # Arguments
    /// * `file_path` - Path to the GGUF file
    /// * `device` - Device to load tensors onto
    ///
    /// # Example
    /// ```no_run
    /// use candle::Device;
    /// use candle_transformers::models::quantized_qwen3::ModelWeights;
    /// use std::path::Path;
    ///
    /// let path = Path::new("model.gguf");
    /// let device = Device::cuda_if_available(0)?;
    /// let model = ModelWeights::from_gguf_by_path(path, &device)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_gguf_by_path(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        // VRAM-aware auto: Precision when the weights leave headroom, else
        // Performance (smaller); Off on CPU. Sized by the GGUF length.
        let model_bytes = std::fs::metadata(file_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        Self::from_gguf_by_path_with_int8(
            file_path,
            device,
            Int8Mode::auto_sized(device, model_bytes),
        )
    }

    /// Like from_gguf_by_path but with an explicit numeric int8mode (the test path selects it
    /// from INT8MODE); from_gguf_by_path defaults it to Int8Mode::auto.
    pub fn from_gguf_by_path_with_int8(
        file_path: &std::path::Path,
        device: &Device,
        int8mode: Int8Mode,
    ) -> Result<Self> {
        use memmap2::MmapOptions;

        // Open file and create memory map for zero-copy access
        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| candle::Error::Msg(format!("Failed to mmap file: {}", e)))?
        };

        // Register mmap with CUDA for DMA-accelerated host-to-device transfers
        #[cfg(feature = "cuda")]
        let _mmap_guard = if matches!(device, Device::Cuda(_)) {
            register_mmap_cuda(&mmap)
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let _mmap_guard: Option<()> = None;

        // Parse GGUF metadata from mmap (23x faster than reading from File!)
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut cursor)?;

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Extract model hyperparameters
        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());

        let rope_freq_base = md_opt_f32("qwen3.rope.freq_base")
            .or_else(|| md_opt_f32("qwen3.rope.theta"))
            .or_else(|| md_opt_f32("rope.freq_base"))
            .or_else(|| md_opt_f32("rope.theta"))
            .unwrap_or(1_000_000f32) as f64;

        let rope_scaling_factor = md_opt_f32("qwen3.rope.scaling.factor")
            .or_else(|| md_opt_f32("qwen3.rope.scale_factor"))
            .or_else(|| md_opt_f32("rope.scaling.factor"))
            .or_else(|| md_opt_f32("rope.scale_factor"))
            .filter(|f| *f > 0.0);

        let rope_scaling_factor =
            infer_rope_scaling_factor(max_position_embeddings, rope_scaling_factor);

        // Extract model's native dtype from metadata
        let dtype = match ct.metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        // Helper to load tensor from mmap
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        let load_qmatmul = |name: &str| -> Result<QMatMul> {
            QMatMul::from_weights_with_mode(load_tensor(name)?.into(), int8mode)
        };

        let load_rms_norm = |name: &str, eps: f64| -> Result<RmsNorm> {
            RmsNorm::from_qtensor(load_tensor(name)?, eps)
        };

        // Load embeddings
        let embed_tensor = load_tensor("token_embd.weight")?;
        let tok_embed = embed_tensor.dequantize(device)?;
        let embeddings = Embedding::new(tok_embed, hidden_size)?;

        // Create rotary embeddings
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            rope_scaling_factor,
            device,
        )?);

        // Load all layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let ln1 = load_rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ln2 = load_rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

            // Load attention weights
            let q_w = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let k_w = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let v_w = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let o_proj = load_qmatmul(&format!("{prefix}.attn_output.weight"))?;
            let q_norm = load_rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
            let k_norm = load_rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

            // q/k/v kept separate KO twins (no concat): the segmented kernel fuses them at launch.
            let q_proj = QMatMul::from_weights_with_mode(q_w.into(), int8mode)?;
            let k_proj = QMatMul::from_weights_with_mode(k_w.into(), int8mode)?;
            let v_proj = QMatMul::from_weights_with_mode(v_w.into(), int8mode)?;

            let self_attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads: num_kv_heads,
                num_kv_groups: num_attention_heads / num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
            };

            // Load MLP weights
            let gate_w = load_tensor(&format!("{prefix}.ffn_gate.weight"))?;
            let up_w = load_tensor(&format!("{prefix}.ffn_up.weight"))?;
            let down_proj = load_qmatmul(&format!("{prefix}.ffn_down.weight"))?;

            let try_fuse = device.is_cuda()
                && gate_w.dtype() == up_w.dtype()
                && !matches!(
                    gate_w.dtype(),
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                );

            let (gate_up_proj, gate_proj, up_proj) = if try_fuse {
                #[cfg(feature = "cuda")]
                {
                    let (gate_n, gate_k) = gate_w.shape().dims2()?;
                    let (up_n, up_k) = up_w.shape().dims2()?;
                    if gate_n != up_n || gate_k != up_k {
                        candle::bail!(
                            "cannot fuse ffn_gate/ffn_up due to shape mismatch: gate=({}, {}) up=({}, {})",
                            gate_n,
                            gate_k,
                            up_n,
                            up_k
                        );
                    }
                    let fused = QTensor::concat_rows_cuda(&[&gate_w, &up_w])?;
                    (
                        Some(QMatMul::from_qtensor_with_mode(fused, int8mode)?),
                        None,
                        None,
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle::bail!("fused gate+up requires the cuda feature");
                }
            } else {
                (
                    None,
                    Some(QMatMul::from_weights_with_mode(gate_w.into(), int8mode)?),
                    Some(QMatMul::from_weights_with_mode(up_w.into(), int8mode)?),
                )
            };

            let mlp = MlpWeights {
                gate_up_proj,
                gate_proj,
                up_proj,
                down_proj,
                act_fn: Activation::Silu,
                span: tracing::span!(tracing::Level::TRACE, "mlp"),
            };

            layers.push(LayerWeights {
                self_attn,
                mlp,
                ln1,
                ln2,
            });
        }

        // Load final norm and output projection
        let norm = load_rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => load_tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights_with_mode(lm_head_tensor.into(), int8mode)?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            span,
            span_output,
        })
    }

    /// Create KV caches for all layers (regular)
    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = (0..self.layers.len())
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Forward pass with strongly-typed sequence context.
    ///
    /// This is the preferred API for continuous batching scenarios where you manage
    /// multiple independent sequences. Each sequence has its own `KvCaches` instance.
    ///
    /// # Continuous Batching Semantics
    ///
    /// **Important**: Sequential calls to this method provide *continuous batching*
    /// (concurrent sequence management with weight reuse) but NOT *true GPU batch
    /// parallelism* (processing multiple sequences simultaneously in one kernel launch).
    ///
    /// Benefits achieved:
    /// - **Weight caching**: Model weights stay in L2 cache across sequences
    /// - **Concurrent management**: Handle multiple sequences in flight
    /// - **Memory efficiency**: Share model weights, separate KV caches
    ///
    /// Benefits NOT achieved:
    /// - **Batch parallelism**: Each forward() processes one sequence independently
    /// - **Tensor core batching**: No amortized kernel launch overhead
    ///
    /// This is the correct design because RoPE position embeddings are per-token
    /// and prevent batching sequences at different offsets without custom kernels.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle::{Device, Tensor};
    /// # use candle_transformers::models::quantized_qwen3::ModelWeights;
    /// # use candle_transformers::models::kv_cache_utils::SequenceContext;
    /// # fn example() -> candle::Result<()> {
    /// # let device = Device::cuda_if_available(0)?;
    /// # let model_path = std::path::Path::new("model.gguf");
    /// # let model = ModelWeights::from_gguf_by_path(model_path, &device)?;
    /// // Manage multiple independent sequences
    /// let mut seq1_caches = model.create_kv_caches(512);
    /// let mut seq2_caches = model.create_kv_caches(512);
    ///
    /// // Process sequence 1
    /// let seq1_tokens = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;
    /// let ctx1 = SequenceContext {
    ///     kv_caches: &mut seq1_caches,
    ///     offset: 0,
    ///     input_ids: &seq1_tokens,
    ///     input_len: 3,
    ///     write_offset_shift: 0,
    /// };
    /// let output1 = model.forward_with_context(ctx1)?;
    ///
    /// // Process sequence 2 (independent, weights cached in L2)
    /// let seq2_tokens = Tensor::new(&[4u32, 5, 6], &device)?.unsqueeze(0)?;
    /// let ctx2 = SequenceContext {
    ///     kv_caches: &mut seq2_caches,
    ///     offset: 0,
    ///     input_ids: &seq2_tokens,
    ///     input_len: 3,
    ///     write_offset_shift: 0,
    /// };
    /// let output2 = model.forward_with_context(ctx2)?;
    /// # Ok(())
    /// # }
    /// ```
    /// Forward pass with typed context struct (preferred API).
    pub fn forward_with_context(&self, ctx: SequenceContext<'_>) -> Result<Tensor> {
        if ctx.kv_caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                ctx.kv_caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let (_b, l) = ctx.input_ids.dims2()?;

        // Derive dtype from KV cache to ensure consistency throughout forward pass
        let embed_dtype = ctx.kv_caches.dtype();
        let mut h = self
            .embeddings
            .forward_as_dtype(ctx.input_ids, embed_dtype)?
            .contiguous()?;

        for (layer, cache) in self.layers.iter().zip(ctx.kv_caches.caches.iter_mut()) {
            h = layer.forward(cache, &h, ctx.offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?.contiguous()?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    /// Forward pass (backwards compatible API).
    pub fn forward(&self, caches: &mut KvCaches, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_with_context(SequenceContext {
            kv_caches: caches,
            offset,
            input_ids: input,
            input_len: input.dims2()?.1,
            write_offset_shift: 0,
        })
    }

    /// Forward pass returning logits for ALL positions (for perplexity evaluation).
    ///
    /// Unlike `forward` which returns only the last token's logits `[batch, vocab]`,
    /// this returns `[batch, seq_len, vocab]` so cross-entropy loss can be computed
    /// against shifted targets.
    pub fn forward_all_logits(
        &self,
        caches: &mut KvCaches,
        input: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        if caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let embed_dtype = caches.dtype();
        let mut h = self
            .embeddings
            .forward_as_dtype(input, embed_dtype)?
            .contiguous()?;
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward(cache, &h, offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        self.lm_head.forward(&h)
    }

    /// Returns the RoPE inverse frequency vector used by this model.
    ///
    /// This includes any RoPE scaling (e.g., for extended context) that was
    /// configured when the model was loaded. Required when wrapping the model
    /// in `BatchedInference` to ensure the RoPE tables match.
    pub fn rope_inv_freq(&self) -> Option<Vec<f32>> {
        self.layers.first().and_then(|l| {
            l.self_attn
                .rotary_emb
                .cis
                .read()
                .ok()
                .and_then(|cis| cis.inv_freq_vec())
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;
    use candle::quantized::gguf_file;

    #[test]
    #[ignore] // Run manually with: cargo test benchmark_large_model_sequential --release --features cuda -- --ignored --nocapture
    fn benchmark_large_model_sequential() -> Result<()> {
        // Benchmark large 7B-8B model with sequential loading
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        // Using Llama-3.2-3B (larger than 0.6B but not too huge for testing)
        let repo = api.model("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string());

        let model_path = repo.get("Llama-3.2-3B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("\n=== Large Model Sequential Loading Benchmark ===");
        println!("Model: Llama-3.2-3B-Instruct-Q4_K_M");
        println!("Model path: {:?}\n", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "GPU required for this benchmark. CUDA error: {}",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Check file size
        let metadata = std::fs::metadata(&model_path)?;
        println!(
            "File size: {:.2} GB\n",
            metadata.len() as f64 / 1_000_000_000.0
        );

        // Warm up run
        println!("Warming up (loading once to populate OS cache)...");
        let mut file = std::fs::File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        use crate::models::quantized_llama::ModelWeights as LlamaModelWeights;
        let _model = LlamaModelWeights::from_gguf(content, &mut file, &device)?;
        println!("Warmup complete.\n");

        // Actual benchmark - run 3 times
        println!("Running 3 timed loads with File→RAM→GPU method...");
        let mut durations = Vec::new();
        for i in 0..3 {
            let mut file = std::fs::File::open(&model_path)?;
            let content = gguf_file::Content::read(&mut file)?;

            let start = std::time::Instant::now();
            let _model = LlamaModelWeights::from_gguf(content, &mut file, &device)?;
            let duration = start.elapsed();

            println!("  Run {}: {:.3}s", i + 1, duration.as_secs_f64());
            durations.push(duration.as_secs_f64());
        }

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== Sequential Loading Results (File→RAM→GPU) ===");
        println!("Average: {:.3}s", avg);
        println!("Min:     {:.3}s", min);
        println!("Max:     {:.3}s", max);

        Ok(())
    }

    #[test]
    #[ignore] // Run manually with: cargo test benchmark_large_model_mmap --release --features cuda -- --ignored --nocapture
    fn benchmark_large_model_mmap() -> Result<()> {
        // Benchmark large 7B-8B model with mmap loading
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        // Using Llama-3.2-3B (larger than 0.6B but not too huge for testing)
        let repo = api.model("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string());

        let model_path = repo.get("Llama-3.2-3B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("\n=== Large Model mmap Loading Benchmark ===");
        println!("Model: Llama-3.2-3B-Instruct-Q4_K_M");
        println!("Model path: {:?}\n", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "GPU required for this benchmark. CUDA error: {}",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Check file size
        let metadata = std::fs::metadata(&model_path)?;
        println!(
            "File size: {:.2} GB\n",
            metadata.len() as f64 / 1_000_000_000.0
        );

        // Warm up run
        println!("Warming up (loading once to populate OS cache)...");
        use crate::models::quantized_llama::ModelWeights as LlamaModelWeights;
        let _model = LlamaModelWeights::from_gguf_by_path(&model_path, &device)?;
        println!("Warmup complete.\n");

        // Actual benchmark - run 3 times
        println!("Running 3 timed loads with File→GPU direct (mmap) method...");
        let mut durations = Vec::new();
        for i in 0..3 {
            let start = std::time::Instant::now();
            let _model = LlamaModelWeights::from_gguf_by_path(&model_path, &device)?;
            let duration = start.elapsed();

            println!("  Run {}: {:.3}s", i + 1, duration.as_secs_f64());
            durations.push(duration.as_secs_f64());
        }

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== mmap Loading Results (File→GPU direct) ===");
        println!("Average: {:.3}s", avg);
        println!("Min:     {:.3}s", min);
        println!("Max:     {:.3}s", max);

        Ok(())
    }

    #[test]
    #[ignore] // Slow without CUDA. Run with: cargo test --release --features cuda -- --ignored test_parallel_batched_forwarding
    fn test_parallel_batched_forwarding() -> Result<()> {
        #[cfg(not(all(feature = "cuda")))]
        println!("⚠ WARNING: This test should be run with --features cuda for optimal performance");
        #[cfg(not(all(feature = "cuda")))]
        println!(
            "⚠ Current build is missing performance-critical features. Results may be slower.\n"
        );

        println!("\n=== Setting up Test Parameters ===\n");

        let num_generate_tokens = 20;
        let dialect = Dialect::chat_ml();

        // Download tokenizer.json (Qwen3) from HuggingFace.
        // We keep this runtime-loaded so the test can validate real token ranges.
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
        let tok_repo = api.model("Qwen/Qwen3-8B".to_string());
        let tokenizer_path = tok_repo.get("tokenizer.json").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download tokenizer.json: {}. This test requires internet access.",
                e
            ))
        })?;
        let tokenizer_json = std::fs::read_to_string(&tokenizer_path).map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to read tokenizer.json from {:?}: {}",
                tokenizer_path, e
            ))
        })?;

        let params = TestParams::new(num_generate_tokens, &tokenizer_json, dialect)
            .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
            .with_suppress_thinking(true)
            .with_print_outputs(false)
            .with_timeout_secs(1200); // 8 minutes for this test (Qwen3-8B is large)

        println!("\n=== Loading Model ===\n");

        let repo = api.repo(hf_hub::Repo::with_revision(
            "unsloth/Qwen3-8B-GGUF".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let model_path = repo.get("Qwen3-8B-Q6_K.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "CUDA required for this test: {}. Use --features cuda",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        let configs = vec![
            // Sequential (non-batched) test using BF16 for better performance on tensor cores
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: false,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            // Batched tests with F16 and BF16 - single repeat for speed
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 16,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 32,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 32,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 128,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 10,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        // Create a logits processor for sampling
        // Use BatchedInference wrapper type
        use crate::models::batched_model::BatchedInference;

        // Inference numeric mode, selected by INT8MODE (default Performance; "off"/"prec").
        let int8mode = match std::env::var("INT8MODE").ok().as_deref() {
            Some("off") => Int8Mode::Off,
            Some("prec") | Some("precision") => Int8Mode::Precision,
            _ => Int8Mode::Performance,
        };
        println!("int8 mode = {int8mode:?}\n");

        // Load the model wrapped in BatchedInference with proper inv_freq
        let load_model = || {
            let model = ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?;
            println!("✓ Model loaded\n");
            let inv_freq = model
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    #[test]
    #[ignore] // cargo test --release --features cuda --lib -p candle-transformers quantized_qwen3::tests::test_ruler_eval -- --ignored --nocapture
    fn test_ruler_eval() -> Result<()> {
        use crate::models::batch_test::ruler_gen::{
            run_ruler_benchmark, RulerBenchConfig, RulerDataSource, RulerTask, QWEN3_EOS_IDS,
        };
        use crate::models::batch_test::test_helpers::{
            download_hf_gguf, load_hf_tokenizer, open_gguf,
        };
        use crate::models::batched_inference::InferenceMode;
        use crate::models::batched_model::BatchedInference;

        #[cfg(not(feature = "cuda"))]
        println!("⚠  No CUDA — performance will be poor");

        // Change GGUF_VARIANT to swap model weight quantisation (Q4_0 < Q4_K_M by ~240 MB).
        const GGUF_VARIANT: &str = "Q6_K"; // options: Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0

        let tokenizer = load_hf_tokenizer("Qwen/Qwen3-8B")?;
        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA device: {e}")))?;
        let gguf_filename = format!("Qwen3-8B-{GGUF_VARIANT}.gguf");
        let model_path = download_hf_gguf("unsloth/Qwen3-8B-GGUF", &gguf_filename, "main")?;
        println!("Model path: {model_path:?}");
        let (content, mut file) = open_gguf(&model_path)?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)?;
        println!("✓ Model loaded");
        let inv_freq = weights
            .rope_inv_freq()
            .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
        let model = BatchedInference::new_with_inv_freq(weights, inv_freq, 98_304, &device)?;

        // Phase A: 4K context, high concurrency (budget = 24×4096 = 98304).
        // Target batches: Q8→24, C5→~64, C8→~115, C9/C10→higher.
        // After results lock in at 4K, bump to: 8K(12), 16K(6), 32K(3).
        let model_name = format!("Qwen3-8B-{GGUF_VARIANT}");
        let cfg = RulerBenchConfig {
            model_name: &model_name,
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 98_304,
            max_gen_tokens: 50,
            modes: &[
                (Some(InferenceMode::Q8_0), "Q8_0"),
                (Some(InferenceMode::Q8_Q4), "Q8/Q4"),
                (Some(InferenceMode::Q4_0), "Q4_0"),
                (Some(InferenceMode::C5), "C5"),
                (Some(InferenceMode::C8), "C8"),
                (Some(InferenceMode::C9), "C9"),
                (Some(InferenceMode::C10), "C10"),
            ],
            // Per-length sample counts sized for ~6 min each (÷2 per 2× context).
            // Adjust after first run if actual timings differ.
            lengths_samples: &[(4_096, 8), (8_192, 4), (16_384, 2), (32_768, 1)],
            tasks: &[
                RulerTask::NiahSingle1,
                RulerTask::NiahMultiKey2,
                RulerTask::Vt,
                RulerTask::Cwe,
            ],
            data_source: RulerDataSource::Generated,
        };

        run_ruler_benchmark(&model, &tokenizer, &cfg)
    }
}
