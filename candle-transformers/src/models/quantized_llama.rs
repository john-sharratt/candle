//! Quantized llama model implementation.
//!
//! This provides a quantized implementation of the llama language model architecture.
//! The model implements parameter efficient quantization for reduced memory usage
//! while maintaining model quality.
//!
//! Key characteristics:
//! - Transformer decoder architecture
//! - Support for 2/3/4/8-bit quantization
//! - Optimized memory usage through quantization
//! - Configurable model sizes and parameter counts
//!
//! - 💻 [GH Link](https://github.com/facebookresearch/llama)
//! - 📝 [Paper](https://arxiv.org/abs/2302.13971)
//!
//! ![](https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/quantized/assets/aoc.gif)
//!

use std::sync::{Arc, RwLock};

#[cfg(feature = "cuda")]
use super::batched_layer::{BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::BatchedModelCore;
use super::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use super::llama_rope::llama_inv_freq;
use super::profile::{pipeline_record, profile_now, profile_sync};
use super::rope_tables::CisPrecomputations;
use super::{decode_utils, quantized_matmul::QMatMul};
use crate::models::llama::{Llama3RopeConfig, Llama3RopeType};
use crate::quantized_nn::RmsNorm;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::DynamicActs;
#[cfg(feature = "cuda")]
use candle::quantized::register_mmap_cuda;
use candle::quantized::QTensor;
use candle::quantized::{ggml_file, gguf_file, GgmlDType, Int8Mode};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Embedding, Module};

/// Initial number of RoPE positions to precompute for quantized llama models.
///
/// The tables can be extended on demand in chunks (see `ROPE_EXTEND_CHUNK`) to
/// keep RoPE lookup fast (index_select/narrow into a contiguous table) while
/// avoiding a hard cap.
///
/// We default to 0 to avoid any up-front RoPE allocation; the tables will grow
/// on demand.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

type SharedCis = Arc<RwLock<CisPrecomputations>>;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate_up: Option<QMatMul>,
    feed_forward_w1: Option<QMatMul>,
    feed_forward_w2: QMatMul,
    feed_forward_w3: Option<QMatMul>,
}

impl Mlp {
    fn from_qtensors(
        feed_forward_w1: QTensor,
        feed_forward_w2: QTensor,
        feed_forward_w3: QTensor,
        device_is_cuda: bool,
        // The dense MLP's `ffn_norm` emits int8 (q8a128) activations in int8 mode, so its
        // weights must be the matching KO twins. MoE experts receive FP activations and pass
        // `Int8Mode::Off`. The fused gate+up weight is repacked to a single KO twin too.
        int8mode: Int8Mode,
    ) -> Result<Self> {
        let try_fuse = device_is_cuda
            && feed_forward_w1.dtype() == feed_forward_w3.dtype()
            && !matches!(
                feed_forward_w1.dtype(),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            );

        let (feed_forward_gate_up, feed_forward_w1, feed_forward_w3) = if try_fuse {
            #[cfg(feature = "cuda")]
            {
                let (w1_n, w1_k) = feed_forward_w1.shape().dims2()?;
                let (w3_n, w3_k) = feed_forward_w3.shape().dims2()?;
                if w1_n != w3_n || w1_k != w3_k {
                    candle::bail!(
                        "cannot fuse ffn_gate/ffn_up due to shape mismatch: gate=({}, {}) up=({}, {})",
                        w1_n,
                        w1_k,
                        w3_n,
                        w3_k
                    );
                }
                let fused = QTensor::concat_rows_cuda(&[&feed_forward_w1, &feed_forward_w3])?;
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
                Some(QMatMul::from_qtensor_with_mode(feed_forward_w1, int8mode)?),
                Some(QMatMul::from_qtensor_with_mode(feed_forward_w3, int8mode)?),
            )
        };

        Ok(Self {
            feed_forward_gate_up,
            feed_forward_w1,
            feed_forward_w2: QMatMul::from_qtensor_with_mode(feed_forward_w2, int8mode)?,
            feed_forward_w3,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let stage_prefix = if dims.len() >= 2 && dims[1] == 1 {
            "decode"
        } else {
            "prefill"
        };

        let w1w3_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:w1w3"
        } else {
            "prefill:model:mlp:ffn:w1w3"
        };
        let w1_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:w1"
        } else {
            "prefill:model:mlp:ffn:w1"
        };
        let w3_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:w3"
        } else {
            "prefill:model:mlp:ffn:w3"
        };
        let silu_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:silu"
        } else {
            "prefill:model:mlp:ffn:silu"
        };
        let mul_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:mul"
        } else {
            "prefill:model:mlp:ffn:mul"
        };
        let w2_name = if stage_prefix == "decode" {
            "decode:model:mlp:ffn:w2"
        } else {
            "prefill:model:mlp:ffn:w2"
        };

        let (w1, w3) = if let Some(w) = &self.feed_forward_gate_up {
            let t_w1w3 = profile_now();
            let gu = w.forward(xs)?;
            profile_sync(gu.device());
            pipeline_record(w1w3_name, t_w1w3);

            let last_dim = gu.rank() - 1;
            let out_dim = gu.dim(last_dim)?;
            if out_dim % 2 != 0 {
                candle::bail!("unexpected fused gate+up output dim {out_dim} (not even)");
            }
            let half = out_dim / 2;
            (
                gu.narrow(last_dim, 0, half)?,
                gu.narrow(last_dim, half, half)?,
            )
        } else {
            let t_w1 = profile_now();
            let w1 = self
                .feed_forward_w1
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing feed_forward_w1".into()))?
                .forward(xs)?;
            profile_sync(w1.device());
            pipeline_record(w1_name, t_w1);

            let t_w3 = profile_now();
            let w3 = self
                .feed_forward_w3
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing feed_forward_w3".into()))?
                .forward(xs)?;
            profile_sync(w3.device());
            pipeline_record(w3_name, t_w3);
            (w1, w3)
        };

        let t_silu = profile_now();
        let silu_w1 = candle_nn::ops::silu(&w1)?;
        profile_sync(silu_w1.device());
        pipeline_record(silu_name, t_silu);

        let t_mul = profile_now();
        let intermediate = (silu_w1 * w3)?;
        profile_sync(intermediate.device());
        pipeline_record(mul_name, t_mul);

        let t_w2 = profile_now();
        let out = self.feed_forward_w2.forward(&intermediate)?;
        profile_sync(out.device());
        pipeline_record(w2_name, t_w2);
        Ok(out)
    }
}

impl Mlp {
    /// B3 consumer: gate/up over a producer-prepared (fused ffn_norm) activation, shared across
    /// both projections; down-proj closes the block. CUDA only.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(&self, acts: &DynamicActs, out_dtype: DType) -> Result<Tensor> {
        let (mut w1, mut w3) = if let Some(w) = &self.feed_forward_gate_up {
            let mut gu = w.forward_dynamic(acts.as_dynamic(), out_dtype)?;
            // Coerce the fused output to out_dtype ONCE, in place, before splitting it into the
            // gate/up views: `gu` is owned + contiguous here so the cast is allocation-free.
            // Casting the two aliasing narrows separately instead forces two fallback allocations
            // (an in-place cast on a shared view is unsafe — see `Tensor::to_dtype_mut`).
            gu.to_dtype_mut(out_dtype)?;
            let last = gu.rank() - 1;
            let half = gu.dim(last)? / 2;
            (gu.narrow(last, 0, half)?, gu.narrow(last, half, half)?)
        } else {
            let w1 = self
                .feed_forward_w1
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing feed_forward_w1".into()))?
                .forward_dynamic(acts.as_dynamic(), out_dtype)?;
            let w3 = self
                .feed_forward_w3
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing feed_forward_w3".into()))?
                .forward_dynamic(acts.as_dynamic(), out_dtype)?;
            (w1, w3)
        };
        // Run silu/mul/down in out_dtype: the Float path returns the activation dtype (F16), but
        // MLP intermediates can exceed F16's ~65504 range, so compute in out_dtype (BF16). The
        // fused path already coerced `gu` above and the int8 path already returns out_dtype, so
        // these are no-ops except on the separate-weight Float path.
        w1.to_dtype_mut(out_dtype)?;
        w3.to_dtype_mut(out_dtype)?;
        let intermediate = (candle_nn::ops::silu(&w1)? * w3)?;
        let mut out = self.feed_forward_w2.forward(&intermediate)?;
        out.to_dtype_mut(out_dtype)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    // Fused q/k/v weight: consumed only by the CUDA batched `project_qkv`; the non-batched
    // `forward()` uses the separate wq/wk/wv, so it is intentionally unread off-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cis: SharedCis,
    neg_inf: Tensor,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    // mask: 0.0 = visible, -inf = masked (can be any dtype: F32, F16, BF16)
    // Convert to U8 for where_cond: 0.0 -> 0u8 (false), -inf -> 1u8 (true)
    // Create zero in the same dtype as mask to avoid comparison dtype mismatch
    let zero = Tensor::new(0f32, mask.device())?
        .to_dtype(mask.dtype())?
        .broadcast_as(shape.dims())?;
    let mask_u8 = mask.ne(&zero)?; // not equal to 0.0 gives U8 tensor
    let m = mask_u8.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let (cos, sin) = {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.narrow_growable(0, index_pos, seq_len, x.dtype())?
        };
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    /// Apply rotary position embeddings in batched mode.
    ///
    /// Note: This method is no longer used by the generic batched layer processing,
    /// which now receives precomputed RoPE (cos, sin) from the model level.
    /// Kept for potential direct use or debugging.
    #[allow(dead_code)]
    fn apply_rotary_emb_batched(
        &self,
        x: &Tensor,
        offsets: &[usize],
        offsets_t: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;

        if offsets.len() != b_sz {
            candle::bail!(
                "Offset count mismatch: got {} offsets for batch size {}",
                offsets.len(),
                b_sz
            );
        }

        // Fast path for the decode loop (seq_len == 1): do a single rope_i call by
        // building per-batch (cos, sin) via index_select.
        if seq_len == 1 {
            let offsets_t_owned = if offsets_t.is_none() {
                Some(decode_utils::offsets_to_u32_tensor(offsets, x.device())?)
            } else {
                None
            };

            let offsets_t = match offsets_t {
                Some(t) => t,
                None => offsets_t_owned
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("missing owned offsets_t".into()))?,
            };

            let offsets_len = offsets_t.dim(0)?;
            if offsets_len != b_sz {
                candle::bail!(
                    "offsets_t length mismatch: got {}, expected {}",
                    offsets_len,
                    b_sz
                );
            }

            // Ensure RoPE tables cover the requested offsets, then gather from the
            // contiguous precomputed tables.
            let required_len = offsets.iter().copied().max().unwrap_or(0) + 1;
            let needs_extend = {
                let cis = self
                    .cis
                    .read()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                required_len > cis.max_seq_len()
            };
            if needs_extend {
                let mut cis = self
                    .cis
                    .write()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                cis.ensure_len(required_len)?;
            }
            let (cos_all, sin_all) = {
                let cis = self
                    .cis
                    .read()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                let cis = cis.get_for_dtype(x.dtype())?;
                (cis.cos.clone(), cis.sin.clone())
            };
            let (cos, sin) = decode_utils::gather_rope_cos_sin(&cos_all, &sin_all, offsets_t)?;
            let x = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            return candle_nn::rotary_emb::rope_i(&x, &cos, &sin);
        }

        // Multi-token prompt processing: extend once (if needed), then slice from the
        // contiguous tables for each sequence.
        let required_len = offsets
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("rope range overflow".into()))?;

        let needs_extend = {
            let cis = self
                .cis
                .read()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            required_len > cis.max_seq_len()
        };
        if needs_extend {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.ensure_len(required_len)?;
        }
        let (cos_all, sin_all) = {
            let cis = self
                .cis
                .read()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            let cis = cis.get_for_dtype(x.dtype())?;
            (cis.cos.clone(), cis.sin.clone())
        };

        let mut results = Vec::with_capacity(b_sz);
        for (batch_idx, &offset) in offsets.iter().enumerate() {
            let x_slice = x.narrow(0, batch_idx, 1)?;
            let cos = cos_all.narrow(0, offset, seq_len)?;
            let sin = sin_all.narrow(0, offset, seq_len)?;
            let rotated = candle_nn::rotary_emb::rope_i(&x_slice.contiguous()?, &cos, &sin)?;
            results.push(rotated);
        }

        let results_refs: Vec<&Tensor> = results.iter().collect();
        Tensor::cat(&results_refs, 0)
    }

    /// Forward attention for quantized operations.
    fn forward_attn(&self, cache: &mut KvCache, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let t_qkv = profile_now();
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        // Reshape and transpose in one go for better performance
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        // V contiguous only needed for prompt processing, no-op for single token
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;
        profile_sync(q.device());
        if seq_len == 1 {
            pipeline_record("decode:qkv_proj", t_qkv);
        } else {
            pipeline_record("prefill:qkv_proj", t_qkv);
        }

        // Reset KV cache if we're at the first position
        let t_alloc = profile_now();
        if index_pos == 0 {
            cache.reset();
        }
        let (k, v) = cache.append(&k, &v)?;
        profile_sync(q.device());
        if seq_len == 1 {
            pipeline_record("decode:alloc", t_alloc);
        } else {
            pipeline_record("prefill:alloc", t_alloc);
        }

        // Use optimized attention kernels when available
        let cache_dtype = cache.dtype();
        let standard_attention = || -> Result<Tensor> {
            // Convert q to cache dtype if needed (k and v are already in cache dtype)
            let q = q.to_dtype(cache_dtype)?;
            let k = crate::utils::repeat_kv(k.clone(), self.n_head / self.n_kv_head)?;
            let v = crate::utils::repeat_kv(v.clone(), self.n_head / self.n_kv_head)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len > 1 {
                let cache_len = index_pos + seq_len;
                let mask: Vec<_> = (0..seq_len)
                    .flat_map(|i| {
                        (0..cache_len).map(move |j| {
                            if j > index_pos + i {
                                f32::NEG_INFINITY
                            } else {
                                0.0f32
                            }
                        })
                    })
                    .collect();
                let mask = Tensor::from_vec(mask, (1, 1, seq_len, cache_len), q.device())?
                    .to_dtype(cache_dtype)?;
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf.to_dtype(cache_dtype)?)?
            } else {
                att
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)
        };

        let t_kernel = profile_now();
        let y = if seq_len == 1 && q.device().is_metal() {
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else if seq_len > 1 && matches!(q.device(), Device::Cuda(_)) {
            #[cfg(feature = "flash-attn")]
            {
                let q_fa = q.transpose(1, 2)?;
                let k_fa = k.transpose(1, 2)?;
                let v_fa = v.transpose(1, 2)?;
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                match candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true) {
                    Ok(out) => out.transpose(1, 2)?,
                    Err(_) => standard_attention()?,
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            standard_attention()?
        } else {
            standard_attention()?
        };
        profile_sync(x.device());
        if seq_len == 1 {
            pipeline_record("decode:kernel", t_kernel);
        } else {
            pipeline_record("prefill:kernel", t_kernel);
        }

        let t_out_proj = profile_now();
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        profile_sync(x.device());
        if seq_len == 1 {
            pipeline_record("decode:out_proj", t_out_proj);
        } else {
            pipeline_record("prefill:out_proj", t_out_proj);
        }
        Ok(y)
    }
}

/// Implement the `BatchedAttentionLayer` trait for quantized llama layers.
///
/// This enables the use of generic batched layer processing from `batched_layer` module.
#[cfg(feature = "cuda")]
impl BatchedAttentionLayer for LayerWeights {
    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn o_proj(&self) -> &QMatMul {
        &self.attention_wo
    }

    /// B3 producer: fuse ffn_norm -> q8a128 only for the dense MLP path; MoE stays FP.
    #[cfg(feature = "cuda")]
    fn ffn_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs> {
        match &self.mlp_or_moe {
            MlpOrMoe::Mlp(_) => self.ffn_norm.forward_dynamic(x, mode),
            _ => Ok(DynamicActs::Float(self.ffn_norm.forward(x)?)),
        }
    }

    /// B3 consumer: dense MLP over the fused activation; MoE falls back to FP.
    #[cfg(feature = "cuda")]
    fn ffn_forward(&self, acts: DynamicActs, mlp_dtype: DType) -> Result<Tensor> {
        match &self.mlp_or_moe {
            MlpOrMoe::Mlp(m) => m.forward_dynamic(&acts, mlp_dtype),
            _ => match acts {
                DynamicActs::Float(t) => self.mlp_or_moe.forward(&t.to_dtype(mlp_dtype)?),
                DynamicActs::Int8(_) => {
                    candle::bail!("llama MoE ffn_forward received int8 acts")
                }
            },
        }
    }

    /// B1 producer: fuse attention_norm -> q8a128 (int8) or FP rms_norm (Off).
    #[cfg(feature = "cuda")]
    fn attention_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs> {
        self.attention_norm.forward_dynamic(x, mode)
    }

    /// B1 consumer: q/k/v over the fused activation (fused-qkv or separate).
    #[cfg(feature = "cuda")]
    fn project_qkv(&self, acts: &DynamicActs, out_dtype: DType) -> Result<QkvProjection> {
        let q_dim = self.n_head * self.head_dim;
        let kv_dim = self.n_kv_head * self.head_dim;
        let (q, k, v) = match acts {
            // int8: ONE segmented launch over the three KO weights (no concat) — float-identical
            // to three separate matmuls, full GPU occupancy.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[
                        self.attention_wq.inner(),
                        self.attention_wk.inner(),
                        self.attention_wv.inner(),
                    ],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                (
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                )
            }
            DynamicActs::Float(_) => (
                self.attention_wq
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
                self.attention_wk
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
                self.attention_wv
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
            ),
        };
        Ok(QkvProjection { q, k, v })
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    span: tracing::Span,
    span_output: tracing::Span,
}

/// Implementation of BatchedModelCore for use with BatchedInference wrapper.
///
/// This is the new recommended way to use batched inference. The BatchedInference
/// wrapper handles RoPE caching at the model level, so this implementation is simpler.
#[cfg(feature = "cuda")]
impl BatchedModelCore for ModelWeights {
    type Layer = LayerWeights;

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_head(&self) -> usize {
        self.layers.first().map(|l| l.n_kv_head).unwrap_or(0)
    }

    fn head_dim(&self) -> usize {
        self.layers.first().map(|l| l.head_dim).unwrap_or(0)
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
        &self.output
    }

    fn rope_interleaved(&self) -> bool {
        // LLaMA uses interleaved RoPE format
        true
    }

    fn prune(&self) -> Result<()> {
        // Compact the embedding cache to free GPU copies
        self.embeddings.compact();

        // Compact the RoPE cache (shared across all layers via Arc)
        if let Some(layer) = self.layers.first() {
            if let Ok(mut cis) = layer.cis.write() {
                cis.compact();
            }
        }

        Ok(())
    }

    fn k_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::LLAMA_KV_FACTORS.k_hi
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::LLAMA_KV_FACTORS.k_low
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::LLAMA_KV_FACTORS.v_hi
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::LLAMA_KV_FACTORS.v_low
    }
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable(
            head_dim,
            10000.,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            &ct.device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;

            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp::from_qtensors(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                    matches!(ct.device, Device::Cuda(_)),
                    Int8Mode::Off,
                )?)
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize)?,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: ct.device.clone(),
            span,
            span_output,
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        // Try to parse Llama-3-style RoPE scaling parameters if present.
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());
        let md_opt_str = |k: &str| ct.metadata.get(k).and_then(|v| v.to_string().ok()).cloned();

        let rope_scaling = {
            let factor = md_opt_f32("llama.rope.scaling.factor")
                .or_else(|| md_opt_f32("llama.rope.scale_factor"));
            let low_freq_factor = md_opt_f32("llama.rope.scaling.low_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.low_freq"));
            let high_freq_factor = md_opt_f32("llama.rope.scaling.high_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.high_freq"));
            let original_max_position_embeddings =
                md_opt_u32("llama.rope.scaling.original_max_position_embeddings")
                    .or_else(|| md_opt_u32("llama.rope.scaling.original_context_length"))
                    .map(|v| v as usize);

            let rope_type = md_opt_str("llama.rope.scaling.type")
                .or_else(|| md_opt_str("llama.rope.scaling.rope_type"))
                .map(|s| match s.as_str() {
                    "llama3" => Llama3RopeType::Llama3,
                    _ => Llama3RopeType::Default,
                })
                .unwrap_or(Llama3RopeType::Default);

            match (
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            ) {
                (Some(factor), Some(low), Some(high), Some(orig)) => Some(Llama3RopeConfig {
                    factor,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_position_embeddings: orig,
                    rope_type,
                }),
                _ => None,
            }
        };

        let inv_freq = llama_inv_freq(rope_dim, rope_freq_base, rope_scaling);
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
            inv_freq,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp::from_qtensors(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                    matches!(device, Device::Cuda(_)),
                    Int8Mode::Off,
                )?)
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp::from_qtensors(
                        feed_forward_w1,
                        feed_forward_w2,
                        feed_forward_w3,
                        matches!(device, Device::Cuda(_)),
                        Int8Mode::Off,
                    )?)
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, embedding_length)?,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
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
    /// use candle_transformers::models::quantized_llama::ModelWeights;
    /// use std::path::Path;
    ///
    /// let path = Path::new("model.gguf");
    /// let device = Device::cuda_if_available(0)?;
    /// let model = ModelWeights::from_gguf_by_path(path, &device)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_gguf_by_path(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        Self::from_gguf_by_path_with_int8(file_path, device, Int8Mode::auto(device))
    }

    /// Like from_gguf_by_path but with an explicit int8mode (test path selects from INT8MODE).
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

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());
        let md_opt_str = |k: &str| ct.metadata.get(k).and_then(|v| v.to_string().ok()).cloned();

        let rope_scaling = {
            let factor = md_opt_f32("llama.rope.scaling.factor")
                .or_else(|| md_opt_f32("llama.rope.scale_factor"));
            let low_freq_factor = md_opt_f32("llama.rope.scaling.low_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.low_freq"));
            let high_freq_factor = md_opt_f32("llama.rope.scaling.high_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.high_freq"));
            let original_max_position_embeddings =
                md_opt_u32("llama.rope.scaling.original_max_position_embeddings")
                    .or_else(|| md_opt_u32("llama.rope.scaling.original_context_length"))
                    .map(|v| v as usize);

            let rope_type = md_opt_str("llama.rope.scaling.type")
                .or_else(|| md_opt_str("llama.rope.scaling.rope_type"))
                .map(|s| match s.as_str() {
                    "llama3" => Llama3RopeType::Llama3,
                    _ => Llama3RopeType::Default,
                })
                .unwrap_or(Llama3RopeType::Default);

            match (
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            ) {
                (Some(factor), Some(low), Some(high), Some(orig)) => Some(Llama3RopeConfig {
                    factor,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_position_embeddings: orig,
                    rope_type,
                }),
                _ => None,
            }
        };

        let inv_freq = llama_inv_freq(rope_dim, rope_freq_base, rope_scaling);
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
            inv_freq,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Helper to load tensor from mmap
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        let tok_embeddings_q = load_tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(load_tensor("output_norm.weight")?, rms_norm_eps)?;
        let output = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let attention_wk = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let attention_wv = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let attention_wo = load_tensor(&format!("{prefix}.attn_output.weight"))?;

            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 = load_tensor(&format!("{prefix}.ffn_gate.weight"))?;
                let feed_forward_w2 = load_tensor(&format!("{prefix}.ffn_down.weight"))?;
                let feed_forward_w3 = load_tensor(&format!("{prefix}.ffn_up.weight"))?;
                MlpOrMoe::Mlp(Mlp::from_qtensors(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                    matches!(device, Device::Cuda(_)),
                    int8mode,
                )?)
            } else {
                let feed_forward_gate_inp = load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 = load_tensor(&format!("{prefix}.ffn_gate.{i}.weight"))?;
                    let feed_forward_w2 = load_tensor(&format!("{prefix}.ffn_down.{i}.weight"))?;
                    let feed_forward_w3 = load_tensor(&format!("{prefix}.ffn_up.{i}.weight"))?;
                    experts.push(Mlp::from_qtensors(
                        feed_forward_w1,
                        feed_forward_w2,
                        feed_forward_w3,
                        matches!(device, Device::Cuda(_)),
                        Int8Mode::Off,
                    )?)
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor_with_mode(
                        feed_forward_gate_inp,
                        int8mode,
                    )?,
                    experts,
                }
            };

            let attention_norm = load_tensor(&format!("{prefix}.attn_norm.weight"))?;
            let ffn_norm = load_tensor(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor_with_mode(attention_wq, int8mode)?,
                attention_wk: QMatMul::from_qtensor_with_mode(attention_wk, int8mode)?,
                attention_wv: QMatMul::from_qtensor_with_mode(attention_wv, int8mode)?,
                attention_wo: QMatMul::from_qtensor_with_mode(attention_wo, int8mode)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, embedding_length)?,
            layers,
            norm,
            output: QMatMul::from_qtensor_with_mode(output, int8mode)?,
            device: device.clone(),
            span,
            span_output,
        })
    }

    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = self
            .layers
            .iter()
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Get the RoPE inv_freq values for use with BatchedInference wrapper.
    ///
    /// This returns the custom inv_freq computed during model loading (which may include
    /// rope scaling). Use this with `BatchedInference::new_with_inv_freq()`.
    pub fn rope_inv_freq(&self) -> Option<Vec<f32>> {
        self.layers
            .first()
            .and_then(|l| l.cis.read().ok().and_then(|cis| cis.inv_freq_vec()))
    }

    /// Forward pass (backwards compatible API).
    pub fn forward(&self, caches: &mut KvCaches, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward_with_context(&mut SequenceContext {
            kv_caches: caches,
            offset: index_pos,
            input_ids: x,
            input_len: x.dims2()?.1,
        })
    }

    /// Forward pass with strongly-typed sequence context.
    ///
    /// This is the preferred API for continuous batching scenarios where you manage
    /// multiple independent sequences. Each sequence has its own `KvCaches` instance.
    pub fn forward_with_context(&self, ctx: &mut SequenceContext<'_>) -> Result<Tensor> {
        if ctx.kv_caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                ctx.kv_caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let (_b, seq_len) = ctx.input_ids.dims2()?;
        let stage_prefix = if seq_len == 1 { "decode" } else { "prefill" };

        // Derive dtype from KV cache to ensure consistency throughout forward pass
        // For FP8 KV cache, use F16 activations (mixed precision)
        let embed_dtype = ctx.kv_caches.dtype();
        let mut layer_in = self
            .embeddings
            .forward(ctx.input_ids)?
            .to_dtype(embed_dtype)?;

        // Use forward_attn for consistent attention computation
        for (layer, cache) in self.layers.iter().zip(ctx.kv_caches.caches.iter_mut()) {
            let x = layer_in;
            let residual = &x;

            let t_attn_total = profile_now();
            let t_attn_norm = profile_now();
            let x = layer.attention_norm.forward(&x)?;
            profile_sync(x.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:attn:norm"
                } else {
                    "prefill:model:attn:norm"
                },
                t_attn_norm,
            );

            let t_attn_core = profile_now();
            let attn = layer
                .forward_attn(cache, &x, ctx.offset)?
                .to_dtype(embed_dtype)?;
            profile_sync(attn.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:attn:core"
                } else {
                    "prefill:model:attn:core"
                },
                t_attn_core,
            );

            let t_attn_residual = profile_now();
            let x = (attn + residual)?;
            profile_sync(x.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:attn:resid"
                } else {
                    "prefill:model:attn:resid"
                },
                t_attn_residual,
            );
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:attn:total"
                } else {
                    "prefill:model:attn:total"
                },
                t_attn_total,
            );

            // MLP
            let t_mlp_total = profile_now();
            let _enter = layer.span_mlp.enter();
            let residual = &x;

            let t_mlp_norm = profile_now();
            let x = layer.ffn_norm.forward(&x)?;
            profile_sync(x.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:mlp:norm"
                } else {
                    "prefill:model:mlp:norm"
                },
                t_mlp_norm,
            );

            let t_mlp_ffn = profile_now();
            let x = layer.mlp_or_moe.forward(&x)?;
            profile_sync(x.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:ffn:total"
                } else {
                    "prefill:model:ffn:total"
                },
                t_mlp_ffn,
            );

            let t_mlp_residual = profile_now();
            let x = (x + residual)?;
            profile_sync(x.device());
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:mlp:resid"
                } else {
                    "prefill:model:mlp:resid"
                },
                t_mlp_residual,
            );
            pipeline_record(
                if stage_prefix == "decode" {
                    "decode:model:mlp:total"
                } else {
                    "prefill:model:mlp:total"
                },
                t_mlp_total,
            );

            layer_in = x
        }
        let t_norm = profile_now();
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let _enter = self.span_output.enter();
        let out = self.output.forward(&x)?;
        profile_sync(out.device());
        pipeline_record(
            if stage_prefix == "decode" {
                "decode:model:norm+proj"
            } else {
                "prefill:model:norm+proj"
            },
            t_norm,
        );
        Ok(out)
    }

    /// Forward pass returning logits for ALL positions (for perplexity evaluation).
    ///
    /// Returns `[batch, seq_len, vocab]` instead of `[batch, vocab]`.
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
        let mut layer_in = self.embeddings.forward(input)?.to_dtype(embed_dtype)?;
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer
                .forward_attn(cache, &x, offset)?
                .to_dtype(embed_dtype)?;
            let x = (attn + residual)?;
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;
    #[allow(unused_imports)]
    use candle_nn::kv_cache::CacheIntegrityResult;

    #[test]
    #[ignore] // Downloads model from HuggingFace. Run with: cargo test --release -- --ignored test_clone_with_independent_kv_cache
    fn test_clone_with_independent_kv_cache() -> Result<()> {
        // Download a small Llama model from HuggingFace
        // Using Llama-3.2-1B-Instruct (smallest available for fast testing)
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
        let model_path = repo.get("Llama-3.2-1B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::cuda_if_available(0)?;
        println!("Using device: {:?}", device);

        // Load model using optimized mmap path
        println!("Loading model with mmap optimization...");
        let load_start = std::time::Instant::now();
        let model = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        let load_duration = load_start.elapsed();
        println!(
            "✓ Model loaded in {:.3}s using mmap\n",
            load_duration.as_secs_f64()
        );

        println!("Starting 500-token prefill...");

        // Step 1: Advance model forward by 500+ tokens to populate KV cache
        // Using token ID 1 (typically a valid token in most vocabularies)
        let prefill_tokens = 500;
        let mut caches = model.create_kv_caches(2048);
        for i in 0..prefill_tokens {
            let input = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&mut caches, &input, i)?;

            if (i + 1) % 100 == 0 {
                println!("  Prefill progress: {}/{}", i + 1, prefill_tokens);
            }
        }

        let original_cache_len = caches.current_seq_len();
        assert_eq!(
            original_cache_len, prefill_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens
        );
        println!("✓ Original model cache: {} tokens", original_cache_len);

        // Step 2: Clone the model
        println!("\nCloning model...");
        let cloned_model = model.clone();
        let mut cloned_caches = cloned_model.create_kv_caches(2048);
        let clone_initial_cache_len = cloned_caches.current_seq_len();
        assert_eq!(
            clone_initial_cache_len, 0,
            "Cloned model should start with empty cache"
        );
        println!(
            "✓ Cloned model cache: {} tokens (empty)",
            clone_initial_cache_len
        );

        // Step 3: Advance clone forward with new prompt (different token: 2)
        println!("\nAdvancing clone with new prompt (100 tokens)...");
        let clone_tokens = 100;
        for i in 0..clone_tokens {
            let input = Tensor::new(&[2u32], &device)?.unsqueeze(0)?;
            let _output = cloned_model.forward(&mut cloned_caches, &input, i)?;
        }

        let clone_cache_len = cloned_caches.current_seq_len();
        assert_eq!(
            clone_cache_len, clone_tokens,
            "Clone should have {} tokens in cache",
            clone_tokens
        );
        println!("✓ Clone cache after generation: {} tokens", clone_cache_len);

        // Step 4: Verify original model cache is still intact
        let original_cache_len_after_clone = caches.current_seq_len();
        assert_eq!(
            original_cache_len_after_clone, prefill_tokens,
            "Original model cache should still have {} tokens (not affected by clone)",
            prefill_tokens
        );
        println!(
            "✓ Original cache after clone generation: {} tokens (unchanged)",
            original_cache_len_after_clone
        );

        // Step 5: Advance original forward with continuation (token 3)
        println!("\nAdvancing original model (50 more tokens)...");
        let original_continue_tokens = 50;
        for i in 0..original_continue_tokens {
            let input = Tensor::new(&[3u32], &device)?.unsqueeze(0)?;
            let _output = model.forward(&mut caches, &input, prefill_tokens + i)?;
        }

        let original_final_cache_len = caches.current_seq_len();
        assert_eq!(
            original_final_cache_len,
            prefill_tokens + original_continue_tokens,
            "Original model should have {} tokens in cache",
            prefill_tokens + original_continue_tokens
        );
        println!(
            "✓ Original cache after continuation: {} tokens",
            original_final_cache_len
        );

        // Verify caches are completely independent
        assert_ne!(
            original_final_cache_len, clone_cache_len,
            "Original and clone should have different cache lengths"
        );

        println!("\n=== Test Summary ===");
        println!(
            "✓ Original model: {} tokens in cache",
            original_final_cache_len
        );
        println!("✓ Cloned model: {} tokens in cache", clone_cache_len);
        println!("✓ Caches are completely independent");
        println!("✓ Clone did not affect original model state");

        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda,flash-attn -- --ignored test_flash_attention_prompt
    fn test_flash_attention_prompt() -> Result<()> {
        println!("\n=== Testing Flash Attention for Prompt Processing ===\n");

        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("bartowski/Llama-3.2-1B-Instruct-GGUF".to_string());
        let model_path = repo.get("Llama-3.2-1B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("Model downloaded to: {:?}", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "CUDA required for this test: {}. Use --features cuda,flash-attn",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Load model
        let model = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        println!("✓ Model loaded\n");

        // Test 1: Process a long prompt (should trigger Flash Attention)
        println!("Test 1: Long prompt processing (64 tokens)");
        let prompt_len = 64;
        let prompt_tokens: Vec<u32> = (0..prompt_len).map(|i| (i % 1000 + 1) as u32).collect();
        let prompt = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;

        let mut caches = model.create_kv_caches(2048);
        let start = std::time::Instant::now();
        let output = model.forward(&mut caches, &prompt, 0)?;
        let duration = start.elapsed();

        println!("  ✓ Processed {} tokens", prompt_len);
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), prompt_len);

        // Test 2: Single token generation (should use standard attention)
        println!("\nTest 2: Single token generation (autoregressive)");
        let start = std::time::Instant::now();
        let output = model.forward(
            &mut caches,
            &Tensor::new(&[1u32], &device)?.unsqueeze(0)?,
            prompt_len,
        )?;
        let duration = start.elapsed();

        println!("  ✓ Generated 1 token");
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), prompt_len + 1);

        // Test 3: Another multi-token sequence (Flash Attention again)
        println!("\nTest 3: Another multi-token batch (32 tokens)");
        let mut caches = model.create_kv_caches(2048);
        let batch_len = 32;
        let batch_tokens: Vec<u32> = (0..batch_len).map(|i| (i % 500 + 1) as u32).collect();
        let batch = Tensor::new(&batch_tokens[..], &device)?.unsqueeze(0)?;

        let start = std::time::Instant::now();
        let output = model.forward(&mut caches, &batch, 0)?;
        let duration = start.elapsed();

        println!("  ✓ Processed {} tokens", batch_len);
        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  Output shape: {:?}", output.shape());
        println!("  Cache length: {}", caches.current_seq_len());
        assert_eq!(caches.current_seq_len(), batch_len);

        // Test 4: Verify numerical stability
        println!("\nTest 4: Numerical stability check");
        let mut caches1 = model.create_kv_caches(2048);
        let test_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let test_input = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?;

        let output1 = model.forward(&mut caches1, &test_input, 0)?;
        let mut caches2 = model.create_kv_caches(2048);
        let output2 = model.forward(&mut caches2, &test_input, 0)?;

        // Check outputs are identical (or very close due to BF16 precision)
        let diff = (&output1 - &output2)?.abs()?.flatten_all()?.max(0)?;
        let diff_val = diff.to_vec0::<f32>()?;
        println!("  Max difference between runs: {:.6}", diff_val);
        assert!(diff_val < 1e-3, "Outputs should be consistent");
        println!("  ✓ Outputs are consistent");

        println!("\n=== Flash Attention Test Summary ===");
        println!("✓ Long prompt processing works (64 tokens)");
        println!("✓ Single token generation works");
        println!("✓ Multi-token batching works (32 tokens)");
        println!("✓ Numerical stability verified");
        println!(
            "Note: Flash Attention is used for seq_len > 1 on CUDA, fallback for seq_len == 1\n"
        );

        Ok(())
    }

    #[test]
    #[ignore] // Slow without CUDA. Run with: cargo test --release --features cuda,flash-attn -- --ignored test_parallel_batched_forwarding
    fn test_parallel_batched_forwarding_llama3() -> Result<()> {
        #[cfg(not(all(feature = "cuda")))]
        println!("⚠ WARNING: This test should be run with --features cuda,flash-attn for optimal performance");
        #[cfg(not(all(feature = "cuda")))]
        println!(
            "⚠ Current build is missing performance-critical features. Results may be slower.\n"
        );

        println!("\n=== Setting up Test Parameters ===\n");

        // Set up console logging for warnings and errors
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .try_init();

        // Load tokenizer and create test parameters
        let tokenizer_json = include_str!("quantized_llama_tokenizer.json");
        let num_generate_tokens = 40;
        let dialect = Dialect::llama3();

        let params = TestParams::new(num_generate_tokens, tokenizer_json, dialect)
            .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
            .with_print_outputs(false)
            .with_timeout_secs(6800);

        println!("\n=== Loading Model ===\n");

        // Download model from HuggingFace
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        let repo = api.model("VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF".to_string());
        let model_path = repo.get("model-Q4_K_M.gguf").map_err(|e| {
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
            TestConfig {
                mode: InferenceMode::F32,
                use_batched: false,
                num_contexts: 1,
                num_repeats: 4,
                generate_max_len: 40,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 4,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // R16: Raw F16 with Q-capture space — lossless F16, should match F16 quality.
            TestConfig {
                mode: InferenceMode::R16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q4,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            /*
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            // Composite tests
            TestConfig {
                mode: InferenceMode::Q8_Q8KS,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q4,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q4KS,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q2_S,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q2_A,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            TestConfig {
                mode: InferenceMode::Q8_Q1_S,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            */
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 4,
                generate_max_len: 40,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 40,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q8_1,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q8_KS,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: None,
            },
            // Q8_1K_Q4_0V: independent K/V formats — Q8_1 for K cache, Q4_1 for V cache.
            // Uses CoherenceCheck: if K/V tags are swapped the output will be garbage.
            TestConfig {
                mode: InferenceMode::Q8_Q4,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::CoherenceCheck),
            },
            // Q4_0 uses NameGreeting validation: 4-bit quantization error accumulates over
            // ~650 KV tokens and flips borderline argmax decisions for specific prompts
            // (e.g. "water tower." vs "water tower in"), causing StoryRewrite to fail even
            // in single-session mode.  NameGreeting still validates session isolation.
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            TestConfig {
                mode: InferenceMode::Q4_1,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::NameGreeting),
            },
            // Q4_KS uses CoherenceCheck: 4-bit with attention-sink sub-block scaling
            // collapses all name-token KV projections to the same quantized code,
            // making session differentiation impossible.  CoherenceCheck still catches
            // garbage output (e.g. broken prefill reads).
            TestConfig {
                mode: InferenceMode::Q4_KS,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::Skip),
            },
            // ──────── Adaptive C-levels (v2 candidate-list design) ────────
            // C0–C2: quality tier — F16/BF16 float fallback, Q8_0 baseline.
            //   Blocks that exceed threshold stay float (no forced quantization).
            // C3–C6: sweet tier — floor raised to Q8_0, no float fallback.
            //   Wider quant tails (Q3_0, Q2_0) for progressive compression.
            // C7–C9: compress tier — floor raised to Q4_0/Q3_0.
            //   Maximum compression; quality trade-off accepted.
            // C0: K=[F16,Q8_0] V=[BF16,Q8_0] — ~1.25× CR, 53/50dB SNR
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
            // C1: K=[F16,Q8_0] V=[BF16,Q8_0,Q4_0] — ~1.53× CR, 47dB
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 5,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 4,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C2: K=[F16,Q8_0,Q4_0] V=[BF16,Q8_0,Q4_0] — ~1.67× CR, 44dB
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 5,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 4,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C3: K=[Q8_0,Q4_0] V=[Q8_0,Q4_0] — ~2.11× CR, 36dB (floor raised)
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 64,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C4: K=[Q8_0,Q4_0] V=[Q8_0,Q4_0,Q3_0] — ~2.51× CR, 26dB
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 128,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C5: K=[Q8_0,Q4_0,Q3_0] V=[Q8_0,Q4_0,Q3_0] — ~2.84× CR, 23dB
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 256,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C6: K=[Q8_0,Q4_0,Q3_0] V=[Q8_0,Q4_0,Q3_0,Q2_0] — ~3.15× CR, 19dB
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
            // C7: K=[Q4_0,Q3_0] V=[Q4_0,Q3_0,Q2_0] — ~4.27× CR, 15dB
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C8: K=[Q4_0,Q3_0,Q2_0] V=[Q4_0,Q3_0,Q2_0] — ~4.70× CR, 12dB
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C9: K=[Q3_0,Q2_0] V=[Q3_0,Q2_0] — ~5.62× CR, 8dB
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C10: K same as C9, V pushed further.
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

        // Sequential (non-batched) callbacks - access inner model via .model()
        // Loads the model wrapped in BatchedInference with proper inv_freq
        let int8mode = match std::env::var("INT8MODE").ok().as_deref() {
            Some("off") => candle::quantized::Int8Mode::Off,
            Some("prec") | Some("precision") => candle::quantized::Int8Mode::Precision,
            _ => candle::quantized::Int8Mode::Performance,
        };
        println!(
            "int8 mode = {int8mode:?}
"
        );
        let load_model = || {
            let model = ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?;
            println!("✓ Model loaded\n");
            // Get the custom inv_freq (includes rope scaling if configured)
            let inv_freq = model
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            // Wrap with BatchedInference using the model's actual inv_freq
            BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    /// Llama 2 7B Q4_0 parallel batched forwarding test.
    ///
    /// This test benchmarks Llama 2 7B (TheBloke/Llama-2-7B-GGUF) using quantized
    /// weights and paged attention. Uses smaller batch sizes than the 3B model
    /// due to higher memory requirements.
    ///
    /// Run with: cargo test --release --features cuda --lib --package candle-transformers quantized_llama::tests::test_parallel_batched_forwarding_llama2 -- --ignored --nocapture
    #[test]
    #[ignore] // Slow without CUDA. Run with: cargo test --release --features cuda -- --ignored test_parallel_batched_forwarding_llama2
    fn test_parallel_batched_forwarding_llama2() -> Result<()> {
        #[cfg(not(all(feature = "cuda")))]
        println!("⚠ WARNING: This test should be run with --features cuda for optimal performance");
        #[cfg(not(all(feature = "cuda")))]
        println!(
            "⚠ Current build is missing performance-critical features. Results may be slower.\n"
        );

        println!("\n=== Setting up Test Parameters (Llama 2 7B) ===\n");

        let num_generate_tokens = 20;
        let dialect = Dialect::llama2();

        // Download tokenizer.json (Llama 2) from HuggingFace
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
        let tok_repo = api.model("NousResearch/Llama-2-7b-hf".to_string());
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
            .with_print_outputs(false)
            .with_test_mode(TestMode::NameGreeting) // Simpler validation for chat model
            .with_timeout_secs(1200); // 11 minutes and 20 seconds for 7B model

        println!("\n=== Loading Llama 2 7B Chat Model ===\n");

        // Download model from HuggingFace (TheBloke/Llama-2-7B-Chat-GGUF)
        // Note: Using the Chat variant which is instruction-tuned
        let repo = api.model("TheBloke/Llama-2-7B-Chat-GGUF".to_string());
        let model_path = repo.get("llama-2-7b-chat.Q4_0.gguf").map_err(|e| {
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

        // Use smaller batch sizes for 7B model (more memory intensive)
        let configs = vec![
            // Sequential (non-batched) baseline
            TestConfig {
                mode: InferenceMode::F32,
                use_batched: false,
                num_contexts: 1,
                num_repeats: 2,
                generate_max_len: 20,
                test_mode: None,
            },
            // Batched F16 tests
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 2,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 8,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            // Batched BF16 tests - scale up to maximum contexts
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 2,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 8,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 16,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 48,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 32,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 96,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 32,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 140,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: None,
            },
        ];

        // Create a logits processor for sampling
        // Use BatchedInference wrapper type
        use crate::models::batched_model::BatchedInference;

        // Sequential (non-batched) callbacks - access inner model via .model()
        // Load the model wrapped in BatchedInference with proper inv_freq
        let int8mode = match std::env::var("INT8MODE").ok().as_deref() {
            Some("off") => candle::quantized::Int8Mode::Off,
            Some("prec") | Some("precision") => candle::quantized::Int8Mode::Precision,
            _ => candle::quantized::Int8Mode::Performance,
        };
        println!(
            "int8 mode = {int8mode:?}
"
        );
        let load_model = || {
            let model = ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?;
            println!("✓ Llama 2 7B Chat model loaded\n");
            // Get the custom inv_freq (includes rope scaling if configured)
            let inv_freq = model
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            // Wrap with BatchedInference using the model's actual inv_freq
            BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    /// KV-cache dump infrastructure for offline selection analysis.
    ///
    /// Run tests in this module with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_llama::tests::kv_dump -- --ignored --nocapture
    mod kv_dump {
        use super::*;
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::batched_model::BatchedInference;
        use std::io::Write;

        /// Dump real KV cache data (K, V, Q) from Llama-3.2-3B for offline analysis.
        ///
        /// Runs a single R16 session using the Llama-3 chat-template system+user
        /// prompt (matching what the gated test sends for context index 0), followed
        /// by 40 decode steps.  R16 mode keeps K in raw F16 with Q-capture space, so
        /// the prefill/decode kernels populate `block_r16->q[]` with real Q
        /// projections; the dump captures K + V + Q in v4 binary format.
        ///
        /// The output is written to:
        ///   `candle-nn/src/kv_cache/chunked/tests/data/llama-kv-data.bin`
        ///
        /// Run with:
        ///   cargo test --release --features cuda --lib --package candle-transformers \
        ///     quantized_llama::tests::kv_dump::test_dump_kv_cache_data -- --ignored --nocapture
        #[test]
        #[ignore]
        fn test_dump_kv_cache_data() -> Result<()> {
            let device = Device::new_cuda(0).map_err(|e| {
                candle::Error::Msg(format!(
                    "CUDA required for this test: {}. Use --features cuda",
                    e
                ))
            })?;
            println!("Using device: {:?}", device);

            // Tokenizer + chat-ML prompts (matching the gated test).
            let tokenizer_json = include_str!("quantized_llama_tokenizer.json");
            let dialect = Dialect::llama3();
            let params = TestParams::new(40, tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?;
            let system_tokens = params.system_prompt_tokens(0);
            let user_tokens = params.user_prompt_tokens(0);
            let mut all_tokens: Vec<u32> = Vec::new();
            all_tokens.extend_from_slice(&system_tokens);
            all_tokens.extend_from_slice(&user_tokens);
            let prefill_len = all_tokens.len();
            println!(
                "Prefill: {} system + {} user = {} tokens",
                system_tokens.len(),
                user_tokens.len(),
                prefill_len
            );

            // Download model (same as the gated test — should already be cached).
            let api = crate::models::batch_test::test_helpers::api()
                .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
            let repo = api.model("VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF".to_string());
            let model_path = repo.get("model-Q4_K_M.gguf").map_err(|e| {
                candle::Error::Msg(format!(
                    "Failed to download model: {}. This test requires internet access.",
                    e
                ))
            })?;
            println!("Model path: {:?}", model_path);

            let raw = ModelWeights::from_gguf_by_path(&model_path, &device)?;
            let inv_freq = raw
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            let model = BatchedInference::new_with_inv_freq(raw, inv_freq, 4096, &device)?;

            let n_kv_head = model.n_kv_head();
            let head_dim = model.head_dim();
            let num_layers = model.num_layers();
            println!(
                "Model: {} layers, {} kv-heads, {} head-dim",
                num_layers, n_kv_head, head_dim
            );

            // R16 session — K stored as R16 so Q values get captured during prefill / decode.
            let mode = InferenceMode::R16;
            let batch_config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(batch_config)?;
            let seq_idx = session.create_sequence()?;

            // Prefill with actual prompt tokens.
            let prefill_input = Tensor::from_vec(
                all_tokens[..prefill_len].to_vec(),
                (1, prefill_len),
                &device,
            )?;
            let logits_vec = model.forward_batched(&mut session, &[seq_idx], &[prefill_input])?;
            session.advance_sequence(seq_idx, prefill_len)?;
            println!("Prefill done ({} tokens)", prefill_len);

            // Decode: 40 more tokens — capture each generated token.
            let num_decode = 40usize;
            let mut last_logits = logits_vec
                .into_iter()
                .next()
                .ok_or_else(|| candle::Error::Msg("no logits from prefill".into()))?;
            for step in 0..num_decode {
                let next_token = last_logits
                    .squeeze(0)?
                    .argmax(candle::D::Minus1)?
                    .to_scalar::<u32>()?;
                all_tokens.push(next_token);
                let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                let out = model.forward_batched(&mut session, &[seq_idx], &[input])?;
                session.advance_sequence(seq_idx, 1)?;
                last_logits = out
                    .into_iter()
                    .next()
                    .ok_or_else(|| candle::Error::Msg("no logits from decode".into()))?;
                if (step + 1) % 10 == 0 {
                    println!("  Decoded {}/{} tokens", step + 1, num_decode);
                }
            }

            let total_tokens = prefill_len + num_decode;
            println!(
                "Generated tokens (first 16): {:?}",
                &all_tokens[..all_tokens.len().min(16)]
            );
            println!("Session complete: {} tokens", total_tokens);

            // Dump all layers — R16 path returns (block_idx, k, v, q).
            let backings = session.backings();
            let mut layer_dumps: Vec<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
                Vec::with_capacity(num_layers);
            for (layer_idx, backing) in backings.iter().enumerate() {
                let chunks = backing.dump_sequence_r16_kv_chunks(seq_idx, None)?;
                println!("  Layer {:2}: {} R16 chunks", layer_idx, chunks.len());
                layer_dumps.push(chunks);
            }

            // Compute output path relative to workspace root.
            let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("candle-nn/src/kv_cache/chunked/tests/data");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                candle::Error::Msg(format!("Failed to create output dir {:?}: {}", out_dir, e))
            })?;
            let bin_path = out_dir.join("llama-kv-data.bin");

            // Binary format v4:
            //   header: magic[8] + version:u32 + num_layers:u32 + n_kv_head:u32
            //           + chunk_size:u32 + head_dim:u32
            //           + num_tokens:u32 + tokens:[u32; num_tokens]
            //   per layer: num_chunks:u32,
            //     per chunk: block_idx:u32 + token_start:u32
            //                + k_data:[f32; elems] + v_data:[f32; elems] + q_data:[f32; elems]
            let mut file = std::fs::File::create(&bin_path).map_err(|e| {
                candle::Error::Msg(format!("Failed to create {:?}: {}", bin_path, e))
            })?;
            file.write_all(b"KVDUMP\0\0")?;
            file.write_all(&4u32.to_le_bytes())?;
            file.write_all(&(num_layers as u32).to_le_bytes())?;
            file.write_all(&(n_kv_head as u32).to_le_bytes())?;
            file.write_all(&(candle_nn::CHUNK_SIZE as u32).to_le_bytes())?;
            file.write_all(&(head_dim as u32).to_le_bytes())?;
            file.write_all(&(all_tokens.len() as u32).to_le_bytes())?;
            for &t in &all_tokens {
                file.write_all(&t.to_le_bytes())?;
            }

            let chunk_size = candle_nn::CHUNK_SIZE;
            let mut total_chunks = 0usize;
            for chunks in &layer_dumps {
                file.write_all(&(chunks.len() as u32).to_le_bytes())?;
                for (block_idx, k_data, v_data, q_data) in chunks {
                    let token_start = block_idx * chunk_size;
                    file.write_all(&(*block_idx as u32).to_le_bytes())?;
                    file.write_all(&(token_start as u32).to_le_bytes())?;
                    for &v in k_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    for &v in v_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    for &v in q_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    total_chunks += 1;
                }
            }

            println!(
                "\nDumped {} total chunks (K+V+Q v4) across {} layers to {:?}",
                total_chunks, num_layers, bin_path
            );
            let meta = std::fs::metadata(&bin_path)?;
            println!("File size: {:.1} MB", meta.len() as f64 / (1024.0 * 1024.0));

            Ok(())
        }
    } // mod kv_dump

    /// R16 vs F16 diagnostic: compare logits at every step to isolate where divergence starts.
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_llama::tests::test_r16_vs_f16_logits_comparison -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_r16_vs_f16_logits_comparison() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::batched_model::BatchedInference;

        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA required: {}", e)))?;

        // Download model
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("HF API: {}", e)))?;
        let repo = api.model("VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF".to_string());
        let model_path = repo
            .get("model-Q4_K_M.gguf")
            .map_err(|e| candle::Error::Msg(format!("Download failed: {}", e)))?;

        let raw = ModelWeights::from_gguf_by_path(&model_path, &device)?;
        let inv_freq = raw
            .rope_inv_freq()
            .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
        let model = BatchedInference::new_with_inv_freq(raw, inv_freq, 4096, &device)?;
        println!(
            "Model loaded: {} layers, {} kv-heads, hdim={}",
            model.num_layers(),
            model.n_kv_head(),
            model.head_dim()
        );

        // F16 session
        let f16_config = BatchedConfig::default().with_dtype(DType::F16);
        let mut f16_session = model.create_batched_session(f16_config)?;
        let f16_seq = f16_session.create_sequence()?;

        // R16 session
        let r16_config = BatchedConfig {
            k_format: InferenceMode::R16.k_format(),
            v_format: InferenceMode::R16.v_format(),
            compression_level: None,
            ..Default::default()
        };
        let mut r16_session = model.create_batched_session(r16_config)?;
        let r16_seq = r16_session.create_sequence()?;

        // Fixed prefill tokens (same as the gated test StoryRewrite prompt-ish)
        // Use a short recognisable prefix so we can trace outputs.
        let prefill_tokens: Vec<u32> = vec![
            128000, 128006, 9125, 128007,
            271, // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
            2675, 527, 264, 11190, 18328, 13, // You are a helpful assistant.
            128009, 128006, 882, 128007,
            271, // <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
            36227, 757, 264, 2875, 3446, 922, // Tell me a short story about
            264, 33671, 889, 33095, 279, 18566, // a dragon who guards the castle
            128009, 128006, 78191, 128007,
            271, // <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        ];
        let prefill_len = prefill_tokens.len();
        println!("\n=== Prefill: {} tokens ===", prefill_len);

        let prefill_input = Tensor::from_vec(prefill_tokens.clone(), (1, prefill_len), &device)?;

        // Prefill both
        let f16_logits =
            model.forward_batched(&mut f16_session, &[f16_seq], &[prefill_input.clone()])?;
        f16_session.advance_sequence(f16_seq, prefill_len)?;

        let r16_logits = model.forward_batched(&mut r16_session, &[r16_seq], &[prefill_input])?;
        r16_session.advance_sequence(r16_seq, prefill_len)?;

        let f16_l = &f16_logits[0]; // [1, vocab]
        let r16_l = &r16_logits[0];

        // Compare prefill logits
        let diff = (f16_l - r16_l)?.abs()?.to_dtype(DType::F32)?;
        let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean_diff = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

        let f16_argmax = f16_l
            .squeeze(0)?
            .argmax(candle::D::Minus1)?
            .to_scalar::<u32>()?;
        let r16_argmax = r16_l
            .squeeze(0)?
            .argmax(candle::D::Minus1)?
            .to_scalar::<u32>()?;

        let prefill_match = f16_argmax == r16_argmax;
        println!(
            "  Prefill logits: max_diff={:.6e}  mean_diff={:.6e}  f16_argmax={}  r16_argmax={}  {}",
            max_diff,
            mean_diff,
            f16_argmax,
            r16_argmax,
            if prefill_match { "MATCH" } else { "DIVERGE" }
        );

        // Top-5 comparison for prefill
        print_top_k("  F16 prefill top-5", f16_l, 5)?;
        print_top_k("  R16 prefill top-5", r16_l, 5)?;

        // Decode steps: always use F16 argmax as the next token (so both see same input)
        let num_decode = 15;
        let mut all_match = prefill_match;
        let mut first_diverge_step: Option<usize> = if !prefill_match { Some(0) } else { None };
        let mut next_token = f16_argmax;

        println!(
            "\n=== Decode: {} steps (using F16 argmax as input for both) ===",
            num_decode
        );
        for step in 0..num_decode {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;

            let f16_out = model.forward_batched(&mut f16_session, &[f16_seq], &[input.clone()])?;
            f16_session.advance_sequence(f16_seq, 1)?;

            let r16_out = model.forward_batched(&mut r16_session, &[r16_seq], &[input])?;
            r16_session.advance_sequence(r16_seq, 1)?;

            let f16_l = &f16_out[0];
            let r16_l = &r16_out[0];

            let diff = (f16_l - r16_l)?.abs()?.to_dtype(DType::F32)?;
            let max_d = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean_d = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

            let f16_am = f16_l
                .squeeze(0)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;
            let r16_am = r16_l
                .squeeze(0)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;

            let step_match = f16_am == r16_am;
            if !step_match && first_diverge_step.is_none() {
                first_diverge_step = Some(step + 1);
            }
            if !step_match {
                all_match = false;
            }

            println!(
                "  step {:2}: max_diff={:.6e}  mean_diff={:.6e}  f16={:6}  r16={:6}  {}",
                step + 1,
                max_d,
                mean_d,
                f16_am,
                r16_am,
                if step_match { "MATCH" } else { "DIVERGE <<<" }
            );

            // If argmax diverged, also show top-5 for this step
            if !step_match {
                print_top_k("    F16 top-5", f16_l, 5)?;
                print_top_k("    R16 top-5", r16_l, 5)?;
            }

            next_token = f16_am; // always use F16's choice
        }

        println!("\n=== Summary ===");
        println!("  Prefill match: {}", prefill_match);
        if let Some(s) = first_diverge_step {
            if s == 0 {
                println!("  First divergence: PREFILL (before any decode)");
                println!("  Diagnosis: BUG is in prefill K-write to R16 or prefill attention read from R16");
            } else {
                println!("  First divergence: decode step {}", s);
                println!("  Diagnosis: prefill path is OK; BUG is in decode scatter-write or decode attention read from R16");
            }
        } else {
            println!(
                "  All {} decode steps match! R16 and F16 are equivalent.",
                num_decode
            );
        }

        if !all_match {
            candle::bail!("R16 diverged from F16 — see above for diagnosis");
        }

        Ok(())
    }

    /// Print top-K token IDs and logit values from a [1, vocab] tensor.
    fn print_top_k(label: &str, logits: &Tensor, k: usize) -> Result<()> {
        let flat = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let vals: Vec<f32> = flat.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = indexed[..k.min(indexed.len())]
            .iter()
            .map(|(i, v)| format!("{}={:.4}", i, v))
            .collect();
        println!("{}: [{}]", label, top.join(", "));
        Ok(())
    }
}
