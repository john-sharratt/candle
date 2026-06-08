//! Qwen3 MoE (Mixture-of-Experts) implementation with quantization support.
//!
//! Supports Qwen3-30B-A3B and other Qwen3 MoE variants using quantized GGUF weights.
//! Implements `BatchedModelCore` + `BatchedAttentionLayer` for `candle-conversation`.
//!
//! Key design:
//! - Non-expert weights: loaded directly to VRAM (small relative to expert weights)
//! - Expert weights: LRU cache in VRAM (dynamic budget), cold storage in mmap
//! - 3D merged expert tensors as primary loading path, 2D per-expert as fallback
//!
//! References:
//! - [Qwen3 MoE Models](https://huggingface.co/Qwen/Qwen3-30B-A3B)

use super::batched_layer::{BatchedAttentionLayer, QkvProjection};
use super::batched_model::BatchedModelCore;
use super::expert_lre::{ExpertCache, ExpertSlot, MmapExpertRef, PipelineStats, ProfileSnapshot};
use super::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use super::profile::profile_now;
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
#[cfg(feature = "cuda")]
use candle::quantized::{get_vram_info, register_mmap_cuda, MmapRegistration};
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Initial number of RoPE positions to precompute.
pub const MAX_ROPE_SEQ_LEN: usize = 0;

/// When extending RoPE tables, grow them in this many positions at a time.
pub const ROPE_EXTEND_CHUNK: usize = 1024;

type SharedCis = Arc<RwLock<CisPrecomputations>>;

fn qwen_inv_freq(head_dim: usize, rope_theta: f32, rope_scaling_factor: Option<f32>) -> Vec<f32> {
    let factor = rope_scaling_factor.unwrap_or(1.0);
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / (factor * rope_theta.powf(i as f32 / head_dim as f32)))
        .collect()
}

// ============================================================================
// GGUF Reader Helper
// ============================================================================

struct Gguf<R: std::io::Read + std::io::Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: std::io::Read + std::io::Seek> Gguf<R> {
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

    fn metadata(&self) -> &HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

// ============================================================================
// Rotary Embeddings (copied from quantized_qwen3)
// ============================================================================

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

// ============================================================================
// Attention Weights (copied from quantized_qwen3)
// ============================================================================

#[derive(Debug, Clone)]
struct AttentionWeights {
    qkv_proj: Option<QMatMul>,
    q_proj: Option<QMatMul>,
    k_proj: Option<QMatMul>,
    v_proj: Option<QMatMul>,
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
    #[inline]
    fn project_qkv_with_compute_type(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        if let Some(qkv_proj) = &self.qkv_proj {
            let qkv = qkv_proj.forward(x)?;
            let q = qkv.narrow(2, 0, q_dim)?;
            let k = qkv.narrow(2, q_dim, kv_dim)?;
            let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;
            Ok((q, k, v))
        } else {
            let q_proj = self
                .q_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing q_proj".into()))?;
            let k_proj = self
                .k_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing k_proj".into()))?;
            let v_proj = self
                .v_proj
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("missing v_proj".into()))?;
            Ok((q_proj.forward(x)?, k_proj.forward(x)?, v_proj.forward(x)?))
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

        if offset == 0 {
            cache.reset();
        }
        let (k, v) = cache.append(&k, &v)?;

        let standard_attention = || -> Result<Tensor> {
            let k = repeat_kv(k.clone(), self.num_kv_groups)?;
            let v = repeat_kv(v.clone(), self.num_kv_groups)?;
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if l > 1 {
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
            #[cfg(feature = "flash-attn")]
            {
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
        };
        let reshaped_ctx =
            ctx.transpose(1, 2)?
                .contiguous()?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

// ============================================================================
// MLP Weights (dense layers, copied from quantized_qwen3)
// ============================================================================

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
}

// ============================================================================
// Sparse MoE Block
// ============================================================================

struct SparseMoeBlock {
    gate: QMatMul,
    cache: Arc<ExpertCache>,
    moe_layer_idx: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let k = self.num_experts_per_tok;
        #[allow(unused_variables)]
        let num_tokens = xs.dim(0)?;

        // ── 1. Route: GPU-side softmax + top-k ──
        //
        // `sort_last_dim` on `[num_tokens, 128]` is safe (128 < CUDA 1024-
        // thread-per-block limit).  We pull only the `[num_tokens, k]`
        // expert indices to CPU for cache scheduling; routing weights stay
        // on GPU and are gathered per-expert via flat index_select.
        let t = profile_now();
        let router_logits = self.gate.forward(&xs)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
        let routing_weights = routing_weights.to_dtype(DType::F32)?;

        let (sorted_w, sorted_idx) = routing_weights.sort_last_dim(false)?;
        let top_k_weights = sorted_w.narrow(1, 0, k)?; // [num_tokens, k]
        let top_k_indices = sorted_idx.narrow(1, 0, k)?.contiguous()?; // [num_tokens, k] u32

        let top_k_weights = if self.norm_topk_prob {
            let sums = top_k_weights.sum(1)?;
            top_k_weights.broadcast_div(&sums.unsqueeze(1)?)?
        } else {
            top_k_weights
        };

        // Flatten weights to 1-D on GPU — stays device-resident.
        let weights_flat = top_k_weights.flatten_all()?.contiguous()?; // [num_tokens * k]

        // ── 1b. Async DtoH for routing indices ──
        //
        // Instead of to_vec2() which drains the compute pipeline, we:
        //   1. Record event E1 on compute stream (marks sort output ready)
        //   2. Routing stream waits for E1 (GPU-side, CPU does not block)
        //   3. Async DtoH on routing stream to pinned buffer
        //   4. Record event E2 on routing stream (marks DtoH done)
        //   5. Send speculative hint to pipeline thread
        //   6. cuEventSynchronize(E2) — CPU blocks only for routing stream
        //   7. Read indices from pinned buffer
        //
        // Fallback: if routing stream or pinned buffer not available,
        // fall back to synchronous to_vec2().
        #[cfg(feature = "cuda")]
        let idx_cpu: Vec<Vec<u32>> = if let Device::Cuda(cuda_dev) = xs.device() {
            let total_indices = num_tokens * k;
            let routing_stream = self.cache.routing_stream();
            let pinned_buf = self.cache.routing_pinned_mut(total_indices);

            if let (Some(rs), Some(buf)) = (routing_stream, pinned_buf) {
                // Step 1: Record event on compute stream after sort output
                let compute_stream = cuda_dev.cuda_stream();
                let e1 = compute_stream
                    .record_event(None)
                    .map_err(candle::Error::wrap)?;

                // Step 2: Routing stream waits for sort to complete (GPU-side)
                rs.wait(&e1).map_err(candle::Error::wrap)?;

                // Step 3: Async DtoH on routing stream to pinned buffer
                let (storage, layout) = top_k_indices.storage_and_layout();
                if let candle::Storage::Cuda(cuda_storage) = &*storage {
                    // Use contiguous_offsets to get the exact element range.
                    // narrow() can leave a CudaSlice larger than the logical
                    // tensor (e.g. [1,8] narrowed from [1,128] — slice is 128
                    // but only 8 elements are valid).
                    if let Some((o1, o2)) = layout.contiguous_offsets() {
                        let elem_count = o2 - o1;
                        cuda_storage.copy_u32_to_host_on_stream(buf, rs, o1, elem_count)?;
                    } else {
                        // Non-contiguous layout: fall back to sync path
                        drop(storage);
                        let idx = top_k_indices.to_vec2::<u32>()?;
                        self.cache.record_profile("fwd_routing", t);

                        // Send hint with previous layer's experts
                        let prev_experts = self.cache.get_prev_layer_experts();
                        if !prev_experts.is_empty() {
                            self.cache.send_hint(self.moe_layer_idx, prev_experts);
                        }

                        return self.forward_with_indices(
                            xs,
                            weights_flat,
                            idx,
                            b_size,
                            seq_len,
                            hidden_dim,
                            k,
                            t,
                        );
                    }
                } else {
                    drop(storage);
                    let idx = top_k_indices.to_vec2::<u32>()?;
                    self.cache.record_profile("fwd_routing", t);
                    return self.forward_with_indices(
                        xs,
                        weights_flat,
                        idx,
                        b_size,
                        seq_len,
                        hidden_dim,
                        k,
                        t,
                    );
                }
                drop(storage);

                // Step 4: Record event on routing stream
                let e2 = rs.record_event(None).map_err(candle::Error::wrap)?;

                self.cache.record_profile("fwd_routing", t);

                // Step 5: Send speculative hint while DtoH is in-flight
                let prev_experts = self.cache.get_prev_layer_experts();
                if !prev_experts.is_empty() {
                    self.cache.send_hint(self.moe_layer_idx, prev_experts);
                }

                // Step 6: Wait for routing DtoH to complete
                let t_wait = profile_now();
                e2.synchronize().map_err(candle::Error::wrap)?;
                self.cache.record_profile("fwd_routing_wait", t_wait);

                // Step 7: Read indices from pinned buffer into Vec<Vec<u32>>
                let pinned_slice = self
                    .cache
                    .routing_pinned_mut(total_indices)
                    .expect("pinned buffer disappeared");
                let mut idx_cpu: Vec<Vec<u32>> = Vec::with_capacity(num_tokens);
                for tok in 0..num_tokens {
                    let start = tok * k;
                    idx_cpu.push(pinned_slice[start..start + k].to_vec());
                }
                idx_cpu
            } else {
                // Fallback: no routing stream or pinned buffer — sync path
                let idx = top_k_indices.to_vec2::<u32>()?;
                self.cache.record_profile("fwd_routing", t);

                // Still send hint even on sync path
                let prev_experts = self.cache.get_prev_layer_experts();
                if !prev_experts.is_empty() {
                    self.cache.send_hint(self.moe_layer_idx, prev_experts);
                }
                idx
            }
        } else {
            let idx = top_k_indices.to_vec2::<u32>()?;
            self.cache.record_profile("fwd_routing", t);
            idx
        };

        #[cfg(not(feature = "cuda"))]
        let idx_cpu: Vec<Vec<u32>> = {
            let idx = top_k_indices.to_vec2::<u32>()?;
            self.cache.record_profile("fwd_routing", t);
            idx
        };

        self.forward_with_indices(xs, weights_flat, idx_cpu, b_size, seq_len, hidden_dim, k, t)
    }

    /// Common path after routing indices are available (sync or async).
    fn forward_with_indices(
        &self,
        xs: Tensor,
        weights_flat: Tensor,
        idx_cpu: Vec<Vec<u32>>,
        b_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        k: usize,
        _routing_start: crate::models::profile::ProfileMark,
    ) -> Result<Tensor> {
        // ── 2. Build flat assignment array sorted by expert ──
        // Pure CPU bookkeeping — trivial cost (< 0.01ms for k=8 single-token).
        // Each entry: (expert_id, token_idx, flat_weight_idx)
        // Sorting by expert_id groups same-expert tokens contiguously,
        // ready for dispatch without any HashMap allocation.
        let t = profile_now();
        let num_assignments = idx_cpu.len() * k;
        let mut assignments: Vec<(u32, u32, u32)> = Vec::with_capacity(num_assignments);
        for (tok, idxs) in idx_cpu.iter().enumerate() {
            for (slot_k, &eid) in idxs.iter().enumerate() {
                assignments.push((eid, tok as u32, (tok * k + slot_k) as u32));
            }
        }
        assignments.sort_unstable_by_key(|a| a.0);

        // Collect unique expert IDs (already sorted)
        let mut expert_ids: Vec<usize> = Vec::new();
        for &(eid, _, _) in &assignments {
            let eid = eid as usize;
            if expert_ids.last() != Some(&eid) {
                expert_ids.push(eid);
            }
        }
        self.cache.record_profile("fwd_cpu_assign", t);

        // Store this layer's expert set for the next layer's speculative hint
        self.cache.set_prev_layer_experts(expert_ids.clone());

        // ── 3. Submit to pipeline (threaded or inline) ──
        //
        // `submit_moe_work` handles both modes:
        //   - Threaded (mmap path): sends work to the background pipeline
        //     thread, which does classify → DMA → compute with &mut self
        //     (no locks).  Blocks until the thread returns the result.
        //   - Inline (reader path): locks the Mutex (uncontended), computes
        //     all experts by slot index, releases.
        //
        // Either way, the caller gets back the output tensor.
        let ys = self.cache.submit_moe_work(
            self.moe_layer_idx,
            expert_ids,
            &xs,
            &weights_flat,
            assignments,
        )?;

        let result = ys.reshape((b_size, seq_len, hidden_dim))?;
        Ok(result)
    }
}

// ============================================================================
// FeedForward enum (dense MLP or MoE)
// ============================================================================

enum FeedForward {
    Mlp(MlpWeights),
    MoE(SparseMoeBlock),
}

// ============================================================================
// Layer Weights
// ============================================================================

pub struct LayerWeights {
    self_attn: AttentionWeights,
    ffn: FeedForward,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

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

    fn attention_norm(&self, x: &Tensor) -> Result<Tensor> {
        self.ln1.forward(x)
    }

    fn ffn_norm(&self, x: &Tensor) -> Result<Tensor> {
        self.ln2.forward(x)
    }

    fn ffn_forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.ffn {
            FeedForward::Mlp(m) => m.forward(x),
            FeedForward::MoE(m) => m.forward(x),
        }
    }

    fn project_qkv(&self, x: &Tensor) -> Result<QkvProjection> {
        let act_dtype = x.dtype();
        let (b_sz, seq_len, _) = x.dims3()?;

        let (mut q, mut k, mut v) = self.self_attn.project_qkv_with_compute_type(x)?;

        if q.dtype() != act_dtype {
            q = q.to_dtype(act_dtype)?;
        }
        if k.dtype() != act_dtype {
            k = k.to_dtype(act_dtype)?;
        }
        if v.dtype() != act_dtype {
            v = v.to_dtype(act_dtype)?;
        }

        let n_head = self.self_attn.num_heads;
        let n_kv_head = self.self_attn.num_kv_heads;
        let head_dim = self.self_attn.head_dim;

        let q = q
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let q_flat = q.flatten(0, 2)?;
        let q_flat = self.self_attn.q_norm.forward(&q_flat)?;
        let q = q_flat
            .reshape((b_sz, n_head, seq_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, n_head * head_dim))?;

        let k = k
            .reshape((b_sz, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let k_flat = self.self_attn.k_norm.forward(&k_flat)?;
        let k = k_flat
            .reshape((b_sz, n_kv_head, seq_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, n_kv_head * head_dim))?;

        Ok(QkvProjection { q, k, v })
    }

    fn output_projection(&self, attn_output: &Tensor) -> Result<Tensor> {
        self.self_attn.o_proj.forward(attn_output)
    }
}

// ============================================================================
// Model Weights
// ============================================================================

pub struct ModelWeights {
    embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    #[allow(dead_code)]
    expert_cache: Option<Arc<ExpertCache>>,
    #[allow(dead_code)]
    _mmap: Option<Arc<memmap2::Mmap>>,
    #[cfg(feature = "cuda")]
    _mmap_registration: Option<MmapRegistration>,
    device: Device,
    span: tracing::Span,
    span_output: tracing::Span,
}

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

    fn expert_stats(&self) -> Option<PipelineStats> {
        self.expert_cache.as_ref().map(|cache| cache.expert_stats())
    }

    fn reset_expert_stats(&self) {
        if let Some(cache) = &self.expert_cache {
            cache.reset_expert_stats();
        }
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.expert_cache
            .as_ref()
            .map_or_else(ProfileSnapshot::default, |cache| cache.snapshot_profiles())
    }

    fn k_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.k_hi
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.k_low
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.v_hi
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_MOE_KV_FACTORS.v_low
    }
}

// ============================================================================
// HuggingFace config.json support
// ============================================================================

/// Relevant fields from a HuggingFace `config.json` file placed next to a GGUF.
///
/// GGUFs sometimes omit optional flags (e.g. `expert_weights_norm`) that are
/// required to match the original training setup. This struct lets callers
/// supply the ground-truth values read from the HF config.
#[derive(Debug, Default)]
pub struct HFModelConfig {
    /// `norm_topk_prob` — whether to renormalize top-k expert weights to sum to 1.
    pub norm_topk_prob: Option<bool>,
    /// `rope_theta` — RoPE frequency base.
    pub rope_theta: Option<f64>,
    /// `max_position_embeddings` — maximum context length.
    pub max_position_embeddings: Option<usize>,
}

/// Try to read `config.json` from `model_dir` and extract relevant fields.
///
/// Returns `HFModelConfig::default()` silently if the file is absent, and
/// prints a warning if the file exists but cannot be parsed.
pub fn read_hf_config(model_dir: &std::path::Path) -> HFModelConfig {
    let cfg_path = model_dir.join("config.json");
    if !cfg_path.exists() {
        return HFModelConfig::default();
    }
    let text = match std::fs::read_to_string(&cfg_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("[config.json] read error: {e}");
            return HFModelConfig::default();
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("[config.json] parse error: {e}");
            return HFModelConfig::default();
        }
    };
    let cfg = HFModelConfig {
        norm_topk_prob: v.get("norm_topk_prob").and_then(|x| x.as_bool()),
        rope_theta: v.get("rope_theta").and_then(|x| x.as_f64()),
        max_position_embeddings: v
            .get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize),
    };
    tracing::debug!(
        "[config.json] norm_topk_prob={:?}  rope_theta={:?}  max_pos={:?}",
        cfg.norm_topk_prob,
        cfg.rope_theta,
        cfg.max_position_embeddings
    );
    cfg
}

// ============================================================================
// VRAM budget helpers
// ============================================================================

/// Detect the GGUF metadata prefix for Qwen3-MoE models.
///
/// Tries `qwen3moe`, `qwen2moe`, then falls back to whatever
/// `general.architecture` says. Returns the prefix string (e.g. "qwen2moe").
fn detect_arch_prefix(metadata: &HashMap<String, gguf_file::Value>) -> String {
    // Check general.architecture first
    if let Some(v) = metadata.get("general.architecture") {
        if let Ok(arch) = v.to_string() {
            let arch = arch.to_string();
            // The architecture value IS the prefix
            if metadata.contains_key(&format!("{arch}.block_count")) {
                tracing::debug!("GGUF arch prefix: '{arch}' (from general.architecture)");
                return arch;
            }
        }
    }
    // Fallback: probe known prefixes
    for prefix in &["qwen3moe", "qwen2moe", "qwen3", "llama"] {
        if metadata.contains_key(&format!("{prefix}.block_count")) {
            tracing::debug!("GGUF arch prefix: '{prefix}' (probed)");
            return prefix.to_string();
        }
    }
    tracing::debug!("GGUF arch prefix: 'qwen2moe' (default fallback)");
    "qwen2moe".to_string()
}

/// Pre-populate the OS page cache for a memory-mapped file.
///
/// 1. On Unix, calls `madvise(MADV_SEQUENTIAL)` then `madvise(MADV_WILLNEED)` so
///    the kernel begins readahead immediately.
/// 2. On all platforms, walks the mapping **backwards** in 8 MiB chunks, touching
///    one byte per 4 KiB page. Reading backwards ensures that if physical RAM is
///    insufficient to hold the entire file, the *beginning* of the file (read last)
///    remains hottest in the page cache — which is exactly what the model needs
///    first when loading layers sequentially.
fn warm_mmap(mmap: &memmap2::Mmap) {
    let len = mmap.len();
    let size_mb = len as f64 / (1024.0 * 1024.0);
    tracing::debug!("warm_mmap: warming {size_mb:.1} MiB …");
    let t0 = std::time::Instant::now();

    // ── 1. madvise (Unix only) ───────────────────────────────────────────
    #[cfg(unix)]
    {
        use memmap2::Advice;
        // Sequential tells the kernel we will walk linearly (enables readahead).
        let _ = mmap.advise(Advice::Sequential);
        // WillNeed asks the kernel to start paging-in the entire region now.
        let _ = mmap.advise(Advice::WillNeed);
    }

    // ── 2. Backwards page-touch ──────────────────────────────────────────
    const CHUNK: usize = 8 * 1024 * 1024; // 8 MiB
    const PAGE: usize = 4096; // OS page size

    let ptr = mmap.as_ptr();
    let mut offset = len;
    let mut _acc: u8 = 0; // accumulator to prevent dead-code elimination

    while offset > 0 {
        let chunk_start = offset.saturating_sub(CHUNK);
        // Touch one byte per page inside this chunk (forward within the chunk).
        let mut pos = chunk_start;
        while pos < offset {
            // SAFETY: pos is always < mmap.len(), ptr valid for the mapping lifetime.
            _acc = _acc.wrapping_add(unsafe { *ptr.add(pos) });
            pos += PAGE;
        }
        offset = chunk_start;
    }

    // Ensure the accumulator is not optimised away.
    std::hint::black_box(_acc);

    let elapsed = t0.elapsed();
    tracing::debug!(
        "warm_mmap: done in {:.2}s ({:.0} MiB/s)",
        elapsed.as_secs_f64(),
        size_mb / elapsed.as_secs_f64()
    );
}

impl ModelWeights {
    /// Load model from GGUF via reader (non-mmap path).
    /// MoE layers load all experts to VRAM (no LRU cache in this path).
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
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

        let p = detect_arch_prefix(gg.metadata());

        let num_attention_heads = md_get(&format!("{p}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads = md_get(&format!("{p}.attention.head_count_kv"))?.to_u32()? as usize;
        let num_layers = md_get(&format!("{p}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{p}.embedding_length"))?.to_u32()? as usize;

        let head_dim = md_opt_u32(&format!("{p}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let max_position_embeddings =
            md_opt_u32(&format!("{p}.context_length")).unwrap_or(32768) as usize;

        let rms_norm_eps =
            md_get(&format!("{p}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;

        let rope_freq_base =
            md_opt_f32(&format!("{p}.rope.freq_base")).unwrap_or(1_000_000f32) as f64;

        let rope_scaling_factor =
            md_opt_f32(&format!("{p}.rope.scaling.factor")).filter(|f| *f > 0.0 && *f != 1.0);

        let n_expert = md_opt_u32(&format!("{p}.expert_count")).unwrap_or(1) as usize;
        let n_expert_used = md_opt_u32(&format!("{p}.expert_used_count")).unwrap_or(1) as usize;
        // Qwen3-MoE always uses norm_topk_prob=true; GGUF often omits this key so default to 1.
        let norm_topk_prob = md_opt_u32(&format!("{p}.expert_weights_norm")).unwrap_or(1) == 1;

        tracing::debug!("GGUF arch: {p} (reader path, no config.json)  layers={num_layers} hidden={hidden_size} eps={rms_norm_eps:.2e} heads={num_attention_heads}Q/{num_kv_heads}KV head_dim={head_dim} ctx={max_position_embeddings} rope_base={rope_freq_base} experts={n_expert}/{n_expert_used} norm={norm_topk_prob}");

        let dtype = DType::F16;

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
        let mut reader_moe_count: usize = 0;
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");
            let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

            // Attention
            let q_w = gg.tensor(&format!("{prefix}.attn_q.weight"))?;
            let k_w = gg.tensor(&format!("{prefix}.attn_k.weight"))?;
            let v_w = gg.tensor(&format!("{prefix}.attn_v.weight"))?;
            let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
            let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
            let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

            let try_fuse = device.is_cuda()
                && q_w.dtype() == k_w.dtype()
                && q_w.dtype() == v_w.dtype()
                && !matches!(
                    q_w.dtype(),
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                );

            let (qkv_proj, q_proj, k_proj, v_proj) = if try_fuse {
                #[cfg(feature = "cuda")]
                {
                    let qkv_w = QTensor::concat_rows_cuda(&[&q_w, &k_w, &v_w])?;
                    (Some(QMatMul::from_qtensor(qkv_w)?), None, None, None)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle::bail!("fused QKV requires the cuda feature");
                }
            } else {
                (
                    None,
                    Some(QMatMul::from_weights(q_w.into())?),
                    Some(QMatMul::from_weights(k_w.into())?),
                    Some(QMatMul::from_weights(v_w.into())?),
                )
            };

            let self_attn = AttentionWeights {
                qkv_proj,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                num_kv_groups: num_attention_heads / num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
            };

            // FFN: detect MoE vs dense by checking for expert tensors
            let has_moe_tensors = gg
                .ct
                .tensor_infos
                .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                || gg
                    .ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

            let ffn = if has_moe_tensors && n_expert > 1 {
                // In the reader path, just load all experts to VRAM (no LRU)
                let gate = gg.qmatmul(&format!("{prefix}.ffn_gate_inp.weight"))?;
                // Try per-expert 2D naming
                let mut experts_data = Vec::new();
                for j in 0..n_expert {
                    let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.{j}.weight"))?;
                    let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.{j}.weight"))?;
                    let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.{j}.weight"))?;
                    experts_data.push((gate_proj, up_proj, down_proj));
                }

                // Reader path: all experts already in VRAM — pre-populate the
                // cache so classify_and_load's hit path finds them immediately.
                // No mmap needed since nothing will ever miss.
                //
                // NOTE: each MoE layer gets its own independent ExpertCache
                // here, unlike the mmap path (from_gguf_by_path) which shares
                // a single global cache across all layers with cross-layer LRU
                // eviction.  This is fine because the reader path keeps every
                // expert resident — there is no eviction pressure.
                let moe_layer_idx = reader_moe_count;
                reader_moe_count += 1;

                let num_experts = experts_data.len();
                let mut slots: Vec<Option<ExpertSlot>> = Vec::with_capacity(num_experts);
                let mut key_to_slot = std::collections::HashMap::new();
                let mut last_used = Vec::with_capacity(num_experts);
                let mut slot_to_key = Vec::with_capacity(num_experts);

                for (j, (gp, up, dp)) in experts_data.into_iter().enumerate() {
                    slots.push(Some(ExpertSlot {
                        gate_proj: gp,
                        up_proj: up,
                        down_proj: dp,
                    }));
                    key_to_slot.insert((moe_layer_idx, j), j);
                    last_used.push(j as u32);
                    slot_to_key.push(Some((moe_layer_idx, j)));
                }
                let generation = num_experts as u32;

                FeedForward::MoE(SparseMoeBlock {
                    gate,
                    cache: Arc::new(ExpertCache::new_prepopulated(
                        slots,
                        key_to_slot,
                        last_used,
                        generation,
                        slot_to_key,
                        device,
                    )),
                    moe_layer_idx,
                    num_experts_per_tok: n_expert_used,
                    norm_topk_prob,
                })
            } else {
                let gate_w = gg.tensor(&format!("{prefix}.ffn_gate.weight"))?;
                let up_w = gg.tensor(&format!("{prefix}.ffn_up.weight"))?;
                let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;

                let try_fuse = device.is_cuda()
                    && gate_w.dtype() == up_w.dtype()
                    && !matches!(
                        gate_w.dtype(),
                        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                    );

                let (gate_up_proj, gate_proj, up_proj) = if try_fuse {
                    #[cfg(feature = "cuda")]
                    {
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
                        Some(QMatMul::from_weights(gate_w.into())?),
                        Some(QMatMul::from_weights(up_w.into())?),
                    )
                };
                FeedForward::Mlp(MlpWeights {
                    gate_up_proj,
                    gate_proj,
                    up_proj,
                    down_proj,
                    act_fn: Activation::Silu,
                    span: tracing::span!(tracing::Level::TRACE, "mlp"),
                })
            };

            layers.push(LayerWeights {
                self_attn,
                ffn,
                ln1,
                ln2,
            });
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            // Reader path: each SparseMoeBlock owns its own ExpertCache
            // with all experts pre-loaded, so no global cache is needed.
            // (The mmap path sets this to Some(...) for cross-layer LRU.)
            expert_cache: None,
            _mmap: None,
            #[cfg(feature = "cuda")]
            _mmap_registration: None,
            device: device.clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    /// Load model from GGUF via mmap with LRU expert cache and VRAM budget management.
    ///
    /// - Non-expert weights: loaded directly to VRAM (small relative to expert weights)
    /// - Expert weights: LRU cache in VRAM (dynamic budget based on free VRAM)
    /// - 3D merged expert tensors as primary, 2D per-expert as fallback
    ///
    /// `progress`, when supplied, is called with `(layers_loaded, num_layers)`
    /// after each layer's weights have been mounted — drives a UI progress
    /// bar without coupling this loader to the daemon's progress type.
    pub fn from_gguf_by_path(
        file_path: &std::path::Path,
        device: &Device,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<Self> {
        use memmap2::MmapOptions;

        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| candle::Error::Msg(format!("Failed to mmap file: {}", e)))?
        };
        let mmap = Arc::new(mmap);

        // Mmap warming is handled by ExpertCache::new() (prewarm_expert_cache)
        // which fills VRAM slots first, then warms remaining pages.
        // For non-MoE models, warm_mmap() is called below after cache building.

        // Register mmap with CUDA for DMA acceleration
        #[cfg(feature = "cuda")]
        let _mmap_guard = if matches!(device, Device::Cuda(_)) {
            register_mmap_cuda(&mmap)
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let _mmap_guard: Option<()> = None;

        // Parse GGUF
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut cursor)?;

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());

        let p = detect_arch_prefix(&ct.metadata);
        // Read config.json from the same directory as the GGUF for fallback values.
        let hf_cfg = file_path
            .parent()
            .map(|d| read_hf_config(d))
            .unwrap_or_default();

        let num_attention_heads = md_get(&format!("{p}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads = md_get(&format!("{p}.attention.head_count_kv"))?.to_u32()? as usize;
        let num_layers = md_get(&format!("{p}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{p}.embedding_length"))?.to_u32()? as usize;

        let head_dim = md_opt_u32(&format!("{p}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let max_position_embeddings = md_opt_u32(&format!("{p}.context_length"))
            .map(|v| v as usize)
            .or(hf_cfg.max_position_embeddings)
            .unwrap_or(32768);

        let rms_norm_eps =
            md_get(&format!("{p}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;

        let rope_freq_base = md_opt_f32(&format!("{p}.rope.freq_base"))
            .map(|v| v as f64)
            .or(hf_cfg.rope_theta)
            .unwrap_or(1_000_000.0);

        let rope_scaling_factor =
            md_opt_f32(&format!("{p}.rope.scaling.factor")).filter(|f| *f > 0.0 && *f != 1.0);

        let n_expert = md_opt_u32(&format!("{p}.expert_count")).unwrap_or(1) as usize;
        let n_expert_used = md_opt_u32(&format!("{p}.expert_used_count")).unwrap_or(1) as usize;
        // Qwen3-MoE always uses norm_topk_prob=true. The GGUF key may be absent;
        // config.json takes precedence, then GGUF, then we default to true.
        let norm_topk_prob = md_opt_u32(&format!("{p}.expert_weights_norm"))
            .map(|v| v == 1)
            .or(hf_cfg.norm_topk_prob)
            .unwrap_or(true);

        tracing::debug!("GGUF arch: {p}  layers={num_layers} hidden={hidden_size} eps={rms_norm_eps:.2e} heads={num_attention_heads}Q/{num_kv_heads}KV head_dim={head_dim} ctx={max_position_embeddings} rope_base={rope_freq_base} experts={n_expert}/{n_expert_used} norm={norm_topk_prob}");

        // Expert FFN size (moe_intermediate_size)
        let _expert_ffn_size =
            md_opt_u32(&format!("{p}.expert_feed_forward_length")).unwrap_or(2048) as usize;

        let dtype_model = DType::F16;

        // ── VRAM info (budget computed later once expert sizes are known) ──
        #[cfg(feature = "cuda")]
        let (free_vram, total_vram) = if matches!(device, Device::Cuda(_)) {
            get_vram_info()?
        } else {
            (0, 0)
        };
        #[cfg(not(feature = "cuda"))]
        let (free_vram, total_vram): (usize, usize) = (0, 0);

        tracing::debug!(
            "VRAM: total={:.1} GB, free={:.1} GB",
            total_vram as f64 / 1e9,
            free_vram as f64 / 1e9,
        );

        // Helper: load tensor directly to VRAM
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        // Load embeddings
        let tok_embed = load_tensor("token_embd.weight")?.dequantize(device)?;
        let embeddings = Embedding::new(tok_embed, hidden_size)?;

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype_model,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            rope_scaling_factor,
            device,
        )?);

        // Count MoE layers for cache indexing
        let mut moe_count: usize = 0;

        // ── Compute expert byte offsets (3D merged primary, 2D fallback) ──
        // We collect these before loading layers so the ExpertCache can be built first.
        // Detect MoE layers by checking for expert tensors (not metadata-based).
        let moe_layer_indices: Vec<usize> = (0..num_layers)
            .filter(|i| {
                let prefix = format!("blk.{i}");
                ct.tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                    || ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
            })
            .collect();
        let num_moe_layers = moe_layer_indices.len();

        let mut all_host_refs: Vec<Vec<MmapExpertRef>> = Vec::with_capacity(num_moe_layers);

        // Determine expert shapes and byte offsets
        if num_moe_layers > 0 {
            for &i in &moe_layer_indices {
                let prefix = format!("blk.{i}");
                let mut layer_refs = Vec::with_capacity(n_expert);

                // Try 3D merged tensors first
                let gate_exps_name = format!("{prefix}.ffn_gate_exps.weight");
                let up_exps_name = format!("{prefix}.ffn_up_exps.weight");
                let down_exps_name = format!("{prefix}.ffn_down_exps.weight");

                if let (Some(gate_info), Some(up_info), Some(down_info)) = (
                    ct.tensor_infos.get(&gate_exps_name),
                    ct.tensor_infos.get(&up_exps_name),
                    ct.tensor_infos.get(&down_exps_name),
                ) {
                    // 3D merged: shape is (num_experts, out_dim, in_dim)

                    let gate_dims = gate_info.shape.dims();
                    let up_dims = up_info.shape.dims();
                    let down_dims = down_info.shape.dims();

                    // Per-expert element count (product of dims after first)
                    let gate_expert_elems: usize = gate_dims[1..].iter().product();
                    let up_expert_elems: usize = up_dims[1..].iter().product();
                    let down_expert_elems: usize = down_dims[1..].iter().product();

                    // Use per-projection dtype for correct byte calculation
                    let gate_expert_bytes = gate_expert_elems / gate_info.ggml_dtype.block_size()
                        * gate_info.ggml_dtype.type_size();
                    let up_expert_bytes = up_expert_elems / up_info.ggml_dtype.block_size()
                        * up_info.ggml_dtype.type_size();
                    let down_expert_bytes = down_expert_elems / down_info.ggml_dtype.block_size()
                        * down_info.ggml_dtype.type_size();

                    let gate_base = (ct.tensor_data_offset + gate_info.offset) as usize;
                    let up_base = (ct.tensor_data_offset + up_info.offset) as usize;
                    let down_base = (ct.tensor_data_offset + down_info.offset) as usize;

                    for j in 0..n_expert {
                        layer_refs.push(MmapExpertRef {
                            gate_offset: gate_base + j * gate_expert_bytes,
                            gate_len: gate_expert_bytes,
                            up_offset: up_base + j * up_expert_bytes,
                            up_len: up_expert_bytes,
                            down_offset: down_base + j * down_expert_bytes,
                            down_len: down_expert_bytes,
                            gate_shape: gate_dims[1..].to_vec(),
                            up_shape: up_dims[1..].to_vec(),
                            down_shape: down_dims[1..].to_vec(),
                            gate_dtype: gate_info.ggml_dtype,
                            up_dtype: up_info.ggml_dtype,
                            down_dtype: down_info.ggml_dtype,
                        });
                    }
                } else {
                    // 2D per-expert fallback
                    for j in 0..n_expert {
                        let gate_name = format!("{prefix}.ffn_gate.{j}.weight");
                        let up_name = format!("{prefix}.ffn_up.{j}.weight");
                        let down_name = format!("{prefix}.ffn_down.{j}.weight");

                        let gate_info = ct.tensor_infos.get(&gate_name).ok_or_else(|| {
                            candle::Error::Msg(format!("tensor {} not found", gate_name))
                        })?;
                        let up_info = ct.tensor_infos.get(&up_name).ok_or_else(|| {
                            candle::Error::Msg(format!("tensor {} not found", up_name))
                        })?;
                        let down_info = ct.tensor_infos.get(&down_name).ok_or_else(|| {
                            candle::Error::Msg(format!("tensor {} not found", down_name))
                        })?;

                        let gate_bytes = gate_info.shape.elem_count()
                            / gate_info.ggml_dtype.block_size()
                            * gate_info.ggml_dtype.type_size();
                        let up_bytes = up_info.shape.elem_count() / up_info.ggml_dtype.block_size()
                            * up_info.ggml_dtype.type_size();
                        let down_bytes = down_info.shape.elem_count()
                            / down_info.ggml_dtype.block_size()
                            * down_info.ggml_dtype.type_size();

                        layer_refs.push(MmapExpertRef {
                            gate_offset: (ct.tensor_data_offset + gate_info.offset) as usize,
                            gate_len: gate_bytes,
                            up_offset: (ct.tensor_data_offset + up_info.offset) as usize,
                            up_len: up_bytes,
                            down_offset: (ct.tensor_data_offset + down_info.offset) as usize,
                            down_len: down_bytes,
                            gate_shape: gate_info.shape.dims().to_vec(),
                            up_shape: up_info.shape.dims().to_vec(),
                            down_shape: down_info.shape.dims().to_vec(),
                            gate_dtype: gate_info.ggml_dtype,
                            up_dtype: up_info.ggml_dtype,
                            down_dtype: down_info.ggml_dtype,
                        });
                    }
                }
                all_host_refs.push(layer_refs);
            }
        }

        // Combined progress denominator for the `from_gguf_by_path`
        // outer callback: expert cache uploads (`num_moe_layers ×
        // n_expert`) followed by the per-layer attention loop
        // (`num_layers`). The bar then advances continuously through
        // both phases, instead of sitting at 0% until the cache
        // finishes.
        let total_expert_ticks = num_moe_layers * n_expert;
        let total_units = total_expert_ticks + num_layers;

        // ── Build Expert Cache ──
        let expert_cache = if !all_host_refs.is_empty() && n_expert > 0 {
            // Determine per-expert shapes from the first layer's first expert
            // Use max expert size across all layers for budget calculation
            let max_expert_size = all_host_refs
                .iter()
                .flat_map(|layer| layer.iter())
                .map(|r| r.gate_len + r.up_len + r.down_len)
                .max()
                .unwrap_or(0);
            let total_experts = num_moe_layers * n_expert;
            let total_expert_bytes = total_experts * max_expert_size;

            // ── VRAM budget for expert LRU cache ──
            // Formula: min(max(free − 5 GB, free × 50%), total_expert_bytes)
            //  • max(free−5GB, 50%) → on small GPUs the 50% floor keeps a usable
            //    pool; on larger GPUs free−5GB greedily takes most of VRAM.
            //  • Crossover is at free=10 GB.
            //  • The min() cap prevents allocating more slots than experts exist,
            //    freeing leftover VRAM for KV cache / context length.
            const RESERVE_BYTES: usize = 5 * 1024 * 1024 * 1024; // 5 GB
            let generous = {
                let option_a = free_vram.saturating_sub(RESERVE_BYTES);
                let option_b = free_vram / 2;
                option_a.max(option_b)
            };
            let expert_budget = generous.min(total_expert_bytes);

            let num_slots = if max_expert_size > 0 {
                let base = expert_budget / max_expert_size;
                // Round up: if leftover VRAM can fit ≥50% of another expert,
                // take it — avoids DMA churn when the model *almost* fits.
                let remainder = expert_budget % max_expert_size;
                let rounded = if remainder >= max_expert_size / 2 {
                    base + 1
                } else {
                    base
                };
                rounded.min(total_experts)
            } else {
                0
            };

            tracing::debug!(
                "Expert cache: {} slots / {} total (budget {:.1} GB / {:.1} GB model, max expert {:.1} KB)",
                num_slots,
                total_experts,
                expert_budget as f64 / 1e9,
                total_expert_bytes as f64 / 1e9,
                max_expert_size as f64 / 1e3,
            );

            // Wrap the outer callback so the cache's per-expert
            // `(done, total_experts)` ticks land on the combined
            // denominator computed above.
            let cache_wrapper = progress
                .map(|cb| move |done: usize, _total: usize| cb(done, total_units));
            let cache_progress: Option<&dyn Fn(usize, usize)> = cache_wrapper
                .as_ref()
                .map(|f| f as &dyn Fn(usize, usize));
            let cache = ExpertCache::new(
                mmap.clone(),
                all_host_refs,
                num_slots,
                device,
                n_expert,
                Some(file_path),
                cache_progress,
            )?;
            Some(Arc::new(cache))
        } else {
            // No MoE layers — warm the entire mmap the simple way.
            warm_mmap(&mmap);
            None
        };

        // ── Load layers ──
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let ln1 = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_norm.weight"))?,
                rms_norm_eps,
            )?;
            let ln2 = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.ffn_norm.weight"))?,
                rms_norm_eps,
            )?;

            // Attention
            let q_w = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let k_w = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let v_w = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let o_proj = QMatMul::from_weights(
                load_tensor(&format!("{prefix}.attn_output.weight"))?.into(),
            )?;
            let q_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_q_norm.weight"))?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_k_norm.weight"))?,
                rms_norm_eps,
            )?;

            let try_fuse = device.is_cuda()
                && q_w.dtype() == k_w.dtype()
                && q_w.dtype() == v_w.dtype()
                && !matches!(
                    q_w.dtype(),
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                );

            let (qkv_proj, q_proj, k_proj, v_proj) = if try_fuse {
                #[cfg(feature = "cuda")]
                {
                    let qkv_w = QTensor::concat_rows_cuda(&[&q_w, &k_w, &v_w])?;
                    (Some(QMatMul::from_qtensor(qkv_w)?), None, None, None)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle::bail!("fused QKV requires the cuda feature");
                }
            } else {
                (
                    None,
                    Some(QMatMul::from_weights(q_w.into())?),
                    Some(QMatMul::from_weights(k_w.into())?),
                    Some(QMatMul::from_weights(v_w.into())?),
                )
            };

            let self_attn = AttentionWeights {
                qkv_proj,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                num_kv_groups: num_attention_heads / num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
            };

            // FFN: detect MoE vs dense by checking for expert tensors
            let has_moe_tensors = ct
                .tensor_infos
                .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                || ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

            let ffn = if has_moe_tensors && n_expert > 1 {
                let gate = QMatMul::from_weights(
                    load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?.into(),
                )?;

                let cache_ref = expert_cache
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("expert_cache is None for MoE layer".into()))?
                    .clone();

                let moe_layer_idx = moe_count;
                moe_count += 1;

                FeedForward::MoE(SparseMoeBlock {
                    gate,
                    cache: cache_ref,
                    moe_layer_idx,
                    num_experts_per_tok: n_expert_used,
                    norm_topk_prob,
                })
            } else {
                // Dense MLP
                let gate_w = load_tensor(&format!("{prefix}.ffn_gate.weight"))?;
                let up_w = load_tensor(&format!("{prefix}.ffn_up.weight"))?;
                let down_proj = QMatMul::from_weights(
                    load_tensor(&format!("{prefix}.ffn_down.weight"))?.into(),
                )?;

                let try_fuse = device.is_cuda()
                    && gate_w.dtype() == up_w.dtype()
                    && !matches!(
                        gate_w.dtype(),
                        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                    );

                let (gate_up_proj, gate_proj, up_proj) = if try_fuse {
                    #[cfg(feature = "cuda")]
                    {
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
                        Some(QMatMul::from_weights(gate_w.into())?),
                        Some(QMatMul::from_weights(up_w.into())?),
                    )
                };
                FeedForward::Mlp(MlpWeights {
                    gate_up_proj,
                    gate_proj,
                    up_proj,
                    down_proj,
                    act_fn: Activation::Silu,
                    span: tracing::span!(tracing::Level::TRACE, "mlp"),
                })
            };

            layers.push(LayerWeights {
                self_attn,
                ffn,
                ln1,
                ln2,
            });

            if (i + 1) % 8 == 0 || i == num_layers - 1 {
                tracing::debug!("Layer {}/{} loaded", i + 1, num_layers);
            }
            // Continue the bar from where the expert cache left off, on
            // the same combined denominator (`total_units`). Each layer
            // is one tick — the layers run far faster than experts but
            // they're a small fraction of the total either way.
            if let Some(cb) = progress {
                cb(total_expert_ticks + i + 1, total_units);
            }
        }

        // Load final norm and output projection
        let norm = RmsNorm::from_qtensor(load_tensor("output_norm.weight")?, rms_norm_eps)?;
        let lm_head_tensor = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => load_tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        tracing::debug!("Model loaded: {} layers ({} MoE)", num_layers, moe_count);

        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            expert_cache,
            _mmap: Some(mmap),
            #[cfg(feature = "cuda")]
            _mmap_registration: _mmap_guard,
            device: device.clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    /// Create KV caches for all layers.
    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = (0..self.layers.len())
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Forward pass with typed context struct.
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

        let embed_dtype = ctx.kv_caches.dtype();
        let mut h = self
            .embeddings
            .forward_as_dtype(ctx.input_ids, embed_dtype)?
            .contiguous()?;

        for (layer, cache) in self.layers.iter().zip(ctx.kv_caches.caches.iter_mut()) {
            let residual = h.clone();
            let normed = layer.ln1.forward(&h)?;
            let attn_out = layer.self_attn.forward(cache, &normed, ctx.offset)?;
            h = (residual + attn_out)?;

            let residual = h.clone();
            let normed = layer.ln2.forward(&h)?;
            let ffn_out = match &layer.ffn {
                FeedForward::Mlp(m) => m.forward(&normed)?,
                FeedForward::MoE(m) => m.forward(&normed)?,
            };
            h = (residual + ffn_out)?;
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
        let mut h = self
            .embeddings
            .forward_as_dtype(input, embed_dtype)?
            .contiguous()?;
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            let residual = h.clone();
            let normed = layer.ln1.forward(&h)?;
            let attn_out = layer.self_attn.forward(cache, &normed, offset)?;
            h = (residual + attn_out)?;
            let residual = h.clone();
            let normed = layer.ln2.forward(&h)?;
            let ffn_out = match &layer.ffn {
                FeedForward::Mlp(m) => m.forward(&normed)?,
                FeedForward::MoE(m) => m.forward(&normed)?,
            };
            h = (residual + ffn_out)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        self.lm_head.forward(&h)
    }

    /// Returns the RoPE inverse frequency vector used by this model.
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::utils::{SequentialCallbacks, TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;

    #[test]
    #[ignore] // Run with: cargo test --release --features cuda --lib --package candle-transformers quantized_qwen3_moe::tests::test_parallel_batched_forwarding -- --ignored --nocapture
    fn test_parallel_batched_forwarding() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ This test requires --features cuda");
            return Ok(());
        }

        println!("\n=== Qwen3-30B-A3B MoE Batched Forwarding Test ===\n");

        let num_generate_tokens = 10;
        let dialect = Dialect::chat_ml();

        // Download tokenizer
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
        let tok_repo = api.model("Qwen/Qwen3-30B-A3B-Instruct-2507".to_string());
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
            .with_timeout_secs(1200); // 20 minutes

        println!("\n=== Loading Model ===\n");

        let repo = api.repo(hf_hub::Repo::with_revision(
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let model_path = repo
            .get("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
            .map_err(|e| {
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
            // F16 single context
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // BF16 single context
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // BF16 multi-context
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // Q8
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 20,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 32,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // Q4
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::Skip),
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 48,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::Skip),
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 2,
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
                num_contexts: 2,
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
                num_contexts: 2,
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
                num_contexts: 2,
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
                num_contexts: 2,
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
                num_contexts: 2,
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
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 64,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 120,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 2,
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

        let make_sampler = || {
            use crate::generation::{LogitsProcessor, Sampling};
            let mut logits_processor = LogitsProcessor::from_sampling(299792458, Sampling::ArgMax);
            move |logits: &Tensor| {
                let logits = logits.squeeze(0)?;
                logits_processor.sample(&logits)
            }
        };

        use crate::models::batched_model::BatchedInference;
        type WrappedModel = BatchedInference<ModelWeights>;

        let sequential = SequentialCallbacks {
            create_cache: |config: &TestConfig, model: &WrappedModel| {
                let mut caches = model.model().create_kv_caches(512);
                caches.force_dtype(config.mode.compute_dtype());
                Ok(caches)
            },
            forward: |ctx: SequenceContext, model: &WrappedModel| {
                model.model().forward_with_context(ctx)
            },
            sample: make_sampler(),
        };

        let load_model = || {
            let model = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
            println!("✓ Model loaded\n");
            let inv_freq = model
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
        };

        params.run(configs, load_model, sequential)?;

        Ok(())
    }

    /// KV-cache dump for offline selection analysis (Qwen3-MoE).
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::kv_dump::test_dump_kv_cache_data -- --ignored --nocapture
    mod kv_dump {
        use super::*;
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::batched_model::BatchedInference;
        use std::io::Write;

        /// Dump real KV cache data (K, V, Q) from Qwen3-30B-A3B MoE.
        ///
        /// Runs a single R16 session using the same ChatML system+user prompt as the
        /// gated test (StoryRewrite), followed by 40 decode steps.  R16 mode keeps K
        /// in raw F16 with Q-capture space, so the prefill/decode kernels populate
        /// `block_r16->q[]` with real Q projections; the dump captures K + V + Q in
        /// v4 binary format for offline selection analysis.
        ///
        /// Output: `candle-nn/src/kv_cache/chunked/tests/data/qwen3-kv-data.bin`
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

            // Download tokenizer
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
            let tok_repo = api.model("Qwen/Qwen3-30B-A3B-Instruct-2507".to_string());
            let tokenizer_path = tok_repo
                .get("tokenizer.json")
                .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
            let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
                .map_err(|e| candle::Error::Msg(format!("Failed to read tokenizer: {}", e)))?;

            let dialect = Dialect::chat_ml();
            let params = TestParams::new(10, &tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
                .with_suppress_thinking(true);

            // Build the full ChatML prompt: system + user + assistant header
            // This matches what the gated test sends for context index 0.
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

            // Download model
            let repo = api.repo(hf_hub::Repo::with_revision(
                "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ));
            let model_path = repo
                .get("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
                .map_err(|e| candle::Error::Msg(format!("Failed to download model: {}", e)))?;
            println!("Model path: {:?}", model_path);

            let raw = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
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

            // Create an R16 session — K stored as R16 so Q values get captured in
            // each block_r16->q[] field during prefill / decode.
            let mode = InferenceMode::R16;
            let batch_config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(batch_config)?;
            let seq_idx = session.create_sequence()?;

            // Prefill with actual prompt tokens
            let prefill_input = Tensor::from_vec(
                all_tokens[..prefill_len].to_vec(),
                (1, prefill_len),
                &device,
            )?;
            let logits_vec = model.forward_batched(&mut session, &[seq_idx], &[prefill_input])?;
            session.advance_sequence(seq_idx, prefill_len)?;
            println!("Prefill done ({} tokens)", prefill_len);

            // Decode: 40 tokens (greedy argmax) — matching production generate_max_len
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
            println!("Session complete: {} total tokens", total_tokens);

            // Dump all layers — R16 path returns (block_idx, k, v, q).
            let backings = session.backings();
            let mut layer_dumps: Vec<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
                Vec::with_capacity(num_layers);
            for (layer_idx, backing) in backings.iter().enumerate() {
                let chunks = backing.dump_sequence_r16_kv_chunks(seq_idx, None)?;
                println!("  Layer {:2}: {} R16 chunks", layer_idx, chunks.len());
                layer_dumps.push(chunks);
            }

            // Output path
            let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("candle-nn/src/kv_cache/chunked/tests/data");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                candle::Error::Msg(format!("Failed to create output dir {:?}: {}", out_dir, e))
            })?;
            let bin_path = out_dir.join("qwen3-kv-data.bin");

            // Binary format v4 — header + tokens + per-chunk (block_idx, token_start, k, v, q).
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
    }

    // ── RULER long-context benchmark ──────────────────────────────────────────

    /// End-to-end RULER evaluation for Qwen3-30B-A3B.
    ///
    /// Generates RULER task data in-process (no external Python required),
    /// runs batched inference for four compression modes, and prints an
    /// accuracy table.
    ///
    /// Run with:
    ///   cargo test --release --features cuda,verbose --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::test_ruler_eval -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_ruler_eval() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        println!("⚠  No CUDA — performance will be poor");

        use crate::models::batch_test::ruler_gen::{
            run_ruler_benchmark, RulerBenchConfig, RulerDataSource, RulerTask, QWEN3_EOS_IDS,
        };
        use crate::models::batch_test::test_helpers::{download_hf_gguf, load_hf_tokenizer};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::batched_model::BatchedInference;

        let tokenizer = load_hf_tokenizer("Qwen/Qwen3-30B-A3B-Instruct-2507")?;
        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA device: {e}")))?;
        let model_path = download_hf_gguf(
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
            "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            "main",
        )?;
        println!("Model path: {model_path:?}");
        let weights = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
        let inv_freq = weights
            .rope_inv_freq()
            .ok_or_else(|| candle::Error::Msg("no inv_freq".into()))?;
        let model = BatchedInference::new_with_inv_freq(weights, inv_freq, 32_768, &device)?;
        println!("✓ Model loaded");

        let cfg = RulerBenchConfig {
            model_name: "Qwen3-30B-A3B-Q4_K_M",
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 32_768,
            max_gen_tokens: 50,
            modes: &[
                (None, "F16 (lossless)"),
                (Some(InferenceMode::Q4_0), "Q4_0 (4.5 BPE)"),
                (Some(InferenceMode::C5), "C5  (PalQuant ~4.4 BPE)"),
                (Some(InferenceMode::Q3_0), "Q3_0 (3.5 BPE)"),
                (Some(InferenceMode::C8), "C8  (PalQuant ~3.3 BPE)"),
            ],
            lengths_samples: &[(4_096, 20), (8_192, 10), (16_384, 5), (32_768, 3)],
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
