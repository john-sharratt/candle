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

#[cfg(feature = "cuda")]
use super::batched_layer::{BatchedAttentionLayer, QkvProjection};
#[cfg(feature = "cuda")]
use super::batched_model::BatchedModelCore;
use super::expert_lre::{
    ExpertCache, ExpertSlot, MmapExpertRef, MoeInput, PipelineStats, ProfileSnapshot,
};
use super::kv_cache_utils::{new_kv_caches, KvCaches};
use super::profile::{profile_now, ProfileMark};
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use crate::quantized_nn::RmsNorm;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{moe_route, DynamicActs};
#[cfg(feature = "cuda")]
use candle::quantized::{get_vram_info, register_mmap_cuda, MmapRegistration};
use candle::quantized::{gguf_file, GgmlDType, Int8Mode, QTensor};
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
}

// ============================================================================
// Attention Weights (copied from quantized_qwen3)
// ============================================================================

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: Option<QMatMul>,
    k_proj: Option<QMatMul>,
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl AttentionWeights {
    /// q/k/v projection over a producer-prepared [`DynamicActs`] (the fused `ln1` output).
    #[cfg(feature = "cuda")]
    fn project_qkv(
        &self,
        acts: &DynamicActs,
        out_dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
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
        match acts {
            // int8: ONE segmented launch over the three KO weights (q/k/v kept separate — no
            // concat). Float-identical to three separate matmuls, with full GPU occupancy so the
            // tiny k/v GEMVs no longer starve. Output [lead.., q_dim+2·kv_dim] → narrow to q/k/v.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[q_proj.inner(), k_proj.inner(), v_proj.inner()],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                Ok((
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                ))
            }
            // FP (Off): three separate matmuls.
            DynamicActs::Float(_) => Ok((
                q_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                k_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                v_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
            )),
        }
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
    /// B3 producer-fused MoE entry: `acts` is the ln2 output as a `DynamicActs` (q8a128 for int8,
    /// Float for Off). The router consumes it via `forward_dynamic`; the experts byte-gather the
    /// q8a128 directly (no gather-then-quantize). CUDA only.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(&self, acts: DynamicActs, out_dtype: DType) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = match &acts {
            DynamicActs::Float(t) => t.dims3()?,
            DynamicActs::Int8(op) => match op.lead.as_slice() {
                &[b, s] => (b, s, op.cols),
                other => candle::bail!(
                    "SparseMoeBlock::forward_dynamic: expected [b, seq] lead, got {other:?}"
                ),
            },
        };
        let num_tokens = b_size * seq_len;
        let k = self.num_experts_per_tok;
        let t = profile_now();
        let router_logits = self
            .gate
            .forward_dynamic(acts.as_dynamic(), out_dtype)?
            .reshape((num_tokens, ()))?;
        // Logits width = the real expert count; the router kernel writes this
        // value as a "no expert" sentinel into empty top-k slots, so downstream
        // must know it to filter them (see `forward_with_indices`).
        let num_experts = router_logits.dim(1)?;
        let (weights_flat, idx_cpu) = self.route_indices(&router_logits, num_tokens, k, t)?;
        let input = match acts {
            DynamicActs::Float(t2) => MoeInput::Float(t2.reshape((num_tokens, hidden_dim))?),
            DynamicActs::Int8(op) => MoeInput::Q8(op),
        };
        self.forward_with_indices(
            input,
            out_dtype,
            weights_flat,
            idx_cpu,
            b_size,
            seq_len,
            hidden_dim,
            k,
            num_experts,
            t,
        )
    }

    /// Route: GPU softmax + top-k → `(flattened routing weights, per-token expert indices)`.
    /// Used by both the FP and q8a128 arms of `forward_dynamic` — operates only on the logits.
    fn route_indices(
        &self,
        router_logits: &Tensor,
        num_tokens: usize,
        k: usize,
        t: ProfileMark,
    ) -> Result<(Tensor, Vec<Vec<u32>>)> {
        // `num_tokens` drives the CUDA async routing DtoH only; the non-CUDA path uses `to_vec2`.
        #[cfg(not(feature = "cuda"))]
        let _ = num_tokens;
        // Fused routing: softmax + top-k select + (optional) renormalize in a single kernel,
        // replacing the `softmax → sort(desc) → narrow(k) → renorm → flatten` op chain (≈6 launches
        // over a tiny `[num_tokens, 128]` tensor). top-k of softmax == top-k of the logits (softmax
        // is monotonic) and renorm cancels the global softmax denominator, so the kernel makes one
        // pass over the experts. Outputs `[num_tokens, k]` weights (f32) and indices (u32) in
        // descending-logit order — identical to the sort path. We pull only the indices to CPU for
        // scheduling; weights stay GPU-resident.
        #[cfg(feature = "cuda")]
        let (top_k_weights, top_k_indices) = moe_route(router_logits, k, self.norm_topk_prob)?;
        #[cfg(not(feature = "cuda"))]
        let (top_k_weights, top_k_indices) = {
            let routing_weights =
                candle_nn::ops::softmax_last_dim(router_logits)?.to_dtype(DType::F32)?;
            let (sorted_w, sorted_idx) = routing_weights.sort_last_dim(false)?;
            let top_k_weights = sorted_w.narrow(1, 0, k)?; // [num_tokens, k]
            let top_k_indices = sorted_idx.narrow(1, 0, k)?.contiguous()?; // [num_tokens, k] u32
            let top_k_weights = if self.norm_topk_prob {
                let sums = top_k_weights.sum(1)?;
                top_k_weights.broadcast_div(&sums.unsqueeze(1)?)?
            } else {
                top_k_weights
            };
            (top_k_weights, top_k_indices)
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
        let idx_cpu: Vec<Vec<u32>> = if let Device::Cuda(cuda_dev) = router_logits.device() {
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

                        return Ok((weights_flat, idx));
                    }
                } else {
                    drop(storage);
                    let idx = top_k_indices.to_vec2::<u32>()?;
                    self.cache.record_profile("fwd_routing", t);
                    return Ok((weights_flat, idx));
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

        Ok((weights_flat, idx_cpu))
    }

    /// Common path after routing indices are available (sync or async).
    #[allow(clippy::too_many_arguments)]
    fn forward_with_indices(
        &self,
        input: MoeInput,
        out_dtype: DType,
        weights_flat: Tensor,
        idx_cpu: Vec<Vec<u32>>,
        b_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        k: usize,
        num_experts: usize,
        _routing_start: ProfileMark,
    ) -> Result<Tensor> {
        // ── 2. Group assignments by expert via a counting sort ──
        // Each entry: (expert_id, token_idx, flat_weight_idx). Same-expert tokens
        // must be contiguous for the grouped-GEMM dispatch. Expert id is a small
        // bounded integer, so we bucket by it in **O(A + E)** (A = token→expert
        // assignments, E = experts) — no comparison sort — keeping the cost
        // linear even for large prefill batches (a sort here is O(A log A) and
        // scaled badly with the batch size we want for expert-stream amortization).
        let t = profile_now();
        let k_u = k as u32;
        // Bucket count = the router's expert count. The router kernel writes
        // `num_experts` itself as a sentinel into any top-k slot that found no
        // valid expert (a token whose logits were all -inf/NaN), so ids `>=
        // num_experts` are skipped in both passes — they aren't real experts and
        // would index past `num_experts` here and the pipeline's expert arrays.
        let n_experts = num_experts;

        // Pass 1: count assignments per expert (skipping sentinels).
        let mut counts = vec![0u32; n_experts];
        for idxs in &idx_cpu {
            for &eid in idxs {
                if (eid as usize) < n_experts {
                    counts[eid as usize] += 1;
                }
            }
        }
        // Prefix-sum into per-expert bucket starts; collect the ascending active
        // expert ids in the same pass.
        let mut cursor = vec![0u32; n_experts];
        let mut expert_ids: Vec<usize> = Vec::new();
        let mut running = 0u32;
        for (e, &c) in counts.iter().enumerate() {
            cursor[e] = running;
            running += c;
            if c > 0 {
                expert_ids.push(e);
            }
        }
        // Pass 2: scatter each assignment into its expert's bucket (stable in
        // token order) → assignments grouped by ascending expert id, exactly as
        // a sort-by-expert would produce. `slot_k` stays the original top-k
        // position so the flat weight index remains aligned even when a
        // sentinel slot is skipped.
        let num_assignments = running as usize;
        let mut assignments: Vec<(u32, u32, u32)> = vec![(0, 0, 0); num_assignments];
        for (tok, idxs) in idx_cpu.iter().enumerate() {
            let tok_u = tok as u32;
            for (slot_k, &eid) in idxs.iter().enumerate() {
                if (eid as usize) >= n_experts {
                    continue;
                }
                let pos = cursor[eid as usize] as usize;
                assignments[pos] = (eid, tok_u, tok_u * k_u + slot_k as u32);
                cursor[eid as usize] += 1;
            }
        }
        self.cache.record_profile("fwd_cpu_assign", t);

        // Store this layer's expert set for the next layer's speculative hint
        self.cache.set_prev_layer_experts(expert_ids.clone());

        // ── Routing-trace capture (inert unless explicitly enabled) ──
        // Records the active expert set + per-expert routing mass for offline
        // predictor evaluation.  The mass DtoH only happens while capturing.
        if crate::models::routing_capture::is_enabled() {
            if let Ok(w) = weights_flat.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                let mut mass = vec![0f32; expert_ids.len()];
                for &(eid, _tok, widx) in &assignments {
                    if let Ok(pos) = expert_ids.binary_search(&(eid as usize)) {
                        if let Some(&wv) = w.get(widx as usize) {
                            mass[pos] += wv;
                        }
                    }
                }
                crate::models::routing_capture::record(self.moe_layer_idx, &expert_ids, &mass);
            }
        }

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
            input,
            out_dtype,
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

    // ── Producer-fused (q8a128) overrides for the batched paged path ──
    // `int8mode` + `output_projection` (B2) come from the trait defaults (o_proj-driven);
    // only the model-specific producers (ln1/ln2 fusion, q/k-norm qkv, MoE ffn) are overridden.

    /// B1 producer: fuse ln1 → q8a128 (int8) or FP rms_norm (Off) in one kernel.
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

    /// B3: ln2 as a producer epilogue. Only the MoE path emits q8a128 (its router + expert gather
    /// consume it); a dense MLP layer stays FP (it has no int8 grouped path).
    #[cfg(feature = "cuda")]
    fn ffn_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs> {
        match &self.ffn {
            FeedForward::MoE(_) => self.ln2.forward_dynamic(x, mode),
            FeedForward::Mlp(_) => Ok(DynamicActs::Float(self.ln2.forward(x)?)),
        }
    }

    /// B3 consumer: the MoE/MLP over the producer-fused ln2 activation. MoE routes the q8a128 (or
    /// Float) through `forward_dynamic`; a dense MLP runs the FP path (with the stability cast).
    #[cfg(feature = "cuda")]
    fn ffn_forward(&self, acts: DynamicActs, mlp_dtype: DType) -> Result<Tensor> {
        match &self.ffn {
            FeedForward::MoE(m) => {
                // FP acts get the F16→BF16 stability cast; q8a128 is range-safe (no cast).
                let acts = match acts {
                    DynamicActs::Float(t) => DynamicActs::Float(t.to_dtype(mlp_dtype)?),
                    int8 => int8,
                };
                m.forward_dynamic(acts, mlp_dtype)
            }
            FeedForward::Mlp(m) => match acts {
                DynamicActs::Float(t) => m.forward(&t.to_dtype(mlp_dtype)?),
                DynamicActs::Int8(_) => candle::bail!(
                    "dense MLP: int8 activation unsupported (ffn_norm emits Float for Mlp)"
                ),
            },
        }
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
    /// Inference numeric mode for the dense (non-expert) projections. Baked into each dense
    /// `QMatMul` at load; retained here for introspection. Experts are unaffected (always FP16).
    #[allow(dead_code)]
    int8mode: Int8Mode,
    /// Fixed VRAM bytes held by the **base** (non-expert) weights — embeddings,
    /// attention projections, norms, router, lm_head. Measured at load as the
    /// driver-used delta across weight loading, minus the resident-expert
    /// footprint. Added to the live resident-expert gauge to report the model's
    /// total (time-varying) weight VRAM for the whole-card decomposition. `0` on
    /// non-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    base_weight_bytes: usize,
}

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

    fn resident_weight_bytes(&self) -> Option<usize> {
        // Fixed base weights + the live resident-expert footprint (0 when no
        // global expert cache). Rises/falls as experts page VRAM↔pinned RAM.
        let experts = self
            .expert_cache
            .as_ref()
            .map_or(0, |cache| cache.resident_vram_bytes());
        Some(self.base_weight_bytes + experts)
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

            // q/k/v kept separate (no concat): the int8 path fuses them at launch via the
            // segmented kernel, and the FP path runs them as three matmuls.
            let q_proj = Some(QMatMul::from_weights(q_w.into())?);
            let k_proj = Some(QMatMul::from_weights(k_w.into())?);
            let v_proj = Some(QMatMul::from_weights(v_w.into())?);

            let self_attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
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
            // Reader path keeps every projection in FP16; int8 dense repack is only wired on the
            // mmap (`from_gguf_by_path`) load path.
            int8mode: Int8Mode::Off,
            // Reader path: experts live in per-block caches (not the global
            // `expert_cache`), so the whole-card weight decomposition can't
            // attribute them here — left unmeasured.
            base_weight_bytes: 0,
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
        // VRAM-aware auto: int8 Precision on int8-MMA-capable GPUs when the
        // weights leave headroom, else Performance (smaller footprint); FP16 Off
        // on CPU. Sized by the GGUF length. Explicit-mode callers use
        // `from_gguf_by_path_with_int8` instead.
        let model_bytes = std::fs::metadata(file_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let int8mode = Int8Mode::auto_sized(device, model_bytes);
        Self::from_gguf_by_path_with_int8(file_path, device, progress, int8mode)
    }

    /// Like [`ModelWeights::from_gguf_by_path`] but selects the inference numeric `int8mode` for
    /// the whole model — dense projections *and* MoE experts. [`Int8Mode::Off`] is the FP16
    /// reference; an int8 mode repacks every dense weight (attention q/k/v/o, MoE router gate,
    /// dense-MLP gate/up/down, lm_head) to its KO twin so forward runs the q8a128 int8 tensor-core
    /// matmul, and stages each expert's gate/up/down as their KO twins through the [`ExpertCache`]
    /// repack-to-host/DMA pipeline so the grouped expert matmul runs int8 too.
    pub fn from_gguf_by_path_with_int8(
        file_path: &std::path::Path,
        device: &Device,
        progress: Option<&dyn Fn(usize, usize)>,
        int8mode: Int8Mode,
    ) -> Result<Self> {
        use memmap2::MmapOptions;

        tracing::info!(
            "Inference int8 mode: {int8mode:?} (dense projections + MoE experts; \
             GPU int8 m16n8k32 MMA capable: {})",
            Int8Mode::auto(device).is_int8(),
        );

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

        // ── VRAM Governor ──
        // Balloon (fast-path skip on a free card) to measure the real resident
        // capacity C, then install it so the expert budget, KV budget, and
        // scheduler all coordinate through one authority (see
        // `docs/vram_governor_design.md`).
        #[cfg(feature = "cuda")]
        let gpu_id = match device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => 0,
        };
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) && candle::vram::get(gpu_id).is_none() {
            match candle::vram::VramGovernor::from_device(device, gpu_id) {
                Ok(gov) => {
                    let mut balloon =
                        candle::vram::balloon::DeviceBalloonAllocator::new(device.clone());
                    match gov.run_balloon(&mut balloon) {
                        Ok(c) => tracing::info!(
                            target: "candle_core::vram",
                            "VRAM governor installed: capacity C={:.1}GB",
                            c as f64 / 1e9
                        ),
                        Err(e) => tracing::warn!("VRAM governor balloon failed: {e}"),
                    }
                    candle::vram::install(gov);
                }
                Err(e) => tracing::warn!("VRAM governor init failed: {e}"),
            }
        }

        // Driver-used VRAM baseline BEFORE any weights load (the governor's
        // balloon has already been freed). The delta from here to the fully-built
        // model, minus the resident-expert footprint, is the fixed base-weight
        // VRAM the whole-card decomposition reports (see `base_weight_bytes`).
        #[cfg(feature = "cuda")]
        let used_before_weights: usize = if matches!(device, Device::Cuda(_)) {
            get_vram_info()
                .map(|(free, total)| total.saturating_sub(free))
                .unwrap_or(0)
        } else {
            0
        };

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
        // Driver-used VRAM bracketing the expert-cache build, so the base-weight
        // measurement can EXCLUDE experts by construction (base = the driver delta
        // OUTSIDE this bracket). Subtracting the expert gauge instead would cancel
        // against the gauge re-added in `resident_weight_bytes`, collapsing the
        // whole figure back to the raw (governor-balloon-polluted) driver delta.
        #[cfg(feature = "cuda")]
        let used_before_experts = super::batched_model::driver_used_bytes(device);
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
            // Preferred: the VRAM Governor computes it from the live measurement
            // at this instant (weights already resident), leaving the KV floor +
            // scratch cushion free so experts can never starve KV
            // (`docs/vram_governor_design.md` §11). Fallback (no governor): the
            // legacy `min(max(free−5GB, free×50%), total_expert_bytes)`.
            #[cfg(feature = "cuda")]
            let gov_budget = candle::vram::get(gpu_id).and_then(|g| g.expert_budget().ok());
            #[cfg(not(feature = "cuda"))]
            let gov_budget: Option<u64> = None;
            let expert_budget = match gov_budget {
                Some(b) => (b as usize).min(total_expert_bytes),
                None => {
                    const RESERVE_BYTES: usize = 5 * 1024 * 1024 * 1024; // 5 GB
                    let generous = free_vram.saturating_sub(RESERVE_BYTES).max(free_vram / 2);
                    generous.min(total_expert_bytes)
                }
            };

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
            let cache_wrapper =
                progress.map(|cb| move |done: usize, _total: usize| cb(done, total_units));
            let cache_progress: Option<&dyn Fn(usize, usize)> =
                cache_wrapper.as_ref().map(|f| f as &dyn Fn(usize, usize));
            let cache = ExpertCache::new(
                mmap.clone(),
                all_host_refs,
                num_slots,
                device,
                n_expert,
                Some(file_path),
                cache_progress,
                int8mode,
            )?;
            // Record the resident expert footprint with the governor (tally +
            // kv_floor base; the availability gate stays the live measurement).
            #[cfg(feature = "cuda")]
            if let Some(g) = candle::vram::get(gpu_id) {
                g.credit_class(
                    candle::vram::AllocClass::Expert,
                    (num_slots * max_expert_size) as u64,
                );
            }
            Some(Arc::new(cache))
        } else {
            // No MoE layers — warm the entire mmap the simple way.
            warm_mmap(&mmap);
            None
        };
        // Driver-used VRAM right after the expert cache built — the delta from
        // `used_before_experts` is the experts' driver footprint (excluded from base).
        #[cfg(feature = "cuda")]
        let used_after_experts = super::batched_model::driver_used_bytes(device);

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
            let o_proj = QMatMul::from_weights_with_mode(
                load_tensor(&format!("{prefix}.attn_output.weight"))?.into(),
                int8mode,
            )?;
            let q_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_q_norm.weight"))?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_k_norm.weight"))?,
                rms_norm_eps,
            )?;

            // q/k/v kept separate KO twins (no concat): the segmented kernel fuses the three at
            // launch — any KO formats, one occupied launch.
            let q_proj = Some(QMatMul::from_weights_with_mode(q_w.into(), int8mode)?);
            let k_proj = Some(QMatMul::from_weights_with_mode(k_w.into(), int8mode)?);
            let v_proj = Some(QMatMul::from_weights_with_mode(v_w.into(), int8mode)?);

            let self_attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
            };

            // FFN: detect MoE vs dense by checking for expert tensors
            let has_moe_tensors = ct
                .tensor_infos
                .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                || ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

            let ffn = if has_moe_tensors && n_expert > 1 {
                let gate = QMatMul::from_weights_with_mode(
                    load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?.into(),
                    int8mode,
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
                let down_proj = QMatMul::from_weights_with_mode(
                    load_tensor(&format!("{prefix}.ffn_down.weight"))?.into(),
                    int8mode,
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
        let lm_head = QMatMul::from_weights_with_mode(lm_head_tensor.into(), int8mode)?;

        tracing::debug!(
            "Model loaded: {} layers ({} MoE), int8mode={:?}",
            num_layers,
            moe_count,
            int8mode
        );

        // Base-weight VRAM = the driver-used growth OUTSIDE the expert-cache
        // bracket: embeddings + rotary/misc (before the cache) plus attention +
        // norms + router + lm_head (the layer loop, after it). Experts are
        // excluded by construction — `resident_weight_bytes` adds the live expert
        // gauge back on top, so nothing cancels. Fixed for the session (dense
        // weights never move); the experts are the only time-varying part.
        #[cfg(feature = "cuda")]
        let base_weight_bytes: usize = if matches!(device, Device::Cuda(_)) {
            let used_after = get_vram_info()
                .map(|(free, total)| total.saturating_sub(free))
                .unwrap_or(0);
            let pre_experts = used_before_experts.saturating_sub(used_before_weights);
            let post_experts = used_after.saturating_sub(used_after_experts);
            let base = pre_experts + post_experts;
            let expert_driver = used_after_experts.saturating_sub(used_before_experts);
            let expert_gauge = expert_cache.as_ref().map_or(0, |c| c.resident_vram_bytes());
            tracing::info!(
                target: "candle_transformers::quantized_qwen3_moe",
                base_gib = base as f64 / 1e9,
                expert_driver_gib = expert_driver as f64 / 1e9,
                expert_gauge_gib = expert_gauge as f64 / 1e9,
                "weight VRAM breakdown (base = non-expert driver delta; experts from gauge)"
            );
            base
        } else {
            0
        };
        #[cfg(not(feature = "cuda"))]
        let base_weight_bytes: usize = 0;

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
            int8mode,
            base_weight_bytes,
        })
    }

    /// Create KV caches for all layers.
    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = (0..self.layers.len())
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
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
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
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

        // Download tokenizer (hf_get falls back from IPv6 to IPv4 if needed).
        use crate::models::batch_test::test_helpers::hf_get;
        let tokenizer_path = hf_get(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )
        .map_err(|e| {
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

        let model_path = hf_get(
            "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
            hf_hub::RepoType::Model,
            "main",
            "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        )
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
                num_contexts: 48,
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
            // BF16 single context (after everything is warm)
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // Q8 (after everything is warm)
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                num_contexts: 20,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::Skip),
            },
        ];

        use crate::models::batched_model::BatchedInference;

        // Inference numeric mode for the whole model — dense projections AND MoE experts (KO
        // twins) — selected by the INT8MODE env var so a run picks a mode without recompiling.
        // Defaults to Performance (same-width KO int8); override with "off" (FP16 reference) or
        // "prec"/"precision" (stepped-up, near-lossless KO int8). One model load: switching mode
        // means a fresh load, which is correct here — it keeps the Markov expert predictor from
        // being mixed across modes.
        let int8mode = match std::env::var("INT8MODE").ok().as_deref() {
            Some("off") => Int8Mode::Off,
            Some("prec") | Some("precision") => Int8Mode::Precision,
            _ => Int8Mode::Performance,
        };
        println!("int8 mode = {int8mode:?}\n");

        let load_model = || {
            let model =
                ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, None, int8mode)?;
            println!("✓ Model loaded\n");
            let inv_freq = model
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    /// Continuous-fair-wave equivalence gate (`docs/continuous_fair_waves.md`):
    /// the co-batched `forward_wave` (decode + prefill through the mixed dispatch +
    /// shared MoE) must produce the SAME logits as running decode and prefill as
    /// separate forwards, and the re-entrant `forward_batched_layers` split sweep
    /// must match a full sweep. GPU + a real model; ignored so it never runs in the
    /// normal (fast) suite.
    ///
    /// Run with:
    ///   cargo test -p candle-transformers --release --features cuda --lib \
    ///     quantized_qwen3_moe::tests::wave_equivalence -- --ignored --nocapture
    #[test]
    #[ignore]
    fn wave_equivalence() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ wave_equivalence requires --features cuda");
            Ok(())
        }
        #[cfg(feature = "cuda")]
        {
            use crate::models::batch_test::test_helpers::hf_get;
            use crate::models::batched_inference::{
                BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveStep,
            };
            use crate::models::batched_model::BatchedInference;

            let device = match Device::new_cuda(0) {
                Ok(d) => d,
                Err(_) => {
                    println!("skip: no CUDA device");
                    return Ok(());
                }
            };
            let model_path = hf_get(
                "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                hf_hub::RepoType::Model,
                "main",
                "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            )
            .map_err(|e| candle::Error::Msg(format!("model download: {e}")))?;
            let raw = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
            let inv_freq = raw
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            let model = BatchedInference::new_with_inv_freq(raw, inv_freq, 4096, &device)?;
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let n = model.num_layers();

            let mk = |t: &[u32]| -> Result<Tensor> { Tensor::new(t, &device)?.unsqueeze(0) };
            // Cosine of two logit rows, robust to tiny per-kernel fp reordering.
            let cos = |a: &Tensor, b: &Tensor| -> Result<f32> {
                let a = a.flatten_all()?.to_dtype(DType::F32)?;
                let b = b.flatten_all()?.to_dtype(DType::F32)?;
                let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
                let na = (&a * &a)?.sum_all()?.to_scalar::<f32>()?.sqrt();
                let nb = (&b * &b)?.sum_all()?.to_scalar::<f32>()?.sqrt();
                Ok(dot / (na * nb + 1e-6))
            };
            // Single-group reference forward through the unified wave entry: one
            // prefill group, full sweep. `forward_wave` reclassifies q=1 rows into
            // the decode group internally, so this is exactly the old `forward_batched`.
            let fb = |session: &mut BatchedInferenceSession,
                      seqs: &[usize],
                      inputs: &[Tensor]|
             -> Result<Vec<Tensor>> {
                Ok(model
                    .forward_wave(session, &[], &[], seqs, inputs, &[], &[], 0, n, None)?
                    .logits
                    .unwrap_or_default())
            };
            // Layer-range reference through the wave entry (the retired
            // `forward_batched_layers`): prefill group, layers `[ls, le)`, residual.
            let fbl = |session: &mut BatchedInferenceSession,
                       seqs: &[usize],
                       inputs: &[Tensor],
                       ls: usize,
                       le: usize,
                       residual: Option<Tensor>|
             -> Result<WaveStep> {
                model.forward_wave(session, &[], &[], seqs, inputs, &[], &[], ls, le, residual)
            };
            // Prefill an identical `ctx` context into `s`, leaving offset at ctx.len().
            let prep =
                |session: &mut BatchedInferenceSession, s: usize, ctx: &[u32]| -> Result<()> {
                    let _ = fb(session, &[s], &[mk(ctx)?])?;
                    session.advance_sequence(s, ctx.len())?;
                    Ok(())
                };

            let ctx_d: Vec<u32> = (100..=115u32).collect();
            let ctx_p: Vec<u32> = (200..=215u32).collect();
            let dec_tok = [42u32];
            let pre_tok = [51u32, 52, 53, 54, 55];

            // ── Test 1: co-batch (forward_wave) == separate forwards ──────────
            let d0 = session.create_sequence()?;
            let d1 = session.create_sequence()?;
            let p0 = session.create_sequence()?;
            let p1 = session.create_sequence()?;
            prep(&mut session, d0, &ctx_d)?;
            prep(&mut session, d1, &ctx_d)?;
            prep(&mut session, p0, &ctx_p)?;
            prep(&mut session, p1, &ctx_p)?;

            let sep_dec = fb(&mut session, &[d0], &[mk(&dec_tok)?])?;
            let sep_pre = fb(&mut session, &[p0], &[mk(&pre_tok)?])?;
            let wave = model
                .forward_wave(
                    &mut session,
                    &[d1],
                    &[mk(&dec_tok)?],
                    &[p1],
                    &[mk(&pre_tok)?],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits
                .expect("full-range wave must produce logits");
            assert_eq!(wave.len(), 2, "wave logits = decode + prefill rows");
            let c_dec = cos(&sep_dec[0], &wave[0])?;
            let c_pre = cos(&sep_pre[0], &wave[1])?;
            println!("forward_wave vs separate: decode cos={c_dec:.5} prefill cos={c_pre:.5}");
            assert!(c_dec > 0.999, "decode logits diverged (cos={c_dec})");
            assert!(c_pre > 0.999, "prefill logits diverged (cos={c_pre})");

            // ── Test 2: re-entrant split sweep == full sweep ──────────────────
            let a = session.create_sequence()?;
            let b = session.create_sequence()?;
            prep(&mut session, a, &ctx_p)?;
            prep(&mut session, b, &ctx_p)?;
            let full = fbl(&mut session, &[a], &[mk(&pre_tok)?], 0, n, None)?
                .logits
                .expect("full sweep logits");
            let k = n / 2;
            let mid = fbl(&mut session, &[b], &[mk(&pre_tok)?], 0, k, None)?
                .residual
                .expect("paused sweep must return a residual");
            let split = fbl(&mut session, &[b], &[mk(&pre_tok)?], k, n, Some(mid))?
                .logits
                .expect("resumed sweep logits");
            let c_split = cos(&full[0], &split[0])?;
            println!("split sweep [0,{k})+[{k},{n}) vs full: cos={c_split:.5}");
            assert!(c_split > 0.999, "re-entrant sweep diverged (cos={c_split})");

            // ── Test 3a: decode-only wave with D>1 (flat single decode group) ──
            // Guards the mixed-dispatch fix: a >1-sequence decode wave packs flat
            // [1,D,h] and must be reshaped to the decode kernel's [D,1,h] and
            // routed by header type, not by dim(1). (D>1 wasn't covered by Test 1.)
            let e0 = session.create_sequence()?;
            let e1 = session.create_sequence()?;
            let e2 = session.create_sequence()?;
            let e3 = session.create_sequence()?;
            for &s in &[e0, e1, e2, e3] {
                prep(&mut session, s, &ctx_d)?;
            }
            let sep_d2 = fb(&mut session, &[e0, e1], &[mk(&dec_tok)?, mk(&dec_tok)?])?;
            let wave_d2 = model
                .forward_wave(
                    &mut session,
                    &[e2, e3],
                    &[mk(&dec_tok)?, mk(&dec_tok)?],
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits
                .expect("decode-only wave logits");
            assert_eq!(wave_d2.len(), 2, "decode-only wave = 2 decode rows");
            let c_d0 = cos(&sep_d2[0], &wave_d2[0])?;
            let c_d1 = cos(&sep_d2[1], &wave_d2[1])?;
            println!("multi-decode wave vs separate: cos={c_d0:.5},{c_d1:.5}");
            assert!(c_d0 > 0.999 && c_d1 > 0.999, "multi-decode wave diverged");

            // ── Test 3b: co-batch a 1-token PREFILL ───────────────────────────
            // A single-token prefill is operationally a decode; the wave routes it
            // through the DECODE kernel (the canonical single-token path — the paged
            // prefill kernel diverges for q_len==1), so it must match the separate
            // 1-token forward, which `forward_batched` also decodes.
            let g0 = session.create_sequence()?;
            let g1 = session.create_sequence()?;
            let dg = session.create_sequence()?;
            let dg2 = session.create_sequence()?;
            prep(&mut session, g0, &ctx_p)?;
            prep(&mut session, g1, &ctx_p)?;
            prep(&mut session, dg, &ctx_d)?;
            prep(&mut session, dg2, &ctx_d)?;
            let one_tok = [61u32];
            let sep_dec1 = fb(&mut session, &[dg], &[mk(&dec_tok)?])?;
            let sep_pre1 = fb(&mut session, &[g0], &[mk(&one_tok)?])?;
            let wave_1 = model
                .forward_wave(
                    &mut session,
                    &[dg2],
                    &[mk(&dec_tok)?],
                    &[g1],
                    &[mk(&one_tok)?],
                    &[],
                    &[],
                    0,
                    n,
                    None,
                )?
                .logits
                .expect("co-batch wave logits");
            assert_eq!(wave_1.len(), 2, "co-batch = decode + 1-token prefill");
            let c_1d = cos(&sep_dec1[0], &wave_1[0])?;
            let c_1p = cos(&sep_pre1[0], &wave_1[1])?;
            println!("1-token-prefill co-batch vs separate: decode={c_1d:.5} prefill={c_1p:.5}");
            assert!(c_1d > 0.999, "co-batch decode diverged (cos={c_1d})");
            assert!(c_1p > 0.999, "1-token prefill misrouted (cos={c_1p})");

            // ── Test 3c: a RAGGED forward_batched batch mixing a multi-token and a
            // 1-token sequence. `max_input_len>1` would route the whole batch as a
            // prefill, sending the 1-token row through the divergent prefill kernel;
            // the mixed-batch delegation must route it through the decode kernel so
            // each row matches its homogeneous single-sequence reference.
            let r0 = session.create_sequence()?;
            let r1 = session.create_sequence()?;
            let r0s = session.create_sequence()?;
            let r1s = session.create_sequence()?;
            prep(&mut session, r0, &ctx_p)?;
            prep(&mut session, r1, &ctx_d)?;
            prep(&mut session, r0s, &ctx_p)?;
            prep(&mut session, r1s, &ctx_d)?;
            // References: each row run alone (r0s = multi-token prefill kernel, r1s
            // = 1-token → decode kernel via `max_input_len==1`).
            let ref_multi = fb(&mut session, &[r0s], &[mk(&pre_tok)?])?;
            let ref_one = fb(&mut session, &[r1s], &[mk(&one_tok)?])?;
            // Mixed ragged batch [5-token, 1-token] in ONE wave prefill group.
            let mixed = fb(&mut session, &[r0, r1], &[mk(&pre_tok)?, mk(&one_tok)?])?;
            assert_eq!(mixed.len(), 2, "ragged batch = one logit row per sequence");
            let c_m = cos(&ref_multi[0], &mixed[0])?;
            let c_s = cos(&ref_one[0], &mixed[1])?;
            println!("ragged mixed-batch vs separate: multi={c_m:.5} single={c_s:.5}");
            assert!(c_m > 0.999, "ragged multi-token row diverged (cos={c_m})");
            assert!(c_s > 0.999, "ragged 1-token row misrouted (cos={c_s})");

            // ── Test 3d: a MIXED cohort split across TWO layer windows. This is the
            // re-entrant residual path with reclassification: single-token prefills
            // are folded into the decode group internally, but the residual crosses
            // the API in CALLER order — `forward_wave` reorders caller→internal on
            // resume, so re-feeding the returned residual round-trips. A broken
            // reorder would corrupt the whole cohort. Split-sweep logits must equal
            // the full-sweep of the same mixed batch (and each row its solo reference).
            let w0 = session.create_sequence()?;
            let w1 = session.create_sequence()?;
            prep(&mut session, w0, &ctx_p)?;
            prep(&mut session, w1, &ctx_d)?;
            let kk = n / 2;
            let mixed_res = fbl(
                &mut session,
                &[w0, w1],
                &[mk(&pre_tok)?, mk(&one_tok)?],
                0,
                kk,
                None,
            )?
            .residual
            .expect("paused mixed cohort must return a residual");
            let mixed_split = fbl(
                &mut session,
                &[w0, w1],
                &[mk(&pre_tok)?, mk(&one_tok)?],
                kk,
                n,
                Some(mixed_res),
            )?
            .logits
            .expect("resumed mixed cohort must produce logits");
            assert_eq!(mixed_split.len(), 2, "split cohort = one logit row per seq");
            let c_sm = cos(&ref_multi[0], &mixed_split[0])?;
            let c_ss = cos(&ref_one[0], &mixed_split[1])?;
            println!("mixed cohort split-sweep vs separate: multi={c_sm:.5} single={c_ss:.5}");
            assert!(
                c_sm > 0.999,
                "split mixed multi-token row diverged (cos={c_sm})"
            );
            assert!(
                c_ss > 0.999,
                "split mixed 1-token row corrupted on resume (cos={c_ss})"
            );

            // ── Test 3e: DECODE + a MULTI-MEMBER GROUP co-batched, then the
            // combined `[decode | group]` residual split by `narrow` at n_dec and
            // each part resumed INDEPENDENTLY — exactly the residual bookkeeping in
            // `decode_forward_cobatched` when section chunks creep alongside the
            // prefill cohort (group = [prefill-multi, section-multi, section-1tok]).
            // The residual is in CALLER order `[decode | group]`, so `narrow(0, n_dec)`
            // is the decode part and `narrow(n_dec, ..)` the group part (held WHOLE,
            // never split within) — even though the 1-token member is folded into the
            // decode kernel internally, caller order keeps it inside the group span.
            // Every resumed row must equal its solo full-sweep — a mis-split would
            // corrupt decode or the ingest.
            let kk2 = n / 2;
            // Solo references (each row alone, full sweep) on their own sequences.
            let rda = session.create_sequence()?;
            let rga = session.create_sequence()?;
            let rgb = session.create_sequence()?;
            let rgc = session.create_sequence()?;
            for &s in &[rda, rga, rgb, rgc] {
                prep(&mut session, s, &ctx_d)?;
            }
            let sref_d = fb(&mut session, &[rda], &[mk(&dec_tok)?])?;
            let sref_a = fb(&mut session, &[rga], &[mk(&pre_tok)?])?;
            let sref_b = fb(&mut session, &[rgb], &[mk(&pre_tok)?])?;
            let sref_c = fb(&mut session, &[rgc], &[mk(&one_tok)?])?;
            // Co-batch decode [da] + group [ga, gb, gc], paused at kk2.
            let da = session.create_sequence()?;
            let ga = session.create_sequence()?;
            let gb = session.create_sequence()?;
            let gc = session.create_sequence()?;
            for &s in &[da, ga, gb, gc] {
                prep(&mut session, s, &ctx_d)?;
            }
            let combined = model
                .forward_wave(
                    &mut session,
                    &[da],
                    &[mk(&dec_tok)?],
                    &[ga, gb, gc],
                    &[mk(&pre_tok)?, mk(&pre_tok)?, mk(&one_tok)?],
                    &[],
                    &[],
                    0,
                    kk2,
                    None,
                )?
                .residual
                .expect("paused co-batch must return a residual");
            // Split: 1 decode token, then the group's tokens (held whole).
            let gtok = pre_tok.len() + pre_tok.len() + one_tok.len();
            let dec_part = combined.narrow(1, 0, 1)?;
            let grp_part = combined.narrow(1, 1, gtok)?;
            // Resume decode alone to N.
            let dec_fin = model
                .forward_wave(
                    &mut session,
                    &[da],
                    &[mk(&dec_tok)?],
                    &[],
                    &[],
                    &[],
                    &[],
                    kk2,
                    n,
                    Some(dec_part),
                )?
                .logits
                .expect("decode resume logits");
            // Resume the group alone to N (members re-passed in the SAME order).
            let grp_fin = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[ga, gb, gc],
                    &[mk(&pre_tok)?, mk(&pre_tok)?, mk(&one_tok)?],
                    &[],
                    &[],
                    kk2,
                    n,
                    Some(grp_part),
                )?
                .logits
                .expect("group resume logits");
            assert_eq!(dec_fin.len(), 1, "decode resume = one row");
            assert_eq!(grp_fin.len(), 3, "group resume = one row per member");
            let c_ed = cos(&sref_d[0], &dec_fin[0])?;
            let c_ea = cos(&sref_a[0], &grp_fin[0])?;
            let c_eb = cos(&sref_b[0], &grp_fin[1])?;
            let c_ec = cos(&sref_c[0], &grp_fin[2])?;
            println!(
                "co-batch split-residual resume: decode={c_ed:.5} group=[{c_ea:.5},{c_eb:.5},{c_ec:.5}]"
            );
            assert!(
                c_ed > 0.999,
                "decode part corrupted by residual split (cos={c_ed})"
            );
            assert!(
                c_ea > 0.999 && c_eb > 0.999 && c_ec > 0.999,
                "group member corrupted by residual split (cos=[{c_ea},{c_eb},{c_ec}])"
            );

            // ── Test 3f: the THREE-group co-batch — decode + creep group + a GLUE
            // group — paused mid-sweep and its `[decode | creep | glue]` residual
            // split THREE ways, each part resumed independently. This is the unified
            // wave step's full geometry (`decode_forward_cobatched` folding deferred
            // glue as a full-sweep member): glue rides the caller-order TAIL, so the
            // creep still narrows contiguously between decode and glue. The creep
            // carries a 1-token member (folded into decode internally) to exercise
            // the caller-order reorder under a trailing glue group. (Here the glue
            // group runs as a plain ragged group — no pending scatter descriptors —
            // which is enough to validate the residual geometry; the glue kernel's
            // scatter/mask is covered by the gap-fill tests.) Every resumed row must
            // equal its solo full-sweep.
            let kk3 = n / 2;
            let hda = session.create_sequence()?;
            let hca = session.create_sequence()?;
            let hcb = session.create_sequence()?;
            let hgl = session.create_sequence()?;
            for &s in &[hda, hca, hcb, hgl] {
                prep(&mut session, s, &ctx_d)?;
            }
            // Solo references (each row alone, full sweep).
            let href_d = fb(&mut session, &[hda], &[mk(&dec_tok)?])?;
            let href_ca = fb(&mut session, &[hca], &[mk(&pre_tok)?])?;
            let href_cb = fb(&mut session, &[hcb], &[mk(&one_tok)?])?;
            let href_gl = fb(&mut session, &[hgl], &[mk(&pre_tok)?])?;
            // Co-batch decode [da] + creep [ca(multi), cb(1tok)] + glue [gl(multi)],
            // paused at kk3. Residual crosses in caller order [decode | creep | glue].
            let combined3 = model
                .forward_wave(
                    &mut session,
                    &[hda],
                    &[mk(&dec_tok)?],
                    &[hca, hcb],
                    &[mk(&pre_tok)?, mk(&one_tok)?],
                    &[hgl],
                    &[mk(&pre_tok)?],
                    0,
                    kk3,
                    None,
                )?
                .residual
                .expect("paused three-group co-batch must return a residual");
            let creep_tok3 = pre_tok.len() + one_tok.len();
            let glue_tok3 = pre_tok.len();
            let dec_part3 = combined3.narrow(1, 0, 1)?;
            let creep_part3 = combined3.narrow(1, 1, creep_tok3)?;
            let glue_part3 = combined3.narrow(1, 1 + creep_tok3, glue_tok3)?;
            // Resume each group alone to N, members re-passed in the SAME order.
            let dec_fin3 = model
                .forward_wave(
                    &mut session,
                    &[hda],
                    &[mk(&dec_tok)?],
                    &[],
                    &[],
                    &[],
                    &[],
                    kk3,
                    n,
                    Some(dec_part3),
                )?
                .logits
                .expect("decode resume logits");
            let creep_fin3 = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[hca, hcb],
                    &[mk(&pre_tok)?, mk(&one_tok)?],
                    &[],
                    &[],
                    kk3,
                    n,
                    Some(creep_part3),
                )?
                .logits
                .expect("creep resume logits");
            let glue_fin3 = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[hgl],
                    &[mk(&pre_tok)?],
                    &[],
                    &[],
                    kk3,
                    n,
                    Some(glue_part3),
                )?
                .logits
                .expect("glue resume logits");
            let c_fd = cos(&href_d[0], &dec_fin3[0])?;
            let c_fca = cos(&href_ca[0], &creep_fin3[0])?;
            let c_fcb = cos(&href_cb[0], &creep_fin3[1])?;
            let c_fgl = cos(&href_gl[0], &glue_fin3[0])?;
            println!(
                "three-group split-residual resume: decode={c_fd:.5} creep=[{c_fca:.5},{c_fcb:.5}] glue={c_fgl:.5}"
            );
            assert!(c_fd > 0.999, "decode corrupted by 3-way split (cos={c_fd})");
            assert!(
                c_fca > 0.999 && c_fcb > 0.999,
                "creep member corrupted by 3-way split (cos=[{c_fca},{c_fcb}])"
            );
            assert!(
                c_fgl > 0.999,
                "glue-tail member corrupted by 3-way split (cos={c_fgl})"
            );

            Ok(())
        }
    }

    /// Per-op decode profiler: prefill a realistic context, then time N single-token
    /// decode steps and dump the `snapshot_profiles` op breakdown so we can see where
    /// the per-token forward spends its time (attention kernel vs qkv/o_proj vs the
    /// MoE `fwd_routing_wait` host sync vs the expert path). Iterates in ~1 min so
    /// it drives forward-latency optimization without the daemon's substrate reload.
    ///
    /// Run with the profiler enabled:
    ///   cargo test -p candle-transformers --release --features cuda,profile --lib \
    ///     quantized_qwen3_moe::tests::decode_profile -- --ignored --nocapture
    #[test]
    #[ignore]
    fn decode_profile() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ decode_profile requires --features cuda,profile");
            Ok(())
        }
        #[cfg(feature = "cuda")]
        {
            use crate::models::batch_test::test_helpers::hf_get;
            use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
            use crate::models::batched_model::BatchedInference;

            let device = match Device::new_cuda(0) {
                Ok(d) => d,
                Err(_) => {
                    println!("skip: no CUDA device");
                    return Ok(());
                }
            };
            let model_path = hf_get(
                "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                hf_hub::RepoType::Model,
                "main",
                "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            )
            .map_err(|e| candle::Error::Msg(format!("model download: {e}")))?;
            let raw = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
            let inv_freq = raw
                .rope_inv_freq()
                .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
            let model = BatchedInference::new_with_inv_freq(raw, inv_freq, 4096, &device)?;
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let mk = |t: &[u32]| -> Result<Tensor> { Tensor::new(t, &device)?.unsqueeze(0) };

            // Batch-size sweep to expose scaling: decode B sequences per step for
            // B in {1, 4, 16}. If the forward is launch-bound, ms/STEP barely grows
            // with B (launches amortize) → ms/token drops ~B× → near-linear tok/s.
            // Sub-linear tok/s ⇒ per-session (non-amortized) work dominates.
            let ctx: Vec<u32> = (0..3000u32).map(|i| (i % 2900) + 100).collect();
            let n_steps = 60usize;
            let nl = model.num_layers();
            let mut baseline_ms_step = 0.0f64;
            for (bi, &batch) in [1usize, 4, 16].iter().enumerate() {
                let seqs: Vec<usize> = (0..batch)
                    .map(|_| session.create_sequence())
                    .collect::<Result<_>>()?;
                for &sq in &seqs {
                    // Context prefill (q>1): prefill group.
                    let _ = model.forward_wave(
                        &mut session,
                        &[],
                        &[],
                        &[sq],
                        &[mk(&ctx)?],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    session.advance_sequence(sq, ctx.len())?;
                }
                let inputs =
                    |t: u32| -> Result<Vec<Tensor>> { (0..batch).map(|_| mk(&[t])).collect() };
                let mut tok = 42u32;
                for _ in 0..6 {
                    // Decode step (q=1): decode group — the path this benchmark times.
                    let _ = model.forward_wave(
                        &mut session,
                        &seqs,
                        &inputs(tok)?,
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    for &sq in &seqs {
                        session.advance_sequence(sq, 1)?;
                    }
                    tok = (tok + 1) % 2900 + 100;
                }
                let _ = model.snapshot_profiles();
                let _ = crate::models::profile::pipeline_snapshot_and_reset();
                let t0 = std::time::Instant::now();
                for _ in 0..n_steps {
                    let _ = model.forward_wave(
                        &mut session,
                        &seqs,
                        &inputs(tok)?,
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    for &sq in &seqs {
                        session.advance_sequence(sq, 1)?;
                    }
                    tok = (tok + 1) % 2900 + 100;
                }
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ms_step = total_ms / n_steps as f64;
                let ms_tok = ms_step / batch as f64;
                let tok_s = 1000.0 / ms_tok;
                if bi == 0 {
                    baseline_ms_step = ms_step;
                }
                let scale = if bi == 0 {
                    1.0
                } else {
                    (1000.0 / ms_tok) / (1000.0 / (baseline_ms_step)) // tok/s vs B=1 tok/s
                };
                println!(
                    "B={batch:2}: {ms_step:6.2} ms/step | {ms_tok:5.2} ms/token | {tok_s:7.0} tok/s | scale×{scale:.1} (ideal ×{batch})"
                );
                // Per-op breakdown only for B=1 (needs --features profile).
                if bi == 0 {
                    let mut snap = model.snapshot_profiles();
                    snap.merge(&crate::models::profile::pipeline_snapshot_and_reset());
                    let mut rows = snap.entries.clone();
                    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (name, ms, cnt) in rows.iter().take(8) {
                        println!("    {name:24} {:6.3} ms/tok  x{cnt}", ms / n_steps as f64);
                    }
                }
            }
            Ok(())
        }
    }

    /// Capture a real MoE routing trace for offline predictor evaluation.
    ///
    /// Runs a single BF16×1 StoryRewrite generation against Qwen3-30B-A3B with
    /// routing capture enabled, then bincode + gzip the trace to the checked-in
    /// fixture ([`routing_capture::FIXTURE_PATH`]).  The offline eval
    /// (`expert_lre::eval`) replays that fixture on CPU in milliseconds.
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::capture_routing_trace -- --ignored --nocapture
    #[test]
    #[ignore]
    fn capture_routing_trace() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ This test requires --features cuda");
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        {
            use crate::models::routing_capture;
            use std::io::Write;

            println!("\n=== Capturing Qwen3-30B-A3B MoE routing trace ===\n");

            let dialect = Dialect::chat_ml();
            use crate::models::batch_test::test_helpers::hf_get;
            let tokenizer_path = hf_get(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                hf_hub::RepoType::Model,
                "main",
                "tokenizer.json",
            )
            .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer.json: {}", e)))?;
            let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
                .map_err(|e| candle::Error::Msg(format!("Failed to read tokenizer.json: {}", e)))?;

            // Diverse prompts driven through a single model load.  Config 0
            // keeps the default StoryRewrite prompt; the rest override it.
            let prompts: Vec<String> = [
                "", // config 0 → story.md (StoryRewrite default)
                "Tell me about the candle repository and how it works",
                "How does AI implement inference in modern engines",
                "Output me a rust program example the is a basic API skeleton",
                "Write example code for a bubble sort of Int64 numbers",
                "Give me a list of jobs that are safe from AI automation",
                "Explain how a B-tree database index works and when to use one",
                "Write a Python function that merges two sorted linked lists",
                "What caused the fall of the Western Roman Empire?",
                "Summarize the key differences between TCP and UDP",
                "Write a haiku about a thunderstorm over the ocean",
                "Implement a least-recently-used cache in Rust with a fixed capacity",
                "Describe how photosynthesis converts sunlight into chemical energy",
                "Compare and contrast supervised and unsupervised machine learning",
                "Give me a step-by-step recipe for a classic margherita pizza",
                "Explain the difference between mutexes and semaphores in concurrency",
                "What are the tradeoffs between monolithic and microservice architectures?",
                "Write a SQL query to find the second highest salary in an employees table",
                "Explain how vaccines train the human immune system",
                "Describe the plot of a short mystery story set on a moving train",
                "List strategies for reducing tail latency in a distributed web service",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();

            // Stop each generation at EOS to avoid post-EOS degenerate routing.
            let eos_tokens: Vec<u32> = {
                let tok = tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).unwrap();
                ["<|im_end|>", "<|endoftext|>"]
                    .iter()
                    .filter_map(|s| tok.token_to_id(s))
                    .collect()
            };

            let params = TestParams::new(256, &tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
                .with_suppress_thinking(true)
                .with_print_outputs(false)
                .with_timeout_secs(3600)
                .with_per_config_prompts(prompts.clone())
                .with_stop_on_eos(eos_tokens);

            let model_path = hf_get(
                "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                hf_hub::RepoType::Model,
                "main",
                "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            )
            .map_err(|e| candle::Error::Msg(format!("Failed to download model: {}", e)))?;

            let device = Device::new_cuda(0)
                .map_err(|e| candle::Error::Msg(format!("CUDA required: {}", e)))?;

            // One config per prompt: BF16, single context, no validation.
            let configs: Vec<TestConfig> = (0..prompts.len())
                .map(|_| TestConfig {
                    mode: InferenceMode::BF16,
                    use_batched: true,
                    num_contexts: 1,
                    num_repeats: 1,
                    generate_max_len: 256,
                    test_mode: Some(TestMode::Skip),
                })
                .collect();

            use crate::models::batched_model::BatchedInference;

            let load_model = || {
                let model = ModelWeights::from_gguf_by_path(&model_path, &device, None)?;
                let inv_freq = model
                    .rope_inv_freq()
                    .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
                BatchedInference::new_with_inv_freq(model, inv_freq, 4096, &device)
            };

            // Capture every config (each is a distinct prompt, tagged by index).
            // Validation outcome is irrelevant here — we only want the routing
            // trace, so a harness validation error must not lose captured data.
            routing_capture::enable_all();
            let run_result = params.run(configs, load_model);
            routing_capture::disable();
            if let Err(e) = &run_result {
                println!("(harness reported: {e} — ignored; writing captured trace)");
            }

            let records = routing_capture::take();
            println!("Captured {} routing records", records.len());
            assert!(!records.is_empty(), "no routing records captured");

            // bincode → gzip → fixture.
            let raw = bincode::serialize(&records)
                .map_err(|e| candle::Error::Msg(format!("bincode serialize failed: {}", e)))?;
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder
                .write_all(&raw)
                .map_err(|e| candle::Error::Msg(format!("gzip write failed: {}", e)))?;
            let compressed = encoder
                .finish()
                .map_err(|e| candle::Error::Msg(format!("gzip finish failed: {}", e)))?;

            let path = std::path::Path::new(routing_capture::FIXTURE_PATH);
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| candle::Error::Msg(format!("mkdir fixtures failed: {}", e)))?;
            }
            std::fs::write(path, &compressed)
                .map_err(|e| candle::Error::Msg(format!("write fixture failed: {}", e)))?;

            println!(
                "Wrote {} records → {} ({} bytes raw, {} bytes gzip)",
                records.len(),
                routing_capture::FIXTURE_PATH,
                raw.len(),
                compressed.len(),
            );
            Ok(())
        }
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
            let api = crate::models::batch_test::test_helpers::api()
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
            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[seq_idx],
                    &[prefill_input],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits
                .unwrap_or_default();
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
                let out = model
                    .forward_wave(
                        &mut session,
                        &[seq_idx],
                        &[input],
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?
                    .logits
                    .unwrap_or_default();
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
