//! Batched transformer model processing for continuous batching.
//!
//! This module provides the [`BatchedModelCore`] trait and [`BatchedInference`] wrapper
//! for model-level batched inference. For layer-level primitives, see [`super::batched_layer`].
//!
//! # Design
//!
//! The design separates concerns:
//! - [`BatchedModelCore`] - Simple trait with just accessor methods (easy to implement)
//! - [`BatchedInference`] - Concrete wrapper that owns RoPE cache and implements forward_batch
//!
//! This avoids duplicating RoPE caching logic across models.
//!
//! # Usage
//!
//! ```ignore
//! // 1. Implement the simple trait for your model
//! impl BatchedModelCore for MyModel {
//!     type Layer = MyLayerWeights;  // must implement BatchedAttentionLayer
//!     
//!     fn num_layers(&self) -> usize { self.layers.len() }
//!     fn n_kv_head(&self) -> usize { self.layers[0].n_kv_head }
//!     fn head_dim(&self) -> usize { self.layers[0].head_dim }
//!     fn device(&self) -> &Device { &self.device }
//!     fn embeddings(&self) -> &Embedding { &self.embeddings }
//!     fn layer(&self, idx: usize) -> &Self::Layer { &self.layers[idx] }
//!     fn final_norm(&self) -> &RmsNorm { &self.norm }
//!     fn output_proj(&self) -> &QMatMul { &self.output }
//! }
//!
//! // 2. Wrap with BatchedInference
//! let model = MyModel::from_gguf(...)?;
//! let batched = BatchedInference::new(model, 10000.0, 4096, &device)?;
//!
//! // 3. Use batched inference
//! let logits = batched.forward_batch(&mut contexts)?;
//! ```

use std::sync::RwLock;

use candle::quantized::pinned_staging::Generation;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::Module;

use super::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionLayer, BatchedAttentionParams, DecodeHeaders,
    WaveAttnGroup,
};
use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use super::kv_cache_utils::SequenceContext;
use super::prefill_utils::SharedPm;
use super::profile::{pipeline_record, profile_now, profile_sync};
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use super::tensor_cat::TensorCat;
use crate::quantized_nn::RmsNorm;
use candle_nn::Embedding;

/// Outcome of a re-entrant [`BatchedInference::forward_batch_layers`] call.
///
/// A wave runs a contiguous layer range. If it stopped short of the final layer
/// it yields the inter-layer residual stream to persist and resume; if it reached
/// the last layer it ran the head and yields per-sequence logits.
pub enum WavePhase {
    /// Layers `[start, end)` ran with `end < num_layers`; the residual stream is
    /// handed back to be persisted and fed as `x_in` on the next wave.
    Residual(TensorCat),
    /// The range reached the final layer; the head ran. One logits row per input
    /// sequence, packed as the current forward's `TensorCat`.
    Logits(TensorCat),
}

// ============================================================================
// Core Model Trait (Simple Accessors Only)
// ============================================================================

/// Minimal trait for models that support batched forward passes.
///
/// This trait only requires simple accessor methods. All complex logic
/// (RoPE caching, forward_batch implementation) lives in [`BatchedInference`].
pub trait BatchedModelCore {
    /// Type of the layer that implements [`BatchedAttentionLayer`].
    type Layer: BatchedAttentionLayer;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Number of KV heads per layer.
    fn n_kv_head(&self) -> usize;

    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;

    /// Device the model is on.
    fn device(&self) -> &Device;

    /// Access the embedding layer.
    fn embeddings(&self) -> &Embedding;

    /// Access a layer by index.
    fn layer(&self, idx: usize) -> &Self::Layer;

    /// Access the final RMS normalization layer.
    fn final_norm(&self) -> &RmsNorm;

    /// Access the output projection (LM head).
    fn output_proj(&self) -> &QMatMul;

    /// Whether RoPE uses interleaved format.
    ///
    /// - `false`: Standard format, uses `rope()`
    /// - `true`: Interleaved format, uses `rope_i()`
    fn rope_interleaved(&self) -> bool;

    /// Prune excess memory usage (e.g., compact embeddings).
    fn prune(&self) -> Result<()>;

    /// Snapshot expert pipeline telemetry counters (if this model has an expert cache).
    fn expert_stats(&self) -> Option<PipelineStats> {
        None
    }

    /// Reset expert pipeline telemetry counters to zero.
    fn reset_expert_stats(&self) {}

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }

    /// Per-model multiplier for the K high adaptive threshold.
    fn k_hi_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the K low adaptive threshold.
    fn k_low_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the V high (strict) adaptive threshold.
    fn v_hi_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    fn v_low_error_threshold_factor(&self) -> f32 {
        1.0
    }
}

// ============================================================================
// Batched Inference Wrapper
// ============================================================================

/// Default initial RoPE table size.
const DEFAULT_ROPE_SEQ_LEN: usize = 4096;
/// Chunk size for extending RoPE tables.
const ROPE_EXTEND_CHUNK: usize = 1024;

/// Concrete wrapper for batched inference with RoPE caching.
///
/// This struct owns the RoPE cache and provides the `forward_batch` implementation.
/// Using a concrete wrapper instead of a trait default ensures:
/// - RoPE tables are cached once at the model level (not per-layer)
/// - No duplication of forward_batch logic across models
/// - Easy to add shared state (attention mask cache, etc.) in the future
pub struct BatchedInference<M: BatchedModelCore> {
    model: M,
    rope_cache: RwLock<CisPrecomputations>,
    /// Per-dimension inverse frequencies for the CUDA paged-attention kernels.
    /// Shape: [head_dim/2], dtype F32, stored on the model device.
    inv_freq_device: Tensor,
    /// Cached precomputed cos/sin table for decode RoPE.
    /// Computed lazily on first decode call, keyed by max_blocks.
    /// Shape: [max_pos, head_dim], dtype F32, on device.
    rope_cs_cache: std::sync::Mutex<Option<(usize, Tensor)>>,
    /// When true, `forward_batch` projects ALL token positions through the LM head
    /// instead of only the last token. Used for perplexity evaluation.
    /// Default: false (near-zero cost when off).
    all_logits: bool,
}

impl<M: BatchedModelCore> BatchedInference<M> {
    /// Create a new batched inference wrapper.
    ///
    /// # Arguments
    /// * `model` - The model to wrap
    /// * `rope_theta` - RoPE base frequency (e.g., 10000.0 for LLaMA, 1000000.0 for Qwen3)
    /// * `max_seq_len` - Initial RoPE table size (will auto-extend if needed)
    /// * `device` - Device for RoPE tables
    pub fn new(model: M, rope_theta: f32, max_seq_len: usize, device: &Device) -> Result<Self> {
        let head_dim = model.head_dim();
        let rope_cache = RwLock::new(CisPrecomputations::new_growable(
            head_dim,
            rope_theta,
            max_seq_len,
            ROPE_EXTEND_CHUNK,
            device,
        )?);
        let half_dim = head_dim / 2;
        let inv_freq_data: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq_device = Tensor::from_vec(inv_freq_data, (half_dim,), device)?;
        Ok(Self {
            model,
            rope_cache,
            inv_freq_device,
            rope_cs_cache: std::sync::Mutex::new(None),
            all_logits: false,
        })
    }

    /// Create with default RoPE table size.
    pub fn new_default(model: M, rope_theta: f32, device: &Device) -> Result<Self> {
        Self::new(model, rope_theta, DEFAULT_ROPE_SEQ_LEN, device)
    }

    /// Create with a custom inv_freq tensor for non-standard RoPE scaling.
    ///
    /// Use this for models with custom RoPE configurations (e.g., scaled RoPE).
    pub fn new_with_inv_freq(
        model: M,
        inv_freq: Vec<f32>,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = inv_freq.len();
        let inv_freq_device = Tensor::from_vec(inv_freq.clone(), (half_dim,), device)?;
        let rope_cache = RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
            inv_freq,
            max_seq_len,
            ROPE_EXTEND_CHUNK,
            device,
        )?);
        Ok(Self {
            model,
            rope_cache,
            inv_freq_device,
            rope_cs_cache: std::sync::Mutex::new(None),
            all_logits: false,
        })
    }

    /// When true, `forward_batch` returns logits for ALL positions, not just last.
    pub fn set_all_logits(&mut self, enabled: bool) {
        self.all_logits = enabled;
    }

    /// Access the underlying model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Access the underlying model mutably.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Consume the wrapper and return the inner model.
    pub fn into_inner(self) -> M {
        self.model
    }

    /// Get RoPE (cos, sin) tables for the given dtype and length.
    fn get_rope_tables(&self, dtype: DType, required_len: usize) -> Result<(Tensor, Tensor)> {
        // Check if extension is needed
        let needs_extend = {
            let cache = self
                .rope_cache
                .read()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            required_len > cache.max_seq_len()
        };

        if needs_extend {
            let mut cache = self
                .rope_cache
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cache.ensure_len(required_len)?;
        }

        // Get tables for the requested dtype
        let cache = self
            .rope_cache
            .read()
            .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
        let cis = cache.get_for_dtype(dtype)?;
        Ok((cis.cos.clone(), cis.sin.clone()))
    }

    /// Process multiple sequences in continuous batching style with parallel GPU execution.
    ///
    /// # Arguments
    /// * `contexts` - Mutable slice of sequence contexts
    ///
    /// # Returns
    /// A `TensorCat` containing the output logits for each sequence.
    pub fn forward_batch(
        &self,
        contexts: &mut [SequenceContext],
        generation: &Generation,
        decode_headers: DecodeHeaders,
    ) -> Result<TensorCat> {
        // A whole-model sweep is the [0, num_layers) case of the re-entrant path,
        // starting from freshly-embedded tokens (`x_in = None`) and running the
        // head. Single source of truth — the wave engine reuses the same body.
        let n = self.model.num_layers();
        match self.forward_batch_layers(contexts, generation, decode_headers, 0, n, None)? {
            WavePhase::Logits(logits) => Ok(logits),
            WavePhase::Residual(_) => {
                candle::bail!("forward_batch: a full [0, num_layers) sweep must return logits")
            }
        }
    }

    /// Re-entrant, layer-range forward — the primitive the continuous-fair-wave
    /// engine (`docs/continuous_fair_waves.md`) builds on. Runs layers
    /// `[layer_start, layer_end)` over `contexts`, threading the inter-layer
    /// residual stream:
    ///
    /// - `x_in = None` embeds `contexts`' input tokens (the entry into layer 0);
    ///   `x_in = Some(residual)` resumes a paused batch from its persisted stream.
    /// - Returns [`WavePhase::Logits`] when `layer_end == num_layers` (the head
    ///   ran), else [`WavePhase::Residual`] carrying the stream to persist and
    ///   resume next wave.
    ///
    /// Positional/RoPE/attention metadata is rebuilt from `contexts` each call
    /// (it is a pure function of the sequences' offsets + query lengths, which do
    /// not change while a prefill is paused), so a resumed batch RoPEs and masks
    /// identically to a single full sweep.
    pub fn forward_batch_layers(
        &self,
        contexts: &mut [SequenceContext],
        generation: &Generation,
        decode_headers: DecodeHeaders,
        layer_start: usize,
        layer_end: usize,
        x_in: Option<TensorCat>,
    ) -> Result<WavePhase> {
        if contexts.is_empty() {
            candle::bail!("Cannot process empty batch");
        }
        let num_layers = self.model.num_layers();
        if layer_start > layer_end || layer_end > num_layers {
            candle::bail!(
                "forward_batch_layers: bad range [{layer_start}, {layer_end}) for {num_layers} layers"
            );
        }
        let _ = &generation;
        let batch_size = contexts.len();

        // Collect offsets, input tensors, and per-sequence new-token lengths.
        let mut offsets = Vec::with_capacity(batch_size);
        let mut input_tensors = Vec::with_capacity(batch_size);
        let mut q_lens = Vec::with_capacity(batch_size);
        for ctx in contexts.iter() {
            offsets.push(ctx.offset);
            input_tensors.push(ctx.input_ids.clone());
            q_lens.push(ctx.input_len);
        }

        // Prefill packs the batch FLAT along the token dimension ([1, Σlen, …])
        // so heterogeneous per-sequence lengths share one varlen forward; decode
        // (1 token/seq) stacks along the batch dimension ([b_sz, 1, …]).
        let is_prefill = matches!(decode_headers, DecodeHeaders::Prefill(_));
        let cat_dim = if is_prefill { 1 } else { 0 };

        // Derive dtype from KV cache to ensure consistency throughout forward pass
        let cache_dtype = contexts
            .first()
            .map(|ctx| ctx.kv_caches.dtype())
            .unwrap_or(DType::F32);
        // For FP8 KV cache, use BF16 for activations (mixed precision mode)
        // F8E4M3 has too limited range for embeddings and intermediate activations
        let embed_dtype = if cache_dtype == DType::F8E4M3 {
            DType::BF16
        } else {
            cache_dtype
        };

        // The inter-layer residual stream: either embed the input tokens (entry
        // at layer_start) or resume from the persisted stream of a paused batch.
        let mut x = match x_in {
            None => {
                let packed = TensorCat::from_tensors(cat_dim, input_tensors.into_iter())?;
                let x_tensor = packed.to_tensor();
                let embedded = self
                    .model
                    .embeddings()
                    .forward_as_dtype(&x_tensor, embed_dtype)?
                    .contiguous()?;
                TensorCat::from_cat_tensor(embedded.to_dtype(embed_dtype)?, 0)?
            }
            Some(resume) => resume,
        };

        // For prefill (flat) `seq_len` is the total packed token count; for decode
        // it is 1. Per-sequence lengths are carried in `q_lens`.
        let seq_len = x.dim(1)?;

        // Pre-compute RoPE (cos, sin) for the batch (ragged per-token positions).
        let rope_cos_sin = self.compute_rope_for_batch(&offsets, &q_lens, embed_dtype)?;

        // Get-or-compute the precomputed RoPE cos/sin table for decode.
        // Lazily initialized on first decode call, then cached for subsequent calls.
        let rope_cs = {
            let max_blocks = contexts
                .first()
                .and_then(|ctx| {
                    ctx.kv_caches
                        .caches
                        .first()
                        .map(|c| c.k_cache().chunked_max_blocks())
                })
                .unwrap_or(0);
            let mut cache = self
                .rope_cs_cache
                .lock()
                .map_err(|_| candle::Error::Msg("poisoned rope_cs lock".into()))?;
            if let Some((cached_mb, ref t)) = *cache {
                if cached_mb == max_blocks {
                    t.clone()
                } else {
                    let t = crate::models::prefill_utils::compute_rope_cs(
                        &self.inv_freq_device,
                        max_blocks,
                        self.model.head_dim(),
                        self.model.device(),
                    )?;
                    *cache = Some((max_blocks, t.clone()));
                    t
                }
            } else {
                let t = crate::models::prefill_utils::compute_rope_cs(
                    &self.inv_freq_device,
                    max_blocks,
                    self.model.head_dim(),
                    self.model.device(),
                )?;
                *cache = Some((max_blocks, t.clone()));
                t
            }
        };

        // Per-forward cache for the layer-invariant prefill position_map: the
        // first layer builds + uploads it, every later layer reuses it. Lives for
        // the whole forward (all layers share it), then drops here.
        let shared_prefill_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);

        // Build params for batched attention
        let params = BatchedAttentionParams::new(
            &rope_cos_sin.0,
            &rope_cos_sin.1,
            self.model.rope_interleaved(),
            &self.inv_freq_device,
            &rope_cs,
            decode_headers,
            &q_lens,
            generation,
            &shared_prefill_pm,
        );

        // Process through this wave's layer range. `x` (the residual stream) is
        // already established above — embedded fresh, or resumed from a paused batch.
        let stage_is_decode = seq_len == 1;
        let t_layers = profile_now();
        for layer_idx in layer_start..layer_end {
            let mut cache_refs: Vec<&mut KvCache> = contexts
                .iter_mut()
                .map(|ctx| &mut ctx.kv_caches.caches[layer_idx])
                .collect();

            // A homogeneous forward is the single-group case of the mixed-wave
            // layer: one row-type over the whole buffer. A true wave passes a
            // decode + prefill + glue group triple instead (same shared FFN/MoE).
            let mut groups = [WaveAttnGroup {
                caches: &mut cache_refs,
                offsets: &offsets,
                params: &params,
                rows: seq_len,
                decode_layout: false,
            }];
            forward_layer_batched_mixed(
                self.model.layer(layer_idx),
                &mut groups,
                &mut x,
                embed_dtype,
                layer_idx,
            )?;
        }
        profile_sync(self.model.device());
        pipeline_record(
            if stage_is_decode {
                "decode:model:layers:total"
            } else {
                "prefill:model:layers:total"
            },
            t_layers,
        );

        // Paused mid-stack: hand the residual stream back so the scheduler can
        // persist it and resume this batch next wave. Only a range that reached
        // the final layer runs the head below.
        if layer_end < num_layers {
            return Ok(WavePhase::Residual(x));
        }

        let t_proj = profile_now();
        // B5: gather the positions that actually need logits BEFORE normalizing (RMSNorm is
        // per-position, so select-then-norm == norm-then-select), then fuse the final RMSNorm into
        // q8a128 and feed `output_proj` directly via `forward_dynamic` — no FP store + standalone
        // quantize on the int8 path. Off mode (every non-int8 model) takes the plain FP path.
        let x_tensor = x.to_tensor();
        let pre_norm = if self.all_logits {
            // All positions (perplexity evaluation mode).
            x_tensor.clone()
        } else if is_prefill {
            // Flat-packed prefill: x is [1, total, hidden]; the last token of sequence i sits at
            // the flat index (Σ_{j<=i} q_lens[j]) - 1. Gather those rows → [n_seqs, hidden].
            let hidden = x_tensor.dim(2)?;
            let x_flat = x_tensor.reshape((seq_len, hidden))?;
            let mut last_idx = Vec::with_capacity(q_lens.len());
            let mut acc = 0u32;
            for &l in &q_lens {
                acc += l as u32;
                last_idx.push(acc - 1);
            }
            let idx = Tensor::from_vec(last_idx, q_lens.len(), x_flat.device())?;
            x_flat.index_select(&idx, 0)?.contiguous()?
        } else {
            // Decode: [b_sz, 1, hidden] → the single (last) token per sequence.
            x_tensor.i((.., seq_len - 1, ..))?.contiguous()?
        };

        let logits = {
            #[cfg(feature = "cuda")]
            {
                let proj = self.model.output_proj();
                let acts = self
                    .model
                    .final_norm()
                    .forward_dynamic(&pre_norm, proj.int8mode())?;
                proj.forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let normed = self.model.final_norm().forward(&pre_norm)?;
                self.model.output_proj().forward(&normed)?
            }
        };

        profile_sync(self.model.device());
        pipeline_record(
            if stage_is_decode {
                "decode:model:norm+proj"
            } else {
                "prefill:model:norm+proj"
            },
            t_proj,
        );
        Ok(WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?))
    }

    /// Co-batched continuous-fair-wave forward (`docs/continuous_fair_waves.md`).
    ///
    /// Packs decode (q=1), prefill (q=N) and glue (q=G) rows into ONE flat
    /// activation buffer and runs the re-entrant layer range with the 3-group
    /// mixed dispatch — each row-type's own attention kernel over its slice, then
    /// the **single shared FFN/MoE grouped GEMM** over the whole buffer (one expert
    /// load per layer serves all three). Since attention is per-type, o_proj is
    /// linear, and the FFN/MoE is token-flat, this is bit-identical to running the
    /// three types as separate forwards through a shared MoE.
    ///
    /// `contexts` are ordered `[decode… | prefill… | glue…]`; `n_decode` /
    /// `n_prefill` give the group boundaries (glue is the remainder). The three
    /// `*_headers` are the per-group attention metadata (Decode / Prefill /
    /// Prefill+glue). When the range reaches the head, returns logits for the
    /// **decode + prefill** rows only (glue rows scatter K/V, they carry no logits).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_wave_contexts(
        &self,
        contexts: &mut [SequenceContext],
        n_decode: usize,
        n_prefill: usize,
        decode_headers: DecodeHeaders,
        prefill_headers: DecodeHeaders,
        glue_headers: DecodeHeaders,
        generation: &Generation,
        layer_start: usize,
        layer_end: usize,
        x_in: Option<TensorCat>,
    ) -> Result<WavePhase> {
        if contexts.is_empty() {
            candle::bail!("forward_wave: empty batch");
        }
        let num_layers = self.model.num_layers();
        if layer_start > layer_end || layer_end > num_layers {
            candle::bail!("forward_wave: bad layer range [{layer_start}, {layer_end})");
        }
        let n_glue = contexts
            .len()
            .checked_sub(n_decode + n_prefill)
            .ok_or_else(|| candle::Error::Msg("forward_wave: group bounds exceed batch".into()))?;

        // Per-group offsets + query lengths, in [decode | prefill | glue] order.
        let offsets: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
        let q_lens: Vec<usize> = contexts.iter().map(|c| c.input_len).collect();
        let (dec_off, rest_off) = offsets.split_at(n_decode);
        let (pre_off, glue_off) = rest_off.split_at(n_prefill);
        let (dec_q, rest_q) = q_lens.split_at(n_decode);
        let (pre_q, glue_q) = rest_q.split_at(n_prefill);
        // Flat token-row counts per group (decode is one row per sequence).
        let pre_rows: usize = pre_q.iter().sum();
        let glue_rows: usize = glue_q.iter().sum();

        let cache_dtype = contexts
            .first()
            .map(|c| c.kv_caches.dtype())
            .unwrap_or(DType::F32);
        let embed_dtype = if cache_dtype == DType::F8E4M3 {
            DType::BF16
        } else {
            cache_dtype
        };

        // Combined residual: embed every row flat `[1, total, hidden]`, or resume
        // a paused wave from its persisted stream.
        let mut x = match x_in {
            None => {
                let inputs: Vec<Tensor> = contexts.iter().map(|c| c.input_ids.clone()).collect();
                let packed = TensorCat::from_tensors(1, inputs.into_iter())?;
                let xt = packed.to_tensor();
                let embedded = self
                    .model
                    .embeddings()
                    .forward_as_dtype(&xt, embed_dtype)?
                    .contiguous()?;
                TensorCat::from_cat_tensor(embedded.to_dtype(embed_dtype)?, 0)?
            }
            Some(resume) => resume,
        };

        // Shared decode rope_cs table (position-indexed lookup used by all groups).
        let rope_cs = {
            let max_blocks = contexts
                .first()
                .and_then(|c| {
                    c.kv_caches
                        .caches
                        .first()
                        .map(|k| k.k_cache().chunked_max_blocks())
                })
                .unwrap_or(0);
            let mut cache = self
                .rope_cs_cache
                .lock()
                .map_err(|_| candle::Error::Msg("poisoned rope_cs lock".into()))?;
            match *cache {
                Some((mb, ref t)) if mb == max_blocks => t.clone(),
                _ => {
                    let t = crate::models::prefill_utils::compute_rope_cs(
                        &self.inv_freq_device,
                        max_blocks,
                        self.model.head_dim(),
                        self.model.device(),
                    )?;
                    *cache = Some((max_blocks, t.clone()));
                    t
                }
            }
        };

        // Per-group RoPE (cos/sin) + prefill position-map caches, all alive for the
        // whole layer loop.
        let dec_rope = self.compute_rope_for_batch(dec_off, dec_q, embed_dtype)?;
        let pre_rope = self.compute_rope_for_batch(pre_off, pre_q, embed_dtype)?;
        let glue_rope = self.compute_rope_for_batch(glue_off, glue_q, embed_dtype)?;
        let dec_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let pre_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let glue_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let interleaved = self.model.rope_interleaved();
        let dec_params = BatchedAttentionParams::new(
            &dec_rope.0,
            &dec_rope.1,
            interleaved,
            &self.inv_freq_device,
            &rope_cs,
            decode_headers,
            dec_q,
            generation,
            &dec_pm,
        );
        let pre_params = BatchedAttentionParams::new(
            &pre_rope.0,
            &pre_rope.1,
            interleaved,
            &self.inv_freq_device,
            &rope_cs,
            prefill_headers,
            pre_q,
            generation,
            &pre_pm,
        );
        let glue_params = BatchedAttentionParams::new(
            &glue_rope.0,
            &glue_rope.1,
            interleaved,
            &self.inv_freq_device,
            &rope_cs,
            glue_headers,
            glue_q,
            generation,
            &glue_pm,
        );

        for layer_idx in layer_start..layer_end {
            let mut cache_refs: Vec<&mut KvCache> = contexts
                .iter_mut()
                .map(|c| &mut c.kv_caches.caches[layer_idx])
                .collect();
            let (dec_c, rest_c) = cache_refs.split_at_mut(n_decode);
            let (pre_c, glue_c) = rest_c.split_at_mut(n_prefill);
            let mut groups: Vec<WaveAttnGroup> = Vec::with_capacity(3);
            if n_decode > 0 {
                groups.push(WaveAttnGroup {
                    caches: dec_c,
                    offsets: dec_off,
                    params: &dec_params,
                    rows: n_decode,
                    decode_layout: true,
                });
            }
            if n_prefill > 0 {
                groups.push(WaveAttnGroup {
                    caches: pre_c,
                    offsets: pre_off,
                    params: &pre_params,
                    rows: pre_rows,
                    decode_layout: false,
                });
            }
            if n_glue > 0 {
                groups.push(WaveAttnGroup {
                    caches: glue_c,
                    offsets: glue_off,
                    params: &glue_params,
                    rows: glue_rows,
                    decode_layout: false,
                });
            }
            forward_layer_batched_mixed(
                self.model.layer(layer_idx),
                &mut groups,
                &mut x,
                embed_dtype,
                layer_idx,
            )?;
        }

        if layer_end < num_layers {
            return Ok(WavePhase::Residual(x));
        }

        // Head over the rows that need logits: every decode row (one token each,
        // flat positions `0..n_decode`) and the last token of every prefill row
        // (within the prefill slice `[n_decode, n_decode + pre_rows)`). Glue rows
        // are excluded — they only scattered K/V.
        let x_tensor = x.to_tensor();
        let hidden = x_tensor.dim(2)?;
        let x_flat = x_tensor.reshape((x_tensor.dim(1)?, hidden))?;
        let mut idx: Vec<u32> = Vec::with_capacity(n_decode + n_prefill);
        for d in 0..n_decode {
            idx.push(d as u32);
        }
        let mut acc = n_decode as u32;
        for &l in pre_q {
            acc += l as u32;
            idx.push(acc - 1);
        }
        let pre_norm = if idx.is_empty() {
            candle::bail!("forward_wave: no decode/prefill rows to head");
        } else {
            let sel = Tensor::from_vec(idx, n_decode + n_prefill, x_flat.device())?;
            x_flat.index_select(&sel, 0)?.contiguous()?
        };
        let logits = {
            #[cfg(feature = "cuda")]
            {
                let proj = self.model.output_proj();
                let acts = self
                    .model
                    .final_norm()
                    .forward_dynamic(&pre_norm, proj.int8mode())?;
                proj.forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let normed = self.model.final_norm().forward(&pre_norm)?;
                self.model.output_proj().forward(&normed)?
            }
        };
        Ok(WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?))
    }

    /// Compute RoPE (cos, sin) for a batch of sequences.
    ///
    /// For decode (seq_len == 1): Returns gathered (cos, sin) at each offset position.
    /// For prefill (seq_len > 1): Returns (cos, sin) reshaped for batch processing.
    fn compute_rope_for_batch(
        &self,
        offsets: &[usize],
        q_lens: &[usize],
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        use super::decode_utils;

        // Required RoPE table length = max over sequences of (offset + q_len).
        let required_len = offsets
            .iter()
            .zip(q_lens.iter())
            .map(|(&o, &l)| o + l)
            .max()
            .unwrap_or(1);

        // RoPE doesn't support F8E4M3, use BF16 instead
        let rope_dtype = if dtype == DType::F8E4M3 {
            DType::BF16
        } else {
            dtype
        };

        // Get RoPE tables (may extend if needed)
        let (cos_all, sin_all) = self.get_rope_tables(rope_dtype, required_len)?;

        // Decode is the uniform single-token case (every q_len == 1): gather the
        // (cos, sin) at the per-sequence offsets → [b_sz, rotary].
        if q_lens.iter().all(|&l| l == 1) {
            let offsets_t = decode_utils::offsets_to_u32_tensor(offsets, self.model.device())?;
            return decode_utils::gather_rope_cos_sin(&cos_all, &sin_all, &offsets_t);
        }

        // Prefill: ragged per-token positions flat-packed in cu_seqlens order.
        // Sequence i's new tokens occupy absolute positions [off_i, off_i+q_len_i).
        let total: usize = q_lens.iter().sum();
        let mut pos = Vec::with_capacity(total);
        for (&off, &l) in offsets.iter().zip(q_lens.iter()) {
            for i in 0..l {
                pos.push((off + i) as u32);
            }
        }
        let pos_flat = Tensor::from_vec(pos, (total,), self.model.device())?;

        let mut cos = cos_all.index_select(&pos_flat, 0)?;
        let mut sin = sin_all.index_select(&pos_flat, 0)?;
        if !cos.is_contiguous() {
            cos = cos.contiguous()?;
        }
        if !sin.is_contiguous() {
            sin = sin.contiguous()?;
        }

        // [total, rotary] -> [1, total, rotary] to match the flat [1, total, …]
        // activation (batch-of-one); the non-paged rope() consumes it directly.
        let rotary_dim = cos.dim(1)?;
        let cos = cos.reshape((1, total, rotary_dim))?;
        let sin = sin.reshape((1, total, rotary_dim))?;

        Ok((cos, sin))
    }
}
