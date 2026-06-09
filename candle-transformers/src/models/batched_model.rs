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
    forward_layer_batched, BatchedAttentionLayer, BatchedAttentionParams, DecodeHeaders,
};
use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use super::kv_cache_utils::SequenceContext;
use super::profile::{pipeline_record, profile_now, profile_sync};
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use super::tensor_cat::TensorCat;
use crate::quantized_nn::RmsNorm;
use candle_nn::Embedding;

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
        if contexts.is_empty() {
            candle::bail!("Cannot process empty batch");
        }
        let _ = &generation;
        let batch_size = contexts.len();

        // Collect offsets, input tensors, and per-sequence new-token lengths.
        let mut offsets = Vec::with_capacity(batch_size);
        let mut input_tensors = Vec::with_capacity(batch_size);
        let mut write_shifts_raw = Vec::with_capacity(batch_size);
        let mut q_lens = Vec::with_capacity(batch_size);
        for ctx in contexts.iter() {
            offsets.push(ctx.offset);
            input_tensors.push(ctx.input_ids.clone());
            write_shifts_raw.push(ctx.write_offset_shift as u32);
            q_lens.push(ctx.input_len);
        }

        // Prefill packs the batch FLAT along the token dimension ([1, Σlen, …])
        // so heterogeneous per-sequence lengths share one varlen forward; decode
        // (1 token/seq) stacks along the batch dimension ([b_sz, 1, …]).
        let is_prefill = matches!(decode_headers, DecodeHeaders::Prefill(_));
        let cat_dim = if is_prefill { 1 } else { 0 };
        let x = TensorCat::from_tensors(cat_dim, input_tensors.into_iter())?;

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

        // Embed tokens
        let x_tensor = x.to_tensor();

        let embedded = self
            .model
            .embeddings()
            .forward_as_dtype(&x_tensor, embed_dtype)?
            .contiguous()?;

        // For prefill (flat) `seq_len` is the total packed token count; for decode
        // it is 1. Per-sequence lengths are carried in `q_lens`.
        let (_, seq_len, _) = embedded.dims3()?;

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
        );

        // Convert to activation dtype before entering layers
        let x_tensor = embedded.to_dtype(embed_dtype)?;
        let mut x = TensorCat::from_cat_tensor(x_tensor, 0)?;

        // Build write_offset_shifts buffer once (zeros when no SSO shifts needed).
        // Use the pinned staging system for a zero-copy device-visible buffer.
        #[cfg(feature = "cuda")]
        let (write_offset_shifts_ptr, _write_offset_shifts_guard) = {
            let byte_len = batch_size * 4;
            let mut buf = generation.alloc(byte_len)?;
            let src = unsafe {
                std::slice::from_raw_parts(write_shifts_raw.as_ptr() as *const u8, byte_len)
            };
            buf.copy_from_slice(src);
            let gpu_buf = generation.submit(buf)?;
            let ptr = gpu_buf.dev_ptr();
            (ptr, gpu_buf)
        };
        #[cfg(not(feature = "cuda"))]
        let write_offset_shifts_ptr: u64 = 0;

        // Process through all layers
        let stage_is_decode = seq_len == 1;
        let t_layers = profile_now();
        for layer_idx in 0..self.model.num_layers() {
            let mut cache_refs: Vec<&mut KvCache> = contexts
                .iter_mut()
                .map(|ctx| &mut ctx.kv_caches.caches[layer_idx])
                .collect();

            forward_layer_batched(
                self.model.layer(layer_idx),
                &mut cache_refs,
                &mut x,
                &offsets,
                &params,
                embed_dtype,
                write_offset_shifts_ptr,
                layer_idx,
            )?;
        }
        profile_sync(embedded.device());
        pipeline_record(
            if stage_is_decode {
                "decode:model:layers:total"
            } else {
                "prefill:model:layers:total"
            },
            t_layers,
        );

        let t_proj = profile_now();
        // Apply final normalization
        let x_tensor = x.to_tensor();
        let x = self.model.final_norm().forward(&x_tensor)?;

        // Project to vocabulary
        let logits = if self.all_logits {
            // All positions (perplexity evaluation mode)
            self.model.output_proj().forward(&x)?
        } else if is_prefill {
            // Flat-packed prefill: x is [1, total, hidden]; the last token of
            // sequence i sits at the flat index (Σ_{j<=i} q_lens[j]) - 1. Gather
            // those rows → [n_seqs, hidden] before the vocab projection.
            let hidden = x.dim(2)?;
            let x_flat = x.reshape((seq_len, hidden))?;
            let mut last_idx = Vec::with_capacity(q_lens.len());
            let mut acc = 0u32;
            for &l in &q_lens {
                acc += l as u32;
                last_idx.push(acc - 1);
            }
            let idx = Tensor::from_vec(last_idx, q_lens.len(), x_flat.device())?;
            let last_hidden = x_flat.index_select(&idx, 0)?.contiguous()?;
            self.model.output_proj().forward(&last_hidden)?
        } else {
            // Decode: [b_sz, 1, hidden] → the single (last) token per sequence.
            let last_hidden = x.i((.., seq_len - 1, ..))?.contiguous()?;
            self.model.output_proj().forward(&last_hidden)?
        };

        profile_sync(embedded.device());
        pipeline_record(
            if stage_is_decode {
                "decode:model:norm+proj"
            } else {
                "prefill:model:norm+proj"
            },
            t_proj,
        );
        TensorCat::from_cat_tensor(logits, 0)
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
