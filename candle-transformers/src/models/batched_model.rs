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
//!     fn embeddings(&self) -> Option<&Embedding> { Some(&self.embeddings) }
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
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_forward, begin_wave, end_wave_transient, plan_wave_transient, LayerPhase, ModelGeometry,
    WavePlan, REGION_BYTES, WAVE_FORWARD_BYTES,
};
use candle_nn::Module;

use super::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionLayer, BatchedAttentionParams, DecodeHeaders,
    WaveAttnGroup,
};
use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use super::kv_cache_utils::SequenceContext;
use super::prefill_utils::SharedPm;
use super::quantized_matmul::QMatMul;
use super::rope_tables::CisPrecomputations;
use super::tensor_cat::TensorCat;
use super::wave_admit::admit_wave_kv;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_root;
use crate::quantized_nn::RmsNorm;
use candle_nn::Embedding;

/// The forward-scoped generation a wave's head outputs were carved from.
///
/// Returned alongside the phase so the caller can keep the span open for as long
/// as it holds the values sitting in it. There is no wave domain without CUDA,
/// so off-CUDA this is a unit and every guard is `None`.
#[cfg(feature = "cuda")]
pub type WaveGuard = candle_nn::kv_cache::WaveGeneration;
#[cfg(not(feature = "cuda"))]
pub type WaveGuard = ();

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

/// Driver-used VRAM right now (`total − free` from `cuMemGetInfo`), in bytes;
/// `0` on non-CUDA / query failure. Dense models capture this at the start and
/// end of weight loading — the delta is their fixed resident-weight footprint,
/// reported through [`BatchedModelCore::resident_weight_bytes`] for the
/// whole-card VRAM decomposition.
#[cfg(feature = "cuda")]
pub(crate) fn driver_used_bytes(device: &Device) -> usize {
    if matches!(device, Device::Cuda(_)) {
        candle::quantized::get_vram_info()
            .map(|(free, total)| total.saturating_sub(free))
            .unwrap_or(0)
    } else {
        0
    }
}

/// The load-time shapes a wave's transient buffers are sized from.
///
/// Deliberately **width-free**: a wave's row count is an argument to the sizing
/// functions, not a field here. Width is what admission is deciding, so baking
/// an assumed width in would make the plan answer a question it is supposed to
/// be asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveShapes {
    /// Model hidden size.
    pub hidden: usize,
    /// FFN intermediate size. On a MoE model this is the *per-expert*
    /// intermediate, not the dense equivalent.
    pub intermediate: usize,
    /// Experts each token routes to. `1` for a dense model, which collapses the
    /// MoE terms to the dense FFN shapes rather than needing a second branch.
    pub experts_per_tok: usize,
    /// Experts the router scores over — the width of the per-token logits the
    /// FFN phase carries. `1` on a dense model, which has no router.
    pub n_experts: usize,
}

/// The dtype activations are carried in, for a KV cache stored as `cache_dtype`.
///
/// Activations follow the cache so the two never need a conversion between them
/// — except for F8E4M3, which is a storage format with no compute kernels, so a
/// cache in it computes in BF16.
///
/// One definition, because two would drift: the forward derives the activation
/// dtype from the session's cache on every wave, and whoever configures the
/// session has to materialise the norm weights in the *same* dtype or the
/// forward refuses.
pub fn activation_dtype(cache_dtype: DType) -> DType {
    if cache_dtype == DType::F8E4M3 {
        DType::BF16
    } else {
        cache_dtype
    }
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

    /// Shapes the transient tier is sized from, captured at load.
    ///
    /// Read from the checkpoint's own config rather than derived from weight
    /// dimensions: `hidden` is not `n_head * head_dim` on every model (Qwen3
    /// carries 2048 against 32x128), so deriving it would be right for some
    /// architectures and quietly wrong for others.
    fn wave_shapes(&self) -> WaveShapes;

    /// The geometry [`candle_nn::kv_cache::WavePlan`] prices a wave from.
    ///
    /// Pairs the load-time shapes with the session's activation dtype, which is
    /// an inference-mode choice rather than a property of the weights. Head
    /// geometry comes from layer 0 — every layer of a model shares it, and the
    /// plan needs one layer's working set, not the whole stack's.
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        let shapes = self.wave_shapes();
        ModelGeometry {
            hidden: shapes.hidden,
            intermediate: shapes.intermediate,
            n_head: self.layer(0).n_head(),
            n_kv_head: self.n_kv_head(),
            head_dim: self.head_dim(),
            experts_per_tok: shapes.experts_per_tok,
            n_experts: shapes.n_experts,
            act_dtype,
            // The int8 tensor-core kernels emit **F32** and the cast back to
            // `act_dtype` is a second buffer, so the plan charges for both
            // rather than for the wider of the two.
            //
            // This said BF16 until the census measured it: the expert GEMM
            // outputs came back at four bytes an element, not two, so every
            // accumulate-dtype term in the FFN phase — gate, up and down, the
            // three largest buffers a MoE layer allocates — was priced at half
            // its size.
            accum_dtype: DType::F32,
        }
    }

    /// Re-materialise every norm weight in the dtype activations will arrive in,
    /// if it is not already.
    ///
    /// Called where a session is created — never inside a wave. A quantized
    /// checkpoint dequantizes its norm weights to F32 while inference runs F16 or
    /// BF16, so the norm kernels need them in the activation dtype. Converting on
    /// demand inside the forward costs one device allocation and one launch per
    /// norm, per layer, per token, and is invisible in a profile: it surfaces
    /// only as a slightly slower forward. So the forward path *refuses* a dtype
    /// it was not prepared for (`RmsNorm::weight_for`) and this is the only place
    /// the conversion happens.
    ///
    /// The weight is reloaded from the retained quantized source rather than cast
    /// from a resident copy, so switching dtypes costs a reload and never leaves
    /// a second copy behind.
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()>;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Number of KV heads per layer.
    fn n_kv_head(&self) -> usize;

    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;

    /// Device the model is on.
    fn device(&self) -> &Device;

    /// Access the resident embedding layer.
    ///
    /// `None` when the table is served from host memory instead — see
    /// [`Self::host_embedding`]. Exactly one of the two is populated.
    fn embeddings(&self) -> Option<&Embedding>;

    /// The token embedding served from the GGUF mmap rather than VRAM.
    ///
    /// `Some` only when the table is large enough relative to the card that
    /// keeping it resident is not worth the VRAM (see
    /// [`crate::models::host_embedding`]); the rows are then gathered per
    /// forward instead. `None` keeps the resident path.
    #[cfg(feature = "cuda")]
    fn host_embedding(&self) -> Option<&crate::models::host_embedding::HostEmbedding> {
        None
    }

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

    /// Ask the weight side to hand ground to the KV side now, answering with the
    /// bytes conceded.
    ///
    /// For a caller whose KV allocation just failed, or which can see it is
    /// about to. The other direction runs only between forwards
    /// ([`Self::reclaim_spare_ground`]), which a wave that cannot allocate never
    /// reaches. `regions` is the quantity the caller measured — never a counter
    /// it drained, which is what this replaced and what took the expert zone
    /// below its working minimum. Zero is an ordinary answer: the zone is at its
    /// floor, or a wave is still in flight.
    ///
    /// Dense models have no movable boundary and return zero.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// The opposite direction: let the weight side take back KV regions that are
    /// standing free.
    ///
    /// **Only legal between forwards.** A boundary move evicts and relocates
    /// expert slots, so it may not run under a live wave generation — see
    /// `ExpertCache::reclaim_spare_ground` for what happened when it was driven
    /// from the expert pipeline's end-of-pass instead.
    ///
    /// Dense models have no movable boundary and do nothing.
    fn reclaim_spare_ground(&self) {}

    /// Live VRAM bytes held by the model's weights — fixed base weights plus the
    /// **time-varying** resident-expert footprint (MoE experts page VRAM↔RAM
    /// under pressure). `None` if the model can't report it (dense models, no
    /// expert cache, non-CUDA). Feeds the whole-card VRAM decomposition.
    fn resident_weight_bytes(&self) -> Option<usize> {
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
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
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
        let embed_dtype = activation_dtype(cache_dtype);

        // **Phase 0: hand back the previous forward's tier.**
        //
        // It is held past its guards on purpose (`release_if_last`), and
        // `plan_wave_transient` used to be what returned it — one phase too
        // late. Admit runs first and claims against a pool the old tier is still
        // capping, so it can be refused by a reservation belonging to a forward
        // that has already finished. Invisible while every wave succeeds, fatal
        // the moment one fails: the failed wave's tier stands, every retry's
        // admit is refused by it, and the engine spins.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = self.model.device() {
            end_wave_transient(&d.cuda_stream());
            // **And the boundary's growing direction, in the one gap it is legal
            // in.** Every guard from the previous forward is dropped and this
            // one has opened none, so no wave generation is live — the condition
            // `set_weight_floor` checks, and the condition a retraction's
            // evictions and relocations actually need. It runs after the tier is
            // handed back because a placed tier caps the pool at a fixed address
            // that the boundary cannot move, which would make the spare-region
            // count this reads an underestimate.
            //
            // The KV side's direction is not here: a claim that runs out buys
            // its ground on the spot (`request_kv_ground`) rather than waiting
            // for a forward that its own failure is preventing.
            self.model.reclaim_spare_ground();
        }

        // **Phase 1: admit.** Claim every KV slot this wave will write, for
        // every layer in the range, before a single byte of it computes — so the
        // arena frontier is final when the transient tier is reserved against it
        // (`docs/elastic_vram_partition.md` §7, `wave_admit`). Decode's claims
        // were made by the caller when it built the position map; this covers
        // the multi-token rows.
        admit_wave_kv(contexts, n_decode, n_prefill, layer_start, layer_end)?;

        // **Phase 2: price and reserve this wave's transient tier.**
        //
        // After admit, so the partition it measures against is the one the whole
        // forward will run on. The tier is sized to *this* wave rather than to
        // the widest one the engine can run: a twenty-session decode prices at a
        // few megabytes where the old fixed constants reserved 912 MiB, and the
        // difference is ground the weight side gets to hold — which is the entire
        // point of the tier sitting between the arenas and the weights (§2).
        //
        // Reserved **once**, for the forward. `begin_wave` lays the three spans
        // out inside the reservation on every phase but never chooses its
        // address, which is what keeps layer *N*'s extents and layer *N+1*'s at
        // the same offsets (§13b).
        #[cfg(feature = "cuda")]
        {
            let rows = n_decode + pre_rows + glue_rows;
            if rows > 0 {
                if let Device::Cuda(d) = self.model.device() {
                    let plan = WavePlan::new(self.model.wave_geometry(embed_dtype));
                    // One region of slack per layer phase. The plan enumerates
                    // every declared buffer, but a phase pays one alignment per
                    // range and the count is not in the plan, so this covers the
                    // rounding rather than an unknown.
                    let pad = |b: usize| b + REGION_BYTES;
                    let per_phase = [
                        pad(plan.phase_bytes(LayerPhase::Attention, rows)),
                        pad(plan.phase_bytes(LayerPhase::Ffn, rows)),
                        // The forward phase carries per-*sequence* metadata —
                        // ragged offsets, RoPE tables — which the plan prices as
                        // zero because it sizes what scales with width. One
                        // region is the floor the tier is carved in anyway.
                        WAVE_FORWARD_BYTES,
                    ];
                    // The tier packs directly against the arena frontier, with no
                    // room reserved above it. Nothing claims a region after this
                    // point: a region claim creates an arena, and arena creation
                    // waits for the gap between forwards
                    // (`BackingInner::arena_window`), which `plan_wave_transient`
                    // in turn waits on before it reads the frontier. The
                    // compressor keeps running through the forward — it fills
                    // arenas that already exist, which moves nothing.
                    plan_wave_transient(&d.cuda_stream(), per_phase)?;
                }
            }
        }

        // **From here to the end of this function, the forward owns the
        // partition.**
        //
        // Held rather than inferred, because the obvious inference is wrong: a
        // wave generation is *not* live for the whole forward — it drops at
        // every phase boundary — so both `enter_arena_window` and the boundary
        // latch (`wave_is_live`) read this flag, and `live_generations` covers
        // only the tail after this returns, where the logits still sit on the
        // head span.
        //
        // **After phase 2, and that is not a detail — in either direction.**
        // Admit is the forward creating its own arenas through the same gate
        // the sealing thread uses, so opening the flag before admit had the
        // forward waiting on itself (the daemon froze mid-load). And phase 2's
        // tier placement may *buy ground* from the weight side, whose
        // `set_weight_floor` refuses while `wave_is_live` — which now reads
        // this flag — so opening it before the placement would refuse the
        // tier's own purchase. The sweep is what needs the partition frozen;
        // the flag opens exactly where the sweep begins.
        #[cfg(feature = "cuda")]
        let _forward_open = match self.model.device() {
            Device::Cuda(d) => Some(begin_forward(&d.cuda_stream())),
            _ => None,
        };

        // Combined residual: embed every row flat `[1, total, hidden]`, or resume
        // a paused wave from its persisted stream.
        let mut x = match x_in {
            None => {
                let inputs: Vec<Tensor> = contexts.iter().map(|c| c.input_ids.clone()).collect();
                let packed = TensorCat::from_tensors(1, inputs.into_iter())?;
                let xt = packed.to_tensor();
                // Prefer the host-served table when the model has one: the rows
                // are gathered from the mmap over PCIe, so the embedding never
                // occupies VRAM. Falls back to the resident lookup otherwise.
                #[cfg(feature = "cuda")]
                let host = self.model.host_embedding();
                #[cfg(not(feature = "cuda"))]
                let host: Option<&()> = None;
                let embedded = match host {
                    #[cfg(feature = "cuda")]
                    Some(he) => {
                        let flat = xt.flatten_all()?;
                        let n = flat.elem_count();
                        // A scope of its own for the gather's staging bytes. The
                        // embedding runs before layer 0, so no phase span is open
                        // and the attention arena is idle — its bytes are already
                        // reserved, so the staging is free. The guard drops here,
                        // before the layer loop opens the same span for real work.
                        let staging = match self.model.device() {
                            Device::Cuda(d) => {
                                Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?)
                            }
                            _ => None,
                        };
                        let rows =
                            he.embed(&flat, self.model.device(), wave_root(staging.as_ref()))?;
                        rows.reshape((1, n, he.layout().ncols))?
                    }
                    #[cfg(not(feature = "cuda"))]
                    Some(_) => unreachable!("host embedding requires the cuda feature"),
                    None => self
                        .model
                        .embeddings()
                        .ok_or_else(|| {
                            candle::Error::Msg(
                                "model has neither a resident nor a host embedding".into(),
                            )
                        })?
                        .forward_as_dtype(&xt, embed_dtype)?
                        .contiguous()?,
                };
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
            return Ok((WavePhase::Residual(x), None));
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
        // A glue-only wave (no decode/prefill rows) has nothing to head — the K/V
        // scatter already happened in the layer loop. Return the residual buffer;
        // the glue caller discards the `WaveStep` (it only needs the side effect).
        if idx.is_empty() {
            return Ok((WavePhase::Residual(x), None));
        }
        let pre_norm = {
            let sel = Tensor::from_vec(idx, n_decode + n_prefill, x_flat.device())?;
            x_flat.index_select(&sel, 0)?.contiguous()?
        };
        // The head's span. It runs after the last layer, so both phase spans are
        // idle, and this one is reset per *forward* — the lifetime the norm and
        // the logits actually have.
        //
        // Seeded from `wave_root`, which yields a `Backing` carrying a ticket
        // rather than a borrow of the guard. That distinction is the whole
        // mechanism: a borrow would bind `'w` to this scope and the logits could
        // not be returned at all, while a ticket leaves them `'static`-typed and
        // physically on the span. What makes that sound is handing the guard back
        // with them, so the span cannot be reclaimed while the caller holds the
        // values — see `WaveResult`.
        #[cfg(feature = "cuda")]
        let head_span = match self.model.device() {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Forward)?),
            _ => None,
        };
        let logits = {
            #[cfg(feature = "cuda")]
            {
                let proj = self.model.output_proj();
                let acts = self.model.final_norm().forward_dynamic(
                    &pre_norm,
                    proj.int8mode(),
                    wave_root(head_span.as_ref()),
                )?;
                proj.forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let normed = self.model.final_norm().forward(&pre_norm)?;
                self.model.output_proj().forward(&normed)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let head_span: Option<WaveGuard> = None;
        Ok((
            WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
            head_span,
        ))
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
