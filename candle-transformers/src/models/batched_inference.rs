//! Batched inference session for efficient multi-sequence processing.
//!
//! This module provides a high-level API for batched inference with paged attention.
//! The [`BatchedInferenceSession`] manages sequence allocation, KV cache backing, and
//! provides a simple interface for running batched forward passes.
//!
//! # Example
//!
//! ```ignore
//! use candle_transformers::models::batched_inference::{BatchedConfig, BatchedInferenceSession};
//!
//! // Create a batch session
//! let mut session = BatchedInferenceSession::new(
//!     num_layers, n_kv_head, head_dim, &device, BatchedConfig::default()
//! )?;
//!
//! // Run batched inference - BatchedInference<M> implements ManagedBatchedModel
//! let outputs = model.forward_batched(&mut session, &seq_indices, &input_tensors)?;
//! ```

use super::expert_lre::PipelineStats;
use super::expert_lre::ProfileSnapshot;
use crate::models::kv_cache_utils::{new_kv_caches, KvCaches, SequenceContext};
use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::GpuBuf;
use candle::quantized::GgmlDType;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    ChunkedKvBacking, CompressionPolicy, HeadGids, KvCache, KvFormat, QuantFormat,
};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "cuda")]
use super::batched_layer::GlueMeta;
use super::batched_layer::{BatchedPrefillMeta, DecodeHeaders};
use super::batched_model::{BatchedInference, BatchedModelCore};
#[cfg(feature = "cuda")]
use crate::models::profile::pipeline_record_duration;
use crate::models::profile::{pipeline_record, profile_now};

/// Inference mode specifying both compute dtype and KV cache storage format.
///
/// This enum provides a high-level way to configure inference:
/// - Float modes (F32, F16, BF16): Use the specified dtype for both compute and KV storage
/// - Quantized modes: Use F32 compute with quantized KV storage for memory savings
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    /// Full precision F32 compute and KV storage
    F32,
    /// Half precision F16 compute and KV storage
    F16,
    /// BFloat16 compute and KV storage
    BF16,
    /// F32 compute with Q8_0 quantized KV cache (8-bit int)
    Q8_0,
    /// F32 compute with Q8_1 quantized KV cache (8-bit int)
    Q8_1,
    /// F32 compute with Q8_KS quantized KV cache (8-bit int with sub-block scales)
    Q8_KS,
    /// F32 compute with Q4_0 quantized KV cache (4-bit int)
    Q4_0,
    /// F32 compute with Q4_1 quantized KV cache (4-bit int)
    Q4_1,
    /// F32 compute with Q4_KS quantized KV cache (4-bit with sub-block scales)
    Q4_KS,
    /// F32 compute with Q3_0 quantized KV cache (3-bit int)
    Q3_0,
    /// F32 compute with Q2_0 quantized KV cache (2-bit int)
    Q2_0,
    /// F16 compute with R16 KV cache (raw F16 with Q-capture space)
    R16,
    /// Composite types
    Q8_Q4,
    Q8_Q8KS,
    Q8_Q4KS,
    Q8_Q2_A,
    Q8_Q2_S,
    Q8_Q1_S,
    /// Adaptive compression level 0: near-lossless. K={Q8_KS,Q8_0,Q2_0}, V={Q8_0,Q4_0,Q2_0}.
    /// Most K blocks stay F16 fallback; V predominantly Q8_0.
    C0,
    /// Adaptive compression level 1. K={Q8_KS,Q8_0,Q4_1,Q2_0}, V={Q8_0,Q4_0,Q2_0}.
    /// K: Q8_0≈81%, small Q8_KS fraction; V: Q8_0≈93%.
    C1,
    /// Adaptive compression level 2. K={Q8_KS,Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_1,Q4_0,Q2_0}.
    /// K: Q8_0≈93%, Q4 formats emerging; V: Q4_1/Q4_0 mix.
    C2,
    /// Adaptive compression level 3. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_1,Q4_0,Q2_0}.
    /// K: Q8_0≈76%, Q4 mix≈21%; V: Q4_0 dominant.
    C3,
    /// Adaptive compression level 4. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_0,Q3_0,Q2_0}.
    /// K: Q4_0≈25%, Q8_0≈24%; V: Q4_0≈81%, Q3_0 entering.
    C4,
    /// Adaptive compression level 5. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q8_0,Q4_0,Q3_0,Q2_0}.
    /// K: Q4_0≈28%, Q3_0≈18%; V: Q3_0≈70%.
    C5,
    /// Adaptive compression level 6. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q3_0≈43%, Q4_0≈20%; V: Q3_0≈70%, Q2_0≈25%.
    C6,
    /// Adaptive compression level 7. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q3_0≈49%, Q2_0≈25%; V: Q2_0≈59%, Q3_0≈37%.
    C7,
    /// Adaptive compression level 8. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q2_0≈48%, Q3_0≈42%; V: Q2_0≈94%.
    C8,
    /// Adaptive compression level 9: maximum compression. K={Q8_0,Q4_KS,Q4_1,Q4_0,Q3_0,Q2_0}, V={Q4_1,Q4_0,Q3_0,Q2_0}.
    /// K: Q2_0≈64%, Q3_0≈30%; V: Q2_0≈95%.
    C9,
    /// Adaptive compression level 10: K same as C9, V pushed further.
    /// K: same as C9; V: Q2_0 dominant.
    C10,
}

impl InferenceMode {
    /// Get the compute dtype for activations.
    pub fn compute_dtype(&self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            // All quantized modes use F32 for compute to maintain precision
            Self::Q8_0
            | Self::Q4_0
            | Self::Q8_1
            | Self::Q4_1
            | Self::Q4_KS
            | Self::Q8_KS
            | Self::Q2_0
            | Self::Q3_0
            | Self::Q8_Q8KS
            | Self::Q8_Q4KS
            | Self::Q8_Q4
            | Self::Q8_Q2_A
            | Self::Q8_Q2_S
            | Self::Q8_Q1_S
            | Self::C0
            | Self::C1
            | Self::C2
            | Self::C3
            | Self::C4
            | Self::C5
            | Self::C6
            | Self::C7
            | Self::C8
            | Self::C9
            | Self::C10 => DType::F32,
            // R16 uses F16 compute (it stores F16 natively)
            Self::R16 => DType::F16,
        }
    }

    /// Get the KV cache storage format for K.
    ///
    /// Returns the first *quantized* candidate if one exists, otherwise the
    /// first entry.  For adaptive modes with float fallback (C0-C2) this
    /// returns the highest-fidelity quant (e.g. Q8_0) so that `storage_policy`
    /// produces `GpuQuant` and reconcile can route chunks through the adaptive
    /// selection kernel.  The float entry is a per-block fallback within the
    /// kernel, not the storage target.
    pub fn k_format(&self) -> KvFormat {
        let cands = self.k_candidates();
        cands
            .iter()
            .find(|f| f.is_quantized())
            .copied()
            .unwrap_or(cands[0])
    }

    /// Get the KV cache storage format for V.
    ///
    /// Same logic as [`k_format`]: returns first quantized candidate when
    /// available, falling back to the first entry for pure-float modes.
    pub fn v_format(&self) -> KvFormat {
        let cands = self.v_candidates();
        cands
            .iter()
            .find(|f| f.is_quantized())
            .copied()
            .unwrap_or(cands[0])
    }

    /// K cache candidate formats for per-block adaptive selection.
    ///
    /// K cache candidate formats for per-block adaptive selection.
    ///
    /// Ordered highest-to-lowest fidelity.  The selection kernel evaluates
    /// candidates in ascending BPE order (best compression first) and selects
    /// the most aggressive format whose magnitude-weighted error stays within
    /// threshold.  If no quantized candidate passes, the block stays F16.
    ///
    /// A single-element list means uniform application (no selection kernel).
    /// Multiple elements trigger the adaptive selection kernel.
    pub fn k_candidates(&self) -> Vec<KvFormat> {
        use KvFormat::{Float, Quantized as Q};
        use QuantFormat::*;

        if let Some(level) = self.compression_level() {
            return CompressionPolicy::production_candidates(level).0;
        }

        match self {
            // -- Composite models - K and V formats differ but are statically defined, no selection kernel --
            Self::Q8_Q4
            | Self::Q8_Q8KS
            | Self::Q8_Q4KS
            | Self::Q8_Q2_S
            | Self::Q8_Q2_A
            | Self::Q8_Q1_S => vec![Q(Q8_1)],
            // ── Uniform modes — single candidate ──
            Self::Q8_0 => vec![Q(Q8_0)],
            Self::Q8_1 => vec![Q(Q8_1)],
            Self::Q8_KS => vec![Q(Q8_KS)],
            Self::Q4_0 => vec![Q(Q4_0)],
            Self::Q4_1 => vec![Q(Q4_1)],
            Self::Q4_KS => vec![Q(Q4_KS)],
            Self::Q3_0 => vec![Q(Q3_0)],
            Self::Q2_0 => vec![Q(Q2_0)],
            Self::R16 => vec![Q(QuantFormat::R16)],
            // ── Float modes ──
            // Paged attention kernels store/read F16/BF16 only, so F32 KV is unrepresentable. The
            // F32 reference mode stores BF16 — matching the F32→BF16 compute collapse already in the
            // prefill/decode paths. Model compute follows the arena dtype (caches.dtype()).
            Self::F32 => vec![Float(DType::BF16)],
            Self::F16 => vec![Float(DType::F16)],
            Self::BF16 => vec![Float(DType::BF16)],
            _ => vec![Float(DType::F16)],
        }
    }

    /// V cache candidate formats for per-block adaptive selection.
    ///
    /// Mirror of [`k_candidates`] for the V cache.  Uses cosine distance
    /// as the error metric.  Always 4 candidates per adaptive level.
    pub fn v_candidates(&self) -> Vec<KvFormat> {
        use KvFormat::{Float, Quantized as Q};
        use QuantFormat::*;

        if let Some(level) = self.compression_level() {
            return CompressionPolicy::production_candidates(level).1;
        }

        match self {
            // -- Composite models - K and V formats differ but are statically defined, no selection kernel --
            Self::Q8_Q8KS => vec![Q(Q8_KS)],
            Self::Q8_Q4KS => vec![Q(Q4_KS)],
            Self::Q8_Q2_S => vec![Q(Q2_S)],
            Self::Q8_Q2_A => vec![Q(Q2_A)],
            Self::Q8_Q1_S => vec![Q(Q1_S)],
            Self::Q8_Q4 => vec![Q(Q4_1)],
            // ── Uniform modes — single candidate ──
            Self::Q8_0 => vec![Q(Q8_0)],
            Self::Q8_1 => vec![Q(Q8_1)],
            Self::Q8_KS => vec![Q(Q8_KS)],
            Self::Q4_0 => vec![Q(Q4_0)],
            Self::Q4_1 => vec![Q(Q4_1)],
            Self::Q4_KS => vec![Q(Q4_KS)],
            Self::Q3_0 => vec![Q(Q3_0)],
            Self::Q2_0 => vec![Q(Q2_0)],
            // R16 is K-only (K@F16 + Q-capture space). V uses plain F16.
            Self::R16 => vec![Float(DType::F16)],
            // ── Float modes ──
            // F32 KV is unrepresentable in the paged kernels (F16/BF16 only); store BF16 to match
            // the F32→BF16 compute collapse.
            Self::F32 => vec![Float(DType::BF16)],
            Self::F16 => vec![Float(DType::F16)],
            Self::BF16 => vec![Float(DType::BF16)],
            _ => vec![Float(DType::F16)],
        }
    }

    /// Get the compression level (0-10) for adaptive modes, or `None` for
    /// uniform/non-compression modes. Presence of a level signals that the
    /// adaptive selection kernel should run.
    pub fn compression_level(&self) -> Option<u8> {
        match self {
            Self::C0 => Some(0),
            Self::C1 => Some(1),
            Self::C2 => Some(2),
            Self::C3 => Some(3),
            Self::C4 => Some(4),
            Self::C5 => Some(5),
            Self::C6 => Some(6),
            Self::C7 => Some(7),
            Self::C8 => Some(8),
            Self::C9 => Some(9),
            Self::C10 => Some(10),
            _ => None,
        }
    }

    /// Check if this mode uses quantized KV cache.
    ///
    /// True when at least one candidate is a quantized format.
    pub fn is_quantized(&self) -> bool {
        self.k_candidates().iter().any(|f| f.is_quantized())
            || self.v_candidates().iter().any(|f| f.is_quantized())
    }
}

/// Configuration for batched inference sessions.
#[derive(Debug, Clone)]
pub struct BatchedConfig {
    /// Initial maximum sequence length (can grow dynamically).
    /// Default: 2048
    pub initial_seq_len: usize,
    /// Storage format for K cache (ceiling format for adaptive modes).
    pub k_format: KvFormat,
    /// Storage format for V cache (ceiling format for adaptive modes).
    pub v_format: KvFormat,
    /// Compression level (0-10) for adaptive per-block selection.
    /// `None` means uniform storage with no per-block selection (legacy modes
    /// like F16/Q8_0). `Some(level)` engages the adaptive selection kernel.
    pub compression_level: Option<u8>,
    /// Per-model multiplier for the K high adaptive threshold.
    pub k_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the K low adaptive threshold.
    pub k_low_error_threshold_factor: f32,
    /// Per-model multiplier for the V high (strict) adaptive threshold.
    pub v_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    pub v_low_error_threshold_factor: f32,
    /// When `Some(fmt)`, the persist-time quantizer forces K storage to
    /// uniform `fmt` with identity pal_map and unit outer scales, bypassing
    /// the selection kernel's K-side choices. V remains fully adaptive per
    /// selection. `None` lets selection's per-(chunk, head, palette) K
    /// state propagate to storage as designed.
    ///
    /// Set via `with_override_k_quant`. Propagates into the
    /// [`CompressionPolicy`] built by [`Self::compression_policy`].
    pub override_k_quant: Option<QuantFormat>,
    /// Symmetric V counterpart to [`Self::override_k_quant`]. Defaults to
    /// `None` because V's decode path already honors selection's full
    /// adaptive state correctly; provided for symmetry / diagnostic use.
    pub override_v_quant: Option<QuantFormat>,
}

impl Default for BatchedConfig {
    fn default() -> Self {
        Self {
            initial_seq_len: 2048,
            k_format: KvFormat::Float(DType::BF16),
            v_format: KvFormat::Float(DType::BF16),
            compression_level: None,
            k_hi_error_threshold_factor: 1.0,
            k_low_error_threshold_factor: 1.0,
            v_hi_error_threshold_factor: 1.0,
            v_low_error_threshold_factor: 1.0,
            override_k_quant: None,
            override_v_quant: None,
        }
    }
}

impl BatchedConfig {
    /// Use standard float dtype for KV storage (same for K and V).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.k_format = KvFormat::Float(dtype);
        self.v_format = KvFormat::Float(dtype);
        self.compression_level = None;
        self
    }

    /// Use quantized format for KV storage (same for K and V).
    pub fn with_quantized(mut self, format: QuantFormat) -> Self {
        self.k_format = KvFormat::Quantized(format);
        self.v_format = KvFormat::Quantized(format);
        self.compression_level = None;
        self
    }

    /// Set the initial sequence length.
    pub fn with_initial_seq_len(mut self, initial_seq_len: usize) -> Self {
        self.initial_seq_len = initial_seq_len;
        self
    }

    /// Set per-model K/V error threshold factors for adaptive compression.
    pub fn with_error_threshold_factors(
        mut self,
        k_hi_factor: f32,
        k_low_factor: f32,
        v_hi_factor: f32,
        v_low_factor: f32,
    ) -> Self {
        self.k_hi_error_threshold_factor = k_hi_factor;
        self.k_low_error_threshold_factor = k_low_factor;
        self.v_hi_error_threshold_factor = v_hi_factor;
        self.v_low_error_threshold_factor = v_low_factor;
        self
    }

    /// Force K storage to a uniform quant format with identity pal_map and
    /// unit outer scales, ignoring selection's K choices. V remains fully
    /// adaptive. `None` propagates selection's K state to storage.
    pub fn with_override_k_quant(mut self, fmt: Option<QuantFormat>) -> Self {
        self.override_k_quant = fmt;
        self
    }

    /// Symmetric V counterpart to [`Self::with_override_k_quant`].
    /// Defaults to `None` — V's decode path already honors selection's
    /// full adaptive state correctly; provided for symmetry / diagnostic use.
    pub fn with_override_v_quant(mut self, fmt: Option<QuantFormat>) -> Self {
        self.override_v_quant = fmt;
        self
    }

    /// Build a [`CompressionPolicy`] from this config's level (if set).
    /// Returns `None` for uniform/non-compression modes.
    pub fn compression_policy(&self) -> Option<CompressionPolicy> {
        self.compression_level.map(|level| {
            CompressionPolicy::new_with_error_threshold_factors(
                level,
                self.k_hi_error_threshold_factor,
                self.k_low_error_threshold_factor,
                self.v_hi_error_threshold_factor,
                self.v_low_error_threshold_factor,
            )
            .with_override_k_quant(self.override_k_quant)
            .with_override_v_quant(self.override_v_quant)
        })
    }
}

/// Per-sequence state managed by the batch session.
#[derive(Debug)]
struct SequenceState {
    /// Current position in the sequence (number of tokens processed).
    offset: usize,
    /// Whether this slot is currently in use.
    active: bool,
    /// KvCaches for this sequence (one cache per layer).
    caches: KvCaches,
}

/// Assemble a [`GlueMeta`] from the wave's per-slot column positions for a
/// gap-fill forward. `col_per_seq[i]` must have length `seq_offsets[i] +
/// input_lens[i]` (sealed prefix ++ glue) — the kernel reads it as the flat
/// `col_actual_pos` over every slot's `[0, kv_len)`. Returns `None` (falls back
/// to plain prefill) only when no slot actually carries glue.
#[cfg(feature = "cuda")]
fn build_glue_meta(
    col_per_seq: Vec<Vec<u32>>,
    seq_offsets: &[usize],
    input_lens: &[usize],
    device: &Device,
) -> Result<Option<GlueMeta>> {
    if col_per_seq.len() != seq_offsets.len() || col_per_seq.len() != input_lens.len() {
        candle::bail!(
            "build_glue_meta: {} col vecs vs {} offsets / {} input_lens",
            col_per_seq.len(),
            seq_offsets.len(),
            input_lens.len()
        );
    }
    let mut flat: Vec<u32> = Vec::with_capacity(col_per_seq.iter().map(|c| c.len()).sum());
    for (i, cols) in col_per_seq.iter().enumerate() {
        let expected = seq_offsets[i] + input_lens[i];
        if cols.len() != expected {
            candle::bail!(
                "build_glue_meta: slot {i} col_actual_pos len {} != kv_len {} (offset {} + glue {})",
                cols.len(),
                expected,
                seq_offsets[i],
                input_lens[i]
            );
        }
        flat.extend_from_slice(cols);
    }
    if flat.is_empty() {
        return Ok(None);
    }
    let n = flat.len();
    let col_actual_pos = Tensor::from_vec(flat, n, device)?;
    // Confirms the gap-fill forward took the paged-glue route (HD128) rather than
    // plain prefill, with the glue shape — one line per reproject. Logged under
    // the scheduler's reproject target so it rides alongside the reproject
    // summary in the normal log view.
    tracing::info!(
        target: "candle_conversation::scheduler::reproject",
        slots = col_per_seq.len(),
        total_glue = input_lens.iter().sum::<usize>(),
        max_prefix = seq_offsets.iter().copied().max().unwrap_or(0),
        "paged-glue route active"
    );
    Ok(Some(GlueMeta { col_actual_pos }))
}

/// Batched inference session that manages KV cache state for multiple sequences.
///
/// This is the main interface for batched inference. It:
/// - Manages the chunked KV cache backing for paged attention
/// - Owns KvCaches for each sequence
/// - Tracks per-sequence state (offsets, active status)
/// - Provides direct access for models implementing [`ManagedBatchedModel::forward_batched`]
pub struct BatchedInferenceSession {
    /// Shared KV cache backing for all layers.
    /// One backing per layer.
    backings: Vec<ChunkedKvBacking>,
    /// Per-sequence state, indexed by sequence index.
    sequences: Vec<Option<SequenceState>>,
    /// Configuration used to create this session.
    config: BatchedConfig,
    /// Number of layers in the model.
    num_layers: usize,
    /// Device the session is on.
    device: Device,
    /// Pending reprojection-glue column positions, set by the wave immediately
    /// before its gap-fill `forward_batched`. One entry per sequence (in the
    /// forward's `seq_indices` order); each is that slot's flat `col_actual_pos`
    /// (sealed prefix ++ glue, TRUE sequence positions). Taken + cleared inside
    /// `forward_batched`, which routes HD128 glue to the paged-glue kernel.
    pending_glue: Option<Vec<Vec<u32>>>,
}

impl BatchedInferenceSession {
    /// Create a new batched inference session.
    pub fn new(
        num_layers: usize,
        n_kv_head: usize,
        head_dim: usize,
        device: &Device,
        config: BatchedConfig,
    ) -> Result<Self> {
        // Create one backing and share it across all layers.
        // All layers have the same (n_kv_head, head_dim, format) so they
        // share arenas, the GID pool, and the arena table. Each layer gets
        // its own per-layer state (sequences, max_blocks).
        let first_backing = ChunkedKvBacking::new_with_format_adaptive(
            1, // Initial batch size of 1, will grow dynamically
            n_kv_head,
            head_dim,
            config.k_format,
            config.v_format,
            device,
            config.initial_seq_len,
            config.compression_policy(),
        )?;

        let mut backings = Vec::with_capacity(num_layers);
        backings.push(first_backing.clone());
        for layer_idx in 1..num_layers {
            backings.push(first_backing.new_layer(layer_idx, 1, config.initial_seq_len));
        }

        Ok(Self {
            backings,
            sequences: Vec::new(),
            config,
            num_layers,
            device: device.clone(),
            pending_glue: None,
        })
    }

    /// Create a session that shares the KV arena pool with an existing session.
    ///
    /// Each `ChunkedKvBacking` is an `Arc<BackingInner>`, so cloning shares the
    /// underlying storage, slot table, and arena table between this new session
    /// and `source`.  This is the correct way to create a sibling session (e.g. a
    /// float prototype session for boundary injection) that needs to borrow chunks
    /// into the main session via `append_borrowed_chunks_cow` — because that
    /// operation scans the same slot table to find chunk handles.
    ///
    /// Slot collisions are avoided by consulting the shared backing when picking
    /// a new slot index in `create_sequence` (see [`BatchedInferenceSession::create_sequence`]).
    pub fn new_with_backings(
        backings: Vec<ChunkedKvBacking>,
        config: BatchedConfig,
        device: &Device,
    ) -> Self {
        let num_layers = backings.len();
        Self {
            backings,
            sequences: Vec::new(),
            config,
            num_layers,
            device: device.clone(),
            pending_glue: None,
        }
    }

    /// Stage reprojection-glue column positions for the next `forward_batched`.
    /// `col_actual_pos_per_seq[i]` is sequence `i`'s flat `col_actual_pos`
    /// (sealed prefix ++ glue), aligned with the `seq_indices` of the imminent
    /// gap-fill forward. Consumed (and cleared) by that single forward.
    pub fn set_pending_glue(&mut self, col_actual_pos_per_seq: Vec<Vec<u32>>) {
        self.pending_glue = Some(col_actual_pos_per_seq);
    }

    /// Take + clear the staged glue column positions (one forward's worth).
    pub fn take_pending_glue(&mut self) -> Option<Vec<Vec<u32>>> {
        self.pending_glue.take()
    }

    /// Get the configuration used for this session.
    pub fn config(&self) -> &BatchedConfig {
        &self.config
    }

    /// Build the session's compression policy from its config.
    /// Returns `None` for uniform / non-compression modes.
    pub fn compression_policy(&self) -> Option<CompressionPolicy> {
        self.config.compression_policy()
    }

    /// Get the device the session is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the number of currently active sequences.
    pub fn active_sequence_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.as_ref().map_or(false, |s| s.active))
            .count()
    }

    /// Get the list of active sequence indices.
    pub fn active_sequences(&self) -> Vec<usize> {
        self.sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| {
                if s.as_ref().map_or(false, |s| s.active) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Create a new sequence and return its index.
    ///
    /// The sequence starts at offset 0 (empty KV cache).
    ///
    /// When multiple sessions share the same `BackingInner` pool (created via
    /// [`new_with_backings`](Self::new_with_backings)), the backing is the
    /// canonical slot allocator.  Reading the first free slot from the backing
    /// guarantees that sibling sessions — e.g. a proto session for boundary
    /// injection — get non-overlapping slot indices even though each session
    /// maintains its own `sequences` metadata vector.
    pub fn create_sequence(&mut self) -> Result<usize> {
        // Ask the first backing for the first unallocated slot.  This is correct
        // for both the single-session case (backing only has our own slots) and
        // the shared-pool case (backing may also have slots owned by a sibling
        // session, so we must skip those to avoid collisions).
        let idx = if let Some(backing) = self.backings.first() {
            backing.first_free_slot()?
        } else {
            // No backing — fall back to scanning our own sequences vector.
            self.sequences
                .iter()
                .position(|s| s.is_none())
                .unwrap_or(self.sequences.len())
        };

        // Ensure slot is allocated in all backings
        for backing in &self.backings {
            backing.ensure_sequence_allocated(idx)?;
        }

        // Create KvCaches for this sequence with chunked backing
        let caches = self.create_kv_caches_for_sequence(idx)?;

        // Grow the sequences vector to accommodate idx, filling any gap with None.
        // A gap arises when a sibling session owns the lower-numbered backing slots.
        if idx >= self.sequences.len() {
            self.sequences.resize_with(idx + 1, || None);
        }
        self.sequences[idx] = Some(SequenceState {
            offset: 0,
            active: true,
            caches,
        });

        Ok(idx)
    }

    /// Create KvCaches for a sequence, wiring up chunked backing.
    fn create_kv_caches_for_sequence(&self, seq_idx: usize) -> Result<KvCaches> {
        let compression = self.compression_policy();
        let mut caches: Vec<KvCache> = Vec::with_capacity(self.num_layers);
        for layer_idx in 0..self.num_layers {
            let mut cache = KvCache::new(2, 64); // Initial capacity, will use backing
                                                 // For float formats, set the dtype on the cache
            if let Some(dtype) = self.config.k_format.dtype() {
                cache.force_dtype(dtype);
            }
            if let Some(backing) = self.backings.get(layer_idx) {
                cache.set_chunked_backing(backing, seq_idx, compression.clone())?;
            }
            caches.push(cache);
        }
        Ok(new_kv_caches(caches, self.device.clone()))
    }

    /// Begin a pinned-stager generation guard.
    ///
    /// All backings share the same `PinnedStager`, so we only need to call this
    /// on the first one.  The returned RAII guard keeps the pinned arena alive
    /// until dropped, ensuring device-mapped pointers remain valid for the
    /// duration of a forward pass.
    pub fn begin_stager_generation(&self) -> Generation {
        self.backings
            .first()
            .expect("begin_stager_generation: no backings")
            .begin_stager_generation_required()
    }

    /// Build decode kernel metadata for ALL layers from scratch.
    ///
    /// Rebuilds slice and header data directly from backing arenas on every call.
    /// Pass `seq_indices` in the same order used for the forward pass batch.
    /// Returns per-layer slice tensors (must stay alive until after the kernel) and
    /// per-layer header GpuBufs. Both vecs are empty when `seq_indices` is empty.
    #[cfg(feature = "cuda")]
    pub fn build_decode_metadata(
        &self,
        seq_indices: &[usize],
        generation: &Generation,
    ) -> Result<(Option<GpuBuf>, Option<GpuBuf>, u64)> {
        let n_active = seq_indices.len();
        if n_active == 0 {
            return Ok((None, None, 0));
        }

        // 24-byte SlotHeader: n_slices, write_slice, slices_ptr, position_map_ptr.
        let header_stride = n_active * 24;
        let mut all_headers: Vec<u8> = Vec::with_capacity(self.num_layers * header_stride);

        // Pre-compute per-sequence offsets once (same for all layers).
        let seq_offsets: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&seq_idx| {
                let offset = self
                    .sequences
                    .get(seq_idx)
                    .and_then(|s| s.as_ref())
                    .map_or(0, |s| s.offset);
                (seq_idx, offset)
            })
            .collect();

        // Build per-sequence position_map covering [0, state.offset + 1).
        // Each entry is u32: (slice_idx << 16) | in_blk.  The map is
        // layer-invariant (slice metadata is uniform), built once, and every
        // layer's SlotHeader points into the per-sequence region.
        // Entry at index state.offset is the write slot for the new token.
        let mut pm_flat: Vec<u32> = Vec::new();
        let mut pm_seq_byte_offsets: Vec<usize> = Vec::with_capacity(n_active);
        // Ensure backings are sized for the upcoming decode write so the
        // slot's chunks reflect the post-write layout when we read them.
        self.backings[0].ensure_for_batch_entries(&seq_offsets, 1)?;
        for &(seq_idx, seq_offset) in &seq_offsets {
            let entry_start = pm_flat.len();
            pm_seq_byte_offsets.push(entry_start * 4);
            let chunks = self.backings[0]
                .live_chunks_as_sealed(seq_idx, &[])
                .unwrap_or_default();
            for (sidx, c) in chunks.iter().enumerate() {
                let base = (sidx as u32) << 16;
                pm_flat.extend(
                    (c.offset as u32..c.offset as u32 + c.token_count as u32)
                        .map(|in_blk| base | in_blk),
                );
            }
            debug_assert_eq!(
                pm_flat.len() - entry_start,
                seq_offset,
                "decode position_map: cum_tokens {} != state.offset {seq_offset} for seq {seq_idx}",
                pm_flat.len() - entry_start,
            );
            // Write-slot entry: the new token lands in the WRITE chunk — the
            // first non-full chunk from `writer_start_idx`, NOT `chunks.last()`
            // (which may be a trailing empty sitting past the writer). This MUST
            // match the `write_slice` rule in `sync_decode_gpu_chunks`, or the
            // kernel scatters the token into one chunk while attention is told
            // (via the position_map) to read it from another.
            let wstart = self.backings[0]
                .writer_start_idx_for_seq(seq_idx)
                .unwrap_or(0);
            let n_ch = chunks.len();
            let wi = if n_ch == 0 {
                0
            } else {
                let start = wstart.min(n_ch - 1);
                (start..n_ch)
                    .find(|&i| (chunks[i].offset as usize + chunks[i].token_count as usize) < 32)
                    .unwrap_or(n_ch - 1)
            };
            let wi_within = chunks
                .get(wi)
                .map_or(0, |c| c.offset as u32 + c.token_count as u32);
            pm_flat.push(((wi as u32) << 16) | wi_within);
        }

        // Upload position_map via the pinned stager — zero-copy PCIe read,
        // same path as all_headers below.  Pad to at least one entry so the
        // device pointer is always valid.
        if pm_flat.is_empty() {
            pm_flat.push(0);
        }
        let pm_byte_len = pm_flat.len() * std::mem::size_of::<u32>();
        let mut pm_pinned = generation.alloc(pm_byte_len)?;
        // SAFETY: u32 has no padding and is trivially copyable; the lengths match.
        let pm_bytes =
            unsafe { std::slice::from_raw_parts(pm_flat.as_ptr() as *const u8, pm_byte_len) };
        pm_pinned.copy_from_slice(pm_bytes);
        let pm_gpu_buf = generation.submit(pm_pinned)?;
        let pm_base_ptr = pm_gpu_buf.dev_ptr();

        let mut slot_rebuild_time = std::time::Duration::ZERO;
        let mut slot_reuse_time = std::time::Duration::ZERO;
        let mut saw_slot_rebuild = false;
        let mut saw_slot_reuse = false;

        for layer_idx in 0..self.num_layers {
            // Ensure the write chunk for each sequence is allocated before reading
            // metadata. At chunk boundaries the write chunk does not exist yet
            // (it is allocated lazily inside paged_decode_attention), so we must
            // pre-allocate it here so the GPU buffer reflects the correct
            // new write chunk rather than the previous sealed tail.
            self.backings[layer_idx].ensure_for_batch_entries(&seq_offsets, 1)?;

            let arena_info = self.backings[layer_idx].resolve_arena_info()?;

            // Incrementally sync each sequence's GPU slot-state buffer.
            // Common case: the cached GPU buffer is already valid and we only
            // reuse its pointer. Chunk-boundary case: the layer rebuilds once
            // from authoritative CPU chunk state.
            let (seq_ptrs, sync_stats) =
                self.backings[layer_idx].sync_decode_gpu_chunks(&seq_offsets, &arena_info)?;
            slot_reuse_time += sync_stats.reuse_time;
            slot_rebuild_time += sync_stats.rebuild_time;
            saw_slot_reuse |= sync_stats.reuses > 0;
            saw_slot_rebuild |= sync_stats.rebuilds > 0;

            // Append this layer's headers (24 bytes × n_active).
            for (i, &(ptr, n_slices, write_slice)) in seq_ptrs.iter().enumerate() {
                let pm_ptr = pm_base_ptr + pm_seq_byte_offsets[i] as u64;
                all_headers.extend_from_slice(&n_slices.to_le_bytes());
                all_headers.extend_from_slice(&write_slice.to_le_bytes());
                all_headers.extend_from_slice(&ptr.to_le_bytes());
                all_headers.extend_from_slice(&pm_ptr.to_le_bytes());
            }
        }

        pipeline_record_duration(
            "decode:slot_reuse",
            slot_reuse_time,
            u64::from(saw_slot_reuse),
        );
        pipeline_record_duration(
            "decode:slot_rebuild",
            slot_rebuild_time,
            u64::from(saw_slot_rebuild),
        );

        // Upload all layers' headers in a single pinned → GPU copy.
        let total = self.num_layers * header_stride;
        let mut pinned = generation.alloc(total)?;
        pinned.copy_from_slice(&all_headers);
        let gpu_buf = generation.submit(pinned)?;
        Ok((Some(pm_gpu_buf), Some(gpu_buf), header_stride as u64))
    }

    /// Free a sequence, returning its resources to the pool.
    ///
    /// The sequence index should not be used after this call.
    pub fn free_sequence(&mut self, idx: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }

        // Free in all backings
        for backing in &self.backings {
            backing.free_sequence(idx)?;
        }

        // Mark as inactive
        self.sequences[idx] = None;

        Ok(())
    }

    /// Fork a sequence, creating a new sequence that shares the KV cache prefix.
    ///
    /// This uses copy-on-write semantics - the forked sequence shares memory
    /// with the source until either is modified.
    pub fn fork_sequence(&mut self, source_idx: usize) -> Result<usize> {
        if source_idx >= self.sequences.len() {
            candle::bail!("invalid source sequence index {}", source_idx);
        }

        let source_state = self.sequences[source_idx]
            .as_ref()
            .ok_or_else(|| candle::Error::Msg(format!("sequence {} not allocated", source_idx)))?;
        if !source_state.active {
            candle::bail!("cannot fork inactive sequence {}", source_idx);
        }

        let seq_len = source_state.offset;

        // Allocate a new sequence
        let new_idx = self.create_sequence()?;

        // Fork in all backings
        for backing in &self.backings {
            backing.fork_sequence(source_idx, new_idx, seq_len)?;
        }

        // Set the offset to match source
        if let Some(ref mut state) = self.sequences[new_idx] {
            state.offset = seq_len;
        }

        Ok(new_idx)
    }

    /// Append the sealed chunks of `sealed_per_layer` onto the tail of
    /// `seq_idx` as live chunk windows.  Pure metadata — no DMA.
    /// Advances the sequence's logical token offset by the appended
    /// token count (sum of usages, taken from layer 0).
    ///
    /// Returns `(block_start, block_end)` from layer 0 (all layers
    /// land at the same range because the metadata is uniform).
    pub fn inject_sealed_at_tail(
        &mut self,
        seq_idx: usize,
        sealed_per_layer: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<(usize, usize)> {
        if sealed_per_layer.len() != self.backings.len() {
            candle::bail!(
                "inject_sealed_at_tail: got {} layers but session has {} backings",
                sealed_per_layer.len(),
                self.backings.len()
            );
        }
        let mut range = (0usize, 0usize);
        let mut tokens_added: usize = 0;
        for (i, (backing, sealed)) in self
            .backings
            .iter()
            .zip(sealed_per_layer.iter())
            .enumerate()
        {
            let r = backing.inject_sealed_at_tail(seq_idx, sealed)?;
            if i == 0 {
                range = r;
                tokens_added = sealed
                    .chunks
                    .iter()
                    .map(|c| c.token_count as usize)
                    .sum::<usize>();
            }
        }
        if tokens_added > 0 {
            if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
                state.offset += tokens_added;
            }
        }
        Ok(range)
    }

    /// Force-push a fresh empty writer chunk onto every layer's view
    /// of `seq_idx`.  Used by cumulative section ingest right after
    /// [`inject_sealed_at_tail`] populates a scratch slot with prefix
    /// sections' substrate chunks: the prefix's last partial chunk
    /// is Arc-shared with the substrate, so writing into it would
    /// mutate bytes other holders see as immutable.  Pushing a fresh
    /// chunk here makes the slot's tail a writer-owned chunk; the
    /// upcoming prefill writes start at that chunk's position 0
    /// (logical position = `prefix_token_count`).
    pub fn push_empty_writer_chunk(&mut self, seq_idx: usize) -> Result<()> {
        for backing in &self.backings {
            backing.push_empty_writer_chunk(seq_idx)?;
        }
        Ok(())
    }

    /// Truncate every backing's view of `seq_idx` to `block_count`
    /// chunks; reset the session's logical offset to the resulting
    /// token count (read from layer 0).
    ///
    /// Lets the scheduler reset a persistent conversation sequence to
    /// its system-prompt baseline before injecting the next turn's
    /// projection.
    pub fn truncate_sequence_to_blocks(
        &mut self,
        seq_idx: usize,
        block_count: usize,
    ) -> Result<()> {
        let mut new_offset: Option<usize> = None;
        for backing in &self.backings {
            let n = backing.truncate_sequence_to_blocks(seq_idx, block_count)?;
            if new_offset.is_none() {
                new_offset = Some(n);
            }
        }
        if let Some(o) = new_offset {
            if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
                state.offset = o;
            }
        }
        Ok(())
    }

    /// Quantize the live K/V chunks of each sequence in `seq_indices` in place
    /// using the session's compression policy, then re-seal.
    ///
    /// Mirrors the substrate scheduler's boundary quantize (`SealAction::Turn` /
    /// `PrimingProjection`): for every layer the live float chunks are
    /// snapshotted with [`ChunkedKvBacking::record_turn`], run through
    /// [`quantize_sealed_in_place`], and swapped back in via
    /// `truncate_sequence_to_blocks(0)` + [`Self::inject_sealed_at_tail`]. The
    /// quantized `SealedSequence`s are held across the truncate, so their
    /// refcounted arena slots stay alive through the swap while the source float
    /// arenas return to the pool.
    ///
    /// With `start_new_chunk`, a fresh writer chunk is pushed so the next writes
    /// (decode) land in a new chunk instead of appending to the now-immutable
    /// quantized tail. Already-quantized chunks pass through the quantizer's
    /// preserve bucket unchanged, so this is safe to call after both prefill and
    /// decode. No-op when the mode is uncompressed, the device is not CUDA, or
    /// `head_dim != 128` (the palette4 quantizer requires 128).
    #[cfg(feature = "cuda")]
    pub fn quantize_and_seal_sequences(
        &mut self,
        seq_indices: &[usize],
        start_new_chunk: bool,
    ) -> Result<()> {
        use candle::quantized::pinned_staging::PinnedBuf;
        use candle_nn::kv_cache::{quantize_sealed_in_place, SealedSequence};

        // Quantization policy, or `None` for uncompressed (F16/BF16). When it's
        // `None` we still SEAL + collapse the layout below (record_turn +
        // re-inject + fresh writer) — only the quantize step is skipped. This is
        // what keeps repeated prefills from leaving phantom empty chunks behind
        // for float modes (which the quantized path already collapses).
        let policy: Option<CompressionPolicy> = match self.compression_policy() {
            Some(p) => Some(p),
            None => {
                // Legacy uniform-quant modes (Q8_0, Q4_0, …) carry no
                // compression_level but a fixed Quantized K/V format. Drive the
                // quantizer with a forced uniform override so they still
                // compress to that format. The level (5) only feeds the
                // discarded candidate scaffolding — both overrides bypass
                // per-block selection entirely.
                match self.config.k_format.as_quant() {
                    Some(k_fmt) => {
                        let v_fmt = self.config.v_format.as_quant().unwrap_or(k_fmt);
                        Some(
                            CompressionPolicy::new_with_error_threshold_factors(
                                5, 1.0, 1.0, 1.0, 1.0,
                            )
                            .with_override_k_quant(Some(k_fmt))
                            .with_override_v_quant(Some(v_fmt)),
                        )
                    }
                    // Uncompressed (F16/BF16): no quantization, but still seal.
                    None => None,
                }
            }
        };
        let copy_stream = match &self.device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => return Ok(()),
        };
        // The fused palette4 quantizer requires head_dim == 128. If it isn't, we
        // still collapse the layout below — just without quantizing.
        let policy = if self.backings.first().map(|b| b.head_dim()) == Some(128) {
            policy
        } else {
            None
        };

        let mut scratch: Option<PinnedBuf> = None;
        for &seq_idx in seq_indices {
            // Snapshot + quantize each layer. Holding the outputs across the
            // truncate below keeps their (refcounted) arena slots alive.
            let mut quantized_per_layer: Vec<SealedSequence> =
                Vec::with_capacity(self.backings.len());
            for backing in &self.backings {
                let live = backing.record_turn(seq_idx)?;
                if live.chunks.is_empty() || policy.is_none() {
                    quantized_per_layer.push(live);
                    continue;
                }
                let out = quantize_sealed_in_place(
                    backing,
                    &[&live],
                    policy.as_ref().unwrap(),
                    &self.device,
                    &copy_stream,
                    &mut scratch,
                )?;
                let q = out.into_iter().next().ok_or_else(|| {
                    candle::Error::Msg(
                        "quantize_and_seal_sequences: quantizer returned no sequence".into(),
                    )
                })?;
                quantized_per_layer.push(q);
            }
            // Convert kernels must finish before we drop the source float chunks
            // (truncate) and before decode reads the quantized bytes. No convert
            // kernel runs when quantization was skipped, so only sync then.
            if policy.is_some() {
                copy_stream
                    .synchronize()
                    .map_err(|e| candle::Error::Msg(format!("quantize sync: {e}")))?;
                // Quantized modes must replace the float chunks with the quantized
                // bytes: truncate the live float layout and re-inject the sealed
                // quantized chunks.
                self.truncate_sequence_to_blocks(seq_idx, 0)?;
                self.inject_sealed_at_tail(seq_idx, &quantized_per_layer)?;
            }
            // Uncompressed (F16/BF16) modes keep their float chunks in place — the
            // prefill's `truncate_caches_to_offset` already collapses phantom
            // chunks from repeated re-prefill, so the record_turn/truncate/re-inject
            // round-trip is unnecessary (and re-injecting Arc-shared chunks under a
            // lone sequence corrupted the decode read). Just open a fresh writer.
            if start_new_chunk {
                self.push_empty_writer_chunk(seq_idx)?;
            }
        }
        Ok(())
    }

    /// Cheap session-level compaction check.
    ///
    /// This only inspects backing state and does not move or free anything.
    pub fn compact_check(&self) -> Result<bool> {
        let t_check = profile_now();
        let mut should_run = false;
        for backing in &self.backings {
            should_run |= backing.needs_compaction()?;
        }
        pipeline_record("session:kv_compact_check", t_check);
        Ok(should_run)
    }

    /// Compact all arena backings to release unused tail arenas.
    ///
    /// Call this after freeing sequences to reclaim GPU memory.
    /// Returns the total number of arenas freed across all layers.
    pub fn compact(&self) -> Result<usize> {
        if !self.compact_check()? {
            return Ok(0);
        }
        let t_compact = profile_now();
        let mut total_freed = 0;
        for backing in &self.backings {
            total_freed += backing.compact()?;
        }
        pipeline_record("session:kv_compact_run", t_compact);
        Ok(total_freed)
    }

    /// Force compaction across all backings (defrag threshold 0) and release
    /// reclaimed arenas. Used by the scheduler's VRAM-pressure backpressure
    /// path, where reclaiming any arena is worth it. Returns arenas freed.
    pub fn compact_forced(&self) -> Result<usize> {
        let mut total_freed = 0;
        for backing in &self.backings {
            total_freed += backing.compact_forced()?;
        }
        Ok(total_freed)
    }

    /// Release fully-empty KV arenas across all backings **without** the
    /// chunk-moving defrag — cheap VRAM relief for the scheduler's pressure
    /// path (the costly speculative defrag is left to the allocation-time OOM
    /// retry). Returns arenas freed.
    pub fn release_empty_arenas(&self) -> Result<usize> {
        let mut total_freed = 0;
        for backing in &self.backings {
            total_freed += backing.release_empty_arenas()?;
        }
        Ok(total_freed)
    }

    /// Device VRAM `(free, total)` in bytes, or `None` on non-CUDA devices /
    /// query failure. Drives the scheduler's VRAM-pressure admission gate.
    pub fn vram_free_total(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(d) = &self.device {
                return d.mem_get_info().ok();
            }
        }
        None
    }

    /// Pool-aware KV-VRAM budget headroom in bytes (`init_free - pool_used -
    /// reserve`), or `None` on non-CUDA / query failure. Unlike
    /// [`Self::vram_free_total`] (the volatile driver `cuMemGetInfo` free, which
    /// our stream-ordered pool's reserved-but-free memory hides from and which
    /// WDDM pollutes with other processes' resident memory), this counts only
    /// *our* live footprint — so KV freed back into the pool registers as
    /// headroom. This is the number the per-arena budget gate already uses.
    pub fn vram_budget_available(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        return candle_nn::kv_cache::vram_budget_available(&self.device);
        #[cfg(not(feature = "cuda"))]
        return None;
    }

    /// True when a forced compaction could free at least one whole arena from
    /// any KV backing. When false, the cache is packed to within a single arena
    /// of free space and compaction would reclaim nothing — the scheduler uses
    /// this to skip a futile compaction pass under VRAM pressure.
    pub fn can_reclaim_arena(&self) -> bool {
        self.backings.iter().any(|b| b.can_reclaim_arena())
    }

    /// Our CUDA memory pool's `(used, reserved)` bytes — what our allocations
    /// actually occupy (model weights + KV + activations) vs the high-water
    /// bytes the pool has reserved from the OS. `reserved - used` is held but
    /// reusable; `reserved` is why the driver's `free` reads near zero. The
    /// real diagnostic for "what's using VRAM". `None` on non-CUDA / failure.
    pub fn vram_pool_stats(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(d) = &self.device {
                if let (Ok(used), Ok(reserved)) = (d.pool_used_bytes(), d.pool_reserved_bytes()) {
                    return Some((used, reserved));
                }
            }
        }
        None
    }

    /// Create a view sequence that borrows KV blocks from a parent.
    ///
    /// Allocates a new sequence slot and populates it with Arc-shared refs to
    /// the specified parent blocks.  Callers write new tokens into the view, then
    /// call [`finalize_view`] to transfer those blocks back to the parent.
    pub fn create_view_sequence(
        &mut self,
        parent_idx: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<ViewSequence> {
        let view_idx = self.create_sequence()?;
        let mut borrowed_block_count = 0;
        let mut borrowed_token_count = 0;
        let mut first = true;
        for backing in &self.backings {
            let (n_blks, n_toks) =
                backing.create_view_sequence(view_idx, parent_idx, visible_block_ranges)?;
            if first {
                borrowed_block_count = n_blks;
                borrowed_token_count = n_toks;
                first = false;
            }
        }
        // The view's write cursor starts right after the borrowed tokens.
        let view_offset = borrowed_token_count;
        if let Some(ref mut state) = self.sequences[view_idx] {
            state.offset = view_offset;
        }
        Ok(ViewSequence {
            view_idx,
            borrowed_block_count,
        })
    }

    /// Transfer newly-written view blocks to the parent and free the view slot.
    ///
    /// `original_view_block_count` is the value returned by [`create_view_sequence`].
    pub fn finalize_view(
        &mut self,
        view_idx: usize,
        parent_idx: usize,
        original_view_block_count: usize,
    ) -> Result<()> {
        // Copy view's final offset to parent before any mutation
        let view_final_offset = self
            .sequences
            .get(view_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.offset)
            .unwrap_or(0);

        for backing in &self.backings {
            backing.finalize_view(view_idx, parent_idx, original_view_block_count)?;
        }

        // Update parent offset to the view's final write position
        if let Some(Some(parent_state)) = self.sequences.get_mut(parent_idx) {
            parent_state.offset = view_final_offset;
        }
        // Remove view slot from session (backing already freed it)
        if view_idx < self.sequences.len() {
            self.sequences[view_idx] = None;
        }
        Ok(())
    }

    /// Get the current offset (position) of a sequence.
    pub fn sequence_offset(&self, idx: usize) -> Option<usize> {
        self.sequences.get(idx)?.as_ref().map(|s| s.offset)
    }

    /// Get the chunk count of a sequence slot — the authoritative
    /// block total used to set borrow ranges, seal ranges, and any
    /// other "how many blocks does this slot have" question.
    ///
    /// **Why not `sequence_offset(idx) / CHUNK_SIZE`?** That formula
    /// matches only when every chunk in the slot is full.  After
    /// `inject_sealed_at_tail` materialises multiple sealed sections
    /// back-to-back, each section's trailing partial chunk stays a
    /// separate `ChunkWindow` — token-count / `CHUNK_SIZE`
    /// under-counts the real chunk total.  Using the divided value
    /// as a borrow-range bound leaves the slot's tail (one chunk
    /// per concatenated section) invisible to the view, which
    /// silently drops up to `(sections - 1) * (CHUNK_SIZE - 1)`
    /// tokens of context.
    pub fn sequence_block_count(&self, idx: usize) -> Option<usize> {
        // All backings share the same chunk metadata layout (number
        // of chunks and per-chunk usages are uniform across layers),
        // so reading from layer 0 is authoritative.
        self.backings.first()?.sequence_block_count(idx)
    }

    /// Get mutable access to a sequence's KvCaches.
    pub fn sequence_caches_mut(&mut self, idx: usize) -> Option<&mut KvCaches> {
        self.sequences.get_mut(idx)?.as_mut().map(|s| &mut s.caches)
    }

    /// Get immutable access to a sequence's KvCaches.
    pub fn sequence_caches(&self, idx: usize) -> Option<&KvCaches> {
        self.sequences.get(idx)?.as_ref().map(|s| &s.caches)
    }

    /// Get mutable references to all active sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for all active sequences.
    /// This allows safe simultaneous mutable access to all caches.
    pub fn active_caches_mut(&mut self) -> Vec<(usize, usize, &mut KvCaches)> {
        self.sequences
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, slot)| {
                slot.as_mut().and_then(|state| {
                    if state.active {
                        Some((idx, state.offset, &mut state.caches))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get mutable references to specific sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for the requested sequences.
    /// The order matches the order of seq_indices.
    pub fn caches_for_sequences(&self, seq_indices: &[usize]) -> Vec<(usize, usize, &KvCaches)> {
        // Create a set of requested indices for O(1) lookup
        let requested: HashSet<usize> = seq_indices.iter().copied().collect();

        // Collect all matching sequences
        let mut result: Vec<(usize, usize, &KvCaches)> = self
            .sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if requested.contains(&idx) {
                    slot.as_ref()
                        .map(|state| (idx, state.offset, &state.caches))
                } else {
                    None
                }
            })
            .collect();

        // Sort by the order in seq_indices
        let index_order: HashMap<usize, usize> = seq_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        result.sort_by_key(|(idx, _, _)| index_order.get(idx).copied().unwrap_or(usize::MAX));

        result
    }

    /// Get mutable references to specific sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for the requested sequences.
    /// The order matches the order of seq_indices.
    pub fn caches_for_sequences_mut(
        &mut self,
        seq_indices: &[usize],
    ) -> Vec<(usize, usize, &mut KvCaches)> {
        // Create a set of requested indices for O(1) lookup
        let requested: HashSet<usize> = seq_indices.iter().copied().collect();

        // Collect all matching sequences
        let mut result: Vec<(usize, usize, &mut KvCaches)> = self
            .sequences
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if requested.contains(&idx) {
                    slot.as_mut()
                        .map(|state| (idx, state.offset, &mut state.caches))
                } else {
                    None
                }
            })
            .collect();

        // Sort by the order in seq_indices
        let index_order: HashMap<usize, usize> = seq_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        result.sort_by_key(|(idx, _, _)| index_order.get(idx).copied().unwrap_or(usize::MAX));

        result
    }

    /// Reset a sequence to the beginning (offset 0).
    ///
    /// This is more efficient than free+create for reusing a sequence slot.
    pub fn reset_sequence(&mut self, idx: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }

        // Free and re-allocate in all backings to clear KV data
        for backing in &self.backings {
            backing.free_sequence(idx)?;
            backing.ensure_sequence_allocated(idx)?;
        }

        // Create fresh caches (without borrowing sequences mutably)
        let caches = self.create_kv_caches_for_sequence(idx)?;

        // Now update the sequence state
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset = 0;
            state.caches = caches;
        }

        Ok(())
    }

    /// Get the KV cache K format.
    pub fn k_format(&self) -> KvFormat {
        self.config.k_format
    }

    /// Get the KV cache V format.
    pub fn v_format(&self) -> KvFormat {
        self.config.v_format
    }

    /// Get the dtype used for KV cache storage (for float formats only).
    ///
    /// Returns BF16 for quantized formats as a default compute dtype.
    pub fn dtype(&self) -> DType {
        self.config.k_format.dtype().unwrap_or(DType::BF16)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the KV backing for a specific layer.
    pub fn backing(&self, layer: usize) -> Option<&ChunkedKvBacking> {
        self.backings.get(layer)
    }

    /// Get all backings (one per layer).
    pub fn backings(&self) -> &[ChunkedKvBacking] {
        &self.backings
    }

    /// Dump raw K, V, and Q float data for R16 chunks of a sequence.
    ///
    /// Returns `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` for every
    /// sealed KV block in `seq_idx` whose K arena is R16.  Blocks stored in
    /// other formats (float, quantized non-R16) are silently skipped.
    ///
    /// Layout of each flat vec: `[head][palette][token][sub_dim]` —
    /// `n_kv_head × N_PALETTE × CHUNK_SIZE × sub_head_dim` values.
    ///
    /// Returns an empty vec when R16 is not in use or the sequence has no
    /// sealed blocks.  `layer_idx` selects which layer's backing to read;
    /// layer 0 is typical for provenance indexing.
    /// Dump R16 KV chunk data for the requested block range.
    ///
    /// Returns one `(block_idx, k_flat, v_flat, q_flat)` per R16 chunk
    /// whose K side resides in an R16 quantized arena.  Non-R16 chunks
    /// are skipped (creating gaps); each tuple carries the absolute
    /// block index so callers can reconstruct the position.
    ///
    /// `block_range = None` walks every chunk in the sequence (legacy
    /// full-sequence semantics — used by turn-end signature extraction
    /// and KV-data debug dumps); `Some((lo, hi))` walks only chunks
    /// in `[lo, hi)`, which is what the scheduler's mid-decode
    /// re-projection wants (probe window is at most a handful of
    /// chunks regardless of context depth).
    pub fn dump_r16_kv_for_provenance(
        &self,
        seq_idx: usize,
        layer_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        match self.backings.get(layer_idx) {
            Some(backing) => backing.dump_sequence_r16_kv_chunks(seq_idx, block_range),
            None => Ok(vec![]),
        }
    }

    /// Fast CUDA gather for provenance: single kernel + one DtoH copy.
    ///
    /// Replaces `dump_r16_kv_for_provenance` in the mid-decode reproject path,
    /// where N synchronous `memcpy_dtov` calls per `\n` token caused ~2 t/s decode.
    /// Falls back to the slow path on CPU or if V is not float-F16.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_for_provenance(
        &self,
        seq_idx: usize,
        layer_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        match self.backings.get(layer_idx) {
            Some(backing) => backing.gather_r16_kv_probe(seq_idx, block_range),
            None => Ok(vec![]),
        }
    }

    /// Gather R16 KV data for multiple provenance layers in one call.
    ///
    /// Calls `gather_r16_kv_probe` for each entry in `layer_indices` and
    /// returns results in the same order.  Exists so callers can retrieve
    /// all three provenance layers (syn, sem, prag) through a single API
    /// boundary rather than three separate calls.
    ///
    /// Each layer's probe is sequential (HtoD → kernel → DtoH per layer).
    /// Falls back to the slow path on CPU.  Returns one result vec per entry
    /// in `layer_indices`.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_provenance_layers(
        &self,
        seq_idx: usize,
        layer_indices: &[usize],
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Vec<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>>> {
        layer_indices
            .iter()
            .map(|&layer_idx| match self.backings.get(layer_idx) {
                Some(backing) => backing.gather_r16_kv_probe(seq_idx, block_range),
                None => Ok(vec![]),
            })
            .collect()
    }

    /// Per-chunk `(offset, len, cum_before)` real-token window for a sequence — the
    /// exact layout attention reads, so a provenance / diagnostic gather can check
    /// only real slots and skip partial-chunk padding. Chunk structure is identical
    /// across layer backings, so layer 0 is authoritative.
    pub fn provenance_chunk_layout(
        &self,
        seq_idx: usize,
        seq_offset: usize,
    ) -> Vec<(u16, u16, usize)> {
        self.backings
            .first()
            .map(|b| b.provenance_chunk_layout(seq_idx, seq_offset))
            .unwrap_or_default()
    }

    /// Number of KV heads in this session's backing.
    pub fn n_kv_head(&self) -> usize {
        self.backings.first().map(|b| b.n_kv_head()).unwrap_or(0)
    }

    /// Head dimension in this session's backing.
    pub fn head_dim(&self) -> usize {
        self.backings.first().map(|b| b.head_dim()).unwrap_or(0)
    }

    /// Print a compact per-chunk palette4 format distribution for layer 0 of the given sequence.
    ///
    /// For each chunk, tallies how many (head × palette) K and V GID slots use each
    /// arena format. A chunk with mixed formats indicates active per-palette routing.
    #[cfg(feature = "verbose")]
    pub fn print_palette4_stats(&self, seq_idx: usize) {
        use candle_nn::kv_cache::{SampleFormat, N_PALETTE};

        let backing = match self.backings.first() {
            Some(b) => b,
            None => {
                println!("[pal4] seq={seq_idx}: no backings");
                return;
            }
        };

        let arena_infos = match backing.resolve_arena_info() {
            Ok(a) => a,
            Err(e) => {
                println!("[pal4] seq={seq_idx}: resolve_arena_info failed: {e}");
                return;
            }
        };
        let chunks = match backing.live_chunks_as_sealed(seq_idx, &arena_infos) {
            Some(c) => c,
            None => {
                println!("[pal4] seq={seq_idx}: not allocated");
                return;
            }
        };
        if chunks.is_empty() {
            println!("[pal4] seq={seq_idx}: no chunks");
            return;
        }

        let n_kv_head = backing.n_kv_head();
        let head_dim = backing.head_dim();

        // Snapshot arena format info once. Kept strongly typed — string
        // conversion happens only at the point where a label is needed
        // (tally display or grid short-label).
        let fmt_map: Vec<Option<KvFormat>> = backing
            .with_arenas(|arenas| {
                let max_idx = arenas.keys().max().copied().unwrap_or(0);
                let mut v = vec![None; max_idx + 1];
                for (&idx, arena) in arenas.iter() {
                    v[idx] = Some(arena.format());
                }
                v
            })
            .unwrap_or_default();

        let gid_fmt = |gid: &candle_nn::kv_cache::ChunkGid| -> Option<KvFormat> {
            if gid.is_empty() {
                None
            } else {
                fmt_map.get(gid.arena_idx()).copied().flatten()
            }
        };
        let gid_is_quant = |gid: &candle_nn::kv_cache::ChunkGid| -> bool {
            !gid.is_empty()
                && matches!(gid_fmt(gid), Some(KvFormat::Quantized(qf)) if qf != QuantFormat::R16)
        };

        // Full label for tally/headline output (e.g. "Q8_0", "F16", "NULL").
        let fmt_long_label = |fmt: Option<KvFormat>| -> String {
            match fmt {
                None => "NULL".to_string(),
                Some(KvFormat::Float(dt)) => format!("{:?}", dt),
                Some(KvFormat::Quantized(qf)) => format!("{:?}", qf),
            }
        };

        // 3-char grid label. Delegates to `SampleFormat::grid_label` for the
        // formats it knows about and falls back to the 3-char prefix for any
        // others (R16, exotic dtypes, etc.).
        let fmt_grid_label = |fmt: Option<KvFormat>| -> String {
            match fmt {
                None => "NUL".to_string(),
                Some(KvFormat::Quantized(QuantFormat::R16)) => "R16".to_string(),
                Some(kv) => SampleFormat::from_kv_format(kv)
                    .map(|sf| sf.grid_label().to_string())
                    .unwrap_or_else(|| fmt_long_label(Some(kv)).chars().take(3).collect()),
            }
        };

        // Compact display for a tally: "Q8_0" if uniform, or "F16×4 Q8_0×28" if mixed.
        let tally_desc = |counts: &std::collections::HashMap<Option<KvFormat>, usize>| -> String {
            if counts.len() == 1 {
                fmt_long_label(*counts.keys().next().unwrap())
            } else {
                let mut parts: Vec<(String, usize)> = counts
                    .iter()
                    .map(|(k, v)| (fmt_long_label(*k), *v))
                    .collect();
                parts.sort_by(|a, b| a.0.cmp(&b.0));
                parts
                    .iter()
                    .map(|(k, v)| format!("{k}×{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };

        // Palette deviation: for one head's pal slice, count how many of the head_dim
        // per-dimension 2-bit palette assignments differ from the identity pattern
        // (identity: dim d → palette d/sub_hd).  Returns 0 for an exact identity map
        // and a positive value for any non-identity layout (including balanced shuffles
        // like d%N_PALETTE which have equal per-palette counts but different dim routing).
        let pal_bytes_per_head = (head_dim / 4).max(1);
        let sub_hd = (head_dim / N_PALETTE).max(1);
        let pal_head_deviation = |pal_data: &[u8], h: usize| -> u64 {
            if pal_data.is_empty() {
                return 0; // empty = shared identity palette, deviation = 0
            }
            let base = h * pal_bytes_per_head;
            let end = (base + pal_bytes_per_head).min(pal_data.len());
            let slice = &pal_data[base..end];
            let mut diff_count: u64 = 0;
            for d in 0..head_dim {
                let byte_idx = d / 4;
                let actual_p = if byte_idx < slice.len() {
                    ((slice[byte_idx] >> ((d % 4) * 2)) & 0x3) as usize
                } else {
                    (d / sub_hd).min(N_PALETTE - 1)
                };
                let identity_p = (d / sub_hd).min(N_PALETTE - 1);
                if actual_p != identity_p {
                    diff_count += 1;
                }
            }
            diff_count
        };

        println!(
            "[pal4] seq={seq_idx}  n_chunks={}  n_kv_head={n_kv_head}  N_PALETTE={N_PALETTE}",
            chunks.len()
        );

        // Chunk selection for the dim grid: last chunk that uses a real quant format.
        // R16 is intentionally excluded here.
        let mut last_q_ci: usize = chunks.len().saturating_sub(1);
        let mut last_q_k_pct: u32 = 100;
        let mut last_q_v_pct: u32 = 100;
        let mut last_q_k_grid: Vec<Vec<String>> = Vec::new();
        let mut last_q_v_grid: Vec<Vec<String>> = Vec::new();
        let mut last_q_k_pal_data: Vec<u8> = Vec::new();
        let mut last_q_v_pal_data: Vec<u8> = Vec::new();
        let mut last_q_k_scales: Vec<f32> = Vec::new();
        let mut last_q_v_scales: Vec<f32> = Vec::new();

        for (ci, chunk) in chunks.iter().enumerate() {
            let mut k_counts: std::collections::HashMap<Option<KvFormat>, usize> =
                std::collections::HashMap::new();
            let mut v_counts: std::collections::HashMap<Option<KvFormat>, usize> =
                std::collections::HashMap::new();
            let mut hd_k: Vec<Vec<String>> = Vec::with_capacity(n_kv_head);
            let mut hd_v: Vec<Vec<String>> = Vec::with_capacity(n_kv_head);
            let mut has_quant = false;
            for h in 0..n_kv_head {
                let mut hk_row = Vec::with_capacity(N_PALETTE);
                let mut hv_row = Vec::with_capacity(N_PALETTE);
                for p in 0..N_PALETTE {
                    let k_gid = chunk.gids.k_gid_pal(h, p);
                    let v_gid = chunk.gids.v_gid_pal(h, p);
                    let kf = gid_fmt(k_gid);
                    let vf = gid_fmt(v_gid);
                    has_quant |= gid_is_quant(k_gid) || gid_is_quant(v_gid);
                    hk_row.push(fmt_grid_label(kf));
                    hv_row.push(fmt_grid_label(vf));
                    *k_counts.entry(kf).or_insert(0) += 1;
                    *v_counts.entry(vf).or_insert(0) += 1;
                }
                hd_k.push(hk_row);
                hd_v.push(hv_row);
            }

            let k_nonid = (0..n_kv_head)
                .filter(|&h| pal_head_deviation(&chunk.k_pal, h) > 0)
                .count();
            let v_nonid = (0..n_kv_head)
                .filter(|&h| pal_head_deviation(&chunk.v_pal, h) > 0)
                .count();
            // k_pct = % of heads with identity palette (100 = all identity).
            let k_pct = 100u32.saturating_sub((k_nonid * 100 / n_kv_head.max(1)) as u32);
            let v_pct = 100u32.saturating_sub((v_nonid * 100 / n_kv_head.max(1)) as u32);
            if has_quant {
                last_q_ci = ci;
                last_q_k_pct = k_pct;
                last_q_v_pct = v_pct;
                last_q_k_grid = hd_k;
                last_q_v_grid = hd_v;
                last_q_k_pal_data = chunk.k_pal.as_ref().clone();
                last_q_v_pal_data = chunk.v_pal.as_ref().clone();
                last_q_k_scales = chunk.k_scale.as_ref().clone();
                last_q_v_scales = chunk.v_scale.as_ref().clone();
            }
            let pct_tag = match (k_pct, v_pct) {
                (100, 100) => String::new(),
                (k, v) => format!("  K={k}% V={v}%"),
            };
            println!(
                "  blk[{ci:3}] toks={:2}  K: {:12}  V: {}{}",
                chunk.token_count,
                tally_desc(&k_counts),
                tally_desc(&v_counts),
                pct_tag
            );
        }

        // Grid view of the last quantized chunk in the sequence.
        if !chunks.is_empty() && !last_q_k_grid.is_empty() {
            let pal_bytes = pal_bytes_per_head;

            let pal_hdr: String = (0..N_PALETTE)
                .map(|p| format!("  p{p}"))
                .collect::<Vec<_>>()
                .join("");
            println!(
                "  grid blk[{last_q_ci:3}] K={last_q_k_pct}% V={last_q_v_pct}%  \
                 (K: {pal_hdr}  | V: {pal_hdr})"
            );
            for h in 0..n_kv_head {
                let k_row: String = last_q_k_grid[h]
                    .iter()
                    .map(|s| format!("{s:>3}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                let v_row: String = last_q_v_grid[h]
                    .iter()
                    .map(|s| format!("{s:>3}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                println!("  h{h:2}: {k_row}  |  {v_row}");
            }
            // Per-head K and V scales (1.0 = no post-dequant scaling).
            let scale_val = |scales: &[f32], h: usize, p: usize| -> f32 {
                scales.get(h * N_PALETTE + p).copied().unwrap_or(1.0)
            };
            for h in 0..n_kv_head {
                let k_sc: String = (0..N_PALETTE)
                    .map(|p| format!("{:.3}", scale_val(&last_q_k_scales, h, p)))
                    .collect::<Vec<_>>()
                    .join("  ");
                let v_sc: String = (0..N_PALETTE)
                    .map(|p| format!("{:.3}", scale_val(&last_q_v_scales, h, p)))
                    .collect::<Vec<_>>()
                    .join("  ");
                println!("  sc{h:2}: {k_sc}  |  {v_sc}");
            }

            // Per-dim palette grid: for K and V, find the head with the largest
            // deviation from the identity mapping and show which sub-band each of the
            // head_dim dims is assigned to. The key shows which format each sub-band uses.
            let print_dim_grid = |side: &str, pal_data: &[u8], hd_grid: &[Vec<String>]| {
                // Pick the head with the most non-identity palette byte assignment.
                let best_h = (0..n_kv_head)
                    .max_by_key(|&h| pal_head_deviation(pal_data, h))
                    .unwrap_or(0);

                // Key: show the format label for each sub-band index in this head.
                // p0=fmt0 p1=fmt1 p2=fmt2 p3=fmt3
                let key: String = (0..N_PALETTE)
                    .map(|p| {
                        let lbl = hd_grid
                            .get(best_h)
                            .and_then(|row| row.get(p))
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!("p{p}={lbl}")
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  pal {side}  blk[{last_q_ci:3}] h{best_h}  key: {key}");

                let base = best_h * pal_bytes;
                let end = (base + pal_bytes).min(pal_data.len());
                let slice: &[u8] = if base < end {
                    &pal_data[base..end]
                } else {
                    &[]
                };

                // Each cell shows the raw sub-band index (0/1/2/3) for that dim.
                // The key shows which format each sub-band uses.
                // For the identity palette this produces 32 of each value in order.
                let cols = 8usize;
                let grid_rows = head_dim.div_ceil(cols);
                for r in 0..grid_rows {
                    let row_str: String = (0..cols)
                        .map(|c| {
                            let d = r * cols + c;
                            if d < head_dim {
                                let byte_idx = d / 4;
                                let pal_idx = if byte_idx < slice.len() {
                                    ((slice[byte_idx] >> ((d % 4) * 2)) & 0x3) as usize
                                } else {
                                    (d / sub_hd).min(N_PALETTE - 1)
                                };
                                format!("{pal_idx}")
                            } else {
                                " ".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!("    {row_str}");
                }
            };

            print_dim_grid("K", &last_q_k_pal_data, &last_q_k_grid);
            print_dim_grid("V", &last_q_v_pal_data, &last_q_v_grid);
        }
    }

    //#[cfg(feature = "verbose")]
    pub fn print_compression_distribution(&self, dist: &ahash::HashMap<GgmlDType, usize>) {
        println!("Compression Distribution (elements)");
        let mut dist = dist.iter().map(|(k, v)| (k, *v)).collect::<Vec<_>>();
        dist.sort_by_key(|(k, _)| k.type_size());
        for (format, cnt) in dist {
            println!("   {:?}: {}", format, cnt);
        }
    }

    /// Update the offset for a sequence after processing tokens.
    pub fn advance_sequence(&mut self, idx: usize, tokens_processed: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset += tokens_processed;
            Ok(())
        } else {
            candle::bail!("sequence {} not allocated", idx);
        }
    }

    /// Record a turn boundary for a sequence, capturing the current KV state.
    ///
    /// Commits the active window to `recorded_metas` and returns a `SealedSequence`
    /// snapshot of the just-committed turn.
    ///
    /// No-op when the sequence has zero tokens.
    pub fn record_turn(&mut self, idx: usize) -> Result<candle_nn::kv_cache::SealedSequence> {
        let backing = self
            .backings
            .first()
            .ok_or_else(|| candle::Error::Msg("record_turn: no backings".into()))?;
        backing
            .record_turn(idx)
            .map_err(|e| candle::Error::Msg(format!("record_turn: {e}")))
    }

    /// Validate that no two sessions share a raw GID across all layers.
    ///
    /// Returns `Ok(())` if clean, or an error listing every aliased GID with
    /// the GPU pointer and both owner locations.  Call before the decode kernel
    /// to confirm or rule out cross-session KV contamination.
    pub fn validate_gid_uniqueness(&self, seq_indices: &[usize]) -> Result<()> {
        let entries: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&i| (i, self.sequence_offset(i).unwrap_or(0)))
            .collect();

        let mut all_violations: Vec<String> = Vec::new();
        for (layer_idx, backing) in self.backings.iter().enumerate() {
            let v = backing.validate_gid_uniqueness(&entries)?;
            for s in v {
                all_violations.push(format!("[layer {layer_idx}] {s}"));
            }
        }

        if all_violations.is_empty() {
            Ok(())
        } else {
            candle::bail!(
                "GID aliasing detected across {} sessions ({} violations):\n{}",
                seq_indices.len(),
                all_violations.len(),
                all_violations.join("\n")
            )
        }
    }

    /// Ensure capacity for writing tokens at the given offsets.
    pub fn ensure_capacity(&self, sequence_indices: &[usize], add: usize) -> Result<()> {
        for backing in &self.backings {
            for &idx in sequence_indices.iter() {
                let offset = self.sequence_offset(idx).unwrap_or(0);
                backing.ensure_for_offset(idx, offset, add)?;
            }
        }
        Ok(())
    }

    /// Inject pre-populated prefix chunks into a sequence across all layers.
    ///
    /// The sequence must already be allocated (via `create_sequence`). Each
    /// layer backing gets the same chunk_ids/rope_positions/seq_len.
    pub fn inject_prefix(
        &self,
        seq_idx: usize,
        chunk_ids: &[HeadGids],
        seq_len: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.inject_prefix_chunks(seq_idx, chunk_ids, seq_len)?;
        }
        Ok(())
    }

    /// **Deprecated**: `no_quantize` is no longer used in the flat-chunks model.
    /// This method is a no-op kept for API compatibility.
    #[allow(unused_variables)]
    pub fn set_no_quantize(&self, seq_idx: usize, no_quantize: bool) -> Result<()> {
        Ok(())
    }

    /// Append borrowed chunk references to an existing sequence across all layers.
    ///
    /// Extends the sequence's block table starting from its current `block_count`.
    /// Does not change `block_usage`. Advances the backing's length tracking
    /// by `token_count`.
    ///
    /// The caller must call `advance_sequence(seq_idx, token_count)` afterwards
    /// to update the session-level offset.
    pub fn append_borrowed_chunks_cow(
        &self,
        seq_idx: usize,
        chunk_ids: &[HeadGids],
        token_count: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.append_borrowed_chunks_cow(seq_idx, chunk_ids, token_count)?;
        }
        Ok(())
    }

    /// Read the per-head GID vectors for a sequence's block table (from layer 0).
    ///
    /// Returns one `HeadGids` per block (length = `2 * n_kv_head`),
    /// preserving per-head arena assignments.
    /// All layers share the same block table, so reading from layer 0 suffices.
    pub fn slot_chunk_ids(&self, seq_idx: usize) -> Result<Vec<HeadGids>> {
        if let Some(backing) = self.backings.first() {
            backing.slot_chunk_ids(seq_idx)
        } else {
            Ok(Vec::new())
        }
    }

    /// Snapshot the current KV state of a sequence as a [`SealedSequence`].
    ///
    /// Captures the sequence's current chunk IDs and token counts.  The last
    /// chunk may be partial (fewer than `chunk_size` tokens); its token count
    /// is computed from the current sequence offset.  In the turn-recording
    /// design this also **commits** the turn boundary (moves the active window
    /// to `recorded_metas`).
    ///
    /// Returns an error if the sequence is not allocated.
    pub fn snapshot_sequence(&self, idx: usize) -> Result<candle_nn::kv_cache::SealedSequence> {
        let backing = self
            .backings
            .first()
            .ok_or_else(|| candle::Error::Msg("snapshot_sequence: no backings".into()))?;
        backing
            .record_turn(idx)
            .map_err(|e| candle::Error::Msg(format!("snapshot_sequence: {e}")))
    }

    /// Snapshot a sequence into per-layer `SealedSequence`s, one
    /// entry per backing.  Used at turn-end to capture per-layer KV
    /// bytes ready for `seal_to_cpu_per_layer`.
    pub fn snapshot_sequence_per_layer(
        &self,
        idx: usize,
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        let mut out = Vec::with_capacity(self.backings.len());
        for backing in &self.backings {
            out.push(
                backing
                    .record_turn(idx)
                    .map_err(|e| candle::Error::Msg(format!("snapshot_sequence_per_layer: {e}")))?,
            );
        }
        Ok(out)
    }

    /// Migrate a per-layer sealed snapshot from the GPU (hot) tier to the CPU (warm) tier.
    ///
    /// `sealed` must have one entry per layer (same length as `self.backings()`).
    /// Each entry is migrated by the matching layer's [`ChunkedKvBacking`].
    /// Returns a new per-layer vec with CPU-resident [`SealedSequence`]s.
    pub fn sealed_to_cpu(
        &self,
        sealed: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        sealed
            .iter()
            .zip(self.backings.iter())
            .map(|(seq, backing)| backing.migrate_sealed_to_cpu(seq))
            .collect()
    }

    /// Migrate a per-layer sealed snapshot from the CPU (warm) tier back to the GPU (hot) tier.
    ///
    /// Symmetric inverse of [`sealed_to_cpu`].
    pub fn sealed_to_gpu(
        &self,
        sealed: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        sealed
            .iter()
            .zip(self.backings.iter())
            .map(|(seq, backing)| backing.migrate_sealed_to_gpu(seq))
            .collect()
    }

    pub fn estimate_quantized_percentage(&self) -> Option<f64> {
        let sequences = self
            .sequences
            .iter()
            .enumerate()
            .filter_map(|(n, v)| v.as_ref().map(|_| n))
            .collect::<Vec<_>>();
        self.estimate_quantized_percentage_by_sequences(&sequences)
    }

    pub fn estimate_quantized_percentage_by_sequences(
        &self,
        sequence_indices: &[usize],
    ) -> Option<f64> {
        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        let mut total_quantized = 0usize;
        let mut total_tokens = 0usize;

        for (seq_idx, _, caches) in caches_and_offsets {
            // Sample the first layer only — all layers share one arena pool so layer 0 is
            // representative for an estimate.  Use seq_idx as batch_idx; that is the slot
            // index assigned when the sequence was registered in the backing.
            if let Some(kv_cache) = caches.caches.first() {
                if let Some(result) = kv_cache.quantized_token_stats(seq_idx) {
                    let Ok((quant, total)) = result else {
                        continue;
                    };
                    total_quantized += quant;
                    total_tokens += total;
                }
            }
        }

        if total_tokens > 0 {
            let percent = (total_quantized as f64 / total_tokens as f64) * 100.0;
            Some(percent)
        } else {
            None
        }
    }

    /// Compression ratio across sequences, measured over all layers.
    ///
    /// Returns the ratio of F16-equivalent bytes to actual bytes. A ratio > 1.0
    /// means compression is happening. For uniform modes this is deterministic;
    /// for adaptive modes it reflects the actual per-chunk format distribution.
    pub fn compression_ratio_by_sequences(&self, sequence_indices: &[usize]) -> Option<f64> {
        let mut total_actual = 0.0f64;
        let mut total_elements = 0usize;

        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        for (seq_idx, _, caches) in &caches_and_offsets {
            for kv_cache in &caches.caches {
                if let Some(result) = kv_cache.compression_bpe(*seq_idx) {
                    total_actual += result.0;
                    total_elements += result.1;
                }
            }
        }

        if total_elements > 1 {
            let bpe = total_actual as f64 / total_elements as f64;
            Some(16.0f64 / bpe)
        } else {
            None
        }
    }

    /// Compression distribution across sequences, measured over all layers.
    pub fn compression_dist_by_sequences(
        &self,
        sequence_indices: &[usize],
    ) -> ahash::HashMap<GgmlDType, usize> {
        let mut ret = ahash::HashMap::default();

        let caches_and_offsets = self.caches_for_sequences(sequence_indices);
        for (seq_idx, _, caches) in &caches_and_offsets {
            for kv_cache in &caches.caches {
                kv_cache.compression_dist(*seq_idx, &mut ret);
            }
        }

        ret
    }

    pub fn get_sequence_stats(&self, seq_idx: usize) -> SequenceStats {
        let mut ret = SequenceStats::default();
        for (actual_seq_idx, _, caches) in self.caches_for_sequences(&[seq_idx]) {
            for kv_cache in &caches.caches {
                if let Some(result) = kv_cache.quantized_token_stats(actual_seq_idx) {
                    let Ok((quant, total)) = result else {
                        continue;
                    };
                    ret.quantized_tokens += quant as u64;
                    ret.total_tokens += total as u64;
                    ret.active_tokens += total as u64;
                }
            }
        }
        ret
    }

    /// Read contiguous K/V float data for a token range across **all** layers.
    ///
    /// Returns one `(K, V)` pair per layer (length = `num_layers`).  Each tensor
    /// is float dtype, shape `(1, n_kv_heads, len, head_dim)`.  For quantized KV
    /// storage the data is dequantized on-the-fly before returning.
    ///
    /// Intended for Hot → Warm eviction: the caller reads the float K/V tensors,
    /// re-quantizes to Q8_0 on CPU, and writes the bytes into a [`WarmPool`].
    ///
    /// # Arguments
    ///
    /// * `seq_idx` — The batch index (sequence slot) whose KV data to read.
    /// * `offset`  — First token position to read.
    /// * `len`     — Number of tokens to read.
    pub fn read_all_layers_contiguous(
        &self,
        seq_idx: usize,
        offset: usize,
        len: usize,
    ) -> Result<Vec<(candle::Tensor, candle::Tensor)>> {
        self.backings
            .iter()
            .map(|backing| backing.read_contiguous(seq_idx, offset, len))
            .collect()
    }

    /// Write float K/V tensors into a sequence across **all** layers.
    ///
    /// The inverse of [`read_all_layers_contiguous`].  Used when restoring
    /// a Warm-tier turn directly from dequantized bytes without a forward
    /// pass.
    ///
    /// # Arguments
    ///
    /// * `seq_idx` — The batch index (sequence slot) to write into.
    /// * `offset`  — First token position to start writing at.
    /// * `kv_per_layer` — One `(K, V)` pair per layer, each shaped
    ///   `(1, n_kv_heads, len, head_dim)`.
    pub fn write_all_layers_contiguous(
        &self,
        seq_idx: usize,
        offset: usize,
        kv_per_layer: &[(candle::Tensor, candle::Tensor)],
    ) -> Result<()> {
        for (backing, (k, v)) in self.backings.iter().zip(kv_per_layer.iter()) {
            backing.write_contiguous(seq_idx, offset, k, v)?;
        }
        Ok(())
    }

    /// Read raw quantized bytes for one block across **all** layers.
    ///
    /// Called per-block during Hot→Warm eviction.  Returns one `(k_bytes, v_bytes)`
    /// pair per layer in the session.  The bytes are in the token-oriented layout
    /// produced by `reconcile` — identical to what the attention kernel reads.
    ///
    /// # Errors
    /// Fails if any layer's block is still in a float arena (not yet reconciled).
    pub fn read_all_layers_raw_sealed_chunk(
        &self,
        seq_idx: usize,
        block_idx: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.backings
            .iter()
            .map(|backing| backing.read_raw_sealed_chunk(seq_idx, block_idx))
            .collect()
    }

    /// Write raw quantized bytes for one block across **all** layers.
    ///
    /// Called per-block during Warm→Hot restore.  The bytes must be in the
    /// same token-oriented layout that `reconcile` produces (as returned by
    /// [`read_all_layers_raw_sealed_chunk`]).  No format conversion is performed.
    ///
    /// After writing all blocks, call `advance_sequence` with the total token count.
    pub fn write_all_layers_raw_sealed_chunk(
        &self,
        seq_idx: usize,
        block_idx: usize,
        kv_per_layer: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<()> {
        for (backing, (k_bytes, v_bytes)) in self.backings.iter().zip(kv_per_layer.iter()) {
            backing.write_raw_sealed_chunk(
                seq_idx,
                block_idx,
                k_bytes,
                v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct SequenceStats {
    /// How many tokens are current active in the conversation window
    pub active_tokens: u64,

    /// How many tokens are there in total that make up the conversation history
    pub total_tokens: u64,

    /// How many of the tokens have been quantized to save space
    pub quantized_tokens: u64,
}

/// Result of a successful [`BatchedInferenceSession::create_view_sequence`] call.
///
/// `view_idx` is the freshly allocated view slot; `borrowed_block_count` is how
/// many blocks were shared from the parent into the view.  Pass
/// `borrowed_block_count` back to [`BatchedInferenceSession::finalize_view`] so
/// it can identify where parent-owned blocks end and newly-generated blocks begin.
#[derive(Debug, Copy, Clone)]
pub struct ViewSequence {
    /// Sequence index of the newly created view.
    pub view_idx: usize,
    /// Number of KV blocks borrowed from the parent into this view.
    /// All borrowed blocks are read-only Arc clones; the view's writable
    /// tail is a fresh chunk pushed past index `borrowed_block_count`.
    pub borrowed_block_count: usize,
}

// ============================================================================
// ManagedBatchedModel Trait and Blanket Implementation
// ============================================================================

/// Trait for models that support session-managed batched inference.
///
/// This trait provides a high-level API for batched inference using
/// [`BatchedInferenceSession`] to manage KV cache state.
///
/// Layer index pairs for the three provenance signature depths.
///
/// Layout: `[syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]`.
///
/// Each band uses two layers — band-start (`l0 = centre − 4`, clamped) and
/// band-centre (`l4`) — whose multi-head Q sign bits are XOR-folded into a
/// single 128-bit signature per token (`MH_XOR_QQ_l0xl4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenanceLayerIndices {
    pub syn_l0: usize,
    pub syn_l4: usize,
    pub sem_l0: usize,
    pub sem_l4: usize,
    pub prag_l0: usize,
    pub prag_l4: usize,
}

impl ProvenanceLayerIndices {
    /// Flat array form `[syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]`.
    pub fn as_array(self) -> [usize; 6] {
        [
            self.syn_l0,
            self.syn_l4,
            self.sem_l0,
            self.sem_l4,
            self.prag_l0,
            self.prag_l4,
        ]
    }
}

/// Static model properties that are cheap to copy and independent of any inference session.
///
/// Captured once (before the model moves to the scheduler thread) via
/// [`ManagedBatchedModel::model_core_properties`] and stored wherever these values are needed.
#[derive(Clone, Copy, Debug)]
pub struct ModelCoreProperties {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub n_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Layer-index pairs for the three provenance signature depth bands.
    pub provenance_layer_indices: ProvenanceLayerIndices,
    /// Per-model multiplier for the K high adaptive threshold.
    pub k_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the K low adaptive threshold.
    pub k_low_error_threshold_factor: f32,
    /// Per-model multiplier for the V high (strict) adaptive threshold.
    pub v_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    pub v_low_error_threshold_factor: f32,
}

/// **You don't need to implement this trait manually.** Any type that implements
/// [`BatchedModel`] automatically gets a `ManagedBatchedModel` implementation
/// via the blanket impl.
pub trait ManagedBatchedModel {
    /// Number of transformer layers.
    fn num_layers(&self) -> usize;
    /// Number of KV heads per layer.
    fn n_kv_head(&self) -> usize;
    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;
    /// Device the model is on.
    fn device(&self) -> &Device;

    /// Snapshot of all static model properties.
    ///
    /// Cheap to copy; use this to capture model metadata before the model is moved to
    /// another thread.  The default derives `provenance_layer_indices` from `num_layers()`
    /// and sets threshold factors to `1.0`.  The blanket impl for `BatchedInference<M>`
    /// overrides this to read per-model threshold factors from the inner `BatchedModelCore`.
    fn model_core_properties(&self) -> ModelCoreProperties {
        let n = self.num_layers();
        let provenance_layer_indices = if n == 0 {
            ProvenanceLayerIndices {
                syn_l0: 0,
                syn_l4: 0,
                sem_l0: 0,
                sem_l4: 0,
                prag_l0: 0,
                prag_l4: 0,
            }
        } else {
            let syn = (n * 15 / 100).max(1);
            let sem = n / 2;
            let prag = (n * 85 / 100).min(n - 1);
            ProvenanceLayerIndices {
                syn_l0: syn.saturating_sub(4),
                syn_l4: syn,
                sem_l0: sem.saturating_sub(4),
                sem_l4: sem,
                prag_l0: prag.saturating_sub(4),
                prag_l4: prag,
            }
        };
        ModelCoreProperties {
            num_layers: n,
            n_kv_heads: self.n_kv_head(),
            head_dim: self.head_dim(),
            provenance_layer_indices,
            k_hi_error_threshold_factor: 1.0,
            k_low_error_threshold_factor: 1.0,
            v_hi_error_threshold_factor: 1.0,
            v_low_error_threshold_factor: 1.0,
        }
    }

    /// Run a batched forward pass for specific sequences.
    ///
    /// This method processes a subset of sequences in parallel, using the session's
    /// chunked KV backing for efficient paged attention.
    ///
    /// # Arguments
    /// * `session` - The batched inference session managing KV caches
    /// * `seq_indices` - Which sequence indices to process
    /// * `inputs` - Input tensors, one per sequence in seq_indices (same order)
    ///
    /// # Returns
    /// A vector of output logits tensors, one per input sequence.
    fn forward_batched(
        &self,
        session: &mut BatchedInferenceSession,
        seq_indices: &[usize],
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>>;

    /// Run a batched forward pass with per-sequence KV write offset shifts.
    ///
    /// This is used by the static chunk cache to right-pack partial blocks: the prefill
    /// kernel writes KV at physical positions `shift..shift+token_count` instead of
    /// `0..token_count`, so the injected prefix reads from the correct physical locations.
    ///
    /// For sequences where `write_offset_shifts[i] == 0`, behaviour is identical to
    /// `forward_batched`.
    ///
    /// The default implementation ignores the shifts (safe for non-paged paths).
    fn forward_batched_with_write_shifts(
        &self,
        session: &mut BatchedInferenceSession,
        seq_indices: &[usize],
        inputs: &[Tensor],
        write_offset_shifts: &[u32],
    ) -> Result<Vec<Tensor>> {
        // Default: ignore shifts (correct for non-CUDA / non-paged paths)
        let _ = write_offset_shifts;
        self.forward_batched(session, seq_indices, inputs)
    }

    /// Create a batched inference session configured for this model.
    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let mut config = config;
        let props = self.model_core_properties();
        config.k_hi_error_threshold_factor *= props.k_hi_error_threshold_factor;
        config.k_low_error_threshold_factor *= props.k_low_error_threshold_factor;
        config.v_hi_error_threshold_factor *= props.v_hi_error_threshold_factor;
        config.v_low_error_threshold_factor *= props.v_low_error_threshold_factor;
        BatchedInferenceSession::new(
            props.num_layers,
            props.n_kv_heads,
            props.head_dim,
            self.device(),
            config,
        )
    }

    /// Create a sibling session that shares the KV arena pool with `source`.
    ///
    /// Each `ChunkedKvBacking` in `source` is an `Arc<BackingInner>`.  Cloning
    /// those arcs into the new session means both sessions operate on the same
    /// underlying tensors, slot table, and arena table.  This is required for
    /// `append_borrowed_chunks_cow` to find chunk handles produced in the sibling
    /// session (e.g. fixed boundary-injection chunks generated in `proto_session`).
    ///
    /// Slot-index collisions between the two sessions are avoided automatically:
    /// `create_sequence` consults the shared backing for the first free slot, so
    /// the sibling gets a slot index that does not overlap with the source session.
    fn create_session_sharing_backings(
        &self,
        source: &BatchedInferenceSession,
        config: BatchedConfig,
    ) -> Result<BatchedInferenceSession> {
        let mut config = config;
        let props = self.model_core_properties();
        config.k_hi_error_threshold_factor *= props.k_hi_error_threshold_factor;
        config.k_low_error_threshold_factor *= props.k_low_error_threshold_factor;
        config.v_hi_error_threshold_factor *= props.v_hi_error_threshold_factor;
        config.v_low_error_threshold_factor *= props.v_low_error_threshold_factor;
        let backings = source.backings().iter().cloned().collect();
        Ok(BatchedInferenceSession::new_with_backings(
            backings,
            config,
            source.device(),
        ))
    }

    /// Prunes excess memory usage.
    fn prune(&self) -> Result<()>;

    /// Snapshot expert pipeline telemetry counters (if the model has an expert cache).
    fn expert_stats(&self) -> Option<PipelineStats> {
        None
    }

    /// Reset expert pipeline telemetry counters to zero.
    fn reset_expert_stats(&self) {}

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }
}

/// Maximum total tokens (batch_size × seq_len) for a single prefill forward pass.
/// Larger prefills are automatically sliced into smaller batches to keep GPU memory
/// bounded. GPU compute saturates around this point for typical models, so slicing
/// costs virtually nothing in throughput. The value is a multiple of 32 for optimal
/// CUDA kernel utilization.
const MAX_PREFILL_TOKENS: usize = 4096;

/// Blanket implementation of `ManagedBatchedModel` for `BatchedInference<M>`.
///
/// This allows models using the new `BatchedModelCore` + `BatchedInference` pattern
/// to work with `BatchedInferenceSession` without implementing `BatchedModel`.
impl<M: BatchedModelCore> ManagedBatchedModel for BatchedInference<M> {
    fn num_layers(&self) -> usize {
        self.model().num_layers()
    }

    fn n_kv_head(&self) -> usize {
        self.model().n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.model().head_dim()
    }

    fn device(&self) -> &Device {
        self.model().device()
    }

    fn model_core_properties(&self) -> ModelCoreProperties {
        let n = self.model().num_layers();
        let provenance_layer_indices = if n == 0 {
            ProvenanceLayerIndices {
                syn_l0: 0,
                syn_l4: 0,
                sem_l0: 0,
                sem_l4: 0,
                prag_l0: 0,
                prag_l4: 0,
            }
        } else {
            let syn = (n * 15 / 100).max(1);
            let sem = n / 2;
            let prag = (n * 85 / 100).min(n - 1);
            ProvenanceLayerIndices {
                syn_l0: syn.saturating_sub(4),
                syn_l4: syn,
                sem_l0: sem.saturating_sub(4),
                sem_l4: sem,
                prag_l0: prag.saturating_sub(4),
                prag_l4: prag,
            }
        };
        ModelCoreProperties {
            num_layers: n,
            n_kv_heads: self.model().n_kv_head(),
            head_dim: self.model().head_dim(),
            provenance_layer_indices,
            k_hi_error_threshold_factor: self.model().k_hi_error_threshold_factor(),
            k_low_error_threshold_factor: self.model().k_low_error_threshold_factor(),
            v_hi_error_threshold_factor: self.model().v_hi_error_threshold_factor(),
            v_low_error_threshold_factor: self.model().v_low_error_threshold_factor(),
        }
    }

    fn forward_batched(
        &self,
        session: &mut BatchedInferenceSession,
        seq_indices: &[usize],
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != seq_indices.len() {
            candle::bail!(
                "Input count {} doesn't match sequence count {}",
                inputs.len(),
                seq_indices.len()
            );
        }

        // Hold a generation guard for the entire forward pass.  This keeps
        // the pinned arena alive so all device-mapped pointers from
        // stager.submit() (used by quantization kernels) remain valid until
        // every layer's kernels have completed.  Dropped at function exit,
        // which syncs the stream and reclaims overflow arenas.
        let stager_generation = session.begin_stager_generation();

        // Calculate max input length for capacity
        let max_input_len = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .max()
            .unwrap_or(1);

        // NOTE: We intentionally do NOT call session.ensure_capacity() here.
        //
        // Both the prefill path (paged_prefill_batched → ensure_chunked_capacity_batch)
        // and the decode path (paged_decode_attention → ensure_chunked_capacity_batch)
        // ensure capacity per-layer, right before each layer processes.
        //
        // Pre-allocating across ALL layers simultaneously is harmful for quantized
        // KV cache modes (Q4_0, Q8_0): it creates float arenas for all 32 layers
        // at once, consuming ~18 GB of VRAM before any reconciliation can free them.
        // Per-layer allocation allows reconcile + consolidation to release float
        // arenas after each layer, keeping peak memory bounded.

        // Build the final decode/prefill headers before the mutable caches borrow.
        // Fetch per-sequence offsets now (immutable borrow) so prefill meta can be
        // built here — eliminating the two-phase override later.
        let seq_offsets: Vec<usize> = seq_indices
            .iter()
            .map(|&i| session.sequence_offset(i).unwrap_or(0))
            .collect();
        // Per-sequence new-token (query) lengths — ragged across the prefill batch.
        let input_lens: Vec<usize> = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .collect();
        #[cfg(feature = "cuda")]
        let (_pm_guard, decode_headers) = if max_input_len == 1 {
            let (pm_guard, buf, stride) =
                session.build_decode_metadata(seq_indices, &stager_generation)?;
            (pm_guard, DecodeHeaders::Decode { buf, stride })
        } else {
            let mut meta =
                BatchedPrefillMeta::new_ragged(&seq_offsets, &input_lens, self.device())?;
            // A reprojection-glue forward stages its per-slot column positions on
            // the session; attach them so the layer routes HD128 to the paged-glue
            // kernel. Consumed once — ordinary prefills leave `glue` None.
            if let Some(glue_cols) = session.take_pending_glue() {
                meta.glue = build_glue_meta(glue_cols, &seq_offsets, &input_lens, self.device())?;
            }
            (None, DecodeHeaders::Prefill(meta))
        };
        #[cfg(not(feature = "cuda"))]
        let decode_headers = if max_input_len == 1 {
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            }
        } else {
            let meta = BatchedPrefillMeta::new_ragged(&seq_offsets, &input_lens, self.device())?;
            DecodeHeaders::Prefill(meta)
        };

        // Get caches for the requested sequences
        let mut caches_data = session.caches_for_sequences_mut(seq_indices);

        // Build contexts from the collected data
        let mut contexts: Vec<SequenceContext<'_>> = Vec::with_capacity(inputs.len());
        for (i, (_seq_idx, offset, caches)) in caches_data.iter_mut().enumerate() {
            contexts.push(SequenceContext {
                offset: *offset,
                kv_caches: caches,
                input_ids: &inputs[i],
                input_len: input_lens[i],
                write_offset_shift: 0,
            });
        }

        // Slice large prefills to bound a single forward's token count. Ragged:
        // group sequences so each slice's TOTAL tokens (Σ input_lens) stays within
        // MAX_PREFILL_TOKENS, rather than count × max_len. Decode (1 token/seq) is
        // cheap and never sliced.
        let total_tokens: usize = input_lens.iter().sum();
        let outputs = if total_tokens > MAX_PREFILL_TOKENS && max_input_len > 1 {
            let mut all_logits = Vec::with_capacity(contexts.len());
            let mut slice_start = 0usize;
            while slice_start < contexts.len() {
                // Grow the slice until the next sequence would exceed the token
                // budget; always take at least one sequence.
                let mut slice_tokens = 0usize;
                let mut slice_end = slice_start;
                while slice_end < contexts.len() {
                    let l = contexts[slice_end].input_len;
                    if slice_end > slice_start && slice_tokens + l > MAX_PREFILL_TOKENS {
                        break;
                    }
                    slice_tokens += l;
                    slice_end += 1;
                }
                let slice = &mut contexts[slice_start..slice_end];
                let offsets: Vec<usize> = slice.iter().map(|c| c.offset).collect();
                let lens: Vec<usize> = slice.iter().map(|c| c.input_len).collect();
                let meta = BatchedPrefillMeta::new_ragged(&offsets, &lens, self.device())?;
                let slice_logits =
                    self.forward_batch(slice, &stager_generation, DecodeHeaders::Prefill(meta))?;
                all_logits.extend(slice_logits.into_vec()?);
                slice_start = slice_end;
            }
            all_logits
        } else {
            // Small enough to process in one shot
            self.forward_batch(&mut contexts, &stager_generation, decode_headers)?
                .into_vec()?
        };

        if max_input_len > 1 {
            let _ = session.compact_check()?;
        }

        Ok(outputs)
    }

    fn forward_batched_with_write_shifts(
        &self,
        session: &mut BatchedInferenceSession,
        seq_indices: &[usize],
        inputs: &[Tensor],
        write_offset_shifts: &[u32],
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != seq_indices.len() || write_offset_shifts.len() != seq_indices.len() {
            candle::bail!(
                "Input count {}, shift count {}, sequence count {} must all match",
                inputs.len(),
                write_offset_shifts.len(),
                seq_indices.len()
            );
        }

        let stager_generation = session.begin_stager_generation();

        let max_input_len = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .max()
            .unwrap_or(1);

        // Build the final decode/prefill headers before the mutable caches borrow.
        let seq_offsets: Vec<usize> = seq_indices
            .iter()
            .map(|&i| session.sequence_offset(i).unwrap_or(0))
            .collect();
        #[cfg(feature = "cuda")]
        let (_pm_guard, decode_headers) = if max_input_len == 1 {
            let (pm_guard, buf, stride) =
                session.build_decode_metadata(seq_indices, &stager_generation)?;
            (pm_guard, DecodeHeaders::Decode { buf, stride })
        } else {
            let meta = BatchedPrefillMeta::new(&seq_offsets, max_input_len, self.device())?;
            (None, DecodeHeaders::Prefill(meta))
        };
        #[cfg(not(feature = "cuda"))]
        let decode_headers = if max_input_len == 1 {
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            }
        } else {
            let meta = BatchedPrefillMeta::new(&seq_offsets, max_input_len, self.device())?;
            DecodeHeaders::Prefill(meta)
        };

        let mut caches_data = session.caches_for_sequences_mut(seq_indices);
        let input_lens: Vec<usize> = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .collect();

        let mut contexts: Vec<SequenceContext<'_>> = Vec::with_capacity(inputs.len());
        for (i, (_seq_idx, offset, caches)) in caches_data.iter_mut().enumerate() {
            contexts.push(SequenceContext {
                offset: *offset,
                kv_caches: caches,
                input_ids: &inputs[i],
                input_len: input_lens[i],
                write_offset_shift: write_offset_shifts[i] as usize,
            });
        }

        let total_tokens = contexts.len() * max_input_len;
        if total_tokens > MAX_PREFILL_TOKENS && max_input_len > 1 {
            let raw_seqs = MAX_PREFILL_TOKENS / max_input_len;
            let seqs_per_slice = if raw_seqs >= 32 {
                (raw_seqs / 32) * 32
            } else {
                raw_seqs.max(1)
            };

            let mut all_logits = Vec::with_capacity(contexts.len());
            for slice_start in (0..contexts.len()).step_by(seqs_per_slice) {
                let slice_end = (slice_start + seqs_per_slice).min(contexts.len());
                let slice = &mut contexts[slice_start..slice_end];
                let offsets: Vec<usize> = slice.iter().map(|c| c.offset).collect();
                let meta = BatchedPrefillMeta::new(&offsets, max_input_len, self.device())?;
                let slice_logits =
                    self.forward_batch(slice, &stager_generation, DecodeHeaders::Prefill(meta))?;
                all_logits.extend(slice_logits.into_vec()?);
            }
            Ok(all_logits)
        } else {
            let result = self
                .forward_batch(&mut contexts, &stager_generation, decode_headers)?
                .into_vec()?;
            Ok(result)
        }
    }

    fn prune(&self) -> Result<()> {
        self.model().prune()
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        self.model().expert_stats()
    }

    fn reset_expert_stats(&self) {
        self.model().reset_expert_stats()
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.model().snapshot_profiles()
    }
}

// ============================================================================
// Backward Compatibility Aliases
