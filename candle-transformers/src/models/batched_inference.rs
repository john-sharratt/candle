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
//! // Run batched inference - BatchedInference<M> implements ManagedBatchedModel.
//! // `forward_wave` is the single forward entry: a full-sweep prefill group here
//! // (empty decode/glue groups, layers `[0, num_layers)`, no resumed residual).
//! let n = model.num_layers();
//! let outputs = model
//!     .forward_wave(&mut session, &[], &[], &seq_indices, &input_tensors, &[], &[], 0, n, None)?
//!     .logits
//!     .unwrap_or_default();
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
    ChunkedKvBacking, CompressionPolicy, GpuArenaClassStats, HeadGids, KvCache, KvFormat,
    ModelGeometry, QuantFormat, WavePlan, WAVE_FFN_BYTES,
};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

#[cfg(feature = "cuda")]
use super::batched_layer::GlueMeta;
use super::batched_model::{BatchedInference, BatchedModelCore, WaveGuard, WavePhase};
use super::wave_driver::{drive_wave, WaveGroups, WaveSweep};
#[cfg(feature = "cuda")]
use crate::models::profile::pipeline_record_duration;
use crate::models::speculative_choice::{AcceptWalk, TokenChooser};
use crate::models::verify_wave::{issue_verify_wave, VerifyPlan, WaveCoBatch};

/// One R16 chunk's unpacked contents: `(block_idx, k_flat, v_flat, q_flat)`.
///
/// Each flat vec holds `n_kv_head × N_PALETTE × CHUNK_SIZE × sub_head_dim`
/// values in `[head][palette][token][sub_dim]` order. `block_idx` is absolute,
/// so a caller can place the chunk even though non-R16 chunks leave gaps.
pub type R16ChunkDump = (usize, Vec<f32>, Vec<f32>, Vec<f32>);

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

/// Assemble a [`GlueMeta`] from the wave's per-slot glue descriptors for a
/// gap-fill forward. Each `pending[i]`'s three vectors have length
/// `input_lens[i]` (the slot's glue-token count); they are concatenated in the
/// forward's flat `q` order. There is no `col_actual_pos`: the kernel derives
/// every column's position from its chunk `rope_base` (`slice_rope`), the same
/// convention decode reads. Returns `None` (falls back to plain prefill) only
/// when no slot actually carries glue.
#[cfg(feature = "cuda")]
pub(crate) fn build_glue_meta(
    pending: Vec<PendingGlue>,
    input_lens: &[usize],
    device: &Device,
) -> Result<Option<GlueMeta>> {
    if pending.len() != input_lens.len() {
        candle::bail!(
            "build_glue_meta: {} glue descriptors vs {} input_lens",
            pending.len(),
            input_lens.len()
        );
    }
    let total: usize = input_lens.iter().sum();
    let mut write_slice: Vec<u32> = Vec::with_capacity(total);
    let mut write_in_blk: Vec<u32> = Vec::with_capacity(total);
    let mut fwd_ahead: Vec<u32> = Vec::with_capacity(total);
    for (i, p) in pending.iter().enumerate() {
        if p.write_slice.len() != input_lens[i]
            || p.write_in_blk.len() != input_lens[i]
            || p.fwd_ahead.len() != input_lens[i]
        {
            candle::bail!(
                "build_glue_meta: slot {i} glue len ({}/{}/{}) != input_len {}",
                p.write_slice.len(),
                p.write_in_blk.len(),
                p.fwd_ahead.len(),
                input_lens[i]
            );
        }
        write_slice.extend_from_slice(&p.write_slice);
        write_in_blk.extend_from_slice(&p.write_in_blk);
        fwd_ahead.extend_from_slice(&p.fwd_ahead);
    }
    if write_slice.is_empty() {
        return Ok(None);
    }
    let n = write_slice.len();
    // Confirms the gap-fill forward took the paged-glue route (HD128) — one line
    // per reproject, under the scheduler's reproject log target.
    tracing::trace!(
        target: "candle_conversation::scheduler::reproject",
        slots = pending.len(),
        total_glue = total,
        "paged-glue route active"
    );
    Ok(Some(GlueMeta {
        glue_write_slice: Tensor::from_vec(write_slice, n, device)?,
        glue_write_in_blk: Tensor::from_vec(write_in_blk, n, device)?,
        fwd_ahead: Tensor::from_vec(fwd_ahead, n, device)?,
    }))
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
    /// Pending reprojection-glue descriptors, set by the wave immediately before
    /// its gap-fill `forward_batched`. One entry per sequence (in the forward's
    /// `seq_indices` order). Taken + cleared inside `forward_batched`, which
    /// routes HD128 glue to the paged-glue kernel.
    pending_glue: Option<Vec<PendingGlue>>,
}

/// Per-slot reprojection-glue descriptor staged on the session for one gap-fill
/// `forward_batched`. The glue tokens are reserved IN PLACE as gap chunks during
/// assembly, so there is no `col_actual_pos`: every column's sequence position
/// comes from its chunk's `rope_base` (`slice_rope`), the same convention decode
/// reads. This carries only what the kernel can't derive from the slot itself —
/// where each glue token scatters, and how far it bridges forward.
#[derive(Clone, Debug)]
pub struct PendingGlue {
    /// Per glue token: the gap chunk's block index its K/V scatters into.
    pub write_slice: Vec<u32>,
    /// Per glue token: the in-block offset within its gap chunk.
    pub write_in_blk: Vec<u32>,
    /// Per glue token: forward bridge window in tokens (`0` == backward-only).
    pub fwd_ahead: Vec<u32>,
}

/// GPU-packed wide-Q provenance sign bits for a scope, produced by
/// [`BatchedInferenceSession::gather_provenance_sign_packed`] and consumed by the
/// scheduler's `gather_wide_sigs` (assemble raw `WideQSig` → fold).
pub struct ProvSignPacked {
    /// Warp-major packed sign bits — `packed[warp * CHUNK_SIZE + token]`, bit `d`
    /// set iff Q dim `d` of that sub-band is `>= 0` (`d` in `0..sub_head_dim`).
    /// Warp index = `((layer*n_blocks + block_pos)*n_kv_head + head)*n_palette + palette`,
    /// where `block_pos` indexes [`Self::block_indices`].
    pub packed: Vec<u64>,
    /// Absolute chunk indices captured (identical across all layers).
    pub block_indices: Vec<usize>,
    pub n_layers: usize,
    pub n_kv_head: usize,
    pub n_palette: usize,
    /// `head_dim / n_palette` — bits packed per palette sub-band (`<= 64`, the
    /// width of a physical R16 band at `head_dim` 256).
    pub sub_head_dim: usize,
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

    /// Stage per-slot reprojection-glue descriptors for the next
    /// `forward_batched`. `pending[i]` aligns with the `seq_indices` of the
    /// imminent gap-fill forward. Consumed (and cleared) by that single forward.
    pub fn set_pending_glue(&mut self, pending: Vec<PendingGlue>) {
        self.pending_glue = Some(pending);
    }

    /// Take + clear the staged glue descriptors (one forward's worth).
    pub fn take_pending_glue(&mut self) -> Option<Vec<PendingGlue>> {
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
            .filter(|s| s.as_ref().is_some_and(|s| s.active))
            .count()
    }

    /// Get the list of active sequence indices.
    pub fn active_sequences(&self) -> Vec<usize> {
        self.sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| {
                if s.as_ref().is_some_and(|s| s.active) {
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

    /// Refresh the persistent decode GPU slot-state for `seq_idx` across every
    /// layer after a multi-token prefill wrote tokens without the decode
    /// kernel's self-increment (a stencil static-run injection, a think-steer
    /// continuation prefill).
    ///
    /// The decode hot path trusts the cached GPU slot buffer, whose tail
    /// length self-increments only on decode steps — without this refresh the
    /// injected tokens sit beyond the buffer's stale tail length, invisible to
    /// subsequent decode attention (and progressively clobbered by the next
    /// decode writes). Only the WRITER chunk's slice is re-serialised (O(1)
    /// per layer); a prefill that crossed a chunk boundary already dropped the
    /// buffer at the mutation site, and a sequence that has not decoded yet
    /// has none — both rebuild fully on the next decode sync. No-op on
    /// contiguous backings.
    pub fn refresh_decode_slot_state(&self, seq_idx: usize) -> Result<()> {
        for backing in &self.backings {
            backing.refresh_decode_writer_slice(&[(seq_idx, 0)])?;
        }
        Ok(())
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
                cache.set_chunked_backing(backing, seq_idx, compression)?;
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
        self.build_decode_metadata_at(0..self.num_layers, seq_indices, generation, &[], &[], &[])
    }

    /// [`Self::build_decode_metadata`] with explicit per-sequence offset
    /// overrides: each `(seq, offset)` pair serializes that slot AS IF the
    /// sequence stood at `offset`, without touching session state. The wave
    /// prefill uses this to pre-build one header snapshot per absorbed token
    /// (token `t` attends `[0, base+t)` + itself — exactly the per-token
    /// decode regime), while other sequences serialize their live offsets.
    ///
    /// `non_writer` names sequences whose rows do NOT scatter through the
    /// decode write slot (glue rows write into explicitly reserved gap chunks
    /// instead). They are EXCLUDED from the `+1` write-chunk ensure: applying
    /// it would allocate a spurious empty chunk past a full/gap tail, which
    /// permanently inflates the slot's block count — the turn-seal range
    /// `[turn_start_parent_blocks, block_count)` then misses the turn's real
    /// blocks and the turn silently never persists.
    ///
    /// `snapshot_seqs` names the sequences whose slot state must be captured as
    /// an IMMUTABLE snapshot copy into `generation` (their `slices_ptr` survives
    /// a later chunk-boundary reallocation) — i.e. any sequence that mutates the
    /// arena during this forward: a prefill absorbing across chunk boundaries, a
    /// glue gap-scatter. Every OTHER sequence (a plain decode row, whose write
    /// chunk is pre-ensured so it never reallocs) keeps the zero-copy LIVE
    /// pointer + on-device write-len commit — the cheap decode path. Passing
    /// `&[]` makes every row live.
    ///
    /// `layers` names the contiguous group of KV layers to describe. The whole
    /// stack (`0..num_layers`) is the ordinary answer, and the shared position
    /// map is what makes it one build rather than N — every layer of the group
    /// must therefore agree on block structure, which
    /// [`ChunkedKvBacking::ensure_for_batch_entries_all`] establishes over the
    /// group before the map is read from its first backing.
    ///
    /// A NARROWER group exists for exactly one reason: a layer that legitimately
    /// stands at a different length from the rest. The MTP draft head is one —
    /// its layer runs one position ahead of the trunk's for each speculative
    /// token it proposes, and a whole-stack build would see that as divergence
    /// and "heal" it by truncating the head's proposal away. Asking for
    /// `head..head + 1` describes that layer against its own length, and the
    /// group is trivially self-consistent.
    ///
    /// Headers are packed for the GROUP, so the kernel index of layer `l` is
    /// `l - layers.start`, not `l`.
    pub fn build_decode_metadata_at(
        &self,
        layers: Range<usize>,
        seq_indices: &[usize],
        generation: &Generation,
        offset_overrides: &[(usize, usize)],
        non_writer: &[usize],
        snapshot_seqs: &[usize],
    ) -> Result<(Option<GpuBuf>, Option<GpuBuf>, u64)> {
        let n_active = seq_indices.len();
        if n_active == 0 {
            return Ok((None, None, 0));
        }
        if layers.start >= layers.end || layers.end > self.num_layers {
            candle::bail!(
                "decode metadata: layer group {:?} is empty or past the session's \
                 {} KV layers",
                layers,
                self.num_layers
            )
        }
        let group = &self.backings[layers.clone()];
        // Per-row snapshot decision (layer-invariant), aligned with `seq_indices`.
        let snapshot_mask: Vec<bool> = seq_indices
            .iter()
            .map(|s| snapshot_seqs.contains(s))
            .collect();

        // 24-byte SlotHeader: n_slices, write_slice, slices_ptr, position_map_ptr.
        let header_stride = n_active * 24;
        let mut all_headers: Vec<u8> = Vec::with_capacity(group.len() * header_stride);

        // Pre-compute per-sequence offsets once (same for all layers).
        let seq_offsets: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&seq_idx| {
                let offset = offset_overrides
                    .iter()
                    .find(|(s, _)| *s == seq_idx)
                    .map(|&(_, o)| o)
                    .unwrap_or_else(|| {
                        self.sequences
                            .get(seq_idx)
                            .and_then(|s| s.as_ref())
                            .map_or(0, |s| s.offset)
                    });
                (seq_idx, offset)
            })
            .collect();

        // Build per-sequence position_map covering [0, state.offset + 1).
        // Each entry is u32: (slice_idx << 16) | in_blk.  The map is
        // invariant across the GROUP (slice metadata is uniform within it),
        // built once, and every layer's SlotHeader points into the
        // per-sequence region.
        // Entry at index state.offset is the write slot for the new token.
        let mut pm_flat: Vec<u32> = Vec::new();
        let mut pm_seq_byte_offsets: Vec<usize> = Vec::with_capacity(n_active);
        // `(slice count, write slice)` the map was built against, per sequence.
        // Both are read from the group's FIRST layer while the map is one buffer
        // every layer's slot header points at, so every layer's own answer is
        // checked against them below rather than assumed equal.
        let mut pm_slot_shape: Vec<(u32, u32)> = Vec::with_capacity(n_active);
        // Ensure backings are sized for the upcoming decode write so the slot's
        // chunks reflect the post-write layout when we read them, and reconcile
        // any block-structure skew between layers — the map built below is one
        // buffer describing all of them.
        //
        // Every layer of the GROUP at once, and once per decode step rather than
        // once per layer: only the layers that need an allocation take the write
        // guard, which is what the per-layer form cost 48 times per decoded
        // token in a steady state where the answer is "nothing to allocate".
        //
        // Only for sequences that actually decode-write — a speculative-verify
        // slot replays already-written positions and must not have a write
        // chunk pre-allocated (see `non_writer`).
        let writer_offsets: Vec<(usize, usize)> = seq_offsets
            .iter()
            .copied()
            .filter(|(s, _)| !non_writer.contains(s))
            .collect();
        ChunkedKvBacking::ensure_for_batch_entries_all(group, &writer_offsets, 1)?;
        for &(seq_idx, seq_offset) in &seq_offsets {
            let entry_start = pm_flat.len();
            pm_seq_byte_offsets.push(entry_start * 4);
            let chunks = group[0].live_chunks_as_sealed(seq_idx).unwrap_or_default();
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
            let wstart = group[0].writer_start_idx_for_seq(seq_idx).unwrap_or(0);
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
            pm_slot_shape.push((n_ch as u32, wi as u32));
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

        for (slot, backing) in group.iter().enumerate() {
            // Capacity for the upcoming write is ensured for EVERY layer of the
            // group above, before this loop — see `ensure_for_batch_entries_all`.
            let layer_idx = layers.start + slot;

            let arena_info = backing.resolve_arena_info()?;

            // Incrementally sync each sequence's GPU slot-state buffer.
            // Common case: the cached GPU buffer is already valid and we only
            // reuse its pointer. Chunk-boundary case: the layer rebuilds once
            // from authoritative CPU chunk state.
            // The wave prefill builds every per-token snapshot up front, then
            // runs the layer loop, so its `slices_ptr` must be an immutable copy
            // that survives later chunk-boundary reallocations of the live
            // buffer. Live decode (one metadata build per step, used immediately)
            // keeps the zero-copy live pointer + on-device write-len commit.
            // Per-row live-or-snapshot: only the rows in `snapshot_seqs` pay the
            // immutable copy; decode rows keep the live pointer (mask all-false ⇒
            // the whole wave is the cheap live path).
            let (seq_ptrs, sync_stats) = backing.sync_decode_gpu_chunks_snapshot(
                &seq_offsets,
                &arena_info,
                generation,
                &snapshot_mask,
            )?;
            slot_reuse_time += sync_stats.reuse_time;
            slot_rebuild_time += sync_stats.rebuild_time;
            saw_slot_reuse |= sync_stats.reuses > 0;
            saw_slot_rebuild |= sync_stats.rebuilds > 0;

            // Append this layer's headers (24 bytes × n_active).
            for (i, &(ptr, n_slices, write_slice)) in seq_ptrs.iter().enumerate() {
                // The kernel SCATTERS the new token through this layer's own
                // `write_slice` and READS it back through the shared map's write
                // slot. A layer whose block table disagrees with the one the map
                // was built from writes the token into one chunk and attends to
                // another — silently, with no fault and no wrong-looking number
                // anywhere. `ensure_for_batch_entries_all` reconciles the layers
                // before this point; this is where that is worth confirming,
                // because both values are already in hand.
                let (exp_slices, exp_write) = pm_slot_shape[i];
                if (n_slices, write_slice) != (exp_slices, exp_write) {
                    candle::bail!(
                        "decode metadata: layer {layer_idx} describes sequence {} as \
                         {n_slices} slices writing slice {write_slice}, but the position map \
                         every layer of {layers:?} shares was built from layer {} as \
                         {exp_slices} slices writing slice {exp_write}",
                        seq_offsets[i].0,
                        layers.start
                    )
                }
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

        // Upload the group's headers in a single pinned → GPU copy.
        let total = group.len() * header_stride;
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

    /// Reserve an in-place glue gap of `n_tokens` slots at the slot tail across
    /// every layer, advance the session offset, and return the gap's block index
    /// (identical across layers). The glue forward later fills the gap by
    /// explicit `(slice, in_blk)` write target; until then its K/V is
    /// uninitialised but never read (the kernel scatters before it streams).
    ///
    /// This is the interleaved-glue primitive: because the gap is a real chunk
    /// with `usage = n_tokens` sitting at its logical position, the
    /// cumulative-usage `rope_base` of every later chunk equals its true
    /// sequence position — so decode and glue share one positional convention
    /// (`slice_rope`) with no `col_actual_pos` side channel.
    /// Reserve a full-by-construction glue gap across every layer's backing.
    /// Returns `(gap_block_index, in_blk_base)` — the block index (identical
    /// across layers) and the first valid slot of the gap's tail window, into
    /// which the glue forward scatters the island's K/V.
    pub fn reserve_glue_gap(&mut self, seq_idx: usize, n_tokens: u32) -> Result<(usize, u32)> {
        // `reserve_glue_gap_chunk` MUTATES each layer (pushes a gap chunk + a writer
        // chunk). This loop must therefore be ATOMIC: if it bails mid-way — because a
        // later layer's gap index diverges, or a per-layer reservation errors — the
        // layers already pushed must be ROLLED BACK, or the slot is left one chunk
        // longer on those layers and EVERY subsequent reservation diverges harder,
        // permanently wedging the sequence. Each layer's returned `idx` is its
        // PRE-push block count (gap_idx = block_count-1 taken right after the gap
        // push), so truncating a pushed layer back to `idx` blocks restores it
        // exactly. (The primary fix is deferring the pinned working set from the
        // warm→cold gather so a live slot's layers stay uniform; this keeps a
        // residual divergence a clean, retryable turn error instead of corruption.)
        // Reconcile per-layer block counts BEFORE reserving. During a windowed creep
        // prefill, layer 0 pushes an empty (0-token) writer chunk for the next window
        // ahead of the layers still pending resume, so the layers' block counts differ
        // by one even though their materialised token counts match (cf. dcd075e0, which
        // fixed the same incremental-fill skew for `sequence_backing_tokens`). A section
        // unit sealed while a slot was mid-creep persists that skew, so even a FRESH
        // conversation re-injecting it hits uneven layers at its very first glue
        // reservation ("layer gap index diverged 33 != 32"). The reservation needs ONE
        // gap index across every layer, so first pad each lagging layer up to the max
        // with the SAME empty writer chunk: it carries 0 tokens, so it shifts no
        // position (contributes 0 to every later chunk's cumulative-usage `rope_base`)
        // and only equalises the block count. Truncating the ahead layer instead would
        // drop a writer chunk a co-batched decode may target, so pad-to-max is safer.
        let max_blocks = (0..self.backings.len())
            .filter_map(|li| self.backings[li].sequence_block_count(seq_idx))
            .max()
            .unwrap_or(0);
        for li in 0..self.backings.len() {
            // `None` (unallocated) reads as `max_blocks`, so the loop is skipped and
            // never spins; a successful push raises the count by one toward `max`.
            while self.backings[li]
                .sequence_block_count(seq_idx)
                .unwrap_or(max_blocks)
                < max_blocks
            {
                self.backings[li].push_empty_writer_chunk(seq_idx)?;
            }
        }

        let mut gap: Option<(usize, u32)> = None;
        let mut pre_counts: Vec<usize> = Vec::with_capacity(self.backings.len());
        let rollback = |backings: &[ChunkedKvBacking], pre: &[usize]| {
            for (li, &c) in pre.iter().enumerate() {
                let _ = backings[li].truncate_sequence_to_blocks(seq_idx, c);
            }
        };
        for li in 0..self.backings.len() {
            let (idx, in_blk_base) =
                match self.backings[li].reserve_glue_gap_chunk(seq_idx, n_tokens) {
                    Ok(v) => v,
                    Err(e) => {
                        rollback(&self.backings, &pre_counts);
                        return Err(e);
                    }
                };
            pre_counts.push(idx);
            match gap {
                None => gap = Some((idx, in_blk_base)),
                Some((g, _)) if g != idx => {
                    rollback(&self.backings, &pre_counts);
                    candle::bail!(
                        "reserve_glue_gap: layer gap index diverged ({g} != {idx}) for slot {seq_idx}"
                    );
                }
                _ => {}
            }
        }
        if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
            state.offset += n_tokens as usize;
        } else {
            rollback(&self.backings, &pre_counts);
            candle::bail!("reserve_glue_gap: sequence {seq_idx} not allocated");
        }
        gap.ok_or_else(|| candle::Error::Msg("reserve_glue_gap: no layers".into()))
    }

    /// Truncate every backing's view of `seq_idx` to `block_count`
    /// chunks; reset the session's logical offset to the resulting
    /// token count (read from layer 0).
    ///
    /// Lets the scheduler reset a persistent conversation sequence to
    /// its system-prompt baseline before injecting the next turn's
    /// projection.
    /// Cut a sequence's chunk list back to `block_count` blocks.
    ///
    /// **A non-zero target is not automatically a rewind of live state.** Two
    /// callers legitimately pass one, and neither leaves a model's per-sequence
    /// state describing tokens the K/V no longer holds:
    ///
    /// - the clean-turn re-prefill, which truncates to the turn boundary *and*
    ///   discards the view's recurrent state in the same step, so both sides
    ///   move together;
    /// - `reserve_glue_gap`'s error-path rollback, which releases chunks a
    ///   failed call had just reserved and that nothing has decoded into.
    ///
    /// The operation that *is* a state-corrupting rewind is the token-granular
    /// [`Self::truncate_sequence_to_tokens`], reached only from speculative
    /// decode, and it is refused up front for a model that
    /// [`ManagedBatchedModel::carries_recurrent_state`].
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

    /// Truncate `seq_idx`'s KV to an exact `target_tokens` count across every backing, and set
    /// the sequence offset to match. Token-granular counterpart to
    /// [`Self::truncate_sequence_to_blocks`] — the primitive the generic speculative-decode
    /// driver uses to roll back the rejected tail of a verified block.
    pub fn truncate_sequence_to_tokens(
        &mut self,
        seq_idx: usize,
        target_tokens: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.truncate_sequence_to_tokens(seq_idx, target_tokens)?;
        }
        if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
            state.offset = target_tokens;
            // **The per-layer caches carry their own length, and it is not the
            // backing's.** Leaving them is how a rewound sequence walks into
            // the next wave with two answers for how long it is: the wave's
            // pre-claim ensures a write chunk at the SESSION offset, while the
            // decode path primes its write slot from `KvCache::current_seq_len`
            // — so a stale cache asks for a chunk nobody claimed, and the arena
            // refuses to create one from inside the forward that owns the
            // partition. That is not a hypothetical: it is what a speculative
            // sequence did on the step after a partial accept, once it had
            // dropped back to a plain decode row.
            for cache in state.caches.caches.iter_mut() {
                cache.truncate_to_offset(target_tokens)?;
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
    /// the shape is unquantizable (the palette4 quantizer supports `head_dim`
    /// 128 and 256; single-latent backings take the per-band latent
    /// compressor at any band-divisible head_dim).
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
        // The fused palette4 quantizer supports head_dim 128 and 256;
        // single-latent backings route through the per-band latent compressor
        // instead (any head_dim divisible by N_PALETTE). Other shapes still
        // collapse the layout below — just without quantizing.
        let quantizable = self.backings.first().is_some_and(|b| {
            candle::quantized::cuda::kvhead_supported_head_dim(b.head_dim())
                || (b.single_latent() && b.head_dim() % 4 == 0)
        });
        let policy = if quantizable { policy } else { None };

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
            // wave's admit phase (`wave_admit`) already collapses phantom
            // chunks from repeated re-prefill, so the record_turn/truncate/re-inject
            // round-trip is unnecessary (and re-injecting Arc-shared chunks under a
            // lone sequence corrupted the decode read). Just open a fresh writer.
            if start_new_chunk {
                self.push_empty_writer_chunk(seq_idx)?;
            }
        }
        Ok(())
    }

    /// Sweep fully-empty KV arenas across all backings, returning their regions
    /// to the free list. Returns arenas freed.
    ///
    /// Runs per-wave, and it is cheap: an arena release is a free-list push and
    /// nothing more. It used to be neither. Releasing an arena unmapped its
    /// slab, so this took the process-global arena-topology write lock — the
    /// migrate's per-head table captured raw base pointers of every arena and
    /// dereferenced them from a kernel with no lock held — and then paid a full
    /// `device.synchronize()` to retire trailing kernels before the unmap.
    /// A whole-device sync, every wave, on the sweep path.
    ///
    /// Under the reservation nothing unmaps: the region stays mapped at the
    /// same address forever and only changes which list names it. The wait that
    /// genuinely is needed moved to where re-tenanting happens, in
    /// `region_pool::claim_region`, where it is paid once per claim instead of
    /// once per wave.
    pub fn release_empty_arenas(&self) -> Result<usize> {
        let mut total_freed = 0;
        for backing in &self.backings {
            total_freed += backing.release_empty_arenas()?;
        }
        Ok(total_freed)
    }

    /// Create the arenas that mid-wave refusals recorded, and answer with how
    /// many were made.
    ///
    /// **Call this from the wave loop, between forwards.** The sealing thread is
    /// what gets refused, but it is the wrong party to satisfy the demand: it has
    /// to *find* a gap, and on this engine the gap between one wave and the next
    /// is narrower than a sealing pass, so it never won one — the `1088 B` class
    /// stayed frozen at 49 arenas with 75 regions free while the hot→warm drain
    /// sat at 634 MiB and never moved.
    ///
    /// The wave loop does not have to find the gap; it *is* the gap. Running here
    /// makes the demand converge in one wave instead of never.
    #[cfg(feature = "cuda")]
    pub fn create_deferred_arenas(&self) -> Result<usize> {
        let mut made = 0;
        for backing in &self.backings {
            made += backing.create_deferred_arenas()?;
        }
        Ok(made)
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

    /// Bytes of KV the reservation can still hold — free regions × the region
    /// size, an exact count. `None` on non-CUDA, or before this device has a
    /// reservation. See [`candle_nn::kv_cache::vram_budget_available`].
    pub fn vram_budget_available(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        return candle_nn::kv_cache::vram_budget_available(&self.device);
        #[cfg(not(feature = "cuda"))]
        return None;
    }

    /// The VRAM Governor for this session's device, if installed. Lets the
    /// scheduler drive admission/eviction off the governor's honest measurement,
    /// forecast, and thresholds.
    pub fn vram_governor(&self) -> Option<std::sync::Arc<candle::vram::VramGovernor>> {
        #[cfg(feature = "cuda")]
        if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
            return candle::vram::get(gpu_id);
        }
        None
    }

    /// Our CUDA memory pool's `(used, reserved)` bytes — what our allocations
    /// actually occupy (model weights + KV + activations) vs the high-water
    /// bytes the pool has reserved from the OS. `reserved - used` is held but
    /// reusable; `reserved` is why the driver's `free` reads near zero. The
    /// real diagnostic for "what's using VRAM". `None` on non-CUDA / failure.
    /// The formats live (unsealed) KV chunks occupy — see
    /// [`candle_nn::kv_cache::active_kv_formats`]. Admission prices candidates
    /// in THESE, not the configured sealed formats.
    pub fn active_kv_formats(
        &self,
    ) -> (candle_nn::kv_cache::KvFormat, candle_nn::kv_cache::KvFormat) {
        candle_nn::kv_cache::active_kv_formats(
            self.config.k_format,
            matches!(self.device, Device::Cuda(_)),
        )
    }

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

    /// GPU KV arena occupancy per size class. Reads the shared GID pool
    /// via layer 0's backing (arenas pool globally across same-config layers, so
    /// one backing is the whole model). `None` when there are no backings.
    pub fn kv_gpu_class_stats(&self) -> Option<GpuArenaClassStats> {
        self.backings.first().map(|b| b.gpu_arena_class_stats())
    }

    /// Create a view sequence that borrows KV blocks from a parent.
    ///
    /// Allocates a new sequence slot and populates it with Arc-shared refs to
    /// the specified parent blocks.  Callers write new tokens into the view, then
    /// call [`finalize_view`] to transfer those blocks back to the parent.
    ///
    /// **`visible_block_ranges` must be non-empty.** It names the blocks to
    /// borrow, so an empty slice borrows nothing and yields a view whose K/V is
    /// empty while it decodes at its parent's position — fluent, plausible text
    /// with no relation to the prompt, and no error anywhere. "Empty means every
    /// block" is the *scheduler wrapper's* convention; it expands to
    /// `[(0, total_blocks)]` before calling here. A caller that genuinely wants
    /// to borrow nothing wants [`Self::create_sequence`]. A zero-length range
    /// (`[(0, 0)]`) is still accepted: that is a parent with no blocks yet.
    pub fn create_view_sequence(
        &mut self,
        parent_idx: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<ViewSequence> {
        if visible_block_ranges.is_empty() {
            candle::bail!(
                "create_view_sequence: no visible block ranges for a view of sequence \
                 {parent_idx}. Pass the ranges to borrow — `[(0, block_count)]` for the \
                 whole parent — or use `create_sequence` for a child that starts empty. \
                 A view that borrows nothing decodes from an empty context and reads \
                 perfectly."
            );
        }
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

    /// The layer-0 writer boundary for a sequence — the first chunk index the
    /// slot may write into (chunks below it are Arc-shared / sealed).
    pub fn writer_start_idx(&self, idx: usize) -> Option<usize> {
        self.backings.first()?.writer_start_idx_for_seq(idx)
    }

    /// Mark every block `idx` currently holds (all layers) as immutable
    /// prefix — later writes append after them in a fresh writer chunk. The
    /// turn-seal re-prefill calls this after its truncate so the clean grid
    /// lands exactly at the seal anchor block.
    pub fn seal_writer_boundary(&self, idx: usize) -> Result<()> {
        for backing in &self.backings {
            backing.seal_writer_boundary(idx)?;
        }
        Ok(())
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
        // The backings are NOT always uniform: a windowed creep prefill can leave
        // layer 0 one empty (0-token) writer chunk AHEAD of the layers still
        // pending resume, so a section unit sealed mid-creep persists that skew.
        // Reading layer 0 would over-report by that phantom block and make a borrow
        // range (`create_view_sequence`) or seal range ask a lagging layer for a
        // block it doesn't have ("parent slot N has M blocks but range requests M").
        // The MIN across layers is the block count every layer actually holds — the
        // safe common prefix, and since the extra block is always the 0-token writer
        // chunk it carries no context to lose. This mirrors dcd075e0, which moved
        // the TOKEN reader (`sequence_backing_tokens`) to the min for the same skew;
        // the block-count reader was left on layer 0 and is the residual half of it.
        // Returns None only when the slot is unallocated on every layer.
        self.backings
            .iter()
            .filter_map(|b| b.sequence_block_count(idx))
            .min()
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

    /// The dtype activations arrive in for this session — what a model's norm
    /// weights must be materialised in before it runs a forward.
    ///
    /// Deliberately **not** [`Self::dtype`], and the difference is not cosmetic.
    /// `dtype` describes *sealed storage* and falls back to BF16 for a quantized
    /// format, which is right for the accounting it feeds. The forward instead
    /// derives its activation dtype from the sequence's live caches, and
    /// [`candle_nn::kv_cache::KvCache::dtype`] reports **F16** for a quantized
    /// backing — the live arena really is F16 (K in `R16`, V in plain F16), so
    /// that is the dtype the norms will actually see. Reading `dtype` here
    /// yields BF16 weights for an F16 forward, which the norm refuses.
    pub fn activation_dtype(&self) -> DType {
        crate::models::batched_model::activation_dtype(
            self.config.k_format.dtype().unwrap_or(DType::F16),
        )
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
    ) -> candle::Result<Vec<R16ChunkDump>> {
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
    ) -> candle::Result<Vec<R16ChunkDump>> {
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
    ) -> candle::Result<Vec<Vec<R16ChunkDump>>> {
        layer_indices
            .iter()
            .map(|&layer_idx| match self.backings.get(layer_idx) {
                Some(backing) => backing.gather_r16_kv_probe(seq_idx, block_range),
                None => Ok(vec![]),
            })
            .collect()
    }

    /// GPU-side wide-Q provenance capture: one kernel launch across ALL layers
    /// reads the R16 Q, signs it, and bit-packs it — only the packed bits come
    /// back to the host. Replaces the per-layer f16 K/Q/V D2H + CPU sign pass
    /// (~48 blocking round-trips per scope) with one HtoD (pointers), one launch
    /// and one DtoH (a few KB). The caller (`gather_wide_sigs`) assembles the raw
    /// per-token `WideQSig` from `packed` and folds it — bit-identical to the CPU
    /// path. Returns `None` (caller falls back to the CPU gather) when not CUDA,
    /// when any layer has no R16 blocks, when the layers' block sets disagree, or
    /// when `sub_head_dim > 64` (can't pack into a u64).
    pub fn gather_provenance_sign_packed(
        &self,
        seq_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Option<ProvSignPacked>> {
        let Some((all_ptrs, block_indices)) =
            self.resolve_provenance_q_ptrs(seq_idx, block_range)?
        else {
            return Ok(None);
        };
        let sub_head_dim = self.prov_sub_head_dim();
        let packed = self.run_prov_sign_pack(&all_ptrs, sub_head_dim)?;
        if packed.is_empty() {
            return Ok(None);
        }
        Ok(Some(ProvSignPacked {
            packed,
            block_indices,
            n_layers: self.backings.len(),
            n_kv_head: self.n_kv_head(),
            // Derived from the same rule as `sub_head_dim` above, and it has to
            // be: the two travel together in this struct, and a consumer that
            // reconstructs `head_dim = n_palette * sub_head_dim` from a derived
            // width beside a constant band count reads 128 for a 256-wide head.
            n_palette: self.prov_n_palette(),
            sub_head_dim,
        }))
    }

    /// How many palette bands a head is **physically stored in** — the number
    /// of Q pointers `provenance_q_ptrs` emits per head.
    ///
    /// **This is a property of the arena, not a free choice.** The R16 chunk
    /// lays each head out in `N_PALETTE` bands of `head_dim / N_PALETTE` dims, the
    /// pointer resolution walks exactly that many bands, and the sign-pack
    /// kernel's contract is `sub_head_dim = head_dim / N_PALETTE`. Deriving a
    /// different band count here does not re-band the storage — it only makes
    /// this side disagree with it.
    ///
    /// It was briefly derived as "the smallest power of two with
    /// `head_dim / p <= 32`" to stop the fast path declining at `head_dim`
    /// 256. That reasoning was right about the symptom and wrong about the
    /// fix: at 256 it yields 8 bands of 32, so the kernel reads only the low
    /// 32 dims of each physical 64-dim band and the host lays band `p` at
    /// `p * 32` instead of `p * 64` — half of every signature dropped, the
    /// rest dim-permuted, and no longer bit-identical to the CPU fold it is
    /// documented to match.
    ///
    /// The band count was never the thing to change. What declined at 256 was
    /// the kernel's **word width**: it packed a band into a `u32`, and the
    /// physical band there is 64 dims. It packs a `u64` now, so a 256-wide head
    /// takes the fast path with its real banding intact.
    pub fn prov_n_palette(&self) -> usize {
        candle_nn::kv_cache::N_PALETTE
    }

    /// The provenance sign-pack sub-band width (`head_dim / n_palette`), or 0
    /// when the geometry cannot be packed into one word — callers treat 0 as
    /// "use the CPU path".
    ///
    /// The bound is 64, the kernel's word width and the width of a physical R16
    /// band at `head_dim` 256 — so both production geometries (128 → 32-dim
    /// bands, 256 → 64-dim bands) take the GPU path. It was 32, which declined
    /// at 256 and sent every seal of the hybrid through a full R16 device→host
    /// copy; the signatures were correct, so nothing but a path assertion could
    /// see it (`hybrid_capture_takes_the_gpu_sign_pack_path`).
    ///
    /// A wider band than 64 still declines rather than silently truncating: the
    /// band is the arena's, not this function's, and narrowing it here would
    /// drop dims (see [`Self::prov_n_palette`]).
    pub fn prov_sub_head_dim(&self) -> usize {
        let head_dim = self.head_dim();
        if head_dim == 0 {
            return 0;
        }
        let sub = head_dim / self.prov_n_palette();
        if sub == 0 || sub > candle_kernels::simple::prov_sign_pack::MAX_SUB_HEAD_DIM {
            0
        } else {
            sub
        }
    }

    /// Resolve the concatenated R16 Q-chunk pointers across ALL layers for a
    /// sequence's blocks — the input to the sign-pack kernel — plus the block
    /// index set (identical across layers). Split from the launch so a caller can
    /// resolve many scopes while their slots are alive, then batch ONE kernel
    /// launch over all of them (the arena chunks stay valid via the sealed K/V's
    /// RAII refs). `None` (→ CPU path) when not CUDA, when any layer has no R16
    /// blocks, when the layers' block sets disagree, or when `sub_head_dim > 32`.
    pub fn resolve_provenance_q_ptrs(
        &self,
        seq_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> candle::Result<Option<(Vec<i64>, Vec<usize>)>> {
        if self.backings.is_empty() || self.n_kv_head() == 0 || self.prov_sub_head_dim() == 0 {
            return Ok(None);
        }
        let mut all_ptrs: Vec<i64> = Vec::new();
        let mut block_indices: Option<Vec<usize>> = None;
        for backing in &self.backings {
            let (ptrs, blocks) = backing.provenance_q_ptrs(seq_idx, block_range)?;
            if ptrs.is_empty() {
                return Ok(None);
            }
            match &block_indices {
                None => block_indices = Some(blocks),
                Some(prev) if *prev != blocks => return Ok(None),
                _ => {}
            }
            all_ptrs.extend(ptrs);
        }
        match block_indices {
            Some(b) if !b.is_empty() => Ok(Some((all_ptrs, b))),
            _ => Ok(None),
        }
    }

    /// Launch the sign-pack kernel over already-resolved (concatenated) Q-chunk
    /// pointers — one kernel + one D2H for the whole batch. Empty on non-CUDA.
    pub fn run_prov_sign_pack(
        &self,
        all_ptrs: &[i64],
        sub_head_dim: usize,
    ) -> candle::Result<Vec<u64>> {
        match self.backings.first() {
            Some(b) => b.run_prov_sign_pack(all_ptrs, sub_head_dim),
            None => Ok(Vec::new()),
        }
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

        let chunks = match backing.live_chunks_as_sealed(seq_idx) {
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

        // Band formats come from the chunk's own tags. A grid built from arena
        // state would report every band of a shared size-class region
        // identically, which is exactly the mixed-format variety this grid
        // exists to display. Kept strongly typed — string conversion happens
        // only where a label is needed (tally display or grid short-label).
        let band_fmt = |tags: &[u8], h: usize, p: usize| -> Option<KvFormat> {
            tags.get(h * N_PALETTE + p)
                .copied()
                .and_then(KvFormat::from_tag)
        };
        let is_real_quant = |fmt: Option<KvFormat>| -> bool {
            matches!(fmt, Some(KvFormat::Quantized(qf)) if qf != QuantFormat::R16)
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
                    let kf = band_fmt(&chunk.k_fmt, h, p);
                    let vf = band_fmt(&chunk.v_fmt, h, p);
                    has_quant |= is_real_quant(kf) || is_real_quant(vf);
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

    /// Token count actually covered by a sequence's live block table — the sum
    /// of live-chunk token counts in the layer-0 backing, i.e. exactly the
    /// `cum` the slot-header build walks (all layers share chunk boundaries).
    /// This is the physical ground truth the wave-boundary offset reconciler
    /// compares `session.offset` against; `current_seq_len` is the write
    /// cursor and reads 0 for freshly injected slots whose tables already
    /// hold sealed tokens, so it cannot serve as the backing length.
    pub fn sequence_backing_tokens(&self, idx: usize) -> Option<usize> {
        let caches = self.sequence_caches(idx)?;
        if caches.caches.is_empty() {
            return None;
        }
        // The token count the live block table covers, taken as the MINIMUM
        // across layers — not layer 0 alone. During a windowed wave the creep
        // prefill advances layers incrementally (layer 0 first), so layer 0 runs
        // AHEAD of the layers still pending resume. Reading only layer 0 would
        // let the forward-entry reconciler advance the slot's offset past those
        // lagging layers; a decode that then reads a lagging layer trips the
        // slot-header count invariant ("block table lost chunks"). The minimum is
        // the token prefix EVERY layer has materialised, so an offset clamped to
        // it is valid on every layer, and as the wave completes and the layers
        // converge the minimum rises to the true length. For a uniform (settled)
        // slot every layer is equal, so this is identical to reading layer 0.
        let mut min_cum: Option<usize> = None;
        for cache in &caches.caches {
            let mut cum: usize = 0;
            cache.k_cache().chunked_visit_live_chunks(|it| {
                for c in it {
                    cum += c.token_count as usize;
                }
            });
            min_cum = Some(min_cum.map_or(cum, |m: usize| m.min(cum)));
        }
        min_cum
    }

    /// Set a sequence's logical offset outright. Used by the wave-boundary
    /// offset reconciler to clamp a slot whose offset ran AHEAD of its physical
    /// backing (a projection that dropped un-liftable sections injects fewer
    /// tokens than the planner counted): every wave's kv metadata is derived
    /// from this offset, and the attention kernels resolve each position
    /// through the slot's physical block table — an offset past the backing
    /// sends them past the end of the slot's staged state into neighboring
    /// uploads. The backing length is the single source of truth at a wave
    /// boundary; positions are slot-relative (slice ropes), so the clamped
    /// value is also the correct RoPE base for new tokens.
    pub fn set_sequence_offset(&mut self, idx: usize, offset: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset = offset;
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

    /// [`Self::snapshot_sequence_per_layer`] restricted to the block-index range
    /// `[start_block, end_block)` on every layer. Cost scales with the range —
    /// the glue-island capture snapshots a couple of chunks out of a
    /// multi-hundred-block slot. The caller guarantees the layers' block tables
    /// are aligned over the range (the projection walk builds them uniformly;
    /// same contract as `slice_per_layer_sealed` over a full snapshot).
    pub fn snapshot_sequence_blocks(
        &self,
        idx: usize,
        start_block: usize,
        end_block: usize,
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        let mut out = Vec::with_capacity(self.backings.len());
        for backing in &self.backings {
            out.push(
                backing
                    .record_turn_blocks(idx, start_block, end_block)
                    .map_err(|e| candle::Error::Msg(format!("snapshot_sequence_blocks: {e}")))?,
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
            let bpe = total_actual / total_elements as f64;
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
    /// How many layers actually have a Q to capture — the fold's layer count.
    ///
    /// Equal to `num_layers` on a uniform transformer and to the **attention**
    /// layer count on a hybrid, where three quarters of the stack is recurrent
    /// and has no Q in a KV cache to read. The provenance fold groups
    /// `[n − 2, 1, 1]` over this, so passing transformer depth on a hybrid
    /// would size the lower group for layers that contribute nothing.
    pub provenance_capture_layers: usize,
    /// Whether this model can compute K/V for tokens inserted **mid-sequence**
    /// — a glue island the projection reserves a gap chunk for.
    ///
    /// **Gap-fill is not the same as prefill.** Appending at the writer tail is
    /// ordinary prefill and every model does it; filling a hole in the middle of
    /// a sequence requires each inserted token's output to depend only on what
    /// precedes it *positionally*, which attention gives for free and a
    /// recurrence cannot give at all — token *t*'s output depends on the
    /// accumulated state over everything before it, in order, and that state has
    /// already moved past the hole.
    ///
    /// A real model property, like `head_dim`. The planner reads it **before**
    /// planning rather than discovering it from a bail inside `forward_wave`,
    /// because by then the projection has already been built around islands the
    /// model cannot fill.
    pub can_gap_fill: bool,
    /// Whether this model keeps per-sequence state outside the KV cache — the
    /// recurrent matrices and conv tails of a hybrid stack.
    ///
    /// The mirror of [`Self::can_gap_fill`], and needed at the same altitude:
    /// the prompt builder has to know, before it starts, whether a branch
    /// checkpoint is something it must compute. A conversation on a model that
    /// carries no such state needs none, and the pass must not run — 96
    /// full-prompt prefills for a quantity nothing will read.
    ///
    /// [`ManagedBatchedModel::carries_recurrent_state`] answers the same
    /// question where a `&dyn` model is in hand; this carries it to code that
    /// only has the copied properties.
    pub carries_recurrent_state: bool,
}

/// Result of a re-entrant [`ManagedBatchedModel::forward_batched_layers`] wave.
///
/// Exactly one of the two fields is populated: `residual` when the wave paused
/// before the last layer (the packed inter-layer stream to persist and feed back
/// next wave), or `logits` when it ran the head (one row per input sequence).
pub struct WaveStep {
    /// Packed residual stream to persist + resume, when the range stopped short
    /// of the final layer.
    pub residual: Option<Tensor>,
    /// Per-sequence logits (in `seq_indices` order), when the range reached the head.
    pub logits: Option<Vec<Tensor>>,
}

/// A [`WaveStep`] together with what keeps it valid.
///
/// The head's outputs are carved from the forward-scoped span, which is
/// reclaimed when its generation drops. Returning the step alone would hand the
/// caller tensors whose memory is reusable the moment the callee returns — sound
/// only by the convention that everyone samples before the next forward, which
/// nothing checks.
///
/// So the guard travels **with** the result. The span cannot be reclaimed while
/// this value is alive, and [`Deref`](std::ops::Deref) is what makes that
/// checkable rather than merely true: reading through it borrows `self`, so a
/// reference to the logits keeps the guard alive, while *moving* the logits out
/// is rejected — you cannot move out of a `Deref`. Every site that only reads
/// compiles unchanged; every site that would have taken the tensors away from
/// their guard is a compile error, which is exactly the set worth looking at.
pub struct WaveResult {
    step: WaveStep,
    /// Held, never read. `None` off-CUDA and for waves that never opened a
    /// forward span.
    #[cfg(feature = "cuda")]
    _forward: Option<candle_nn::kv_cache::WaveGeneration>,
}

impl WaveResult {
    /// Wrap a step whose outputs do not sit on a forward span.
    pub fn owned(step: WaveStep) -> Self {
        Self {
            step,
            #[cfg(feature = "cuda")]
            _forward: None,
        }
    }

    /// Wrap a step whose outputs were carved from `forward`'s span.
    #[cfg(feature = "cuda")]
    pub fn on_span(step: WaveStep, forward: Option<candle_nn::kv_cache::WaveGeneration>) -> Self {
        Self {
            step,
            _forward: forward,
        }
    }

    /// The logits, copied off the span so they outlive this result.
    ///
    /// The sanctioned escape, and it really copies. Prefer reading through
    /// [`Deref`](std::ops::Deref) — `result.logits` — which costs nothing and
    /// keeps the guard doing its job. This exists for callers that genuinely
    /// need the values after the span is reclaimed: a caller that returns them
    /// upward, or one that accumulates across several forwards.
    ///
    /// Empty when the wave paused before the head: a glue-only wave carries no
    /// logits and that is not a failure. A *failed copy* is a different thing and
    /// propagates — the two used to collapse into the same empty `Vec`, which the
    /// scheduler reads as "this wave produced nothing" and turns into a silently
    /// token-less turn.
    pub fn logits_owned(&self) -> Result<Vec<Tensor>> {
        match self.step.logits.as_ref() {
            None => Ok(Vec::new()),
            Some(ls) => ls.iter().map(|t| t.to_owned_tensor()).collect(),
        }
    }

    /// Take the residual stream, which is pool-backed and outlives the span.
    ///
    /// Deliberately available by value where the logits are not: a paused wave's
    /// residual is persisted and resumed on a *later* forward, so it cannot live
    /// on a span that resets at the end of this one — and does not.
    pub fn into_residual(mut self) -> Option<Tensor> {
        self.step.residual.take()
    }
}

impl std::ops::Deref for WaveResult {
    type Target = WaveStep;

    fn deref(&self) -> &WaveStep {
        &self.step
    }
}

/// **You don't need to implement this trait manually.** Any type that implements
/// [`BatchedModel`] automatically gets a `ManagedBatchedModel` implementation
/// via the blanket impl.
pub trait ManagedBatchedModel {
    /// The geometry [`candle_nn::kv_cache::WavePlan`] prices a wave from.
    ///
    /// Exposed here as well as on [`BatchedModelCore`] because the scheduler
    /// holds the model behind this trait object, and admission has to price a
    /// wave's transient buffers before it decides what to admit into it.
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry;

    /// Widest prefill this model will run in one forward, in tokens.
    ///
    /// The smallest of three unrelated ceilings
    /// (`docs/elastic_vram_partition.md` §7: `R = min(8192, transient-fits,
    /// KV-fits)`):
    ///
    /// * `MAX_PREFILL_TOKENS` — where GPU compute saturates. Above it a wider
    ///   forward costs the same per token, so slicing is free.
    /// * [`candle_nn::kv_cache::WavePlan::max_rows_within`] — how many rows the
    ///   FFN span actually holds, computed from this model's geometry. On a MoE
    ///   model the expert chain sees `rows × experts_per_tok`, so the same token
    ///   count needs several times the span a dense model would.
    /// * **What the KV side can still hold.** The admit phase claims every chunk
    ///   the wave will write before it computes anything, so a wave admitted
    ///   wider than the free regions can back is one that fails partway through
    ///   claiming — with some layers extended and some not. Capping here turns
    ///   that into a narrower wave, which is the outcome the whole partition is
    ///   for: under pressure the engine slows down instead of failing.
    ///
    /// Taking the min is what stops a wave being sized by a constant that has
    /// never seen the model. Each term falls back to permissive when it cannot
    /// answer — no reservation yet, or a plan that cannot price a single row —
    /// because a zero-width wave makes no progress, and refusing here would abort
    /// a forward that can still run.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        let mut cap = MAX_PREFILL_TOKENS;
        let fits = WavePlan::new(self.wave_geometry(act_dtype)).max_rows_within(WAVE_FFN_BYTES);
        if fits > 0 {
            cap = cap.min(fits);
        }
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            cap = cap.min(kv_fits);
        }
        cap
    }

    /// Rows the KV side has room to admit, or `None` when it cannot say.
    ///
    /// Every token writes one K and one V element per KV head per layer, so the
    /// per-row cost is fixed by geometry and the only variable is how many free
    /// regions remain. Deliberately measured against `free + blocked` rather
    /// than against the elastic floor: the boundary moves at the expert
    /// pipeline's end of pass, so ground the *weight side* is holding is not
    /// available to this wave however willing it might be to give it up later —
    /// but ground the previous forward's transient tier has blocked is. This
    /// cap is read at the top of `forward_wave`, when that tier is still
    /// standing; the forward it sizes releases the tier in its own phase 0,
    /// before any claim priced here runs. Counting only `free` made every wide
    /// prefill's own tier throttle the next one: the tier's ground read as
    /// gone, the cap collapsed to a few hundred tokens, and a ten-context
    /// prefill fragmented into a dozen narrow waves that each re-loaded the
    /// expert working set — bulk prefill fell from ~2.1K t/s to ~700.
    ///
    /// One region of margin, because a sequence's chunks do not pack a region
    /// exactly and the last one is partly wasted.
    fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        let stats = candle_nn::kv_cache::region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_sub(1);
        let per_row = 2 * self.n_kv_head() * self.head_dim() * act_dtype.size_in_bytes();
        let per_row_all_layers = per_row.checked_mul(self.num_layers())?;
        if per_row_all_layers == 0 {
            return None;
        }
        // Admissible KV = what stands free PLUS what the model's elastic
        // boundary would cede to a stuck claim ([`Self::reclaimable_kv_bytes`]).
        // Counting only the free regions under-reports capacity by whatever the
        // weight side happens to be holding above its floor (tens of GB on the
        // streamed-expert engine) and pre-slices pure-prefill sweeps against a
        // number the first stuck claim would have doubled. The result is still
        // bounded above by `MAX_PREFILL_TOKENS` in the callers.
        let kv_bytes = free
            .saturating_mul(candle_nn::kv_cache::REGION_BYTES)
            .saturating_add(self.reclaimable_kv_bytes());
        let rows = kv_bytes / per_row_all_layers;
        // **Never zero.** A width cap of nought is not a narrow wave, it is no
        // wave — and once the KV side is full it would be permanent: no forward
        // runs, so nothing completes, so nothing is freed, so the cap stays at
        // nought. The partition's answer to genuine exhaustion is a refused
        // claim and the relief pass behind it, both of which need a forward to
        // have been attempted.
        Some(rows.max(1))
    }

    /// KV-side bytes this model's memory layout could free ON DEMAND beyond
    /// what stands free — e.g. an elastic weight/KV boundary that cedes expert
    /// ground to a stuck KV claim. Counted by [`Self::kv_width_cap`] when
    /// sizing a prefill wave, so the wave is sliced against what the partition
    /// CAN admit, not what it happens to have standing free. `0` for models
    /// with a static layout (nothing to cede).
    fn reclaimable_kv_bytes(&self) -> usize {
        0
    }

    /// Re-materialise every norm weight in the activation dtype — see
    /// [`crate::models::batched_model::BatchedModelCore::maybe_change_dtype`].
    ///
    /// On this trait too because the scheduler holds the model behind it, and
    /// the activation dtype is chosen where the KV cache is configured rather
    /// than at load.
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()>;

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
            // Attention computes each token's output from what precedes it
            // positionally, so a hole in the middle of a sequence is fillable.
            // The default is the uniform-transformer answer; a model whose
            // per-sequence memory is a recurrence overrides it.
            // Uniform transformer: every layer attends, so every layer has a Q.
            provenance_capture_layers: self.num_layers(),
            can_gap_fill: !self.carries_recurrent_state(),
            carries_recurrent_state: self.carries_recurrent_state(),
        }
    }

    /// Co-batched continuous-fair-wave forward: run decode (q=1) + prefill (q=N) +
    /// glue (q=G) rows through ONE forward (layer range `[layer_start, layer_end)`)
    /// with the mixed attention dispatch and the single shared FFN/MoE
    /// (`docs/continuous_fair_waves.md`). Glue rows are staged on the session via
    /// `set_pending_glue` before the call, as for a plain glue forward.
    ///
    /// Returns a [`WaveStep`]: `residual` (persist + resume) when the range stops
    /// short of the head, else `logits` for the **decode + prefill** rows only
    /// (in `decode_seqs ++ prefill_seqs` order — glue rows carry no logits).
    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual_in: Option<Tensor>,
    ) -> Result<WaveResult>;

    // ── Speculative decoding (model-agnostic hook) ──────────────────────────────
    //
    // Three composable methods let ANY model do lossless speculative decoding through the
    // generic `speculative_decode_step` driver. A model with no drafter inherits the defaults
    // and `speculative_decode_step` degrades to a single plain decode — so the hook is always
    // safe to call. A model with a drafter overrides `speculative_draft` (and, for the actual
    // speedup, `verify_block`). The accepted tokens are always drawn by the caller's own
    // `TokenChooser` from this model's own logits, so the output is distributed exactly as plain
    // decoding through that chooser would be, regardless of draft quality — bit-identical under
    // `GreedyChooser`, and drawn from the identical distribution under a sampler.

    /// Tokens this model wants each sequence to draft on a wave of `width` sequences.
    ///
    /// **Zero means take a plain decode row**, and that is the default: a model with no drafter
    /// never speculates, and one whose ladder has not been measured does not guess.
    ///
    /// The budget belongs to the model because the trade does. A verify block scores `k + 1` rows
    /// where a plain decode scores one, and writes `k + 1` KV entries where a plain decode writes
    /// one. Both are nearly free while the wave is memory-bandwidth-bound — a narrow decode reads
    /// the entire weight set to score a handful of rows — and both cost what they weigh once it is
    /// compute-bound. Where that turns over depends on the checkpoint's shape: how much weight a
    /// step reads, whether experts stream, how wide the KV rows are. It is measured per model, by
    /// the `speculative_decode_*` gates' width sweep, and recorded on the model.
    ///
    /// Called once per wave, so it must be cheap and must not touch the device.
    fn draft_budget(&self, width: usize) -> usize {
        let _ = width;
        0
    }

    /// Draft up to `max_len` speculative next-tokens for **every** sequence in the step's
    /// cohort, each following its own `committed` token, using the model's own drafter (e.g.
    /// an MTP / DSpark head). Proposals only — the caller verifies them and keeps the
    /// converging prefix. Returns one proposal list per entry of `seqs`, in order; an empty
    /// list means that sequence takes a plain decode row instead of a verify block.
    ///
    /// **Takes the whole cohort, not one sequence, because a drafter is weight-bound.** The
    /// recurrence is serial *within* a sequence — step `j` needs `embed(argmax)` of step
    /// `j-1` — but step `j` of one sequence is independent of step `j` of every other, so the
    /// cohort's `j`-th steps are one batched pass. That matters because the dominant cost is
    /// reading weights, not arithmetic: on the 9B a drafted token spends 1.5 ms of 2.7 ms in
    /// one full read of the ~795 MiB output projection to score a single row, and four rows
    /// measured 1.69 ms against one row's 1.66. Drafting per sequence therefore reads the same
    /// weight once per session per step; drafting per cohort reads it once. That is also why
    /// a drafter's break-even used to rise with width — the answer is to batch the drafter,
    /// not to stop drafting at width.
    ///
    /// **Default: no drafter → no proposals**, and the step below degrades to one plain decode.
    /// A model drafts by overriding this with a head its checkpoint actually carries — the
    /// `qwen35` lineage's NextN/MTP block (`qwen35::mtp`), DeepSeek's DSpark. There is
    /// deliberately no generic fallback proposer: one that guesses from the sequence's own text
    /// can only re-propose what the sequence already said, which is worth nothing on the
    /// reasoning and first-draft tokens a decode loop actually spends its time on, and it costs
    /// a per-sequence token history that grows with context to buy it.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let _ = (session, committed, max_len);
        Ok(vec![Vec::new(); seqs.len()])
    }

    /// Verify a block of `tokens` for `seq`: append them and return one next-token logits row
    /// `[1, vocab]` per input token (the model's prediction after each prefix). Default: run the
    /// tokens as sequential `forward_wave` decode steps — correct for ANY model, but no speedup
    /// (this is what makes speculative decode *lossless* by default). Models override with a
    /// single batched forward over the whole block for the throughput win. Advances the sequence
    /// by `tokens.len()`; the driver truncates back to the accepted length.
    ///
    /// **A model that overrides [`Self::speculative_draft`] but not this should not be driven
    /// speculatively.** The default spends one forward per block position to buy at most that
    /// many tokens, so a drafted step is break-even at best and a loss whenever a proposal is
    /// rejected. It stays correct so a new drafter can be brought up against it and then made
    /// fast; it is not a throughput path.
    fn verify_block(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: &[u32],
        layer_end: usize,
    ) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            let t = Tensor::from_vec(vec![tok], (1, 1), self.device())?;
            let step = self.forward_wave(
                session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                layer_end,
                None,
            )?;
            // Copied off the span (`logits_owned`): the rows are accumulated
            // across this loop's forwards, so each iteration's span-backed
            // views would be invalidated by the next forward.
            let logits = step.logits_owned()?.pop().ok_or_else(|| {
                candle::Error::msg("verify_block: forward_wave produced no logits")
            })?;
            session.advance_sequence(seq, 1)?;
            out.push(logits);
        }
        Ok(out)
    }

    /// Whether this model carries per-sequence state that **cannot be rewound**.
    ///
    /// A recurrent state is an accumulated sum with no per-token decomposition:
    /// there is no suffix to remove and no inverse to apply. Rewinding the K/V
    /// under one is silent corruption — the model answers from tokens the cache
    /// no longer holds, fluently and without an error.
    ///
    /// This is a real model property, like `head_dim`, not a feature flag. It
    /// gates the one operation in the engine that means "put this sequence back
    /// the way it was `n` tokens ago" ([`Self::truncate_sequence`], reached
    /// only from speculative decode), refusing it up front rather than letting
    /// it corrupt state a layer down. `false` for every model whose entire
    /// per-sequence memory is the paged K/V.
    ///
    /// It also gates the snapshot / fork / branch-checkpoint machinery, so what
    /// qualifies matters. **Per-sequence state outside the K/V is not enough —
    /// the state must be irrecoverable.** `latent_moe`'s engine (DeepSeek-V4)
    /// carries a per-sequence compressor and provenance gallery, and answers
    /// `false` deliberately: both are *derived*, rebuildable from the corpus
    /// that is already durable, so there is nothing to snapshot that a replay
    /// could not reproduce. A delta-rule matrix has no such source — it is the
    /// only record of the tokens that built it — which is why the hybrid is the
    /// one lineage that answers `true`.
    ///
    /// The consequence of getting this wrong is asymmetric. A model that
    /// answers `true` without needing to pays ~63 MiB of export per seal for
    /// bytes nothing reads. One that answers `false` when it should not gets
    /// silently zeroed state and reads fluently, which is the failure this whole
    /// path exists to remove.
    fn carries_recurrent_state(&self) -> bool {
        false
    }

    /// Whether a speculative block's rejected tail can be rolled back on this model.
    ///
    /// **Distinct from [`Self::carries_recurrent_state`], which used to answer both.**
    /// Carrying recurrent state means there is something outside the session's KV to
    /// snapshot; for as long as no model could rewind such a state, one declaration
    /// served for both questions and the speculative entry point refused on it. The
    /// hybrid's verify replay separates them. It still carries state — nothing else
    /// records the tokens that built `S` — and it can now rewind *inside a block it
    /// just verified*, by replaying the mixer over the accepted prefix from the
    /// ping-pong entering state the wave left untouched (`qwen35::spec`).
    ///
    /// Default: a model carrying recurrent state cannot rewind, which keeps every
    /// model that has not implemented a replay refused exactly as before.
    ///
    /// Overriding to `true` is a claim about [`Self::truncate_sequences`], not about
    /// the state: that it restores the recurrence exactly for the targets it accepts.
    /// It is not a claim that every offset is rewindable — a replay covers the block
    /// it stashed operands for and nothing else, so `truncate_sequences` is what
    /// refuses a target with no rewind point, at the one place that knows.
    fn can_rewind_speculative_block(&self) -> bool {
        !self.carries_recurrent_state()
    }

    /// Roll `seq` back to exactly `tokens` tokens after a speculative verify — called by the
    /// driver with `pos + kept` once the accepted prefix is known. Default: the session-level KV
    /// truncation. A model whose forward maintains per-sequence streaming state OUTSIDE the
    /// session's KV (compressors, corpus galleries) overrides this to roll that state back in the
    /// same call — otherwise a partial accept leaves rejected draft tokens absorbed in state the
    /// truncation can't see, and later attention reads corrupted context.
    ///
    /// **Reached only by the speculative verify path**, despite the broad name.
    /// A model that cannot express a rewind at all leaves
    /// [`Self::can_rewind_speculative_block`] at its default `false` and is
    /// refused at the entry point, so it never arrives here — the guard is the
    /// declaration, not a bail inside an override raised after the driver has
    /// committed. A model that *can* rewind, but only within a block it stashed
    /// operands for, accepts here and refuses the uncovered offsets itself.
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        // Already there: nothing to roll back. This is the common case, not a
        // corner one — every decode step ends by reconciling each sequence to
        // what it kept, and a step that kept everything it wrote (any plain
        // decode row, and a fully accepted block) reconciles to the offset the
        // sequence already stands at. Without this the KV backing would seal,
        // release and re-claim chunks around the live tail once per sequence per
        // step, against the compressor and the sealing thread.
        if session.sequence_offset(seq) == Some(tokens) {
            return Ok(());
        }
        session.truncate_sequence_to_tokens(seq, tokens)
    }

    /// The whole step's rollbacks in one call — `(seq, tokens)` per sequence.
    /// Default: [`Self::truncate_sequence`] per target, which is correct for
    /// any model. A model whose rollback does real per-sequence work overrides
    /// this to batch it — the qwen35 lineage's recurrent replay runs one
    /// launch pair per DeltaNet layer for the whole cohort here, against one
    /// per layer PER SEQUENCE through the per-target default.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        for &(seq, tokens) in targets {
            self.truncate_sequence(session, seq, tokens)?;
        }
        Ok(())
    }

    /// A sequence id is going away: release whatever per-sequence state this
    /// model keeps OUTSIDE the session's KV.
    ///
    /// The session owns the KV and frees it with the slot. A model whose
    /// forward carries per-sequence state of its own — a recurrent store, a
    /// streaming compressor, a corpus gallery — owns that separately and
    /// outlives every session, so nothing else can free it.
    ///
    /// Slot ids are pool indices and **are recycled**, which is what makes this
    /// a correctness hook rather than a leak fix: a recycled id whose first
    /// wave carries a non-zero offset finds the previous conversation's entry
    /// still sitting in the map and inherits it. The model then answers from a
    /// history the cache never had, fluently and without an error.
    fn release_sequence(&self, _seq: usize) -> Result<()> {
        Ok(())
    }

    /// How many sequences this model currently holds recurrent memory for.
    ///
    /// The leak gauge for [`Self::release_sequence`]. Slot ids are recycled pool
    /// indices, so memory that outlives its sequence is not merely wasted VRAM:
    /// the next conversation to land on that id inherits it. `0` for a model
    /// that carries none.
    fn recurrent_memory_count(&self) -> usize {
        0
    }

    /// `child` begins as a copy of `parent`'s per-sequence model state.
    ///
    /// The KV analogue is `create_view_sequence`, which lets a child slot
    /// borrow the parent's blocks zero-copy. State cannot be borrowed — the
    /// child is about to advance it — so it is copied, device-to-device.
    ///
    /// Called at every turn's view carve, not only on an explicit user fork:
    /// the turn loop decodes on a child slot, and a child that starts from
    /// nothing is a model running on whatever fraction of its stack is not
    /// recurrent.
    fn fork_recurrent(&self, _parent: usize, _child: usize) -> Result<()> {
        Ok(())
    }

    /// Read `seq`'s recurrent state back for the turn-seal snapshot record.
    ///
    /// Returns `(schedule_hash, layers)`, or `None` for a model that carries no
    /// such state — so the seal site is model-agnostic and a non-recurrent model
    /// pays nothing.
    ///
    /// **The layer rows, not the record.** `candle-conversation` depends on this
    /// crate and not the reverse, so `SnapshotPayload` cannot be named here; the
    /// scheduler assembles it from these rows, which are field-for-field its
    /// `SnapshotLayer`. Moving the record type down into the model crate would
    /// invert the dependency to save one struct literal.
    ///
    /// Refuses mid-wave — a snapshot must capture a sealed boundary, never a
    /// wave in flight. The seal runs outside the wave, so this is an assertion
    /// on that ordering rather than a case to handle.
    fn export_recurrent(
        &self,
        _seq: usize,
    ) -> Result<Option<(u64, Vec<crate::models::delta_net::ExportedLayerState>)>> {
        Ok(None)
    }

    /// Scatter a snapshot back into `seq`'s recurrent state — the resume path.
    ///
    /// Returns `true` when the state was installed, `false` when this model
    /// carries none. **Errors** on a hash or geometry mismatch rather than
    /// silently starting from zero: falling back to zeros is the same fluent
    /// amnesia this whole path exists to remove, so the caller has to see it
    /// and log it distinguishably.
    fn restore_recurrent(
        &self,
        _seq: usize,
        _schedule_hash: u64,
        _layers: &[crate::models::delta_net::ExportedLayerState],
    ) -> Result<bool> {
        Ok(false)
    }

    /// `child`'s state becomes `parent`'s — the linear join.
    ///
    /// A **move**, not a merge. A view is a linear continuation of its parent:
    /// it saw exactly the parent's tokens plus its own, so its state simply
    /// *is* the parent's new state. (Merging two divergent children has no
    /// defined arithmetic and is out of scope — `S` is an accumulated sum over
    /// one token order, and there is no operation meaning "and also these
    /// other tokens".)
    ///
    /// The rule for when to call it: **the state follows the K/V.** Wherever
    /// the view's decoded blocks transfer to the parent, this transfers with
    /// them; wherever the blocks are abandoned, this is not called and the
    /// state is dropped with them. Getting that backwards is silent in both
    /// directions — a missing move loses a turn's decode, and a spurious one
    /// re-introduces the `<think>` skew.
    fn move_recurrent(&self, _child: usize, _parent: usize) -> Result<()> {
        Ok(())
    }

    /// One speculative step's whole forward: decode every PLAIN sequence's committed token
    /// (`plain` = `(seq, committed)` pairs — sequences whose drafter proposed nothing) and verify
    /// one block per drafted sequence (append `blocks[i]` to `seqs[i]`), returning each plain
    /// row's logits and each block's per-position next-token logits rows. Default: one batched
    /// plain decode wave + sequential [`Self::verify_block`] calls — correct for any model. A
    /// model overrides with ONE wave carrying BOTH cohorts (plain rows leading, verify rows
    /// trailing): the per-wave fixed costs (MoE routing readbacks, expert DMA, launch overhead)
    /// then amortize across every session AND the two cohorts stop paying two launch floors per
    /// step. Advances plain sequences by 1 and drafted sequences by their block length; the
    /// driver truncates drafted sequences back to the accepted lengths.
    /// Plan the rows this verify step's wave carries, and do whatever must
    /// happen *before* that wave opens.
    ///
    /// Returning `None` means this model has no one-wave verify — it has not
    /// promised the head will score every row of a multi-token member, which is
    /// the one thing a verify block needs — and [`Self::verify_blocks`] falls
    /// back to sequential single-token forwards. A model with a drafter
    /// overrides this and [`Self::end_verify`] together.
    ///
    /// **Split from the wave on purpose.** The setup here is real work with real
    /// ordering constraints — the `qwen35` lineage sizes every verifying
    /// sequence's rewind stash before the forward opens, because the arena
    /// refuses a device allocation from inside a wave, and arms the MTP seed
    /// capture over both cohorts; DeepSeek snapshots the streaming
    /// compressor/gallery state its blocks are about to absorb. None of it wants
    /// to know what else is riding the wave, and the caller that *does* know —
    /// the scheduler, folding these rows into the continuous-fair wave — cannot
    /// hand a `&mut self` borrow down into a model method. So the model plans,
    /// the caller runs the wave, and the model reads the rows back.
    ///
    /// `budget` is what the caller asked each sequence to draft this step, and
    /// it is **not** recoverable from `seqs`/`blocks`: an empty `seqs` means
    /// either that drafting was switched off for this wave *or* that the
    /// drafter had nothing to propose from yet — the first step after a prefill,
    /// which is precisely when the setup here has to run so a seed exists to
    /// draft from next time. A model that skips work when nothing will be
    /// drafted must read this, not the block shapes.
    ///
    /// On a wave that fails, the caller calls [`Self::abort_verify`] instead of
    /// [`Self::end_verify`].
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        let _ = (session, blocks, budget);
        // A model that has not overridden this has not promised its head scores
        // every row of a multi-token member, so it cannot verify a block. It can
        // still plan the *undrafted* case, and that is not a courtesy: it is what
        // lets the scheduler run one decode path for every model instead of a
        // speculative one and a plain one. A model with no drafter proposes
        // nothing, every sequence lands in `plain`, and this plan is an ordinary
        // one-token-per-sequence decode wave.
        if !seqs.is_empty() {
            return Ok(None);
        }
        Ok(Some(VerifyPlan {
            decode_seqs: plain.iter().map(|&(s, _)| s).collect(),
            // **The model's own device, not the host.** A plan's rows are cat'd
            // with whatever else the caller has on the wave — the scheduler's
            // creep group and glue live on the device — so a host-side row makes
            // the wave's own concatenation fail on a device mismatch. It only
            // looked safe while a verify wave carried nothing but its own rows.
            decode_inputs: plain
                .iter()
                .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), self.device()))
                .collect::<Result<_>>()?,
            verify_seqs: Vec::new(),
            verify_inputs: Vec::new(),
            rows: plain.len(),
        }))
    }

    /// Read a verify wave's scored rows back, and undo whatever
    /// [`Self::begin_verify`] armed.
    ///
    /// `logits` is the wave's `[decode | verify]` prefix in plan order. Returns
    /// the plain cohort's rows and one row per position of each block, which is
    /// what the accept walk reads. Advancing the sequences by what the wave
    /// wrote belongs here too — the driver truncates back to the accepted
    /// prefix afterwards.
    fn end_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        let _ = blocks;
        if !seqs.is_empty() {
            candle::bail!(
                "end_verify: {} verify blocks came back to a model that planned none",
                seqs.len()
            );
        }
        if logits.len() != plain.len() {
            candle::bail!(
                "end_verify: wave scored {} rows for {} undrafted sequences",
                logits.len(),
                plain.len()
            );
        }
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        Ok((logits, Vec::new()))
    }

    /// Release whatever [`Self::begin_verify`] armed, after a wave that failed.
    ///
    /// The wave rolled its own state back, so a stash or snapshot taken before
    /// it names a rewind point that no longer exists — left in place, a later
    /// truncate would replay from it.
    fn abort_verify(&self, seqs: &[usize]) {
        let _ = seqs;
    }

    fn verify_blocks(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        layer_end: usize,
        budget: usize,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        // The standalone shape: one wave carrying nothing but this step's own
        // rows. The scheduler does not come through here — it runs the same
        // three phases around its own co-batched wave.
        if let Some(plan) = self.begin_verify(session, plain, seqs, blocks, budget)? {
            let issued = issue_verify_wave(
                self,
                session,
                &plan.decode_seqs,
                &plan.decode_inputs,
                &plan.verify_seqs,
                &plan.verify_inputs,
                &WaveCoBatch::standalone(),
                layer_end,
            );
            return match issued {
                Ok(out) => {
                    if out.logits.len() != plan.rows {
                        self.abort_verify(seqs);
                        candle::bail!(
                            "verify_blocks: wave scored {} rows, plan wanted {}",
                            out.logits.len(),
                            plan.rows
                        );
                    }
                    self.end_verify(session, plain, seqs, blocks, out.logits)
                }
                Err(e) => {
                    self.abort_verify(seqs);
                    Err(e)
                }
            };
        }
        let mut plain_out = Vec::with_capacity(plain.len());
        if !plain.is_empty() {
            let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
            let dinputs: Vec<Tensor> = plain
                .iter()
                .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), &Device::Cpu))
                .collect::<Result<_>>()?;
            let step = self.forward_wave(
                session,
                &dseqs,
                &dinputs,
                &[],
                &[],
                &[],
                &[],
                0,
                layer_end,
                None,
            )?;
            plain_out = step.logits_owned()?;
            if plain_out.len() != plain.len() {
                candle::bail!(
                    "verify_blocks: plain wave scored {} rows for {} seqs",
                    plain_out.len(),
                    plain.len()
                );
            }
            for &(seq, _) in plain {
                session.advance_sequence(seq, 1)?;
            }
        }
        let mut out = Vec::with_capacity(seqs.len());
        for (i, &seq) in seqs.iter().enumerate() {
            out.push(self.verify_block(session, seq, &blocks[i], layer_end)?);
        }
        Ok((plain_out, out))
    }

    /// One lossless speculative-decode step for `seq` (model-agnostic). `committed` is the last
    /// accepted token, held OUT of the KV; it is placed at the current sequence offset. Drafts a
    /// block, verifies `[committed, drafts…]`, accepts the longest prefix whose proposals agree
    /// with what `chooser` draws for this model, and **emits each accepted token to `emit`, one at
    /// a time** — the model's own continuation, in order. This is what keeps speculative decode a
    /// *transparent accelerator*: the caller's main loop runs its normal per-token handling (stop
    /// sequences, EOS, sampling/steering decisions) on each token exactly as for plain decode,
    /// instead of the driver re-implementing any of it. `emit` returns `false` to stop generating
    /// after that token; the driver rolls the KV back to the emitted prefix. Returns
    /// `Some(next_committed)` — the seed to pass as `committed` on the next call (already emitted,
    /// held out of the KV) — or `None` when `emit` asked to stop. With no drafter this is one plain
    /// decode (emits exactly one token). At least one token is always emitted.
    // The argument list is the step's inputs: the session, the sequence, its committed
    // seed, the sampling policy, and the emit sink. Grouping them into a struct would
    // put a shape between the caller and the step it is driving.
    #[allow(clippy::too_many_arguments)]
    fn speculative_decode_step(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        committed: u32,
        max_draft: usize,
        layer_end: usize,
        chooser: &mut dyn TokenChooser,
        emit: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Option<u32>> {
        // The batch-of-1 case of the batched driver — one implementation.
        let mut emits: Vec<Box<dyn FnMut(u32) -> bool + '_>> = vec![Box::new(emit)];
        let next = self.speculative_decode_step_batch(
            session,
            &[seq],
            &[committed],
            max_draft,
            layer_end,
            chooser,
            &mut emits,
        )?;
        Ok(next[0])
    }

    /// One lossless speculative-decode step for MANY sequences — semantics identical to running
    /// [`Self::speculative_decode_step`] once per sequence, with the expensive parts batched:
    /// every block is verified in ONE `verify_blocks` call (a single wave when the model overrides
    /// it), and every scored row of the step is stacked once so `chooser` pays one dispatch per
    /// block position rather than one per row (per-row `to_scalar` round-trips are a
    /// launch-overhead wall at batch width). Each sequence keeps its own emit sink and truncates
    /// to its own accepted prefix. Returns each sequence's next `committed` seed (`None` where its
    /// emit stopped).
    ///
    /// `chooser` decides what each scored row commits, which is what makes the step lossless
    /// under sampling as well as under greedy decode — see [`speculative_choice`] for why a
    /// greedy drafter reduces the textbook accept/reject rule to "sample the row, accept the
    /// proposal iff the sample agrees". Pass [`GreedyChooser`] for bit-identical greedy output.
    // As [`Self::speculative_decode_step`], one slice per per-sequence input.
    #[allow(clippy::too_many_arguments)]
    fn speculative_decode_step_batch(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_draft: usize,
        layer_end: usize,
        chooser: &mut dyn TokenChooser,
        emits: &mut [Box<dyn FnMut(u32) -> bool + '_>],
    ) -> Result<Vec<Option<u32>>> {
        if seqs.len() != committed.len() || seqs.len() != emits.len() {
            candle::bail!(
                "speculative_decode_step_batch: {} seqs, {} committed, {} emits",
                seqs.len(),
                committed.len(),
                emits.len()
            );
        }
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        // **Refused up front for a model that cannot rewind.**
        //
        // Speculative decode is built on "decode the whole draft block, then put
        // the sequence back to the accepted prefix". The check belongs here, at
        // the one entry point, and not inside the rewind: by the time the driver
        // has drafted and verified, refusing is an error raised against work
        // already done, and *not* refusing is a state that has absorbed rejected
        // tokens the K/V no longer holds.
        //
        // The question is whether the model can REWIND, not whether it carries
        // recurrent state — the hybrid does both. Its verify replay re-runs the
        // mixer over the accepted prefix from the entering state the wave's
        // ping-pong left intact, which reconstructs exactly the state those
        // tokens alone would have produced (`qwen35::spec`,
        // `docs/deltanet_state_persistence.md` §5.4). Which offsets are
        // rewindable is a property of the stashed block, so `truncate_sequences`
        // is what refuses one it has no rewind point for.
        if !self.can_rewind_speculative_block() {
            candle::bail!(
                "speculative decode is unavailable on this model: accepting k of n \
                 drafted tokens requires rewinding the sequence, and this model \
                 carries recurrent state with no per-token decomposition to rewind \
                 to. A model whose `truncate_sequences` can restore the recurrence \
                 — as the qwen35 lineage's replay does — declares \
                 `can_rewind_speculative_block`; without that, run it without a \
                 drafter."
            );
        }
        // Draft per sequence. A sequence whose drafter proposes NOTHING (no
        // drafter, or the acceptance fallback holding it back) takes a PLAIN
        // decode wave instead of a 1-token verify: the verify path's
        // snapshot/rollback + slot-rebuild machinery costs measurably more
        // than a decode wave, which is exactly the loss the fallback exists
        // to avoid. Drafted sequences still verify together in one wave.
        let t_draft = std::time::Instant::now();
        let mut poss = Vec::with_capacity(seqs.len());
        for &seq in seqs {
            poss.push(session.sequence_offset(seq).ok_or_else(|| {
                candle::Error::msg("speculative_decode_step_batch: unknown sequence")
            })?);
        }
        // ONE call for the whole cohort: the drafter batches its own passes so
        // the weights it reads are read once for the step, not once per session.
        let drafts = self.speculative_draft(session, seqs, committed, max_draft)?;
        if drafts.len() != seqs.len() {
            candle::bail!(
                "speculative_decode_step_batch: drafter returned {} proposal lists for {} sequences",
                drafts.len(),
                seqs.len()
            );
        }
        let blocks: Vec<Vec<u32>> = drafts
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut block = Vec::with_capacity(d.len() + 1);
                block.push(committed[i]);
                block.extend_from_slice(d);
                block
            })
            .collect();
        pipeline_record_duration("spec:draft", t_draft.elapsed(), 1);
        let plain: Vec<usize> = (0..seqs.len()).filter(|&i| blocks[i].len() == 1).collect();
        let spec: Vec<usize> = (0..seqs.len()).filter(|&i| blocks[i].len() > 1).collect();

        // ONE forward for both cohorts: every undrafted sequence's committed
        // token decodes as an ordinary plain row (live slot, on-device
        // write-len commit — the sequence advances inside, and the later
        // uniform truncate is a no-op for it since kept is always 1) and every
        // drafted block verifies as virtual rows, in the same wave when the
        // model overrides `verify_blocks` (one launch floor per step, not
        // two).
        let t_verify = std::time::Instant::now();
        let plain_pairs: Vec<(usize, u32)> =
            plain.iter().map(|&i| (seqs[i], committed[i])).collect();
        let spec_seqs: Vec<usize> = spec.iter().map(|&i| seqs[i]).collect();
        let spec_blocks: Vec<Vec<u32>> = spec.iter().map(|&i| blocks[i].clone()).collect();
        let (mut plain_rows, spec_logits) = self.verify_blocks(
            session,
            &plain_pairs,
            &spec_seqs,
            &spec_blocks,
            layer_end,
            max_draft,
        )?;
        pipeline_record_duration("spec:verify", t_verify.elapsed(), 1);
        let mut plain_logits: Vec<Option<Tensor>> = vec![None; seqs.len()];
        for &i in plain.iter().rev() {
            plain_logits[i] = plain_rows.pop();
        }

        // Every scored row of BOTH waves, stacked once: plain rows lead (in
        // `plain` order), then each verify block's rows. `row_of[i]` locates
        // sequence `i`'s run inside it, so the accept walk lifts one block
        // position across the whole cohort with a single `index_select` rather
        // than slicing per sequence.
        let rows: Vec<Tensor> = plain
            .iter()
            .map(|&i| plain_logits[i].as_ref().expect("filled above").squeeze(0))
            .chain(spec_logits.iter().flatten().map(|t| t.squeeze(0)))
            .collect::<Result<_>>()?;
        let stacked = Tensor::stack(&rows, 0)?; // [R, vocab]
        let mut row_of: Vec<(usize, usize)> = vec![(0, 0); seqs.len()];
        for (k, &i) in plain.iter().enumerate() {
            row_of[i] = (k, 1);
        }
        let mut cur = plain.len();
        for (k, &i) in spec.iter().enumerate() {
            let n = spec_logits[k].len();
            row_of[i] = (cur, n);
            cur += n;
        }
        // A block scores exactly one row per token and the walk indexes rows by
        // block position, so a verify that returned a different count would read
        // some other sequence's row instead of failing.
        for (i, &(_, n_rows)) in row_of.iter().enumerate() {
            if n_rows != blocks[i].len() {
                candle::bail!(
                    "speculative_decode_step_batch: sequence {} verified {n_rows} rows for a \
                     {}-token block",
                    seqs[i],
                    blocks[i].len()
                );
            }
        }

        // Position-major accept, then one batched rollback over every target.
        //
        // Positions run in order and a sequence leaves `alive` as soon as it
        // commits a token that ends its block — either the model's own token
        // diverged from the proposal (that token IS the correction) or the block
        // ran out (its last row is a free bonus token). The cohort therefore
        // narrows as the walk goes, and the step costs at most `max_draft + 1`
        // chooser dispatches over the sequences still standing.
        //
        // Walking positions rather than deciding one stacked argmax over
        // everything is what lets a sampling chooser be exact: row `j` is scored
        // under the draft prefix that reaches it, so a chooser carrying
        // repetition penalties or a grammar stencil advances its per-sequence
        // state along exactly the path this loop commits.
        let t_accept = std::time::Instant::now();
        let mut walk = AcceptWalk::new(&blocks);
        while !walk.finished() {
            let rows = walk.rows();
            // One `index_select` lifts this position's rows across the cohort
            // out of the stack, rather than a slice per sequence.
            let idx = Tensor::from_vec(
                walk.alive()
                    .iter()
                    .map(|&i| (row_of[i].0 + walk.position()) as u32)
                    .collect::<Vec<_>>(),
                walk.alive().len(),
                stacked.device(),
            )?;
            let tokens = chooser.choose(&stacked.index_select(&idx, 0)?, &rows)?;
            walk.commit(&tokens, |i, token| (emits[i])(token))?;
        }
        let (next, kept) = walk.finish();
        let targets: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| (seq, poss[i] + kept[i]))
            .collect();
        // ONE rollback call for the step: every sequence's target at once, so a
        // model whose rollback does real work (the recurrent replay) batches it
        // across the cohort instead of paying it per sequence.
        self.truncate_sequences(session, &targets)?;
        pipeline_record_duration("spec:accept", t_accept.elapsed(), 1);
        Ok(next)
    }

    /// Create a batched inference session configured for this model.
    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let mut config = config;
        let props = self.model_core_properties();
        config.k_hi_error_threshold_factor *= props.k_hi_error_threshold_factor;
        config.k_low_error_threshold_factor *= props.k_low_error_threshold_factor;
        config.v_hi_error_threshold_factor *= props.v_hi_error_threshold_factor;
        config.v_low_error_threshold_factor *= props.v_low_error_threshold_factor;
        let session = BatchedInferenceSession::new(
            props.num_layers,
            props.n_kv_heads,
            props.head_dim,
            self.device(),
            config,
        )?;
        // Materialise the norm weights for this session's activation dtype, here
        // rather than at each call site. A session is where the dtype is decided,
        // it is created outside any wave, and the forward *refuses* a mismatch —
        // so leaving the call to callers would mean every one of them has to
        // remember, and the one that forgets fails at its first forward instead
        // of at the line that was wrong.
        self.maybe_change_dtype(session.activation_dtype())?;
        Ok(session)
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
        let backings = source.backings().to_vec();
        let session = BatchedInferenceSession::new_with_backings(backings, config, source.device());
        // The other way a session comes into being, and it decides an activation
        // dtype just as `create_batched_session` does — so it materialises the
        // norm weights the same way.
        self.maybe_change_dtype(session.activation_dtype())?;
        Ok(session)
    }

    /// Prunes excess memory usage.
    fn prune(&self) -> Result<()>;

    /// Snapshot expert pipeline telemetry counters (if the model has an expert cache).
    fn expert_stats(&self) -> Option<PipelineStats> {
        None
    }

    /// Buy `regions` of weight-side ground for the KV side, answering with the
    /// bytes conceded. See `BatchedModel::request_kv_ground` — this is the path a
    /// stalled scheduler uses to break a wave that cannot allocate.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// Live VRAM held by the model's weights (fixed base + time-varying resident
    /// experts), for the whole-card VRAM decomposition. `None` if unavailable.
    fn resident_weight_bytes(&self) -> Option<usize> {
        None
    }

    /// Reset expert pipeline telemetry counters to zero.
    fn reset_expert_stats(&self) {}

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }
}

/// Compute-side ceiling on the tokens one prefill forward carries.
///
/// This is a *throughput* limit, not a memory one: beyond roughly this width the
/// prefill kernels are compute-bound, so a wider forward costs the same per token
/// and slicing above it costs only the per-slab fixed wave overhead. A multiple
/// of 32, for kernel utilisation. (16384 was tried against the [1,4,8,16,1]
/// sweep and measured as a no-op there — that path's waves are shaped by the
/// scheduler's turn admission, not this slicer — so the value stays at the
/// established compute-saturation point until the pure-prefill ingest path is
/// measured wider.)
///
/// **It reserves nothing.** The transient tier is
/// `WAVE_ATTN_BYTES + WAVE_FFN_BYTES + MIGRATION_STAGING_CAP_BYTES`, carved once
/// at first use whatever this value is. Raising it does not cost VRAM; it permits
/// a wider wave, and whether that wave *fits* is the separate question
/// [`ManagedBatchedModel::prefill_width_cap`] asks the wave plan (the KV-admission
/// cap and the activation-pool cushion still bound the real wave). The narrower
/// of the two wins, so this can lead the span rather than having to trail it.
pub(crate) const MAX_PREFILL_TOKENS: usize = 8192;

/// The width cap plus 25% slack — the ceiling a single prefill slab may
/// actually reach. The asymmetry is measured: a wave's FIXED cost (~2.37 s —
/// the full per-layer routing readback + expert sweep) is paid per slab
/// regardless of width, while the compute-saturation cap is soft — tokens 25%
/// past it cost the same ~0.87 ms each as the ones before. So a 128-token
/// straggler slab after an 8192 slab spends a whole fixed sweep on 1.5% of the
/// tokens (~25% extra wall), where absorbing it into one 8320-token wave costs
/// per-token rate only.
pub(crate) fn prefill_slack_cap(width_cap: usize) -> usize {
    width_cap + width_cap / 4
}

/// Pack pure-prefill sequences into token-bounded slabs, returned as
/// `start..end` index ranges over `lens`.
///
/// Greedy whole-sequence packing against `width_cap`, with the
/// [`prefill_slack_cap`] tail rule: at the point a slab would close, if
/// EVERYTHING still unpacked fits within the slack ceiling it is absorbed into
/// this final slab instead of becoming one or more remainder waves. Only the
/// final slab may overshoot, so ordinary slabs still respect the cap; a single
/// sequence longer than the cap always travels alone and uncut (this packer
/// never splits inside a sequence).
pub(crate) fn pack_prefill_slabs(lens: &[usize], width_cap: usize) -> Vec<(usize, usize)> {
    let slack_cap = prefill_slack_cap(width_cap);
    let mut slabs = Vec::new();
    let mut start = 0usize;
    while start < lens.len() {
        let mut toks = 0usize;
        let mut end = start;
        while end < lens.len() {
            let l = lens[end];
            if end > start && toks + l > width_cap {
                let tail: usize = lens[end..].iter().sum();
                if toks + tail <= slack_cap {
                    end = lens.len();
                }
                break;
            }
            toks += l;
            end += 1;
        }
        slabs.push((start, end));
        start = end;
    }
    slabs
}

/// Blanket implementation of `ManagedBatchedModel` for `BatchedInference<M>`.
///
/// This allows models using the new `BatchedModelCore` + `BatchedInference` pattern
/// to work with `BatchedInferenceSession` without implementing `BatchedModel`.
impl<M: BatchedModelCore> ManagedBatchedModel for BatchedInference<M> {
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        self.model().wave_geometry(act_dtype)
    }

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        self.model().maybe_change_dtype(dtype)
    }

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
            // Uniform transformer: every layer attends, so every layer has a Q.
            provenance_capture_layers: ManagedBatchedModel::num_layers(self),
            can_gap_fill: !self.carries_recurrent_state(),
            carries_recurrent_state: self.carries_recurrent_state(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual_in: Option<Tensor>,
    ) -> Result<WaveResult> {
        drive_wave(
            self,
            session,
            decode_seqs,
            decode_inputs,
            prefill_seqs,
            prefill_inputs,
            glue_seqs,
            glue_inputs,
            layer_start,
            layer_end,
            residual_in,
        )
    }
    fn prune(&self) -> Result<()> {
        self.model().prune()
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        self.model().expert_stats()
    }

    fn request_kv_ground(&self, regions: usize) -> u64 {
        self.model().request_kv_ground(regions)
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        self.model().resident_weight_bytes()
    }

    fn reset_expert_stats(&self) {
        self.model().reset_expert_stats()
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.model().snapshot_profiles()
    }
}

/// A uniform transformer's half of a wave: the same layer body at every index.
///
/// Everything around it — group assembly, the 1-token reroute, the token
/// permutation, KV rollback, the decode advance — is [`drive_wave`], shared with
/// every other model that runs waves.
impl<M: BatchedModelCore> WaveSweep for BatchedInference<M> {
    fn device(&self) -> &Device {
        self.model().device()
    }

    fn num_layers(&self) -> usize {
        self.model().num_layers()
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    fn sweep(
        &self,
        contexts: &mut [SequenceContext],
        groups: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        self.forward_wave_contexts(
            contexts,
            groups.n_decode,
            groups.n_prefill,
            groups.decode_headers,
            groups.prefill_headers,
            groups.glue_headers,
            groups.generation,
            groups.layer_start,
            groups.layer_end,
            groups.x_in,
        )
    }
}

#[cfg(test)]
mod slab_tests {
    use super::{pack_prefill_slabs, prefill_slack_cap};

    #[test]
    fn straggler_is_absorbed_within_slack() {
        // 8192 + 128 = 8320 ≤ 10240 slack: ONE slab, no straggler wave.
        assert_eq!(pack_prefill_slabs(&[8192, 128], 8192), vec![(0, 2)]);
        // Right at the slack ceiling: still one slab.
        assert_eq!(pack_prefill_slabs(&[8192, 2048], 8192), vec![(0, 2)]);
        // One past the ceiling: split.
        assert_eq!(
            pack_prefill_slabs(&[8192, 2049], 8192),
            vec![(0, 1), (1, 2)]
        );
    }

    #[test]
    fn mid_fleet_slabs_respect_the_bare_cap() {
        // Only the FINAL slab may overshoot: slab 1 closes at the cap because
        // absorbing the remaining 8500 would blow past the slack ceiling; the
        // tail then packs together (500 + 8000 = 8500 ≤ 10240 absorbs at ITS
        // closing point).
        assert_eq!(
            pack_prefill_slabs(&[8000, 500, 8000], 8192),
            vec![(0, 1), (1, 3)]
        );
    }

    #[test]
    fn tail_larger_than_slack_splits_normally() {
        // 10 × 1030: greedy packs 7 (7210); the 3-seq tail (3090) would land at
        // 10300 > 10240, so no absorb — but the tail then fits one slab alone.
        let lens = [1030usize; 10];
        assert_eq!(pack_prefill_slabs(&lens, 8192), vec![(0, 7), (7, 10)]);
    }

    #[test]
    fn oversize_sequences_travel_alone_and_uncut() {
        // A single sequence past even the slack cap is never split.
        assert_eq!(pack_prefill_slabs(&[30000], 8192), vec![(0, 1)]);
        assert_eq!(
            pack_prefill_slabs(&[30000, 100, 30000], 8192),
            vec![(0, 1), (1, 2), (2, 3)]
        );
    }

    #[test]
    fn slack_is_a_quarter_of_the_cap() {
        assert_eq!(prefill_slack_cap(8192), 10240);
        assert_eq!(prefill_slack_cap(100), 125);
    }
}
