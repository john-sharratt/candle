//! Holds the settings a batched session is created from: storage formats, the
//! adaptive compression level, threshold multipliers and the activation dtype.
//!
//! A `None` compression level means uniform storage with no per-block selection,
//! and `Some(level)` engages the adaptive selection kernel. The K and V overrides
//! force one side to a uniform format and leave the other side adaptive, and both
//! default to `None` so that selection's own state reaches storage. The
//! activation dtype is the width a model's reference implementation computes in.
//! When it is unset the width is derived from the K format, which is right only
//! while K is stored as float.

use super::*;

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

    /// The width activations flow between layers in, when the model's own
    /// reference implementation names one.
    ///
    /// `None` derives it from [`Self::k_format`], which is where it came from
    /// historically — see [`BatchedInferenceSession::activation_dtype`] for why
    /// that derivation is only right when K is stored float, and why a model
    /// that knows its own compute width must be able to say so instead.
    pub activation_dtype: Option<DType>,
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
            activation_dtype: None,
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
        self.compression_level
            .and_then(|level| self.compression_policy_at(level))
    }

    /// The same policy at an explicit level, for a caller that seals one layer
    /// differently from the rest. `None` when the session is uncompressed —
    /// a level cannot conjure compression the mode does not have.
    pub fn compression_policy_at(&self, level: u8) -> Option<CompressionPolicy> {
        self.compression_level.map(|_| level).map(|level| {
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
