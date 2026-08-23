//! Compression policy for adaptive per-block format selection.
//!
//! `CompressionPolicy` captures the candidate formats and aggressiveness level
//! for the adaptive quantization path. Its presence is the signal that
//! adaptive selection should run; its absence (i.e. `Option<CompressionPolicy>::None`)
//! means uniform storage with no per-block selection. There is no separate
//! "adaptive" boolean — the policy itself *is* the toggle.

pub use super::sampled_selection::params::{
    production_adaptive_candidates, KvErrorThresholdFactors, LLAMA2_KV_FACTOR, LLAMA3_KV_FACTOR,
    LLAMA_KV_FACTORS, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
    PRODUCTION_LEVEL_TIER, PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
    QWEN35_0_8B_KV_FACTORS, QWEN35_9B_KV_FACTORS, QWEN35_MOE_KV_FACTORS, QWEN36_MOE_KV_FACTORS,
    QWEN38_KV_FACTORS, QWEN3_8B_KV_FACTORS, QWEN3_MOE_KV_FACTORS,
};
use crate::kv_cache::{KvFormat, QuantFormat};

/// Policy controlling adaptive per-block quantization format selection.
///
/// Carried by the conversation/session layer rather than by the arena storage,
/// because format-selection policy belongs to the inference pipeline, not the
/// memory allocator. When code wants "no compression", it passes `None` for
/// the policy rather than a "disabled" sentinel.
#[derive(Debug, Clone, Copy)]
pub struct CompressionPolicy {
    /// Compression level (0-10) for the shared adaptive production profile.
    pub compression_level: u8,
    /// Per-model multiplier for the K high (strict) adaptive threshold.
    pub k_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the K low (lenient) adaptive threshold.
    pub k_low_error_threshold_factor: f32,
    /// Per-model multiplier for the V high (strict) adaptive threshold.
    pub v_hi_error_threshold_factor: f32,
    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    pub v_low_error_threshold_factor: f32,
    /// When set, force K storage to the given uniform quant format with
    /// identity pal_map and unit (1.0) outer scales — bypassing the
    /// selection kernel's K-side choices entirely. V is unaffected (still
    /// fully adaptive per selection).
    ///
    /// This is the path that exercises decode's known-correct K-side
    /// invariants: identity layout + unit scales + a single uniform format.
    /// Non-identity K pal_map combined with non-unit K outer scales is a
    /// decode-kernel bug surface; the override sidesteps it while still
    /// preserving full V adaptivity.
    ///
    /// `None` propagates selection's full per-(chunk, head, palette) K
    /// state to storage — currently fails recall and is preserved as the
    /// default for diagnostic work.
    pub override_k_quant: Option<QuantFormat>,
    /// Symmetric V counterpart to [`Self::override_k_quant`]. Defaults to
    /// `None` because the V-side decode path correctly honors selection's
    /// per-(chunk, head, palette) pal_maps + scales; the override exists
    /// only for symmetry with K (e.g., diagnostic isolation, future use).
    pub override_v_quant: Option<QuantFormat>,
}

impl Default for CompressionPolicy {
    fn default() -> Self {
        Self {
            compression_level: 0,
            k_hi_error_threshold_factor: 1.0,
            k_low_error_threshold_factor: 1.0,
            v_hi_error_threshold_factor: 1.0,
            v_low_error_threshold_factor: 1.0,
            override_k_quant: None,
            override_v_quant: None,
        }
    }
}

impl CompressionPolicy {
    /// Create a compression policy at the given level using the shared
    /// production candidate profile and unit threshold factors.
    pub fn new(compression_level: u8) -> Self {
        Self::new_with_error_threshold_factors(compression_level, 1.0, 1.0, 1.0, 1.0)
    }

    /// Create a compression policy with per-model K and V hi/lo threshold factors.
    pub fn new_with_error_threshold_factors(
        compression_level: u8,
        k_hi_error_threshold_factor: f32,
        k_low_error_threshold_factor: f32,
        v_hi_error_threshold_factor: f32,
        v_low_error_threshold_factor: f32,
    ) -> Self {
        Self {
            compression_level: compression_level.min(10),
            k_hi_error_threshold_factor: k_hi_error_threshold_factor.max(0.0),
            k_low_error_threshold_factor: k_low_error_threshold_factor.max(0.0),
            v_hi_error_threshold_factor: v_hi_error_threshold_factor.max(0.0),
            v_low_error_threshold_factor: v_low_error_threshold_factor.max(0.0),
            override_k_quant: None,
            override_v_quant: None,
        }
    }

    /// Builder-style setter for the K quantization override.
    /// `Some(fmt)` forces K storage to uniform `fmt` with identity pal_map
    /// and unit outer scales; `None` lets selection's per-(chunk, head,
    /// palette) K choices propagate to storage.
    pub fn with_override_k_quant(mut self, fmt: Option<QuantFormat>) -> Self {
        self.override_k_quant = fmt;
        self
    }

    /// Builder-style setter for the V quantization override (symmetric to
    /// [`Self::with_override_k_quant`]). Defaults to `None` — V's decode
    /// path already handles selection's full adaptive state correctly.
    pub fn with_override_v_quant(mut self, fmt: Option<QuantFormat>) -> Self {
        self.override_v_quant = fmt;
        self
    }

    /// K cache candidate formats from the shared production profile.
    pub fn k_candidates(&self) -> Vec<KvFormat> {
        production_adaptive_candidates(self.compression_level).0
    }

    /// V cache candidate formats from the shared production profile.
    pub fn v_candidates(&self) -> Vec<KvFormat> {
        production_adaptive_candidates(self.compression_level).1
    }

    /// Shared production candidate lists for a given compression level.
    pub fn production_candidates(level: u8) -> (Vec<KvFormat>, Vec<KvFormat>) {
        production_adaptive_candidates(level)
    }
}
