//! The per-model calibration of adaptive KV quantization: the multipliers each
//! model applies to the shared thresholds, and the tables they start from.
//!
//! `KvErrorThresholdFactors` holds one model's multipliers over the shared
//! `PRODUCTION_*_THRESHOLDS`. The runtime selector and the offline tuning report
//! both read the same named constants, so they cannot drift apart. The rows for
//! each family of models live beside the shared tables they scale.
//!
//! - `dense_factors`: the rows for the dense models.
//! - `moe_factors`: the rows for the MoE models.
//! - `production_tables`: the shared candidate lists and thresholds.

use crate::kv_cache::{KvFormat, QuantFormat};

/// Per-model multipliers applied on top of the shared `PRODUCTION_*_THRESHOLDS`
/// before they reach the runtime selector or the offline tuning report.
///
/// Single source of truth: production model `BatchedModelCore` overrides AND the
/// `test_candidate_list_compression_curve` projection report read these. Update
/// the named per-model constants below and both paths track each other.
#[derive(Debug, Clone, Copy)]
pub struct KvErrorThresholdFactors {
    pub k_hi: f32,
    pub k_low: f32,
    pub v_hi: f32,
    pub v_low: f32,
}

impl KvErrorThresholdFactors {
    pub const IDENTITY: Self = Self {
        k_hi: 1.0,
        k_low: 1.0,
        v_hi: 1.0,
        v_low: 1.0,
    };
}

mod dense_factors;
mod moe_factors;
mod production_tables;

pub use dense_factors::*;
pub use moe_factors::*;
pub use production_tables::*;
