use crate::kv_cache::{KvFormat, QuantFormat};

pub const SELECT_BLOCK: usize = 32;
pub const ERROR_MARGIN_ABS: f32 = 0.001;
/// Denominator floor for normalized reconstruction error metrics.
pub const ERROR_NORM_EPS: f32 = 1.0e-8;

/// Default arena span used by the report harness when building paged selector metadata.
pub const DEFAULT_REPORT_ARENA_CHUNKS: usize = 8192;

/// Smaller arena span used by the calibration sweep over the sampled subset.
pub const DEFAULT_CALIBRATION_ARENA_CHUNKS: usize = 2048;

/// Shared production candidate ladders for adaptive runtime selection.
const FULL_K_CANDIDATE_LADDER: &[QuantFormat] = &[
    QuantFormat::Q0_V,
    QuantFormat::Q1_S,
    QuantFormat::Q2_A,
    QuantFormat::Q2_S,
    QuantFormat::Q3_0,
    QuantFormat::Q3_1,
    QuantFormat::Q4_0,
    QuantFormat::Q4_1,
    QuantFormat::Q4_KS,
    QuantFormat::Q8_0,
    QuantFormat::Q8_1,
    QuantFormat::Q8_KS,
];

const FULL_V_CANDIDATE_LADDER: &[QuantFormat] = &[
    QuantFormat::Q0_V,
    QuantFormat::Q1_S,
    QuantFormat::Q2_A,
    QuantFormat::Q2_S,
    QuantFormat::Q3_0,
    QuantFormat::Q3_1,
    QuantFormat::Q4_0,
    QuantFormat::Q4_1,
    QuantFormat::Q8_0,
];

#[rustfmt::skip]
pub const PRODUCTION_K_CANDIDATE_FORMATS: [&[QuantFormat]; 11] =
    [
        // C0
        &[
            QuantFormat::Q8_KS,
        ],
        // C1
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_KS,
        ],
        // C2
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C3
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C4
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C5
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
        ],
        // C6 (midpoint between old C6 and old C7 — union of formats)
        &[
            QuantFormat::Q0_V,
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C7
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C8 (copied from C7)
        &[
            QuantFormat::Q0_V,
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
        ],
        // C9
        &[
            QuantFormat::Q0,
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q1_A,
            QuantFormat::Q1_S,
            QuantFormat::Q0_M4,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
        ],
        // C10
        &[
            QuantFormat::Q0,
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q1_A,
            QuantFormat::Q1_S,
            QuantFormat::Q0_M4,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
        ],
    ];

#[rustfmt::skip]
pub const PRODUCTION_V_CANDIDATE_FORMATS: [&[QuantFormat]; 11] =
    [
        // C0
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q8_0,
        ],
        // C1
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q8_0,
        ],
        // C2
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
        ],
        // C3
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
        ],
        // C4
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
        ],
        // C5
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
        ],
        // C6
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C7
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
        ],
        // C8 (copied from C7)
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
        ],
        // C9
        &[
            QuantFormat::Q0,
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q1_A,
            QuantFormat::Q1_S,
            QuantFormat::Q0_M4,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
        ],
        // C10
        &[
            QuantFormat::Q0,
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q1_A,
            QuantFormat::Q1_S,
            QuantFormat::Q0_M4,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
        ],
    ];

pub fn production_adaptive_candidates(level: u8) -> (Vec<KvFormat>, Vec<KvFormat>) {
    let idx = level.min(10) as usize;
    // Order is preserved here — the GPU launcher
    // (`select_kv_format_palette4_paged_batched_raw_from_device_ptrs`) does a
    // stable sort by ascending BPE before the kernel launch, so the order
    // we return is treated as a *priority hint* within each BPE tier
    // (equal-BPE formats keep their relative order from this list).
    (
        PRODUCTION_K_CANDIDATE_FORMATS[idx]
            .iter()
            .copied()
            .map(KvFormat::Quantized)
            .collect(),
        PRODUCTION_V_CANDIDATE_FORMATS[idx]
            .iter()
            .copied()
            .map(KvFormat::Quantized)
            .collect(),
    )
}

/// Shared production q-relevance threshold tables used by runtime selection and offline analysis.
/// Values are dimensionless relative errors: fraction of per-head absolute max.
/// e.g. 0.010 = 1.0% of the head's dynamic range.
/// ⚠ All values are provisional starting points — re-derive after switching to
/// the real-quant-roundtrip selection kernel (see docs/real-quant-roundtrip-selection.md).
#[rustfmt::skip]
pub const PRODUCTION_K_QREL_HIGH_THRESHOLDS: [f32; 11] = [
    0.003096, // C0  (provisional — needs re-derivation)
    0.004725, // C1  (provisional) — must be < K_LOW[C1]
    0.008944, // C2  (provisional)
    0.014703, // C3  (provisional)
    0.018199, // C4  (provisional)
    0.020700, // C5  (provisional)
    0.020758, // C6  (provisional — midpoint between old C6 and old C7)
    0.021735, // C7  (provisional)
    0.018771, // C8  (provisional)
    0.025236, // C9  (provisional)
    0.031321, // C10 (provisional)
];

#[rustfmt::skip]
pub const PRODUCTION_K_QREL_LOW_THRESHOLDS: [f32; 11] = [
    0.011315, // C0  (provisional)
    0.051794, // C1  (provisional) — must be > K_HIGH[C1]
    0.072130, // C2  (provisional)
    0.102114, // C3  (provisional)
    0.136622, // C4  (provisional)
    0.216643, // C5  (provisional)
    0.232942, // C6  (provisional — midpoint between old C6 and old C7)
    0.248296, // C7  (provisional)
    0.284827, // C8  (provisional)
    0.274433, // C9  (provisional)
    0.453389, // C10 (provisional)
];

/// V high (strict) q-relevance error thresholds passed to the CUDA selection kernel.
/// Must be ≤ corresponding LOW value so the kernel's hi/lo scaling stays sane.
#[rustfmt::skip]
pub const PRODUCTION_V_QREL_HIGH_THRESHOLDS: [f32; 11] = [
    0.012232, // C0
    0.018664, // C1
    0.015596, // C2
    0.019366, // C3
    0.022474, // C4
    0.023001, // C5
    0.023768, // C6 (midpoint between old C6 and old C7)
    0.024000, // C7
    0.022167, // C8
    0.023852, // C9
    0.024824, // C10
];

/// V low (lenient) q-relevance error thresholds passed to the CUDA selection kernel.
/// See `PRODUCTION_V_QREL_HIGH_THRESHOLDS`.
#[rustfmt::skip]
pub const PRODUCTION_V_QREL_LOW_THRESHOLDS: [f32; 11] = [
    0.012730, // C0
    0.025898, // C1
    0.022541, // C2
    0.030230, // C3
    0.050119, // C4
    0.153698, // C5
    0.170920, // C6 (midpoint between old C6 and old C7)
    0.187766, // C7
    0.215390, // C8
    0.250035, // C9
    0.653093, // C10
];

/// Mirror of the CUDA `k_threshold_scaled` device function.
///
/// Applies IQR-standardised exponential scaling to the K threshold.
/// When `q_spread` ≤ 1e-8 (degenerate distribution), falls back to the
/// geometric mean `sqrt(threshold_lo * threshold_hi)`.
#[inline]
pub fn k_threshold_scaled_rust(
    threshold_lo: f32,
    threshold_hi: f32,
    q_relevance: f32,
    q_median: f32,
    q_spread: f32,
) -> f32 {
    if q_spread <= 1.0e-8 {
        return (threshold_lo * threshold_hi).sqrt();
    }
    let z = (q_relevance - q_median) / q_spread;
    let multiplier = (-z).exp();
    let base = (threshold_lo * threshold_hi).sqrt();
    let scaled = base * multiplier;
    scaled.clamp(threshold_hi, threshold_lo)
}

pub const PRODUCTION_LEVEL_TIER: [&str; 11] = [
    "quality ", "quality ", "sweet   ", "sweet   ", "sweet   ", "compress", "compress", "compress",
    "compress", "compress", "compress",
];

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

/// Tuned for Qwen3-30B-A3B (MoE).
pub const QWEN3_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.475,
    k_low: 1.200,
    v_hi: 1.225,
    v_low: 2.700,
};

/// Tuned for Qwen3-8B.
pub const QWEN3_8B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.900,
    k_low: 1.450,
    v_hi: 0.900,
    v_low: 2.600,
};

/// Llama 3.x family. Currently identity but kept as a named constant so the
/// production trait override and the offline report stay aligned when it diverges.
pub const LLAMA_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors::IDENTITY;
