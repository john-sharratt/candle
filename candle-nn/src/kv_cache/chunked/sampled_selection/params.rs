use crate::kv_cache::{KvFormat, QuantFormat};

pub const SELECT_BLOCK: usize = 32;
pub const ERROR_MARGIN_ABS: f32 = 0.001;
/// Denominator floor for normalized reconstruction error metrics.
pub const ERROR_NORM_EPS: f32 = 1.0e-8;

/// Default arena span used by the report harness when building paged selector metadata.
pub const DEFAULT_REPORT_ARENA_CHUNKS: usize = 8192;

/// Smaller arena span used by the calibration sweep over the sampled subset.
pub const DEFAULT_CALIBRATION_ARENA_CHUNKS: usize = 2048;

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
    0.028884, // C10 (re-derived 2026-08-16, see the C10 note on the LOW table)
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
    // C10: re-derived 2026-08-16 against the current unsloth Qwen3-8B-Q6_K
    // snapshot. The original C10 row (K_HIGH 0.031321, K_LOW 0.453389,
    // V_HIGH 0.024824, V_LOW 0.653093) was tuned 2026-05-05 against the
    // April upload of that file; unsloth replaced the GGUF on May 9 and
    // May 13, and against the replacement those values fail the gate's
    // C10×5 StoryRewrite config 1/5 (deterministically, both int8 modes).
    // Derivation: geometric interpolation from the C9 row (t=0) to the old
    // C10 row (t=1), hi and lo probed separately. Measured pass/fail edges:
    // hi in (0.75, 0.875], lo in (0.5, 0.625]. These values sit one 0.125
    // step inside each edge — hi at t=0.625, lo at t=0.375 — trading the
    // old 7.51x for 6.21x compression at 5/5 quality with margin on both
    // sides, so a small upstream drift does not put the row back on an edge.
    0.331283, // C10
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
    0.024455, // C10 (re-derived 2026-08-16, see the C10 note on the K LOW table)
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
    0.358398, // C10 (re-derived 2026-08-16, see the C10 note on the K LOW table)
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

/// Per-generation scalar on top of [`LLAMA_KV_FACTORS`]: one multiplier
/// applied to all four rows, carried by the model as its
/// `compression_error_factor`. Llama 2 runs the base row; Llama 3 runs 10%
/// tighter.
pub const LLAMA2_KV_FACTOR: f32 = 1.0;
/// See [`LLAMA2_KV_FACTOR`].
pub const LLAMA3_KV_FACTOR: f32 = 0.9;

/// Qwen3.5-0.8B (dense hybrid) — attention layers at `head_dim 256`.
///
/// **Derived 2026-08-23** on the 0.8B C-ladder gate
/// (`quantized_qwen35::tests::test_parallel_batched_forwarding_0_8b`) to the
/// lineage's calibration target: **the whole range C0–C10 passes**, with the
/// C10×10 rung sitting just under the breaking edge. Sweep facts that
/// remain true for the next re-derivation: the critical blocks respond to
/// the geometric mean of an axis's hi·lo pair, not to either factor alone
/// (single-sided probes barely move them), and V is the sensitive axis on
/// this model — K-only tightening made a second session diverge in the
/// wide-rung sweep.
pub const QWEN35_0_8B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.85,
    k_low: 0.85,
    v_hi: 0.6,
    v_low: 0.6,
};

/// Qwen3.5-9B (dense hybrid).
///
/// **Derived 2026-08-23** on the 9B C-ladder gate
/// (`quantized_qwen35::tests::test_parallel_batched_forwarding_9b`) to the
/// lineage target: C0–C10 all pass, C10×10 just under the breaking edge.
/// The 9B has real headroom over the 0.8B (its ladder passes at identity
/// with room to spare), so the row loosens to sell that headroom for
/// compression. Sweep facts that remain true: V alone saturates before it
/// breaks the top rungs — K is this model's edge axis — and a
/// hi-tight/low-loose redistribution at the same geometric means measured
/// strictly worse than the symmetric split.
pub const QWEN35_9B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.1,
    k_low: 1.1,
    v_hi: 1.9,
    v_low: 1.9,
};

/// Qwen3.5-35B-A3B (routed hybrid).
///
/// **Derived 2026-08-23** on the 35B C-ladder gate
/// (`quantized_qwen35_moe::tests::test_parallel_batched_forwarding_35b`) to
/// the lineage target: C0–C10 all pass, C10×10 just under the breaking
/// edge. The routed 35B is the most quantization-robust of the lineage
/// (its ladder passes at identity with the most headroom), so its row is
/// the loosest. Sweep fact that remains true: V is the fine-grained lever
/// at the top of the ladder — the highest rungs' V candidate floors
/// (Q0/Q1) are strictly worse than the rungs below (which keep Q4
/// fallbacks), so V-loosening moves C10 differentially while C9 holds.
///
/// **V retuned 2026-08-23, 2.3 → 2.0.** At 2.3 the C10×10 rung sat *on* the
/// edge rather than under it: one session's name token flipped in 11 of 11
/// runs at 7.93x, and the same rung passed 3 of 4 runs on the parent commit,
/// so the rung was marginal rather than broken. 2.0 holds it at 7.39x across
/// three alternating runs (six with the 3.6 gate, 6/6), with the C10 ratio
/// identical run to run — a stable selection rather than a coin flip.
pub const QWEN35_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.5,
    k_low: 1.5,
    v_hi: 2.0,
    v_low: 2.0,
};

/// Qwen3.6-35B-A3B (routed hybrid point release).
///
/// **Derived 2026-08-23** on the 3.6 C-ladder gate
/// (`quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b`)
/// to the lineage target: C0–C10 all pass, C10×10 just under the breaking
/// edge. Derivation caution that remains true: wave-width changes (e.g. the
/// VRAM-governor fix widening the spans) shift accumulation order and move
/// marginal edge sessions — re-verify this row after any admission or width
/// change.
///
/// **Retuned 2026-08-23, k 1.5 → 1.2 and v 2.2 → 2.0**, holding C10×10 at
/// 6.80x across three alternating runs (six with the 3.5 gate, 6/6).
///
/// The two rows are no longer identical, because the point release does not
/// share its base model's edge axis: **3.5 is V-limited, 3.6 is K-limited.**
/// 3.6's failing session is inert to V — 2.2, 2.0 and 1.9 all fail it at the
/// same session and character while costing ratio — and inert to a one-notch
/// K step (1.4 fails). K at 1.2 is what clears it. Probe K first on this
/// model; a V sweep here measures nothing but lost compression.
pub const QWEN36_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.2,
    k_low: 1.2,
    v_hi: 2.0,
    v_low: 2.0,
};

/// Qwen3.8-27B (dense flagship hybrid). **Extrapolated, not derived**: the
/// dense 27B is build-only on the 16 GB dev card, so this row cannot be
/// measured here. The measured lineage rows order by capacity — the 0.8B
/// tightest, the 9B looser, the 35B/3.6 MoE loosest — and the 27B sits
/// between the 9B and the MoE pair in capacity, so this row sits just above
/// the 9B's, deliberately conservative (thresholds err tight, quality errs
/// safe). The derivation gate
/// (`quantized_qwen38::tests::test_parallel_batched_forwarding_27b`)
/// replaces it with a measured row on the workstation, tuned to the same
/// target as the rest of the lineage: C0–C10 all pass with C10×10 just
/// under the breaking edge.
pub const QWEN38_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.5,
    k_low: 1.5,
    v_hi: 2.0,
    v_low: 2.0,
};
