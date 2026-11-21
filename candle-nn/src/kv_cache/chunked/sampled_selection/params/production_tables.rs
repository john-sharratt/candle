//! The shared production tables that every model's adaptive selection starts
//! from: the candidate formats each compression level may choose among, and the
//! q-relevance thresholds that decide between them.
//!
//! Each level's candidate list is ordered as a priority hint for the GPU
//! launcher, which stable-sorts by bits per element before it launches. A top
//! rung needs a high-quality fallback in its list, because threshold tuning
//! cannot buy a format that is not offered. A format that is expensive to decode
//! can still be load-bearing: removing `Q0_V` from the top levels leaves the
//! compression ratio unchanged but breaks the model gates, so changes to the
//! lists have to be judged on the model gates and not on a synthetic benchmark.
//! The thresholds are relative errors against each head's dynamic range.

use super::*;

pub const SELECT_BLOCK: usize = 32;

pub const ERROR_MARGIN_ABS: f32 = 0.001;

/// Denominator floor for normalized reconstruction error metrics.
pub const ERROR_NORM_EPS: f32 = 1.0e-8;

/// Default arena span used by the report harness when building paged selector metadata.
pub const DEFAULT_REPORT_ARENA_CHUNKS: usize = 8192;

/// Smaller arena span used by the calibration sweep over the sampled subset.
pub const DEFAULT_CALIBRATION_ARENA_CHUNKS: usize = 2048;

/// **Q0_V is load-bearing at C6, C8, C9 and C10. Do not remove it — that was
/// tried, and it costs output validation on three models.**
///
/// It is by far the most expensive format here to decode, which is what invites
/// the removal. The whole 32-element block is a two-byte header naming a curve,
/// a scale and a centroid in `__constant__` tables, and constant memory
/// broadcasts only when a warp reads ONE address; every lane in the decode
/// kernel is on a different dim, so a different block, so a different curve row,
/// and each of the three lookups per element serialises across the warp.
/// Measured on the decode A/B harness at head_dim 256, 8 slots, 2048 tokens
/// (`decode_ab bench --formats c10`): **332.9 µs against 45.3 µs for BF16 —
/// 7.35×**, where every other candidate at this level sits between 1.26× and
/// 1.71×.
///
/// **That cost is real and it is still not a reason to drop the format.**
/// Removing it from these four levels leaves the compression ratio identical —
/// 7.10× on Qwen3.5-35B, unchanged — because those blocks fall back to Q0_X,
/// which is also two bytes. What changes is the reconstruction: Q0_X carries no
/// curve, so a block Q0_V fitted well can fit it badly. Measured on the
/// batched-forward gates, same build, this list the only difference:
///
/// | | Q0_V removed | Q0_V present |
/// |---|---|---|
/// | Qwen3.5-35B-A3B, C10 x32 / x64 | 31/32, 63/64 sessions | **pass** |
/// | Qwen3.8-Flash-Next, C10 x8 | 7/8 sessions | **pass** |
/// | Qwen3.6-35B-A3B, C10 x64 | 60/64 sessions | 61/64, its own standing level |
///
/// Reproducible — the failures were identical across two runs each way.
///
/// One trap worth naming, since it is what made the removal look safe. A
/// uniform-format bench cannot say whether a format matters to an *adaptive*
/// cache, and `decode_ab`'s fixture is **synthetic** K/V whose statistics need
/// not match a model's. On that fixture an adaptive C10 cache measured 73.6 µs
/// with Q0_V against 73.2 µs without — no difference, which reads as "the
/// selector never picks it". On real KV it plainly does. Any future case for
/// changing this list has to come from the model gates, not the harness.
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
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C4 — K stays at 4 bits or better through C5: keys decide what
        // attention reads, so the strong rungs spend their bits there and push
        // V instead. 3-bit K starts at C6. Q5_0/Q5_1 (5.5/6.0 bpe) sit in the
        // gap between Q4_1 (5.0) and Q8_0 (8.5) on C3–C7, so a block that
        // misses 4 bits takes one more bit, not three and a half.
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C5
        &[
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
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
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
        ],
        // C7 — the last K rung with a Q8 ceiling, so the last where Q5 buys
        // compression; above it Q5 would be a new ceiling and cost it.
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
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
        // C10 — Q4_0/Q4_1 restored 2026-09-01. C10 topped out at Q3_1 while C9
        // carried both, so the top rung had no high-quality fallback: a block
        // the thresholds said needed better than Q3_1 got Q3_1 anyway, because
        // that was the whole list. That is why tuning the error factors did
        // nothing across six settings — the selector was already at the top of
        // its candidates and the threshold had nothing left to buy. A ceiling is
        // not an aggressiveness knob.
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
            QuantFormat::Q8_0,
        ],
        // C3
        &[
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q5_0,
            QuantFormat::Q8_0,
        ],
        // C4 — Q5_0 fills the Q4_0 → Q8_0 gap on C3–C6, as on K. Of the
        // asymmetric (QX_1) formats V keeps only Q3_1: at 3 bits a block's own
        // offset costs a level; from 4 bits up the symmetric formats lose
        // nothing, and Q4_1/Q5_1/Q8_1 are never selected on this side.
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q5_0,
            QuantFormat::Q8_0,
        ],
        // C5
        &[
            QuantFormat::Q1_S,
            QuantFormat::Q2_A,
            QuantFormat::Q2_S,
            QuantFormat::Q3_0,
            QuantFormat::Q3_1,
            QuantFormat::Q4_0,
            QuantFormat::Q5_0,
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
            QuantFormat::Q5_0,
            QuantFormat::Q8_0,
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
        ],
        // C10 — Q4_0 is the ceiling: a block the thresholds say needs better
        // than Q3_1 takes Q4_0, so the top rung keeps a high-quality fallback.
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
    0.006654, // C2  (2026-10-01 monotone ladder, iteration 2)
    0.006324, // C3  (2026-10-01 monotone ladder, iteration 2)
    0.004000, // C4  (2026-10-01 Q5 K buy-back, iteration 6)
    0.006300, // C5  (2026-10-01 Q5 K buy-back, iteration 6)
    0.012200, // C6  (2026-10-01 Q5 K buy-back, iteration 7)
    0.014000, // C7  (2026-10-01 Q5 K buy-back, iteration 8)
    0.016500, // C8  (2026-10-01 Q5 K buy-back, iteration 9)
    0.017700, // C9  (2026-10-01 Q5 K buy-back, iteration 10)
    0.028884, // C10 (re-derived 2026-08-16, see the C10 note on the LOW table)
];

#[rustfmt::skip]
pub const PRODUCTION_K_QREL_LOW_THRESHOLDS: [f32; 11] = [
    0.011315, // C0  (provisional)
    0.051794, // C1  (provisional) — must be > K_HIGH[C1]
    0.043430, // C2  (2026-10-01 monotone ladder, iteration 2)
    0.041170, // C3  (2026-10-01 monotone ladder, iteration 2)
    0.028500, // C4  (2026-10-01 Q5 K buy-back, iteration 6)
    0.056000, // C5  (2026-10-01 Q5 K buy-back, iteration 6)
    0.101000, // C6  (2026-10-01 Q5 K buy-back, iteration 7)
    0.133000, // C7  (2026-10-01 Q5 K buy-back, iteration 8)
    0.268000, // C8  (2026-10-01 Q5 K buy-back, iteration 9)
    0.233000, // C9  (2026-10-01 Q5 K buy-back, iteration 10)
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
    0.021300, // C2 (2026-10-01 Q5 V buy-back, iteration 2)
    0.029800, // C3 (2026-10-01 Q5 V buy-back, iteration 2)
    0.029800, // C4 (2026-10-01 Q5 V buy-back, iteration 2)
    0.031500, // C5 (2026-10-01 Q5 V buy-back, iteration 2)
    0.034000, // C6 (2026-10-01 Q5 V buy-back, iteration 2)
    0.034000, // C7 (2026-10-01 Q5 V buy-back, iteration 2)
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
    0.038000, // C2 (2026-10-01 Q5 V buy-back, iteration 2)
    0.085000, // C3 (2026-10-01 Q5 V buy-back, iteration 2)
    0.102000, // C4 (2026-10-01 Q5 V buy-back, iteration 2)
    0.119000, // C5 (2026-10-01 Q5 V buy-back, iteration 2)
    0.145000, // C6 (2026-10-01 Q5 V buy-back, iteration 2)
    0.187000, // C7 (2026-10-01 Q5 V buy-back, iteration 2)
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
