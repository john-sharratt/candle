use crate::kv_cache::{KvFormat, QuantFormat};

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
///
/// Re-derived 2026-09-20: the factored RoPE table build (`rope_schedule::table`)
/// removed a bogus 1.25x linear-scaling factor that `infer_rope_scaling_factor`
/// had been reading out of this model's `context_length: 40960` and feeding into
/// the paged kernels via `new_with_inv_freq`. `v_hi`/`v_low` were calibrated
/// against that miscalculated RoPE, so once corrected they no longer covered one
/// sweep session (`quantized_qwen3::tests::test_parallel_batched_forwarding`,
/// C9/C10). `k_hi`/`k_low` are unaffected — `v_hi: 0.900 -> 0.899`,
/// `v_low: 2.600 -> 2.598` is the smallest step off the committed row (found by
/// bisecting between the committed value and a much larger, unnecessary cut) that
/// passes the gate twice in a row; C9 moves 5.47x -> 5.46x, C10 is unchanged at
/// 5.84x.
pub const QWEN3_8B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.900,
    k_low: 1.450,
    v_hi: 0.899,
    v_low: 2.598,
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
/// **Re-derived 2026-08-25** on the 0.8B C-ladder gate
/// (`quantized_qwen35::tests::test_parallel_batched_forwarding_0_8b`) to the
/// lineage's calibration target: **the whole range C0–C10 passes**, with the
/// C10×10 rung sitting just under the breaking edge. Green twice with identical
/// ratios (C8 3.88×, C9 4.15×, C10 4.68×).
///
/// **The row it replaces was fit to a different numeric path.** The 2026-08-23
/// row (k 0.85, v 0.60) was derived while the gate pinned the *unquantized* BF16
/// conversion, which forced `Int8Mode::Off` and left this the only model in the
/// lineage on the FP matmul path. The gate now pins Q8_0 and takes `auto`, so it
/// runs int8 like its siblings — and like a deployment. Only V needed moving:
/// 0.60 fails C10 by one session, 0.55 and 0.45 both pass, and the whole usable
/// band is 4.64×–4.68× against the failing row's 4.71%. K stays at 0.85.
///
/// Sweep facts for the next re-derivation:
/// * **V is the lever at the top rung, K is not** — C8 passes at k 0.85
///   throughout, and C10 moves on V alone. (On the old FP path the opposite held,
///   which is a warning that these facts belong to a numeric path, not a model.)
/// * V barely moves compression here — 0.10 of factor is ~0.9% of ratio — so
///   margin is nearly free. Prefer a value that passes repeatably over the
///   largest one that passes once; a rung sitting *on* the edge flips with any
///   numerical change, and this ladder is statistical, not deterministic.
/// * The critical blocks respond to the geometric mean of an axis's hi·lo pair,
///   not to either factor alone.
///
/// **Re-derived 2026-09-01, v 0.55 → 0.45.** C10×10 had gone red by one session
/// in ten with nothing in this file altered — the same drift the 9B row records
/// from 2026-08-28, and the reason the note above says to prefer a value that
/// passes repeatably over the largest one that passes once. 0.45 is the other
/// value that sweep measured passing, so this steps to a bracketed point rather
/// than to a fresh guess, and V costs ~0.9% of ratio per 0.10 of factor.
pub const QWEN35_0_8B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.85,
    k_low: 0.85,
    v_hi: 0.45,
    v_low: 0.45,
};

/// Qwen3.5-9B (dense hybrid).
///
/// **Derived 2026-08-23** on the 9B C-ladder gate
/// (`quantized_qwen35::tests::test_parallel_batched_forwarding_9b`) to the
/// lineage target: C0–C10 all pass, C10×10 just under the breaking edge.
/// The 9B has real headroom over the 0.8B (its ladder passes at identity
/// with room to spare), so the row loosens to sell that headroom for
/// compression. A hi-tight/low-loose redistribution at the same geometric
/// means measured strictly worse than the symmetric split.
///
/// **Re-derived 2026-09-10, k 1.09 → 1.07 and v 1.9 → 1.85**, after the wave
/// admission and driver rework. Same cause as the re-derivation before it: wave
/// widths move, accumulation order moves with them, and the top rung — which
/// this row deliberately parks one notch under its break — goes red. C10×10 had
/// fallen to 9/10 sessions with nothing else altered. Both axes were stepped
/// together and the first step passed, so the new edges are **not** bracketed;
/// the pair below is a passing point, not a measured boundary.
///
/// Cost of the step: C10×10 6.30× → 6.19×, about 1.7% of ratio.
///
/// The 2026-08-28 measurements, for whoever brackets this next — they were taken
/// on the previous wave geometry, so treat them as the shape of the surface
/// rather than as live numbers:
///
/// * **K edge 1.09 ✓ / 1.1 ✗** at v 1.9 — a margin of one hundredth.
/// * **V edge 1.9 ✓ / 2.0 ✗** at k 1.075. V is not the inert axis an earlier
///   note claimed: it broke the top rung one notch above its value, so probe
///   both axes here rather than K alone.
///
/// **This row will drift again.** Twice now it has gone red on a change that
/// touched neither quantization nor this model — only the width of a wave. A
/// row tuned to sit one notch under the break cannot survive that, by
/// construction; if the re-derivations become tiresome, the fix is to buy
/// standing margin rather than to keep re-finding the edge.
///
/// Merge, 2026-09-12: this branch had 1.07/1.85; main re-derived to 0.95/1.65
/// after restoring Q4_0/Q4_1 to the C10 candidates. Main's is the LOWER pair and
/// is kept — a tighter error bound costs ratio, never quality — and its evidence
/// below is the fuller of the two.
///
/// C10×10 at 6.30×, identical across two confirmation runs.
/// **Re-derived 2026-09-01, k 1.09 → 0.95 and v 1.9 → 1.65, after restoring
/// Q4_0/Q4_1 to the C10 candidate lists.** Green twice at C10 5.49×.
///
/// **The candidates were the blocker, not this row.** C10 topped out at Q3_1 on
/// both axes while C9 carried Q4_0/Q4_1, so the top rung had no high-quality
/// fallback. Six threshold settings were walked first and the failure never
/// moved a character:
///
/// | k | v | C10 | result |
/// |---|---|---|---|
/// | 1.09 | 1.90 | 6.30× | session 9 diverges at char 23 |
/// | 1.08 | 1.90 | 6.28× | ” |
/// | 1.07 | 1.85 | 6.21× | ” |
/// | 1.09 | 1.80 | 6.18× | ” |
/// | 1.05 | 1.90 | 6.23× | ” |
/// | 1.00 | 1.75 | 5.99× | ” |
///
/// That is the signature of a **ceiling, not an aggressiveness knob**: the
/// selector was already at the top of its list, so tightening the error bound
/// bought nothing — it asked for a better format and none existed. The ratio
/// moved (blocks were being reassigned) while the divergence sat still.
///
/// With Q4 restored, the same lever works immediately:
///
/// | k | v | C10 | result |
/// |---|---|---|---|
/// | 1.09 | 1.90 | 5.97× | fails |
/// | 1.02 | 1.77 | 5.72× | fails |
/// | **0.95** | **1.65** | **5.49×** | **passes, twice** |
/// | 0.85 | 1.45 | 5.15× | passes |
/// | 0.60 | 1.00 | 4.43× | passes |
///
/// Settled at 0.95/1.65 rather than creeping to the 1.02/1.77 edge. This row has
/// now broken twice by sitting one hundredth from its break (1.1 → 1.09 in
/// August, then red again in September); a margin of 0.07 on K and 0.12 on V is
/// the point of the exercise, not a rounding of it.
///
/// What it cost: 6.30× → 5.49×, ~13%. Worth stating plainly — but the 6.30× row
/// did not pass, so the comparison is against a gate that was red, not against
/// working compression.
pub const QWEN35_9B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    // Merge, 2026-09-12: the table above is main's, measured on main. This
    // branch's row was 1.07/1.85, and the merge first took main's 0.95/1.65 on
    // the reasoning that the lower pair is the tighter error bound and so the
    // safe one. **Measured here, that is backwards:**
    //
    // | k    | v    | C10 ×10 | ratio |
    // |------|------|---------|-------|
    // | 0.95 | 1.65 | 9/10 ✗  | 5.48× |
    // | 1.07 | 1.85 | 10/10 ✓ | 5.87× |
    //
    // Both runs deterministic, the 0.95 row twice. So this branch's pair passes
    // AND compresses better, and the "lower cannot cost quality" reasoning is
    // false — the ladder is not monotone in these factors, exactly as main's own
    // MoE row records ((1.1, 2.35) passes while (1.15, 2.35) and (1.1, 2.43)
    // both fail).
    //
    // Main's table is not wrong; it is a measurement of main. A calibration row
    // belongs to the code it was derived against, and picking between two
    // branches' rows by their VALUES rather than by re-measuring is how a merge
    // ships a combination neither side ever ran.
    //
    // Re-derived 2026-09-14 on the RTX 3090 (sm_86, gate at int8 `prec`):
    // 1.07/1.85 — 10/10 on the sm_120 box two days earlier — scored 9/10 here
    // three runs straight (session 9 diverges at char 23, the single-name
    // signature), at C10 5.86×. So a row also belongs to the CARD it was
    // measured on: the int8 numeric path differs per arch, and the edge moves
    // with it. Stepped straight to main's deep-margin pair rather than walking
    // the edge (this row's own note: buy standing margin):
    //
    // | k    | v    | C10 ×10 (sm_86) | ratio |
    // |------|------|-----------------|-------|
    // | 1.07 | 1.85 | 9/10 ✗ ×3       | 5.86× |
    // | 0.85 | 1.45 | 10/10 ✓ ×3      | 5.13× |
    //
    // Cost: 5.86× → 5.13× (the 5.86× row was red, so against working
    // compression the baseline is the sm_120 box's 5.87×). The sm_120 box has
    // NOT run 0.85/1.45 on this branch — its standing 10/10 is 1.07/1.85
    // (2026-09-12): confirm there on its next sweep, and if this pair fails
    // there, the row has outgrown a single constant (the 27B's checkpoint is
    // already chosen per card; a row may have to be).
    //
    // Confirmed 2026-09-15 on the sm_120 box (RTX PRO 5000, int8 `prec`):
    // 0.85/1.45 validated every C row, C10 ×10 included, in two sweeps, at C10
    // 5.13× against 1.07/1.85's 5.87× there. One constant holds on both cards;
    // the price on this one is 13 % of the top rung's compression.
    k_hi: 0.85,
    k_low: 0.85,
    v_hi: 1.45,
    v_low: 1.45,
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
///
/// **Retuned 2026-08-26 for the widened top rung: K 1.5 → 1.2, V 2.0 → 2.5.**
/// The two moves are opposite in direction and that is the point — the axes
/// turned out to do different jobs here.
///
/// The gate's C10 rows moved from a single ×10 to ×8 and ×16, and the caution
/// above came true as written: at the old row the ×8 rung held (7.31x) while
/// ×16 lost one session of sixteen, diverging 35 characters in. A wider cohort
/// reassociates the batched reductions, so a selection marginally inside the
/// edge at ten sessions sits marginally outside it at sixteen.
///
/// **K is what moves that session; V is inert to it.** Stepping V 2.0 → 1.8
/// cost ratio (7.31x → 6.99x) and failed the *same session at the same
/// character* — the signature this file already records for the 3.6 row. So the
/// "3.5 is V-limited" fact above was derived at ten sessions and does not
/// survive the width: **the edge axis is a property of the cohort as well as of
/// the checkpoint.** At sixteen this model behaves like its point release.
/// K 1.2 clears it.
///
/// Being inert also makes V free to spend, which is where the ratio came back
/// and then some. Bracketed against the widened rung:
///
/// | V   | C10×8 | C10×16 | ratio |
/// |-----|-------|--------|-------|
/// | 2.0 | pass  | pass   | 6.79x |
/// | 2.5 | pass  | pass   | **7.57x** |
/// | 2.8 | pass  | 15/16  | 8.06x |
/// | 3.0 | 15/16 | 9/16   | 8.42x |
///
/// 2.5 rather than 2.7-ish: 2.8 already loses the wide rung and 3.0 loses both,
/// so 2.5 sits under the edge with room rather than on it — the distinction the
/// 2.3 → 2.0 note above was written about. Net against the row this replaces,
/// **7.31x → 7.57x while gaining the ×16 rung it used to fail.**
///
/// # This row is coupled to the draft ladder, in another crate
///
/// The rungs it was derived on — C10 at ×8 and ×16 in `quantized_qwen35_moe` —
/// run under `DraftBudget::Adaptive`, and both speculate only because
/// `candle_transformers::models::draft_ladder`'s bracket reaches 16. Pull that
/// bracket in and the ×16 rung silently reverts to plain decode, which changes
/// the batched reduction order and moves the marginal C10 session. The symptom
/// is a red KV gate caused by a speculation constant, with neither file naming
/// the other — so re-verify this row after a ladder change, exactly as after an
/// admission or width change.
///
/// **Re-derived 2026-09-10, K 1.1 → 1.08 and V 2.35 → 2.30**, after the wave
/// admission and driver rework — the width/accumulation drift this row has now
/// been caught by four times. C10×64 had gone red at 63/64 sessions. Both axes
/// were stepped together and the first step passed, so these are a passing
/// point, not bracketed edges. Cost: C10 7.10× → 6.98×, about 1.7% of ratio,
/// now 7.02/6.97/6.98/6.98× (×8/16/32/64).
///
/// **Four re-derivations, none of them provoked by a quantization or model
/// change.** Every one followed a change to wave width. A row parked one notch
/// under its break cannot survive that; buying standing margin would end the
/// cycle more cheaply than continuing to re-find the edge.
///
/// The 2026-08-28 measurements, taken on the previous wave geometry — the shape
/// of the surface rather than live numbers:
///
/// * **K edge 1.1 ✓ / 1.15 ✗** at v 2.35.
/// * **V edge 2.35 ✓ / 2.43 ✗** at k 1.1.
///
/// **The "K moves it, V is inert" note above is now false**, and the way it
/// failed is worth keeping. Walking V alone 2.5 → 2.2 did not merely fail to
/// help — it made the gate *worse*, losing ×32 as well as ×64 while costing
/// half a turn of ratio (7.55× → 7.07×). Walking K alone 1.2 → 1.1 → 1.0 never
/// cleared ×64 either, at 0.47× of ratio. Only a joint move clears it, and the
/// axes are not separable near the edge: the pair (1.1, 2.35) passes while both
/// (1.15, 2.35) and (1.1, 2.43) fail.
///
/// The lesson generalises past this row — a single-axis sweep here can read as
/// "inert" when the truth is that the other axis had to move too, and stopping
/// at that reading leaves ratio on the table.
///
/// C10 at 7.13/7.08/7.09/7.10× (×8/16/32/64), identical across two
/// confirmation runs.
/// **Re-derived 2026-09-01, k 1.1 → 1.17 and v 2.35 → 2.47**, recovering the
/// ratio that restoring Q4_0/Q4_1 to the C10 candidates cost. Those candidates
/// are shared, so this model paid for the 9B's fix without needing it: C10 fell
/// 7.10× → 6.74× at the old factors with nothing else changed. Better formats it
/// never asked for are, from this row's point of view, simply headroom.
///
/// Edge bracketed: **1.17/2.47 ✓ 7.05×**, 1.25/2.60 ✗ 7.41× (session 2). Settled
/// on the passing side — 0.08 on K and 0.13 on V — rather than creeping toward
/// the break for the last 0.05× (see the 9B row for what edge-sitting costs).
pub const QWEN35_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    // **Re-derived 2026-09-12 for the origin/main merge, k 1.08 → 1.00 and
    // v 2.30 → 2.10.** This is the re-verification the caution above asks for
    // after an admission change, and the merge is the largest one this row has
    // seen: main restored Q4_0/Q4_1 to the shared C10 candidates.
    //
    // | k    | v    | C10 ×64 | ratio |
    // |------|------|---------|-------|
    // | 1.08 | 2.30 | 63/64 ✗ | 6.63× |  (this branch's pre-merge row)
    // | 1.17 | 2.47 | 62/64 ✗ | 7.05× |  (main's, re-derived post-Q4)
    // | 1.08 | 2.10 | 63/64 ✗ | 6.36× |  V alone — inert, as the note below says
    // | 1.00 | 2.10 | 64/64 ✓ | 6.21× |
    //
    // Note the second row: main's pair reproduces main's reported 7.05× exactly,
    // so the candidate list and the selection ARE behaving as main's do — the
    // factors simply do not transfer, because the two branches' numerics differ.
    // The third row is this model's documented V-limitation asserting itself the
    // other way: only the JOINT move cleared ×64, exactly as the bracket above
    // records. ×8/×16/×32 were green throughout, so a narrower rung would have
    // called this fixed three probes early.
    //
    // **A lower factor is NOT automatically safe**, which is what this comment
    // claimed when the merge first resolved it. The 9B row falsified that in the
    // same session: main's *lower* 0.95/1.65 failed 9/10 where this branch's
    // higher 1.07/1.85 passed 10/10 at a better ratio. The ladder is not
    // monotone in these factors and "lower is the tighter bound, so it cannot
    // cost quality" is not a substitute for running the gate.
    k_hi: 1.00,
    k_low: 1.00,
    v_hi: 2.10,
    v_low: 2.10,
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
/// K step (1.4 fails). K at 1.2 cleared it *then*; it no longer does, see the
/// 2026-08-28 entry below. Probe K first on this model; a V sweep here
/// measures nothing but lost compression.
///
/// **Re-derived 2026-08-28, K 1.2 → 1.15**, after the dense weights moved into
/// the device reservation. C10×64 had gone red; V stayed at 2.0.
///
/// Both axes walked separately, both edges bracketed:
///
/// * **K edge 1.15 ✓ / 1.2 ✗** at v 2.0.
/// * **V edge 2.0 ✓ / 2.1 ✗** at k 1.15 — so V is *not* inert here either, it
///   simply had no room left to give above its current value.
///
/// The "probe K first on this model" advice above still holds, and this time it
/// was enough on its own — unlike the 3.5, which needed both axes together.
///
/// **Re-derived 2026-09-03, K 1.15 → 1.05.** C10 had gone red at ×32 *and* ×64
/// (31/32 and 61/64 sessions) while ×8 and ×16 stayed green — the row had drifted
/// since 2026-08-28, which is exactly the re-verification the caution above asks
/// for. Nothing in the compression path had changed; the drift is the wave-width
/// and admission movement that caution names.
///
/// K probed alone again, and again it was enough — but it took two notches, not
/// one, and the two rungs cleared at different points:
///
/// * **K 1.15** — ×32 ✗ (31/32), ×64 ✗ (61/64).
/// * **K 1.10** — ×32 ✓, ×64 ✗ (63/64). The wide rung is the last to go.
/// * **K 1.05** — ×8/16/32/64 all ✓, twice.
///
/// V held at 2.0 throughout and was not swept: the 2026-08-28 bracket put its
/// edge at 2.0 ✓ / 2.1 ✗, so it had no room to give and a sweep would only have
/// cost ratio, per this row's standing advice.
///
/// Worth noting for the next re-derivation: **×32 recovering does not mean the
/// row is clear.** At K 1.10 three of four rungs passed and only ×64 was left,
/// one session short — a stopping point that looks like success from every rung
/// but the widest.
///
/// C10 at 6.50/6.47/6.46/6.47× (×8/16/32/64), identical across two confirmation
/// runs — 6.65× → 6.47× at ×64, the ratio this costs.
pub const QWEN36_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    // **Re-derived 2026-09-12 for the origin/main merge, k 1.05 → 1.00**, V held
    // at 2.0. Main's row (1.32/2.29) was not taken: factors do not transfer
    // between branches whose numerics differ — see the measured table in
    // `QWEN35_MOE_KV_FACTORS` above, where main's pair reproduced main's ratio
    // exactly and still failed here.
    //
    // | k    | v   | C10 ×64 | ratio |
    // |------|-----|---------|-------|
    // | 1.05 | 2.0 | 63/64 ✗ | 6.47× |  (green on four runs BEFORE the merge)
    // | 1.00 | 2.0 | 64/64 ✓ | 6.04× |
    //
    // One K notch, and K alone, exactly as this row's standing advice says to
    // probe it — the note from the 09-01 derivation ("3.6 is K-limited… the next
    // derivation probes K first there") paid for itself immediately. ×8/×16/×32
    // were green at both factors; only ×64 moved.
    //
    // That the pre-merge pair was four-runs-green and is now one session short
    // is this row drifting on an admission change, which is precisely the
    // failure mode its caution predicts — not a defect elsewhere.
    //
    // **Re-derived 2026-09-16, k 1.00 → 0.95**, V held at 2.0, on the 72 GB
    // RTX PRO 5000 Blackwell. The row also serves the AntiLoop + StyleTune
    // hybrid, and the hybrid on `Int8Mode::Performance` sat one session past
    // the edge: C10×64 at 63/64 on three identical runs, the same session
    // diverging at the same character each time. That was the first
    // measurement of that width on that path; the only earlier run of the gate
    // was on a 24 GB card, where ×32/×64 do not run. K alone, one notch, per
    // this row's standing advice:
    //
    // | gate                 | k 1.00 C10 ×64 | k 0.95 C10 ×64 | ratio ×64     |
    // |----------------------|----------------|----------------|---------------|
    // | stock, Performance   | 64/64 ✓        | 64/64 ✓        | 6.04× → 5.94× |
    // | hybrid, Precision    | 64/64 ✓        | 64/64 ✓        | 6.08× → 5.99× |
    // | hybrid, Performance  | 63/64 ✗ (×3)   | 64/64 ✓ (×2)   | 6.07× → 5.97× |
    //
    // Every other rung and width is green at both factors. The cost is about
    // 1.6% of ratio at ×64 on each gate.
    k_hi: 0.95,
    k_low: 0.95,
    v_hi: 2.0,
    v_low: 2.0,
};

/// Qwen3.8-27B (dense flagship hybrid).
///
/// The 27B is dense — no expert streaming to shrink the resident set — and its
/// Q4_K_M weighs 16.5 GB before KV, so it is build-only on the **16 GB** dev
/// card and measurable on anything larger. The gate is
/// `quantized_qwen38::tests::test_parallel_batched_forwarding_27b`, tuned to
/// the same target as the rest of the lineage: C0–C10 all pass with the top
/// rung just under the breaking edge.
///
/// **Derived 2026-08-28 — the first measured row this model has had.** Both
/// axes walked to failure on the C10 rung and bracketed:
///
/// * **K edge 1.3 ✓ / 1.4 ✗** at v 2.3.
/// * **V edge 2.3 ✓ / 2.4 ✗** at k 1.3.
///
/// C0–C10 all pass, C10×10 at 7.03×, identical across two confirmation runs.
///
/// # Two things had to be corrected before it could be measured at all
///
/// The row read "cannot be measured *here*", where *here* meant the 16 GB card
/// — a qualifier that does not travel with the file and reads on any larger
/// machine as a claim the row is underivable. It is not: the gate runs on a
/// 72 GB card. Anchoring a constant's provenance to the machine that happened
/// to write it is how a row stays extrapolated long after the reason expired.
///
/// The real blocker was the checkpoint. The pinned `UD-Q4_K_M` is an Unsloth
/// *Dynamic* quant — per-tensor type choice — and this model's recipe mixes in
/// **IQ4_XS**, gguf dtype 23, which this codebase does not implement. It
/// downloads 16.5 GB and then fails to load, on any machine. The gate now pins
/// single-type files (`quantized_qwen38::checkpoint_for_this_card`), chosen by
/// card size: Q6_K above 32 GB, Q4_K_M below.
///
/// # Weight precision is not a term in this calibration
///
/// The row was derived twice, on Q4_0 and on Q6_K — two bits of weight
/// precision apart. **The ladder moved 6.27× → 6.30× at identical thresholds**,
/// inside run-to-run noise, while bulk throughput fell 5.5% on the heavier
/// weights. So the KV thresholds are what bind the C10 rung, not the weight
/// quant, and the earlier worry that deriving above production quant would
/// yield a too-loose row does not materialise. Deriving on either is sound;
/// Q6_K is pinned because it matches the lineage's other **dense** gate (the
/// 9B), the MoE gates being the Q4_K_M ones.
///
/// # Re-derived on Q3_K_M, and weight precision IS a term after all
///
/// The section above concluded the opposite from Q4_0 against Q6_K, and it was
/// right about that pair. It does not extend to **Q3_K_M**, which is what a
/// 16 GB card runs: at the old row the gate's C10×10 scored **7/10** and C9 was
/// intermittent. The failure is not garbage — the model emits fluent text and
/// picks a *different character name* ("Marcus" where BF16 gives "Ian"), which
/// is what a threshold one step too loose looks like when the KV it reads is
/// still nearly right.
///
/// So the row is now derived per axis on the 16 GB card, Q3_K_M, against the
/// full C0–C10 gate. Nine runs; each pair is one whole gate:
///
/// | K | V | C9 | C10 | compress |
/// |---|---|----|-----|----------|
/// | 1.3 | 2.3 | ✓ | 7/10 | 6.91× |
/// | 1.0 | 2.3 | ✓ | 9/10 | 6.39× |
/// | 0.8 | 2.3 | ✓ | 9/10 | 6.03× |
/// | 1.3 | 1.8 | 4/5 | 9/10 | 6.27× |
/// | 1.0 | 1.8 | 4/5 | 9/10 | 5.84× |
/// | 0.85 | 1.8 | 4/5 | **✓** | 5.61× |
/// | 0.7 | 2.3 | ✓ | 9/10 | 5.86× |
/// | 0.7 | 2.0 | ✓ | 9/10 | 5.58× |
/// | **0.7** | **1.8** | **✓** | **✓** | **5.39×** |
/// | 0.7 | 1.4 | ✓ | ✓ | 5.04× |
///
/// **Both axes bind, and they bind on different rungs.** V's edge is C10: 1.8
/// passes and 2.0 fails at any K. K's edge is **C9**, not C10 — 0.85 clears
/// C10 outright and still loses C9, so tuning against the deepest rung alone
/// would have shipped a row that fails a shallower one. That is the whole
/// reason the two are walked separately.
///
/// `0.7 / 1.8` is the loosest pair that passes the entire gate, with 0.15 of
/// margin on K and 0.2 on V below their measured edges. Retighten here rather
/// than widening tolerances if the row ever goes red.
///
/// Ordering against the lineage has inverted with it: K 0.7 now sits *below*
/// the 9B's 1.09 and the MoE pair's 1.1/1.15. The 27B has the least headroom
/// here, not the most — at the quant a 16 GB card actually runs.
/// **Re-derived 2026-09-01, k 0.7 → 0.8 and v 1.8 → 2.05**, after the shared C10
/// candidate lists regained Q4_0/Q4_1 (see the 9B row). C10 5.12× → **5.50×**.
///
/// **The edge is NOT bracketed** — passed on the first step and was not walked
/// further, so the margin is unmeasured. Same caveat as the 3.6 row.
///
/// **Re-derived 2026-09-14 on the RTX 3090** (24 GB → the gate pins Q4_K_M;
/// sm_86, int8 `perf`) after 0.8/2.05 scored 9/10 three runs straight
/// (session 4, char 32, a single verb — "had"→"has"). Walked down this row's
/// own measured surface, and the divergence MOVED with the step (session 4 →
/// session 1, verb → name), so this was the knob and not a candidate ceiling:
///
/// | k   | v    | C10 ×10 (sm_86, Q4_K_M) | ratio |
/// |-----|------|--------------------------|-------|
/// | 0.8 | 2.05 | 9/10 ✗ ×3                | 5.52× |
/// | 0.7 | 1.8  | 9/10 ✗                   | —     |
/// | 0.7 | 1.4  | 10/10 ✓ ×3               | 4.78× |
///
/// "Weight precision is a term" (above) now has a third point: Q6_K holds
/// 0.8/2.05, Q3_K_M held 0.7/1.8, and Q4_K_M on sm_86 needs 0.7/1.4. The
/// sm_120 box's standing 10/10 at 0.8/2.05 (2026-09-13, Q6_K) has NOT been
/// re-run at this pair: confirm there on its next sweep — a fail there means
/// the row needs the same per-card resolution the checkpoint already has.
///
/// **Confirmed 2026-09-15 on the sm_120 box** (RTX PRO 5000, Q6_K, int8
/// `perf`): 0.7/1.4 validated every C row, C10 ×40 included, in two sweeps, at
/// C10 4.81× against 0.8/2.05's 5.56× there. One constant holds on both cards;
/// the price on this one is 13 % of the top rung's compression.
pub const QWEN38_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.7,
    k_low: 0.7,
    v_hi: 1.4,
    v_low: 1.4,
};

/// Qwen3.8-Flash-Next (`qwen4exp`) — the 512-expert sparse-attention hybrid.
///
/// **Derived 2026-08-31** on
/// `quantized_qwen38_moe::tests::test_parallel_batched_forwarding`, whose
/// ladder gained C0–C10 rungs for this row. Two properties of this model
/// shape the derivation and are worth stating before the numbers:
///
/// * **Only 12 of 48 layers hold K/V at all.** The other 36 mix through a
///   recurrence, so a compression level here touches a quarter of the stack —
///   the per-layer error a level admits is diluted across three layers that
///   cannot propagate it, which is not true of any dense row above.
/// * **QSA caps the read at 2051 cells.** Past that depth a query reads a
///   *selected* subset, so a block's quantization error reaches the output
///   only when the indexer selects that block. Compression error and selection
///   interact, and this row is derived at gate depth (~713 tokens), where
///   selection is the identity and every cell is read. **A row derived below
///   the budget is a bound, not a measurement, for behaviour above it** — the
///   deep-context rung is recorded as outstanding rather than assumed.
///
/// Walked from the lineage's neighbour (the 3.6 row, k 1.15 / v 2.0), which
/// passed the whole ladder on the first run at 5.54× — so the row was loose
/// and the derivation is a search *outward* for the edge, not inward from a
/// failure. Six runs of the C-ladder gate:
///
/// | k | v | C10×2 | C10×8 | ratio |
/// |---|---|-------|-------|-------|
/// | 1.15 | 2.0 | pass | pass | 5.54× |
/// | 1.5  | 3.0 | pass | pass | 6.95× |
/// | 1.65 | 3.0 | pass | pass | **7.15×** |
/// | 1.8  | 3.0 | pass | pass | 7.36× |
/// | 1.65 | 3.3 | **fail** | **fail** | 7.56× |
/// | 1.8  | 3.6 | **fail** | **fail** | 8.13× |
/// | 2.2  | 4.5 | **fail** | **fail** | 10.14× |
///
/// * **V edge 3.0 ✓ / 3.3 ✗** at k 1.65 — V is the binding axis here.
/// * **K 1.8 ✓** at v 3.0. K's own edge above 1.8 is not bracketed: the next
///   probe that failed (1.8 / 3.6) moved V as well, so what is known is that
///   1.8 passes and that V is what gives way first.
///
/// # Why the row sits one notch under the C10 edge rather than further back
///
/// **C10 is the calibration probe; C5 is the operating point.** zend runs
/// `compression_level(5)` (`session.rs`, `repo_scan`), so the margin that has
/// to be comfortable is C5's, and C10 is run only because a row tuned at the
/// level production uses cannot tell you how much edge is left. Held to the
/// tighter of the two readings, this row would be paying real ratio at C5 to
/// buy headroom at a level nothing runs.
///
/// The evidence that C5 is nowhere near an edge here: **C8 still passed at
/// 6.87× under thresholds loose enough to break C10 entirely** (k 2.2 /
/// v 4.5). Everything at or below C8 has a wide margin at this row; only the
/// top rung is close to anything.
///
/// Confirmed across two ladders at k 1.8 / v 3.0: the first ran
/// C0/C4/C8/C10×2/C10×8, the second swapped the middle rung to **C5 — the
/// level zend runs**. C0, C8 and both C10 widths each passed twice.
///
/// # The row this model did NOT need, and why it is worth recording
///
/// When the Gated Residual fused, C10×2 went red and this row was retightened
/// to k 1.65 / v 2.8 — costing ratio at **C5, the level production runs**, to
/// fix a marginal session at C10, a rung that exists only as a probe. That was
/// the wrong currency, and the wrong diagnosis.
///
/// The cause was not reassociation. The first fused kernel rolled its own
/// `1/(1 + __expf(-x))` where the eager path calls `fast_exp::sigmoid` — a
/// cubic polynomial with ~0.009% error. The kernel was therefore ~400× MORE
/// accurate than the reference it replaced, a ~9e-5 relative change on every
/// gate value, which is orders of magnitude above any last-ulp effect. Fixing
/// the kernel to use the shared primitive put the arithmetic back and this row
/// held at k 1.8 / v 3.0 unchanged.
///
/// **The rule this leaves behind:** when a KV rung moves after a kernel
/// change, establish whether the arithmetic actually changed before spending
/// compression on it — and a fusion that improves precision is an arithmetic
/// change, not a neutral one. A calibration derived against a reference is
/// only valid for code that computes what the reference computes.
///
/// Both C10 widths fail *together* at every failing point measured, so this
/// model gives no warning rung between pass and fail: a red C10 here is the
/// first signal, not the second.
///
/// Ratio lands in line with the lineage (3.5 at 7.13×, 3.6 at 6.8×) despite
/// only a quarter of the stack holding K/V at all.
pub const QWEN4EXP_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.8,
    k_low: 1.8,
    v_hi: 3.0,
    v_low: 3.0,
};
