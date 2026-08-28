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
pub const QWEN35_0_8B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.85,
    k_low: 0.85,
    v_hi: 0.55,
    v_low: 0.55,
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
/// **Re-derived 2026-08-28, k 1.1 → 1.09**, after the dense weights moved into
/// the device reservation. That change shifts wave widths and therefore
/// accumulation order, which is exactly the drift the rows below warn about:
/// C10×10 had gone red at k 1.1 with nothing else altered.
///
/// Each axis was walked separately and both edges are now bracketed, so a
/// future retune starts from measurements rather than a sweep:
///
/// * **K edge 1.09 ✓ / 1.1 ✗** at v 1.9. The margin is one hundredth — this row
///   sits as close to the break as the lineage target asks for, and is
///   correspondingly fragile.
/// * **V edge 1.9 ✓ / 2.0 ✗** at k 1.075. V is no longer the inert axis the
///   earlier note claimed: it breaks the top rung one notch above its current
///   value, so probe both axes here rather than K alone.
///
/// C10×10 at 6.30×, identical across two confirmation runs.
pub const QWEN35_9B_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.09,
    k_low: 1.09,
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
/// **Re-derived 2026-08-28, K 1.2 → 1.1 and V 2.5 → 2.35**, after the dense
/// weights moved into the device reservation — the width/accumulation drift
/// this row has now been caught by three times. C10×64 had gone red.
///
/// Both axes walked separately, both edges bracketed:
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
pub const QWEN35_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.1,
    k_low: 1.1,
    v_hi: 2.35,
    v_low: 2.35,
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
/// C10 at 6.67/6.65/6.65/6.65× (×8/16/32/64), identical across two
/// confirmation runs.
pub const QWEN36_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.15,
    k_low: 1.15,
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
/// # It is the loosest row in the lineage, and that is now measured
///
/// K 1.3 sits above the 9B's 1.09 and the MoE pair's 1.1/1.15 — the ordering
/// the old extrapolation asserted but got backwards when its anchors moved
/// (it sat at 1.5, looser than every measured row). The 27B genuinely does
/// have the most headroom here; it simply had to be measured to know it.
pub const QWEN38_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.3,
    k_low: 1.3,
    v_hi: 2.3,
    v_low: 2.3,
};
