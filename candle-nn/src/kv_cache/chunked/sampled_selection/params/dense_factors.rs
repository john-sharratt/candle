//! The error-threshold factors for the dense models, each one a row measured
//! against that model's C-ladder gate.
//!
//! Each row multiplies the shared production thresholds for one model, and each
//! value is a pair that the gate passed on repeated runs. A row is chosen to pass
//! repeatably rather than to sit on the edge of its break. The comments tie a row
//! to the numeric path and the card it was measured on: the same model on another
//! card, or at another weight quantization, can need a different pair, and a row
//! parked one notch under its break fails on any change to the wave's width or
//! to the arithmetic.

use super::*;

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
