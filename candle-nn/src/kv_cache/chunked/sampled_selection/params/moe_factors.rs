//! The error-threshold factors for the MoE models, each measured against the
//! C-ladder gate of its model.
//!
//! The rows are re-derived as the engine changes. Their comments record that a
//! change to wave width, to admission or to a merge moved the edge sessions and
//! sent a row red, and that a row which passed on one branch does not transfer to
//! another, because the numerics differ. Together they establish that a lower
//! factor is not automatically safer: the error ladder is not monotone in these
//! factors, so only running the gate shows that a pair holds. A single-axis probe
//! can mislead near the edge, because the two axes can need to move together.

use super::*;

/// Tuned for Qwen3-30B-A3B (MoE).
///
/// Re-derived 2026-09-25 (`k_low: 1.200 -> 1.000`) — the first time this row has
/// been touched since it was imported, and the change that brings **C10 into the
/// gate for this model at all**. Until now `quantized_qwen3_moe`'s C10 rung was
/// commented out as "the compression is just too much"; it is measured here as a
/// threshold, not a model limit.
///
/// **K was the aggressive side, not V, and the level's own doc comment is what
/// hid it.** `InferenceMode::C10` is described as "K same as C9, V pushed
/// further", which is true of the *candidate format list* and not of the
/// thresholds: from C9 to C10 the shared tables move K high +14.5% and K low
/// +20.7% against V high's +2.5%. V low does take the largest single step
/// (+43.3%), which is why the V side looks like the obvious lever and is the
/// wrong one here. This model is also the widest lo/hi clamp range in the table —
/// `k_low/k_hi` = 29x, against Qwen3-8B's 18.5x — so a below-median block's K
/// threshold could scale further out than on any other row.
///
/// Measured on the RTX PRO 5000, C10x2 StoryRewrite, sessions passing validation:
///
/// | side | value | C10 | mean C0-C9 ratio |
/// |---|---|---|---|
/// | K (chosen) | `k_low` 1.200 | 0/2 | 3.77x (baseline) |
/// | K | `k_low` 1.100 | 1/2 | — |
/// | K | `k_low` **1.000** | **2/2** | 3.75x (**-0.6%**) |
/// | V (rejected) | `v_hi` 1.062 | 1/2 | — |
/// | V (rejected) | `v_hi` 0.899 + `v_low` 2.598 | 2/2 | 3.59x (-4.7%) |
///
/// The graded 0/2 -> 1/2 -> 2/2 response is what identifies this as a threshold:
/// a model-accuracy fault does not walk toward passing as the bound tightens.
/// The V route reaches the same pass at eight times the cost and a *lower* top
/// rung (C10 5.36x vs 5.50x), because tightening V spends ratio on all eleven
/// levels to fix one. 1.000 sits one 0.100 step inside the measured edge, the
/// same margin the C10 threshold note below uses, so ordinary upstream drift does
/// not put the row back on an edge.
///
/// Cost at the rungs that already passed: C0 unchanged at 1.98x, C9 5.31x ->
/// 5.24x, and C10 arrives at 5.50x — above the old C9, so the rung earns its
/// place rather than merely existing.
///
/// # Re-derived 2026-09-26 (`k_hi: 0.475 -> 0.425`) — C10 at the *daemon's* context
///
/// The row above was calibrated against the gate's declared context of 262,144. The
/// engine declares whatever its profile asks for — 8,192 for the fragmentation probe —
/// and a different declared context is a different RoPE table, so it is a different
/// numeric configuration. C10 failed 1/2 there, reproducibly, while every other rung
/// passed and the 262,144 gate stayed 2/2.
///
/// **`k_hi` is the lever at this context, and `k_low` is not.** Stepping `k_low` down was
/// the obvious move — it is what the 2026-09-25 note above identifies as *the* K lever —
/// and it is the wrong one here. Measured at 8,192, C10x2:
///
/// | side | value | C10 sessions | C10 ratio |
/// |---|---|---|---|
/// | `k_low` | 1.000 (as calibrated) | 1/2 | 5.46x |
/// | `k_low` | 0.900 | 1/2 | 5.42x |
/// | `k_low` | 0.800 | 1/2 | 5.37x |
/// | `k_hi` | 0.475 (as calibrated) | 1/2 | 5.46x |
/// | `k_hi` | **0.425** | **2/2** (twice) | 5.38x |
///
/// Three `k_low` steps cost 0.09x of ratio and moved the pass rate not at all — which is
/// the diagnostic, and it is worth stating because it is the opposite of the graded
/// 0/2 -> 1/2 -> 2/2 response that identified `k_low` at the other context. **Monotonic
/// cost with a flat pass rate means the lever is not connected to the failure**; the
/// failing block is above the median, so its K threshold scales by `k_hi`.
///
/// `k_low` is therefore left at its calibrated 1.000. 0.425 is one 0.050 step inside the
/// measured edge, the same margin discipline the note above uses.
///
/// Cost, both contexts re-measured: at 8,192 C9 5.22x -> 5.16x and C10 5.46x -> 5.38x; at
/// 262,144 C9 5.18x and C10 5.42x, every rung still passing. C0 unchanged at 1.98x in
/// both. So the tightening is paid at the top two rungs only, and the published gate keeps
/// a C10 above its own C9.
pub const QWEN3_MOE_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 0.425,
    k_low: 1.000,
    v_hi: 1.225,
    v_low: 2.700,
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
    //
    // **Re-set 2026-10-03, k 0.95 → 0.90 and v 2.0 → 1.9**, after the decode
    // work of that day (warp-per-row DeltaNet decode step, batched greedy picks)
    // moved the hybrid on `Int8Mode::Performance` one session past the edge
    // again: C10×64 at 63/64, every other gate and rung green, ratio 5.97×.
    // Both axes one notch down together.
    //
    // **The stock gate runs `Int8Mode::auto` from 2026-10-04**, which resolves
    // to Precision on an int8-MMA card — the mode the engine loads it in, so
    // gate and engine measure the same numerics. The "stock, Performance" row
    // above is therefore no longer what that gate measures; the factors were
    // not changed for it.
    k_hi: 0.90,
    k_low: 0.90,
    v_hi: 1.9,
    v_low: 1.9,
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
///
/// **Re-set 2026-10-03: k 1.8 → 1.7, v 3.0 → 2.8.** The PLE block now runs
/// over the whole wave, so its `key`/`value` projections are one matmul at
/// M = the wave's rows where they were one per sequence (M = 1 in decode).
/// That is a different F32 summation order inside cuBLAS — about two ulps on
/// the layer-1 projections, not a loss of accuracy — but it moves every later
/// layer's K/V by rounding, and C10, calibrated just under the edge, carries
/// sessions on a near-tie that the perturbation tips. C10×8 failed 6/8 at
/// 1.8 / 3.0 after passing in every run that day, and 7/8 at 1.7 / 3.0 in two
/// runs out of two (6.63×). At 1.7 / 2.8 it passed 8/8 at 6.45× (C10×2) /
/// 6.43× (C10×8), against 6.75× before — the price of keeping the wave-wide
/// projection, which reads the 131 MB of PLE weights once per wave instead of
/// once per sequence. V at 2.8 is the value the lineage's earlier retightening
/// measured.
///
/// **Re-set 2026-10-05: v 2.8 → 2.6.** The forward gate's prefill slabs
/// widened: the width cap is now what the card can place
/// (`Qwen4ExpBatched::prefill_width_cap`), so the C10×8 fleet prefills as one
/// 4,120-row slab where it was two. A different composition is a different
/// summation order in the wave-wide projections, and at 1.7 / 2.8 one session
/// tipped on a near-tie — 7/8 (`watched` → `watch`) in two runs out of two at
/// 7.33×, every other config passing. At 2.7 the same session still failed,
/// diverging earlier (7.23×). V, the binding axis on this model, moves two
/// notches.
pub const QWEN4EXP_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.7,
    k_low: 1.7,
    v_hi: 2.6,
    v_low: 2.6,
};

/// **Qwen3.8-Flash-Next at `Q2_KO` experts** — measured on the RTX 4090 Mobile
/// while `Q2_KO` was the 16 and 24 GiB cards' rung (`quant_ladder`).
///
/// Those cards now run `Q3_KO` experts under this row (`qwen4exp::kv_row`):
/// wider experts have more margin, so a row placed against `Q2_KO`'s edge
/// stays valid for them at some cost in ratio. The brackets below are `Q2_KO`'s.
///
/// A row per expert format because the design calls for one
/// (`docs/qwen38_flash_next.md` §Phase 6: "per machine and per expert format"),
/// and the measurement agrees: at [`QWEN4EXP_KV_FACTORS`] — derived on `Q4_KO`
/// experts, one notch under C10's edge — the `Q2_KO` model passes BF16 and
/// C0/C5/C8 at 100% but fails C10 (0/2 at ×2, 3/8 at ×8). Narrower experts
/// leave the model less margin for K/V error, so the edge moves inward; taking
/// that from the shared row would charge the `Q4_KO` machines compression at C5
/// for a margin only this rung needs.
///
/// Bracketed on the C-ladder gate (2026-09-19, RTX 4090 Mobile):
///
/// | k | v | C10×2 | C10×8 | C10 ratio | C5 ratio |
/// |---|---|-------|-------|-----------|----------|
/// | 1.8  | 3.0 | 0/2 | 3/8 | 7.75× | 4.22× |
/// | 1.8  | 2.7 | 1/2 | 3/8 | 7.41× | 4.20× |
/// | 1.65 | 2.4 | 2/2 | 4/8 | 6.84× | 4.13× |
/// | **1.5** | **2.2** | **2/2** | **8/8** | **6.38×** | **4.06×** |
///
/// The failures are early — the first divergent character is at 8–84 of ~350
/// — which is what a row past the edge looks like here, not a slow drift. At
/// the bracketed row the whole ladder (BF16 ×1/×4/×8, C0/C5/C8/C10×2/C10×8)
/// passed, and C5 — the level zend runs — gave up 3.8% of its ratio against
/// the `Q4_KO` row, the price of the narrower experts' margin.
///
/// **Re-derived 2026-09-20: v 2.2 → 2.1.** At 1.5 / 2.2 a later gate run
/// passed C10×2 (6.37×) but failed C10×8 at 7/8 — one session dropped a word at
/// character 83, the early-divergence shape of a row on the edge. V is the axis
/// that gives way first at the top rung, so only V moves, by the smallest step
/// the bracket above resolves.
///
/// Reproduced on this branch's engine 2026-09-29 before the row was wired in:
/// the `Q4_KO` row failed C10 1/2 at ×2 and 3/8 at ×8 with every other rung
/// passing — the same shape the bracket above starts from.
///
/// **Re-derived 2026-10-01: k 1.5 → 1.45.** Q0_V's K blocks are now sealed and
/// read under the K codebook (they had been sealed and read under V), which
/// moves which codes the K side stores. With every other rung passing, C10×8
/// failed 7/8 twice — one session repeating a word at character 145, a row
/// sitting on the edge — so K, the axis that change moved, takes the smallest
/// step. At 1.45 / 2.1 the whole ladder passed on two consecutive runs, C10×8
/// 8/8 both times; C10 went 5.43× → 5.38× and C5, the level zend runs,
/// 4.01× → 3.98×.
///
/// **Re-derived 2026-10-03: k 1.45 → 1.40, v 2.1 → 2.0.** The quantizer no
/// longer reads a partial chunk's dead slots, and Q0 is offered only flat
/// blocks; at 1.45 / 2.1 that took this model's C10 from 5.38× to 6.14×, and
/// C10×2 then dropped one word at character 256 (C10×8 8/8) — the early-edge
/// shape above. Both axes take the smallest step. The whole ladder passed in one
/// gate run: C10×2 2/2 at 5.97×, C10×8 8/8 at 5.96×. The step costs C10 2.8%
/// (6.14× → 5.97×), C8 1.6% and C5, the level zend runs, 0.3% (3.44× → 3.43×).
///
/// **Re-derived 2026-10-04: k 1.40 → 1.35, v 2.0 → 1.9.** At 1.40 / 2.0 the gate
/// failed C10×8 at 7/8 on three consecutive runs, with every other rung
/// passing and C10 at the same 5.96× the 2026-10-03 pass recorded — a row on the
/// edge again, so both axes take the smallest step. The whole ladder passed on
/// two consecutive gate runs, C10×8 8/8 both times. The step costs C10 2.3%
/// (5.96× → 5.82× at ×2, 5.80× at ×8), C8 2.0% (4.97× → 4.87×) and C5, the
/// level zend runs, 1.3% (3.85× → 3.80×).
pub const QWEN4EXP_Q2KO_KV_FACTORS: KvErrorThresholdFactors = KvErrorThresholdFactors {
    k_hi: 1.35,
    k_low: 1.35,
    v_hi: 1.9,
    v_low: 1.9,
};
