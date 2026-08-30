//! How far ahead a checkpoint drafts, as a function of decode-wave width.
//!
//! Speculation trades compute for latency: a verify block scores `k + 1` rows
//! where a plain decode scores one, and writes `k + 1` KV entries where a plain
//! decode writes one. Both are nearly free while the wave is
//! memory-bandwidth-bound — a narrow decode reads the entire weight set to score
//! a handful of rows, so scoring a few more costs almost nothing — and both cost
//! what they weigh once the wave is wide enough to be compute-bound.
//!
//! **The turnover is not a single on/off point.** The proposals fall off one at
//! a time: the second stops paying well before the first does, so a ladder that
//! could only choose between "full budget" and "none" would be wrong on both
//! sides of the middle. Hence a table of brackets rather than a threshold.
//!
//! Where the brackets sit is a property of the checkpoint — how much weight a
//! step reads, whether its experts stream, how wide its KV rows are — so each
//! model carries its own row, derived by the `width_ladder_*` sweeps and pinned
//! here beside the checkpoint it was measured on. This is the same arrangement
//! the KV threshold factor rows use, for the same reason.
//!
//! # Measuring a row
//!
//! **One point per process, and let the card settle between them.**
//! `quantized_qwen35::tests::cold_speculative_point` takes a single width and a
//! single budget and runs exactly one config; the `cold_*` tests beside it are
//! the points. Run them singly, waiting for the card to return to idle
//! temperature and clocks in between, and read each budget against its own
//! width's budget-0 point.
//!
//! That protocol is not fussiness. A laptop card holds full boost for the first
//! tens of seconds of load and then roughly halves its clock, so a sweep that
//! runs several configs against one loaded model measures *position in the run*
//! as much as the variable it varies. The same config measured 247 tok/s as the
//! first config of a load and 103 as the third, four repetitions running, with
//! the KV region ledger byte-identical between them — no leak, no pressure, just
//! a spent boost budget.
//!
//! An earlier sweep here did exactly that, width-major, and every "width curve"
//! it produced was that artefact: width rose with position, so boost depletion
//! read as a cliff. It has been deleted rather than documented, because a
//! confounded instrument left in the tree gets run again.
//! `decay_across_configs_9b` is kept as the demonstration — one config three
//! times over, which is where the effect is unmistakable.

/// A checkpoint's draft budget as a function of decode-wave width.
///
/// Brackets are `(max width, budget)`, read in order: the first whose width
/// covers the wave wins, and a wave wider than every bracket does not
/// speculate. They must ascend in width with non-increasing budgets —
/// [`DraftLadder::check`] asserts it, and the module's tests run it over every
/// row here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftLadder {
    brackets: &'static [(usize, usize)],
}

impl DraftLadder {
    pub const fn new(brackets: &'static [(usize, usize)]) -> Self {
        Self { brackets }
    }

    /// A checkpoint that never speculates — no drafter, or one not yet measured.
    pub const NONE: Self = Self::new(&[]);

    /// Tokens each sequence drafts on a wave of `width` sequences.
    pub fn budget(&self, width: usize) -> usize {
        if width == 0 {
            return 0;
        }
        self.brackets
            .iter()
            .find(|&&(max, _)| width <= max)
            .map_or(0, |&(_, budget)| budget)
    }

    pub fn brackets(&self) -> &'static [(usize, usize)] {
        self.brackets
    }

    /// Why the ordering is load-bearing: [`Self::budget`] takes the FIRST
    /// covering bracket, so a wider bracket listed earlier silently shadows
    /// every narrower one — the ladder would still answer, just with the wrong
    /// budget at every width below the shadow. A budget that *rose* with width
    /// would contradict the curve the table exists to encode.
    #[cfg(test)]
    fn check(&self) -> Result<(), String> {
        for pair in self.brackets.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(format!("widths must ascend: {pair:?}"));
            }
            if pair[0].1 < pair[1].1 {
                return Err(format!("budgets must not rise with width: {pair:?}"));
            }
        }
        Ok(())
    }
}

/// The Qwen3.5 lineage's shared starting row, pending per-checkpoint sweeps.
///
/// Every drafting checkpoint in the lineage runs the same NextN/MTP head — one
/// transformer block applied recurrently — so they start from one row and split
/// only where measurement says they differ.
///
/// **Full budget or none, and it holds much wider than it first appeared.**
///
/// Measured on the 9B with the `cold_*` points — one width and one budget per
/// process, the card cooled to ~52 °C between every point, each budget against
/// its own width's budget-0 baseline:
///
/// | width | budget 1 | budget 2 |
/// |-------|----------|----------|
/// | 10    | 1.25x    | **1.53x** |
/// | 16    | 1.01x    | **1.05x** |
/// | 20    | 1.00x    | 0.39x     |
///
/// Two things come out of it. Budget 2 beats budget 1 wherever speculation pays
/// at all, so there is no middle rung: a step buys one draft pass and one verify
/// wave whatever `k` is, and only the extra scored row scales with it, so budget
/// 1 pays nearly the full price for two-thirds of the tokens. And the win decays
/// smoothly to break-even around 16 before collapsing by 20 — so the bracket
/// sits at 16, the widest width measured to still pay.
///
/// **Every earlier reading of this curve was an artefact, including the ones
/// that were in this comment.** A laptop card holds full boost for the first
/// tens of seconds of load and then halves its clock, so a gate running several
/// configs against one loaded model measures position in the run as much as
/// width — the same config measured 247 tok/s first and 103 tok/s third, four
/// times over, with the KV region ledger byte-identical between them. Width rose
/// with position in every sweep, so boost depletion read as a width cliff. An
/// earlier version of this table put the bracket at 8 and called width 10 a
/// 0.45x collapse; measured cold, width 10 is a 1.53x win.
///
/// So: measure one point per process, cool between points, and distrust any row
/// that shared a process with the row before it. `cold_speculative_point` is the
/// instrument.
///
/// Above 20 is extrapolation, not measurement — 32 sessions do not fit this
/// card's budget for a 256-token run. The collapse at 20 is steep enough that
/// zero is the safe answer beyond the bracket either way.
///
/// **The bracket sits on its own edge, and that is a deliberate but thin
/// choice.** 16 is the widest width measured to pay, and it pays by only
/// 5–10%; 17, 18 and 19 are unmeasured; 20 loses badly. So the top of the range
/// is nearly free either way, and the case for 16 over something with margin is
/// that widths 13–16 are common and their gain is real. If a later measurement
/// finds the turn is below 16 rather than between 16 and 20, pull the bracket
/// in — the asymmetry favours it, because the loss past the turn (0.39x) dwarfs
/// the gain before it.
///
/// The 35B-A3B and 27B carry this row **unmeasured**. The 3.6-35B is the only
/// routed checkpoint with cold points of its own, and it tracked the dense 9B
/// closely enough to justify sharing — but "same head, similar curve" is
/// evidence for two checkpoints, not for four.
///
/// # The routed checkpoints share this row, and now by measurement
///
/// A streaming-expert MoE looked like it should want a *narrower* bracket: a
/// verify block scores `k + 1` positions per sequence and each routes to its own
/// top-8 of 256, so the wave's routed union widens with the block, and on this
/// card expert traffic crosses PCIe. Measured on the 3.6-35B with the same cold
/// points (`cold36_*`), it does not — the curve tracks the dense 9B closely:
///
/// | width | 9B (dense) | 3.6-35B (routed) |
/// |-------|-----------|------------------|
/// | 4     | —         | **1.91x** |
/// | 10    | 1.53x     | **1.48x** |
/// | 16    | 1.05x     | **1.10x** |
///
/// **Expert bandwidth moves the other way from the intuition.** Over the same
/// 4,080 generated tokens at width 16, host-to-device expert loads were
/// **244,180 at budget 0 and 180,691 at budget 2** — speculation costs 26%
/// *fewer*, with late loads falling from 26 to 0. The union per wave is wider
/// and the hit rate does drop (40.6% → 36.0%), but a speculative step emits
/// about three tokens where a decode step emits one, so a third as many waves
/// each load experts once. Fewer waves beats wider unions.
///
/// The practical consequence is the opposite of the worry: speculation is *more*
/// attractive on a streaming-expert model than on a dense one, because it
/// amortises the expert loads that dominate its step.
///
/// Qwen3-30B-A3B is absent from this table because it cannot speculate at all —
/// no NextN tensors in any conversion, so it takes the trait default of zero and
/// has no ladder to carry.
/// # Measured on two checkpoints, applied to four
///
/// The bracket comes from cold points on Qwen3.5-9B (1.53× at 10, 1.05× at 16,
/// 0.39× at 20) and Qwen3.6-35B-A3B (1.91× at 4, 1.48× at 10, 1.10× at 16).
/// Widths 17–19 are unmeasured, and `QWEN35_35B_A3B_DRAFT` and
/// `QWEN38_27B_DRAFT` have no points of their own — they inherit this row on the
/// argument that the lineage shares a decode loop, which is a hypothesis rather
/// than a measurement.
///
/// The asymmetry is what makes that worth stating: a bracket set a little short
/// costs the 5–10% still on the table just before the turn, while one set a
/// little long costs the 0.39× just after it — about 2.5×. So when a checkpoint
/// here is measured for the first time and disagrees, give it its own row rather
/// than moving this one, and expect the correction to be *inward*.
///
/// # Changing it re-opens a KV calibration in another crate
///
/// `candle_nn`'s `QWEN35_MOE_KV_FACTORS` was tuned against C10 rungs — ×8, ×16,
/// ×32 and ×64 as the gate now sweeps them — of which ×8 and ×16 speculate
/// *because* this bracket reaches 16. Pulling it in below that turns those
/// rungs into plain decode and moves the marginal session, so the KV gate goes
/// red for a reason that lives here.
const LINEAGE_START: &[(usize, usize)] = &[(16, 2)];

/// Qwen3.5-9B (dense). Has a NextN head.
pub const QWEN35_9B_DRAFT: DraftLadder = DraftLadder::new(LINEAGE_START);

/// Qwen3.5-0.8B (dense). **No NextN head in any conversion**, so no ladder —
/// this is the checkpoint's real capability, not a gap awaiting measurement.
pub const QWEN35_0_8B_DRAFT: DraftLadder = DraftLadder::NONE;

/// Qwen3.5-35B-A3B (routed). Has a NextN head, itself a full routed block.
pub const QWEN35_35B_A3B_DRAFT: DraftLadder = DraftLadder::new(LINEAGE_START);

/// Qwen3.6-35B-A3B (routed). Has a NextN head, itself a full routed block.
pub const QWEN36_35B_A3B_DRAFT: DraftLadder = DraftLadder::new(LINEAGE_START);

/// Qwen3.8-27B (dense). Has a NextN head, dense rather than routed — shipped as
/// a **sidecar** GGUF rather than embedded (`mtp-Qwen3.8-27B-Q4_0.gguf`).
///
/// **Its own row, deeper and wider than the lineage's.**
///
/// The lineage bracket is 2, set on the 9B where a third proposal cost more than
/// it returned. The depth of 4 was set here against a *streaming* 27B, on the
/// argument that a verify forward dragged ~18 GB over PCIe while a draft step
/// touched one resident block — roughly 45:1, so depth was close to free.
///
/// **That argument no longer applies, and the depth is kept on new evidence
/// rather than on it.** The two-tier weight zone (`docs/qwen38_layer_streaming.md`
/// §14) holds all 64 layers resident on the 16 GB card, so a verify forward moves
/// no weight bytes at all and the 45:1 is gone. Re-measured resident, tokens/sec
/// at 1 / 4 contexts over two sweeps:
///
/// | budget | 1 ctx | 4 ctx |
/// |--------|-------|-------|
/// | 2      | 61.1 / 62.3 | 173.6 / 185.3 |
/// | 3      | 58.1 / 75.9 | 179.1 / 226.9 |
/// | 4      | 50.4 / 77.7 | 149.0 / 212.8 |
///
/// **Those are gate-sweep numbers, which this module's header says are the
/// confounded instrument.** Several configs run against one loaded model, so
/// position in the run is measured alongside the variable — and the divergence
/// lands exactly where that predicts: the two sweeps agree within 2% at budgets
/// 0–2, the early configs, and disagree by 31% and 54% at 3–4, the last ones, in
/// opposite directions. One sweep read them as a curve turning over at 2; the
/// other has it climbing to 4.
///
/// So **the knee is not resolved**, 4 is retained as the incumbent rather than
/// re-derived, and the table above is recorded to show what is and is not known —
/// not as a measurement. `cold_speculative_point` is the instrument that would
/// settle it: one width, one budget, one process, card cooled between points.
///
/// # The width is 32, and it is a request rather than a measurement
///
/// The lineage's 16 comes from the 9B's cold points, where width 20 measured
/// **0.39×** — speculation actively harmful past the bracket. That is the risk
/// this row now runs toward: the 27B is resident like the 9B, so the mechanism
/// that produced the 9B's cliff (a verify wave carrying rows that are discarded,
/// with no streamed bytes to amortise them) applies here too, and the argument
/// that used to exempt this checkpoint has gone with the streaming.
///
/// Widths 5–32 are **unmeasured on this checkpoint** — the gate samples 1, 4, 5,
/// 10 and 20 contexts, and only 1 and 4 carry speculative points. If a wide rung
/// regresses, this bracket is the first thing to pull back in.
// **Width 16, not the 32 that was asked for — and this is a parked decision,
// not a rejection.** Measured on the 16 GB card by direct control experiment:
// at (32, 4) the gate's C8×20 config takes `CUDA_ERROR_OUT_OF_MEMORY`; at
// (16, 4), with nothing else in the tree different, it passes 20/20. The cause
// is the rewind stash, which is `contexts × (budget + 1)` rows across every
// DeltaNet layer — 60.4 MiB at budget 0 against 301.8 MiB at budget 4, at 20
// contexts.
//
// `HybridBatched::affordable_draft_budget` exists to make (32, 4) safe by
// lowering the *depth* when a cohort is too wide to stash, so speculation
// degrades instead of the wave failing. It does not yet bind: the budget it
// reads reports ~1.4 GiB claimable where the device then refuses, so the clamp
// permits a depth that does not fit. Until that disagreement is understood,
// this row stays at the width that is measured to work.
const QWEN38_27B_BRACKETS: &[(usize, usize)] = &[(16, 4)];
pub const QWEN38_27B_DRAFT: DraftLadder = DraftLadder::new(QWEN38_27B_BRACKETS);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every ladder shipped here must satisfy the ordering `budget` relies on —
    /// checked over the real rows, so a mis-edited table fails here rather than
    /// quietly answering the wrong budget in production.
    #[test]
    fn every_shipped_ladder_is_well_formed() {
        for (name, ladder) in [
            ("9B", QWEN35_9B_DRAFT),
            ("0.8B", QWEN35_0_8B_DRAFT),
            ("35B-A3B", QWEN35_35B_A3B_DRAFT),
            ("3.6-35B-A3B", QWEN36_35B_A3B_DRAFT),
            ("27B", QWEN38_27B_DRAFT),
        ] {
            if let Err(e) = ladder.check() {
                panic!("{name} ladder is malformed: {e}");
            }
        }
    }

    /// Full budget up to the bracket, plain decode past it — no middle rung.
    #[test]
    fn budget_is_full_or_nothing() {
        let l = QWEN35_9B_DRAFT;
        assert_eq!(l.budget(1), 2);
        assert_eq!(l.budget(10), 2);
        assert_eq!(l.budget(16), 2);
        assert_eq!(l.budget(17), 0);
        assert_eq!(l.budget(20), 0);
        assert_eq!(l.budget(4096), 0);
        // Nothing in the lineage's shipped rows asks for a single proposal.
        assert!(
            !LINEAGE_START.iter().any(|&(_, b)| b == 1),
            "a budget-1 bracket reappeared — the cold points measured budget 2 ahead of \
             budget 1 at every width where speculation pays at all"
        );
    }

    /// An empty wave drafts nothing rather than indexing the first bracket.
    #[test]
    fn an_empty_wave_drafts_nothing() {
        assert_eq!(QWEN35_9B_DRAFT.budget(0), 0);
    }

    /// A checkpoint with no drafter answers zero at every width, including the
    /// ones its siblings speculate at.
    #[test]
    fn a_ladderless_checkpoint_never_speculates() {
        for width in [0, 1, 4, 8, 16, 64] {
            assert_eq!(QWEN35_0_8B_DRAFT.budget(width), 0);
        }
    }

    /// **The 27B's row is its own, and widening it must not reach the others.**
    ///
    /// It carries its own *depth* (4 against the lineage's 2), and the two rows
    /// are separate consts precisely so that stays true. A future edit that
    /// "simplifies" them back onto `LINEAGE_START` would take the 27B's measured
    /// depth away, and a future widening of the 27B's row must not hand three
    /// other checkpoints a width measured at 0.39× past 16.
    #[test]
    fn only_the_27b_carries_its_own_depth() {
        assert_eq!(QWEN38_27B_DRAFT.budget(16), 4, "the 27B's own depth");
        // The width is parked at 16 pending the stash-affordability work — see
        // the const's own note. Pinned so re-widening is a deliberate edit here
        // rather than a number that drifts.
        assert_eq!(QWEN38_27B_DRAFT.budget(17), 0, "parked at width 16");
        for (name, ladder) in [
            ("9B", QWEN35_9B_DRAFT),
            ("35B-A3B", QWEN35_35B_A3B_DRAFT),
            ("3.6-35B-A3B", QWEN36_35B_A3B_DRAFT),
        ] {
            assert_eq!(ladder.budget(16), 2, "{name} keeps the lineage budget");
            assert_eq!(
                ladder.budget(17),
                0,
                "{name} inherited the 27B's width — the rows have been merged"
            );
        }
        assert_eq!(
            QWEN35_0_8B_DRAFT.budget(17),
            0,
            "0.8B has no drafter at all"
        );
    }

    /// A shadowed bracket answers the wrong budget rather than failing, which is
    /// exactly why `check` exists — pin that it catches both malformations.
    #[test]
    fn check_rejects_shadowed_and_rising_brackets() {
        assert!(DraftLadder::new(&[(16, 2), (8, 1)]).check().is_err());
        assert!(DraftLadder::new(&[(8, 1), (16, 2)]).check().is_err());
        assert!(DraftLadder::new(&[(8, 2), (16, 1)]).check().is_ok());
    }
}
