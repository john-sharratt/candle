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

/// Qwen3.8-27B (dense). Has a NextN head, dense rather than routed.
pub const QWEN38_27B_DRAFT: DraftLadder = DraftLadder::new(LINEAGE_START);

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

    /// A shadowed bracket answers the wrong budget rather than failing, which is
    /// exactly why `check` exists — pin that it catches both malformations.
    #[test]
    fn check_rejects_shadowed_and_rising_brackets() {
        assert!(DraftLadder::new(&[(16, 2), (8, 1)]).check().is_err());
        assert!(DraftLadder::new(&[(8, 1), (16, 2)]).check().is_err());
        assert!(DraftLadder::new(&[(8, 2), (16, 1)]).check().is_ok());
    }
}
