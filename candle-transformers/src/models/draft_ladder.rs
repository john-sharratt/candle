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
//! `quantized_qwen35::tests::speculative_width_ladder` sweeps width × budget in
//! one run, comparing each budget against **its own width's** budget-0 baseline
//! so the ratios are within-run. Two things to respect when reading its output:
//!
//! * **Run it cold.** These are long sweeps on a laptop-class card and a
//!   thermally throttled tail does not merely add noise, it inverts rows. A 9B
//!   sweep's 12-wide row came back with budget 1 losing and budget 2 winning,
//!   contradicting both its neighbours, after six prior configs had been run
//!   back to back.
//! * **Check the baselines before the ratios.** If aggregate throughput falls as
//!   width rises, that row's ratio is between two degraded numbers and says more
//!   about the card than the ladder.

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
/// **There is no middle rung: it is full budget or none.** Measured cold on the
/// 9B (`middle_rung_9b`, 256 tokens, each budget against its own width's
/// budget-0 baseline):
///
/// | width | budget 1 | budget 2 |
/// |-------|----------|----------|
/// | 8     | 1.41x    | **1.46x** |
/// | 10    | 0.42x    | 0.45x     |
///
/// Budget 2 beats budget 1 at *both* widths — including the one where both
/// lose. A step pays for one draft pass and one verify wave whatever `k` is,
/// and only the extra scored row scales with it, so budget 1 buys about
/// two-thirds of the tokens for very nearly the same price. It is never the best
/// answer, and the ladder has one bracket rather than two.
///
/// Acceptance says the same thing from the other side: at budget 2 it is 2.93
/// per session at width 8 and 2.93 at width 10, identical. Width 10 does not
/// lose because the drafter got worse — it loses because the wave got
/// expensive.
///
/// **The cliff is between 8 and 10**, and it is a cliff: 1.46x to 0.45x with
/// nothing in between measured. Width 9 is untested, so the bracket sits at the
/// widest width observed to win rather than at a fitted crossing.
///
/// An earlier *hot* sweep read width 8 the other way round (budget 1 at 1.13x,
/// budget 2 at 0.50x) and had this table stepping through 1 on the way down.
/// That row was inverted by thermal throttling — the cold run above is the one
/// to trust, and it is why this module tells you to run these sweeps cold.
const LINEAGE_START: &[(usize, usize)] = &[(8, 2)];

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
        assert_eq!(l.budget(8), 2);
        assert_eq!(l.budget(9), 0);
        assert_eq!(l.budget(4096), 0);
        // Nothing in the lineage's shipped rows asks for a single proposal.
        assert!(
            !LINEAGE_START.iter().any(|&(_, b)| b == 1),
            "a budget-1 bracket reappeared — `middle_rung_9b` measured it as never optimal"
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
