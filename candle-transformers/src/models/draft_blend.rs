//! A checkpoint's draft ceiling as each card measured it, blended by VRAM.
//!
//! The width ladder ([`super::draft_ladder`]) is a property of the checkpoint
//! *and the card*: how much a verify block costs beside a step depends on the
//! card's bandwidth, its expert residency and its compute. Each card's row is
//! fitted from its own sweep (`spec_sweep_wNN`: every budget's decode rate at a
//! width, StoryRewrite, one process per width), and a card between two
//! measured ones takes the row between them by its VRAM.
//!
//! **The fit, in three steps, all reproducible from the measured rates.**
//! 1. At each width, the smallest budget whose rate is within [`NEAR_BEST`] of
//!    the width's best: neighbouring budgets sit within a few per cent of each
//!    other, so the raw best flips on noise.
//! 2. Across widths, a non-increasing step function by pool-adjacent-violators:
//!    a verify block costs more as the wave widens, never less, so a ceiling
//!    that rose with width would be the noise again.
//! 3. Between cards, each width's budget is the VRAM-weighted mean of the two
//!    rows, rounded; a card outside the two takes the nearer one's row.
//!
//! **A ceiling, not a depth.** The sweep runs the rewrite, the text a head
//! predicts best, so it finds the most depth that ever pays at a width; each
//! sequence drafts below it by its own acceptance (`super::draft_depth`), which
//! is how free text and dialogue draft shallower under the same ceiling.

/// A budget within this fraction of a width's best rate counts as the best.
pub const NEAR_BEST: f64 = 0.01;

/// One width of a card's sweep: the decode rate at each budget, from 0, as far
/// as the sweep ran it (it stops at the first budget slower than the one
/// before).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepWidth {
    pub width: usize,
    pub rates: &'static [f64],
}

/// A card's measured sweep: its VRAM and its widths, ascending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CardSweep {
    pub vram_mib: u64,
    pub widths: &'static [SweepWidth],
}

/// The smallest budget whose rate is within [`NEAR_BEST`] of the best at this
/// width; 0 when no budget beats plain decode.
fn near_best(rates: &[f64]) -> usize {
    let best = rates.iter().copied().fold(f64::MIN, f64::max);
    rates
        .iter()
        .position(|&r| r >= best * (1.0 - NEAR_BEST))
        .unwrap_or(0)
}

/// `values` as the closest non-increasing sequence, by pool-adjacent-violators:
/// each run that rises is replaced by its mean.
fn non_increasing(values: &[f64]) -> Vec<f64> {
    // Blocks of (mean, count), merged while a later block's mean exceeds the
    // one before it.
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(values.len());
    for &v in values {
        blocks.push((v, 1));
        while blocks.len() > 1 {
            let (m2, n2) = blocks[blocks.len() - 1];
            let (m1, n1) = blocks[blocks.len() - 2];
            if m2 <= m1 {
                break;
            }
            blocks.pop();
            let last = blocks.last_mut().expect("two blocks were there");
            *last = (
                (m1 * n1 as f64 + m2 * n2 as f64) / (n1 + n2) as f64,
                n1 + n2,
            );
        }
    }
    blocks
        .into_iter()
        .flat_map(|(m, n)| std::iter::repeat_n(m, n))
        .collect()
}

impl CardSweep {
    /// This card's fitted ceiling at every width from 1 to the widest it
    /// measured, as a real number before rounding: the near-best budget of each
    /// measured width, made non-increasing, each unmeasured width between two
    /// taking the earlier one's.
    fn fitted(&self) -> Vec<f64> {
        let measured: Vec<f64> = self
            .widths
            .iter()
            .map(|w| near_best(w.rates) as f64)
            .collect();
        let fit = non_increasing(&measured);
        let widest = self.widths.last().map_or(0, |w| w.width);
        let mut out = Vec::with_capacity(widest);
        let mut i = 0;
        for width in 1..=widest {
            while i + 1 < self.widths.len() && self.widths[i + 1].width <= width {
                i += 1;
            }
            out.push(fit[i]);
        }
        out
    }
}

/// The per-width draft ceiling for a card of `vram_mib`, between the card swept
/// with less VRAM (`lo`) and the one with more (`hi`): widths 1 to the widest
/// either measured, then that last ceiling carried on (index `w - 1` is width
/// `w`).
pub fn blend(lo: &CardSweep, hi: &CardSweep, vram_mib: u64) -> Vec<usize> {
    let span = hi.vram_mib.saturating_sub(lo.vram_mib).max(1) as f64;
    let t = (vram_mib.saturating_sub(lo.vram_mib) as f64 / span).clamp(0.0, 1.0);
    let (a, b) = (lo.fitted(), hi.fitted());
    let widest = a.len().max(b.len());
    let at = |row: &[f64], w: usize| row.get(w).or(row.last()).copied().unwrap_or(0.0);
    (0..widest)
        .map(|w| (at(&a, w) * (1.0 - t) + at(&b, w) * t).round() as usize)
        .collect()
}

/// A resolved ceiling: the blended row, read for any width. A width past the
/// row's end takes its last entry — the widest the cards measured, carried on —
/// and a width of 0 drafts nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftTable {
    budgets: Vec<usize>,
}

impl DraftTable {
    pub fn new(budgets: Vec<usize>) -> Self {
        Self { budgets }
    }

    pub fn budget(&self, width: usize) -> usize {
        if width == 0 {
            return 0;
        }
        self.budgets
            .get(width - 1)
            .or(self.budgets.last())
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The best within 1% wins at its smallest budget, and a width where no
    /// budget beats plain decode drafts nothing.
    #[test]
    fn the_near_best_budget_is_the_smallest_within_a_percent() {
        assert_eq!(
            near_best(&[100.0, 150.0, 200.0, 199.0]),
            2,
            "199 is within 1% of 200"
        );
        assert_eq!(near_best(&[100.0, 150.0, 200.0, 210.0]), 3);
        assert_eq!(near_best(&[100.0, 90.0, 80.0]), 0);
    }

    /// A rising run is pooled to its mean; a non-increasing sequence is left as
    /// it is.
    #[test]
    fn a_rise_is_pooled_away() {
        let pooled = non_increasing(&[4.0, 3.0, 4.0, 4.0, 2.0]);
        assert_eq!(pooled[0], 4.0);
        assert!((pooled[1] - 11.0 / 3.0).abs() < 1e-9 && pooled[1] == pooled[3]);
        assert_eq!(pooled[4], 2.0);
        assert_eq!(
            non_increasing(&[4.0, 4.0, 3.0, 1.0]),
            vec![4.0, 4.0, 3.0, 1.0]
        );
    }

    const LO: CardSweep = CardSweep {
        vram_mib: 24_000,
        widths: &[
            SweepWidth {
                width: 1,
                rates: &[10.0, 20.0, 30.0],
            },
            SweepWidth {
                width: 4,
                rates: &[10.0, 20.0, 19.0],
            },
        ],
    };
    const HI: CardSweep = CardSweep {
        vram_mib: 72_000,
        widths: &[
            SweepWidth {
                width: 1,
                rates: &[10.0, 20.0, 30.0, 40.0, 50.0],
            },
            SweepWidth {
                width: 2,
                rates: &[10.0, 20.0, 30.0, 40.0, 50.0],
            },
        ],
    };

    /// **Each card at its own VRAM reads its own row; between, the mean.** The
    /// low card's width 2 and 3 take its width 1 (the earlier measured width),
    /// and past either card's widest the last ceiling is carried on.
    #[test]
    fn a_card_between_two_takes_the_row_between_them() {
        assert_eq!(blend(&LO, &HI, 24_000), vec![2, 2, 2, 1]);
        assert_eq!(blend(&LO, &HI, 72_000), vec![4, 4, 4, 4]);
        assert_eq!(blend(&LO, &HI, 48_000), vec![3, 3, 3, 3], "2.5 rounds up");
        assert_eq!(
            blend(&LO, &HI, 8_000),
            blend(&LO, &HI, 24_000),
            "clamped below"
        );
        assert_eq!(
            blend(&LO, &HI, 96_000),
            blend(&LO, &HI, 72_000),
            "clamped above"
        );
    }

    /// A table answers past its end with its last entry, and nothing at 0.
    #[test]
    fn a_table_carries_its_widest_entry_on() {
        let t = DraftTable::new(vec![4, 3, 2]);
        assert_eq!(
            (t.budget(0), t.budget(1), t.budget(3), t.budget(40)),
            (0, 4, 2, 2)
        );
        assert_eq!(DraftTable::new(Vec::new()).budget(5), 0);
    }
}
