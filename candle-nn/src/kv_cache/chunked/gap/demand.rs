//! How many slots of each class a gap leases (`docs/kv_quantize_in_the_gap.md`
//! §3.2).
//!
//! The demand comes from the summaries. The claim counts every band of its
//! table into a class, deferred or not, so a summary row names the class of
//! every band it saw, and a gap's demand is two parts:
//!
//! - **Retried blocks, exactly.** A deferred block's rows name each band's
//!   format, so its demand per class is known to the slot. Deferred blocks lead
//!   the next table in groups of their own, ahead of every group of new blocks,
//!   and the claim hands out slots in table order across groups, so leasing
//!   their exact demand places them.
//! - **New blocks, by measured share.** Their band count is exact; only the
//!   split across classes is not yet known. It is each class's peak share of
//!   claims over the recent summaries, kept apart for prefill and decode gaps,
//!   capped at the band count. Before any summary, each reachable class takes
//!   an even part of the count firm and the rest of it spare.
//!
//! Firm demand may buy ground from the weight side; spare demand takes only
//! ground standing free. An over-estimate costs slots allocated and returned
//! within one gap; an under-estimate defers the blocks past it, whose exact
//! demand the next summary reports.

use std::collections::VecDeque;

use candle::Result;

use super::super::size_class::SizeClass;
use super::summary::HeadRow;

/// The gaps of each kind whose claims a peak share is taken over.
pub(crate) const SHARE_WINDOW: usize = 32;

/// One value per size class, in ladder order.
pub type PerClass<T> = [T; SizeClass::COUNT];

/// Which forwards a gap's blocks came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapKind {
    Prefill,
    Decode,
}

/// The recent shares of claims each class took, per gap kind.
#[derive(Debug, Clone, Default)]
pub(crate) struct PeakShares {
    prefill: VecDeque<PerClass<f32>>,
    decode: VecDeque<PerClass<f32>>,
}

impl PeakShares {
    fn window(&self, kind: GapKind) -> &VecDeque<PerClass<f32>> {
        match kind {
            GapKind::Prefill => &self.prefill,
            GapKind::Decode => &self.decode,
        }
    }

    /// Record one gap's claims per class. A gap that claimed nothing says
    /// nothing about the split and is not recorded.
    pub(crate) fn record(&mut self, kind: GapKind, claims: &PerClass<usize>) {
        let total: usize = claims.iter().sum();
        if total == 0 {
            return;
        }
        let share = claims.map(|c| c as f32 / total as f32);
        let window = match kind {
            GapKind::Prefill => &mut self.prefill,
            GapKind::Decode => &mut self.decode,
        };
        if window.len() == SHARE_WINDOW {
            window.pop_front();
        }
        window.push_back(share);
    }

    /// Each class's largest share over the recorded window, or `None` before
    /// any gap of `kind` has reported.
    pub(crate) fn peak(&self, kind: GapKind) -> Option<PerClass<f32>> {
        let window = self.window(kind);
        if window.is_empty() {
            return None;
        }
        let mut peak = [0f32; SizeClass::COUNT];
        for share in window {
            for (p, s) in peak.iter_mut().zip(share) {
                *p = p.max(*s);
            }
        }
        Some(peak)
    }
}

/// A gap's demand per class, in two parts that differ in what ground they may
/// take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GapDemand {
    /// Measured demand, and a cold gap's even split: leased from free ground
    /// first and, past it, from ground bought from the weight side.
    pub firm: PerClass<usize>,
    /// A cold gap's guess past its split: leased from free ground only, never
    /// bought — no expert is evicted for slots that are likely returned.
    pub spare: PerClass<usize>,
}

impl GapDemand {
    /// Both parts of a class.
    pub(crate) fn total(&self, c: usize) -> usize {
        self.firm[c] + self.spare[c]
    }
}

/// The slots each class leases for `new_bands` bands of blocks no summary has
/// seen: `new_bands × peak share`, rounded up and capped at `new_bands`, for
/// every class in `reachable`; zero for a class the level never selects. All
/// of it firm.
///
/// With no `peak` — the first gap of its kind — the split is unknown, and real
/// K/V concentrates in a class or two, so an even split alone defers most of
/// the first gap: a prompt's worth of K/V left native for a gap more. Each
/// reachable class leases its even part firm and the rest of the count spare,
/// so the gap lands whole wherever free ground covers the guess, buys only one
/// count's worth in all, and costs only slots returned at its catch-up, once.
pub(crate) fn new_block_demand(
    new_bands: usize,
    peak: Option<&PerClass<f32>>,
    reachable: &PerClass<bool>,
) -> GapDemand {
    let mut demand = GapDemand {
        firm: [0; SizeClass::COUNT],
        spare: [0; SizeClass::COUNT],
    };
    if new_bands == 0 {
        return demand;
    }
    let n_reachable = reachable.iter().filter(|&&r| r).count();
    for c in 0..SizeClass::COUNT {
        if !reachable[c] {
            continue;
        }
        match peak {
            Some(peak) => {
                let share = (new_bands as f64 * f64::from(peak[c])).ceil() as usize;
                demand.firm[c] = share.min(new_bands);
            }
            None => {
                let split = new_bands.div_ceil(n_reachable);
                demand.firm[c] = split;
                demand.spare[c] = new_bands - split;
            }
        }
    }
    demand
}

/// A gap's lease, per class: the retried blocks' exact demand, firm, plus the
/// new blocks' demand.
pub(crate) fn gap_demand(
    retried: &PerClass<usize>,
    new_bands: usize,
    peak: Option<&PerClass<f32>>,
    reachable: &PerClass<bool>,
) -> GapDemand {
    let mut demand = new_block_demand(new_bands, peak, reachable);
    for (f, r) in demand.firm.iter_mut().zip(retried) {
        *f += r;
    }
    demand
}

/// The bands of `rows` counted by the class each claimed from, through
/// `class_of_fmt` (a record format byte's class). Refuses a band whose format
/// byte has no class: the row is not one the claim wrote against these tables.
pub(crate) fn count_bands<'r>(
    rows: impl IntoIterator<Item = &'r HeadRow>,
    class_of_fmt: impl Fn(u8) -> Option<usize>,
) -> Result<PerClass<usize>> {
    let mut counts = [0usize; SizeClass::COUNT];
    for row in rows {
        for &fmt in row.k_fmt.iter().chain(&row.v_fmt) {
            let class = class_of_fmt(fmt)
                .filter(|&c| c < SizeClass::COUNT)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("gap demand: record format {fmt} has no class"))
                })?;
            counts[class] += 1;
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = SizeClass::COUNT;

    fn only(classes: &[usize]) -> PerClass<bool> {
        let mut r = [false; N];
        for &c in classes {
            r[c] = true;
        }
        r
    }

    fn claims(pairs: &[(usize, usize)]) -> PerClass<usize> {
        let mut c = [0; N];
        for &(class, n) in pairs {
            c[class] = n;
        }
        c
    }

    #[test]
    fn before_any_summary_each_class_leases_its_split_firm_and_the_rest_spare() {
        let d = new_block_demand(10, None, &only(&[0, 2, 3]));
        assert_eq!(d.firm, claims(&[(0, 4), (2, 4), (3, 4)]));
        assert_eq!(d.spare, claims(&[(0, 6), (2, 6), (3, 6)]));
        for c in [0, 2, 3] {
            assert_eq!(d.total(c), 10, "class {c}: the whole count in all");
        }
        // The firm parts cover the count: what is bought is one count's worth.
        assert!(d.firm.iter().sum::<usize>() >= 10);
    }

    #[test]
    fn a_cold_gap_with_one_reachable_class_leases_it_all_firm() {
        let d = new_block_demand(10, None, &only(&[5]));
        assert_eq!((d.firm[5], d.spare[5]), (10, 0));
    }

    #[test]
    fn new_bands_follow_the_recorded_peak_share() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &claims(&[(1, 3), (2, 1)]));
        let peak = shares.peak(GapKind::Decode).unwrap();
        let d = new_block_demand(100, Some(&peak), &only(&[1, 2]));
        assert_eq!(d.firm, claims(&[(1, 75), (2, 25)]));
        assert_eq!(d.spare, [0; N], "measured demand is all firm");
    }

    #[test]
    fn a_class_never_takes_more_than_the_new_band_count() {
        let mut peak = [0f32; N];
        peak[0] = 1.0;
        assert_eq!(new_block_demand(40, Some(&peak), &only(&[0])).firm[0], 40);
        // A peak share over one is still the count.
        peak[0] = 1.5;
        assert_eq!(new_block_demand(40, Some(&peak), &only(&[0])).firm[0], 40);
    }

    #[test]
    fn a_share_never_rounds_below_its_part_of_the_count() {
        for bands in [3usize, 300, 301, 65_535 * 64] {
            let mut shares = PeakShares::default();
            shares.record(GapKind::Decode, &claims(&[(0, 1), (1, 1), (2, 1)]));
            let peak = shares.peak(GapKind::Decode).unwrap();
            let d = new_block_demand(bands, Some(&peak), &only(&[0, 1, 2]));
            for (c, &n) in d.firm.iter().take(3).enumerate() {
                assert!(n * 3 >= bands, "bands {bands} class {c}: {n}");
                assert!(n <= bands);
            }
        }
    }

    #[test]
    fn a_reachable_class_with_no_recorded_share_leases_nothing_new() {
        let mut peak = [0f32; N];
        peak[0] = 1.0;
        let d = new_block_demand(50, Some(&peak), &only(&[0, 1]));
        assert_eq!((d.firm[0], d.firm[1]), (50, 0));
    }

    #[test]
    fn no_bands_or_no_reachable_class_leases_nothing() {
        for d in [
            new_block_demand(0, None, &only(&[0, 1])),
            new_block_demand(100, None, &[false; N]),
        ] {
            assert_eq!((d.firm, d.spare), ([0; N], [0; N]));
        }
    }

    #[test]
    fn retried_blocks_lease_exactly_their_demand_on_top_firm() {
        let retried = claims(&[(1, 7), (4, 3)]);
        let d = gap_demand(&retried, 0, None, &only(&[1, 4]));
        assert_eq!(d.firm, retried, "no new blocks: exactly the retried demand");
        assert_eq!(d.spare, [0; N]);
        let d = gap_demand(&retried, 8, None, &only(&[1, 4]));
        assert_eq!(d.firm, claims(&[(1, 11), (4, 7)]));
        assert_eq!(d.spare, claims(&[(1, 4), (4, 4)]));
    }

    #[test]
    fn a_retried_class_the_level_does_not_reach_still_gets_its_slots() {
        // A block deferred under one level, retried after the level changed:
        // its exact demand stands whatever the new level reaches.
        let d = gap_demand(&claims(&[(5, 2)]), 4, None, &only(&[0]));
        assert_eq!((d.firm[0], d.firm[5]), (4, 2));
    }

    #[test]
    fn a_share_recorded_for_a_class_the_level_no_longer_reaches_leases_nothing_new() {
        // Shares recorded under a deeper level; the level now reaches only 0.
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &claims(&[(0, 1), (6, 3)]));
        let peak = shares.peak(GapKind::Decode).unwrap();
        let d = new_block_demand(40, Some(&peak), &only(&[0]));
        assert_eq!((d.firm[0], d.firm[6]), (10, 0));
    }

    #[test]
    fn peaks_over_several_gaps_may_sum_past_one() {
        // Each class's peak comes from a different gap: the lease covers either
        // split, at the cost of slots returned within the gap.
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &claims(&[(0, 9), (1, 1)]));
        shares.record(GapKind::Decode, &claims(&[(0, 1), (1, 9)]));
        let peak = shares.peak(GapKind::Decode).unwrap();
        let d = new_block_demand(100, Some(&peak), &only(&[0, 1]));
        assert_eq!((d.firm[0], d.firm[1]), (90, 90));
    }

    #[test]
    fn prefill_and_decode_windows_fill_and_forget_independently() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Prefill, &claims(&[(2, 1)]));
        for _ in 0..SHARE_WINDOW + 5 {
            shares.record(GapKind::Decode, &claims(&[(3, 1)]));
        }
        assert_eq!(shares.peak(GapKind::Prefill).unwrap()[2], 1.0);
        assert_eq!(shares.peak(GapKind::Decode).unwrap()[2], 0.0);
    }

    #[test]
    fn the_peak_is_the_largest_share_in_the_window_per_class() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Prefill, &claims(&[(0, 1), (1, 3)]));
        shares.record(GapKind::Prefill, &claims(&[(0, 1), (1, 1)]));
        let peak = shares.peak(GapKind::Prefill).unwrap();
        assert_eq!((peak[0], peak[1]), (0.5, 0.75));
        assert!(shares.peak(GapKind::Decode).is_none());
    }

    #[test]
    fn the_window_keeps_its_last_gaps_and_forgets_the_oldest() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &claims(&[(0, 1)]));
        for _ in 1..SHARE_WINDOW {
            shares.record(GapKind::Decode, &claims(&[(1, 1)]));
        }
        assert_eq!(shares.peak(GapKind::Decode).unwrap()[0], 1.0, "still in");
        shares.record(GapKind::Decode, &claims(&[(1, 1)]));
        let peak = shares.peak(GapKind::Decode).unwrap();
        assert_eq!((peak[0], peak[1]), (0.0, 1.0), "pushed out");
    }

    #[test]
    fn a_gap_that_claimed_nothing_is_not_recorded() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &[0; N]);
        assert!(shares.peak(GapKind::Decode).is_none());
    }

    fn row(k: [u8; 4], v: [u8; 4]) -> HeadRow {
        HeadRow {
            k_pal: Vec::new(),
            v_pal: Vec::new(),
            k_fmt: k,
            v_fmt: v,
            k_scale: [1.0; 4],
            v_scale: [1.0; 4],
            k_ptr: [0; 4],
            v_ptr: [0; 4],
            lease_idx: [0; 8],
        }
    }

    #[test]
    fn a_rows_bands_are_counted_by_their_formats_class() {
        // Format bytes 7 and 9 in class 2, 15 in class 0.
        let class = |f: u8| match f {
            7 | 9 => Some(2),
            15 => Some(0),
            _ => None,
        };
        let rows = [row([7, 7, 15, 9], [15, 15, 15, 7]), row([9; 4], [7; 4])];
        let c = count_bands(&rows, class).unwrap();
        assert_eq!(c, claims(&[(2, 12), (0, 4)]));
        assert_eq!(c.iter().sum::<usize>(), 16, "every band counted once");
    }

    #[test]
    fn a_band_whose_format_has_no_class_is_refused() {
        let class = |f: u8| (f == 7).then_some(1);
        assert!(count_bands(&[row([7; 4], [7, 7, 7, 3])], class).is_err());
        // A class past the ladder is no class either.
        assert!(count_bands(&[row([7; 4], [7; 4])], |_| Some(N)).is_err());
    }

    #[test]
    fn no_rows_count_nothing() {
        assert_eq!(count_bands(&[], |_| Some(0)).unwrap(), [0; N]);
    }
}
