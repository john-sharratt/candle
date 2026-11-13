//! The slots each class holds for the gap's claim, and how many it should hold
//! (`docs/kv_quantize_in_the_gap.md` §3.2).
//!
//! **Sizing.** The host built the gap's table, so it knows exactly how many
//! bands the gap can claim: blocks × heads × `N_PALETTE` × 2. Only the split
//! across classes is unknown until the device has chosen. Each class's lease
//! is topped up to that count times the class's recent peak share of claims,
//! plus one arena's headroom, and never more than the count itself. Before any
//! gap has reported its claims the count is split evenly across the reachable
//! classes. The share is a fraction, not a volume, so a prefill arriving after
//! an idle stretch is leased for its own size, and it is kept apart for prefill
//! and decode gaps, whose K/V compresses differently.
//!
//! **Holding.** A lease owns its slots — taken out of the pool before the gap
//! that may write them is launched, so nothing else can hand them out
//! (§3.8 rule 3). The device sees them as one stack of addresses, one run per
//! class, uploaded whole each gap. A slot leaves the lease only at the
//! catch-up: into the block the summary names it for, by the index the claim
//! gave it.

use std::collections::VecDeque;

use candle::Result;

use super::super::size_class::SizeClass;

/// The gaps of each kind whose claims a peak share is taken over.
pub(crate) const SHARE_WINDOW: usize = 32;

/// Which forwards a gap's blocks came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GapKind {
    Prefill,
    Decode,
}

/// One value per size class, in ladder order.
pub(crate) type PerClass<T> = [T; SizeClass::COUNT];

/// The recent shares of claims each class took, per gap kind.
#[derive(Debug, Clone, Default)]
pub(crate) struct PeakShares {
    prefill: VecDeque<PerClass<f32>>,
    decode: VecDeque<PerClass<f32>>,
}

impl PeakShares {
    fn window(&mut self, kind: GapKind) -> &mut VecDeque<PerClass<f32>> {
        match kind {
            GapKind::Prefill => &mut self.prefill,
            GapKind::Decode => &mut self.decode,
        }
    }

    /// Record one installed gap's claims per class. A gap that claimed
    /// nothing says nothing about the split and is not recorded.
    pub(crate) fn record(&mut self, kind: GapKind, claims: &PerClass<usize>) {
        let total: usize = claims.iter().sum();
        if total == 0 {
            return;
        }
        let share = claims.map(|c| c as f32 / total as f32);
        let window = self.window(kind);
        if window.len() == SHARE_WINDOW {
            window.pop_front();
        }
        window.push_back(share);
    }

    /// Each class's largest share over the recorded window, or `None` before
    /// any gap of `kind` has reported.
    pub(crate) fn peak(&self, kind: GapKind) -> Option<PerClass<f32>> {
        let window = match kind {
            GapKind::Prefill => &self.prefill,
            GapKind::Decode => &self.decode,
        };
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

/// The slots each class's lease should hold for a gap of `bands` bands:
/// `bands × peak share + headroom`, capped at `bands`, for every class in
/// `reachable`; zero for a class the level never selects. With no `peak`,
/// `bands` split evenly (rounded up) across the reachable classes, plus their
/// headroom, capped the same way.
pub(crate) fn lease_targets(
    bands: usize,
    peak: Option<&PerClass<f32>>,
    reachable: &PerClass<bool>,
    headroom: &PerClass<usize>,
) -> PerClass<usize> {
    let mut targets = [0usize; SizeClass::COUNT];
    if bands == 0 {
        return targets;
    }
    let n_reachable = reachable.iter().filter(|&&r| r).count();
    if n_reachable == 0 {
        return targets;
    }
    let even = bands.div_ceil(n_reachable);
    for c in 0..SizeClass::COUNT {
        if !reachable[c] {
            continue;
        }
        let base = match peak {
            Some(peak) => (bands as f64 * f64::from(peak[c])).ceil() as usize,
            None => even,
        };
        targets[c] = (base + headroom[c]).min(bands);
    }
    targets
}

/// The slots to add to a lease that holds `held` to reach `target`: a lease
/// is only ever topped up, and a slot it holds past its target stays for the
/// next gap.
pub(crate) fn top_up(held: usize, target: usize) -> usize {
    target.saturating_sub(held)
}

/// The lease as the device reads it: every slot's address, class by class,
/// with each class's first entry and length.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StackImage {
    pub addrs: Vec<u64>,
    pub off: PerClass<u32>,
    pub len: PerClass<u32>,
}

/// The slots held for the gap's claim, per class, in stack order. `T` is the
/// slot's owning handle — a `ChunkGid` in production, which returns the slot to
/// its pool when dropped.
#[derive(Debug)]
pub(crate) struct Lease<T> {
    slots: PerClass<Vec<T>>,
}

impl<T> Default for Lease<T> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl<T> Lease<T> {
    /// The slots each class holds.
    pub(crate) fn held(&self) -> PerClass<usize> {
        std::array::from_fn(|c| self.slots[c].len())
    }

    /// Add `slots` to class `class`'s run, after the ones it holds.
    pub(crate) fn add(&mut self, class: usize, slots: impl IntoIterator<Item = T>) {
        self.slots[class].extend(slots);
    }

    /// The stack image the claim reads, each slot's address from `addr`.
    pub(crate) fn stack_image(
        &self,
        mut addr: impl FnMut(&T) -> Result<u64>,
    ) -> Result<StackImage> {
        let mut image = StackImage::default();
        for (c, run) in self.slots.iter().enumerate() {
            image.off[c] = u32::try_from(image.addrs.len())
                .map_err(|_| candle::Error::Msg("gap lease: stack past u32".into()))?;
            image.len[c] = u32::try_from(run.len())
                .map_err(|_| candle::Error::Msg("gap lease: class run past u32".into()))?;
            for slot in run {
                image.addrs.push(addr(slot)?);
            }
        }
        Ok(image)
    }

    /// Take class `class`'s slots at `indices` — positions in the run as the
    /// last stack image laid it out — out of the lease, in the order named.
    /// The slots left keep their order. Refuses an index past the run or one
    /// named twice, taking nothing: either is a summary that does not match
    /// the lease it was claimed against.
    pub(crate) fn take(&mut self, class: usize, indices: &[u32]) -> Result<Vec<T>> {
        let run = &mut self.slots[class];
        let mut named = vec![false; run.len()];
        for &i in indices {
            let i = i as usize;
            if i >= run.len() || std::mem::replace(&mut named[i], true) {
                candle::bail!(
                    "gap lease: class {class} index {i} is past its {} slots or named twice",
                    run.len()
                );
            }
        }
        let mut taken: Vec<Option<T>> = std::iter::repeat_with(|| None).take(run.len()).collect();
        let mut kept = Vec::with_capacity(run.len() - indices.len());
        for (i, slot) in run.drain(..).enumerate() {
            if named[i] {
                taken[i] = Some(slot);
            } else {
                kept.push(slot);
            }
        }
        *run = kept;
        Ok(indices
            .iter()
            .map(|&i| taken[i as usize].take().expect("named exactly once"))
            .collect())
    }
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

    #[test]
    fn before_any_summary_the_count_is_split_evenly() {
        let t = lease_targets(10, None, &only(&[0, 2, 3]), &[0; N]);
        let mut want = [0; N];
        want[0] = 4;
        want[2] = 4;
        want[3] = 4;
        assert_eq!(t, want);
    }

    #[test]
    fn the_split_follows_the_recorded_peak_share_plus_headroom() {
        let mut shares = PeakShares::default();
        let mut claims = [0; N];
        claims[1] = 3;
        claims[2] = 1;
        shares.record(GapKind::Decode, &claims);
        let peak = shares.peak(GapKind::Decode).unwrap();
        let mut headroom = [0; N];
        headroom[1] = 8;
        headroom[2] = 8;
        let t = lease_targets(100, Some(&peak), &only(&[1, 2]), &headroom);
        assert_eq!(t[1], 75 + 8);
        assert_eq!(t[2], 25 + 8);
        assert_eq!(t[0], 0);
    }

    #[test]
    fn a_lease_is_never_more_than_the_gaps_band_count() {
        let mut headroom = [0; N];
        headroom[0] = 512;
        let mut peak = [0f32; N];
        peak[0] = 1.0;
        let t = lease_targets(40, Some(&peak), &only(&[0]), &headroom);
        assert_eq!(t[0], 40);
    }

    #[test]
    fn an_empty_table_leases_nothing() {
        assert_eq!(lease_targets(0, None, &only(&[0, 1]), &[64; N]), [0; N]);
    }

    #[test]
    fn the_peak_is_the_largest_share_in_the_window_per_class() {
        let mut shares = PeakShares::default();
        let mut a = [0; N];
        a[0] = 1;
        a[1] = 3;
        let mut b = [0; N];
        b[0] = 1;
        b[1] = 1;
        shares.record(GapKind::Prefill, &a);
        shares.record(GapKind::Prefill, &b);
        let peak = shares.peak(GapKind::Prefill).unwrap();
        assert_eq!(peak[0], 0.5);
        assert_eq!(peak[1], 0.75);
        assert!(shares.peak(GapKind::Decode).is_none());
    }

    #[test]
    fn the_window_forgets_its_oldest_gap() {
        let mut shares = PeakShares::default();
        let mut heavy = [0; N];
        heavy[0] = 1;
        shares.record(GapKind::Decode, &heavy);
        let mut light = [0; N];
        light[1] = 1;
        for _ in 0..SHARE_WINDOW {
            shares.record(GapKind::Decode, &light);
        }
        let peak = shares.peak(GapKind::Decode).unwrap();
        assert_eq!(peak[0], 0.0);
        assert_eq!(peak[1], 1.0);
    }

    #[test]
    fn a_gap_that_claimed_nothing_is_not_recorded() {
        let mut shares = PeakShares::default();
        shares.record(GapKind::Decode, &[0; N]);
        assert!(shares.peak(GapKind::Decode).is_none());
    }

    #[test]
    fn a_lease_is_only_topped_up() {
        assert_eq!(top_up(3, 10), 7);
        assert_eq!(top_up(12, 10), 0);
    }

    /// Slots as plain addresses, so the stack is checked without a device.
    fn lease(runs: &[(usize, &[u64])]) -> Lease<u64> {
        let mut l = Lease::default();
        for &(c, slots) in runs {
            l.add(c, slots.iter().copied());
        }
        l
    }

    #[test]
    fn the_stack_lays_each_class_out_as_one_run_in_ladder_order() {
        let l = lease(&[(2, &[0x200, 0x210]), (0, &[0x100]), (2, &[0x220])]);
        let img = l.stack_image(|&a| Ok(a)).unwrap();
        assert_eq!(img.addrs, vec![0x100, 0x200, 0x210, 0x220]);
        assert_eq!(img.off[0], 0);
        assert_eq!(img.len[0], 1);
        assert_eq!(img.off[1], 1);
        assert_eq!(img.len[1], 0);
        assert_eq!(img.off[2], 1);
        assert_eq!(img.len[2], 3);
        assert_eq!(img.off[3], 4);
        assert_eq!(l.held()[2], 3);
    }

    #[test]
    fn a_take_returns_slots_in_the_order_named_and_keeps_the_rest_in_order() {
        let mut l = lease(&[(1, &[10, 11, 12, 13, 14])]);
        assert_eq!(l.take(1, &[3, 0]).unwrap(), vec![13, 10]);
        let img = l.stack_image(|&a| Ok(a)).unwrap();
        assert_eq!(img.addrs, vec![11, 12, 14]);
    }

    #[test]
    fn a_take_naming_a_slot_twice_or_past_the_run_takes_nothing() {
        let mut l = lease(&[(0, &[1, 2, 3])]);
        assert!(l.take(0, &[1, 1]).is_err());
        assert!(l.take(0, &[3]).is_err());
        assert_eq!(l.held()[0], 3);
    }
}
