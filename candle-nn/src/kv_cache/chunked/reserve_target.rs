//! **How many free slots each reserve class holds** — the R16-sized reserve of
//! `docs/kv_drain_under_pressure.md` §2a.
//!
//! Hot KV is written native (R16/F16), and the hot→warm migrate quantizes each
//! native chunk slot into a slot of the class its block was chosen for, plus a
//! record. So the live native slots are the backlog the reserve serves, and the
//! split of that backlog across the quant classes is read from what VRAM holds
//! now: each band class's share of the live quant band slots, and each record
//! class's live slots over the same total.
//!
//! ```text
//! factor(c) = live(c) / Σ live(quant band classes)
//! target(c) = clamp(ceil(pending × factor(c)), floor = one arena,
//!                   cap = RESERVE_CAP_PERCENT of the KV ground × factor(c))
//! ```
//!
//! Every term is a counter the pools already keep, so this is arithmetic over a
//! handful of numbers and nothing here allocates or scans.

use super::arena::{ArenaKey, ArenaKind};

/// The share of the KV ground the reserve may hold in free slots, so a burst
/// of native backlog never hands the whole card to empty arenas.
pub(super) const RESERVE_CAP_PERCENT: usize = 25;

/// One reserve class's target and the figures it came from, for the
/// `kv_reserve` log line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ReserveTarget {
    pub key: ArenaKey,
    /// Free slots the class holds: never below one arena's worth.
    pub slots: usize,
    /// The class's live slots over all live quant band slots.
    pub factor: f64,
    /// The backlog asked for more than the class's share of the cap.
    pub capped: bool,
}

/// The target of each quant class in `classes` — `(key, live slots)`, band and
/// record keys alike — for `pending` live native slots, with the reserve held
/// to `cap_bytes` of free slots split by factor.
///
/// With no quant band slot live there is no split to read, and every class sits
/// at its one-arena floor.
pub(super) fn reserve_targets(
    pending: usize,
    classes: &[(ArenaKey, usize)],
    cap_bytes: usize,
) -> Vec<ReserveTarget> {
    let band_total: usize = classes
        .iter()
        .filter(|(key, _)| matches!(key.kind, ArenaKind::Band))
        .map(|&(_, live)| live)
        .sum();
    classes
        .iter()
        .map(|&(key, live)| {
            let floor = key.chunks();
            if band_total == 0 {
                return ReserveTarget {
                    key,
                    slots: floor,
                    factor: 0.0,
                    capped: false,
                };
            }
            let share = |n: usize| (n as u128 * live as u128).div_ceil(band_total as u128);
            let wanted = share(pending) as usize;
            let cap = (share(cap_bytes) / key.slot_stride().max(1) as u128) as usize;
            let capped = wanted > cap.max(floor);
            ReserveTarget {
                key,
                slots: wanted.min(cap).max(floor),
                factor: live as f64 / band_total as f64,
                capped,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::arena_table::ArenaLocation;
    use crate::kv_cache::chunked::size_class::SizeClass;

    fn band(class: usize) -> ArenaKey {
        ArenaKey::new(SizeClass::at(class), ArenaLocation::Gpu)
    }

    /// **The backlog splits by the live quant mix**: two band classes holding
    /// three and one parts of the live slots take three and one parts of the
    /// backlog, and a record class rides at its live slots over the band total.
    #[test]
    fn the_backlog_splits_by_the_live_quant_mix() {
        let a = band(2);
        let b = band(4);
        let rec = ArenaKey::for_record_stride(ArenaLocation::Gpu, 1536);
        let (fa, fb, fr) = (a.chunks(), b.chunks(), rec.chunks());
        let pending = 40 * fa.max(fb).max(fr);
        let t = reserve_targets(
            pending,
            &[(a, 3_000), (b, 1_000), (rec, 2_000)],
            usize::MAX / 2,
        );
        assert_eq!(t[0].slots, (pending * 3).div_ceil(4));
        assert_eq!(t[1].slots, pending.div_ceil(4));
        assert_eq!(t[2].slots, pending.div_ceil(2));
        assert_eq!((t[0].factor, t[1].factor, t[2].factor), (0.75, 0.25, 0.5));
        assert!(t.iter().all(|t| !t.capped));
    }

    /// **No class goes below one arena**, whatever its share: a class with
    /// nothing live and a backlog of one slot both sit at the floor.
    #[test]
    fn no_class_goes_below_one_arena() {
        let a = band(2);
        let b = band(4);
        let t = reserve_targets(1, &[(a, 10), (b, 0)], usize::MAX / 2);
        assert_eq!(t[0].slots, a.chunks());
        assert_eq!(t[1].slots, b.chunks());
    }

    /// **With no quant slot live, every class is at its floor** — the cold
    /// start, which is the one-arena reserve.
    #[test]
    fn a_cold_start_holds_every_class_at_its_floor() {
        let a = band(2);
        let rec = ArenaKey::for_record_stride(ArenaLocation::Gpu, 1536);
        let t = reserve_targets(1_000_000, &[(a, 0), (rec, 0)], usize::MAX / 2);
        assert_eq!(t[0].slots, a.chunks());
        assert_eq!(t[1].slots, rec.chunks());
        assert!(t.iter().all(|t| t.factor == 0.0 && !t.capped));
    }

    /// **The cap is split by factor, in each class's own slot bytes**: one
    /// class with all the live slots and a backlog far past the cap holds the
    /// cap's bytes in its slots, and says it was capped.
    #[test]
    fn the_cap_holds_a_burst_to_its_share_of_the_ground() {
        let a = band(2);
        let cap_bytes = 10 * a.chunks() * a.slot_stride();
        let t = reserve_targets(1_000 * a.chunks(), &[(a, 5)], cap_bytes);
        assert_eq!(t[0].slots, 10 * a.chunks());
        assert!(t[0].capped);
    }
}
