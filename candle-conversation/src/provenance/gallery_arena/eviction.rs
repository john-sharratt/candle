//! Which resident turns the gallery gives up: the ones whose ground is worth
//! more to the weights than their residency is to the scans.
//!
//! The decision only — the arena gathers the candidates and drops the runs.
//! Kept apart so the rule tests without a device.
//!
//! # The trade, in one currency
//!
//! A gallery page is a device copy of host signatures (the substrate's), so an
//! evicted turn loses nothing but the time its next scan spends uploading it
//! again. Keeping a turn resident therefore saves, every time a scan uses it,
//! its upload — `bytes × upload cost per byte` — and a turn scanned every `T`
//! saves `upload cost per byte / T` per byte per second. The same ground under
//! the weights saves the expert stalls it would avoid: the weight side's
//! marginal value per byte per second: the miss stall one more slot spares,
//! measured from the expert cache's misses and the link's rate
//! (`Scheduler::update_gallery_keep`).
//! So a turn earns its ground while it is scanned more often than
//!
//! ```text
//! T* = upload cost per byte / weight value per byte per second
//! ```
//!
//! and gives it up once the longer of its scan interval and its idle time
//! passes `T*` — [`past_keep`]. With no weight pressure `T*` is unbounded and
//! everything stays; under a reader workload whose experts stream, it falls to
//! the seconds a working set's scans actually come at.
//!
//! # Why not a byte ceiling with a recency exemption
//!
//! That was the rule: a 512 MiB ceiling, and any turn scanned within ten
//! minutes exempt from it. The exemption fixed a real thrash — a 1.44 GB
//! working set re-uploaded every reprojection under a byte cap, a 10 ms scan
//! taking 5 s — and `T*` keeps that fix, since a working set scanned at every
//! reprojection is the turn most worth keeping. But under steady reading every
//! turn was recent, so the ceiling bound nothing: on WhiteWorld (2026-10-19) the
//! gallery held 3.9 GB, the weights sat at their floor, and prefill ran at
//! 267 tok/s on a 15% expert hit rate. Two constants stood where the measured
//! trade belongs.
//!
//! **Pressure relief comes through here too** (`GalleryArena::evict_past_keep`):
//! scattered pages return no region, so shedding a working set on relief only
//! made the next scan upload it again — the same rule decides there.

use std::time::Duration;

/// One resident turn, as the trade sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resident {
    /// Admission order: lower is older.
    pub lru: u64,
    /// Time since a scan last used it.
    pub idle: Duration,
    /// How long between the scans that use it, averaged; `None` until a second
    /// scan has.
    pub interval: Option<Duration>,
    /// Page bytes its run holds.
    pub bytes: u64,
    /// Read by an in-flight scan right now.
    pub pinned: bool,
}

/// Indices into `turns` whose residency is worth less than its ground — the
/// unpinned turns not scanned within `keep_within`, by the longer of their
/// interval and their idle time, oldest first. `None` keeps everything: no
/// measured weight pressure to trade against.
pub fn past_keep(turns: &[Resident], keep_within: Option<Duration>) -> Vec<usize> {
    let Some(keep) = keep_within else {
        return Vec::new();
    };
    let mut out: Vec<usize> = (0..turns.len())
        .filter(|&i| {
            let t = &turns[i];
            !t.pinned && t.interval.map_or(t.idle, |iv| iv.max(t.idle)) > keep
        })
        .collect();
    out.sort_by_key(|&i| turns[i].lru);
    out
}

/// `T*`: how often a turn must be scanned to earn its ground — the upload cost
/// per byte over the weight side's marginal value per byte per second, both in
/// microseconds. `None` when the weights lose nothing to the ground (no miss
/// measured), and the gallery keeps what it holds. So too when they lose so
/// little that `T*` is past what a `Duration` holds: a vanishing marginal value
/// divided up to overflow, and panicked the scheduler thread with it
/// (BlackWorld, 2026-10-31).
pub fn keep_within(upload_us_per_byte: f64, weight_us_per_byte_sec: f64) -> Option<Duration> {
    if !(weight_us_per_byte_sec > 0.0 && upload_us_per_byte > 0.0) {
        return None;
    }
    Duration::try_from_secs_f64(upload_us_per_byte / weight_us_per_byte_sec).ok()
}

/// Which cached scan indices go before one of `incoming` bytes joins them:
/// every one no scan has stored or reused for more than `idle_ms`, then, while
/// the rest and the newcomer are over `cap_bytes`, the least recently used.
/// `entries` are `(fingerprint, last use, device bytes)`, use in milliseconds
/// on one clock; the returned fingerprints are the ones to remove.
///
/// **Idleness first, room only as a backstop** (`docs/gallery_residency.md`
/// §5). An index addresses pages, a few bytes each, so the cap holds dozens of
/// even the largest scan's; it exists because the count of distinct indices is
/// bounded by how fast readers make new segment sets, not by the gallery — a
/// gate leaving each probe's own turn out built 110 in two minutes, kept
/// them past the scans that used them, and ran the card out of memory
/// (BlackWorld, 2026-11-16). Nothing here is exempt: an index is only a cache
/// of work the next scan can redo, and the pages it names stay resident.
pub fn index_cache_evictions(
    entries: &[(u64, u64, u64)],
    incoming: u64,
    now_ms: u64,
    idle_ms: u64,
    cap_bytes: u64,
) -> Vec<u64> {
    let mut out = Vec::new();
    let mut live: Vec<&(u64, u64, u64)> = Vec::new();
    for e in entries {
        if now_ms.saturating_sub(e.1) > idle_ms {
            out.push(e.0);
        } else {
            live.push(e);
        }
    }
    live.sort_by_key(|&&(_, used, _)| used);
    let mut held: u64 = live.iter().map(|&&(_, _, b)| b).sum();
    for &&(fp, _, bytes) in &live {
        if held + incoming <= cap_bytes {
            break;
        }
        out.push(fp);
        held -= bytes;
    }
    out
}

/// The share of a card's memory the cached scan indices may hold together:
/// one forty-eighth.
const INDEX_CACHE_SHARE: u64 = 48;

/// The least the cached scan indices may hold, whatever the card.
const INDEX_CACHE_FLOOR: u64 = 256 << 20;

/// Device bytes the cached scan indices may hold together on a card of
/// `total` bytes: a forty-eighth of it, never under [`INDEX_CACHE_FLOOR`] —
/// memory in the driver pool, outside the span, that nothing else accounts
/// for. ~1.5 GB on the 72 GB card, dozens of the largest scan's index.
pub fn index_cache_budget(total: u64) -> u64 {
    (total / INDEX_CACHE_SHARE).max(INDEX_CACHE_FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// **An idle index goes, however small**: one used within the limit
    /// stays, one used exactly at it too, and a cache under its cap loses
    /// nothing else.
    #[test]
    fn an_idle_index_goes_and_nothing_else_under_the_cap() {
        let entries = [
            (1, 100_000, MIB),
            (2, 39_999, MIB),
            (3, 40_000, MIB),
            (4, 0, MIB),
        ];
        assert_eq!(
            index_cache_evictions(&entries, MIB, 100_000, 60_000, 256 * MIB),
            vec![2, 4]
        );
        assert!(index_cache_evictions(&entries, MIB, 60_000, 60_000, 256 * MIB).is_empty());
        assert!(index_cache_evictions(&[], MIB, 1, 0, 256 * MIB).is_empty());
    }

    /// **Over the cap, the least recently used live ones go**, after the idle
    /// ones and only until the newcomer fits.
    #[test]
    fn over_the_cap_the_oldest_live_go_until_the_newcomer_fits() {
        let entries = [
            (1, 90, 100 * MIB),
            (2, 30, 100 * MIB),
            (3, 60, 50 * MIB),
            (4, 0, 70 * MIB),
        ];
        // At 100 with an 80 limit: 4 is idle. 250 held + 60 incoming against
        // 256: drop the oldest live (2, used 30) and stop — 150 + 60 fits.
        assert_eq!(
            index_cache_evictions(&entries, 60 * MIB, 100, 80, 256 * MIB),
            vec![4, 2]
        );
    }

    /// **A forty-eighth of the card, never under 256 MiB** — on each
    /// development card's reported total.
    #[test]
    fn the_index_cache_holds_a_share_of_the_card() {
        let gib = 1024 * MIB;
        assert_eq!(index_cache_budget(72 * gib), 1_536 * MIB);
        assert_eq!(index_cache_budget(24 * gib), 512 * MIB);
        assert_eq!(index_cache_budget(8 * gib), 256 * MIB, "the floor");
    }

    fn turn(lru: u64, idle_s: u64, interval_s: Option<u64>, pinned: bool) -> Resident {
        Resident {
            lru,
            idle: Duration::from_secs(idle_s),
            interval: interval_s.map(Duration::from_secs),
            bytes: 100 * MIB,
            pinned,
        }
    }

    /// With no weight pressure measured, nothing goes, however stale.
    #[test]
    fn without_weight_pressure_everything_stays() {
        let turns = [turn(0, 3600, None, false)];
        assert!(past_keep(&turns, None).is_empty());
    }

    /// **The regression the old exemption fixed, kept.** A working set scanned
    /// at every reprojection stays, however large, while its scans come within
    /// the threshold.
    #[test]
    fn a_working_set_scanned_within_the_threshold_stays() {
        let turns = [turn(0, 2, Some(3), false), turn(1, 1, Some(3), false)];
        assert!(past_keep(&turns, Some(Duration::from_secs(10))).is_empty());
    }

    /// A turn goes when the longer of its interval and its idle time passes the
    /// threshold — a turn scanned often but not lately goes as surely as one
    /// scanned rarely — oldest first; a pinned one never.
    #[test]
    fn turns_past_the_threshold_go_oldest_first_and_pinned_ones_stay() {
        let turns = [
            turn(5, 1, Some(60), false), // scanned rarely
            turn(1, 90, Some(2), false), // scanned often, but not lately
            turn(3, 1, Some(2), false),  // in use
            turn(0, 900, None, true),    // pinned
        ];
        assert_eq!(past_keep(&turns, Some(Duration::from_secs(30))), vec![1, 0]);
    }

    /// `T*` is the upload cost over the weight value per byte, and unbounded
    /// when the weights lose nothing.
    #[test]
    fn the_threshold_is_the_upload_over_the_weights_value() {
        assert_eq!(keep_within(1e-4, 0.0), None, "no weight pressure, no limit");
        assert_eq!(keep_within(1e-4, 1e-5), Some(Duration::from_secs(10)));
    }

    /// A marginal value too small to divide by is no limit, not a panic.
    #[test]
    fn a_vanishing_weight_pressure_keeps_everything() {
        assert_eq!(keep_within(1e-4, f64::MIN_POSITIVE), None);
        assert_eq!(keep_within(f64::MAX, 1e-300), None, "infinity");
        assert_eq!(keep_within(1e-4, f64::NAN), None);
        assert_eq!(keep_within(f64::NAN, 1e-5), None);
    }
}
