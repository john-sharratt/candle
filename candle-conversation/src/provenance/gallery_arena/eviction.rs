//! Which resident turns the gallery gives up to get back under its ceiling.
//!
//! The decision only — the arena gathers the candidates and drops the runs.
//! Kept apart so the rule tests without a device.
//!
//! # The ceiling bounds stale turns, never a working set
//!
//! Every reprojection scans several belief groups, one after another, and each
//! scan's working set is every turn in its group's scope. When the working sets
//! together outgrow the ceiling, a byte cap enforced at every admission evicts
//! the least-recently-used turns — which are exactly the turns the *previous*
//! group's scan just used and the next reprojection will need again. The arena
//! then re-uploads most of the corpus every reprojection, and every upload bumps
//! the residency generation, so every scan's cached index is rebuilt too.
//! Measured on the live daemon at 1,143 ingested files: a 1.44 GB working set
//! against a 512 MiB ceiling, and a belief scan of 5.1–5.6 s per reprojection
//! where the same scan took 6–10 ms on a corpus that fit.
//!
//! So a turn any scan has used within [`RECENT_USE`] is not a candidate. What
//! the ceiling still bounds is the corpus nobody is scanning — turns left behind
//! by a conversation that went quiet, or a scope that moved on.
//!
//! **Pressure relief comes through here too.** The scheduler's VRAM relief
//! enforces the same ceiling (`GalleryArena::evict_to_cap`). It used to shed by
//! plain LRU, which on the live daemon threw out 892 MiB of a dialogue's working
//! set on every decode relief — to no effect, since scattered pages return no
//! region (`relieved=false` every time) — and the next scan uploaded it again.

use std::time::Duration;

/// How long after its last scan a turn stays exempt from the ceiling.
///
/// Long enough to span a user reading a reply and writing the next message, so
/// a conversation's working set survives the pause between its turns; short
/// enough that a conversation abandoned for the afternoon gives its pages back.
/// A turn evicted after that costs one re-upload on the scan that next wants it.
pub const RECENT_USE: Duration = Duration::from_secs(10 * 60);

/// One resident turn, as the ceiling sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resident {
    /// Admission order: lower is older.
    pub lru: u64,
    /// Time since a scan last used it.
    pub idle: Duration,
    /// Page bytes its run holds.
    pub bytes: u64,
    /// Read by an in-flight scan right now.
    pub pinned: bool,
}

/// Indices into `turns` to evict, oldest first, so that `held` falls to `cap` —
/// taking only turns that are neither pinned nor used within `recent`.
///
/// Returns fewer than needed when the rest is in use: the working set is then
/// larger than the ceiling, and it is served rather than thrashed.
pub fn over_cap(turns: &[Resident], held: u64, cap: u64, recent: Duration) -> Vec<usize> {
    if held <= cap {
        return Vec::new();
    }
    let mut stale: Vec<usize> = (0..turns.len())
        .filter(|&i| !turns[i].pinned && turns[i].idle >= recent)
        .collect();
    stale.sort_by_key(|&i| turns[i].lru);
    let mut freed = 0u64;
    let want = held - cap;
    stale
        .into_iter()
        .take_while(|&i| {
            let more = freed < want;
            freed += turns[i].bytes;
            more
        })
        .collect()
}

/// Which cached scan indices to drop, oldest first, so that one more of
/// `incoming` bytes fits inside `cap_entries` entries and `cap_bytes` bytes.
/// `entries` are `(fingerprint, last use, device bytes)`; the returned
/// fingerprints are the ones to remove.
///
/// Unlike [`over_cap`] nothing here is exempt: an index is only a cache of work
/// the next scan can redo, and the pages it points at stay resident either way.
pub fn index_cache_evictions(
    entries: &[(u64, u64, u64)],
    incoming: u64,
    cap_entries: usize,
    cap_bytes: u64,
) -> Vec<u64> {
    let mut by_age: Vec<&(u64, u64, u64)> = entries.iter().collect();
    by_age.sort_by_key(|&&(_, used, _)| used);
    let mut count = entries.len();
    let mut held: u64 = entries.iter().map(|&(_, _, b)| b).sum();
    let mut out = Vec::new();
    for &&(fp, _, bytes) in &by_age {
        if count < cap_entries && held + incoming <= cap_bytes {
            break;
        }
        out.push(fp);
        count -= 1;
        held -= bytes;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// Within both bounds nothing goes.
    #[test]
    fn an_index_that_fits_evicts_nothing() {
        let entries = [(1, 5, 10 * MIB), (2, 6, 10 * MIB)];
        assert!(index_cache_evictions(&entries, 10 * MIB, 16, 256 * MIB).is_empty());
    }

    /// Over the byte bound, the least recently used go first, and only until
    /// the newcomer fits.
    #[test]
    fn bytes_are_freed_oldest_first_until_the_newcomer_fits() {
        let entries = [(1, 9, 100 * MIB), (2, 3, 100 * MIB), (3, 6, 50 * MIB)];
        // 250 held + 60 incoming against 256: drop the oldest (2, used 3) and
        // stop — 150 + 60 fits.
        assert_eq!(
            index_cache_evictions(&entries, 60 * MIB, 16, 256 * MIB),
            vec![2]
        );
    }

    /// At the entry bound one goes, however small everything is.
    #[test]
    fn the_entry_bound_makes_room_for_one() {
        let entries = [(7, 2, MIB), (8, 1, MIB), (9, 3, MIB)];
        assert_eq!(index_cache_evictions(&entries, MIB, 3, 256 * MIB), vec![8]);
    }

    fn turn(lru: u64, idle_s: u64, mib: u64, pinned: bool) -> Resident {
        Resident {
            lru,
            idle: Duration::from_secs(idle_s),
            bytes: mib * MIB,
            pinned,
        }
    }

    #[test]
    fn under_the_cap_nothing_goes() {
        let turns = [turn(0, 3600, 100, false)];
        assert!(over_cap(&turns, 100 * MIB, 512 * MIB, RECENT_USE).is_empty());
    }

    /// **The regression.** Two groups' working sets, both used in the last
    /// reprojection, together past the cap: neither is evicted.
    #[test]
    fn a_working_set_past_the_cap_is_kept_whole() {
        let turns = [
            turn(0, 2, 700, false), // code_reading's scan, two seconds ago
            turn(1, 1, 740, false), // repo_map's scan, one second ago
        ];
        assert!(over_cap(&turns, 1440 * MIB, 512 * MIB, RECENT_USE).is_empty());
    }

    /// Stale turns go oldest first, and only as many as bring `held` under.
    #[test]
    fn stale_turns_go_oldest_first_until_under() {
        let turns = [
            turn(5, 3600, 300, false),
            turn(1, 3600, 300, false),
            turn(3, 3600, 300, false),
            turn(9, 10, 300, false), // in use
        ];
        // 1200 held, 700 cap: 500 to free — the two oldest stale (lru 1, 3).
        assert_eq!(
            over_cap(&turns, 1200 * MIB, 700 * MIB, RECENT_USE),
            vec![1, 2]
        );
    }

    /// A pinned turn is never a candidate, however old.
    #[test]
    fn a_pinned_turn_is_never_evicted() {
        let turns = [turn(0, 3600, 900, true), turn(1, 3600, 100, false)];
        assert_eq!(over_cap(&turns, 1000 * MIB, 512 * MIB, RECENT_USE), vec![1]);
    }

    /// Exactly `recent` idle is stale; one second less is in use.
    #[test]
    fn the_recent_window_boundary() {
        let at = RECENT_USE.as_secs();
        let turns = [turn(0, at, 100, false), turn(1, at - 1, 100, false)];
        assert_eq!(over_cap(&turns, 200 * MIB, 50 * MIB, RECENT_USE), vec![0]);
    }
}
