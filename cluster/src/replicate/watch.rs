//! Finding events nothing will deliver (§7.1). Every `MEMBERS` names each
//! member's `high_water`, its own highest `oseq`; an origin announcing more
//! than this member holds for [`GAP_WAIT`] gets a catch-up addressed to it,
//! which finds a lost last event even when nothing after it ever comes.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use applied_set::AppliedSet;

/// How long an origin may stay ahead before it is asked directly.
pub const GAP_WAIT: Duration = Duration::from_secs(5);

/// How often a member with a socket down catches up on the sockets still up.
pub const PERIODIC: Duration = Duration::from_secs(30);

/// Whether a periodic catch-up is due: a socket is down, and [`PERIODIC`] has
/// passed since the last one. A gateway this member cannot reach may still be
/// relaying events to others that it will not otherwise see.
pub fn periodic_due(socket_down: bool, last: Instant, now: Instant) -> bool {
    socket_down && now.duration_since(last) >= PERIODIC
}

#[derive(Debug, Default)]
pub struct GapWatch {
    /// Each origin seen ahead, and since when.
    ahead: HashMap<u64, Instant>,
    /// Each origin last said to be [`Self::blind`], by the top it was said at.
    blind: HashMap<u64, u64>,
}

/// An origin this member holds gaps of that no catch-up will ever ask for: what
/// it holds stands above what the origin announces, so the announced high reads
/// as complete. BlackWorld's zend announced one `oseq` for hours on 2026-10-31,
/// and WhiteWorld sat on 97,323 missing records with nothing fetching them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blind {
    pub origin: u64,
    pub high: u64,
    pub top: u64,
    pub missing: u64,
}

impl GapWatch {
    /// The origins `highs` leaves blind that were not already reported at this
    /// top: said once per top, so a stuck origin is one line per change.
    pub fn blind(
        &mut self,
        highs: impl IntoIterator<Item = (u64, u64)>,
        applied: &AppliedSet,
    ) -> Vec<Blind> {
        let mut found = Vec::new();
        let mut seen = Vec::new();
        for (origin, high) in highs {
            let Some(held) = applied.origin(origin) else {
                continue;
            };
            let missing: u64 = held.gaps().iter().map(|(s, e)| e - s + 1).sum();
            if high == 0 || held.top() <= high || missing == 0 {
                continue;
            }
            seen.push(origin);
            if self.blind.insert(origin, held.top()) != Some(held.top()) {
                found.push(Blind {
                    origin,
                    high,
                    top: held.top(),
                    missing,
                });
            }
        }
        self.blind.retain(|o, _| seen.contains(o));
        found
    }

    /// The origins `highs` names as ahead of `applied` for [`GAP_WAIT`]: each
    /// is due a catch-up addressed to it, and is watched afresh from now.
    pub fn observe(
        &mut self,
        highs: impl IntoIterator<Item = (u64, u64)>,
        applied: &AppliedSet,
        now: Instant,
    ) -> Vec<u64> {
        let mut due = Vec::new();
        let mut seen = Vec::new();
        for (origin, high) in highs {
            if high == 0
                || applied
                    .origin(origin)
                    .is_some_and(|o| o.contains_range(1, high))
            {
                continue;
            }
            seen.push(origin);
            let since = self.ahead.entry(origin).or_insert(now);
            if now.duration_since(*since) >= GAP_WAIT {
                due.push(origin);
                *since = now;
            }
        }
        self.ahead.retain(|o, _| seen.contains(o));
        due.sort_unstable();
        due.dedup();
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An origin held past what it announces is reported once per top**,
    /// and one with no gap, or announcing all that is held, never.
    #[test]
    fn an_origin_held_past_its_announcement_is_reported_once_per_top() {
        let mut applied = AppliedSet::default();
        for oseq in [1, 2, 3, 6, 7] {
            applied.insert(9, oseq);
        }
        let mut watch = GapWatch::default();
        assert_eq!(
            watch.blind([(9, 3)], &applied),
            [Blind {
                origin: 9,
                high: 3,
                top: 7,
                missing: 2
            }]
        );
        assert!(watch.blind([(9, 3)], &applied).is_empty(), "said once");
        applied.insert(9, 8);
        assert_eq!(watch.blind([(9, 3)], &applied).len(), 1, "the top moved");
        assert!(watch.blind([(9, 8)], &applied).is_empty(), "announced");
        assert!(watch.blind([(9, 0)], &applied).is_empty(), "nothing said");
        for oseq in [4, 5] {
            applied.insert(9, oseq);
        }
        assert!(watch.blind([(9, 3)], &applied).is_empty(), "no gap");
    }

    /// Every 30 s while a socket is down, and never while every socket is up.
    #[test]
    fn a_periodic_catch_up_runs_every_30_s_while_a_socket_is_down() {
        let t0 = Instant::now();
        assert!(!periodic_due(true, t0, t0 + Duration::from_secs(29)));
        assert!(periodic_due(true, t0, t0 + PERIODIC));
        assert!(!periodic_due(false, t0, t0 + 10 * PERIODIC));
    }

    /// An origin ahead for 5 s is due; one that catches up before then is
    /// forgotten; one still ahead is due again 5 s later.
    #[test]
    fn an_origin_ahead_for_five_seconds_is_asked_directly() {
        let t0 = Instant::now();
        let mut w = GapWatch::default();
        let mut applied = AppliedSet::default();
        applied.insert_range(7, 1, 4);
        applied.insert_range(8, 1, 9);
        assert!(w.observe([(7, 5), (8, 9)], &applied, t0).is_empty());
        assert!(w
            .observe([(7, 5)], &applied, t0 + Duration::from_secs(4))
            .is_empty());
        assert_eq!(w.observe([(7, 5)], &applied, t0 + GAP_WAIT), vec![7]);
        assert!(w
            .observe([(7, 5)], &applied, t0 + GAP_WAIT + Duration::from_secs(1))
            .is_empty());
        assert_eq!(w.observe([(7, 5)], &applied, t0 + 2 * GAP_WAIT), vec![7]);
        applied.insert(7, 5);
        assert!(w.observe([(7, 5)], &applied, t0 + 4 * GAP_WAIT).is_empty());
        assert!(w.ahead.is_empty());
    }
}
