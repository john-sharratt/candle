//! Saying when one gap has capped what this node may announce of itself.
//!
//! A gap in a node's own records is normal for a moment: a record is stamped
//! when its value changes and appended when the log's owner reaches its job. One
//! that stays is not, and nothing else says so: a peer reads the announced
//! `oseq` as all there is, so it asks for nothing, and the rollout that needed
//! a record above the gap simply never happens (2026-10-31).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long one gap may cap the count before it is said.
const SAY_AFTER: Duration = Duration::from_secs(60);

/// The gap holding the count, by its first `oseq`, and since when. Atomics so
/// the read that observes it stays a read.
#[derive(Debug, Default)]
pub(super) struct OwnHold {
    start: AtomicU64,
    since_ms: AtomicU64,
    said: AtomicU64,
    detached: AtomicBool,
}

impl OwnHold {
    /// Note the gap `hold` capping the count now, or that none is.
    pub(super) fn observe(&self, hold: Option<(u64, u64)>, top: u64, open_top: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        if let Some((start, end)) = self.note(hold, now) {
            tracing::warn!(
                gap_start = start,
                gap_end = end,
                top,
                open_top,
                "one gap in this node's own records has capped what it may announce for over a \
                 minute: the records above it reach no peer that trusts the announcement"
            );
        }
    }

    /// Note a record this node wrote cut off: untapped and unannounced until a
    /// rejoin fixes it. Said for the first one of a process, as a node that is
    /// cut off without knowing it otherwise only looks idle to its peers.
    pub(super) fn wrote_detached(&self, oseq: u64) {
        if self.detached.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::warn!(
            oseq,
            "this node is writing records under the detached generation: they reach no peer \
             until a rejoin fixes them"
        );
    }

    /// Whether this process has said it writes detached.
    #[cfg(test)]
    pub(super) fn said_detached(&self) -> bool {
        self.detached.load(Ordering::Relaxed)
    }

    /// The gap to say at `now_ms`: one that has held [`SAY_AFTER`], once.
    fn note(&self, hold: Option<(u64, u64)>, now_ms: u64) -> Option<(u64, u64)> {
        let Some((start, end)) = hold else {
            self.start.store(0, Ordering::Relaxed);
            return None;
        };
        if self.start.swap(start, Ordering::Relaxed) != start {
            self.since_ms.store(now_ms, Ordering::Relaxed);
            return None;
        }
        let held = now_ms.saturating_sub(self.since_ms.load(Ordering::Relaxed));
        (held >= SAY_AFTER.as_millis() as u64 && self.said.swap(start, Ordering::Relaxed) != start)
            .then_some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A gap is said once, after it has held for a minute**; a gap that
    /// clears starts over.
    #[test]
    fn a_gap_is_said_once_after_a_minute() {
        let hold = OwnHold::default();
        assert_eq!(hold.note(Some((5, 5)), 1_000), None, "first seen");
        assert_eq!(hold.note(Some((5, 5)), 60_999), None, "under a minute");
        assert_eq!(hold.note(Some((5, 5)), 61_000), Some((5, 5)));
        assert_eq!(hold.note(Some((5, 5)), 90_000), None, "said once");
        assert_eq!(hold.note(None, 91_000), None);
        assert_eq!(hold.note(Some((9, 9)), 92_000), None, "a new gap waits");
        assert_eq!(hold.note(Some((9, 9)), 152_000), Some((9, 9)));
    }
}
