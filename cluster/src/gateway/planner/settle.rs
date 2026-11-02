//! A member's fitness to host, settled (`docs/static_placement.md` §5.2): a
//! change the plan acts on only once it has held for [`SETTLE`].
//!
//! A member flipping in and out of lag faster than that moves nothing. On
//! 2026-11-02 WhiteWorld's weight flipped between 0 and 245 every one to two
//! minutes, and a character asked to move and then not ran nowhere. A member's
//! first fitness — it has joined and caught up — counts at once: there is
//! nothing yet for it to flip back from.

use std::time::{Duration, Instant};

/// How long a member's fitness must hold before the plan acts on it.
pub const SETTLE: Duration = Duration::from_secs(60);

/// One member's fitness: what the plan acts on, and what it has seen since.
#[derive(Debug, Clone, Copy)]
pub struct Settled {
    settled: bool,
    seen: bool,
    since: Instant,
    /// Whether the plan has ever lent to it.
    hosted: bool,
}

impl Settled {
    /// A member first seen at `now`, taken as it is: there is nothing yet to
    /// settle against.
    pub fn new(eligible: bool, now: Instant) -> Settled {
        Settled {
            settled: eligible,
            seen: eligible,
            since: now,
            hosted: eligible,
        }
    }

    /// What the member is at `now`.
    pub fn observe(&mut self, eligible: bool, now: Instant) {
        if eligible != self.seen {
            self.seen = eligible;
            self.since = now;
        }
        let first = eligible && !self.hosted;
        if self.seen != self.settled && (first || now.duration_since(self.since) >= SETTLE) {
            self.settled = self.seen;
            self.hosted |= self.settled;
        }
    }

    /// Whether the plan lends to it.
    pub fn eligible(&self) -> bool {
        self.settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// **A change counts once it has held a minute**: a flip back inside the
    /// minute is forgotten, and the minute starts again at each change.
    #[test]
    fn a_change_counts_once_it_has_held_a_minute() {
        let t0 = Instant::now();
        let mut s = Settled::new(true, t0);
        assert!(s.eligible(), "taken as first seen");
        s.observe(false, t0 + 10 * S);
        s.observe(false, t0 + 69 * S);
        assert!(s.eligible(), "59 s is not a minute");
        s.observe(true, t0 + 70 * S);
        s.observe(false, t0 + 80 * S);
        s.observe(false, t0 + 139 * S);
        assert!(s.eligible(), "the minute started again at 80 s");
        s.observe(false, t0 + 140 * S);
        assert!(!s.eligible(), "held a minute");
        s.observe(true, t0 + 150 * S);
        s.observe(true, t0 + 209 * S);
        assert!(!s.eligible(), "back, but not for a minute yet");
        s.observe(true, t0 + 210 * S);
        assert!(s.eligible(), "and back once that has held a minute");
    }

    /// **A member's first fitness counts at once**: one that joins syncing is
    /// lent to the moment it has caught up — and only then does a flip have to
    /// hold a minute.
    #[test]
    fn a_first_fitness_counts_at_once() {
        let t0 = Instant::now();
        let mut s = Settled::new(false, t0);
        assert!(!s.eligible());
        s.observe(true, t0 + 5 * S);
        assert!(s.eligible(), "caught up: hosts at once");
        s.observe(false, t0 + 6 * S);
        s.observe(true, t0 + 7 * S);
        assert!(s.eligible(), "a flip after it hosted is settled");
    }
}
