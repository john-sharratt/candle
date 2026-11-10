//! Whether a member's own events stopped being receipted (§6.3): events in
//! flight that no `RECEIPT` covers for a long while are a gateway that stopped
//! taking them in, or a lane that never drains — the replication status shows
//! the count, but nothing said when it began. Watched once a second; a stall
//! is logged once it has lasted [`STALL`], again every [`REPEAT`] while it
//! lasts, and its end once the count falls.

use std::time::{Duration, Instant};

/// How long the in-flight count may stand without falling before it is a stall.
pub const STALL: Duration = Duration::from_secs(60);

/// How often a stall that goes on is logged again.
pub const REPEAT: Duration = Duration::from_secs(600);

/// What one look at the in-flight count says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seen {
    /// Nothing worth a line.
    Quiet,
    /// The count has not fallen for `secs`, standing at `in_flight`.
    Stalled { in_flight: usize, secs: u64 },
    /// It fell again after a stall that lasted `secs`.
    Resumed { in_flight: usize, secs: u64 },
}

#[derive(Debug, Default)]
pub struct ReceiptWatch {
    /// Since when the count has not fallen, and what it stood at last.
    since: Option<(Instant, usize)>,
    /// When the stall was last logged.
    logged: Option<Instant>,
}

impl ReceiptWatch {
    /// The count `in_flight` at `now`. Any fall is progress: a receipt, or a
    /// socket that dropped and owes its events again.
    pub fn observe(&mut self, in_flight: usize, now: Instant) -> Seen {
        let progressed = match self.since {
            None => in_flight == 0,
            Some((_, last)) => in_flight < last || in_flight == 0,
        };
        if progressed {
            let seen = match (self.logged, self.since) {
                (Some(_), Some((at, _))) => Seen::Resumed {
                    in_flight,
                    secs: now.duration_since(at).as_secs(),
                },
                _ => Seen::Quiet,
            };
            self.logged = None;
            self.since = (in_flight > 0).then_some((now, in_flight));
            return seen;
        }
        let at = match self.since {
            Some((at, _)) => at,
            None => now,
        };
        self.since = Some((at, in_flight));
        let stalled = now.duration_since(at) >= STALL;
        let due = self
            .logged
            .is_none_or(|logged| now.duration_since(logged) >= REPEAT);
        if stalled && due {
            self.logged = Some(now);
            return Seen::Stalled {
                in_flight,
                secs: now.duration_since(at).as_secs(),
            };
        }
        Seen::Quiet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A count that stands is a stall at a minute, logged again every ten
    /// while it lasts, and its end is logged once it falls.** Growing is no
    /// progress; nothing in flight is never a stall.
    #[test]
    fn a_standing_count_is_a_stall_and_its_fall_ends_it() {
        let t0 = Instant::now();
        let s = |n: u64| t0 + Duration::from_secs(n);
        let mut w = ReceiptWatch::default();
        assert_eq!(w.observe(0, s(0)), Seen::Quiet);
        assert_eq!(w.observe(56, s(1)), Seen::Quiet);
        assert_eq!(w.observe(60, s(30)), Seen::Quiet, "growing is no progress");
        assert_eq!(
            w.observe(60, s(61)),
            Seen::Stalled {
                in_flight: 60,
                secs: 60
            }
        );
        assert_eq!(w.observe(60, s(62)), Seen::Quiet, "logged once");
        assert_eq!(
            w.observe(60, s(661)),
            Seen::Stalled {
                in_flight: 60,
                secs: 660
            },
            "again ten minutes on"
        );
        assert_eq!(
            w.observe(12, s(700)),
            Seen::Resumed {
                in_flight: 12,
                secs: 699
            }
        );
        assert_eq!(w.observe(12, s(759)), Seen::Quiet, "a fresh minute");
        assert_eq!(w.observe(5, s(800)), Seen::Quiet, "falling, never stalled");
        assert_eq!(w.observe(0, s(900)), Seen::Quiet);
    }

    /// A count that keeps falling while events keep going out is never a
    /// stall, however long something stays in flight.
    #[test]
    fn a_count_that_keeps_falling_is_no_stall() {
        let t0 = Instant::now();
        let mut w = ReceiptWatch::default();
        for i in 0..200u64 {
            let n = if i % 2 == 0 { 40 } else { 30 };
            assert_eq!(
                w.observe(n, t0 + Duration::from_secs(i)),
                Seen::Quiet,
                "at {i} s"
            );
        }
    }
}
