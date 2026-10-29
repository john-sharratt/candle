//! Rule 5's triggers (`docs/cooperative_admission.md` §4.5): when a card is
//! busy with latency or bulk work, judged so that a pause in either moves
//! nothing — busy quickly, free slowly.

use std::time::{Duration, Instant};

/// How long a card stays latency-busy after the last latency unit on it: a
/// conversation pauses while its person reads and types.
pub const LATENCY_LINGER: Duration = Duration::from_secs(10 * 60);

/// How long bulk work must have been outstanding before the card is
/// bulk-busy: a single commit's re-ingest moves nothing.
pub const BULK_ONSET: Duration = Duration::from_secs(60);

/// How long a card stays bulk-busy after its bulk work empties: ingest ends
/// and restarts unit by unit.
pub const BULK_LINGER: Duration = Duration::from_secs(5 * 60);

/// One card's triggers, as the plan has seen them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Busy {
    latency_last: Option<Instant>,
    bulk_since: Option<Instant>,
    bulk_last: Option<Instant>,
    bulk_on: bool,
}

impl Busy {
    /// What the card was doing at `now`: latency work in flight, bulk work
    /// outstanding.
    pub fn observe(&mut self, now: Instant, latency: bool, bulk: bool) {
        if latency {
            self.latency_last = Some(now);
        }
        if bulk {
            let since = *self.bulk_since.get_or_insert(now);
            self.bulk_last = Some(now);
            if now.duration_since(since) >= BULK_ONSET {
                self.bulk_on = true;
            }
        } else {
            self.bulk_since = None;
            if self
                .bulk_last
                .is_none_or(|last| now.duration_since(last) >= BULK_LINGER)
            {
                self.bulk_on = false;
            }
        }
    }

    /// Latency work has been on the card within [`LATENCY_LINGER`].
    pub fn latency(&self, now: Instant) -> bool {
        self.latency_last
            .is_some_and(|last| now.duration_since(last) < LATENCY_LINGER)
    }

    /// Bulk work has held past [`BULK_ONSET`], and not been empty for
    /// [`BULK_LINGER`] since.
    pub fn bulk(&self) -> bool {
        self.bulk_on
    }

    pub fn busy(&self, now: Instant) -> bool {
        self.latency(now) || self.bulk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// **Busy at the first latency unit, free ten minutes after the last**: a
    /// person reading between turns keeps the card busy.
    #[test]
    fn latency_is_busy_at_once_and_free_slowly() {
        let t0 = Instant::now();
        let mut b = Busy::default();
        b.observe(t0, false, false);
        assert!(!b.busy(t0));
        b.observe(t0 + 5 * S, true, false);
        assert!(b.latency(t0 + 5 * S));
        b.observe(t0 + 10 * S, false, false);
        assert!(b.busy(t0 + 10 * 60 * S), "within the linger");
        b.observe(t0 + 605 * S, false, false);
        assert!(!b.busy(t0 + 605 * S), "ten minutes after the last unit");
    }

    /// **Bulk is busy once it has held a minute, and free five minutes after
    /// it empties**: a single re-ingest moves nothing, and a pause between
    /// units does not end it.
    #[test]
    fn bulk_holds_a_minute_and_lingers_five() {
        let t0 = Instant::now();
        let mut b = Busy::default();
        b.observe(t0, false, true);
        b.observe(t0 + 30 * S, false, true);
        assert!(!b.bulk(), "thirty seconds is a re-ingest");
        b.observe(t0 + 35 * S, false, false);
        b.observe(t0 + 40 * S, false, true);
        b.observe(t0 + 95 * S, false, true);
        assert!(!b.bulk(), "the minute restarts after an empty look");
        b.observe(t0 + 100 * S, false, true);
        assert!(b.bulk(), "held a minute");
        b.observe(t0 + 105 * S, false, false);
        b.observe(t0 + 300 * S, false, false);
        assert!(b.bulk(), "empty, but within the linger");
        b.observe(t0 + 405 * S, false, false);
        assert!(!b.bulk(), "five minutes empty");
    }
}
