//! When a card is busy (`docs/static_placement.md` §5.1): from its first live
//! conversation, until it has had none for [`RETURN_AFTER`].

use std::time::{Duration, Instant};

/// How long a card stays busy after its last live conversation: the pause
/// while a person reads a reply and types the next message.
pub const RETURN_AFTER: Duration = Duration::from_secs(120);

/// One card's busy state, as the plan has seen it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Busy {
    last: Option<Instant>,
}

impl Busy {
    /// Whether a live conversation was in flight on the card at `now`.
    pub fn observe(&mut self, now: Instant, live: bool) {
        if live {
            self.last = Some(now);
        }
    }

    /// A live conversation has been on the card within [`RETURN_AFTER`].
    pub fn busy(&self, now: Instant) -> bool {
        self.last
            .is_some_and(|last| now.duration_since(last) < RETURN_AFTER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// **Busy at the first live conversation, free two minutes after the
    /// last**: a person reading between turns keeps the card busy.
    #[test]
    fn busy_at_once_and_free_two_minutes_after() {
        let t0 = Instant::now();
        let mut b = Busy::default();
        b.observe(t0, false);
        assert!(!b.busy(t0));
        b.observe(t0 + 5 * S, true);
        assert!(b.busy(t0 + 5 * S));
        b.observe(t0 + 10 * S, false);
        assert!(b.busy(t0 + 124 * S), "within two minutes of the last");
        assert!(!b.busy(t0 + 125 * S), "two minutes after the last");
    }
}
