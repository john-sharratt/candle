//! Releasing dialogue conversations nobody is using.
//!
//! A resident conversation holds a scheduler slot, and on a recurrent model the
//! slot's state store and its QSA index caches stand on the card for as long as
//! the conversation stays in [`crate::session`]'s map — which, without a sweep,
//! was until it was deleted. Every dialogue the daemon had served since it
//! started kept its ground. Measured on `.5`, 2026-10-17: 14 idle test
//! conversations held 2.7 GB of index caches beside their stores, with the
//! expert zone at 24.6 GB of a 29.7 GB floor and a deep decode's expert hit rate
//! at 75–85%.
//!
//! A released conversation loses nothing: its next request forks it again from
//! the substrate (`fork_resuming`), the path a daemon restart takes.

use std::time::{Duration, Instant};

/// How long a dialogue conversation stays resident after it was last used —
/// the passthrough conversations' figure (`passthrough::IDLE_TTL`).
pub const CONV_IDLE_TTL: Duration = Duration::from_secs(5 * 60);

/// How often idle conversations are looked for.
pub const CONV_SWEEP_EVERY: Duration = Duration::from_secs(30);

/// What one sweep does with a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sweep {
    /// Leave it, and its clock.
    Keep,
    /// Something holds it — a turn is running on it — so it is in use now.
    Touch,
    /// Nothing holds it and it has been idle for the whole TTL.
    Release,
}

/// The verdict for a conversation last used at `last`, given whether anything
/// beyond the map holds it.
pub fn sweep(last: Instant, now: Instant, ttl: Duration, held: bool) -> Sweep {
    if held {
        Sweep::Touch
    } else if now.saturating_duration_since(last) >= ttl {
        Sweep::Release
    } else {
        Sweep::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_held_conversation_is_in_use_however_long_ago_it_started() {
        let t0 = Instant::now();
        let now = t0 + 10 * CONV_IDLE_TTL;
        assert_eq!(sweep(t0, now, CONV_IDLE_TTL, true), Sweep::Touch);
    }

    #[test]
    fn an_unheld_conversation_goes_once_the_ttl_has_passed_and_not_before() {
        let t0 = Instant::now();
        let ttl = CONV_IDLE_TTL;
        assert_eq!(sweep(t0, t0, ttl, false), Sweep::Keep);
        assert_eq!(
            sweep(t0, t0 + ttl - Duration::from_secs(1), ttl, false),
            Sweep::Keep
        );
        assert_eq!(sweep(t0, t0 + ttl, ttl, false), Sweep::Release);
    }

    #[test]
    fn a_clock_ahead_of_now_reads_as_fresh() {
        let now = Instant::now();
        let later = now + Duration::from_secs(60);
        assert_eq!(sweep(later, now, CONV_IDLE_TTL, false), Sweep::Keep);
    }
}
