//! When a conversation takes its turn (`docs/npc_conversations.md` §4.3, §8).
//!
//! - **The burst window.** A turn starts [`BURST_MS`] after the last unanswered
//!   line, and at most [`BURST_MAX_MS`] after the first: one answer to three
//!   quick lines.
//! - **One turn at a time.** What arrives during a turn waits for the turn's
//!   end, and the next turn takes it.
//! - **Activations.** An NPC↔NPC exchange is capped per activation, and an
//!   activation starts at a purpose, or at the first line that arrives after
//!   the conversation went quiet — a journal turn or a goodbye — so the
//!   answering side's count starts again too.

use std::mem;

/// How long a turn waits after the last unanswered line.
pub const BURST_MS: u64 = 2_500;

/// The longest a turn waits after the first unanswered line.
pub const BURST_MAX_MS: u64 = 8_000;

/// How long a conversation whose turn could not start waits before it tries
/// again, the first time.
pub const RETRY_FIRST_MS: u64 = 5_000;

/// The longest it waits, however many times in a row its turn could not start.
pub const RETRY_MAX_MS: u64 = 300_000;

/// How long a conversation waits before trying again after `failed` turns in a
/// row could not start: [`RETRY_FIRST_MS`], doubling each time, at most
/// [`RETRY_MAX_MS`]. A fault that outlasts every retry — a mind with no
/// schema — costs one try every five minutes, not one every burst window.
pub fn retry_delay(failed: u32) -> u64 {
    RETRY_FIRST_MS
        .saturating_mul(1u64 << failed.min(16))
        .min(RETRY_MAX_MS)
}

/// What a conversation is doing, as its clock sees it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    /// Nothing waiting.
    #[default]
    Idle,
    /// Something arrived; the window runs from `first_ms` to its due time.
    Gathering { first_ms: u64, last_ms: u64 },
    /// A turn is running; `queued` is whether something arrived during it.
    Turning { queued: Option<(u64, u64)> },
}

impl Phase {
    /// Something arrived at `at_ms`.
    pub fn arrive(&mut self, at_ms: u64) {
        *self = match mem::take(self) {
            Phase::Idle => Phase::Gathering {
                first_ms: at_ms,
                last_ms: at_ms,
            },
            Phase::Gathering { first_ms, .. } => Phase::Gathering {
                first_ms,
                last_ms: at_ms,
            },
            Phase::Turning { queued } => Phase::Turning {
                queued: Some(match queued {
                    Some((first, _)) => (first, at_ms),
                    None => (at_ms, at_ms),
                }),
            },
        };
    }

    /// When the turn is due, while gathering.
    pub fn due_ms(&self) -> Option<u64> {
        match self {
            Phase::Gathering { first_ms, last_ms } => {
                Some((last_ms + BURST_MS).min(first_ms + BURST_MAX_MS))
            }
            _ => None,
        }
    }

    /// Whether the turn should start at `now_ms`; if so, it has.
    pub fn start_if_due(&mut self, now_ms: u64) -> bool {
        match self.due_ms() {
            Some(due) if now_ms >= due => {
                *self = Phase::Turning { queued: None };
                true
            }
            _ => false,
        }
    }

    /// The turn could not start: everything waiting gathers afresh from
    /// `at_ms`, whenever it arrived, so the wait before trying again is not cut
    /// short by [`BURST_MAX_MS`] from a line that came in long before.
    pub fn retry_at(&mut self, at_ms: u64) {
        *self = Phase::Gathering {
            first_ms: at_ms,
            last_ms: at_ms,
        };
    }

    /// The turn ended: what arrived during it gathers from when it arrived.
    pub fn end_turn(&mut self) {
        *self = match mem::take(self) {
            Phase::Turning {
                queued: Some((first_ms, last_ms)),
            } => Phase::Gathering { first_ms, last_ms },
            _ => Phase::Idle,
        };
    }
}

/// How the last turn ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    /// It said something to the party.
    Spoke,
    /// It wrote to the journal, or said goodbye: it went quiet.
    Quiet,
}

/// What starts a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// The character meant to say something (§10).
    Purpose,
    /// The party said something.
    Line,
}

/// The activation's turn count for a turn on `input`, after `previous`
/// turns that ended `last`: a purpose, or a line after a quiet end, starts a
/// new activation.
pub fn activation(previous: u32, last: Option<Ended>, input: Input) -> u32 {
    match (input, last) {
        (Input::Purpose, _) | (Input::Line, Some(Ended::Quiet) | None) => 1,
        (Input::Line, Some(Ended::Spoke)) => previous + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_quick_lines_are_one_turn_after_the_last() {
        let mut p = Phase::default();
        p.arrive(1_000);
        p.arrive(1_800);
        p.arrive(2_400);
        assert_eq!(p.due_ms(), Some(4_900));
        assert!(!p.start_if_due(4_899));
        assert!(p.start_if_due(4_900));
        assert_eq!(p, Phase::Turning { queued: None });
    }

    #[test]
    fn a_steady_stream_is_answered_eight_seconds_after_its_first_line() {
        let mut p = Phase::default();
        for at in (0..=10_000).step_by(2_000) {
            p.arrive(at);
        }
        assert_eq!(p.due_ms(), Some(8_000));
    }

    #[test]
    fn a_line_during_a_turn_waits_for_the_next() {
        let mut p = Phase::default();
        p.arrive(0);
        assert!(p.start_if_due(2_500));
        p.arrive(3_000);
        p.arrive(3_500);
        assert_eq!(p.due_ms(), None, "no second turn while one runs");
        p.end_turn();
        assert_eq!(
            p,
            Phase::Gathering {
                first_ms: 3_000,
                last_ms: 3_500
            }
        );
        assert_eq!(p.due_ms(), Some(6_000));
        assert!(p.start_if_due(6_000));
        p.end_turn();
        assert_eq!(p, Phase::Idle);
    }

    #[test]
    fn a_retry_waits_longer_each_time_up_to_five_minutes() {
        assert_eq!(retry_delay(0), 5_000);
        assert_eq!(retry_delay(1), 10_000);
        assert_eq!(retry_delay(5), 160_000);
        assert_eq!(retry_delay(6), RETRY_MAX_MS);
        assert_eq!(retry_delay(u32::MAX), RETRY_MAX_MS);
    }

    #[test]
    fn an_activation_starts_at_a_purpose_or_after_a_quiet_end() {
        assert_eq!(activation(0, None, Input::Line), 1);
        assert_eq!(activation(3, Some(Ended::Spoke), Input::Line), 4);
        assert_eq!(activation(7, Some(Ended::Quiet), Input::Line), 1);
        assert_eq!(activation(7, Some(Ended::Spoke), Input::Purpose), 1);
    }
}
