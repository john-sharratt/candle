//! Decides how a turn ended, so the worker settles on the right outcome.
//!
//! A turn ends either as an ending the worker settles on or as a move to another
//! member. A turn that failed is a failure whatever else it said, since nothing it
//! said can be trusted. A turn closed by the repeat guard, the round cap or a stop
//! keeps its whole text. A closing draft that does not end before its deadline is
//! recorded as such rather than waited on.

use super::*;

/// How a turn ended, before the worker settles.
pub(super) enum Ended {
    /// It ended on its own: its whole text.
    Replied(String),
    /// Its tools were closed — by the repeat guard, the round cap, or Claude
    /// stopping it (`by`, a [`StreamItem::Closing`] reason): its whole text.
    Closed {
        text: String,
        by: &'static str,
    },
    Failed(String),
}

/// What a turn came to: an ending the worker settles on, or a move.
pub(super) enum Turned {
    Ended(Ended),
    /// It stopped at round `iteration` to move to member `to`, its call turn
    /// `call_turn` sealed and its tools not run ([`StreamItem::Handed`]).
    Handed {
        to: u64,
        call_turn: u32,
        iteration: usize,
    },
    /// The member left its clusters under the turn ([`StreamItem::Left`]): the
    /// turn stopped where it stood, its round's entries kept, for another
    /// member to resume (`docs/turn_handover.md` §4).
    Left,
}

impl Ended {
    /// Whether Claude stopped the turn (`job_cancel`).
    pub(super) fn stopped(&self) -> bool {
        matches!(self, Ended::Closed { by, .. } if *by == CLOSED_BY_CLAUDE)
    }
}

/// What a turn that streamed `text` came to: a failure first, since nothing
/// it said can be trusted; then a move, since its round is not over; then a
/// close; else a reply. A turn the member left under is [`Turned::Left`]
/// whatever else it streamed, read before this.
pub(super) fn turned(
    text: String,
    failed: Option<String>,
    closed: Option<&'static str>,
    handed: Option<(u64, u32, usize)>,
) -> Turned {
    match (failed, closed, handed) {
        (Some(e), _, _) => Turned::Ended(Ended::Failed(e)),
        (None, _, Some((to, call_turn, iteration))) => Turned::Handed {
            to,
            call_turn,
            iteration,
        },
        (None, Some(by), None) => Turned::Ended(Ended::Closed { text, by }),
        (None, None, None) => Turned::Ended(Ended::Replied(text)),
    }
}

/// How a turn's closing draft went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drafted {
    /// A draft over the turn's stretch ran and ended: its record is the
    /// journal's last.
    Ended,
    /// None ran, for this reason.
    NotRun(NoDraft),
    /// The conversation keeps no journal.
    NoJournal,
    /// The turn or its draft did not end before [`DRAFT_DEADLINE`].
    TimedOut,
}
