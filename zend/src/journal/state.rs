//! A session's journal, and when its stretch of dialogue is due a draft.
//!
//! What the journal holds is [`Journal`]'s. What is the session's is the cadence
//! ([`SessionJournal::due`]), counted in the user's requests: a request is the
//! user's turn and every tool round it takes, until a reply ends without a call
//! (`docs/zend_journal.md` §6).
//!
//! - **A finished request** is always due a draft, and the draft may not skip:
//!   every request the user made is in the journal.
//! - **Within a long request** a draft is due every [`EVERY_ROUNDS`] of its
//!   sealed turns, or sooner when the dialogue's projection has started
//!   leaving out turns no draft has looked at — a turn is looked at before the
//!   session loses sight of it. Such a draft may skip only once this request
//!   has an entry, so a skip files its stretch under the request's own entry
//!   and never under the request before it.
//!
//! At most one draft runs at a time. A request that finishes while one runs is
//! owed its entry, which starts when that draft ends ([`SessionJournal::owe`]).
//!
//! The journal counts in parts ([`super::origin`]), whose numbering is the
//! turns' own and stable across restarts, so a journal restored from its record
//! carries on from the last part its entries covered.

use std::mem;
use std::time::{Duration, Instant};

use journal_core::journal::{Holds, Journal};
use journal_core::section::Kept;
use journal_core::span::{Span, SpanTurn};
use journal_core::style::Style;
use journal_core::workflow::Skip;

use crate::journal::origin::{reply_part, turn_of, user_part};
use crate::journal::style::CODING;

/// Sealed turns of one request since the journal last looked after which a
/// progress entry is drafted.
pub const EVERY_ROUNDS: u32 = 16;

/// Where the request a sealed turn belongs to stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    /// The request's first turn: the one holding the user's message.
    pub from: u32,
    /// The turn just sealed ended the request: its reply made no call.
    pub finished: bool,
}

/// A draft that is due, and whether it may skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Due {
    pub skip: Skip,
}

/// How often open conversations are asked whether a period of their journal
/// has ended and is due to fold. A conversation's own rounds ask as well; this
/// is for the day that turns over while it sits open and quiet.
pub const FOLD_SWEEP_EVERY: Duration = Duration::from_secs(5 * 60);

/// How long after a fold was given up on before the same journal folds again.
/// A fold that failed its rounds would otherwise be asked again on every round
/// and every sweep — four decodes each time, against the dialogue's slot.
pub const FOLD_RETRY_AFTER: Duration = Duration::from_secs(60 * 60);

#[derive(Debug)]
pub struct SessionJournal {
    journal: Journal,
    /// When a fold was last given up on, if one was this run.
    fold_failed_at: Option<Instant>,
    /// The wording its entries are kept in ([`super::style::style_for`]).
    style: &'static Style,
    /// A request finished while a draft ran: its entry starts when that draft
    /// ends.
    owed: bool,
}

impl Default for SessionJournal {
    fn default() -> Self {
        Self::new()
    }
}

impl Holds for SessionJournal {
    fn journal(&self) -> &Journal {
        &self.journal
    }

    fn journal_mut(&mut self) -> &mut Journal {
        &mut self.journal
    }
}

impl SessionJournal {
    /// An empty journal of a coding session.
    pub fn new() -> Self {
        Self {
            journal: Journal::default(),
            fold_failed_at: None,
            style: &CODING,
            owed: false,
        }
    }

    /// Rebuild from the record the conversation's metadata held. Part numbers
    /// are the conversation's own, so what the record says is covered — by an
    /// entry or by a draft that found nothing to add — stays covered.
    pub fn restore(kept: Kept) -> Self {
        let covered_to = kept.covered_to;
        Self {
            journal: Journal::restore(kept, covered_to),
            ..Self::new()
        }
    }

    /// This journal, kept in `style`.
    pub fn in_style(self, style: &'static Style) -> Self {
        Self { style, ..self }
    }

    /// The wording its entries are kept in.
    pub fn style(&self) -> &'static Style {
        self.style
    }

    /// A fold was given up on at `now`.
    pub fn fold_failed(&mut self, now: Instant) {
        self.fold_failed_at = Some(now);
    }

    /// Whether a fold may start at `now`: none has been given up on within
    /// [`FOLD_RETRY_AFTER`].
    pub fn may_fold(&self, now: Instant) -> bool {
        self.fold_failed_at
            .is_none_or(|at| now.saturating_duration_since(at) >= FOLD_RETRY_AFTER)
    }

    /// Turns sealed since the last draft was handed a stretch, `newest` being
    /// the newest sealed turn's index.
    pub fn fresh_turns(&self, newest: u32) -> u32 {
        match turn_of(self.journal.looked_to()) {
            Some(looked) => newest.saturating_sub(looked),
            None => newest + 1,
        }
    }

    /// The first turn no entry and no "nothing to write" covers.
    pub fn first_uncovered_turn(&self) -> u32 {
        turn_of(self.journal.covered_to()).map_or(0, |t| t + 1)
    }

    /// Whether a draft is due now that turn `newest` of `request` has sealed,
    /// and whether it may skip. `pressed` is whether the dialogue's last
    /// projection left out a turn at or past [`Self::first_uncovered_turn`]
    /// ([`super::pressure`]). Whether a draft is running is the caller's to
    /// ask: a finished request whose draft cannot start now is owed it
    /// ([`Self::owe`]).
    pub fn due(&self, newest: u32, pressed: bool, request: Request) -> Option<Due> {
        if request.finished {
            return Some(Due {
                skip: Skip::Refused,
            });
        }
        (self.request_turns(newest, request.from) >= EVERY_ROUNDS || pressed).then(|| Due {
            skip: self.progress_skip(request.from),
        })
    }

    /// Turns of the request opened at turn `from` sealed since the journal
    /// last looked, `newest` being the newest one's index.
    fn request_turns(&self, newest: u32, from: u32) -> u32 {
        let since = turn_of(self.journal.looked_to()).map_or(from, |looked| (looked + 1).max(from));
        (newest + 1).saturating_sub(since)
    }

    /// Whether a progress draft of the request opened at turn `from` may skip:
    /// only once the journal's latest step is the request's own, which a skip
    /// files its stretch under.
    fn progress_skip(&self, from: u32) -> Skip {
        let own = self
            .journal
            .latest_step()
            .is_some_and(|e| e.from_turn >= user_part(from));
        if own {
            Skip::Allowed
        } else {
            Skip::Refused
        }
    }

    /// A finished request's entry could not start: a draft is running. It
    /// starts when that draft ends ([`Self::take_owed`]).
    pub fn owe(&mut self) {
        self.owed = true;
    }

    /// Whether a finished request's entry is owed, clearing it.
    pub fn take_owed(&mut self) -> bool {
        mem::take(&mut self.owed)
    }

    /// Start a draft over `parts`, the parts past what the journal covers, up to
    /// turn `newest`. `None` while one is running or when there are no parts.
    pub fn begin(&mut self, parts: Vec<SpanTurn>, newest: u32) -> Option<Span> {
        self.journal.begin(parts, reply_part(newest))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use journal_core::entry::{Entry, Kind};
    use journal_core::period::Period;

    use super::*;
    use crate::journal::origin::user_part;

    /// Both parts of each turn in `turns`.
    fn parts(turns: RangeInclusive<u32>) -> Vec<SpanTurn> {
        turns
            .flat_map(|t| [user_part(t), reply_part(t)])
            .map(|id| SpanTurn {
                id,
                at_ms: id * 100,
                evidence: Some(Kind::Did),
                text: format!("part {id}"),
                full: format!("part {id}"),
            })
            .collect()
    }

    fn entry_to(part: u64) -> Entry {
        Entry {
            id: 0,
            from_turn: 1,
            to_turn: part,
            from_ms: 0,
            to_ms: 1,
            summary: "Went on.".into(),
            claims: vec![],
            intend: vec!["carry on".into()],
            opened: vec![],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
        }
    }

    fn ongoing(from: u32) -> Request {
        Request {
            from,
            finished: false,
        }
    }

    fn finished(from: u32) -> Request {
        Request {
            from,
            finished: true,
        }
    }

    const REFUSED: Option<Due> = Some(Due {
        skip: Skip::Refused,
    });

    /// **Every finished request is due an entry, and its draft may not
    /// skip** — a one-turn request included, and whatever the journal holds.
    #[test]
    fn a_finished_request_is_always_due_and_may_not_skip() {
        let mut j = SessionJournal::new();
        assert_eq!(j.due(0, false, finished(0)), REFUSED);
        j.begin(parts(0..=0), 0).unwrap();
        j.journal_mut().kept(entry_to(reply_part(0)));
        assert_eq!(j.due(1, false, finished(1)), REFUSED);
    }

    /// Within a request, a draft is due once sixteen of its turns have sealed
    /// since the journal last looked, and not before.
    #[test]
    fn a_long_request_is_due_a_draft_every_sixteen_of_its_turns() {
        let j = SessionJournal::new();
        assert_eq!(
            j.due(14, false, ongoing(0)),
            None,
            "turns 0..=14 are fifteen"
        );
        assert_eq!(
            j.due(15, false, ongoing(0)),
            REFUSED,
            "turns 0..=15 are sixteen"
        );
        // A request opened at turn 10 has had six of its turns by turn 15.
        assert_eq!(j.due(15, false, ongoing(10)), None);
        assert_eq!(j.due(25, false, ongoing(10)), REFUSED);
    }

    #[test]
    fn selection_pressure_makes_a_draft_due_before_the_cadence() {
        let j = SessionJournal::new();
        assert_eq!(j.due(2, true, ongoing(0)), REFUSED);
    }

    /// The count runs from where the journal last looked within the request:
    /// after a progress draft over turns 0..=15, the next is due at turn 31.
    #[test]
    fn the_cadence_counts_from_the_last_look() {
        let mut j = SessionJournal::new();
        let span = j.begin(parts(0..=15), 15).unwrap();
        j.journal_mut().nothing_to_write(&span);
        assert_eq!(j.fresh_turns(20), 5);
        assert_eq!(j.due(30, false, ongoing(0)), None);
        assert!(j.due(31, false, ongoing(0)).is_some());
    }

    /// **A progress draft may skip only once its request has an entry**: the
    /// skip files the stretch under the journal's latest step, which must be
    /// this request's and never the one before it.
    #[test]
    fn a_progress_draft_skips_only_under_its_own_requests_entry() {
        let mut j = SessionJournal::new();
        // The previous request, turns 0..=2, has its entry.
        j.begin(parts(0..=2), 2).unwrap();
        j.journal_mut().kept(Entry {
            from_turn: user_part(0),
            ..entry_to(reply_part(2))
        });
        // A request opened at turn 3 has none yet: its first draft writes.
        assert_eq!(j.due(18, false, ongoing(3)), REFUSED);
        // Once it has a progress entry, the next one may skip.
        j.begin(parts(3..=18), 18).unwrap();
        j.journal_mut().kept(Entry {
            from_turn: user_part(3),
            ..entry_to(reply_part(18))
        });
        assert_eq!(
            j.due(34, false, ongoing(3)),
            Some(Due {
                skip: Skip::Allowed
            })
        );
    }

    /// A request that finishes while a draft runs is owed its entry, once.
    #[test]
    fn a_finished_request_behind_a_running_draft_is_owed_its_entry() {
        let mut j = SessionJournal::new();
        assert!(!j.take_owed());
        j.owe();
        assert!(j.take_owed());
        assert!(!j.take_owed(), "taken once");
    }

    #[test]
    fn a_running_draft_holds_the_single_flight() {
        let mut j = SessionJournal::new();
        j.begin(parts(0..=20), 20).unwrap();
        assert!(j.begin(parts(0..=20), 20).is_none(), "single flight");
    }

    #[test]
    fn a_no_covers_its_stretch_and_the_next_starts_after_it() {
        let mut j = SessionJournal::new();
        let span = j.begin(parts(0..=15), 15).unwrap();
        j.journal_mut().nothing_to_write(&span);
        assert_eq!(j.first_uncovered_turn(), 16);
    }

    #[test]
    fn an_abandoned_draft_leaves_its_turns_uncovered() {
        let mut j = SessionJournal::new();
        j.begin(parts(0..=15), 15).unwrap();
        j.journal_mut().abandon();
        assert_eq!(j.first_uncovered_turn(), 0);
        assert_eq!(
            j.due(16, false, ongoing(0)),
            None,
            "looked at, so the cadence starts over"
        );
        assert!(
            j.due(16, true, ongoing(0)).is_some(),
            "but pressure on an unread turn still asks"
        );
    }

    #[test]
    fn a_kept_entry_covers_through_the_last_part_it_names() {
        let mut j = SessionJournal::new();
        j.begin(parts(0..=15), 15).unwrap();
        j.journal_mut().kept(entry_to(reply_part(15)));
        assert_eq!(j.first_uncovered_turn(), 16);
    }

    #[test]
    fn a_restored_journal_carries_on_from_what_it_covered() {
        let kept = Kept {
            entries: vec![Entry {
                id: 4,
                ..entry_to(reply_part(30))
            }],
            open: vec![],
            next_item: 1,
            covered_to: reply_part(33),
            removed: vec![],
        };
        let j = SessionJournal::restore(kept);
        assert_eq!(
            j.first_uncovered_turn(),
            34,
            "a stretch skipped after the newest entry stays covered"
        );
        assert_eq!(j.fresh_turns(40), 7);
        assert_eq!(j.journal().written(), 4);
        assert!(!j.journal().pending());
    }

    #[test]
    fn a_fold_given_up_on_waits_an_hour_before_the_next() {
        let mut j = SessionJournal::new();
        let at = Instant::now();
        assert!(j.may_fold(at));
        j.fold_failed(at);
        assert!(!j.may_fold(at + FOLD_RETRY_AFTER - Duration::from_secs(1)));
        assert!(j.may_fold(at + FOLD_RETRY_AFTER));
    }

    #[test]
    fn a_new_session_counts_turn_zero() {
        let j = SessionJournal::new();
        assert_eq!(j.fresh_turns(0), 1);
        assert_eq!(j.first_uncovered_turn(), 0);
    }
}
