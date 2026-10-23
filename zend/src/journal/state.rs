//! A session's journal, and when its stretch of dialogue is due a draft.
//!
//! What the journal holds is [`Journal`]'s. What is the session's is the cadence
//! ([`SessionJournal::due`]): a draft every [`EVERY_TURNS`] sealed dialogue
//! turns, or sooner when the dialogue's projection has started leaving out turns
//! no draft has looked at — a turn is looked at before the session loses sight
//! of it. At most one draft runs at a time.
//!
//! The journal counts in parts ([`super::origin`]), whose numbering is the
//! turns' own and stable across restarts, so a journal restored from its record
//! carries on from the last part its entries covered.

use journal_core::journal::{Holds, Journal};
use journal_core::section::Kept;
use journal_core::span::{Span, SpanTurn};

use crate::journal::origin::{reply_part, turn_of};

/// Sealed dialogue turns that must land since the last look before the next
/// draft.
pub const EVERY_TURNS: u32 = 16;

#[derive(Debug, Default)]
pub struct SessionJournal {
    journal: Journal,
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild from the record the conversation's metadata held. Part numbers
    /// are the conversation's own, so what the record says is covered — by an
    /// entry or by a draft that found nothing to add — stays covered.
    pub fn restore(kept: Kept) -> Self {
        let covered_to = kept.covered_to;
        Self {
            journal: Journal::restore(kept, covered_to),
        }
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

    /// Whether a draft is due now that turn `newest` has sealed. `pressed` is
    /// whether the dialogue's last projection left out a turn at or past
    /// [`Self::first_uncovered_turn`] ([`super::pressure`]).
    pub fn due(&self, newest: u32, pressed: bool) -> bool {
        if self.journal.pending() {
            return false;
        }
        self.fresh_turns(newest) >= EVERY_TURNS || pressed
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
            claims: vec![],
            intend: vec!["carry on".into()],
            opened: vec![],
            resolved: vec![],
        }
    }

    #[test]
    fn nothing_is_due_before_sixteen_turns_and_a_draft_is_after() {
        let j = SessionJournal::new();
        assert!(!j.due(EVERY_TURNS - 2, false), "turns 0..=14 are fifteen");
        assert!(j.due(EVERY_TURNS - 1, false), "turns 0..=15 are sixteen");
    }

    #[test]
    fn selection_pressure_makes_a_draft_due_before_the_cadence() {
        let j = SessionJournal::new();
        assert!(j.due(2, true));
    }

    #[test]
    fn a_running_draft_is_never_due_again() {
        let mut j = SessionJournal::new();
        j.begin(parts(0..=20), 20).unwrap();
        assert!(!j.due(60, true));
        assert!(j.begin(parts(0..=20), 20).is_none(), "single flight");
    }

    #[test]
    fn the_cadence_counts_from_the_last_look() {
        let mut j = SessionJournal::new();
        let span = j.begin(parts(0..=15), 15).unwrap();
        j.journal_mut().nothing_to_write(&span);
        assert_eq!(j.fresh_turns(20), 5);
        assert!(!j.due(30, false));
        assert!(j.due(31, false));
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
        assert!(!j.due(16, false), "looked at, so the cadence starts over");
        assert!(j.due(16, true), "but pressure on an unread turn still asks");
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
    fn a_new_session_counts_turn_zero() {
        let j = SessionJournal::new();
        assert_eq!(j.fresh_turns(0), 1);
        assert_eq!(j.first_uncovered_turn(), 0);
    }
}
