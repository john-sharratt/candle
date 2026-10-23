//! A character's journal, and when a stretch of its window is worth asking about.
//!
//! What the journal holds is [`Journal`]'s. What is the character's is the
//! cadence: a draft every [`EVERY_TURNS`] turns of its window, staggered across
//! the cast; one forced before unread turns fall off the window; and the turns a
//! day roll-over clears, kept until a draft takes them.

use journal_core::journal::{Holds, Journal};
use journal_core::record;
use journal_core::section::{JournalPrompt, Kept};
use journal_core::span::Span;
use serde_json::Value;

use crate::engine::journal::origin::span_turn;
use crate::engine::journal::style::NPC;
use crate::engine::window::{Turn, Window};

/// Turns that must land since the last look before the next draft.
pub const EVERY_TURNS: u64 = 16;
/// How near the window's cap the oldest unlooked turn may sit before a draft is
/// forced, so a slow or failing draft cannot let turns fall off unread.
pub const EVICTION_MARGIN: usize = 4;

/// A stretch of a character's life its journal does not cover yet, as the
/// guardian is shown it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waiting {
    pub turns: usize,
    pub from_ms: u64,
    pub to_ms: u64,
    /// Whether the journal holds no entry at all. The character's prompt has no
    /// journal section then, so there is nothing for it to weigh a stretch
    /// against and the stretch is written up without asking.
    pub empty: bool,
}

#[derive(Debug)]
pub struct JournalState {
    journal: Journal,
    stagger: u64,
    /// Turns a day roll-over cleared from the window before the journal covered
    /// them, oldest first. The window is the only place turns live, so they are
    /// kept here until a draft takes them.
    stranded: Vec<Turn>,
}

impl Holds for JournalState {
    fn journal(&self) -> &Journal {
        &self.journal
    }

    fn journal_mut(&mut self) -> &mut Journal {
        &mut self.journal
    }
}

impl JournalState {
    pub fn new(npc_id: u64) -> Self {
        Self::holding(npc_id, Journal::new())
    }

    /// Rebuild from what the substrate held.
    ///
    /// Turn ids belong to one run of the window, which numbers from 1 again when
    /// the daemon starts, so what the entries covered says nothing about the new
    /// window's turns: coverage starts at zero and the first draft covers
    /// whatever has landed since the character woke.
    pub fn restore(npc_id: u64, kept: Kept) -> Self {
        Self::holding(npc_id, Journal::restore(kept, 0))
    }

    fn holding(npc_id: u64, journal: Journal) -> Self {
        Self {
            journal,
            // Spreads the cast's first drafts over a cadence rather than
            // landing them all on the same tick.
            stagger: npc_id % EVERY_TURNS,
            stranded: Vec::new(),
        }
    }

    /// The journal as the character's conversation should hold it now.
    pub fn prompt(&self) -> JournalPrompt {
        self.journal.prompt(&NPC)
    }

    /// Everything the journal API says about this journal.
    pub fn view(&self) -> Value {
        record::view(&self.journal, &NPC)
    }

    /// Keep the turns a day roll-over is about to clear, so a day's last stretch
    /// is asked about like any other. Turns an unfinished draft already holds are
    /// not taken twice.
    pub fn strand(&mut self, window: &Window) {
        let floor = if self.journal.pending() {
            self.journal.looked_to()
        } else {
            self.journal.covered_to()
        };
        self.stranded.extend(window.after(floor).cloned());
    }

    /// The stretch a draft would be about now, when one is worth asking about:
    /// the turns the window holds past what the journal covers, and any a
    /// roll-over stranded, once enough have landed or the window is nearly full.
    pub fn waiting(&self, window: &Window) -> Option<Waiting> {
        if !self.due(window) {
            return None;
        }
        let turns = self.stretch(window);
        let (first, last) = (turns.first()?, turns.last()?);
        Some(Waiting {
            turns: turns.len(),
            from_ms: first.at_ms,
            to_ms: last.at_ms,
            empty: self.journal.entries().next().is_none(),
        })
    }

    fn stretch(&self, window: &Window) -> Vec<Turn> {
        let mut turns = self.stranded.clone();
        turns.extend(window.after(self.journal.covered_to()).cloned());
        turns
    }

    fn due(&self, window: &Window) -> bool {
        if self.journal.pending() {
            return false;
        }
        if !self.stranded.is_empty() {
            return true;
        }
        let looked_to = self.journal.looked_to();
        let fresh = window.after(looked_to).count() as u64;
        let threshold = match looked_to {
            0 => EVERY_TURNS + self.stagger,
            _ => EVERY_TURNS,
        };
        if fresh >= threshold {
            return true;
        }
        let nearly_full = window.len() + EVICTION_MARGIN >= window.cap();
        nearly_full && window.oldest_id().is_some_and(|id| id > looked_to)
    }

    /// Start a draft over the turns since the last entry. `None` when one is
    /// already running or there is nothing new to draft from.
    pub fn begin(&mut self, window: &Window) -> Option<Span> {
        let turns = self.stretch(window).iter().map(span_turn).collect();
        let span = self.journal.begin(turns, window.newest_id())?;
        self.stranded.clear();
        Some(span)
    }
}

#[cfg(test)]
mod tests {
    use journal_core::entry::Entry;

    use super::*;
    use crate::engine::event::{Event, EventKind, Salience};

    fn said(window: &mut Window, n: u64) {
        for i in 0..n {
            window.push_npc(format!("tell — to Pax: hello {i}"), 1000 * i);
        }
    }

    fn entry_to(to_turn: u64) -> Entry {
        Entry {
            id: 0,
            from_turn: 1,
            to_turn,
            from_ms: 0,
            to_ms: 1,
            claims: vec![],
            intend: vec!["carry on".into()],
            opened: vec![],
            resolved: vec![],
        }
    }

    #[test]
    fn looking_moves_looked_to_and_not_covered_to() {
        let mut w = Window::with_default_cap();
        said(&mut w, 5);
        let mut s = JournalState::new(0);
        s.begin(&w).unwrap();
        let j = s.journal();
        assert_eq!((j.looked_to(), j.covered_to()), (w.newest_id(), 0));
    }

    #[test]
    fn nothing_is_due_before_the_cadence() {
        let mut w = Window::with_default_cap();
        let s = JournalState::new(0);
        said(&mut w, EVERY_TURNS - 1);
        assert!(!s.due(&w));
        said(&mut w, 1);
        assert!(s.due(&w));
    }

    #[test]
    fn the_first_draft_is_staggered_by_character_and_later_ones_are_not() {
        let mut w = Window::with_default_cap();
        said(&mut w, EVERY_TURNS + 5);
        assert!(!JournalState::new(6).due(&w), "stagger 6 is not yet due");
        assert!(JournalState::new(5).due(&w));
        let mut s = JournalState::new(6);
        said(&mut w, 1);
        assert!(s.due(&w));
        let span = s.begin(&w).unwrap();
        s.journal_mut().nothing_to_write(&span);
        said(&mut w, EVERY_TURNS);
        assert!(
            s.due(&w),
            "after the first look the stagger no longer applies"
        );
    }

    #[test]
    fn two_characters_get_different_first_thresholds() {
        let mut w = Window::with_default_cap();
        said(&mut w, EVERY_TURNS);
        assert!(JournalState::new(16).due(&w));
        assert!(!JournalState::new(17).due(&w));
    }

    #[test]
    fn a_pending_draft_is_never_due_again() {
        let mut w = Window::with_default_cap();
        let mut s = JournalState::new(0);
        said(&mut w, 40);
        assert!(s.begin(&w).is_some());
        assert!(!s.due(&w));
        assert!(s.begin(&w).is_none(), "single-flight");
    }

    #[test]
    fn a_no_is_final_for_its_stretch_and_the_next_draft_starts_after_it() {
        let mut w = Window::with_default_cap();
        let mut s = JournalState::new(0);
        said(&mut w, 20);
        let span = s.begin(&w).unwrap();
        s.journal_mut().nothing_to_write(&span);
        assert_eq!(s.journal().covered_to(), 20);
        assert!(!s.due(&w), "the same twenty turns are not examined again");
        said(&mut w, EVERY_TURNS);
        assert!(s.due(&w));
        let next = s.begin(&w).unwrap();
        assert_eq!(
            next.from_turn, 21,
            "only what happened since is asked about"
        );
    }

    #[test]
    fn an_abandoned_draft_retries_the_same_turns() {
        let mut w = Window::with_default_cap();
        let mut s = JournalState::new(0);
        said(&mut w, 20);
        let first = s.begin(&w).unwrap();
        s.journal_mut().abandon();
        assert!(!s.journal().pending());
        said(&mut w, EVERY_TURNS);
        let again = s.begin(&w).unwrap();
        assert_eq!(again.from_turn, first.from_turn);
        assert!(again.to_turn > first.to_turn);
    }

    #[test]
    fn a_kept_entry_moves_covered_to_and_the_next_span_starts_after_it() {
        let mut w = Window::with_default_cap();
        let mut s = JournalState::new(0);
        said(&mut w, 20);
        let span = s.begin(&w).unwrap();
        s.journal_mut().kept(entry_to(span.to_turn));
        assert_eq!(s.journal().covered_to(), 20);
        said(&mut w, 3);
        assert_eq!(s.begin(&w).unwrap().from_turn, 21);
    }

    #[test]
    fn nothing_to_begin_from_is_none() {
        let mut w = Window::with_default_cap();
        let mut s = JournalState::new(0);
        assert!(s.begin(&w).is_none());
        said(&mut w, 2);
        let span = s.begin(&w).unwrap();
        s.journal_mut().kept(entry_to(span.to_turn));
        assert!(s.begin(&w).is_none(), "everything is already covered");
    }

    #[test]
    fn the_eviction_backstop_fires_when_unlooked_turns_are_about_to_fall_off() {
        let mut w = Window::new(20);
        let s = JournalState::new(15);
        said(&mut w, 15);
        assert!(!s.due(&w));
        said(&mut w, 1);
        assert!(
            s.due(&w),
            "window within the margin of full and nothing looked at"
        );
        let mut looked = JournalState::new(15);
        looked.begin(&w).unwrap();
        looked.journal_mut().abandon();
        said(&mut w, 1);
        assert!(
            !looked.due(&w),
            "what has been looked at falling off is not a loss"
        );
    }

    #[test]
    fn nothing_is_waiting_before_the_cadence_and_a_stretch_is_after() {
        let mut w = Window::with_default_cap();
        let s = JournalState::new(0);
        said(&mut w, 3);
        assert_eq!(s.waiting(&w), None);
        said(&mut w, EVERY_TURNS);
        let waiting = s.waiting(&w).expect("a stretch has landed");
        assert_eq!(waiting.turns as u64, EVERY_TURNS + 3);
        assert!(waiting.empty);
    }

    #[test]
    fn nothing_is_waiting_while_a_draft_runs() {
        let mut w = Window::with_default_cap();
        said(&mut w, 20);
        let mut s = JournalState::new(0);
        s.begin(&w).unwrap();
        assert_eq!(s.waiting(&w), None);
    }

    #[test]
    fn turns_a_roll_over_clears_are_kept_for_the_next_draft() {
        let mut w = Window::with_default_cap();
        said(&mut w, 3);
        let mut s = JournalState::new(0);
        s.strand(&w);
        w.roll_over();
        said(&mut w, 2);
        let waiting = s
            .waiting(&w)
            .expect("stranded turns are always worth asking about");
        assert_eq!(waiting.turns, 5);
        let span = s.begin(&w).unwrap();
        assert_eq!(span.turns.len(), 5);
        assert_eq!(s.waiting(&w), None, "they are taken once");
    }

    #[test]
    fn a_restored_journal_drafts_the_whole_of_a_new_window() {
        let kept = Kept {
            entries: vec![Entry {
                id: 3,
                ..entry_to(120)
            }],
            open: Vec::new(),
            next_item: 1,
        };
        let mut s = JournalState::restore(0, kept);
        assert_eq!(s.journal().written(), 3);
        let mut w = Window::with_default_cap();
        said(&mut w, 20);
        let span = s
            .begin(&w)
            .expect("turns past the old run's ids are still new");
        assert_eq!((span.from_turn, span.to_turn), (1, 20));
    }

    #[test]
    fn a_span_lists_what_may_be_cited_and_nothing_ambient() {
        let mut w = Window::with_default_cap();
        w.push_npc("tell — to Pax: hi", 1);
        w.push_event(&Event::new(
            0,
            2,
            Salience::IDLE,
            EventKind::Description {
                text: "The conduit hums.".into(),
            },
        ));
        w.push_npc("reflect — thinking it over", 3);
        let mut s = JournalState::new(0);
        let span = s.begin(&w).unwrap();
        assert_eq!(span.citable_ids(), vec![1]);
        assert!(span.turn(2).is_some() && span.turn(9).is_none());
    }

    #[test]
    fn the_prompt_and_the_view_are_rendered_in_the_characters_voice() {
        let mut s = JournalState::new(0);
        let kept = s.journal_mut().kept(entry_to(4));
        assert_eq!(s.prompt().sections[0].text, kept.render(&NPC));
        let view = s.view();
        assert_eq!(view["written"], 1);
        assert_eq!(view["entries"][0]["text"], kept.render(&NPC));
    }
}
