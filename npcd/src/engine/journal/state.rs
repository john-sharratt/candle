//! What one character's journal holds, and when a stretch is worth asking about.
//!
//! The journal is the newest [`IN_PROMPT`] entries and the open items: exactly
//! what the character reads in its system prompt, and what the substrate holds
//! for it as sections of its conversation ([`crate::engine::journal::section`]).
//! It is rebuilt from that record at start-up ([`JournalState::restore`]). The
//! rest is bookkeeping: which turns the journal covers, which have been looked
//! at, and whether a draft is running.

use std::collections::VecDeque;

use crate::engine::journal::entry::{Entry, Item};
use crate::engine::journal::record::DraftRecord;
use crate::engine::journal::section::{JournalPrompt, Kept};
use crate::engine::window::{Turn, Window};

/// Turns that must land since the last look before the next draft.
pub const EVERY_TURNS: u64 = 16;
/// How near the window's cap the oldest unlooked turn may sit before a draft is
/// forced, so a slow or failing draft cannot let turns fall off unread.
pub const EVICTION_MARGIN: usize = 4;
/// Open items held at once. Past it the oldest is dropped: a list of everything
/// ever left unfinished is the same problem the journal exists to solve.
pub const MAX_OPEN: usize = 6;
/// Entries the journal holds, and the character reads: when a newer one is kept
/// the oldest falls out of the prompt and out of the substrate with it.
pub const IN_PROMPT: usize = 5;
/// Draft records kept for the API.
pub const KEPT_DRAFTS: usize = 16;

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

/// The window turns a draft is about, copied out so the draft can run with no
/// lock held.
#[derive(Clone, Debug)]
pub struct Span {
    pub from_turn: u64,
    pub to_turn: u64,
    pub from_ms: u64,
    pub to_ms: u64,
    pub turns: Vec<Turn>,
}

impl Span {
    /// The ids a claim may cite.
    pub fn citable_ids(&self) -> Vec<u64> {
        self.turns
            .iter()
            .filter(|t| t.origin.citable())
            .map(|t| t.id)
            .collect()
    }

    pub fn turn(&self, id: u64) -> Option<&Turn> {
        self.turns.iter().find(|t| t.id == id)
    }
}

#[derive(Debug)]
pub struct JournalState {
    entries: VecDeque<Entry>,
    open: Vec<Item>,
    covered_to: u64,
    /// The newest turn id at the last draft that started. A "no" and an
    /// abandoned draft both count as having looked, so a draft
    /// starts only every [`EVERY_TURNS`] turns, never on each one.
    looked_to: u64,
    pending: bool,
    next_entry: u64,
    next_item: u64,
    stagger: u64,
    drafts: VecDeque<DraftRecord>,
    /// Turns a day roll-over cleared from the window before the journal covered
    /// them, oldest first. The window is the only place turns live, so they are
    /// kept here until a draft takes them.
    stranded: Vec<Turn>,
}

impl JournalState {
    pub fn new(npc_id: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            open: Vec::new(),
            covered_to: 0,
            looked_to: 0,
            pending: false,
            next_entry: 1,
            next_item: 1,
            // Spreads the cast's first drafts over a cadence rather than
            // landing them all on the same tick.
            stagger: npc_id % EVERY_TURNS,
            drafts: VecDeque::new(),
            stranded: Vec::new(),
        }
    }

    /// Note how a draft ended, oldest dropped past [`KEPT_DRAFTS`].
    pub fn record(&mut self, draft: DraftRecord) {
        self.drafts.push_back(draft);
        while self.drafts.len() > KEPT_DRAFTS {
            self.drafts.pop_front();
        }
    }

    /// The most recent drafts, oldest first.
    pub fn drafts(&self) -> impl Iterator<Item = &DraftRecord> {
        self.drafts.iter()
    }

    pub fn looked_to(&self) -> u64 {
        self.looked_to
    }

    /// Rebuild from the entries the substrate held, oldest first, and the open
    /// items that followed them. `next_item` is the next open-item id to hand
    /// out: an item that was resolved and has left the open set still used its id.
    ///
    /// Turn ids belong to one run of the window, which numbers from 1 again when
    /// the daemon starts, so what the entries covered says nothing about the new
    /// window's turns: `covered_to` starts at zero and the first draft covers
    /// whatever has landed since the character woke.
    pub fn restore(npc_id: u64, kept: Vec<Entry>, open: Vec<Item>, next_item: u64) -> Self {
        let mut s = Self::new(npc_id);
        for e in kept {
            s.apply(e);
        }
        s.open = open;
        s.next_item = s.next_item.max(next_item);
        s.covered_to = 0;
        s.pending = false;
        s
    }

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter()
    }

    /// Entries ever kept for this character, including those from earlier runs
    /// that have left memory.
    pub fn written(&self) -> u64 {
        self.next_entry - 1
    }

    pub fn open(&self) -> &[Item] {
        &self.open
    }

    /// The id the next new open item will take.
    pub fn next_item(&self) -> u64 {
        self.next_item
    }

    pub fn covered_to(&self) -> u64 {
        self.covered_to
    }

    pub fn pending(&self) -> bool {
        self.pending
    }

    /// Keep the turns a day roll-over is about to clear, so a day's last stretch
    /// is asked about like any other. Turns an unfinished draft already holds are
    /// not taken twice.
    pub fn strand(&mut self, window: &Window) {
        let floor = if self.pending {
            self.looked_to
        } else {
            self.covered_to
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
            empty: self.entries.is_empty(),
        })
    }

    fn stretch(&self, window: &Window) -> Vec<Turn> {
        let mut turns = self.stranded.clone();
        turns.extend(window.after(self.covered_to).cloned());
        turns
    }

    fn due(&self, window: &Window) -> bool {
        if self.pending {
            return false;
        }
        if !self.stranded.is_empty() {
            return true;
        }
        let fresh = window.after(self.looked_to).count() as u64;
        let threshold = match self.looked_to {
            0 => EVERY_TURNS + self.stagger,
            _ => EVERY_TURNS,
        };
        if fresh >= threshold {
            return true;
        }
        let nearly_full = window.len() + EVICTION_MARGIN >= window.cap();
        nearly_full && window.oldest_id().is_some_and(|id| id > self.looked_to)
    }

    /// Start a draft over the turns since the last entry. `None` when one is
    /// already running or there is nothing new to draft from.
    pub fn begin(&mut self, window: &Window) -> Option<Span> {
        if self.pending {
            return None;
        }
        let turns = self.stretch(window);
        let (first, last) = (turns.first()?, turns.last()?);
        let span = Span {
            from_turn: first.id,
            to_turn: last.id,
            from_ms: first.at_ms,
            to_ms: last.at_ms,
            turns,
        };
        self.stranded.clear();
        self.pending = true;
        self.looked_to = window.newest_id();
        Some(span)
    }

    /// The entry with the ids it would be kept under, without keeping it. The
    /// substrate write needs the ids (they are in the entry's tag and its JSON)
    /// before the entry is in memory, so a write that fails leaves nothing here
    /// to take back. A staged entry is passed to [`Self::kept`] unchanged.
    pub fn stage(&self, mut entry: Entry) -> Entry {
        entry.id = self.next_entry;
        let fresh = entry.opened.iter_mut().filter(|i| i.id == 0);
        for (item, id) in fresh.zip(self.next_item..) {
            item.id = id;
        }
        entry
    }

    /// The draft was written. Assigns the entry its id and any new open item
    /// its id, and returns the entry as kept.
    pub fn kept(&mut self, entry: Entry) -> Entry {
        let entry = self.stage(entry);
        self.apply(entry.clone());
        self.pending = false;
        entry
    }

    /// The gate said there was nothing worth writing in the span. The verdict
    /// is final for those turns: `covered_to` moves past them, so the next draft
    /// is asked only about what has happened since.
    pub fn nothing_to_write(&mut self, span: &Span) {
        self.covered_to = span.to_turn;
        self.pending = false;
    }

    /// The draft did not finish. `covered_to` stays put, so the next trigger
    /// covers the same turns again.
    pub fn abandon(&mut self) {
        self.pending = false;
    }

    /// Take the entries numbered `ids` out of the journal, and return the ones
    /// that were held. Ids already used are not handed out again, and the open
    /// items stay: they belong to the character, not to the page that raised
    /// them.
    pub fn forget(&mut self, ids: &[u64]) -> Vec<u64> {
        let gone: Vec<u64> = self
            .entries
            .iter()
            .map(|e| e.id)
            .filter(|id| ids.contains(id))
            .collect();
        self.entries.retain(|e| !gone.contains(&e.id));
        gone
    }

    /// The journal as a conversation should hold it now.
    pub fn prompt(&self) -> JournalPrompt {
        JournalPrompt::of(&self.snapshot())
    }

    /// The journal as a conversation should hold it once `staged` is kept, for
    /// the substrate to take before the state does.
    pub fn prompt_after(&self, staged: &Entry) -> JournalPrompt {
        let mut next = Self::new(0);
        next.entries = self.entries.clone();
        next.open = self.open.clone();
        next.next_entry = self.next_entry;
        next.next_item = self.next_item;
        next.apply(staged.clone());
        next.prompt()
    }

    fn snapshot(&self) -> Kept {
        Kept {
            entries: self.entries.iter().cloned().collect(),
            open: self.open.clone(),
            next_item: self.next_item,
        }
    }

    /// The open items once `entry` is kept: settled ones gone, restated ones
    /// updated in place, new ones appended, the oldest dropped past the cap.
    pub fn open_after(&self, entry: &Entry) -> Vec<Item> {
        let mut open = self.open.clone();
        open.retain(|i| !entry.resolved.contains(&i.id));
        for item in &entry.opened {
            match open.iter_mut().find(|o| o.id == item.id) {
                Some(held) => held.text = item.text.clone(),
                None => open.push(item.clone()),
            }
        }
        while open.len() > MAX_OPEN {
            open.remove(0);
        }
        open
    }

    fn apply(&mut self, entry: Entry) {
        self.next_entry = self.next_entry.max(entry.id + 1);
        self.covered_to = self.covered_to.max(entry.to_turn);
        for item in &entry.opened {
            self.next_item = self.next_item.max(item.id + 1);
        }
        self.open = self.open_after(&entry);
        self.entries.push_back(entry);
        while self.entries.len() > IN_PROMPT {
            self.entries.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
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
            intend: vec![],
            opened: vec![],
            resolved: vec![],
        }
    }

    fn item(id: u64, text: &str) -> Item {
        Item {
            id,
            text: text.into(),
        }
    }

    #[test]
    fn only_the_latest_drafts_are_kept() {
        use crate::engine::journal::workflow::Outcome;
        let mut s = JournalState::new(0);
        let span = |n: u64| Span {
            from_turn: n,
            to_turn: n,
            from_ms: 0,
            to_ms: 0,
            turns: vec![],
        };
        for n in 1..=(KEPT_DRAFTS as u64 + 3) {
            s.record(DraftRecord::of(
                &span(n),
                std::time::Duration::ZERO,
                &Outcome::Nothing { why: String::new() },
            ));
        }
        let firsts: Vec<u64> = s.drafts().map(|d| d.from_turn).collect();
        assert_eq!(firsts.len(), KEPT_DRAFTS);
        assert_eq!(firsts[0], 4);
        assert_eq!(*firsts.last().unwrap(), KEPT_DRAFTS as u64 + 3);
    }

    #[test]
    fn looking_moves_looked_to_and_not_covered_to() {
        let mut w = Window::with_default_cap();
        said(&mut w, 5);
        let mut s = JournalState::new(0);
        s.begin(&w).unwrap();
        assert_eq!((s.looked_to(), s.covered_to()), (w.newest_id(), 0));
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
        s.nothing_to_write(&span);
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
        s.nothing_to_write(&span);
        assert_eq!(s.covered_to(), 20);
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
        s.abandon();
        assert!(!s.pending());
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
        s.kept(entry_to(span.to_turn));
        assert_eq!(s.covered_to(), 20);
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
        s.kept(entry_to(span.to_turn));
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
        looked.abandon();
        said(&mut w, 1);
        assert!(
            !looked.due(&w),
            "what has been looked at falling off is not a loss"
        );
    }

    #[test]
    fn ids_are_assigned_at_keep_and_never_reused() {
        let mut s = JournalState::new(0);
        let mut e = entry_to(3);
        e.opened = vec![item(0, "a"), item(0, "b")];
        let kept = s.kept(e);
        assert_eq!(kept.id, 1);
        assert_eq!(
            kept.opened.iter().map(|i| i.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        let mut e = entry_to(4);
        e.opened = vec![item(0, "c")];
        let kept = s.kept(e);
        assert_eq!(kept.id, 2);
        assert_eq!(kept.opened[0].id, 3);
    }

    #[test]
    fn forgetting_an_entry_removes_it_and_leaves_the_numbering_and_open_items() {
        let mut s = JournalState::new(0);
        let mut first = entry_to(3);
        first.opened = vec![item(0, "the gate")];
        s.kept(first);
        s.kept(entry_to(6));
        s.kept(entry_to(9));

        assert_eq!(s.forget(&[2, 7]), vec![2], "only a held entry is forgotten");
        let left: Vec<u64> = s.entries().map(|e| e.id).collect();
        assert_eq!(left, vec![1, 3]);
        assert_eq!(s.open().len(), 1, "the open item is the character's");
        assert_eq!(s.written(), 3, "an id once used is not used again");
        assert_eq!(
            s.prompt().sections.len(),
            3,
            "two entries and the open items"
        );
        assert!(s.forget(&[2]).is_empty(), "forgetting twice finds nothing");
    }

    #[test]
    fn a_staged_entry_changes_nothing_and_is_kept_under_the_same_ids() {
        let mut s = JournalState::new(0);
        let mut e = entry_to(3);
        e.opened = vec![item(0, "a"), item(0, "b")];
        let staged = s.stage(e);
        assert_eq!(s.entries().count(), 0);
        assert!(s.open().is_empty());
        assert_eq!(
            s.stage(staged.clone()),
            staged,
            "staging twice is staging once"
        );
        assert_eq!(s.open_after(&staged).len(), 2);
        assert!(
            s.open().is_empty(),
            "the open set is not touched until kept"
        );
        assert_eq!(s.kept(staged.clone()), staged);
        assert_eq!(s.open().len(), 2);
    }

    #[test]
    fn a_resolved_item_leaves_and_a_restated_one_is_updated_in_place() {
        let mut s = JournalState::new(0);
        let mut e = entry_to(3);
        e.opened = vec![
            item(0, "who holds shields?"),
            item(0, "is the door jammed?"),
        ];
        s.kept(e);
        let mut e = entry_to(6);
        e.opened = vec![item(2, "is the east door still jammed?")];
        e.resolved = vec![1];
        s.kept(e);
        assert_eq!(s.open(), &[item(2, "is the east door still jammed?")]);
    }

    #[test]
    fn open_items_are_capped_and_the_oldest_goes_first() {
        let mut s = JournalState::new(0);
        for n in 0..(MAX_OPEN as u64 + 2) {
            let mut e = entry_to(n + 1);
            e.opened = vec![item(0, &format!("q{n}"))];
            s.kept(e);
        }
        assert_eq!(s.open().len(), MAX_OPEN);
        assert_eq!(s.open()[0].text, "q2");
    }

    #[test]
    fn only_the_newest_entries_are_held() {
        let mut s = JournalState::new(0);
        for n in 0..(IN_PROMPT as u64 + 3) {
            s.kept(entry_to(n + 1));
        }
        assert_eq!(s.entries().count(), IN_PROMPT);
        assert_eq!(s.entries().next().unwrap().id, 4);
    }

    #[test]
    fn the_prompt_after_a_staged_entry_is_what_the_prompt_is_once_it_is_kept() {
        let mut s = JournalState::new(0);
        for n in 0..(IN_PROMPT as u64 + 1) {
            s.kept(entry_to(n + 1));
        }
        let staged = s.stage(entry_to(50));
        let before = s.prompt();
        let after = s.prompt_after(&staged);
        assert_eq!(s.prompt(), before, "staging must not change the state");
        s.kept(staged);
        assert_eq!(s.prompt(), after);
        assert_eq!(after.aged_out.as_deref(), Some("journal/2"));
    }

    #[test]
    fn an_empty_journal_has_an_empty_prompt() {
        assert!(JournalState::new(0).prompt().is_empty());
    }

    #[test]
    fn restoring_replays_entries_and_does_not_reuse_their_ids() {
        let mut a = entry_to(3);
        a.id = 5;
        a.opened = vec![item(9, "open one")];
        let mut b = entry_to(7);
        b.id = 6;
        let mut s = JournalState::restore(0, vec![a, b], vec![item(9, "open one")], 12);
        assert_eq!(s.covered_to(), 0, "a new window numbers its turns afresh");
        assert_eq!(s.open(), &[item(9, "open one")]);
        let next = s.kept(entry_to(9));
        assert_eq!(next.id, 7);
        let mut e = entry_to(10);
        e.opened = vec![item(0, "fresh")];
        assert_eq!(
            s.kept(e).opened[0].id,
            12,
            "an item that was resolved still used its id"
        );
    }

    #[test]
    fn the_count_written_survives_entries_leaving_the_journal() {
        let mut s = JournalState::new(0);
        assert_eq!(s.written(), 0);
        for n in 0..(IN_PROMPT as u64 + 3) {
            s.kept(entry_to(n + 1));
        }
        assert_eq!(s.written(), IN_PROMPT as u64 + 3);
        assert_eq!(s.entries().count(), IN_PROMPT);
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
        let mut s = JournalState::restore(0, vec![entry_to(120)], Vec::new(), 0);
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
}
