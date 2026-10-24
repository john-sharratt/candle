//! What one journal holds, and the bookkeeping of what it covers.
//!
//! The journal is its entries and the open items: exactly what its writer reads
//! in its system prompt, and what the substrate holds for it as sections of its
//! conversation ([`crate::section`]). It stays short by folding, not by
//! forgetting: the steps of a day that is over fold into one entry for the day,
//! a week's days into the week, and so on up to years ([`crate::fold`]). The
//! rest is bookkeeping: which turns the journal covers, which have been looked
//! at, and whether a draft or a fold is running.
//!
//! **When a draft is due is the caller's.** A character in a game drafts every
//! sixteen turns of its perception window and before a turn falls off it; a
//! coding session drafts on its own cadence. Both hand [`Journal::begin`] the
//! turns that have landed past [`Journal::covered_to`].

use std::collections::VecDeque;

use crate::entry::{Entry, Item};
use crate::fold::Fold;
use crate::period::Period;
use crate::record::DraftRecord;
use crate::section::{entry_section, JournalPrompt, Kept};
use crate::span::{Span, SpanTurn};
use crate::style::Style;

/// Open items held at once. Past it the oldest is dropped: a list of everything
/// ever left unfinished is the same problem the journal exists to solve.
pub const MAX_OPEN: usize = 6;
/// Draft records kept for inspection.
pub const KEPT_DRAFTS: usize = 16;

/// Where a journal is held, so one draft workflow runs against any holder: a
/// bare [`Journal`], or a caller's state that keeps its own cadence beside it.
pub trait Holds: Send {
    fn journal(&self) -> &Journal;
    fn journal_mut(&mut self) -> &mut Journal;
}

#[derive(Debug, Clone)]
pub struct Journal {
    entries: VecDeque<Entry>,
    open: Vec<Item>,
    covered_to: u64,
    /// The newest turn id at the last draft that started. A "no" and an
    /// abandoned draft both count as having looked.
    looked_to: u64,
    pending: bool,
    next_entry: u64,
    next_item: u64,
    drafts: VecDeque<DraftRecord>,
    /// Entries a fold or a forget took whose sections no reconcile has swept
    /// yet ([`Self::swept`]); kept in the record, so a run that restarts in
    /// between still names them.
    removed: Vec<u64>,
}

impl Default for Journal {
    fn default() -> Self {
        Self::new()
    }
}

impl Holds for Journal {
    fn journal(&self) -> &Journal {
        self
    }

    fn journal_mut(&mut self) -> &mut Journal {
        self
    }
}

impl Journal {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            open: Vec::new(),
            covered_to: 0,
            looked_to: 0,
            pending: false,
            next_entry: 1,
            next_item: 1,
            drafts: VecDeque::new(),
            removed: Vec::new(),
        }
    }

    /// Rebuild from what the substrate held: the entries, oldest first, the open
    /// items that followed them, and the next open-item id to hand out — an item
    /// that was resolved and has left the open set still used its id.
    ///
    /// `covered_to` is the caller's: the turn ids the entries cover mean nothing
    /// to a caller whose turns are numbered afresh on each run, and everything to
    /// one whose turn ids are the conversation's own.
    pub fn restore(kept: Kept, covered_to: u64) -> Self {
        let mut j = Self::new();
        for e in kept.entries {
            j.apply(e);
        }
        j.open = kept.open;
        j.removed = kept.removed;
        j.next_item = j.next_item.max(kept.next_item);
        j.covered_to = covered_to;
        j.looked_to = covered_to;
        j
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

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter()
    }

    /// The latest entry when it is a step: the one a skipped stretch runs on
    /// over ([`Self::nothing_to_write`]).
    pub fn latest_step(&self) -> Option<&Entry> {
        self.entries.back().filter(|e| e.period == Period::Step)
    }

    /// Start a fold: single flight with drafts, so neither works from a
    /// journal the other is changing. `false` while either is running.
    pub fn begin_fold(&mut self) -> bool {
        if self.pending {
            return false;
        }
        self.pending = true;
        true
    }

    /// The journal once `staged` — the entry a fold made, under the id
    /// [`Self::stage`] gave it — has replaced the entries `fold` names, in
    /// the place the first of them held.
    fn folded(&self, fold: &Fold, staged: &Entry) -> Self {
        let mut next = self.clone();
        let at = next
            .entries
            .iter()
            .position(|e| fold.ids.contains(&e.id))
            .unwrap_or(next.entries.len());
        next.entries.retain(|e| !fold.ids.contains(&e.id));
        next.entries
            .insert(at.min(next.entries.len()), staged.clone());
        next.next_entry = next.next_entry.max(staged.id + 1);
        next.removed.extend(fold.ids.iter().copied());
        next
    }

    /// The journal as a conversation should hold it once `staged` has replaced
    /// the entries `fold` names, for the substrate to take before the state
    /// does; it names their sections as gone.
    pub fn prompt_after_fold(&self, fold: &Fold, staged: &Entry, style: &Style) -> JournalPrompt {
        self.folded(fold, staged).prompt(style)
    }

    /// A reconcile has swept the sections `prompt` named as removed: they need
    /// not be named again. Entries removed since that prompt was taken stay.
    pub fn swept(&mut self, prompt: &JournalPrompt) {
        self.removed
            .retain(|id| !prompt.removed.contains(&entry_section(*id)));
    }

    /// The fold was kept: `staged` replaces the entries `fold` names.
    pub fn kept_fold(&mut self, fold: &Fold, staged: Entry) {
        *self = Self {
            pending: false,
            ..self.folded(fold, &staged)
        };
    }

    /// Entries ever kept, including those that have left memory.
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

    /// Start a draft over `turns`, the turns past what the journal covers, oldest
    /// first; `newest` is the newest turn the caller holds, which counts as looked
    /// at. `None` while a draft is running or when there are no turns.
    pub fn begin(&mut self, turns: Vec<SpanTurn>, newest: u64) -> Option<Span> {
        if self.pending {
            return None;
        }
        let span = Span::of(turns)?;
        self.pending = true;
        self.looked_to = newest.max(span.to_turn);
        Some(span)
    }

    /// The entry with the ids it would be kept under, without keeping it. The
    /// substrate write needs the ids before the entry is in memory, so a write
    /// that fails leaves nothing here to take back. A staged entry is passed to
    /// [`Self::kept`] unchanged.
    pub fn stage(&self, mut entry: Entry) -> Entry {
        entry.id = self.next_entry;
        let fresh = entry.opened.iter_mut().filter(|i| i.id == 0);
        for (item, id) in fresh.zip(self.next_item..) {
            item.id = id;
        }
        entry
    }

    /// [`Self::stage`], and the id taken at once: an entry kept while this one
    /// is still being written down — a closing entry beside a fold in flight
    /// ([`Self::keep_closing`]) — takes the next. A reservation never kept is
    /// an id skipped, as an id once used is never reused.
    pub fn reserve(&mut self, entry: Entry) -> Entry {
        let entry = self.stage(entry);
        self.next_entry = self.next_entry.max(entry.id + 1);
        entry
    }

    /// The draft was written. Assigns the entry its id and any new open item its
    /// id, and returns the entry as kept.
    pub fn kept(&mut self, entry: Entry) -> Entry {
        let entry = self.stage(entry);
        self.apply(entry.clone());
        self.pending = false;
        entry
    }

    /// Keep an entry the caller wrote itself, beside whatever draft is in flight,
    /// which it leaves to finish as it would have.
    pub fn keep_closing(&mut self, entry: Entry) -> Entry {
        let entry = self.stage(entry);
        self.apply(entry.clone());
        entry
    }

    /// The stretch was no step of its own — more of the task the latest entry
    /// is about — so it is that entry's: its span runs on to the stretch's
    /// end, and the turns are found under it. The verdict is final for those
    /// turns: `covered_to` moves past them, so the next draft is asked only
    /// about what has happened since. With no step to run on — no entry yet,
    /// or only folded ones — the stretch is only covered.
    pub fn nothing_to_write(&mut self, span: &Span) {
        if let Some(latest) = self.entries.back_mut().filter(|e| e.period == Period::Step) {
            latest.to_turn = latest.to_turn.max(span.to_turn);
            latest.to_ms = latest.to_ms.max(span.to_ms);
        }
        self.covered_to = self.covered_to.max(span.to_turn);
        self.pending = false;
    }

    /// The draft did not finish. `covered_to` stays put, so the next draft covers
    /// the same turns again.
    pub fn abandon(&mut self) {
        self.pending = false;
    }

    /// Take the entries numbered `ids` out of the journal, and return the ones
    /// that were held. Ids already used are not handed out again, and the open
    /// items stay: they belong to the writer, not to the page that raised them.
    pub fn forget(&mut self, ids: &[u64]) -> Vec<u64> {
        let gone: Vec<u64> = self
            .entries
            .iter()
            .map(|e| e.id)
            .filter(|id| ids.contains(id))
            .collect();
        self.entries.retain(|e| !gone.contains(&e.id));
        self.removed.extend(gone.iter().copied());
        gone
    }

    /// What the substrate should hold for this journal now.
    pub fn snapshot(&self) -> Kept {
        Kept {
            entries: self.entries.iter().cloned().collect(),
            open: self.open.clone(),
            next_item: self.next_item,
            covered_to: self.covered_to,
            removed: self.removed.clone(),
        }
    }

    /// The journal as a conversation should hold it once `span` is settled
    /// with nothing written ([`Self::nothing_to_write`]): the same lines, and a
    /// record in which the latest entry runs on over the stretch.
    pub fn prompt_covering(&self, span: &Span, style: &Style) -> JournalPrompt {
        let mut next = self.clone();
        next.nothing_to_write(span);
        next.prompt(style)
    }

    /// The journal as a conversation should hold it now.
    pub fn prompt(&self, style: &Style) -> JournalPrompt {
        JournalPrompt::of(&self.snapshot(), style)
            .without(self.removed.iter().map(|id| entry_section(*id)).collect())
    }

    /// The journal as a conversation should hold it once `staged` is kept, for
    /// the substrate to take before the state does.
    pub fn prompt_after(&self, staged: &Entry, style: &Style) -> JournalPrompt {
        let mut next = self.clone();
        next.apply(staged.clone());
        next.prompt(style)
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
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::entry::testing::{entry_to, item};
    use crate::entry::Kind;
    use crate::fold;
    use crate::section::restore;
    use crate::span::testing::turn;
    use crate::style::testing::PLAIN;
    use crate::workflow::Outcome;

    /// `n` citable turns numbered from `from`.
    fn turns(from: u64, n: u64) -> Vec<SpanTurn> {
        (from..from + n)
            .map(|id| turn(id, id * 1000, Some(Kind::Did), "said something"))
            .collect()
    }

    #[test]
    fn only_the_latest_drafts_are_kept() {
        let mut j = Journal::new();
        for n in 1..=(KEPT_DRAFTS as u64 + 3) {
            let span = Span::of(turns(n, 1)).unwrap();
            j.record(DraftRecord::of(
                &span,
                Duration::ZERO,
                &Outcome::Nothing { why: String::new() },
            ));
        }
        let firsts: Vec<u64> = j.drafts().map(|d| d.from_turn).collect();
        assert_eq!(firsts.len(), KEPT_DRAFTS);
        assert_eq!(firsts[0], 4);
        assert_eq!(*firsts.last().unwrap(), KEPT_DRAFTS as u64 + 3);
    }

    #[test]
    fn looking_moves_looked_to_and_not_covered_to() {
        let mut j = Journal::new();
        j.begin(turns(1, 5), 5).unwrap();
        assert_eq!((j.looked_to(), j.covered_to()), (5, 0));
    }

    #[test]
    fn looked_to_is_never_behind_the_span_it_was_handed() {
        let mut j = Journal::new();
        j.begin(turns(1, 5), 0).unwrap();
        assert_eq!(j.looked_to(), 5);
    }

    #[test]
    fn a_running_draft_is_single_flight() {
        let mut j = Journal::new();
        assert!(j.begin(turns(1, 3), 3).is_some());
        assert!(j.pending());
        assert!(j.begin(turns(1, 3), 3).is_none(), "single-flight");
    }

    #[test]
    fn nothing_to_begin_from_is_none_and_starts_nothing() {
        let mut j = Journal::new();
        assert!(j.begin(Vec::new(), 9).is_none());
        assert!(!j.pending());
    }

    #[test]
    fn a_no_is_final_for_its_stretch() {
        let mut j = Journal::new();
        let span = j.begin(turns(1, 20), 20).unwrap();
        j.nothing_to_write(&span);
        assert_eq!(j.covered_to(), 20);
        assert!(!j.pending());
    }

    #[test]
    fn an_abandoned_draft_leaves_its_turns_uncovered() {
        let mut j = Journal::new();
        j.begin(turns(1, 20), 20).unwrap();
        j.abandon();
        assert!(!j.pending());
        assert_eq!(j.covered_to(), 0);
    }

    #[test]
    fn a_kept_entry_moves_covered_to() {
        let mut j = Journal::new();
        let span = j.begin(turns(1, 20), 20).unwrap();
        j.kept(entry_to(span.to_turn));
        assert_eq!(j.covered_to(), 20);
        assert!(!j.pending());
    }

    #[test]
    fn ids_are_assigned_at_keep_and_never_reused() {
        let mut j = Journal::new();
        let mut e = entry_to(3);
        e.opened = vec![item(0, "a"), item(0, "b")];
        let kept = j.kept(e);
        assert_eq!(kept.id, 1);
        assert_eq!(
            kept.opened.iter().map(|i| i.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        let mut e = entry_to(4);
        e.opened = vec![item(0, "c")];
        let kept = j.kept(e);
        assert_eq!(kept.id, 2);
        assert_eq!(kept.opened[0].id, 3);
    }

    #[test]
    fn forgetting_an_entry_removes_it_and_leaves_the_numbering_and_open_items() {
        let mut j = Journal::new();
        let mut first = entry_to(3);
        first.opened = vec![item(0, "the gate")];
        j.kept(first);
        j.kept(entry_to(6));
        j.kept(entry_to(9));

        assert_eq!(j.forget(&[2, 7]), vec![2], "only a held entry is forgotten");
        let left: Vec<u64> = j.entries().map(|e| e.id).collect();
        assert_eq!(left, vec![1, 3]);
        assert_eq!(j.open().len(), 1, "the open item is the writer's");
        assert_eq!(j.written(), 3, "an id once used is not used again");
        assert_eq!(
            j.prompt(&PLAIN).sections.len(),
            3,
            "two entries and the open items"
        );
        assert!(j.forget(&[2]).is_empty(), "forgetting twice finds nothing");
    }

    #[test]
    fn a_staged_entry_changes_nothing_and_is_kept_under_the_same_ids() {
        let mut j = Journal::new();
        let mut e = entry_to(3);
        e.opened = vec![item(0, "a"), item(0, "b")];
        let staged = j.stage(e);
        assert_eq!(j.entries().count(), 0);
        assert!(j.open().is_empty());
        assert_eq!(
            j.stage(staged.clone()),
            staged,
            "staging twice is staging once"
        );
        assert_eq!(j.open_after(&staged).len(), 2);
        assert!(
            j.open().is_empty(),
            "the open set is not touched until kept"
        );
        assert_eq!(j.kept(staged.clone()), staged);
        assert_eq!(j.open().len(), 2);
    }

    #[test]
    fn a_resolved_item_leaves_and_a_restated_one_is_updated_in_place() {
        let mut j = Journal::new();
        let mut e = entry_to(3);
        e.opened = vec![
            item(0, "who holds shields?"),
            item(0, "is the door jammed?"),
        ];
        j.kept(e);
        let mut e = entry_to(6);
        e.opened = vec![item(2, "is the east door still jammed?")];
        e.resolved = vec![1];
        j.kept(e);
        assert_eq!(j.open(), &[item(2, "is the east door still jammed?")]);
    }

    #[test]
    fn open_items_are_capped_and_the_oldest_goes_first() {
        let mut j = Journal::new();
        for n in 0..(MAX_OPEN as u64 + 2) {
            let mut e = entry_to(n + 1);
            e.opened = vec![item(0, &format!("q{n}"))];
            j.kept(e);
        }
        assert_eq!(j.open().len(), MAX_OPEN);
        assert_eq!(j.open()[0].text, "q2");
    }

    #[test]
    fn every_entry_is_held_until_it_folds() {
        let mut j = Journal::new();
        for n in 0..12 {
            j.kept(entry_to(n + 1));
        }
        assert_eq!(j.entries().count(), 12);
    }

    #[test]
    fn the_prompt_after_a_staged_entry_is_what_the_prompt_is_once_it_is_kept() {
        let mut j = Journal::new();
        for n in 0..3 {
            j.kept(entry_to(n + 1));
        }
        let staged = j.stage(entry_to(50));
        let before = j.prompt(&PLAIN);
        let after = j.prompt_after(&staged, &PLAIN);
        assert_eq!(
            j.prompt(&PLAIN),
            before,
            "staging must not change the state"
        );
        j.kept(staged);
        assert_eq!(j.prompt(&PLAIN), after);
        assert!(after.removed.is_empty());
    }

    /// Steps 1 and 2 of day 1, step 3 of day 2.
    fn two_days() -> Journal {
        let mut j = Journal::new();
        for (to_turn, from_ms, to_ms) in [(3, 1_100, 1_200), (6, 1_500, 1_900), (9, 2_100, 2_200)] {
            j.kept(Entry {
                from_ms,
                to_ms,
                ..entry_to(to_turn)
            });
        }
        j
    }

    fn day_one() -> Fold {
        Fold {
            into: Period::Day,
            ids: vec![1, 2],
        }
    }

    #[test]
    fn a_fold_replaces_its_entries_in_their_place_under_the_next_id() {
        let mut j = two_days();
        assert!(j.begin_fold());
        let all: Vec<Entry> = j.entries().cloned().collect();
        let staged = j.stage(fold::entry(
            &day_one(),
            &all,
            "Day one.".into(),
            "All of it.".into(),
        ));
        assert_eq!(staged.id, 4);
        let after = j.prompt_after_fold(&day_one(), &staged, &PLAIN);
        assert_eq!(j.entries().count(), 3, "nothing changes until it is kept");
        j.kept_fold(&day_one(), staged);
        let held: Vec<(u64, Period)> = j.entries().map(|e| (e.id, e.period)).collect();
        assert_eq!(held, [(4, Period::Day), (3, Period::Step)]);
        assert!(!j.pending());
        assert_eq!(j.prompt(&PLAIN).sections, after.sections);
        assert_eq!(after.removed, ["journal/1", "journal/2"]);
        assert_eq!(j.kept(entry_to(12)).id, 5, "ids keep counting up");
    }

    /// The review's case: a closing entry kept while a fold is still being
    /// written down takes the id after the fold's, not the same one.
    #[test]
    fn an_entry_kept_beside_a_fold_in_flight_takes_the_next_id() {
        let mut j = two_days();
        assert!(j.begin_fold());
        let all: Vec<Entry> = j.entries().cloned().collect();
        let staged = j.reserve(fold::entry(
            &day_one(),
            &all,
            "Day one.".into(),
            "All of it.".into(),
        ));
        let closing = j.keep_closing(entry_to(12));
        assert_eq!((staged.id, closing.id), (4, 5));
        j.kept_fold(&day_one(), staged);
        let ids: Vec<u64> = j.entries().map(|e| e.id).collect();
        assert_eq!(ids, [4, 3, 5], "every id held once");
    }

    /// The review's case: the sections a fold took are named by every prompt
    /// — the record included, so a restart still sweeps them — until a
    /// reconcile has swept them.
    #[test]
    fn folded_sections_are_named_until_a_reconcile_sweeps_them() {
        let mut j = two_days();
        assert!(j.begin_fold());
        let all: Vec<Entry> = j.entries().cloned().collect();
        let staged = j.reserve(fold::entry(
            &day_one(),
            &all,
            "Day one.".into(),
            "All of it.".into(),
        ));
        j.kept_fold(&day_one(), staged);
        let prompt = j.prompt(&PLAIN);
        assert_eq!(prompt.sweep(), ["journal/1", "journal/2"]);
        let restored = Journal::restore(restore(prompt.record()).unwrap(), 0);
        assert_eq!(restored.prompt(&PLAIN).sweep(), ["journal/1", "journal/2"]);

        j.forget(&[3]);
        j.swept(&prompt);
        assert_eq!(
            j.prompt(&PLAIN).sweep(),
            ["journal/3"],
            "what was forgotten after the sweep's prompt is still named"
        );
    }

    #[test]
    fn a_fold_is_single_flight_with_drafts() {
        let mut j = two_days();
        j.begin(turns(10, 2), 11).unwrap();
        assert!(!j.begin_fold(), "a draft is running");
        j.abandon();
        assert!(j.begin_fold());
        assert!(j.begin(turns(10, 2), 11).is_none(), "a fold is running");
    }

    #[test]
    fn a_skip_after_a_fold_runs_on_over_no_entry() {
        let mut j = Journal::new();
        j.kept(Entry {
            period: Period::Day,
            ..entry_to(5)
        });
        assert!(j.latest_step().is_none());
        let span = j.begin(turns(6, 2), 7).unwrap();
        j.nothing_to_write(&span);
        assert_eq!(
            j.entries().next().unwrap().to_turn,
            5,
            "a day is not run on"
        );
        assert_eq!(j.covered_to(), 7);
    }

    #[test]
    fn an_empty_journal_has_an_empty_prompt() {
        assert!(Journal::new().prompt(&PLAIN).is_empty());
    }

    #[test]
    fn restoring_replays_entries_and_does_not_reuse_their_ids() {
        let mut a = entry_to(3);
        a.id = 5;
        a.opened = vec![item(9, "open one")];
        let mut b = entry_to(7);
        b.id = 6;
        let kept = Kept {
            entries: vec![a, b],
            open: vec![item(9, "open one")],
            next_item: 12,
            covered_to: 7,
            removed: vec![],
        };
        let mut j = Journal::restore(kept, 0);
        assert_eq!(j.covered_to(), 0, "the caller says what is covered");
        assert_eq!(j.open(), &[item(9, "open one")]);
        let next = j.kept(entry_to(9));
        assert_eq!(next.id, 7);
        let mut e = entry_to(10);
        e.opened = vec![item(0, "fresh")];
        assert_eq!(
            j.kept(e).opened[0].id,
            12,
            "an item that was resolved still used its id"
        );
    }

    #[test]
    fn a_restore_that_keeps_its_coverage_looks_only_past_it() {
        let kept = Kept {
            entries: vec![entry_to(40)],
            open: Vec::new(),
            next_item: 1,
            covered_to: 52,
            removed: vec![],
        };
        let j = Journal::restore(kept.clone(), kept.covered_to);
        assert_eq!((j.covered_to(), j.looked_to()), (52, 52));
        assert!(!j.pending());
    }

    #[test]
    fn a_stretch_settled_with_nothing_is_in_the_record() {
        let mut j = Journal::new();
        j.kept(entry_to(40));
        let span = j
            .begin(
                vec![
                    turn(41, 41, Some(Kind::Did), "a"),
                    turn(52, 52, Some(Kind::Did), "b"),
                ],
                52,
            )
            .unwrap();
        let covering = j.prompt_covering(&span, &PLAIN);
        assert_eq!(j.covered_to(), 40, "nothing moves until the record is kept");
        let recorded = restore(covering.record()).unwrap();
        assert_eq!(recorded.covered_to, 52);
        assert_eq!(
            recorded.entries[0].to_turn, 52,
            "the stretch is the latest entry's: its span runs on over it"
        );
        assert_eq!(
            covering.sections,
            j.prompt(&PLAIN).sections,
            "the entry's line does not change, so nothing is sealed again"
        );
        j.nothing_to_write(&span);
        assert_eq!(j.prompt(&PLAIN), covering);
    }

    #[test]
    fn a_stretch_settled_before_any_entry_is_only_covered() {
        let mut j = Journal::new();
        let span = j.begin(vec![turn(1, 1, Some(Kind::Did), "a")], 1).unwrap();
        j.nothing_to_write(&span);
        assert_eq!((j.covered_to(), j.entries().count()), (1, 0));
    }

    #[test]
    fn the_count_written_survives_entries_leaving_the_journal() {
        let mut j = Journal::new();
        assert_eq!(j.written(), 0);
        for n in 0..4 {
            j.kept(entry_to(n + 1));
        }
        j.forget(&[1, 2]);
        assert_eq!(j.written(), 4);
        assert_eq!(j.entries().count(), 2);
    }

    #[test]
    fn a_closing_entry_is_kept_beside_a_draft_in_flight() {
        let mut j = Journal::new();
        j.begin(turns(1, 3), 3).unwrap();
        let closing = j.keep_closing(entry_to(2));
        assert_eq!(closing.id, 1);
        assert!(j.pending(), "the draft in flight carries on");
    }
}
