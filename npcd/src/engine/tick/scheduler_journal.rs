//! The scheduler's access to each character's journal: reading it, starting a
//! draft over the stretch it does not yet cover, and forgetting or closing
//! parts of it.
//!
//! A draft is marked in flight before it is returned, so two callers cannot be
//! handed the same stretch of a character's life. The journal is read under its
//! lock without being cloned out, and the Pulse feed is filtered before anything
//! is copied, so one character's feed copies only its own ticks.

use super::*;

impl Scheduler {
    /// The stretch of this character's life its journal does not cover yet,
    /// once there is enough of it to be worth asking about. `None` before the
    /// cadence, while a draft is running, and for a character nobody has woken.
    pub fn journal_waiting(&self, npc_id: u64) -> Option<Waiting> {
        let inboxes = self.inboxes.lock().unwrap();
        let inbox = inboxes.get(&npc_id)?;
        let waiting = inbox.journal.lock().unwrap().waiting(&inbox.window);
        waiting
    }

    /// Start a draft now. `None` for a character nobody has woken, while a draft
    /// is running, and when nothing is new.
    ///
    /// The state is marked as having a draft in flight before this returns, so
    /// two callers cannot both be handed the stretch.
    pub fn journal_begin(&self, npc_id: u64) -> Option<Due> {
        self.inboxes.lock().unwrap().get_mut(&npc_id)?.begin_draft()
    }

    /// Read a character's journal under its lock, without cloning it out.
    /// `None` for a character nobody has woken.
    pub fn journal_of<T, F>(&self, npc_id: u64, f: F) -> Option<T>
    where
        F: FnOnce(&JournalState) -> T,
    {
        let journal = {
            let inboxes = self.inboxes.lock().unwrap();
            Arc::clone(&inboxes.get(&npc_id)?.journal)
        };
        let held = journal.lock().unwrap();
        Some(f(&held))
    }

    /// Take entries out of a character's journal. Returns the ids that were held
    /// and the journal as it now reads, or `None` for a character nobody has woken.
    pub fn journal_forget(&self, npc_id: u64, ids: &[u64]) -> Option<(Vec<u64>, JournalPrompt)> {
        self.journal_of_mut(npc_id, |j| {
            let gone = j.journal_mut().forget(ids);
            (gone, j.prompt())
        })
    }

    /// Close a chapter of a character's journal: keep the entry recording the
    /// mission it handed in ([`closing::closing_entry`]), at world time `at_ms`.
    /// Returns the journal as it now reads, or `None` for a character nobody
    /// has woken.
    pub fn journal_close_chapter(
        &self,
        npc_id: u64,
        mission: &Mission,
        at_ms: u64,
    ) -> Option<JournalPrompt> {
        self.journal_of_mut(npc_id, |j| {
            let journal = j.journal_mut();
            let covered_to = journal.covered_to();
            journal.keep_closing(closing::closing_entry(mission, at_ms, covered_to));
            j.prompt()
        })
    }

    pub(super) fn journal_of_mut<T, F>(&self, npc_id: u64, f: F) -> Option<T>
    where
        F: FnOnce(&mut JournalState) -> T,
    {
        let journal = {
            let inboxes = self.inboxes.lock().unwrap();
            Arc::clone(&inboxes.get(&npc_id)?.journal)
        };
        let mut held = journal.lock().unwrap();
        Some(f(&mut held))
    }

    /// Give a character the journal it kept before — rebuilt from the
    /// substrate at start-up — in place of the empty one it woke with.
    pub fn set_journal(&self, npc_id: u64, state: JournalState) {
        if let Some(inbox) = self.inboxes.lock().unwrap().get_mut(&npc_id) {
            inbox.journal = Arc::new(Mutex::new(state));
        }
    }

    /// Of the newest `limit` ticks, those `keep` keeps, newest last — filtered
    /// before any is copied, so a feed for one character copies only its own.
    /// `keep` runs under the ring's lock and must take none of its own. The
    /// Pulse view's feed.
    pub fn recent_where(
        &self,
        limit: usize,
        keep: impl Fn(&TickRecord) -> bool,
    ) -> Vec<TickRecord> {
        // Once, outside the walk, so every row in one page is measured against
        // the same instant.
        let now = self.born.elapsed();
        let r = self.recent.lock().unwrap();
        let skip = r.len().saturating_sub(limit);
        r.iter()
            .skip(skip)
            .filter(|t| keep(t))
            .cloned()
            .map(|mut t| {
                t.ms_ago = now.saturating_sub(t.at).as_millis() as u64;
                t
            })
            .collect()
    }

    /// The newest tick of `npc` still in the ring, copying nothing: where a
    /// stream of its acts starts from.
    pub fn latest_tick_of(&self, npc: u64) -> Option<u64> {
        self.recent
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|t| t.npc_id == npc)
            .map(|t| t.tick)
    }

    /// A snapshot of every character's loop state.
    pub fn census(&self) -> Vec<Census> {
        // Read once, outside the map, so every row in one census is measured
        // against the same instant — otherwise a large cast reports ages that
        // disagree by however long the walk took.
        let now = self.born.elapsed();
        let inboxes = self.inboxes.lock().unwrap();
        let mut v: Vec<Census> = inboxes
            .values()
            .map(|i| {
                // One guard for all three reads: guards taken in a struct
                // literal live to the end of the statement, so a second `lock()`
                // there would wait on the first.
                let (journal_written, journal_open, journal_drafting) = {
                    let held = i.journal.lock().unwrap();
                    let j = held.journal();
                    (j.written(), j.open().len(), j.pending())
                };
                Census {
                    npc_id: i.npc_id,
                    readiness: i.readiness(),
                    inbox_depth: i.depth(),
                    heartbeat_ms: i.heartbeat.as_millis() as u64,
                    ticks: i.ticks,
                    events_seen: i.events_seen,
                    window_turns: i.window.len(),
                    window_cap: i.window.cap(),
                    faded: i.window.faded(),
                    journal_written,
                    journal_open,
                    journal_drafting,
                    day: i.day.current(),
                    acted_ms_ago: i
                        .last_act_at
                        // `saturating_sub`: a tick that landed between the reading
                        // above and this line would otherwise be an age in the
                        // future, which renders as a very large number.
                        .map(|at| now.saturating_sub(at).as_millis() as u64),
                    last_news_ms: i.last_news_ms,
                    last_nudge_ms: i.last_nudge_ms,
                }
            })
            .collect();
        v.sort_by_key(|c| c.npc_id);
        v
    }
}
