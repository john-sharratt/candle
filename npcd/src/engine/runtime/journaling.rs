//! Drafting a character's journal entries as tasks that do not hold up its turns.
//!
//! A stretch of turns waiting in the journal is drafted in a task of its own, so
//! the character carries on turn by turn while the draft runs in conversations of
//! its own. A draft that does not land costs only its entry, and the same turns
//! are covered again at the next draft. Writing an entry also folds the days,
//! weeks and months it closes, and the journal keeper already holds those folds,
//! so they need not be named again.

use super::*;

impl Runtime {
    /// What a journal draft checks its claims against: the tower's state when the
    /// character has a body in a world that has one, nothing otherwise.
    pub(super) fn journal_world(&self, npc_id: u64) -> Snapshot {
        self.body_of(npc_id)
            .and_then(|(hosted, _)| {
                hosted.with_both(|_, sim| sim.tower.as_ref().map(Snapshot::of_tower))
            })
            .unwrap_or_default()
    }

    /// Start a draft of the stretch waiting in `npc_id`'s journal, as a task of
    /// its own. `false` when there is no stretch to draft: another draft is
    /// running, the stretch was taken since the guardian looked, or the
    /// character is suspended or at rest.
    pub fn draft_journal(self: &Arc<Self>, npc_id: u64) -> bool {
        if self.scheduler.is_held(npc_id) {
            return false;
        }
        let Some(due) = self.scheduler.journal_begin(npc_id) else {
            return false;
        };
        self.spawn_journal(npc_id, due);
        true
    }

    /// Draft the stretch in `due` as a task of its own: one answer writes the
    /// entry or, once the journal holds one, skips the stretch with a reason.
    ///
    /// Nobody waits on it: the character carries on turn by turn while the draft
    /// runs in conversations of its own. A runtime with no executor to spawn on
    /// cannot draft, and says so, handing the stretch back to be covered later.
    pub fn spawn_journal(self: &Arc<Self>, npc_id: u64, due: Due) {
        match &self.spawner {
            Some(handle) => {
                let rt = Arc::clone(self);
                handle.spawn(async move { rt.journal_now(npc_id, due).await });
            }
            None => {
                due.state.lock().unwrap().journal_mut().abandon();
                tracing::warn!("npc {npc_id}: no executor to write a journal entry on");
            }
        }
    }

    /// Draft one journal entry and log how it ended. Logged, never returned: an
    /// entry that did not land costs an entry, and the same turns are covered
    /// again at the next draft.
    pub(super) async fn journal_now(&self, npc_id: u64, due: Due) {
        let started = Instant::now();
        let not_started = |why: String| {
            let mut held = due.state.lock().unwrap();
            let state = held.journal_mut();
            state.abandon();
            state.record(DraftRecord::not_started(&due.span, started.elapsed(), why));
        };
        let Some(minds) = self.minds.read().unwrap().clone() else {
            not_started("no mind to ask".into());
            return;
        };
        let Some(who) = self.persona_of_async(npc_id).await else {
            not_started("no persona for this character".into());
            return;
        };
        let world = self.journal_world(npc_id);
        let within = self.within(npc_id);
        let drafted = minds
            .journal(npc_id, &who.as_persona(), &within, &due, &world)
            .await;
        match &drafted {
            Ok((outcome, timing)) => due
                .state
                .lock()
                .unwrap()
                .journal_mut()
                .record(DraftRecord::of(&due.span, started.elapsed(), outcome).timed(*timing)),
            Err(e) => {
                tracing::warn!("npc {npc_id}: the journal entry could not start — {e:#}");
                not_started(format!("{e:#}"));
                return;
            }
        }
        match drafted {
            Ok((
                Drafted::Wrote {
                    entry,
                    notes,
                    rounds,
                },
                _,
            )) => tracing::info!(
                "npc {npc_id}: journal entry {} kept in {:?} after {rounds} round(s), {} check note(s)",
                entry.id,
                started.elapsed(),
                notes.len(),
            ),
            Ok((Drafted::Nothing { why }, _)) => tracing::info!(
                "npc {npc_id}: nothing worth a journal entry in turns {}–{} — {why}",
                due.span.from_turn,
                due.span.to_turn,
            ),
            Ok((Drafted::Abandoned(why), _)) => {
                tracing::warn!("npc {npc_id}: the journal entry was not written — {why:?}")
            }
            Err(_) => return,
        }
        // A character's time is its own: the newest turn just covered says
        // which of its world's days, weeks and months are over, and those fold.
        let folds = minds
            .fold_journal(
                npc_id,
                &who.as_persona(),
                &within,
                &due.state,
                due.span.to_ms,
            )
            .await;
        // Each kept fold was held by `Minds::keep_journal`, whose reconcile
        // swept the sections it took: they need not be named again.
        if folds.iter().any(|f| matches!(f, Folded::Kept { .. })) {
            let mut held = due.state.lock().unwrap();
            let swept = held.journal().prompt(&NPC);
            held.journal_mut().swept(&swept);
        }
        for folded in folds {
            match folded {
                Folded::Kept { fold, entry } => tracing::info!(
                    "npc {npc_id}: journal {} {:?} folded into entry {} — {}",
                    fold.into.name(),
                    fold.ids,
                    entry.id,
                    entry.summary,
                ),
                Folded::Abandoned(why) => {
                    tracing::warn!("npc {npc_id}: a journal fold was not kept — {why:?}")
                }
                Folded::Nothing => {}
            }
        }
    }
}
