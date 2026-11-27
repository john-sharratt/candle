//! Writes, folds and restores a character's journal inside its live
//! conversation.
//!
//! The journal is held as sections of the conversation, so the entries a
//! character has written reach it as part of what it reads. Which entries are
//! held is recorded in the conversation's metadata, and a character waking
//! rebuilds its journal from the conversation that holds it, taking the one
//! that kept the newest entry when several do. A character with no such record
//! wakes with an empty journal. Entries are weighed against the latest one
//! rather than logged request by request. Forgetting a journal empties its
//! record without tombstoning the conversation, which is the character's day;
//! the sections it holds are tombstoned with that conversation.

use super::*;

/// The sections a journal should be carried as.
pub(super) fn journal_wants(prompt: Option<&JournalPrompt>) -> Vec<Want<'_>> {
    prompt
        .map(|p| {
            p.sections
                .iter()
                .map(|s| Want {
                    name: &s.name,
                    text: &s.text,
                    sharing: Sharing::Owned,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The journal sections a conversation that did not see them leave may still hold.
pub(super) fn journal_sweep(prompt: Option<&JournalPrompt>) -> Vec<&str> {
    prompt.map(JournalPrompt::sweep).unwrap_or_default()
}

impl Minds {
    /// Write one journal entry, if the stretch in `due` warrants one — see
    /// [`crate::engine::journal`].
    ///
    /// Its write is put to the character's live conversation, as [`Minds::ask`]
    /// puts a question: on an ephemeral slot that writes nothing back, between
    /// the character's turns, holding its lock for the length of the decode. An
    /// entry that is kept reaches the character as a section of its journal — see
    /// [`Minds::keep_journal`].
    ///
    /// Whichever way it ends, `due`'s state has been told how.
    pub async fn journal(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &tools::Within,
        due: &Due,
        world: &(dyn World + Sync),
    ) -> anyhow::Result<(Outcome, Timing)> {
        if self.projected().is_none() {
            due.state.lock().unwrap().journal_mut().abandon();
            anyhow::bail!("no schema, so no journal collection to write to");
        }
        let mut desk = EngineDesk::new(self, npc_id, persona, within, WRITE_TOKENS);
        // A character's journal weighs every stretch against its latest entry:
        // it has no requests to log one by one.
        Ok(workflow::run(
            &mut desk,
            &*due.state,
            &due.span,
            world,
            &NPC,
            Skip::Allowed,
        )
        .await)
    }

    /// Fold every period of the character's journal that is over at world
    /// instant `now_ms`, one after another ([`journal_core::folding`]), asked
    /// of the character as a draft is. Each fold's outcome, in order; the last
    /// is never [`Folded::Kept`].
    pub async fn fold_journal<S: Holds>(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &tools::Within,
        state: &Mutex<S>,
        now_ms: u64,
    ) -> Vec<Folded> {
        let mut desk = EngineDesk::new(self, npc_id, persona, within, FOLD_TOKENS);
        let mut out = Vec::new();
        loop {
            let folded = folding::run(&mut desk, state, now_ms, &NPC).await;
            let more = matches!(folded, Folded::Kept { .. });
            out.push(folded);
            if !more {
                return out;
            }
        }
    }

    /// Hold `prompt` in the character's live conversation as its journal: submit
    /// the sections that are new, remove the ones that have aged out, and record
    /// the entries in the conversation's metadata so a restart can rebuild them.
    ///
    /// The sections are not waited on. They seal in the background, and each
    /// joins the prompt from the first turn after it has.
    pub async fn keep_journal(&self, npc_id: u64, prompt: &JournalPrompt) -> anyhow::Result<()> {
        self.hold_journal(npc_id, prompt, &[]).await
    }

    /// Hold `prompt` as [`Minds::keep_journal`] does, and also take the sections
    /// of the entries numbered `removed` out of the conversation, including ones
    /// an earlier run sealed into it.
    pub async fn hold_journal(
        &self,
        npc_id: u64,
        prompt: &JournalPrompt,
        removed: &[u64],
    ) -> anyhow::Result<()> {
        let gone: Vec<String> = removed.iter().map(|id| entry_section(*id)).collect();
        let sweep: Vec<&str> = journal_sweep(Some(prompt))
            .into_iter()
            .chain(gone.iter().map(String::as_str))
            .collect();
        let resident = self.resident_of(npc_id).ok_or_else(|| {
            anyhow::anyhow!("npc {npc_id} has no live conversation to journal in")
        })?;
        let mut live = resident.conversation.lock().await;
        let Live {
            sequence, journal, ..
        } = &mut *live;
        journal
            .reconcile(
                sequence,
                identity::JOURNAL,
                &journal_wants(Some(prompt)),
                Linking::Chain,
                Changes::Any,
                &sweep,
                None,
                tokio::time::sleep,
            )
            .await;
        self.record_journal(npc_id, &mut live, prompt.record())
    }

    /// Write the journal record to the conversation's metadata, unless it is
    /// already what that holds.
    pub(super) fn record_journal(
        &self,
        npc_id: u64,
        live: &mut Live,
        record: &str,
    ) -> anyhow::Result<()> {
        if live.journal_record == record {
            return Ok(());
        }
        let kept = BTreeMap::from([
            (META_OF.to_string(), npc_id.to_string()),
            (META_KEPT.to_string(), record.to_string()),
        ]);
        self.engine
            .set_conversation_metadata_many(live.sequence.timeline_id(), &kept)
            .map_err(|e| anyhow::anyhow!("the journal could not be recorded: {e:?}"))?;
        live.journal_record = record.to_string();
        Ok(())
    }

    /// Rebuild this character's journal from the conversation that holds it.
    /// Called as the character wakes, before its conversation is opened.
    ///
    /// A character with no such conversation, or none whose record parses, wakes
    /// with an empty journal. Of several, the one that kept the newest entry wins.
    pub fn restore_journal(&self, npc_id: u64) -> JournalState {
        let engine = &self.engine;
        let kept = engine
            .find_conversations_by_metadata(META_OF, &npc_id.to_string())
            .into_iter()
            .filter_map(|tl| engine.conversation_metadata_value(tl, META_KEPT))
            .filter_map(|record| section::restore(&record))
            .max_by_key(|k| k.newest());
        match kept {
            Some(k) => JournalState::restore(npc_id, k),
            None => JournalState::new(npc_id),
        }
    }

    /// Empty the record this character's journal is restored from, so it wakes
    /// with none. Returns how many conversations held one.
    ///
    /// The conversation is not tombstoned — it is the character's day. The
    /// sections it holds stay with it and are not read: a character whose journal
    /// is empty selects none, and an entry written later replaces the section of
    /// the same name. They are tombstoned with the conversation.
    pub fn forget_journal(&self, npc_id: u64) -> usize {
        let empty = JournalPrompt::of(
            &Kept {
                entries: Vec::new(),
                open: Vec::new(),
                next_item: 1,
                covered_to: 0,
                removed: Vec::new(),
            },
            &NPC,
        );
        let engine = &self.engine;
        let mut emptied = 0;
        for timeline in engine.find_conversations_by_metadata(META_OF, &npc_id.to_string()) {
            match engine.set_conversation_metadata(timeline, META_KEPT, empty.record()) {
                Ok(()) => emptied += 1,
                Err(e) => tracing::warn!(
                    "npc {npc_id}: journal record of conversation {timeline} could not be \
                     emptied: {e:?} — it is restored at the next start"
                ),
            }
        }
        emptied
    }
}
