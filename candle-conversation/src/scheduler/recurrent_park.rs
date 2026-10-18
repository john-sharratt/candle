//! An idle conversation's recurrent state lives in its record, not on the device.
//!
//! A conversation keeps its slot between turns for minutes, and on a hybrid
//! its recurrent store with it — on blackworld (2026-10-18) 155 regions,
//! 2.37 GB, mostly held by conversations no forward was running. The seal
//! already writes that state to the substrate: the store is exported at the
//! turn's seal, after its closing tail has been prefilled, and the slot then
//! sits idle holding exactly what the record holds. So once the record is
//! durable the store is dropped ([`ManagedBatchedModel::park_recurrent`]), and
//! the slot's next use stages the state back from the record
//! ([`Scheduler::unpark_from_record`]) — no host copy kept in between, and no
//! readback beyond the seal's own.

use super::*;

impl Scheduler {
    /// Drop the store of every idle slot whose last seal the persistence layer
    /// now reports durable. A slot back in a forward since its seal is left
    /// alone: its next seal is what will park it.
    pub(super) fn park_durable_states(&mut self) {
        if self.sealed_snapshots.is_empty() {
            return;
        }
        let durable: HashSet<(TimelineId, u32)> = self
            .distinct_substrates()
            .iter()
            .flat_map(|c| c.drain_durable_snapshots())
            .collect();
        if durable.is_empty() {
            return;
        }
        let reached: Vec<SequenceId> = self
            .sealed_snapshots
            .iter()
            .filter(|(_, sealed)| durable.contains(sealed))
            .map(|(&slot, _)| slot)
            .collect();
        for slot in reached {
            let Some(sealed) = self.sealed_snapshots.remove(&slot) else {
                continue;
            };
            if !self.slot_idle(slot) {
                continue;
            }
            match self.model.park_recurrent(slot.0) {
                Ok(true) => {
                    self.parked_at.insert(slot, sealed);
                    tracing::debug!(
                        target: "candle_conversation::scheduler::recurrent_park",
                        slot = slot.0,
                        "an idle slot's recurrent state is parked in its record"
                    );
                }
                Ok(false) => {}
                Err(e) => tracing::warn!("failed to park slot {slot}'s recurrent state: {e}"),
            }
        }
    }

    /// Whether `slot`'s timeline is marked unresumed — content read and never
    /// extended, which no slot continues — so its state is never read after a
    /// seal. Not a transient timeline: that one persists nothing, but its turns
    /// may continue in-process on the same slot.
    fn never_resumed(&self, slot: SequenceId) -> bool {
        match (
            self.slot_conversations.get(&slot),
            self.slot_targets.get(&slot),
        ) {
            (Some(conversation), Some(target)) => {
                conversation.read().is_timeline_unresumed(target.timeline)
            }
            _ => false,
        }
    }

    /// Drop a just-sealed slot's store when its timeline is unresumed. Such a
    /// timeline writes no snapshot, so the durable signal that parks
    /// every other slot never comes for it, and its store stood until the slot
    /// was released — minutes, for an ingested document.
    pub(super) fn park_never_resumed(&mut self, slot: SequenceId) {
        if !self.never_resumed(slot) || !self.slot_idle(slot) {
            return;
        }
        self.sealed_snapshots.remove(&slot);
        if let Err(e) = self.model.park_recurrent(slot.0) {
            tracing::warn!("failed to park slot {slot}'s recurrent state: {e}");
        }
    }

    /// Whether nothing is running or waiting on `slot`: no turn view carved
    /// from it, no decode or prefill of its own, no section ingest, and no
    /// ephemeral fork waiting for it.
    pub(super) fn slot_idle(&self, slot: SequenceId) -> bool {
        !self.turn_views.values().any(|v| v.parent_id == slot)
            && !self.active_decodes.contains_key(&slot)
            && !self
                .active_prefills
                .iter()
                .any(|p| p.work.sequence_id == slot)
            && !self.prefill_queue.iter().any(|w| w.sequence_id == slot)
            && !self.section_queue.iter().any(|s| s.sequence_id == slot)
            && !self
                .active_section_ingests
                .iter()
                .any(|s| s.sequence_id == slot)
            && !self.fork_waits_on(slot)
    }

    /// Stage `slot`'s state back from its record if it is parked, before
    /// anything reads it. A parked slot whose record cannot be read is an
    /// error, not a slot that starts over: that would read fluently and
    /// remember nothing.
    pub(super) fn unpark_from_record(&mut self, slot: SequenceId) -> Result<(), ConversationError> {
        if !self.model.recurrent_parked(slot.0) {
            return Ok(());
        }
        // A never-resumed timeline has no record of its own, and its lineage's
        // would be some ancestor's state: wrong for a slot that continued it.
        if self.never_resumed(slot) {
            return Err(ConversationError::Channel(format!(
                "slot {slot} continues a timeline marked unresumed; its recurrent \
                 state was dropped at its last seal"
            )));
        }
        // **Only what parking dropped comes back: the GDN layers.** The slot
        // kept its PLE history and its QSA index, so the record's aux blob —
        // which a resume installs into an empty slot — must not be laid over
        // them: installed on 3432 it put a second copy of the index on a live
        // one, and every wave after found the index 602 tokens off its K/V.
        // And from this slot's own record, at the turn it parked at, never an
        // ancestor's through the lineage.
        let staged = match (
            self.parked_at.get(&slot).copied(),
            self.slot_conversations.get(&slot).cloned(),
        ) {
            (Some((timeline, turn)), Some(conversation)) => {
                match conversation.read_recurrent_snapshot(timeline) {
                    Ok(Some(payload)) if payload.turn_index == turn => {
                        let layers: Vec<ExportedLayerState> = payload
                            .layers
                            .iter()
                            .map(|l| ExportedLayerState {
                                layer_index: l.layer_index,
                                n_v_heads: l.n_v_heads,
                                d_v: l.d_v,
                                d_k: l.d_k,
                                state: l.state.clone(),
                                conv_channels: l.conv_channels,
                                conv_tail_cols: l.conv_tail_cols,
                                conv_tail: l.conv_tail.clone(),
                            })
                            .collect();
                        self.model
                            .stage_recurrent(slot.0, payload.schedule_hash, Arc::from(layers))
                            .map_err(ConversationError::Model)
                            .map(|_| ())
                    }
                    Ok(Some(payload)) => Err(ConversationError::Channel(format!(
                        "slot {slot} parked at turn {turn} but its record holds turn {}",
                        payload.turn_index
                    ))),
                    Ok(None) => Err(ConversationError::Channel(format!(
                        "slot {slot} parked at turn {turn} but its record holds no snapshot"
                    ))),
                    Err(e) => Err(e),
                }
            }
            _ => Err(ConversationError::Channel(format!(
                "slot {slot} is parked with no record of where"
            ))),
        };
        if let Err(e) = staged {
            return Err(ConversationError::Channel(format!(
                "slot {slot}'s recurrent state is parked and its record could not be read back: {e}"
            )));
        }
        self.parked_at.remove(&slot);
        if self.model.recurrent_parked(slot.0) {
            return Err(ConversationError::Channel(format!(
                "slot {slot}'s recurrent state is parked and its record could not be read back"
            )));
        }
        Ok(())
    }
}
