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
            self.sealed_snapshots.remove(&slot);
            if !self.slot_idle(slot) {
                continue;
            }
            match self.model.park_recurrent(slot.0) {
                Ok(true) => tracing::debug!(
                    target: "candle_conversation::scheduler::recurrent_park",
                    slot = slot.0,
                    "an idle slot's recurrent state is parked in its record"
                ),
                Ok(false) => {}
                Err(e) => tracing::warn!("failed to park slot {slot}'s recurrent state: {e}"),
            }
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
        if let (Some(conversation), Some(target)) = (
            self.slot_conversations.get(&slot).cloned(),
            self.slot_targets.get(&slot).copied(),
        ) {
            self.restore_recurrent_state(slot, &conversation, target.timeline);
        }
        if self.model.recurrent_parked(slot.0) {
            return Err(ConversationError::Channel(format!(
                "slot {slot}'s recurrent state is parked and its record could not be read back"
            )));
        }
        Ok(())
    }
}
