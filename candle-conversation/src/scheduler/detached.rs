//! What a detached substrate's handles may still send, and why none of it
//! lands.
//!
//! A tenant detaching from a shared engine ([`SchedulerRequest::DetachSubstrate`])
//! has its slots freed and its timelines forgotten, but a `Sequence` handle of
//! its can outlive the detach on another thread — a task the tenant had not
//! yet joined. Slot ids recycle, so without care that handle's next request
//! would land on whatever another tenant's newest sequence took the id: a
//! `FreeSequence` freeing a live conversation, a `SubmitTurn` running one
//! tenant's prompt on another's slot and sealing it into the wrong substrate.
//!
//! So a detach reserves each of its slot ids in the session until its own
//! handle frees it — no new sequence takes the id — and every request naming a
//! reserved slot or a detached substrate is dropped unanswered: the caller sees
//! its reply channel close, exactly as it would on a stopped scheduler.

use std::collections::HashSet;

use super::SchedulerRequest;
use crate::projection::{Conversation, SubstrateRef};
use crate::sequence_handle::SequenceId;

/// The slots and substrates detached from this scheduler.
#[derive(Default)]
pub(super) struct Detached {
    /// Slot ids reserved until their handle frees them.
    slots: HashSet<SequenceId>,
    /// Substrates detached and not attached again; each is pruned once no
    /// handle on it remains to send anything.
    substrates: Vec<SubstrateRef>,
}

impl Detached {
    /// `conversation`'s substrate detached, its `slots` freed and reserved.
    pub(super) fn record(
        &mut self,
        conversation: &Conversation,
        slots: impl IntoIterator<Item = SequenceId>,
    ) {
        self.substrates.retain(|s| !s.is_gone());
        self.substrates.push(conversation.substrate_ref());
        self.slots.extend(slots);
    }

    /// `conversation`'s substrate attached again: its requests are live.
    pub(super) fn reattached(&mut self, conversation: &Conversation) {
        self.substrates
            .retain(|s| !s.is(conversation) && !s.is_gone());
    }

    /// Whether `slot` was reserved by a detach — and is no longer, its handle
    /// having freed it.
    pub(super) fn release(&mut self, slot: SequenceId) -> bool {
        self.slots.remove(&slot)
    }

    /// Whether `request` comes from a detached substrate's handle.
    pub(super) fn names(&self, request: &SchedulerRequest) -> bool {
        request.slots().iter().any(|s| self.slots.contains(s))
            || request
                .substrate()
                .is_some_and(|c| self.substrates.iter().any(|d| d.is(c)))
    }
}

impl SchedulerRequest {
    /// The slots a request acts on. Exhaustive, so a new request is classified
    /// when it is added.
    pub(super) fn slots(&self) -> Vec<SequenceId> {
        match self {
            Self::NewSequence { parent, .. } => parent.iter().copied().collect(),
            Self::ForkEphemeralSequence { parent, .. } => vec![*parent],
            Self::SubmitTurn { sequence_id, .. }
            | Self::FreeSequence { sequence_id }
            | Self::ResetSequence { sequence_id, .. }
            | Self::SetSequenceAdapter { sequence_id, .. }
            | Self::IngestSection { sequence_id, .. }
            | Self::SeedRecurrentFromLineage { sequence_id, .. }
            | Self::MemoryCatchUp { sequence_id, .. }
            | Self::PrimingProjection { sequence_id, .. }
            | Self::ExtractRawKvq { sequence_id, .. }
            | Self::ProbeWideSigs { sequence_id, .. } => vec![*sequence_id],
            #[cfg(any(test, feature = "test-helpers"))]
            Self::ReadRecurrentMemory { sequence_id, .. }
            | Self::ReadTurnKvDigest { sequence_id, .. } => vec![*sequence_id],
            Self::InstallRecurrentState { sequence_ids, .. } => sequence_ids.clone(),
            Self::NewEphemeralSequence { .. }
            | Self::ResumeSequence { .. }
            | Self::ScoreSealedTurn { .. }
            | Self::BranchCheckpointPass { .. }
            | Self::RestoreSection { .. }
            | Self::RetireSections { .. }
            | Self::OffloadCollectionMembers { .. }
            | Self::SubmitSummaryProbe { .. }
            | Self::ReconstructSubstrate { .. }
            | Self::AttachSubstrate { .. }
            | Self::DetachSubstrate { .. }
            | Self::WarmCollectionNormalization { .. }
            | Self::PreloadGalleries { .. }
            | Self::WarmIngestSlice { .. }
            | Self::ScanGroupVotes { .. }
            | Self::DemoteTimelinesHot { .. }
            | Self::Wake
            | Self::Settle { .. }
            | Self::Shutdown => Vec::new(),
            #[cfg(any(test, feature = "test-helpers"))]
            Self::CountRecurrentMemories { .. } => Vec::new(),
        }
    }

    /// The substrate a request acts on, when it names one. An attach and a
    /// detach name theirs too, but are the engine's own requests and never a
    /// stale handle's, so they are not counted here.
    pub(super) fn substrate(&self) -> Option<&Conversation> {
        match self {
            Self::NewSequence { conversation, .. }
            | Self::NewEphemeralSequence { conversation, .. }
            | Self::ResumeSequence { conversation, .. }
            | Self::RestoreSection { conversation, .. }
            | Self::RetireSections { conversation, .. }
            | Self::OffloadCollectionMembers { conversation, .. }
            | Self::ReconstructSubstrate { conversation, .. }
            | Self::WarmCollectionNormalization { conversation, .. }
            | Self::PreloadGalleries { conversation, .. }
            | Self::WarmIngestSlice { conversation, .. }
            | Self::ScanGroupVotes { conversation, .. }
            | Self::DemoteTimelinesHot { conversation, .. } => Some(conversation),
            Self::ScoreSealedTurn { substrate, .. } => Some(substrate),
            Self::ForkEphemeralSequence { .. }
            | Self::SubmitTurn { .. }
            | Self::FreeSequence { .. }
            | Self::ResetSequence { .. }
            | Self::SetSequenceAdapter { .. }
            | Self::IngestSection { .. }
            | Self::InstallRecurrentState { .. }
            | Self::SeedRecurrentFromLineage { .. }
            | Self::MemoryCatchUp { .. }
            | Self::BranchCheckpointPass { .. }
            | Self::PrimingProjection { .. }
            | Self::ExtractRawKvq { .. }
            | Self::ProbeWideSigs { .. }
            | Self::SubmitSummaryProbe { .. }
            | Self::AttachSubstrate { .. }
            | Self::DetachSubstrate { .. }
            | Self::Wake
            | Self::Settle { .. }
            | Self::Shutdown => None,
            #[cfg(any(test, feature = "test-helpers"))]
            Self::ReadRecurrentMemory { .. }
            | Self::ReadTurnKvDigest { .. }
            | Self::CountRecurrentMemories { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use flume::bounded;

    fn free(slot: usize) -> SchedulerRequest {
        SchedulerRequest::FreeSequence {
            sequence_id: SequenceId(slot),
        }
    }

    fn reset(slot: usize) -> SchedulerRequest {
        let (tx, _rx) = bounded(1);
        SchedulerRequest::ResetSequence {
            sequence_id: SequenceId(slot),
            response_tx: tx,
        }
    }

    fn new_on(conversation: &Conversation) -> SchedulerRequest {
        let (tx, _rx) = bounded(1);
        SchedulerRequest::NewSequence {
            conversation: conversation.clone(),
            target: None,
            parent: None,
            response_tx: tx,
        }
    }

    /// A detach names its slots and its substrate; another tenant's requests
    /// — on its own slots, through its own substrate — are not named.
    #[test]
    fn only_the_detached_tenants_requests_are_named() {
        let gone = Conversation::new();
        let live = Conversation::new();
        let mut detached = Detached::default();
        detached.record(&gone, [SequenceId(5), SequenceId(6)]);

        assert!(detached.names(&reset(5)));
        assert!(detached.names(&new_on(&gone)));
        assert!(!detached.names(&reset(7)));
        assert!(!detached.names(&new_on(&live)));
        assert!(!detached.names(&SchedulerRequest::Wake));
    }

    /// A reserved slot is released once, by its own handle's free; after that
    /// its id belongs to whoever takes it next.
    #[test]
    fn a_reserved_slot_is_released_by_its_own_free() {
        let gone = Conversation::new();
        let mut detached = Detached::default();
        detached.record(&gone, [SequenceId(5)]);
        assert!(detached.names(&free(5)));
        assert!(detached.release(SequenceId(5)));
        assert!(!detached.release(SequenceId(5)));
        assert!(!detached.names(&free(5)));
    }

    /// A substrate attached again takes requests again, and one whose every
    /// handle has gone is forgotten.
    #[test]
    fn a_reattached_or_vanished_substrate_is_no_longer_named() {
        let back = Conversation::new();
        let mut detached = Detached::default();
        detached.record(&back, []);
        detached.reattached(&back);
        assert!(!detached.names(&new_on(&back)));

        let vanished = Conversation::new();
        detached.record(&vanished, []);
        drop(vanished);
        detached.record(&Conversation::new(), []);
        assert_eq!(detached.substrates.len(), 1, "the vanished one was pruned");
    }
}
