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
//!
//! A detached substrate is still a holder of KV for as long as it lives: its
//! residences keep their resident copies until its last handle goes. So the
//! compaction sweep visits every detached substrate still alive
//! ([`Detached::rewrite_for_compaction`]) — out of the attached set, it is named
//! by nothing else the sweep reaches.

use std::collections::HashSet;
use std::sync::Arc;

use candle_nn::kv_cache::Sweep;

use super::SchedulerRequest;
use crate::projection::{Conversation, SubstrateRef};
use crate::sequence_handle::SequenceId;
use crate::substrate::SubstrateLock;

/// A detached substrate held alive for one compaction pass — see
/// [`Detached::alive`].
pub(super) struct Swept(Arc<SubstrateLock>);

impl Swept {
    /// Rewrite its residences through the pass's sweep, returning how many
    /// were rewritten.
    pub(super) fn rewrite_for_compaction(&self, sweep: &mut Sweep<'_>) -> candle::Result<usize> {
        self.0.read().rewrite_for_compaction(sweep)
    }
}

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

    /// Every detached substrate still alive, held for a compaction pass — a
    /// detached tenant's residences hold resident KV until its last handle
    /// goes, and nothing else the sweep visits names them.
    ///
    /// Held for the whole pass and let go after it, never upgraded inside it:
    /// a tenant dropping its last handle mid-pass would otherwise leave the
    /// pass's own upgrade as the last, and the substrate would free its chunks
    /// into the pool the pass is relocating, on the thread that is relocating
    /// them.
    pub(super) fn alive(&self) -> Vec<Swept> {
        self.substrates
            .iter()
            .filter_map(SubstrateRef::upgrade)
            .map(Swept)
            .collect()
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

    use std::sync::Arc;

    use candle_nn::kv_cache::{
        ArenaLocation, ChunkGid, CompactionMap, HeadGids, SealedChunk, SealedSequence, GID_STRIDE,
    };
    use flume::bounded;

    use crate::projection::{GroupId, LayerId, TimelineId};
    use crate::substrate::residence::own_device_frees;
    use crate::substrate::{ResidenceIndex, TurnPartWrite};

    /// A one-layer sequence of one chunk at raw gid `raw`.
    fn seq_at(raw: i64) -> SealedSequence {
        SealedSequence {
            chunks: vec![SealedChunk {
                gids: HeadGids::uniform(ChunkGid::detached(raw), 1),
                offset: 0,
                token_count: 32,
                k_pal: Arc::new(Vec::new()),
                v_pal: Arc::new(Vec::new()),
                k_scale: Arc::new(Vec::new()),
                v_scale: Arc::new(Vec::new()),
                k_fmt: Arc::new(Vec::new()),
                v_fmt: Arc::new(Vec::new()),
                byte_size: 0,
                meta: None,
            }],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// A substrate holding one turn whose hot copy is one chunk at `raw`.
    fn hot_at(raw: i64) -> (Conversation, ResidenceIndex) {
        let conversation = Conversation::new();
        let timeline = TimelineId::for_test(1);
        conversation.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));
        let residence = conversation.update(|s| {
            let write = TurnPartWrite {
                sealed_gpu: Some(Arc::new(vec![seq_at(raw)])),
                token_count: 32,
                ..Default::default()
            };
            let idx = s
                .append_complete(timeline, write, |seqs| Ok(seqs.to_vec()))
                .unwrap();
            s.turn_residence(timeline, idx).unwrap()
        });
        (conversation, residence)
    }

    /// The raw gid of `residence`'s hot copy's one chunk.
    fn hot_raw(conversation: &Conversation, residence: ResidenceIndex) -> i64 {
        let slab = conversation.residences();
        let hot = slab.get(residence).hot().expect("still hot");
        hot[0].chunks[0].gids.as_slice()[0].raw()
    }

    /// **A detached substrate still alive is a holder the sweep reaches**: its
    /// resident copy that moved comes back naming the destination, and one whose
    /// every handle has gone is skipped, holding nothing.
    #[test]
    fn a_detached_substrate_is_swept_while_it_lives() {
        // The payloads dropped here are this thread's to free, not the shared
        // reclaim queue's.
        own_device_frees();
        let src = (5 * GID_STRIDE + 2) as i64;
        let dst = (GID_STRIDE + 1) as i64;
        let (alive, residence) = hot_at(src);
        let mut detached = Detached::default();
        detached.record(&alive, []);
        let gone = Conversation::new();
        detached.record(&gone, []);
        drop(gone);

        let mut map = CompactionMap::new();
        map.insert(src, ChunkGid::detached(dst), 0x4000);
        let mut sweep = Sweep::new(&map);

        let held = detached.alive();
        assert_eq!(held.len(), 1, "the gone one is not held");
        let rewritten: usize = held
            .iter()
            .map(|s| s.rewrite_for_compaction(&mut sweep).unwrap())
            .sum();
        assert_eq!(rewritten, 1);
        assert_eq!(hot_raw(&alive, residence), dst);
        assert_eq!(sweep.allocations_rewritten(), 1);
    }

    /// **A detached substrate held for a pass outlives its last handle until
    /// the pass lets it go**, so it is never freed inside the pass.
    #[test]
    fn a_held_substrate_outlives_its_last_handle_until_released() {
        own_device_frees();
        let (substrate, _) = hot_at((3 * GID_STRIDE + 1) as i64);
        let mut detached = Detached::default();
        detached.record(&substrate, []);
        let held = detached.alive();
        drop(substrate);
        assert_eq!(detached.alive().len(), 1, "the pass's hold keeps it alive");
        drop(held);
        assert!(detached.alive().is_empty(), "let go once the pass is over");
    }

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
