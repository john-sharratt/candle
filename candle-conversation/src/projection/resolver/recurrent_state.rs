//! Writes and reads a conversation's recurrent-state snapshots and a prompt
//! branch's checkpoint in the redo log.
//!
//! Recurrent state is fixed-size per sequence whatever the token count, so a
//! timeline that is ephemeral, or kept but never resumed, writes no snapshot. A
//! snapshot is a single tail that supersedes the one before it, and retiring a tail
//! writes an empty one. An unreadable record where the index points is an error,
//! because starting from zeros is the silent amnesia this path exists to prevent. A
//! branch checkpoint is memoised for the next conversation opened on that prompt.

use super::*;

impl Conversation {
    /// Enqueue a conversation's recurrent-state snapshot (the encoded
    /// [`SnapshotPayload`]) under its snapshot stream id. Fire-and-forget from
    /// the seal path: the writer thread appends it and registers the fresh
    /// location, superseding every earlier snapshot for the conversation (the
    /// single-tail contract). Nothing is mirrored in RAM here — the state that
    /// produced the payload is still live on the device; the record exists for
    /// restart and fork-resume.
    pub fn enqueue_recurrent_snapshot(&self, timeline: TimelineId, payload: SnapshotPayload) {
        // **Ephemeral timelines get no snapshot.** The record exists for restart
        // and fork-resume, and a timeline marked by
        // `Substrate::mark_timeline_transient` is by definition never resumed:
        // a code_read scope fork is tombstoned, a calibration exemplar is
        // archived and then distilled to `ProvenanceOnly`. Writing it spends
        // real bytes on state nothing will ever read.
        //
        // Not a micro-optimisation. Recurrent state is fixed-size **per
        // sequence, independent of token count**, so a 19-token calibration
        // question costs exactly as much as a full trajectory. Measured on this
        // model: ~65 MB per turn — 2,998 exemplars would write ~195 GB of
        // snapshots, which dwarfed even their KV and was enough on its own to
        // fill the disk and wedge the daemon on failing fsyncs.
        //
        // Nor do timelines that are kept but never resumed
        // (`Substrate::mark_timeline_unresumed`), for the same reason.
        {
            let s = self.read();
            if s.is_timeline_transient(timeline) || s.is_timeline_unresumed(timeline) {
                return;
            }
        }
        let stream_id = snapshot_stream_id(timeline.raw());
        self.persistence
            .enqueue(WriteJob::Snapshot { stream_id, payload });
    }

    /// Mark `timeline` as kept but never resumed
    /// (`Substrate::mark_timeline_unresumed`): no further snapshot is written
    /// for it, and the tail it already has is retired.
    ///
    /// **Retired by superseding, not by deletion.** An empty snapshot — no
    /// layers, no aux — is enqueued under the timeline's snapshot stream, and
    /// the single-tail rule makes it the tail: the full one reads as dead to
    /// maintenance, which reclaims it, and the empty one replicates like any
    /// snapshot, so every member that copied the full one retires it too.
    /// [`Self::read_recurrent_snapshot`] reads an empty tail as none kept.
    ///
    /// Enqueued unconditionally rather than only when a tail is indexed: a seal
    /// that ran just before this may still be in the writer's queue, unindexed,
    /// and the queue's order is what puts the empty record after it.
    pub fn mark_timeline_unresumed(&self, timeline: TimelineId) {
        self.note_timeline_unresumed(timeline);
        let stream_id = snapshot_stream_id(timeline.raw());
        self.persistence.enqueue(WriteJob::Snapshot {
            stream_id,
            payload: SnapshotPayload::retired(timeline.raw()),
        });
    }

    /// [`Self::mark_timeline_unresumed`] without the retiring write: the flag
    /// alone, for a timeline whose tail is already retired or absent.
    pub fn note_timeline_unresumed(&self, timeline: TimelineId) {
        self.update(|s| s.mark_timeline_unresumed(timeline));
    }

    /// Bytes of the recurrent snapshot `timeline` keeps, if any — a retired
    /// (empty) tail reads as a few dozen bytes. For a sweep deciding whether a
    /// tail still needs retiring, without reading it.
    pub fn recurrent_snapshot_bytes(&self, timeline: TimelineId) -> Option<u64> {
        self.recurrent_snapshot_loc(timeline).map(|l| l.payload_len)
    }

    /// Enqueue a **prompt branch's** recurrent checkpoint — the state after the
    /// whole system prompt for one selector assignment, keyed by the branch's
    /// cumulative content prefix.
    ///
    /// Rides the identical single-tail path as a conversation's snapshot: the
    /// writer appends it under this stream id and registers the location,
    /// superseding any earlier checkpoint for the same branch. Superseding is
    /// the right behaviour here too — a branch has exactly one state, and a
    /// second computation of it can only be a recomputation.
    pub fn enqueue_branch_checkpoint(
        &self,
        prefix: ContentHash,
        payload: Arc<BranchCheckpointPayload>,
    ) {
        let stream_id = branch_checkpoint_stream_id(prefix);
        self.persistence
            .enqueue(WriteJob::BranchCheckpoint { stream_id, payload });
    }

    /// Read a prompt branch's recurrent checkpoint, if one was computed.
    ///
    /// `None` is ordinary: no checkpoint has been built for this branch, or the
    /// prompt changed and this branch's content prefix with it. An indexed but
    /// unreadable record is an **error** for the same reason it is on the
    /// conversation path — silently starting from zeros is the amnesia this
    /// exists to remove.
    pub fn read_branch_checkpoint(
        &self,
        prefix: ContentHash,
    ) -> Result<Option<Arc<BranchCheckpointPayload>>, ConversationError> {
        // Served decoded when this is the branch we last read — see
        // `branch_checkpoint`. The hash key makes the hit exact: a different
        // branch misses and refills rather than installing the wrong state.
        if let Some(memo) = self.branch_checkpoint.load_full() {
            if memo.0 == prefix {
                return Ok(Some(Arc::clone(&memo.1)));
            }
        }
        let stream_id = branch_checkpoint_stream_id(prefix);
        let Some(loc) = self.read().branch_checkpoint_loc(stream_id) else {
            return Ok(None);
        };
        let bytes = self
            .read_located(loc, |stale| {
                self.read()
                    .branch_checkpoint_loc(stream_id)
                    .filter(|now| *now != stale)
            })
            .map_err(|e| {
                ConversationError::Channel(format!(
                    "branch checkpoint {prefix:?} is indexed but unreadable: {e}"
                ))
            })?;
        let payload = BranchCheckpointPayload::decode(&bytes).map_err(|e| {
            ConversationError::Channel(format!(
                "branch checkpoint {prefix:?} failed to decode: {e}"
            ))
        })?;
        if payload.prefix_hash != prefix {
            return Err(ConversationError::Channel(format!(
                "branch checkpoint under {prefix:?} states prefix {:?} — the stream \
                 id collided, and restoring it would seed a conversation with \
                 another branch's prompt state",
                payload.prefix_hash
            )));
        }
        let payload = Arc::new(payload);
        self.memo_branch_checkpoint(prefix, &payload);
        Ok(Some(payload))
    }

    /// Keep `state` as the state section stream `section` ended on — the last
    /// section of a shared run (`docs/shared_journal_sections.md` §4.1). On the
    /// snapshot single tail under the section's end-state stream, so it
    /// replicates and goes with the section's generation.
    pub fn enqueue_section_end_state(&self, section: StreamId, state: &SharedState) {
        if self.read_only {
            return;
        }
        self.persistence.enqueue(WriteJob::SectionEndState {
            stream_id: section_end_state_stream_id(section),
            payload: section_end_state::to_payload(state),
        });
    }

    /// The state section stream `section` ended on, when its stream keeps
    /// one. `None` is ordinary — a section that ended no shared run, or whose
    /// state was never kept. An indexed but unreadable record is an error, as a
    /// checkpoint's is: starting from zeros instead is the silent amnesia.
    pub fn read_section_end_state(
        &self,
        section: StreamId,
    ) -> Result<Option<SharedState>, ConversationError> {
        let stream_id = section_end_state_stream_id(section);
        let Some(loc) = self.read().recurrent_snapshot_loc(stream_id) else {
            return Ok(None);
        };
        let bytes = self
            .read_located(loc, |stale| {
                self.read()
                    .recurrent_snapshot_loc(stream_id)
                    .filter(|now| *now != stale)
            })
            .map_err(|e| {
                ConversationError::Channel(format!(
                    "the end state of section stream {section:?} is indexed but unreadable: {e}"
                ))
            })?;
        let payload = SnapshotPayload::decode(&bytes).map_err(|e| {
            ConversationError::Channel(format!(
                "the end state of section stream {section:?} failed to decode: {e}"
            ))
        })?;
        Ok(section_end_state::from_payload(&payload))
    }

    /// Hold `payload` as the decoded checkpoint for `prefix`, so the next
    /// conversation opened on that branch installs it without touching the log.
    /// Called both after a read and after a fresh computation — a branch this
    /// process just computed is exactly the one the following conversations
    /// will ask for.
    pub fn memo_branch_checkpoint(
        &self,
        prefix: ContentHash,
        payload: &Arc<BranchCheckpointPayload>,
    ) {
        self.branch_checkpoint
            .store(Some(Arc::new((prefix, Arc::clone(payload)))));
    }

    /// The live recurrent-state snapshot location for `timeline`, if one is
    /// on disk. Resume reads the payload through the persistence handle and
    /// validates `(schedule_hash, turn_index)` before scattering state.
    pub fn recurrent_snapshot_loc(&self, timeline: TimelineId) -> Option<RecordLoc> {
        self.read()
            .recurrent_snapshot_loc(snapshot_stream_id(timeline.raw()))
    }

    /// Read and decode a timeline's recurrent-state snapshot — the resume read.
    ///
    /// `None` means "no snapshot for this timeline", which is an ordinary
    /// outcome: a conversation sealed by a model that carries no recurrent
    /// state has none, and so does one whose first turn has not sealed yet. A
    /// snapshot that is present but unreadable is an **error**, not a `None`:
    /// falling back to zeros there is the silent amnesia this whole path exists
    /// to remove, so the caller has to see it and say so.
    pub fn read_recurrent_snapshot(
        &self,
        timeline: TimelineId,
    ) -> Result<Option<SnapshotPayload>, ConversationError> {
        let Some(loc) = self.recurrent_snapshot_loc(timeline) else {
            return Ok(None);
        };
        // The scheduler calls this at every admission of a hybrid-model turn,
        // so a durable snapshot is read without the log's owner at all.
        let bytes = self
            .read_located(loc, |stale| self.moved_snapshot_loc(timeline, stale))
            .map_err(|e| {
                ConversationError::Channel(format!(
                    "recurrent snapshot for timeline {} is indexed but unreadable: {e}",
                    timeline.raw()
                ))
            })?;
        Self::decode_kept_snapshot(timeline, &bytes)
    }

    /// Every recurrent snapshot whose record became durable since the last
    /// call, as `(timeline, turn index)` — the cue for a slot holding exactly
    /// that state to release it, since [`Self::read_recurrent_snapshot`] now
    /// reads the record back without the log's owner. Lock-free: it waits on
    /// neither the owner nor the substrate. A snapshot appended but not yet
    /// committed is reported by a later call.
    pub fn drain_durable_snapshots(&self) -> Vec<(TimelineId, u32)> {
        self.persistence
            .written_snapshots()
            .take_durable(|loc| self.sealed.is_durable(loc))
            .into_iter()
            .filter_map(|(raw, turn)| Some((TimelineId::from_raw(raw)?, turn)))
            .collect()
    }

    /// Whether the recurrent state `timeline` reached at `turn` — or a later
    /// turn's — is in a durable record, so a resume on a new slot restores it.
    ///
    /// Non-consuming: unlike [`Self::drain_durable_snapshots`], whose queue
    /// is the scheduler's alone, asking this takes nothing from anyone. It is
    /// what a caller waits on before it drops a sequence whose next turn may
    /// run on another slot. Lock-free on the log, like the drain.
    pub fn snapshot_durable(&self, timeline: TimelineId, turn: u32) -> bool {
        self.persistence
            .written_snapshots()
            .latest(timeline.raw())
            .is_some_and(|(written, loc)| written >= turn && self.sealed.is_durable(&loc))
    }

    /// The timeline's snapshot location if the index now names a different one
    /// than `stale` — the record was relocated since `stale` was read. `None`
    /// when it has not moved, or when no snapshot is indexed any more.
    pub(super) fn moved_snapshot_loc(
        &self,
        timeline: TimelineId,
        stale: RecordLoc,
    ) -> Option<RecordLoc> {
        self.recurrent_snapshot_loc(timeline)
            .filter(|now| *now != stale)
    }

    /// Decode a recurrent snapshot record, naming the timeline on failure.
    pub(super) fn decode_snapshot(
        timeline: TimelineId,
        bytes: &[u8],
    ) -> Result<SnapshotPayload, ConversationError> {
        SnapshotPayload::decode(bytes).map_err(|e| {
            ConversationError::Channel(format!(
                "recurrent snapshot for timeline {} failed to decode: {e}",
                timeline.raw()
            ))
        })
    }

    /// [`Self::decode_snapshot`], reading a retired tail
    /// ([`Self::mark_timeline_unresumed`]) as no snapshot kept.
    pub(super) fn decode_kept_snapshot(
        timeline: TimelineId,
        bytes: &[u8],
    ) -> Result<Option<SnapshotPayload>, ConversationError> {
        Self::decode_snapshot(timeline, bytes).map(|p| (!p.is_retired()).then_some(p))
    }
}
