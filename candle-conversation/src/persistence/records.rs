//! Appends each kind of record a substrate writes: stream data, metadata, snapshots, NPC state, custom objects and the tombstones that retire them.
//!
//! Snapshots and branch checkpoints keep a single tail per stream, so the store credits the previous one as dead as the new one lands, and a retraction withdraws the tail without a replacement. A timeline's distillation retires its recurrent state at the source, because state a distilled conversation can never resume is content, not provenance. An NPC record supersedes the character's previous record by stamp, and a deletion is an edit that marks the character tombstoned rather than a delete record.

use super::*;

impl SubstratePersistence {
    /// Retire a timeline's recurrent-state tail when its `Distilled` marker is
    /// appended — the runtime twin of the `Distilled` arm in
    /// [`record_snapshot_loc`], and the exact counterpart of the removal
    /// [`Self::write_tombstone`] performs.
    ///
    /// Distillation sheds a timeline's content and keeps its provenance
    /// signatures. The recurrent state is content: it is derived from the token
    /// stream and is meaningless without the K/V it was computed against, so a
    /// distilled conversation's snapshot describes a resume that can never
    /// happen. Retiring the tail here means it stops being live at the *source*,
    /// which is what lets both rewrite paths do the right thing without either
    /// having to consult the distilled set.
    ///
    /// Keyed off the payload, not the header: a `Distilled` record carries its
    /// `timeline_id` in the payload and leaves `stream_id` zero, so unlike
    /// `Snapshot` / `Npc` the identity cannot be read from the header alone.
    pub(super) fn retire_snapshot_on_distill(&mut self, h: &RecordHeader, payload: &[u8]) {
        if h.record_type != RecordType::Distilled {
            return;
        }
        if let Ok(p) = record::DistillPayload::decode(payload) {
            self.snapshot_locs
                .remove(&snapshot_stream_id(p.timeline_id).0);
        }
    }

    /// Append one character's durable state.
    ///
    /// The `npc_id` goes in the header's `stream_id`, which is what makes this
    /// supersede the character's previous record mechanically: the accounting
    /// credits the old copy as dead the moment this one lands, and
    /// [`Self::npc_locs`] now points here. There is no delete record — an edit
    /// supersedes, and a deletion is an edit that sets `state: "tombstoned"`.
    ///
    /// Writes are expected to be rare: the payload is durable configuration
    /// only, so a character that is merely *running* never writes at all.
    pub fn write_npc(&mut self, npc: &NpcPayload) -> Result<RecordIdentity> {
        let bytes = npc.encode();
        let at = self.append_record(RecordType::Npc, 0, npc.npc_id, 0, 0, 0, &bytes)?;
        Ok(at.identity)
    }

    /// Append custom-object record `id`. The header's `stream_id` is the key's
    /// stable hash, so of two writes with the same key the greater stamp is the
    /// object on replay. The caller applies it to the in-RAM [`Substrate`]
    /// under the same identity, so it re-emits on compaction — see
    /// [`SharedSubstrate::put_custom_object`].
    pub fn write_custom_object(
        &mut self,
        obj: &CustomObjectPayload,
        id: RecordIdentity,
    ) -> Result<()> {
        let bytes = obj.encode();
        self.append_carried(
            RecordType::CustomObject,
            0,
            obj.stream_id(),
            0,
            0,
            0,
            &bytes,
            id,
        )?;
        Ok(())
    }

    /// Locations rather than payloads, because the payloads are not held here —
    /// the registry that owns them lives in the daemon. Compaction and segment
    /// liveness both read this to know which bytes are still worth keeping.
    pub fn npc_locs(&self) -> &StampedLocs<u64> {
        &self.npc_locs
    }

    /// Read back one character's current record.
    ///
    /// The load path does not use this — it collects payloads from the recovery
    /// walk's sink in one sweep, which is a great deal cheaper than a seek per
    /// character. This exists for the one-off: re-reading a single character
    /// after a write, or a diagnostic.
    pub fn read_npc(&mut self, npc_id: u64) -> Result<Option<NpcPayload>> {
        let Some(loc) = self.npc_locs.get(&npc_id).copied() else {
            return Ok(None);
        };
        let record = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(Some(NpcPayload::decode(&record.payload)?))
    }

    /// Append a stream's `Tokens` record. The caller MUST fold its location and
    /// identity into the substrate index (`apply_tokens_loc`) — otherwise the
    /// in-RAM index has no record of the tokens, and a subsequent segment
    /// compaction / relocation can't carry them forward and reclaims them (the
    /// KV path already registers its chunk locations; tokens must too, or a
    /// live turn silently loses its text).
    pub fn append_tokens(&mut self, stream_id: StreamId, tokens: &[u8]) -> Result<Appended> {
        self.append_record(RecordType::Tokens, 0, stream_id.0, 0, 0, 0, tokens)
    }

    /// Append a turn's `ProjectionEvents` record `id` (opaque JSON payload).
    ///
    /// Returns the record's location, which the writer settles the substrate's
    /// held copy to ([`crate::substrate::Substrate::apply_projection_events_loc`]).
    pub fn append_projection_events(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<RecordLoc> {
        let at = self.append_carried(
            RecordType::ProjectionEvents,
            0,
            stream_id.0,
            0,
            0,
            0,
            payload,
            id,
        )?;
        Ok(at.loc(payload.len() as u64))
    }

    /// Append a turn's `WideQSig` record `id` (opaque wide-Q window payload),
    /// keyed by stream id.
    ///
    /// Returns the record's location, which the writer settles the substrate's
    /// held copy to ([`crate::substrate::Substrate::apply_wide_q_sigs_loc`]).
    pub fn append_wide_q_sigs(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<RecordLoc> {
        let at = self.append_carried(RecordType::WideQSig, 0, stream_id.0, 0, 0, 0, payload, id)?;
        Ok(at.loc(payload.len() as u64))
    }

    /// Append a turn's QSA index page, record `id` (opaque payload), keyed by
    /// stream id.
    ///
    /// The counterpart of [`Self::append_wide_q_sigs`]: both carry a blob that
    /// belongs to a turn and that no amount of re-reading the turn's K/V can
    /// reconstruct, because it was computed from hidden states.
    ///
    /// Returns the record's location, which the writer settles the substrate's
    /// held copy to ([`crate::substrate::Substrate::settle_index_page`]): a
    /// page is held only until it is durable, then read from the log.
    pub fn append_turn_index_page(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<RecordLoc> {
        let at = self.append_carried(
            RecordType::TurnIndexPage,
            0,
            stream_id.0,
            0,
            0,
            0,
            payload,
            id,
        )?;
        Ok(at.loc(payload.len() as u64))
    }

    /// Append a conversation's recurrent-state snapshot under its snapshot
    /// stream id. Single tail: the accounting layer credits the previous
    /// snapshot as dead the moment this one lands, and `snapshot_locs` keeps
    /// exactly this copy alive through segment maintenance. Returns the
    /// record's location for the caller's in-RAM index.
    ///
    /// The payload comes as its parts ([`record::SnapshotPayload::parts`]) and is
    /// written from them, so the state reaches the file without being copied
    /// into one buffer first.
    pub fn write_snapshot(
        &mut self,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        self.write_recurrent_record(RecordType::Snapshot, stream_id, payload)
    }

    /// Append a snapshot retraction for `stream_id`: an empty `Snapshot`
    /// record, after which the stream has no live snapshot on this run or any
    /// reload ([`RecordHeader::retracts_snapshot`]). The writer appends one
    /// when a seal's snapshot fails to land, because the previous snapshot
    /// would otherwise stay the tail while the turn's records land behind it.
    /// Returns the retraction's identity, which outranks the tail it withdraws.
    pub fn retract_snapshot(&mut self, stream_id: StreamId) -> Result<RecordIdentity> {
        let at = self.append_record(RecordType::Snapshot, 0, stream_id.0, 0, 0, 0, &[])?;
        Ok(at.identity)
    }

    /// Append a prompt branch's recurrent checkpoint — the same single-tail
    /// append under a `BranchCheckpoint` record, from its parts
    /// ([`record::BranchCheckpointPayload::parts`]).
    pub fn write_branch_checkpoint(
        &mut self,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        self.write_recurrent_record(RecordType::BranchCheckpoint, stream_id, payload)
    }

    pub(super) fn write_recurrent_record(
        &mut self,
        record_type: RecordType,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        let at = self.append_recurrent_parts(record_type, stream_id.0, payload)?;
        Ok(Stamped {
            value: at.loc(payload.iter().map(|p| p.len() as u64).sum()),
            id: at.identity,
        })
    }

    /// Append a `Commit` record marking `stream_id` durable through
    /// `through_index`. Returns the identity it was written with.
    pub fn commit_stream(
        &mut self,
        stream_id: StreamId,
        through_index: u64,
    ) -> Result<RecordIdentity> {
        let at =
            self.append_record(RecordType::Commit, 0, stream_id.0, through_index, 0, 0, &[])?;
        Ok(at.identity)
    }

    /// Append `TreeMetadata` record `id` for one `(timeline_id, turn_index)`
    /// summary-tree node — the greatest stamp per node on replay. Callers
    /// check idempotency against substrate state.
    pub fn write_tree_metadata(
        &mut self,
        payload: TreeMetadataPayload,
        id: RecordIdentity,
    ) -> Result<()> {
        let bytes = payload.encode();
        self.append_carried(RecordType::TreeMetadata, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append a [`RecordType::Tombstone`] record marking
    /// `timeline_id` as logically deleted.  Walker replay applies it
    /// to the substrate; the compactor drops every record bound to
    /// the timeline on the next compaction pass.  Idempotent —
    /// duplicate tombstones replay identically. `reason` is a diagnostic
    /// note (e.g. the corrupt-reload detail) recorded in the payload; pass
    /// `None` for an ordinary deletion.
    pub fn write_tombstone(&mut self, timeline_id: u64, reason: Option<&str>) -> Result<()> {
        let id = self.minter.mint(0);
        self.append_timeline_tombstone(timeline_id, reason, id)
    }

    /// [`Self::write_tombstone`] for a **repair**: this member's judgement
    /// about its own bytes — a turn it finds incomplete or unreadable at
    /// reload — not a deletion. Housekeeping, written with no identity, so it
    /// is never sent to another member, which may hold the conversation whole
    /// (§6.5).
    pub fn write_repair_tombstone(&mut self, timeline_id: u64, reason: &str) -> Result<()> {
        self.append_timeline_tombstone(timeline_id, Some(reason), RecordIdentity::NONE)
    }

    pub(super) fn append_timeline_tombstone(
        &mut self,
        timeline_id: u64,
        reason: Option<&str>,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = TombstonePayload {
            timeline_id,
            turn_index: None,
            turn_writer: 0,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::Tombstone, 0, 0, 0, 0, 0, &bytes, id)?;
        // A dead conversation has no live snapshot tail: drop its entry from
        // the persistence-side map so a maintenance execute planned before
        // this tombstone skips relocating the (now dead) snapshot record —
        // the same removal the substrate index performs in `apply_tombstone`,
        // and the same one `record_snapshot_loc` replays on reload.
        self.snapshot_locs
            .remove(&snapshot_stream_id(timeline_id).0);
        // Its file events are orphans now: out of the index, so no rewrite
        // carries them, and their bytes dead.
        vfs::retire_timeline(&mut self.vfs_index, &mut self.accounting, timeline_id);
        Ok(())
    }

    /// Append a **turn-scoped** tombstone — kill only the turn `mark` names,
    /// leaving the rest of the timeline live. Used by the per-layer `drop_turn`
    /// corrupt-turn policy so one unrecoverable turn (e.g. a partial write) does
    /// not drop its whole conversation the way a timeline tombstone would, and
    /// by retention. The mark names the turn as its writer wrote it
    /// ([`Substrate::tombstone_turn`] returns it).
    pub fn write_turn_tombstone(&mut self, mark: TurnMark, reason: Option<&str>) -> Result<()> {
        let id = self.minter.mint(0);
        self.append_turn_tombstone(mark, reason, id)
    }

    /// [`Self::write_turn_tombstone`] for a **repair** — see
    /// [`Self::write_repair_tombstone`].
    pub fn write_repair_turn_tombstone(&mut self, mark: TurnMark, reason: &str) -> Result<()> {
        self.append_turn_tombstone(mark, Some(reason), RecordIdentity::NONE)
    }

    pub(super) fn append_turn_tombstone(
        &mut self,
        mark: TurnMark,
        reason: Option<&str>,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = TombstonePayload {
            timeline_id: mark.timeline.raw(),
            turn_index: Some(mark.turn_index),
            turn_writer: mark.writer,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::Tombstone, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append a [`RecordType::SectionTombstone`] record marking the
    /// content-addressed section stream `stream_id` as logically deleted.
    /// Walker replay applies it via [`Substrate::apply_section_tombstone`],
    /// which reads as "not persisted" on the next ingest triage, so the
    /// section re-prefills fresh instead of restoring the tombstoned chunks.
    /// Idempotent — duplicate tombstones for the same stream replay into the
    /// same set. `generation` is the one it ends
    /// ([`Substrate::tombstone_section`] returns it); `None` ends the
    /// unstamped stream. `reason` is a diagnostic note (the confirmed
    /// divergence detail); pass `None` when there is nothing more to say than
    /// "gone". A generation this node was writing stops being stamped on its
    /// records.
    pub fn write_section_tombstone(
        &mut self,
        stream_id: u64,
        generation: Option<SectionGeneration>,
        reason: Option<&str>,
    ) -> Result<()> {
        let payload = SectionTombstonePayload {
            generation,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_record(RecordType::SectionTombstone, 0, stream_id, 0, 0, 0, &bytes)?;
        if let Some(g) = generation {
            if g.writer == self.node() {
                self.minter.end_section(stream_id, g.oseq);
            }
        }
        Ok(())
    }

    /// Append a [`RecordType::TurnCoupling`] record joining `from_turn` to the
    /// tool response that follows it.
    ///
    /// Written before the response turn is submitted — the one window where the
    /// round-trip is certain — so replay can never observe the response turn
    /// without its coupling. Idempotent: duplicates replay into the same set.
    pub fn write_turn_coupling(
        &mut self,
        timeline_id: u64,
        from_turn: u32,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = record::TurnCouplingPayload {
            timeline_id,
            from_turn,
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::TurnCoupling, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append `DebugId` record `id` for `timeline_id` — the greatest stamp per
    /// timeline on replay. Callers check idempotency against substrate state.
    pub fn write_debug_id(
        &mut self,
        timeline_id: u64,
        debug_id: &str,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = DebugIdPayload {
            timeline_id,
            debug_id: debug_id.to_string(),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::DebugId, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }
}
