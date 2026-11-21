//! Retires and releases a conversation's timelines, sections and turns, and marks
//! how their storage may be shed.
//!
//! A tombstone marks a timeline logically deleted in RAM at once and on disk
//! through a record, and the compactor drops its records later, so reads never see
//! them. A tombstoned turn keeps its index and loses its content, which keeps
//! numbering stable and lets an unbounded transcript be reclaimed. Section releases
//! write tombstones so a restart does not restore them. Distilled, transient and
//! splice-source marks say when a timeline's key-value state may be shed.

use super::*;

impl Conversation {
    /// Tombstone `timeline` — marks it logically deleted both
    /// in-RAM (so projection retrieval stops surfacing its turns on
    /// the next query) and on disk (via a
    /// [`crate::persistence::record::RecordType::Tombstone`]
    /// record).  The compactor drops the underlying records on the
    /// next compaction pass; ordinary reads never see them. On a read-only
    /// substrate the in-RAM tombstone is all there is.
    ///
    /// Sections the timeline owns ([`Self::release_owned_sections`]) go with it.
    pub fn tombstone_timeline(&self, timeline: TimelineId) -> candle::Result<()> {
        // A unit's face stands only for it, so it goes with it.
        let face = self
            .read()
            .face_timeline(timeline)
            .filter(|&face| face != timeline);
        self.release_owned_sections(timeline)?;
        self.update(|s| s.tombstone_timeline(timeline));
        if !self.read_only {
            // The append runs on the log's owner; this returns without waiting.
            self.persistence.enqueue(WriteJob::TimelineTombstone {
                timeline_id: timeline.raw(),
                reason: None,
            });
        }
        match face {
            Some(face) if !self.read().is_tombstoned(face) => self.tombstone_timeline(face),
            _ => Ok(()),
        }
    }

    /// Tombstone a section stream — the section counterpart of
    /// [`Self::tombstone_timeline`]. Marks it logically deleted both in-RAM
    /// (so the ingest triage's `section_exists` / `section_stream_is_persisted`
    /// checks stop treating it as present or restorable on the very next
    /// request, in this same process) and on disk (via a
    /// [`crate::persistence::record::RecordType::SectionTombstone`] record,
    /// so a restarted daemon doesn't reload the same corruption). The
    /// compactor drops the underlying `Chunk`/`Tokens`/`StreamDecl` records
    /// on the next compaction pass; ordinary reads never see them once this
    /// lands.
    pub fn tombstone_section(
        &self,
        stream_id: StreamId,
        reason: Option<&str>,
    ) -> candle::Result<()> {
        let generation = self.update(|s| s.tombstone_section(stream_id));
        if self.read_only {
            return Ok(());
        }
        self.persistence.enqueue(WriteJob::SectionTombstone {
            stream_id: stream_id.0,
            generation,
            reason: reason.map(str::to_string),
        });
        Ok(())
    }

    /// End a section stream's persisted generation so its next request prefills
    /// it fresh — the rebuild counterpart of [`Self::tombstone_section`], which
    /// is for corrupted bytes and releases them. Resident copies keep serving
    /// ([`crate::substrate::Substrate::retire_section_generation`]); the
    /// durable [`crate::persistence::record::RecordType::SectionTombstone`] is
    /// the same record, so a restart rebuilds the section.
    pub fn retire_section_generation(
        &self,
        stream_id: StreamId,
        reason: &str,
    ) -> candle::Result<()> {
        let generation = self.update(|s| s.retire_section_generation(stream_id));
        if self.read_only {
            return Ok(());
        }
        self.persistence.enqueue(WriteJob::SectionTombstone {
            stream_id: stream_id.0,
            generation,
            reason: Some(reason.to_string()),
        });
        Ok(())
    }

    /// The persisted stream `section` is registered against, if it is
    /// registered — what [`Self::tombstone_section`] and
    /// [`Self::retire_section_generation`] take.
    pub fn section_stream(&self, section: SectionId) -> Option<StreamId> {
        self.read().section_stream(section)
    }

    /// Re-read a persisted section's per-layer chunk windows off disk and
    /// run the generic [`first_divergent_chunk`] check against them —
    /// confirmation from what is actually persisted, not a trust of a live
    /// seal failure alone. Returns the first divergence found, or `None`
    /// when the stream holds no durable chunks (nothing to confirm) or its
    /// persisted layers agree (the seal failure that prompted the caller to
    /// ask must have another cause — see the reactive hook at
    /// `scheduler::Scheduler::perform_seal_and_write`).
    ///
    /// Read-only by design: this only answers the question. A caller that
    /// wants to act on a confirmed divergence calls [`Self::tombstone_section`]
    /// itself, so a caller that only wants the answer (a future boot-time
    /// pass, a diagnostic) never also accepts the side effect.
    pub fn check_section_window_integrity(
        &self,
        stream_id: StreamId,
        n_layers: usize,
    ) -> candle::Result<Option<WindowDivergence>> {
        let substrate = self.read();
        let windows = self
            .persistence
            .call(|p| read_persisted_section_windows(p, &substrate, stream_id, n_layers))
            .map_err(|e| candle::Error::Msg(format!("check_section_window_integrity: {e}")))?;
        Ok(windows.and_then(|w| first_divergent_chunk(&w)))
    }

    /// Tombstone **one turn** of a live timeline — in-RAM and on disk — leaving
    /// the rest of the conversation untouched.
    ///
    /// The difference from [`Self::tombstone_timeline`] is what survives. A
    /// timeline tombstone retires a whole conversation; this drops one turn's
    /// *content* while the turn itself stays in the index, so the compactor
    /// sheds its bulk records and a reload restores it as an empty hole. Turn
    /// numbering is therefore stable across the drop — nothing downstream has to
    /// renumber, and nothing dangles.
    ///
    /// **This is what makes an endless conversation affordable.** A transcript
    /// that never ends is a transcript no compaction pass can ever reclaim: a
    /// sealed segment is only relocated once enough of it is dead, and turns
    /// that stay live forever never make it dead. Retiring the tail past a depth
    /// is what turns an unbounded log into a bounded one.
    pub fn tombstone_turn(&self, timeline: TimelineId, turn_index: u32) -> candle::Result<()> {
        let mark = self.update(|s| s.tombstone_turn(timeline, turn_index));
        if self.read_only {
            return Ok(());
        }
        // Retention calls this in a loop of up to a few hundred turns, so the
        // append is the owner's and this returns without waiting.
        self.persistence.enqueue(WriteJob::TurnTombstone {
            mark,
            reason: Some("retention".to_string()),
        });
        Ok(())
    }

    /// Whether `(timeline, turn)` has already been dropped by a turn-scoped
    /// tombstone. Callers retiring a tail consult this so a turn is written
    /// once rather than on every pass.
    pub fn is_turn_tombstoned(&self, timeline: TimelineId, turn_index: u32) -> bool {
        self.read().is_turn_tombstoned(timeline, turn_index)
    }

    /// Mark a code_read scope fork's timeline as transient scratch — its sealed KV is
    /// spliced by reference onto the file timeline and never persisted to cold, so a
    /// cold copy can't be stranded as an orphan when the fork is tombstoned. In-memory
    /// only (a transient fork is never reloaded), so unlike [`Self::tombstone_timeline`]
    /// there is no redo-log record.
    pub fn mark_timeline_transient(&self, timeline: TimelineId) {
        self.update(|s| s.mark_timeline_transient(timeline));
    }

    /// Mark a code_read scope fork's timeline as a **splice source**: its turns
    /// keep their hot copy — the only copy [`Self::adopt_turn`] reads — until the
    /// fork is tombstoned. See `Substrate::mark_timeline_splice_source`.
    pub fn mark_timeline_splice_source(&self, timeline: TimelineId) {
        self.update(|s| s.mark_timeline_splice_source(timeline));
    }

    /// Mark a throwaway conversation's frame as transient — see
    /// [`crate::substrate::Substrate::mark_section_transient`]. In-memory only.
    pub fn mark_section_transient(&self, section: SectionId) {
        self.update(|s| s.mark_section_transient(section));
    }

    /// Whether a section is transient — see
    /// [`crate::substrate::Substrate::is_section_transient`].
    pub fn is_section_transient(&self, section: SectionId) -> bool {
        self.read().is_section_transient(section)
    }

    /// Remove a transient section — see
    /// [`crate::substrate::Substrate::retire_section`].
    pub fn retire_section(&self, section: SectionId) -> bool {
        self.update(|s| s.retire_section(section))
    }

    /// The next unused id for a section a conversation adds at runtime — see
    /// [`crate::substrate::Substrate::allocate_owned_section`].
    pub fn allocate_owned_section(&self) -> Option<SectionId> {
        self.update(|s| s.allocate_owned_section())
    }

    /// Record that `owner` owns `section` under `name`.
    pub fn register_owned_section(&self, owner: TimelineId, name: &str, section: SectionId) {
        self.update(|s| s.register_owned_section(owner, name, section));
    }

    /// The sections `owner` owns, in submission order.
    pub fn owned_sections_of(&self, owner: TimelineId) -> Vec<OwnedSection> {
        self.read().owned_sections_of(owner)
    }

    /// The section `owner` owns under `name`.
    pub fn owned_section_named(&self, owner: TimelineId, name: &str) -> Option<SectionId> {
        self.read().owned_section_named(owner, name)
    }

    /// Release a section `owner` owns: drop it from the substrate and write the
    /// [`crate::persistence::record::RecordType::SectionTombstone`] for its
    /// stream, so a restart does not restore it. A section `owner` does not own
    /// is left alone.
    pub fn release_owned_section(
        &self,
        owner: TimelineId,
        section: SectionId,
    ) -> candle::Result<()> {
        let stream = self.update(|s| s.release_owned_section(owner, section));
        self.write_owned_tombstones(stream)
    }

    /// Release everything `owner` owns — see [`Self::release_owned_section`] —
    /// returning the ids, so the caller can clear the scheduler's tables keyed
    /// by them.
    pub fn release_owned_sections(&self, owner: TimelineId) -> candle::Result<Vec<SectionId>> {
        let (ids, streams): (Vec<SectionId>, _) = self.update(|sub| {
            let ids = sub
                .owned_sections_of(owner)
                .into_iter()
                .map(|o| o.section)
                .collect();
            (ids, sub.release_owned_sections(owner))
        });
        self.write_owned_tombstones(streams)?;
        self.release_persisted_owned_sections(owner, None, None)?;
        Ok(ids)
    }

    /// Tombstone the streams `owner` has persisted that no live section of its
    /// holds: those declared under `name`, or under any name when `name` is
    /// `None`, except `keep`. Returns how many it ended.
    ///
    /// The ownership registry is empty after a restart, so a stream sealed
    /// before it is claimed by nobody until its owner submits the section
    /// again, and one its owner never submits again — a finished mission, a
    /// conversation retired since — would stay on disk forever. The stream
    /// declarations are persisted and name their owner, so this finds them
    /// there.
    pub fn release_persisted_owned_sections(
        &self,
        owner: TimelineId,
        name: Option<&str>,
        keep: Option<StreamId>,
    ) -> candle::Result<usize> {
        let stale: Vec<StreamId> = self
            .read()
            .persisted_owned_streams(owner, name)
            .into_iter()
            .filter(|stream| Some(*stream) != keep)
            .collect();
        for stream in &stale {
            self.tombstone_section(*stream, Some("owner released"))?;
        }
        Ok(stale.len())
    }

    pub(super) fn write_owned_tombstones(
        &self,
        streams: impl IntoIterator<Item = (StreamId, Option<SectionGeneration>)>,
    ) -> candle::Result<()> {
        if self.read_only {
            return Ok(());
        }
        for (stream, generation) in streams {
            self.persistence.enqueue(WriteJob::SectionTombstone {
                stream_id: stream.0,
                generation,
                reason: Some("owner released".to_string()),
            });
        }
        Ok(())
    }

    /// Couple `from_turn` to the tool response that follows it — in-RAM (so this
    /// session's summariser groups the exchange immediately) and on disk (a
    /// [`crate::persistence::record::RecordType::TurnCoupling`] record, so the
    /// grouping survives reload).
    ///
    /// Must be called before the response turn is submitted: that ordering is
    /// what stops the summariser observing half an exchange and freezing a leaf
    /// over it. On a read-only substrate the in-RAM coupling is all there is.
    pub fn couple_turn(&self, timeline: TimelineId, from_turn: u32) -> candle::Result<()> {
        let identity = self.minter.mint(0);
        self.update(|s| s.couple_turn(timeline, from_turn, identity));
        if self.read_only {
            return Ok(());
        }
        self.persistence
            .call(|p| p.write_turn_coupling(timeline.raw(), from_turn, identity))
            .map_err(|e| candle::Error::Msg(format!("write_turn_coupling: {e}")))?;
        Ok(())
    }

    /// Whether `timeline` has been tombstoned.
    pub fn is_timeline_tombstoned(&self, timeline: TimelineId) -> bool {
        self.read().is_tombstoned(timeline)
    }

    /// Mark `timeline` for distillation at `mode` — in-RAM (so the compactor
    /// sees it this session) and on disk (a
    /// [`crate::persistence::record::RecordType::Distilled`] record, so it
    /// survives reload). Its turns shed content at the next compaction. A later
    /// call may upgrade the mode (e.g. provenance-only → text-only on archive).
    pub fn distill_timeline(&self, timeline: TimelineId, mode: DistillMode) -> candle::Result<()> {
        self.update(|s| s.distill_timeline(timeline, mode));
        let payload = DistillPayload {
            timeline_id: timeline.raw(),
            mode,
        }
        .encode();
        self.persistence.enqueue(WriteJob::ConvMeta {
            record: RecordType::Distilled,
            payload,
            identity: self.minter.mint(0),
        });
        Ok(())
    }

    /// Whether `timeline` is marked for distillation.
    pub fn is_timeline_distilled(&self, timeline: TimelineId) -> bool {
        self.read().is_distilled(timeline)
    }
}
