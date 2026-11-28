//! Tracks which timelines, sections and tombstones are ephemeral or owned, and
//! what releasing each one frees and writes to the log.
//!
//! A transient timeline's KV is never written to disk, because nothing reads it
//! back: a code_read fork's turns are spliced by reference onto a file timeline,
//! and a calibration exemplar's KV is distilled away on load. A section tombstone
//! ends one generation of a stream, so a later declaration of the same content
//! starts a new one and the earlier values are refused. Releasing an owned
//! section ends its persisted generation, while a shared section's stream
//! outlives any one conversation that reads it. Distillation releases both
//! resident tiers and the cold reference, since the compactor sheds the record
//! that reference names.

use super::*;
use crate::files::is_member_stream_name;

impl Substrate {
    /// A section tombstone ends one **generation** of a stream, not the stream.
    ///
    /// A section's stream id is a pure function of its content address, so a
    /// fresh prefill of the same text lands on the same id. The generation
    /// `generation` names is ended for good: its declaration and every value it
    /// wrote leave the index, and whatever of it arrives later is refused
    /// ([`Self::apply_stream_decl`], `admits`). Compaction and segment
    /// maintenance copy only what the index names, so its records are
    /// reclaimed rather than carried forward. A later generation's declaration
    /// and values stand.
    ///
    /// `None` is a tombstone written before generations: it drops the
    /// unstamped stream as the log stands at that point, and the next
    /// unstamped declaration starts the stream again.
    pub(super) fn end_section(
        &mut self,
        stream_id: StreamId,
        generation: Option<SectionGeneration>,
    ) {
        // A section is no turn, and no group's scan reads it; a turn the
        // stream held is its timeline's.
        let held_turn = self.stream_turn_timeline(StreamKey {
            stream_id,
            writer: generation.map_or(0, |g| g.writer),
        });
        match generation {
            Some(g) => {
                self.ended_sections.end(stream_id, g);
                let key = StreamKey {
                    stream_id,
                    writer: g.writer,
                };
                let declared_here = self.streams.physical(key).is_some_and(|rt| {
                    matches!(rt.decl, Some(StreamDecl::PromptSection(_)))
                        && rt.ids.decl.oseq == g.oseq
                });
                if declared_here {
                    self.streams.undeclare(key);
                }
                self.streams.prune(key, |held| held.epoch != g.oseq);
            }
            None => {
                self.ended_sections.end_unstamped(stream_id);
                let key = StreamKey {
                    stream_id,
                    writer: 0,
                };
                self.streams.remove_writer(key);
                self.tagged_turns.remove(key);
            }
        }
        // The state the section's seal ended on goes with the generation that
        // sealed it. Not retired for good: a later generation of the same
        // content ends on the same state and keeps it again.
        self.drop_snapshot_loc(section_end_state_stream_id(stream_id));
        self.gallery_epoch += 1;
        if let Some(timeline) = held_turn {
            self.timeline_content_changed(timeline);
        }
    }

    /// Mark `timeline`'s durable state as **ephemeral**: nothing will ever read
    /// it back from disk, so neither its KV nor its recurrent snapshot is
    /// written. Flags residences already allocated; `append_complete` flags
    /// future ones via the `transient_timelines` set, and
    /// `Conversation::enqueue_recurrent_snapshot` skips the timeline outright.
    /// In-memory only.
    ///
    /// The invariant is "no durable consumer", which two very different callers
    /// satisfy for different reasons:
    ///
    /// - **code_read scope forks.** Their sealed turns are spliced by REFERENCE
    ///   onto a file timeline and the fork is then tombstoned. The file timeline
    ///   holds the durable copy; a cold copy of the fork would only strand an
    ///   unreconstructable orphan once the fork's decl drops.
    /// - **Calibration exemplars.** The seal captures the turn's wide-Q
    ///   signature, which persists on its own record, and the exemplar's designed
    ///   end state is `DistillMode::ProvenanceOnly` — KV dropped, signature kept.
    ///   Cold-persisting the KV therefore writes bytes that the next load's
    ///   distillation deletes. Measured on the 2,998-case corpus: ~4 GB of redo
    ///   log per minute, ~108 GB in total, every byte of it discarded.
    ///
    /// Safe for a kept (non-tombstoned) timeline because a chunkless turn is
    /// legal on reload: `classify_turn` only reports `MissingKv` when the tokens
    /// record is *also* gone, so a turn that kept its tokens classifies `Ok`.
    ///
    /// **The warm copy needs this flag to be reclaimable, and reads it.**
    /// `purge_warm_to_budget` normally refuses to drop a warm-only residence,
    /// because without a hot or cold copy behind it the warm bytes are the
    /// turn's last. For a timeline marked here the cold copy is never coming, so
    /// that refusal would hold forever — which it did: warm reached 8.8 GiB on a
    /// full calibration and could not be shed. The purge therefore treats
    /// `no_cold_persist` warm-only residences as disposable, which is exactly
    /// what this mark declares them to be.
    pub fn mark_timeline_transient(&mut self, timeline: TimelineId) {
        self.transient_timelines.insert(timeline);
        let residences: Vec<ResidenceIndex> = match self.timelines.get(&timeline) {
            Some(entry) => entry.turns.values().map(|t| t.content.residence).collect(),
            None => return,
        };
        for r in residences {
            self.residences.get(r).set_flag(NO_COLD_PERSIST);
        }
    }

    /// Whether `timeline`'s durable state is ephemeral — see
    /// [`Self::mark_timeline_transient`]. Read by the recurrent-snapshot enqueue,
    /// which skips such a timeline entirely.
    pub fn is_timeline_transient(&self, timeline: TimelineId) -> bool {
        self.transient_timelines.contains(&timeline)
    }

    /// Mark `timeline` as **never resumed**: its turns are kept — sealed,
    /// cold-persisted, projectable — but no slot will ever continue it, so it
    /// keeps no recurrent snapshot. The weaker sibling of
    /// [`Self::mark_timeline_transient`], for content that is read and never
    /// extended: a document ingested as a conversation, a kept dream.
    ///
    /// A snapshot is a whole recurrent state, fixed-size per sequence whatever
    /// its token count — ~112 MiB on Flash-Next — and the single-tail rule keeps
    /// one per live conversation forever. A store of thousands of one-turn
    /// documents therefore held hundreds of gigabytes of state nothing would
    /// ever read. In-memory; `Conversation::mark_timeline_unresumed` writes the
    /// empty snapshot that retires a tail already on disk.
    pub fn mark_timeline_unresumed(&mut self, timeline: TimelineId) {
        self.unresumed_timelines.insert(timeline);
    }

    /// Whether `timeline` keeps no recurrent snapshot — see
    /// [`Self::mark_timeline_unresumed`].
    pub fn is_timeline_unresumed(&self, timeline: TimelineId) -> bool {
        self.unresumed_timelines.contains(&timeline)
    }

    /// Whether `timeline` is declared a **file store** — its whole content file
    /// events, with no conversation registered on it — by a live custom object
    /// under [`file_store_key`]. A rewrite of the log carries a store's events
    /// (`persistence::vfs::carried`); undeclared, it reads as an orphan and
    /// every compaction drops it. The declaration is a record like any other:
    /// it replicates, it is in place when the log is read back before
    /// maintenance can start, and it is never released: a store's discard
    /// tombstones its events, and a release racing another member's next
    /// save would leave that save undeclared.
    pub fn is_file_store(&self, timeline: TimelineId) -> bool {
        self.custom_object(&file_store_key(timeline)).is_some()
    }

    /// Mark `timeline` as a **splice source**: its turns keep their hot copy
    /// until the timeline is tombstoned (see [`residence::SPLICE_SOURCE`]).
    /// Call it right after minting a code_read scope fork, before its first turn
    /// seals — like [`Self::mark_timeline_transient`], it flags the residences
    /// already allocated and every later one.
    pub fn mark_timeline_splice_source(&mut self, timeline: TimelineId) {
        self.splice_source_timelines.insert(timeline);
        let residences: Vec<ResidenceIndex> = match self.timelines.get(&timeline) {
            Some(entry) => entry.turns.values().map(|t| t.content.residence).collect(),
            None => return,
        };
        for r in residences {
            self.residences.get(r).set_flag(SPLICE_SOURCE);
        }
    }

    /// Mark `section`'s durable state as **ephemeral** — the frame of a
    /// conversation opened for one job and thrown away.
    ///
    /// The section counterpart of [`Self::mark_timeline_transient`], for the
    /// same reason: nothing will read it back from disk, so writing it costs a
    /// redo-log write compaction would only have to take back. Its residence is
    /// flagged [`residence::NO_COLD_PERSIST`] — now when it is already
    /// installed, at install otherwise — the scheduler's seal skips its stream
    /// declaration and token record, and [`Self::retire_section`] removes it
    /// once its last user is done.
    ///
    /// Call it before the section seals: a seal that has already declared the
    /// stream has written the record this exists to avoid.
    pub fn mark_section_transient(&mut self, section: SectionId) {
        self.transient_sections.insert(section);
        if let Some(entry) = self.sections.get(&section) {
            self.residences
                .get(entry.residence)
                .set_flag(NO_COLD_PERSIST);
        }
    }

    /// Whether `section`'s durable state is ephemeral — see
    /// [`Self::mark_section_transient`].
    pub fn is_section_transient(&self, section: SectionId) -> bool {
        self.transient_sections.contains(&section)
    }

    /// Remove a transient section: its entry, and its hot and warm KV.
    ///
    /// **Transient sections only.** A persisted section has a stream on disk
    /// that the next load registers again, so removing its entry here would
    /// only make this process disagree with the next one; it is refused.
    /// Returns whether a section was removed.
    ///
    /// The KV chunks are reference-counted, so a slot still holding the
    /// section's injected copy keeps its own until the slot is freed — dropping
    /// the substrate's reference releases nothing another holder is using.
    pub fn retire_section(&mut self, section: SectionId) -> bool {
        if self.transient_sections.remove(&section).is_none() {
            return false;
        }
        let Some(entry) = self.sections.remove(&section) else {
            return false;
        };
        self.section_token_total = self.section_token_total.saturating_sub(entry.token_count);
        // Transient, so never cold-persisted: the kill takes warm as well.
        self.residences
            .get(entry.residence)
            .set_flag(NO_COLD_PERSIST);
        self.residences.kill(entry.residence);
        true
    }

    /// The next unused id for a section a conversation adds at runtime — see
    /// [`OwnedSections`].
    pub fn allocate_owned_section(&mut self) -> Option<SectionId> {
        self.owned_sections.allocate()
    }

    /// Record that `owner` owns `section` under `name`.
    pub fn register_owned_section(&mut self, owner: TimelineId, name: &str, section: SectionId) {
        self.owned_sections.register(owner, name, section);
    }

    /// The sections `owner` owns, in submission order.
    pub fn owned_sections_of(&self, owner: TimelineId) -> Vec<OwnedSection> {
        self.owned_sections.of(owner).to_vec()
    }

    /// The section `owner` owns under `name`.
    pub fn owned_section_named(&self, owner: TimelineId, name: &str) -> Option<SectionId> {
        self.owned_sections.named(owner, name)
    }

    /// Release one owned section: forget `owner`'s claim, end its stream's
    /// persisted generation, and drop its entry with its hot and warm K/V.
    ///
    /// The persisted counterpart of [`Self::retire_section`]. An owned
    /// section's address is salted with its owner, so its stream is read by no
    /// one else and ending it is safe. A shared one's is not ended: a stream
    /// every conversation reads must outlive any one of them, so only the
    /// claim and the entry go ([`Self::drop_owned_entry`]).
    ///
    /// Returns the stream the caller must tombstone **on disk**, or `None`
    /// when `owner` does not own `section` or it never sealed (nothing was
    /// written).
    pub fn release_owned_section(
        &mut self,
        owner: TimelineId,
        section: SectionId,
    ) -> Option<(StreamId, Option<SectionGeneration>)> {
        if !self.owned_sections.release(owner, section) {
            return None;
        }
        self.drop_owned_entry(section)
    }

    /// Release everything `owner` owns, returning the streams to tombstone on
    /// disk with the generation each ended — see
    /// [`Self::release_owned_section`].
    pub fn release_owned_sections(
        &mut self,
        owner: TimelineId,
    ) -> Vec<(StreamId, Option<SectionGeneration>)> {
        self.owned_sections
            .release_all(owner)
            .into_iter()
            .filter_map(|o| self.drop_owned_entry(o.section))
            .collect()
    }

    /// The `debug_name` an owned section's stream is declared under. The
    /// declaration is persisted, so it names the owner after a restart, when
    /// the in-memory ownership registry is empty.
    pub fn owned_stream_name(owner: TimelineId, name: &str) -> String {
        format!("conv/{}/{name}", owner.raw())
    }

    /// The live section streams `owner` has persisted: those declared under
    /// `name`, or under any name when `name` is `None`. Read from the stream
    /// declarations, so it holds across a restart; a tombstoned stream is not
    /// listed.
    pub fn persisted_owned_streams(&self, owner: TimelineId, name: Option<&str>) -> Vec<StreamId> {
        let mut found = match name {
            Some(n) => self
                .streams
                .sections_named(&Self::owned_stream_name(owner, n)),
            None => self
                .streams
                .sections_named_under(&format!("conv/{}/", owner.raw())),
        };
        found.sort_unstable_by_key(|s| s.0);
        found
    }

    /// End `section`'s stream and remove its entry; the stream, if it sealed,
    /// with the generation ended.
    ///
    /// **A shared section's stream is not ended**: every conversation that
    /// would seal exactly it reads it, so one leaving drops only its own entry,
    /// and the stream is the sweep's to retire
    /// (`docs/shared_journal_sections.md` §6). Nothing is returned to
    /// tombstone on disk.
    pub(super) fn drop_owned_entry(
        &mut self,
        section: SectionId,
    ) -> Option<(StreamId, Option<SectionGeneration>)> {
        let stream = self.section_stream(section)?;
        let shared = self.is_shared_section_stream(stream);
        let generation = if shared {
            None
        } else {
            self.tombstone_section(stream)
        };
        let entry = self.sections.remove(&section)?;
        self.section_token_total = self.section_token_total.saturating_sub(entry.token_count);
        self.residences.take_warm(entry.residence);
        (!shared).then_some((stream, generation))
    }

    /// Whether section stream `stream` is a shared section's — declared under
    /// a [`is_shared_stream_name`] name.
    pub fn is_shared_section_stream(&self, stream: StreamId) -> bool {
        matches!(
            self.streams.get(stream).and_then(|s| s.decl.as_ref()),
            Some(StreamDecl::PromptSection(d)) if is_shared_stream_name(&d.debug_name)
        )
    }

    /// Every shared journal section stream the substrate holds, by id — what
    /// the sweep chooses from (`docs/shared_journal_sections.md` §6).
    ///
    /// **Never a knowledge-base member.** A member is declared under the shared
    /// prefix too (`shared/files/…`), but no journal run holds it and, on a
    /// fresh start, no conversation does either — so the sweep, given every
    /// shared stream, retired the whole knowledge base as "superseded" on the
    /// first pass after a restart, and segment maintenance then reclaimed the
    /// segments its chunks lived in as dead (2026-11-28: 17,658 members
    /// retired, nine segments unlinked). A member ends only when its file
    /// does, through the knowledge base's own records.
    pub fn shared_section_streams(&self) -> Vec<StreamId> {
        let mut found: Vec<StreamId> = self
            .streams
            .sections_named_under(SHARED_SECTION_NAME_PREFIX)
            .into_iter()
            .filter(|&s| {
                !matches!(
                    self.streams.get(s).and_then(|st| st.decl.as_ref()),
                    Some(StreamDecl::PromptSection(d)) if is_member_stream_name(&d.debug_name)
                )
            })
            .collect();
        found.sort_unstable_by_key(|s| s.0);
        found.dedup();
        found
    }

    /// The section streams a live section entry reads — a shared stream one of
    /// this member's conversations still holds.
    pub fn held_section_streams(&self) -> HashSet<StreamId> {
        self.sections
            .values()
            .map(|e| self.residences.get(e.residence).stream_id)
            .collect()
    }

    /// Whether `timeline` has been tombstoned.
    pub fn is_tombstoned(&self, timeline: TimelineId) -> bool {
        self.tombstoned_timelines.contains(&timeline)
    }

    /// Direct read of the tombstoned-timeline set.  Used by the
    /// compactor to filter dead records out of the next compacted
    /// log.
    pub fn tombstoned_timelines(&self) -> &PSet<TimelineId> {
        &self.tombstoned_timelines
    }

    /// Replay a persisted [`DistillPayload`] — marks the timeline for
    /// distillation at its recorded [`DistillMode`]. A later record upgrades the
    /// mode (last-writer-wins), so a conversation distilled provenance-only and
    /// then archived ends up `TextOnly`.
    ///
    /// Routed through [`Self::distill_timeline`] rather than inserting into the
    /// map directly, so replay retires the recurrent tail exactly as the live
    /// marking does. Inserting here — which it used to — meant a *reloaded*
    /// distilled timeline had its tail back in the index, and the incremental
    /// maintenance that runs against a reloaded store carried it forward again.
    pub fn apply_distill(&mut self, payload: &DistillPayload) {
        if let Some(timeline) = TimelineId::from_raw(payload.timeline_id) {
            self.distill_timeline(timeline, payload.mode);
        }
    }

    /// Mark `timeline` for distillation in-RAM at `mode` (callers also write the
    /// matching `Distilled` record so the marker survives reload).
    pub fn distill_timeline(&mut self, timeline: TimelineId, mode: DistillMode) {
        self.distilled_timelines.insert(timeline, mode);
        // The recurrent-state tail goes with the content, exactly as a timeline
        // tombstone drops it. The state is derived from the token stream this
        // marker sheds and is unusable without the K/V it was computed against,
        // so a distilled timeline's snapshot describes a resume that can never
        // happen — distillation is for calibration exemplars, which are never
        // resumed.
        //
        // Retiring it at the index, rather than gating each rewrite on the
        // distilled set, is what keeps the two rewrite paths from disagreeing.
        // They did: full compaction gated, incremental maintenance did not, and
        // a production store carried 766 distilled tails of ~116 MiB each —
        // ~89 GB of a 120 GB log — forward on every pass.
        self.retire_snapshot_stream(timeline);
        self.release_distilled_kv(timeline);
    }

    /// Drop the hot and warm KV of every turn in a timeline just marked for
    /// distillation. Returns how many residences were released.
    ///
    /// **Both [`DistillMode`]s drop KV chunks**, so the mark is a declaration
    /// that this timeline's K/V is disposable — the signature and whatever text
    /// the mode keeps live on their own records. Waiting for the compactor to
    /// act on that declaration is what made it expensive: the mark went into a
    /// map and the bytes stayed in RAM until a compaction that may not run for
    /// the rest of the phase.
    ///
    /// Measured on a full tool calibration: 2,222 archived exemplars, every one
    /// marked `ProvenanceOnly` the moment its signature was captured, holding
    /// **7.2 GiB** of warm host RAM between them. Nothing could ever read it —
    /// `belief_gallery` scores against the persisted signature blob, never the
    /// K/V — and the retire path had explicitly demoted it hot→warm to keep VRAM
    /// flat, so the garbage was moved into host RAM rather than released. That
    /// 7.2 GiB is what pushed the box to 0.9 GiB free and 1,302 pages/sec.
    ///
    /// **The cold REFERENCE goes with them, because the compactor sheds the
    /// record it names.** `keep_chunks` is `distill.is_none()`, so every chunk
    /// this residence points at is dropped from the log on the next pass — and
    /// nothing else ever removes the reference. Maintenance repoints relocated
    /// records and has no notion of a dropped one; the in-RAM stream index is
    /// insert-only. So a cold ref left standing here does not become stale
    /// later, it becomes a pointer into a segment that gets unlinked, and it
    /// still reads as a cold copy to everything that asks.
    ///
    /// That is not theoretical. An archived conversation is marked `TextOnly`,
    /// which keeps its tokens and sheds its chunks — so its turns kept a
    /// non-zero `token_count` and a cold tier that no longer existed. Selection
    /// chose them on the strength of both, `elevate_to_hot` answered
    /// `cold-load found no chunks`, `apply_projection` dropped them, and the
    /// whole thing repeated on the next turn and every turn after, because
    /// nothing had changed. Clearing it here makes the turn tier-less, which is
    /// what it is: the projection then skips it on the branch that already
    /// exists for K/V-less turns instead of failing a load to discover it.
    ///
    /// `released` still counts only hot and warm — it reports memory handed
    /// back, and a cold ref is not memory.
    pub(super) fn release_distilled_kv(&mut self, timeline: TimelineId) -> usize {
        let residences: Vec<ResidenceIndex> = match self.timelines.get(&timeline) {
            Some(entry) => entry.turns.values().map(|t| t.content.residence).collect(),
            None => return 0,
        };
        let mut released = 0usize;
        for r in residences {
            let slot = self.residences.get(r);
            // The same two exclusions every hot-drop path honours: a residence
            // the current wave is attending, and one whose `hot` is the interim
            // native form a quantize drain still owes. The second is rare here —
            // a case is marked only once archived — and the compactor reclaims
            // anything this pass skips, exactly as it did before.
            if slot.is_pinned() || slot.has_flag(PENDING_QUANTIZE) {
                continue;
            }
            let hot = self.residences.take_hot(r).is_some();
            let warm = self.residences.take_warm(r).is_some();
            if hot || warm {
                released += 1;
            }
            self.residences.set_cold(r, None);
        }
        released
    }
}
