//! Reads and writes the prompt sections the substrate holds, such as system
//! prompts and tool catalogs, with their index pages and tier state.
//!
//! A section is sealed once and addressed by a content-addressed stream id, so
//! every conversation that seals the same text reads the same stream. A section
//! moves between hot, warm and cold without the turn machinery, and a section
//! still owed a quantize is held off the persistence thread until the scheduler
//! replaces its native bytes. A tombstoned section reads as absent, so the next
//! triage prefills it fresh rather than reloading corrupt chunks. A section
//! restored from the log hands the model the index page its prefill would have
//! sealed.

use super::*;

impl Substrate {
    pub fn reset(&mut self) {
        // `timelines` owns the per-turn store now, so clearing it
        // drops every turn alongside its parent timeline entry.
        self.timelines.clear();
        self.timelines_by_group.clear();
        self.sections.clear();
        self.transient_sections.clear();
        self.owned_sections.clear_ownership();
        self.timeline_token_totals.clear();
        self.calibrations_of.clear();
        self.units_by_key.clear();
        self.section_token_total = 0;
        self.gallery_epoch += 1;
        self.all_groups_content_changed();
        self.sig_rewrites += 1;
        self.galleries_changed();
    }

    /// The stored projection-event record payload for a turn, if any.
    pub fn projection_events_blob(&self, timeline: TimelineId, index: TurnIndex) -> Option<&[u8]> {
        self.streams
            .get(turn_stream_id(timeline.raw(), index.0))
            .and_then(|s| s.projection_events.as_deref())
    }

    /// The stored wide-Q signature window payload for a turn, if any.
    pub fn wide_q_sigs_blob(&self, timeline: TimelineId, index: TurnIndex) -> Option<&[u8]> {
        self.streams
            .get(turn_stream_id(timeline.raw(), index.0))
            .and_then(|s| s.wide_q_sigs.as_deref())
    }

    /// Where a turn's QSA index page is, if it has one — held, or located in
    /// the log ([`IndexPage`]).
    ///
    /// What a projection hands the model alongside a borrowed turn's K/V, read
    /// through [`crate::Conversation::turn_index_page`] when it is in the log.
    /// `None` means the turn's keys can be borrowed but not indexed — the slot
    /// will hold tokens no cache accounts for.
    pub fn index_page_at(&self, timeline: TimelineId, index: TurnIndex) -> Option<IndexPage> {
        self.index_page_of(turn_stream_id(timeline.raw(), index.0))
    }

    /// Where a stream's QSA index page is — a turn's, or a prompt section's by
    /// its content-addressed stream id — the page a section restored from the
    /// log hands the model in place of the one its prefill would have sealed.
    pub fn index_page_of(&self, stream_id: StreamId) -> Option<IndexPage> {
        self.streams
            .get(stream_id)
            .and_then(|s| s.index_page.clone())
    }

    /// [`Self::index_page_of`] with the identity of the record that holds the
    /// page — what a reader keys the bytes it read by, since a relocated copy
    /// keeps it and a new page for the stream does not.
    pub fn index_page_record_of(&self, stream_id: StreamId) -> Option<(IndexPage, RecordIdentity)> {
        self.streams
            .get(stream_id)
            .and_then(|s| s.index_page.clone().map(|page| (page, s.ids.index_page)))
    }

    /// A sealed turn's gather-scope tags, as persisted on its `TurnDecl`.
    ///
    /// These are what route a seal-time normalization observation to the scopes
    /// the turn actually belongs to: a `tool`-tagged turn teaches the tool lens,
    /// an untagged dialogue turn does not. Empty for an untagged turn and for a
    /// turn that is not present.
    pub fn turn_tags(&self, timeline: TimelineId, index: TurnIndex) -> Vec<String> {
        let Some(stream) = self.streams.get(turn_stream_id(timeline.raw(), index.0)) else {
            return Vec::new();
        };
        match &stream.decl {
            Some(StreamDecl::Turn(decl)) => decl.tags.clone(),
            _ => Vec::new(),
        }
    }

    /// Whether the turn carries any of `tags` — [`Self::turn_tags`] read in
    /// place, for a scan that asks it of every turn of every conversation in
    /// a group and wants no copy of each turn's tags to do it.
    pub fn turn_has_any_tag(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        tags: &[String],
    ) -> bool {
        let Some(stream) = self.streams.get(turn_stream_id(timeline.raw(), index.0)) else {
            return false;
        };
        match &stream.decl {
            Some(StreamDecl::Turn(decl)) => decl.tags.iter().any(|t| tags.contains(t)),
            _ => false,
        }
    }

    /// The turn's user-half span in its real-KV grid — the Concept F question
    /// window (`docs/provenance_adaptive_projection.md` §8). Read from the
    /// persisted turn layout's `User` segments; `gather_wide_sigs` emits one
    /// signature per real token 1:1 with this grid, so the range indexes the
    /// decoded sig window directly. `None` when the turn has no user segment.
    pub fn user_sig_span(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<std::ops::Range<usize>> {
        self.turn_phase_span(turn_stream_id(timeline.raw(), index.0), Phase::User)
    }

    /// The signature sub-range one `phase` of a stored turn occupies, or `None`
    /// when the turn is unknown, is not a turn, or has no real K/V for that
    /// phase. Signatures are 1:1 with the turn's real tokens, so the range
    /// indexes its signature window directly.
    pub fn turn_phase_span(
        &self,
        stream_id: StreamId,
        phase: Phase,
    ) -> Option<std::ops::Range<usize>> {
        let stream = self.streams.get(stream_id)?;
        let Some(StreamDecl::Turn(decl)) = &stream.decl else {
            return None;
        };
        phase_span_of(&decl.segments, phase)
    }

    // ── Section accessors ────────────────────────────────────────────────────

    /// Create a section entry atomically with all data including sealed KV.
    ///
    /// `sealed` is the K/V the section's hot tier takes — the caller's
    /// migration of its GPU snapshot, done before the version edit so a
    /// retried edit never runs it twice.
    ///
    /// `block_start`/`block_end` are omitted — written later by
    /// `set_section_block_range` when the section is injected.
    ///
    /// `stream_id` is the content-addressed stream id under which this
    /// section's chunks land in the redo log.  The substrate uses it
    /// for the residence so the persistence thread's section-persist
    /// pass picks the section up by snapshotting hot bytes for any
    /// section residence whose `stream_id != default && cold == None`.
    pub fn set_section_full(
        &mut self,
        section: SectionId,
        stream_id: StreamId,
        token_count: usize,
        sealed: Vec<SealedSequence>,
        tokens: Arc<Vec<u32>>,
    ) -> ResidenceIndex {
        let residence = self.alloc_section_residence(stream_id);
        // A throwaway conversation's frame never reaches the cold tier — see
        // `mark_section_transient`.
        if self.transient_sections.contains(&section) {
            self.residences.get(residence).set_flag(NO_COLD_PERSIST);
        }
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            tokens,
            residence,
        };
        let prev = self
            .sections
            .insert(section, entry)
            .map_or(0, |e| e.token_count);
        self.section_token_total = self.section_token_total + token_count - prev;
        if !sealed.is_empty() {
            self.install_hot(residence, sealed);
        }
        residence
    }

    /// Section variant of [`Self::restore_turn`] — install a section
    /// recovered from the redo log directly into the hot tier without
    /// going through a fresh forward pass.
    ///
    /// `sealed_hot` is the per-layer K/V already materialised in VRAM
    /// (by the caller's `load_section_into_hot` cold-load).  `cold` is
    /// the on-disk reference list returned by `recover_section_cold_refs`;
    /// installing it pre-emptively marks the section as already
    /// persisted so the persistence thread's section-persist pass
    /// skips it.
    pub fn restore_section(
        &mut self,
        section: SectionId,
        stream_id: StreamId,
        token_count: usize,
        sealed_hot: Vec<SealedSequence>,
        cold: Vec<StoredSequence>,
        tokens: Arc<Vec<u32>>,
    ) {
        let residence = self.alloc_section_residence(stream_id);
        self.residences.set_cold(residence, Some(cold));
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            tokens,
            residence,
        };
        let prev = self
            .sections
            .insert(section, entry)
            .map_or(0, |e| e.token_count);
        self.section_token_total = self.section_token_total + token_count - prev;
        if !sealed_hot.is_empty() {
            self.install_hot(residence, sealed_hot);
        }
    }

    /// True when `section` already has a hot-resident entry in the
    /// substrate — used by the ingest loop's skip-if-present check to
    /// avoid re-prefilling sections that recovered from the redo log.
    pub fn section_is_hot(&self, section: SectionId) -> bool {
        match self.sections.get(&section) {
            Some(entry) => self.residences.get(entry.residence).has_hot(),
            None => false,
        }
    }

    /// True when `section` is recorded in the substrate's section map
    /// at any tier (hot, warm, or cold-marker only).  Stricter than
    /// `section_is_hot`: a cold-marker section restored from the redo
    /// log returns true here even before any projection has elevated
    /// it to hot.  Used by the ingest loop to skip re-issuing a
    /// `RestoreSection` for a section the substrate already knows
    /// about (preventing duplicate residence allocations).
    ///
    /// A tombstoned section (see [`Self::tombstone_section`]) reads as
    /// absent here even though its entry is still in [`Self::sections`] —
    /// mirroring how a tombstoned timeline stays in [`Self::timelines`] and
    /// is filtered at every read site instead of being purged. The next
    /// triage that asks for this section id falls through to the ingest
    /// path, which re-prefills fresh and overwrites the tombstoned entry.
    pub fn section_exists(&self, section: SectionId) -> bool {
        match self.sections.get(&section) {
            Some(entry) => {
                !self.is_section_tombstoned(self.residences.get(entry.residence).stream_id)
            }
            None => false,
        }
    }

    /// Per-layer chunk count for the section's hot residence, or
    /// `None` if the section isn't hot.  Used by the ingest loop's
    /// skip path for the per-section block-count diagnostic.
    pub fn section_block_count(&self, section: SectionId) -> Option<usize> {
        let entry = self.sections.get(&section)?;
        let hot = self.residences.get(entry.residence).hot()?;
        hot.first().map(|s| s.chunks.len())
    }

    /// Reverse of [`Self::section_id_for_debug_name`]: the symbolic `debug_name`
    /// for a `SectionId` (tool name, `system.frame`, …). `SectionId` → residence →
    /// `stream_id` → `PromptSection.debug_name`. Used by the promote tracker.
    pub fn section_debug_name(&self, id: SectionId) -> Option<String> {
        let entry = self.sections.get(&id)?;
        let stream_id = self.residences.get(entry.residence).stream_id;
        match self.streams.get(stream_id)?.decl.as_ref()? {
            StreamDecl::PromptSection(s) => Some(s.debug_name.clone()),
            StreamDecl::Turn(_) => None,
        }
    }

    /// Look up a section by its persistence `debug_name` (the symbolic id
    /// passed to `insert_section` at ingest time): the stream declared under
    /// that name, through the stream index's name index, then the in-RAM
    /// `sections` entry that stream was installed as. Used by calibration
    /// consumers that pick out individual scenarios by their human-readable
    /// identifier after loading the workspace's full substrate. Returns the
    /// lowest-numbered stream when duplicate `debug_name`s exist (callers
    /// should keep names unique per workspace).
    pub fn section_id_for_debug_name(&self, debug_name: &str) -> Option<SectionId> {
        let stream_id = self
            .streams
            .sections_named(debug_name)
            .into_iter()
            .min_by_key(|s| s.0)?;
        // Map stream_id back to the in-RAM SectionId via the residence
        // slab — the residence holds the stream id we installed at
        // ingest time.
        self.sections.iter().find_map(|(sid, entry)| {
            if self.residences.get(entry.residence).stream_id == stream_id {
                Some(*sid)
            } else {
                None
            }
        })
    }

    /// Replace a section residence's hot bytes with `new_hot`, the quantized
    /// form of `native` — only while the residence still holds `native`;
    /// `false` when it does not (retired or replaced since), and nothing is
    /// installed. Used by the scheduler's section-quantize drain after it
    /// quantizes the section's K/V to the section-compression policy (C0 by
    /// default) — the new Q-format sequences land here so subsequent
    /// in-session reads see the same bytes the cold tier will reproduce on
    /// the next reload.
    pub fn replace_section_hot(
        &self,
        residence: ResidenceIndex,
        native: &Arc<HotKv>,
        new_hot: Vec<SealedSequence>,
    ) -> bool {
        debug_assert!(
            !new_hot.is_empty(),
            "replace_section_hot called with empty Vec"
        );
        self.residences.replace_hot(residence, native, new_hot)
    }

    /// Snapshot the section residences that have hot bytes installed
    /// but haven't been written to the cold tier yet — the work list
    /// for the persistence thread's section-persist pass.  Mirrors
    /// [`Self::snapshot_pending_cold`] but walks the `sections` map,
    /// since sections go hot→cold without a warm copy.  Skips
    /// sentinel-stream residences (`StreamId::default()` is never
    /// persisted).
    pub fn snapshot_pending_section_cold(
        &self,
    ) -> Vec<(ResidenceIndex, StreamId, Vec<SealedSequence>)> {
        self.sections
            .values()
            .filter_map(|entry| {
                let slot = self.residences.get(entry.residence);
                if !Self::section_cold_pending(slot) {
                    return None;
                }
                slot.hot()
                    .map(|hot| (entry.residence, slot.stream_id, hot.to_vec()))
            })
            .collect()
    }

    /// Whether a section residence awaits its redo-log write.
    pub(super) fn section_cold_pending(slot: &Residence) -> bool {
        slot.has_hot()
            && !slot.has_cold()
            && slot.stream_id != StreamId::default()
            && !slot.has_flag(PENDING_QUANTIZE)
    }

    /// Count of section residences awaiting their redo-log write — the COUNT
    /// companion to [`Self::snapshot_pending_section_cold`], with its exact
    /// filter, so the shutdown drain keeps passing until every sealed section
    /// is durable rather than stopping once the turn tiers are empty.
    pub fn pending_section_cold_count(&self) -> usize {
        self.sections
            .values()
            .filter(|entry| Self::section_cold_pending(self.residences.get(entry.residence)))
            .count()
    }

    /// Mark a section residence as awaiting the scheduler's quantize
    /// drain.  Called from `SealAction::Section` right after
    /// `set_section_full` installs the native bytes, when a
    /// `compression_policy` is configured.  While this flag is set,
    /// `snapshot_pending_section_cold` ignores the residence — the
    /// persistence thread holds off on writing native bytes that the
    /// scheduler is about to replace with their quantized form.
    pub fn mark_section_pending_quantize(&self, residence: ResidenceIndex) {
        self.residences.get(residence).set_flag(PENDING_QUANTIZE);
    }

    /// Clear the pending-quantize flag — called from the same drain
    /// that calls `replace_section_hot`.  After this the persistence
    /// thread is free to gather the residence's (now final) hot bytes
    /// and persist them.
    pub fn clear_section_pending_quantize(&self, residence: ResidenceIndex) {
        self.residences.get(residence).clear_flag(PENDING_QUANTIZE);
    }

    pub fn set_section_block_range(
        &mut self,
        section: SectionId,
        block_start: u64,
        block_end: u64,
    ) {
        if let Some(e) = self.sections.get_mut(&section) {
            e.block_range = (block_start, block_end);
        }
    }

    pub fn section_sealed_of(&self, section: SectionId) -> Option<Arc<Vec<SealedSequence>>> {
        let residence = self.sections.get(&section)?.residence;
        let hot = self.residences.get(residence).hot()?;
        Some(Arc::new(hot.to_vec()))
    }

    pub fn section_tokens_of(&self, section: SectionId) -> Arc<Vec<u32>> {
        self.sections
            .get(&section)
            .map(|e| Arc::clone(&e.tokens))
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    pub fn section_block_range(&self, section: SectionId) -> (u64, u64) {
        self.sections
            .get(&section)
            .map_or((0, 0), |e| e.block_range)
    }

    pub fn all_sections(&self) -> impl Iterator<Item = SectionId> + '_ {
        self.sections.keys().copied()
    }
}
