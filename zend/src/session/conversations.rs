//! Lists, archives and tombstones the conversations a session holds, as the
//! sidebar sees them.
//!
//! The sidebar is built from the substrate's own label and lifecycle records,
//! so the list survives restarts without a separate index. Archiving is
//! one-way: the conversation is distilled to text, its KV is dropped and no
//! tool runs on it again, while a worker's journal text is kept because its
//! turns are the evidence for its outcome. Tombstoning hides a timeline at
//! once and leaves compaction to reclaim its records. Listing is gated on the
//! redo log still existing, so a wiped substrate shows no ghost conversations.

use super::*;

/// Sidebar entry surfaced by `GET /v1/conversations`. Built directly from
/// the substrate — `RecordType::Label` records carry the `conv_id`
/// string and the human label; `RecordType::ConvState` carries the
/// `archived` lifecycle flag. Together they're the whole sidebar
/// contract.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConvEntry {
    /// Client-supplied `conv_id` string — stable across daemon restarts,
    /// used as the sidebar's `id` and echoed back in chat requests.
    pub id: String,
    /// Human-readable title. Empty during the brief window between
    /// first-submit and titler-completion.
    pub label: String,
    /// Number of recovered turns for this conversation, from the
    /// substrate. Advisory display field.
    pub turn_count: u32,
    /// Whether the user has archived (closed) this conversation. The
    /// sidebar hides archived entries by default; the "show archived"
    /// checkbox toggles them back in via `?include_archived=true`.
    pub archived: bool,
    /// The entry's place in the listing, most recently used first — see
    /// [`crate::conv_order`]. The top entry holds the highest value. A rank,
    /// NOT a millisecond clock: the substrate keeps no time. The wire name is
    /// kept for the frontend's existing sort key.
    pub updated_ms: u64,
}

/// Release the images the deleted `timeline` used, retiring each no other
/// timeline reads (`vision::tenancy`). A failure leaves an image kept, never a
/// timeline undeleted.
fn release_images(state: &InferenceState, timeline: projection::TimelineId) {
    let substrate = state.engine.conversation().shared();
    match release_timeline(&substrate, timeline) {
        Ok(retired) if !retired.is_empty() => {
            tracing::info!(
                timeline = timeline.raw(),
                retired = retired.len(),
                "its images retired"
            );
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(
            timeline = timeline.raw(),
            "releasing its images failed: {e}"
        ),
    }
}

impl ZendSession {
    /// Name `starter` as the conversation `conv_id`, a task not yet begun,
    /// begins on: its branches and its changes ([`conv_inherit`]). `None`
    /// while the model is not loaded.
    pub fn begin_on(&self, conv_id: &str, starter: &str) -> Option<EngineResult<()>> {
        let state = self.loaded()?;
        Some(conv_inherit::mark(
            &state.engine,
            conversation_timeline(conv_id),
            starter,
        ))
    }

    /// Record `steps` as what the conversation `conv_id`'s answers account
    /// for ([`answer_steps`]). `None` while the model is not loaded.
    pub fn answer_steps(&self, conv_id: &str, steps: &[String]) -> Option<EngineResult<()>> {
        let state = self.loaded()?;
        Some(answer_steps::mark(
            &state.engine,
            conversation_timeline(conv_id),
            steps,
        ))
    }

    pub fn list_conversations(&self, include_archived: bool) -> Vec<ConvEntry> {
        // On-disk gate: if the workspace's redo log is gone (no segment files
        // in `substrate/`), return empty regardless of the in-RAM cache. The
        // daemon keeps running and any new turn re-mints the segment set.
        let sub_dir = self.config.workspace.root().join(SUBSTRATE_DIR);
        let has_segments = std::fs::read_dir(&sub_dir)
            .ok()
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("seg-"))
                })
            })
            .unwrap_or(false);
        if !has_segments {
            return Vec::new();
        }

        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        // Polled every few seconds by every open tab, so it reads what each
        // listed conversation holds and nothing more: a turn
        // count is its own timeline's, never a walk of every turn the substrate
        // holds (the ingest layers are most of them).
        let engine = &state.engine;
        let known = engine.known_conversations();
        let passthrough_tagged = engine.conversations_with_metadata_key(passthrough::METADATA_KEY);
        let conv = engine.conversation();
        let view = conv.read();
        let active_of = |tl| view.timeline_entry(tl).map_or(0, |e| e.active);
        let mut entries: Vec<(ConvEntry, conv_order::Ranks)> = known
            .into_iter()
            .filter(|(_, _, _, archived, _)| include_archived || !*archived)
            .map(|(tl, conv_id, label, archived, order)| {
                let entry = ConvEntry {
                    id: conv_id,
                    label,
                    turn_count: view.turn_count_of(tl),
                    archived,
                    updated_ms: 0,
                };
                (entry, (active_of(tl), order))
            })
            .collect();
        // Passthrough conversations stored before they carried a `conv_id` are
        // found by their tag; the next call to one names it for good.
        entries.extend(passthrough_tagged.into_iter().filter_map(|(tl, _)| {
            let entry = view.timeline_entry(tl)?;
            if entry.conv_id.is_some() || (entry.archived && !include_archived) {
                return None;
            }
            let first_user = view
                .turn_indices(tl)
                .min()
                .map(|i| view.user_text_of(tl, i))
                .unwrap_or_default();
            Some((
                ConvEntry {
                    id: passthrough::conv_id_of(tl),
                    label: passthrough::label_for(&first_user),
                    turn_count: view.turn_count_of(tl),
                    archived: entry.archived,
                    updated_ms: 0,
                },
                (entry.active, entry.order),
            ))
        }));
        conv_order::by_last_use(entries)
    }

    /// Archive a conversation — one-way. Sets the archived lifecycle flag and
    /// marks the timeline for `TextOnly` distillation, so the next compaction
    /// sheds its KV chunks + signatures + projections and leaves a read-only
    /// text record. There is no unarchive: distillation is lossy (the KV is
    /// gone), so an archived conversation can't be resumed. Its journal keeps
    /// its entries, and the text behind them is ceased: no tool runs on an
    /// archived conversation again. Returns `None` when the model isn't loaded
    /// yet.
    /// Whether this daemon holds the conversation `conv_id`: registered, and not
    /// deleted. Archiving or deleting one it does not hold changes nothing, so
    /// the routes answer it as not found rather than as done. `None` while the
    /// model is not loaded.
    pub fn holds_conversation(&self, conv_id: &str) -> Option<bool> {
        let state = self.loaded()?;
        let timeline = conversation_timeline(conv_id);
        let engine = &state.engine;
        Some(
            engine.timeline_group_and_order(timeline).is_some()
                && !engine.is_timeline_tombstoned(timeline),
        )
    }

    pub fn archive_conversation(&self, conv_id: &str) -> Option<EngineResult<()>> {
        let (state, timeline, distilled) = self.distill_archived(conv_id)?;
        let engine = &state.engine;
        Some(distilled.and_then(|()| cease_journal_text(engine, timeline)))
    }

    /// Archive a worker `/mcp` started (`crate::mcp::workers`): distilled like
    /// any archived conversation, but its journal's text is **kept**. The
    /// turns behind a worker's entries are the evidence for its task's outcome
    /// — exactly what Claude reads later, through `journal_read` with an entry
    /// — so they stay until the worker's thread is done with
    /// (`docs/claude_zend_gateway.md` §4). `None` when the model isn't loaded.
    pub fn archive_worker(&self, conv_id: &str) -> Option<EngineResult<()>> {
        self.distill_archived(conv_id)
            .map(|(_, _, distilled)| distilled)
    }

    /// Record that `conv_id` decodes at `band` in place of its layer's
    /// priority: read when its slot is made, so it is set before the turn
    /// that makes it (`docs/claude_zend_gateway.md` §3.4). `None` when the
    /// model isn't loaded.
    pub fn set_decode_band(&self, conv_id: &str, band: DecodePriority) -> Option<EngineResult<()>> {
        let state = self.loaded()?;
        let engine = &state.engine;
        Some(engine.set_conversation_metadata(
            conversation_timeline(conv_id),
            DecodePriority::METADATA_KEY,
            band.name(),
        ))
    }

    /// What every archive does: set the archived flag and mark the timeline
    /// for `TextOnly` distillation. `None` when the model isn't loaded.
    pub(super) fn distill_archived(
        &self,
        conv_id: &str,
    ) -> Option<(Arc<InferenceState>, TimelineId, EngineResult<()>)> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let engine = &state.engine;
        let archived = engine.set_conversation_archived(timeline, true);
        // Archived, it runs no more rounds, whether or not it distils.
        if archived.is_ok() {
            close_tool_sessions(&state, conv_id, "archived");
        }
        let distilled =
            archived.and_then(|()| engine.distill_timeline(timeline, DistillMode::TextOnly));
        Some((state, timeline, distilled))
    }

    /// Permanently tombstone a conversation — the delete path behind
    /// `DELETE /v1/conversations/{id}`. Writes a durable `Tombstone` record so the
    /// timeline is hidden from listings/resume immediately, and its records
    /// (turns, KV, provenance) are physically reclaimed at the next compaction
    /// (lazy — the tombstone is the commit, compaction does the delete). Also
    /// drops the conversation from the in-RAM map so it vanishes from the sidebar
    /// at once, and retires every image no other conversation reads. Returns
    /// `None` when the model isn't loaded yet.
    pub fn tombstone_conversation(&self, conv_id: &str) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let result = {
            let engine = &state.engine;
            engine
                .tombstone_timeline(timeline)
                .and_then(|()| cease_journal_text(engine, timeline))
        };
        if result.is_ok() {
            close_tool_sessions(&state, conv_id, "deleted");
            state.conversations.remove(conv_id);
            state.journals.forget(conv_id);
            release_images(&state, timeline);
            // After the tombstone, so a turn woken by it finds its timeline
            // tombstoned.
            state.deletions.deleted();
        }
        Some(result)
    }

    /// Tombstone a timeline by its RAW id — the derived-layer counterpart to
    /// [`Self::tombstone_conversation`], which can only address timelines whose
    /// id is `hash(conv_id)`.
    ///
    /// Backs `DELETE /v1/substrate/zend/timeline/{tl}`. Nothing is removed from the
    /// dialogue conversation map, because a derived-layer timeline was never in
    /// it: these are `repo_map` / `code_reading` ingests, addressed only by the
    /// id the substrate views print. The images a turn on it read are released
    /// as the conversation delete releases them.
    ///
    /// `None` when the model is not loaded or `raw` is not a valid timeline id
    /// (zero) — the same shape the sibling routes return.
    pub fn tombstone_timeline_raw(&self, raw: u64) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.load_full()?;
        let timeline = projection::TimelineId::from_raw(raw)?;
        let result = state.engine.tombstone_timeline(timeline);
        if result.is_ok() {
            release_images(&state, timeline);
        }
        Some(result)
    }

    /// The conversation's sidebar title — the titler's label, or the one an
    /// upload gave it. `None` when the model isn't loaded or nothing has
    /// labelled the conversation yet.
    pub fn conversation_label(&self, conv_id: &str) -> Option<String> {
        let state = self.inference.load_full()?;
        let timeline = passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id));
        let label = state.engine.conversation_label_of(timeline)?;
        (!label.is_empty()).then_some(label)
    }

    /// How many turns the conversation holds, read from the substrate's
    /// snapshot — `0` when the model isn't loaded or the conversation is
    /// unknown. What a job's view and the conversation's history both answer,
    /// so a page that fetched one knows when the other has moved.
    pub fn conversation_turn_count(&self, conv_id: &str) -> u32 {
        let Some(state) = self.inference.load_full() else {
            return 0;
        };
        let timeline = passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id));
        state.engine.conversation().read().turn_count_of(timeline)
    }

    /// The branch the conversation works on in each repository, by repository
    /// name — see [`conv_branches`]. Empty when the model isn't loaded or the
    /// conversation is unknown.
    pub fn conversation_branches(&self, conv_id: &str) -> BTreeMap<String, String> {
        let Some(state) = self.inference.load_full() else {
            return BTreeMap::new();
        };
        let timeline = passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id));
        let state = state.engine.conversation_state(timeline);
        state.map(|s| s.branches).unwrap_or_default()
    }

    /// [`Self::conversation_projections`] as the history carries it: each
    /// point light, without the selection and spine only the projection panel
    /// reads, under a fresh display id. One bucket per assistant turn, in order.
    pub fn conversation_spans(&self, conv_id: &str) -> Vec<Vec<ProjectionSpanOut>> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = &state.base_conv;
            base.recovered_projection_points(timeline)
        };
        raw.into_iter()
            .enumerate()
            .map(|(turn, points)| {
                points
                    .into_iter()
                    .enumerate()
                    .map(|(event, point)| {
                        let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ProjectionSpanOut::of(seq, point, turn, event)
                    })
                    .collect()
            })
            .collect()
    }

    /// Projection-event buckets banked for a conversation this daemon session,
    /// one per assistant turn (in order). Backs the timeline-dot replay on a
    /// browser reload. Empty when the conversation isn't resident (e.g. it was
    /// recovered from disk but hasn't been chatted with since startup).
    pub fn conversation_projections(&self, conv_id: &str) -> Vec<Vec<ProjectionEventOut>> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = &state.base_conv;
            base.recovered_projection_events(timeline)
        };
        // Re-attach the display id/region/step the wire shape needs (these are
        // GUI concerns, not persisted); one bucket per assistant turn, in order.
        raw.into_iter()
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| {
                        let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ProjectionEventOut::answer(seq, event)
                    })
                    .collect()
            })
            .collect()
    }
}
