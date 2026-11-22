//! Reads a conversation's recorded history back from the substrate, as the
//! bubbles the conversation view and the projection replay show.
//!
//! Turns are stored verbatim, so history is rebuilt from each turn's user and
//! assistant text with no re-tokenising or decoding. Summary turns are skipped
//! unless a caller asks for them, because they exist for provenance and
//! projection rather than for display. Labels, metadata and projection-event
//! timelines come from the same substrate, so a reloaded conversation shows what
//! the session held.

use super::*;

impl Sequence {
    /// Recover the conversation in `timeline` as bubbles — one per non-empty
    /// half of each turn, in order: the user's message (exactly what
    /// `submit_turn` received), then the assistant's reply (the decoded body).
    /// Both texts come straight off the turn's `user_text` and `assistant_text`
    /// — no re-tokenising, no marker scanning, no decoding. See
    /// [`RecoveredMessage`] for what else each bubble carries.
    ///
    /// When `include_ghost_summaries` is false, the ghost summary turns the
    /// summariser appends to the timeline (`SummaryOfTurns` /
    /// `SummaryOfSummaries` tree nodes) are skipped — they exist for
    /// provenance/projection, not for the conversation view. Pass `true` for
    /// substrate-level views that legitimately surface them.
    pub fn recovered_history(
        &self,
        timeline: TimelineId,
        include_ghost_summaries: bool,
    ) -> Vec<RecoveredMessage> {
        self.recovered_history_from(timeline, include_ghost_summaries, 0)
    }

    /// [`Self::recovered_history`] from turn `from_turn` on: the bubbles of
    /// the turns before it are neither read nor built.
    pub fn recovered_history_from(
        &self,
        timeline: TimelineId,
        include_ghost_summaries: bool,
        from_turn: u32,
    ) -> Vec<RecoveredMessage> {
        let read = self.substrate.read();
        // Only a turn whose reasoning K/V was dropped needs this: its length is
        // then an estimate from the prose it kept.
        let estimate = |text: &str| {
            self.tokenizer
                .encode(text, false)
                .map(|e| e.len() as u32)
                .unwrap_or(0)
        };
        // The ids of the marker that opens each tool result, for measuring a
        // tool-response turn's blocks in its own sealed ids.
        let response_open: Vec<u32> = self
            .tokenizer
            .encode(TOOL_RESPONSE_OPEN, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        recovered_from(
            &read,
            timeline,
            include_ghost_summaries,
            from_turn,
            estimate,
            &response_open,
        )
    }

    /// Every turn of `timeline` as the substrate holds it — its sealed ids and
    /// its index pages — in turn order, the summariser's summary turns skipped
    /// as [`Self::recovered_history`] skips them. What a projection hands the
    /// model when it borrows each turn, so two timelines that agree here are the
    /// same context.
    pub fn sealed_turns(&self, timeline: TimelineId) -> Vec<SealedTurn> {
        // The turns and their ids under the read lock, the pages after it: a
        // page left in the log is read through the persistence path, which
        // takes its own locks.
        let turns: Vec<(TurnIndex, Vec<u32>)> = {
            let read = self.substrate.read();
            read.turn_indices(timeline)
                .filter(|&idx| {
                    !read
                        .tree_meta_of(timeline, idx)
                        .is_some_and(|m| m.kind.is_summary())
                })
                .map(|idx| (idx, read.token_ids_of(timeline, idx)))
                .collect()
        };
        turns
            .into_iter()
            .map(|(idx, token_ids)| {
                let page = self
                    .substrate
                    .turn_index_page(timeline, idx)
                    .unwrap_or_else(|e| {
                        tracing::warn!("sealed turn {timeline:?}/{}: {e}", idx.0);
                        None
                    });
                SealedTurn {
                    index: idx.0,
                    token_ids,
                    pages: SealedPages::of(page.as_deref().map(StoredPages::bytes)),
                }
            })
            .collect()
    }

    /// How many tokens `text` becomes as a turn's user half — encoded exactly
    /// as a submitted turn is, markup and literal pieces each by their own
    /// tokenizer. A tool result measured this way before it is submitted has
    /// the length its block shows once the turn is sealed
    /// ([`RecoveredMessage::tool_tokens`]).
    pub fn encoded_len(&self, text: &TurnText) -> usize {
        self.encode_text(text).map_or(0, |ids| ids.len())
    }

    /// The two verbatim halves — `(user_text, assistant_text)` — of turn `index`
    /// in `timeline`, read from the substrate. Returned UNFRAMED (no dialect
    /// markers): the caller places the glue around and between them. Both halves
    /// are stored verbatim (decoded for dialogue turns, supplied for prefill
    /// ingests like repo_map/code_reading; see `insert_turn` → the seal).
    ///
    /// `timeline` is the turn's resolved identity, stamped by the projection at
    /// selection time (`SelectedTurn::timeline`) — it is NEVER re-derived from a
    /// group here, because the shared substrate registers many conversations under
    /// one group and that re-derivation is non-deterministic. `None` if the turn
    /// isn't found or is entirely empty.
    pub fn resolve_turn_text(&self, timeline: TimelineId, index: u32) -> Option<(String, String)> {
        let read = self.substrate.read();
        let idx = TurnIndex(index);
        let user = read.user_text_of(timeline, idx);
        let assistant = read.assistant_text_of(timeline, idx);
        if user.is_empty() && assistant.is_empty() {
            return None;
        }
        Some((user, assistant))
    }

    /// The ENTIRE turn `index` in `timeline` as one continuous string: the full
    /// sealed token range — user content, the baked intra-turn boundary
    /// (`user_end` / `assistant_start`), and assistant content — decoded verbatim,
    /// exactly as it sits in the KV. `timeline` is the turn's resolved identity
    /// (`SelectedTurn::timeline`); see [`Self::resolve_turn_text`]. `None` if the
    /// turn isn't found or has no tokens.
    pub fn resolve_turn_full_text(&self, timeline: TimelineId, index: u32) -> Option<String> {
        let read = self.substrate.read();
        let ids = read.token_ids_of(timeline, TurnIndex(index));
        if ids.is_empty() {
            return None;
        }
        self.tokenizer.decode(&ids, false).ok()
    }

    /// The segment-vector [`TurnLayout`] for turn `index` in `timeline` — the
    /// complete, validated description of its K/V (user / thinking / assistant /
    /// boundary glue). Built at seal time and stored on the turn, so this is a
    /// direct fetch. `timeline` is the turn's resolved identity
    /// (`SelectedTurn::timeline`); see [`Self::resolve_turn_text`]. `None` if the
    /// turn isn't found.
    pub fn turn_layout(&self, timeline: TimelineId, index: u32) -> Option<TurnLayout> {
        let read = self.substrate.read();
        read.turn_layout(timeline, TurnIndex(index))
    }

    /// Couple `from_turn` — the sealed **call turn** — to the tool response about
    /// to be submitted as the next turn, the two halves of one tool round-trip.
    ///
    /// Call this **after** the tools have returned real output and **before**
    /// submitting the response turn. In that window the round-trip is certain:
    /// the call turn is sealed (so its index is final) and the response turn is
    /// guaranteed to follow. Writing it here — rather than at either turn's seal
    /// — is what makes the record authoritative (a call that is never executed,
    /// or yields nothing, simply never couples).
    ///
    /// `from_turn` must be the call turn's **own** index, captured from its seal
    /// (`TurnResponse::seal`). Do not infer it from the current turn count: the
    /// async summariser can seal a `SummaryOfTurns`/`SummaryOfSummaries` turn
    /// into the same index space between the call turn's seal and this call, so
    /// "the last turn" may be a summary node, not the call — and coupling that
    /// index is silently dropped when the forest is derived (it maps to no
    /// `Normal`), leaving the exchange unformed.
    pub fn couple_turn(&self, from_turn: u32) -> crate::Result<()> {
        self.substrate
            .couple_turn(self.target.timeline, from_turn)
            .map_err(ConversationError::Model)
    }

    /// Persist this conversation's projection-event timeline for the most
    /// recently sealed turn. Called by the caller after `finish_turn` with the
    /// events streamed during (and at the end of) that turn's decode; survives
    /// a daemon restart via the `ProjectionEvents` redo-log record.
    pub fn persist_projection_events(&self, events: &[ProjectionEvent]) -> crate::Result<()> {
        let timeline = self.target.timeline;
        let count = self.substrate.read().turn_count(timeline);
        if count == 0 {
            return Ok(());
        }
        let stream_id = turn_stream_id(timeline.raw(), count - 1);
        let payload = crate::projection::encode_events(events);
        self.substrate.persist_projection_events(stream_id, payload);
        Ok(())
    }

    /// Recovered projection-event timelines for a conversation, one `Vec` per
    /// turn in turn order (empty for turns that have none). Mirrors
    /// [`recovered_history`](Self::recovered_history); backs the GUI timeline
    /// replay after reload.
    /// [`Self::recovered_projection_events`], light: each turn's points as the
    /// conversation's timeline draws them, without the selection or spine the
    /// projection panel reads — what loading a conversation's history needs, at
    /// a fraction of the decode.
    pub fn recovered_projection_points(&self, timeline: TimelineId) -> Vec<Vec<ProjectionPoint>> {
        let read = self.substrate.read();
        let indices: Vec<TurnIndex> = read.turn_indices(timeline).collect();
        indices
            .into_iter()
            .filter(|&idx| {
                !read
                    .tree_meta_of(timeline, idx)
                    .is_some_and(|m| m.kind.is_summary())
                    && read.has_assistant_text(timeline, idx)
            })
            .map(|idx| {
                read.projection_events_blob(timeline, idx)
                    .ok()
                    .flatten()
                    .map(|b| decode_points(&b))
                    .unwrap_or_default()
            })
            .collect()
    }

    pub fn recovered_projection_events(&self, timeline: TimelineId) -> Vec<Vec<ProjectionEvent>> {
        let read = self.substrate.read();
        let indices: Vec<TurnIndex> = read.turn_indices(timeline).collect();
        indices
            .into_iter()
            // One bucket per assistant bubble — filter to turns with a non-empty
            // assistant reply, and skip ghost summary turns, so this lines up
            // index-for-index with the assistant entries the conversation view
            // (`recovered_history(.., false)`) produces.
            .filter(|&idx| {
                !read
                    .tree_meta_of(timeline, idx)
                    .is_some_and(|m| m.kind.is_summary())
                    && read.has_assistant_text(timeline, idx)
            })
            .map(|idx| {
                read.projection_events_blob(timeline, idx)
                    .ok()
                    .flatten()
                    .map(|b| decode_events(&b))
                    .unwrap_or_default()
            })
            .collect()
    }

    /// The substrate-backed sidebar label for `timeline`, or `None`.
    pub fn conversation_label_of(&self, timeline: TimelineId) -> Option<String> {
        self.substrate.label_of(timeline)
    }

    /// This sequence's timeline id.
    pub fn timeline_id(&self) -> TimelineId {
        self.target.timeline
    }

    /// Set this conversation's sidebar label, persisting it to the redo
    /// log. First-write-wins.
    pub fn set_conversation_label(&self, label: &str) -> crate::Result<()> {
        self.substrate
            .set_conversation_label(self.target.timeline, label)
            .map_err(ConversationError::Model)
    }

    /// Merge a `(key, value)` into this conversation's free-form `custom`
    /// metadata bag and persist it (see [`ConvMeta::custom`]). Utility
    /// ingests tag conversations with a content hash + descriptive fields
    /// for the restart-resume cache.
    pub fn set_metadata(&self, key: &str, value: &str) -> crate::Result<()> {
        self.substrate
            .set_conversation_metadata(self.target.timeline, key, value)
            .map_err(ConversationError::Model)
    }

    /// Merge several `(key, value)` pairs into this conversation's
    /// `custom` metadata in one persisted record.
    pub fn set_metadata_many(
        &self,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> crate::Result<()> {
        self.substrate
            .set_conversation_metadata_many(self.target.timeline, kv)
            .map_err(ConversationError::Model)
    }

    /// This conversation's `custom` metadata bag, or `None` if the
    /// timeline isn't registered yet.
    pub fn metadata(&self) -> Option<std::collections::BTreeMap<String, String>> {
        self.substrate.conversation_metadata(self.target.timeline)
    }

    /// Every timeline (across the whole substrate) whose `custom`
    /// metadata contains `key == value`. Used by utility ingests to skip
    /// re-building units already present after substrate load.
    pub fn find_conversations_by_metadata_including_distilled(
        &self,
        key: &str,
        value: &str,
    ) -> Vec<TimelineId> {
        self.substrate
            .find_timelines_by_metadata_including_distilled(key, value)
    }

    pub fn find_conversations_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.substrate.find_timelines_by_metadata(key, value)
    }
}
