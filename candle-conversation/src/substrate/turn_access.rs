//! Reads a turn's content, layout and K/V references for the projection, and sets
//! its conversation label and metadata.
//!
//! A turn's layout is built at seal time and read in place, so a predicate or a
//! span does not copy the whole turn. Its reasoning is windowed out of the sealed
//! K/V and the index pages together, located by position, so the two halves agree
//! by construction. A span that misses the page boundaries is refused rather than
//! injected whole, because a turn that silently keeps its reasoning is a defect. A
//! model that seals no index pages has nothing to window, and its turns are
//! injected whole.

use super::*;

impl Substrate {
    /// Sum of bytes across every hot-resident turn, from each residence's
    /// cached `byte_size`. O(N_hot) integer sum, no chunk-walking.
    pub fn hot_turn_bytes(&self) -> usize {
        self.hot_turns_oldest_first()
            .into_iter()
            .map(|idx| self.residences.get(idx).byte_size() as usize)
            .sum()
    }

    /// Sum of bytes across every hot section — the system-prompt / catalog
    /// KV. Counts toward the same VRAM budget hot-tier eviction uses.
    pub fn section_bytes(&self) -> usize {
        self.sections
            .values()
            .map(|e| self.residences.get(e.residence))
            .filter(|r| r.has_hot())
            .map(|r| r.byte_size() as usize)
            .sum()
    }

    /// Bytes a single hot-resident turn currently holds in VRAM.
    /// `None` for unknown turns or turns with no hot bytes.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        let entry = self.turn(timeline, index)?;
        let r = self.residences.get(entry.content.residence);
        r.has_hot().then(|| r.byte_size() as usize)
    }

    /// The user's message text for this turn — exactly as
    /// `submit_turn` received it.
    pub fn user_text_of(&self, timeline: TimelineId, index: TurnIndex) -> String {
        self.turn(timeline, index)
            .map(|e| e.content.layout.user_text().to_string())
            .unwrap_or_default()
    }

    /// `f` of this turn's user text, read in place — for a check over many
    /// turns whose user halves are long (a tool round's results), which
    /// [`Self::user_text_of`] would copy each time. `None` for a turn the
    /// substrate does not hold.
    pub fn with_user_text<R>(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        f: impl FnOnce(&str) -> R,
    ) -> Option<R> {
        self.turn(timeline, index)
            .map(|e| f(e.content.layout.user_text()))
    }

    /// The assistant's decoded reply text for this turn — the full message
    /// (reasoning block + answer), reconstructed by [`TurnLayout::assistant_text`].
    pub fn assistant_text_of(&self, timeline: TimelineId, index: TurnIndex) -> String {
        self.turn(timeline, index)
            .map(|e| e.content.layout.assistant_text().unwrap_or_default())
            .unwrap_or_default()
    }

    /// A turn's reasoning length ([`TurnLayout::thinking_length`]), read from
    /// its layout in place.
    pub fn thinking_length_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        estimate: impl Fn(&str) -> u32,
    ) -> Option<ThinkingLength> {
        self.turn(timeline, index)?
            .content
            .layout
            .thinking_length(estimate)
    }

    /// The ids of a turn's user body, where its layout places them in the
    /// turn's grid — borrowed, not copied.
    pub fn user_body_ids(&self, timeline: TimelineId, index: TurnIndex) -> Option<&[u32]> {
        let e = self.turn(timeline, index)?;
        let span = e.content.layout.user_span();
        e.content
            .token_ids
            .get(span.offset as usize..span.end() as usize)
    }

    /// Whether [`Self::assistant_text_of`] would be non-empty, without
    /// building the text.
    pub fn has_assistant_text(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.turn(timeline, index)
            .is_some_and(|e| e.content.layout.has_assistant_text())
    }

    /// A turn's user half, read in place.
    pub fn user_half(&self, timeline: TimelineId, index: TurnIndex) -> Option<UserHalf<'_>> {
        let e = self.turn(timeline, index)?;
        let span = e.content.layout.user_span();
        let ids = e
            .content
            .token_ids
            .get(span.offset as usize..span.end() as usize)?;
        Some(UserHalf {
            offset: span.offset as usize,
            ids,
            text: e.content.layout.user_text(),
        })
    }

    /// Turn token IDs as an owned `Vec` (clones the buffer).
    pub fn token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> Vec<u32> {
        self.turn(timeline, index)
            .map(|e| e.content.token_ids[..].to_vec())
            .unwrap_or_default()
    }

    /// The turn's [`TurnLayout`] — its segment-vector description: the complete,
    /// validated description of its K/V (user / thinking / assistant / boundary
    /// glue). Built at seal time and stored on the turn, so this is a direct
    /// clone with no re-derivation. `None` if the turn isn't found.
    pub fn turn_layout(&self, timeline: TimelineId, index: TurnIndex) -> Option<TurnLayout> {
        self.turn(timeline, index).map(|e| e.content.layout.clone())
    }

    /// Whether the turn's grid carries its own boundary markers — the
    /// projection assembler's per-turn question, answered without cloning the
    /// layout (which [`Self::turn_layout`] does, and which is far too much for
    /// a predicate consulted once per turn per assembly).
    ///
    /// `false` for a turn with no recorded layout: that is the oldest shape of
    /// turn there is, and it certainly did not bake its own boundaries.
    pub fn turn_bakes_own_boundaries(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.turn(timeline, index)
            .is_some_and(|e| e.content.layout.bakes_own_boundaries())
    }

    /// The turn's slot block extent `(block_start, block_end)` — needed by the
    /// ordered-merge splice ([`crate::projection::resolver::Conversation::adopt_turn`])
    /// to re-record a forked scope turn onto the file timeline preserving its
    /// range verbatim. `None` if the turn isn't found.
    pub fn turn_block_range(&self, timeline: TimelineId, index: TurnIndex) -> Option<(u64, u64)> {
        self.turn(timeline, index).map(|e| e.block_range)
    }

    /// Turn token IDs as a borrowed slice (zero-copy).
    pub fn assistant_token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> &[u32] {
        self.turn(timeline, index)
            .map_or(&[][..], |e| &e.content.token_ids[..])
    }

    pub fn set_block_range(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        block_start: u64,
        block_end: u64,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.block_range = (block_start, block_end);
        }
    }

    pub fn extend_turn(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        additional_tokens: usize,
        new_block_end: u64,
    ) {
        let Some(entry) = self.turn_mut(timeline, index) else {
            return;
        };
        entry.content.token_count = entry.content.token_count.saturating_add(additional_tokens);
        entry.block_range.1 = new_block_end;
        // `entry`'s borrow ends here, so the counter bump can re-borrow self.
        *self.timeline_token_totals.entry(timeline).or_default() += additional_tokens;
    }

    pub fn block_range_of(&self, timeline: TimelineId, index: TurnIndex) -> (u64, u64) {
        self.turn(timeline, index).map_or((0, 0), |e| e.block_range)
    }

    /// Sealed K/V for the turn — Arc-cloned per-layer
    /// `SealedSequence` snapshot of the content residence's hot
    /// tier.  Used by the projection assembler when injecting a
    /// `SealedKind::Turn` segment onto a slot.
    pub fn turn_sealed_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        let residence = self.turn(timeline, index)?.content.residence;
        let hot = self.residences.get(residence).hot()?;
        Some(Arc::new(hot.to_vec()))
    }

    /// The turn's sealed K/V and index pages with its `<think>…</think>` span
    /// windowed out — what a projection injects for every turn but the most
    /// recent.
    ///
    /// The K/V is the turn's own chunks either side of the reasoning, as two
    /// zero-copy [`window_sealed_tokens`] views concatenated per layer. The
    /// boundary chunks are windowed by `offset` / `token_count`, so a span that
    /// begins and ends inside one physical chunk yields two views of that same
    /// chunk sharing one refcounted `HeadGids` — no copy, and no alignment
    /// requirement on where the reasoning falls.
    ///
    /// Positions **compact**: K is stored un-rotated and RoPE is applied at read
    /// time from cumulative usage of the layout as injected, so omitting a span
    /// leaves no positional gap. The answer lands exactly where a reasoning-free
    /// prefill would have put it.
    ///
    /// **Both halves are returned by one call and cannot be requested
    /// separately.** The index pages are filtered to those outside the span in
    /// the same breath, because a span injected without its rows is silent below
    /// the QSA identity threshold and a hard refusal above it. They agree by
    /// construction rather than by two computations happening to match: the page
    /// boundaries and the `KvSpan` are the same grid positions, recorded once
    /// during decode.
    ///
    /// A turn with no real `Thinking` segment — a prefilled assistant half, or
    /// one sealed before this build — returns its full sealing and its full page
    /// list, unchanged.
    ///
    /// `pages` is the turn's index-page payload as the caller resolved it —
    /// resident first, then the stored record — because the projection knows
    /// which copy it holds and the substrate does not.
    /// `Ok(None)` when the turn is not held here — the caller's existing
    /// "skipped turn" path. `Err` when the turn IS held but its reasoning cannot
    /// be windowed, which is a defect rather than a condition to tolerate.
    pub fn turn_sealed_without_thinking(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        pages: Option<Arc<StoredPages>>,
    ) -> Result<Option<(Arc<Vec<SealedSequence>>, Option<Arc<StoredPages>>)>, ConversationError>
    {
        let (Some(full), Some(turn)) = (
            self.turn_sealed_of(timeline, index),
            self.turn(timeline, index),
        ) else {
            return Ok(None);
        };
        let layout = &turn.content.layout;
        // No reasoning recorded: the turn is whole because there is nothing to
        // take out, which is a different thing from failing to take it out.
        let Some(span) = layout.segments.iter().find_map(|s| match s {
            TurnSegment::Thinking { kv, .. } => *kv,
            _ => None,
        }) else {
            return Ok(Some((full, pages)));
        };
        // **Both halves or neither.** The rows are filtered to the pages outside
        // the span, located by POSITION — the page count is not stable, because
        // a mid-decode reprojection closes an extra one, so an ordinal would
        // window only the turns that never reprojected and silently leave the
        // rest whole.
        //
        // **A span that does not fall on page boundaries is a FAILURE, not a
        // fallback.** The cuts exist to guarantee this alignment; if it does not
        // hold, something upstream did not cut where it said it did.
        //
        // This used to hand the turn over WHOLE — "visibly imperfect rather than
        // quietly inconsistent". It is neither: the turn injects, reads
        // perfectly, and silently keeps showing reasoning the projection was
        // asked to remove, forever. The one case it fired on in production was a
        // live turn, not the legacy data it was written for, so it was masking a
        // real defect rather than tolerating an old one. Refusing puts the
        // failure where it can be fixed.
        // **No pages at all is not a misalignment.** A model whose
        // `carries_positional_state()` is false seals none — the seal only warns
        // about their absence for models that do carry it — so there is nothing
        // to window and nothing that can disagree with the span. Refusing here
        // failed the projection of every turn after the first on Qwen3-8B,
        // Qwen3-30B-A3B and Llama-3.2-3B, which is the whole of what those
        // models do. The turn goes over whole, matching what the assembler
        // already does for a turn with no index page.
        let Some(stored) = pages.as_deref() else {
            return Ok(Some((full, pages)));
        };
        let filter_span = profile::span("window:filter_pages");
        let filtered = stored.without_span(span.range());
        filter_span.end();
        let Some(filtered) = filtered else {
            // The boundaries, not just the fact that the span missed them. The
            // question a refusal always raises is WHICH cut was missing, and
            // that is read straight off the two positions the span sits between.
            let bounds = index_pages::boundaries(stored.bytes())
                .map(|b| format!("{b:?}"))
                .unwrap_or_else(|| "absent".to_string());
            return Err(ConversationError::Channel(format!(
                "turn {}#{} has a reasoning span [{}..{}) that does not fall on its index \
                 page boundaries {bounds} — the K/V can be windowed but the rows cannot, \
                 and injecting one beside the other is the divergence the pairing exists to \
                 prevent. The cuts that make a turn's reasoning occupy whole pages did not \
                 hold for this turn.",
                timeline.raw(),
                index.0,
                span.offset,
                span.end(),
            )));
        };

        let _g = profile::span("window:sealed_tokens");
        let total = full.first().map_or(0, |s| s.token_count);
        let head = window_sealed_tokens(&full, 0, span.offset as usize);
        let tail = window_sealed_tokens(&full, span.end() as usize, total);
        let windowed: Vec<SealedSequence> = head
            .into_iter()
            .zip(tail)
            .map(|(h, t)| SealedSequence {
                token_count: h.token_count + t.token_count,
                chunks: h.chunks.into_iter().chain(t.chunks).collect(),
                chunk_size: h.chunk_size,
                location: h.location,
            })
            .collect();
        Ok(Some((Arc::new(windowed), Some(Arc::new(filtered)))))
    }

    /// Token ids of the turn's *assistant-response body* `[asst_start, total)`,
    /// which the compression path prefills as text.
    pub fn turn_assistant_token_ids(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Vec<u32>> {
        let toks = self.token_ids_of(timeline, index);
        let layout = &self.turn(timeline, index)?.content.layout;
        let total = toks.len();
        let start = (layout.assistant_content_start() as usize).min(total);
        Some(toks[start..total].to_vec())
    }

    /// Token ids of the turn's *user-input body* `[user_start, user_end)`. The
    /// compression path text-prefills this alongside the assistant half so each
    /// half-summary is grounded in the full exchange rather than confabulating
    /// from one half in isolation (see [`Self::turn_assistant_token_ids`]).
    pub fn turn_user_token_ids(&self, timeline: TimelineId, index: TurnIndex) -> Option<Vec<u32>> {
        let toks = self.token_ids_of(timeline, index);
        let layout = &self.turn(timeline, index)?.content.layout;
        let total = toks.len();
        let start = (layout.user_content_start() as usize).min(total);
        let end = (layout.user_content_end() as usize).min(total).max(start);
        Some(toks[start..end].to_vec())
    }

    /// Turn token count — pinned bytes the seal recorded.
    pub fn turn_token_count_of(&self, timeline: TimelineId, index: TurnIndex) -> usize {
        self.turn(timeline, index)
            .map_or(0, |e| e.content.token_count)
    }

    /// Whether this turn was generated with thinking suppressed (the
    /// `/no_think` dial active at submit).  The projection re-injects the
    /// `/no_think` soft-switch into this turn's user opener when it re-renders
    /// the turn as history.  `false` for unknown turns.
    pub fn turn_no_think(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.turn(timeline, index)
            .is_some_and(|e| e.content.layout.no_think())
    }

    pub fn turn_count(&self, timeline: TimelineId) -> u32 {
        self.timelines
            .get(&timeline)
            .map_or(0, |t| t.turns.len() as u32)
    }

    /// Corpus size for a conversation — `timeline`'s turn tokens plus the
    /// shared section (workspace) tokens. This is the denominator the GUI shows
    /// as "materialized M / N tokens": the size of the unbounded store this
    /// projection draws from, against which its materialized subset is compared.
    ///
    /// O(1) — served from the maintained running counters, not an O(corpus)
    /// re-sum (which would be called on every reprojection during decode and
    /// scale with depth, defeating the engine's O(1) premise).
    pub fn total_token_count(&self, timeline: TimelineId) -> usize {
        self.timeline_token_totals
            .get(&timeline)
            .copied()
            .unwrap_or(0)
            + self.section_token_total
    }

    /// `timeline`'s own sealed turn tokens — [`Self::total_token_count`] without the
    /// shared sections, which belong to no one conversation. What a per-conversation
    /// figure that is summed across conversations must use, or every conversation
    /// counts the whole tool catalog again. O(1).
    pub fn timeline_token_total(&self, timeline: TimelineId) -> usize {
        self.timeline_token_totals
            .get(&timeline)
            .copied()
            .unwrap_or(0)
    }

    /// Tokens sealed in the whole substrate: every live timeline's turns and the shared
    /// sections, each counted once. O(timelines).
    pub fn corpus_token_total(&self) -> usize {
        self.timeline_token_totals
            .iter()
            .filter(|(tl, _)| !self.tombstoned_timelines.contains(tl))
            .map(|(_, tokens)| tokens)
            .sum::<usize>()
            + self.section_token_total
    }

    pub fn turn_indices(&self, timeline: TimelineId) -> impl Iterator<Item = TurnIndex> + '_ {
        self.timelines
            .get(&timeline)
            .into_iter()
            .flat_map(|t| t.turns.keys().copied())
    }

    /// How many turns `timeline` holds — `0` for one the substrate does not.
    pub fn turn_count_of(&self, timeline: TimelineId) -> u32 {
        self.timelines
            .get(&timeline)
            .map_or(0, |t| t.turns.len() as u32)
    }

    pub fn all_turns(&self) -> impl Iterator<Item = TurnKey> + '_ {
        self.timelines
            .iter()
            .flat_map(|(tl, t)| t.turns.keys().map(move |idx| TurnKey::new(*tl, *idx)))
    }

    /// The conversation's sidebar label, or `None` if none has been
    /// recorded yet (the titler writes it after the first user turn).
    pub fn label_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|e| e.label.as_deref())
    }

    /// Set a timeline's sidebar label, as `Label` record `id` — a label-only
    /// [`Self::apply_conv_meta`]. Empty values are ignored (so the first-submit
    /// placeholder doesn't clobber a real title written earlier).
    pub fn set_label(&mut self, timeline: TimelineId, label: &str, id: RecordIdentity) {
        let meta = ConvMeta {
            label: label.to_string(),
            ..ConvMeta::default()
        };
        self.apply_conv_meta(timeline.raw(), &meta, id);
    }

    /// The client-supplied `conv_id` string for `timeline`, or `None`
    /// if no submit has been recorded yet.
    pub fn conv_id_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|e| e.conv_id.as_deref())
    }

    /// Set a timeline's `conv_id`, as `Label` record `id` — a conv_id-only
    /// [`Self::apply_conv_meta`], which also places the conversation in
    /// creation order. Empty values are ignored.
    pub fn set_conv_id(&mut self, timeline: TimelineId, conv_id: &str, id: RecordIdentity) {
        let meta = ConvMeta {
            conv_id: conv_id.to_string(),
            ..ConvMeta::default()
        };
        self.apply_conv_meta(timeline.raw(), &meta, id);
    }

    /// Merge key/value pairs into a timeline's `custom` metadata bag, as
    /// `Label` record `id` — each key keeps the greatest stamp that set it.
    pub fn merge_custom(
        &mut self,
        timeline: TimelineId,
        kv: &BTreeMap<String, String>,
        id: RecordIdentity,
    ) {
        let meta = ConvMeta {
            custom: kv.clone(),
            ..ConvMeta::default()
        };
        self.apply_conv_meta(timeline.raw(), &meta, id);
    }

    /// The `custom` metadata bag for `timeline`, or `None` if unregistered.
    pub fn custom_of(&self, timeline: TimelineId) -> Option<&POrdMap<String, String>> {
        self.timelines.get(&timeline).map(|e| &e.custom)
    }

    /// Whether a record `id` would set `key` of `timeline`'s `custom` metadata:
    /// nothing set it yet, or `id` supersedes the record that did.
    pub fn custom_field_takes(&self, timeline: TimelineId, key: &str, id: RecordIdentity) -> bool {
        self.timelines
            .get(&timeline)
            .and_then(|e| e.ids.custom.get(key))
            .is_none_or(|held| id.supersedes(held))
    }
}
