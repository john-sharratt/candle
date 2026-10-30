//! What a submitted turn's projection will put on the device, priced before it
//! does.
//!
//! A turn's working set reaches the device at its projection: the selected
//! history lifted warm/cold → hot, and that history's index pages placed. Both
//! are ground the weight side loses exactly as a turn's own K/V is, and the
//! admission fill is what decides whether a turn may spend it — so it has to be
//! priced from the selection itself, unit by unit, not estimated. Measured on
//! blackworld (2026-10-18) under a 25-turn heavy-read: the QSA index tenant at
//! 5.06 GB and the KV it indexes beside it, spent at submit with no admission
//! decision, and every forward then paid a 38%-hit expert sweep.

use std::time::{Duration, Instant};

use super::*;
use crate::index_pages::StoredPages;

/// Hot bytes lifting `sections` and `turns` would claim: the warm and cold
/// units among them that are not resident, at the sizes `elevate_to_hot`
/// scatters them at. Shared by the lift's price and the elevate's own
/// eviction, so the two cannot disagree.
pub(super) fn incoming_hot_bytes(
    conversation: &Conversation,
    sections: &[SectionId],
    turns: &[TurnKey],
) -> u64 {
    let view = conversation.read();
    let plan = view.snapshot_promotion_state(sections, turns);
    let cold: u64 = plan
        .cold_to_hot
        .iter()
        .flat_map(|c| c.cold.iter())
        .flat_map(|s| s.chunks.iter())
        .map(|c| c.record_len)
        .sum();
    let warm: u64 = plan
        .warm_to_hot
        .iter()
        .map(|w| sealed_total_bytes(&w.warm))
        .sum();
    cold + warm
}

impl Scheduler {
    /// Device bytes projecting `sections` and `turns` would newly claim: the
    /// K/V lifted hot ([`incoming_hot_bytes`]) and the index pages placed for
    /// them, each page priced by the model at what placing it would claim —
    /// zero for a piece already resident, since a placed page is shared.
    pub(super) fn projection_lift(
        &self,
        conversation: &Conversation,
        sections: &[SectionId],
        turns: &[TurnKey],
    ) -> u64 {
        let kv = incoming_hot_bytes(conversation, sections, turns);
        let price = |stored: &StoredPages| -> u64 {
            let Ok(pages) = crate::index_pages::decode(stored.bytes()) else {
                return 0;
            };
            pages
                .iter()
                .zip(stored.page_keys())
                .map(|((_, page), key)| {
                    self.model.positional_lift_bytes(page, key).unwrap_or(0) as u64
                })
                .sum()
        };
        let mut index = 0u64;
        for t in turns {
            if let Some(stored) = self.turn_positional.get(t) {
                index += price(stored);
            }
        }
        let read = conversation.read();
        for &s in sections {
            if let Some(stored) = read
                .section_stream(s)
                .and_then(|stream| self.section_positional.get(stream))
            {
                index += price(stored);
            }
        }
        kv + index
    }
}

/// A submitted turn whose projection is chosen but not yet on the device: what
/// the admission fill offers in its place, priced with its [`lift`].
///
/// [`lift`]: PendingSubmit::lift
pub(crate) struct PendingSubmit {
    /// The turn as it will be queued once admitted, addressed to its parent
    /// until its view is carved.
    pub work: PrefillWork,
    /// Device bytes the projection will claim ([`Scheduler::projection_lift`]).
    pub lift: u64,
    /// Tokens the projected context will hold — the depth the turn's prefill
    /// rows attend over, which its tier is priced at.
    pub depth: usize,
    pub(super) diag_to_write: Option<(TimelineId, SelectionDiagnostics)>,
    pub(super) projected_sections: Vec<SectionId>,
    pub(super) projected_segments: Vec<ProjectionSegment>,
    pub(super) turn_keys_for_elevate: Vec<TurnKey>,
    pub(super) slot_target: Option<ProjectionTarget>,
    pub(super) skip_projection: bool,
    pub(super) seal_group: Option<Vec<TurnContent>>,
}

impl PendingSubmit {
    /// The parent slot the turn is submitted on.
    pub(super) fn parent(&self) -> SequenceId {
        self.work.sequence_id
    }
}

/// How long a submitted turn or a section waits before its wait is reported.
const LONG_WAIT: Duration = Duration::from_secs(120);

/// How often long waits are reported while they last.
const TELL_EVERY: Duration = Duration::from_secs(60);

impl Scheduler {
    /// Say what has waited longer than [`LONG_WAIT`] to be admitted or to
    /// begin — the oldest of each kind, how many more, and at what price — at
    /// most once every [`TELL_EVERY`].
    ///
    /// **A turn the admission never takes says nothing.** A table generation
    /// on WW sat submitted for over an hour, the waves running beside it, and
    /// not one line said it was there or what held it.
    pub(super) fn report_waiting_submits(&mut self) {
        if self
            .waiting_told
            .is_some_and(|at| at.elapsed() < TELL_EVERY)
        {
            return;
        }
        let mut told = false;
        let pending: Vec<&PendingSubmit> = self
            .pending_submits
            .iter()
            .filter(|p| p.work.submitted_at.elapsed() >= LONG_WAIT)
            .collect();
        if let Some(p) = pending.iter().max_by_key(|p| p.work.submitted_at.elapsed()) {
            tracing::info!(
                target: "candle_conversation::scheduler::throttle",
                seq = %p.parent(),
                waited_s = p.work.submitted_at.elapsed().as_secs(),
                tokens = p.work.tokens.len(),
                depth = p.depth,
                lift_mib = p.lift >> 20,
                band = ?self.decode_priority_or_high(p.parent()),
                waiting = pending.len(),
                lift_total_mib = pending.iter().map(|p| p.lift).sum::<u64>() >> 20,
                "submitted turns are still waiting for admission"
            );
            told = true;
        }
        // A section waiting is reported too: a turn whose conversation's own
        // frame has not sealed waits behind it, held before it is ever a
        // submission. The oldest, and how many have waited as long.
        let waiting: Vec<&QueuedSection> = self
            .section_queue
            .iter()
            .filter(|s| s.queued_at.elapsed() >= LONG_WAIT)
            .collect();
        if let Some(s) = waiting.iter().max_by_key(|s| s.queued_at.elapsed()) {
            tracing::info!(
                target: "candle_conversation::scheduler::throttle",
                section = %s.debug_name,
                waited_s = s.queued_at.elapsed().as_secs(),
                tokens = s.tokens.len(),
                band = ?s.band,
                waiting = waiting.len(),
                queued = self.section_queue.len(),
                "sections are still waiting for admission"
            );
            told = true;
        }
        let carved: Vec<&PrefillWork> = self
            .prefill_queue
            .iter()
            .filter(|w| w.submitted_at.elapsed() >= LONG_WAIT)
            .collect();
        if let Some(w) = carved.iter().max_by_key(|w| w.submitted_at.elapsed()) {
            tracing::info!(
                target: "candle_conversation::scheduler::throttle",
                seq = %w.sequence_id,
                waited_s = w.submitted_at.elapsed().as_secs(),
                tokens = w.tokens.len(),
                waiting = carved.len(),
                "admitted turns are still waiting to begin"
            );
            told = true;
        }
        if told {
            self.waiting_told = Some(Instant::now());
        }
    }
}

impl Scheduler {
    /// Put an admitted submission's working set on the device — the lift, the
    /// projection's injection, the view's carve — and answer the turn as it is
    /// to begin, addressed to its view. `None` when the turn could not be
    /// carved (its caller has the error) or went to a turn group instead.
    pub(super) fn materialise_submit(&mut self, pending: PendingSubmit) -> Option<PrefillWork> {
        let PendingSubmit {
            mut work,
            diag_to_write,
            projected_sections,
            projected_segments,
            turn_keys_for_elevate,
            slot_target,
            skip_projection,
            seal_group,
            ..
        } = pending;
        let parent_id = work.sequence_id;
        // The diagnostic side-channel.  No-op when projection used
        // the rule-based path (no tree on the target timeline).
        // Ephemeral probe slots write NOTHING to the substrate — not even
        // this diagnostic side-channel on the (shared) target timeline.
        if let Some((timeline, diag)) = diag_to_write {
            if !self.ephemeral_slots.contains(&parent_id) {
                if let Some(conv) = self.slot_conversations.get(&parent_id) {
                    conv.update(|s| s.set_last_selection(timeline, diag.clone()));
                }
            }
        }

        // Step 1.5: select-promote (cold → warm → hot).
        //
        // Batched warm→hot scatter (one `kv_migrate_on` per
        // layer on `elevate_copy_stream`) + per-item
        // cold→hot recover for any turn whose sealed bytes
        // live in the redo log. After this returns,
        // `apply_projection`'s per-unit `ensure_*_hot`
        // calls hit the hot branch immediately.
        //
        // Skipped when nothing was projected — `elevate_to_hot`
        // would no-op anyway, but the substrate read-lock
        // snapshot is cheap to avoid.
        let mut projected_segments = projected_segments;
        if let Some(conversation) = self.slot_conversations.get(&parent_id).cloned() {
            // Free VRAM held by yesterday's working set (warm-backed hot
            // residences NOT in the incoming projection), then batch
            // select-promote the projected sections/turns into hot before
            // `apply_projection` injects them.
            // Warm → hot for everything the projection selected that is not
            // already resident. A tier crossing, so it is bounded by PCIe rather
            // than by compute, and it runs before any forward can start.
            let _g = profile::span("drain:elevate");
            self.elevate_projection_working_set(
                &conversation,
                &projected_sections,
                &turn_keys_for_elevate,
                "submit",
            );
            // Every other conversation whole, or not this time.
            Self::keep_conversations_whole(
                &conversation,
                slot_target.map(|t| t.timeline),
                &mut projected_segments,
                "submit",
            );
        }

        // When reprojection is disabled and the slot is already
        // seeded, leave the cumulative slot intact — prefill appends
        // onto it below. `apply_projection(_, BlockCount(0), _)` would
        // reset it to empty, so it must be skipped here (not just fed
        // empty segments, which is the RULER/summarisation reset path).
        if !skip_projection {
            // Injecting the projected context into the parent slot: the K/V
            // scatter and the block-table writes for every selected segment. The
            // heaviest single step in the handler, and the one that scales with
            // how much context the projection chose.
            let _g = profile::span("drain:apply_projection");
            if let Err(e) = self.apply_projection(parent_id, BlockCount(0), &projected_segments) {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return None;
            }
        }

        // Step 2: borrowed ranges cover the whole
        // post-injection parent — the projection itself
        // already selected what's visible.  Use the
        // *chunk-count* API (`sequence_block_count`) rather
        // than `offset / CHUNK_SIZE`: each projected section
        // ends in its own partial trailing chunk that the
        // divided value silently under-counts, leaving the
        // tail invisible to the view.  See
        // `BatchedInferenceSession::sequence_block_count`.
        let parent_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        let parent_offset_for_log = self.session.sequence_offset(parent_id.0).unwrap_or(0);
        tracing::trace!(
            target: "candle_conversation::scheduler::view_create",
            parent = parent_id.0,
            parent_block_count,
            parent_offset = parent_offset_for_log,
            offset_div_ceil = parent_offset_for_log.div_ceil(self.chunk_size),
            "view borrow plan",
        );
        // Rolling window over an append-only ingest (design §4.7): on the
        // disable-reprojection path (`skip_projection`), borrow just the
        // system prompt + the last `CODE_READ_WINDOW_TURNS` sealed turns
        // instead of the whole growing parent — bounding otherwise-
        // unbounded ingest context. `window == 0` is the whole-parent
        // borrow.
        let window = if skip_projection {
            CODE_READ_WINDOW_TURNS
        } else {
            0
        };
        let effective_ranges: Vec<BlockRange> = if window > 0 {
            self.slot_targets
                .get(&parent_id)
                .map(|t| t.timeline)
                .zip(self.slot_conversations.get(&parent_id))
                .map(|(tl, c)| c.windowed_ingest_ranges(tl, window, parent_block_count))
                .unwrap_or_else(|| {
                    if parent_block_count == 0 {
                        Vec::new()
                    } else {
                        vec![(0, parent_block_count)]
                    }
                })
                .into_iter()
                .map(|(s, e)| BlockRange::new(s, e))
                .collect()
        } else if parent_block_count == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_block_count)]
        };

        // A turn group carves one view per case, each over the same
        // ranges, instead of the one view below.
        if let Some(cases) = seal_group {
            // Every case is sealed as a turn, so a group has the same
            // preconditions as one: a projection and a substrate target
            // to write into.
            if !matches!(work.seal_action, SealAction::Turn) {
                let _ = work
                    .event_tx
                    .send(TurnEvent::Error(ConversationError::Channel(format!(
                        "submit_turn: a turn group on slot {parent_id} has no substrate \
                                 target to seal its cases into"
                    ))));
                return None;
            }
            self.start_turn_group(
                parent_id,
                effective_ranges,
                cases,
                work.sampling,
                work.belief,
                work.event_tx,
            );
            return None;
        }

        // Step 3: carve the view sequence. Its per-sequence state is
        // forked when admission takes the turn, not here — see
        // `ViewState::state_forked`.
        let (view_id, borrowed) = match self.carve_view(parent_id, &effective_ranges) {
            Ok(t) => t,
            Err(e) => {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return None;
            }
        };

        // Step 4: anchor the active turn's start index in the
        // view's chunk list.
        //
        // Under the read-only projection model, all borrowed
        // chunks (including any partial tail) are shared Arc
        // clones; the view's writable tail is a fresh active
        // chunk pushed at view index `borrowed.0` by
        // `create_view_sequence`. So `borrowed.0` is the
        // anchor for mid-turn captures and final seals.
        let turn_start_parent_blocks = borrowed.0;
        tracing::trace!(
            target: "candle_conversation::scheduler::view_create",
            "view carve: parent {} borrowed {} blocks, parent layout {:?}",
            parent_id,
            borrowed.0,
            self.session
                .provenance_chunk_layout(parent_id.0, parent_offset_for_log),
        );

        // Step 5: register the view so cleanup_finished can auto-finalize.
        self.turn_views.insert(
            view_id,
            ViewState {
                parent_id,
                original_borrowed: borrowed,
                turn_start_parent_blocks,
                question_tokens: work.user_content_end as usize,
                applied_identity: turn_start_identity(parent_block_count),
                state_forked: false,
                holds_parent_state: false,
            },
        );
        work.sequence_id = view_id;
        Some(work)
    }
}

/// Tokens the slot will hold once `segments` are projected onto it: the depth a
/// submitted turn's prefill rows attend over, read from the substrate's own
/// counts for its sealed units.
pub(super) fn projected_depth(
    conversation: &Conversation,
    segments: &[ProjectionSegment],
) -> usize {
    let read = conversation.read();
    segments
        .iter()
        .map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                read.section_tokens_of(rs.id).len()
            }
            ProjectionSegment::Sealed(SealedKind::Turn(rt, _))
            | ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => rt
                .timeline
                .map_or(0, |tl| read.turn_token_count_of(tl, rt.index())),
            ProjectionSegment::Generated { tokens, .. }
            | ProjectionSegment::NewUserMessage { tokens } => tokens.len(),
        })
        .sum()
}
