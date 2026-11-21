//! Queues and runs the reprojections that a decode triggers, which rebuild a
//! view's context from fresh provenance and fold the gap-fill into one shared
//! forward.
//!
//! Every trigger goes through one queue that deduplicates the pending views, so
//! its invariants live in one place. Preparation scores the view and injects its
//! sealed prefix. One batched gap-fill forward then computes the boundary glue for
//! every prepared view together, which amortises the fixed cost of a forward
//! across them. Each view is completed afterwards, independently. Ground refused
//! before a view's state has moved skips that reprojection and lets the turn go
//! on, since failing a live turn over a full span is worse than a slightly stale
//! view.

use super::*;

impl Scheduler {
    /// Log the expert-pipeline cost of one gap-fill wave as a before/after
    /// counter delta: expert activations routed by the forward, VRAM hit rate,
    /// DMA traffic, and Markov-prediction precision. This is the ground truth
    /// for why a glue forward costs what it does — a low hit rate means the
    /// expert GEMMs' workers pulled the missing experts across PCIe from pinned
    /// memory, and `cold_misses` counts the ones they waited on the stager for.
    pub(super) fn log_glue_expert_telemetry(
        before: &PipelineStats,
        after: &PipelineStats,
        glue_ms: u64,
        glue_tokens: usize,
    ) {
        let hits = after.expert_hits.saturating_sub(before.expert_hits);
        let misses = after.expert_misses.saturating_sub(before.expert_misses);
        let activations = hits + misses;
        let predicted_hits = after.predicted_hits.saturating_sub(before.predicted_hits);
        let predicted_total = after.predicted_total.saturating_sub(before.predicted_total);
        let pct = |num: usize, den: usize| {
            if den == 0 {
                100.0
            } else {
                100.0 * num as f64 / den as f64
            }
        };
        tracing::trace!(
            target: "candle_conversation::scheduler::reproject",
            glue_ms,
            glue_tokens,
            activations,
            hits,
            misses,
            hit_rate_pct = format!("{:.1}", pct(hits, activations)).as_str(),
            worker_promotions = after.worker_promotions.saturating_sub(before.worker_promotions),
            ahead_claims = after.ahead_claims.saturating_sub(before.ahead_claims),
            pred_precision_pct =
                format!("{:.1}", pct(predicted_hits, predicted_total)).as_str(),
            cold_misses = after.worker_cold.saturating_sub(before.worker_cold),
            ahead_mb = after.ahead_bytes.saturating_sub(before.ahead_bytes) >> 20,
            resident_mb = after.resident_vram_bytes >> 20,
            "gap-fill wave: expert telemetry",
        );
    }

    /// Render a profile snapshot as one compact `name=totalms/count` list,
    /// sorted by total time descending, for a single trace line.
    pub(super) fn format_profile_spans(snap: &ProfileSnapshot) -> String {
        let mut entries = snap.entries.clone();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1));
        entries
            .iter()
            .map(|(name, total_ms, count)| format!("{name}={total_ms:.1}ms/{count}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Queue a view for reprojection at the next drain, deduplicating against
    /// entries already pending. Every reprojection trigger — cadence,
    /// punctuation, an opening's stride, a played-back turn's script — enqueues through
    /// here so the queue's invariants live in one place. An associated fn over
    /// the queue field (not `&mut self`) so trigger sites that hold a
    /// `DecodeState` borrow can still call it.
    pub(in super::super) fn queue_reprojection(pending: &mut Vec<SequenceId>, seq_id: SequenceId) {
        if !pending.contains(&seq_id) {
            pending.push(seq_id);
        }
    }

    /// Drain the queue of views whose decoded count crossed an
    /// `every_n_tokens` boundary in the last `batch_decode_step`.
    ///
    /// Each entry triggers a provenance scan + projection + view swap; on
    /// success the view's `SequenceId` changes (the new id is internal
    /// to the scheduler — never visible to the caller).  Failures mark
    /// the view as finished with an error event.
    pub(in super::super) fn drain_pending_reprojections(&mut self) {
        if self.pending_reprojections.is_empty() {
            return;
        }
        let _t_drain = super::super::PhaseTimer::new("drain_reprojections");
        let _diag = decode_diag::reproject_span();
        let pending = std::mem::take(&mut self.pending_reprojections);

        // Phase 1 — prepare each view: provenance scan + projection + tier elevate +
        // inject the sealed prefix + build the gap-fill descriptor. Removes the
        // view's `DecodeState` into the in-flight; does NOT fire the forward.
        let mut inflights: Vec<super::super::ReprojectInFlight> = Vec::new();
        for view_id in pending {
            // The view may have finished (EOS/health) between trigger and
            // drain; an opening stopped at a boundary is reprojected like a
            // decoding view (`held_turn`).
            let decoding = self
                .active_decodes
                .get(&view_id)
                .is_some_and(|s| !s.finished);
            if !decoding && self.awaiting_opening(view_id).is_none() {
                continue;
            }
            // A request's first reading is scored and the turn submitted again
            // with the seeds (`first_reading`), not reprojected.
            if self
                .awaiting_opening(view_id)
                .is_some_and(|p| p.work.rereads)
            {
                self.read_first(view_id);
                continue;
            }
            if decode_diag::fresh() {
                self.diag_forget_placement(view_id);
            }
            match self.reproject_view_prepare(view_id) {
                Ok(Some(inflight)) => inflights.push(inflight),
                // Nothing to swap — a selection the slot already holds, or no
                // ground for a scan: the turn goes on as it stood.
                Ok(None) => self.resume_unreprojected(view_id),
                // **Refused ground before the turn's state moved: skip, not
                // fail.** A reprojection is a refresh of a view the turn is
                // still on, and nothing of that view has changed yet, so the
                // turn goes on and a later trigger tries again — where failing
                // it ended a live turn because the span was full that moment
                // (`.5`, 2026-10-18).
                Err(e) if self.turn_reading(view_id).is_some() && is_ground_refusal(&e) => {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::reproject",
                        view_id = view_id.0, err = %e,
                        "reprojection skipped: no ground for it now; the turn goes on on its view"
                    );
                    self.resume_unreprojected(view_id);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::reproject",
                        view_id = view_id.0, err = %e,
                        "reprojection prepare failed; aborting the turn on this view"
                    );
                    // A failure before prepare took the turn's state leaves it
                    // here to fail; one after has told the turn already
                    // (`reproject_abandon.rs`).
                    self.fail_turn_in_place(view_id, e);
                }
            }
        }
        if inflights.is_empty() {
            return;
        }

        // Phase 2 — the cross-conversation wave: ONE batched multi-slot gap-fill
        // forward computes every prepared conversation's boundary glue at once,
        // amortising the per-forward fixed cost (MoE expert loads, layer loop,
        // launch) across all of them.
        let plan_refs: Vec<&super::super::projection_assembler::GapFillPlan> =
            inflights.iter().map(|i| &i.plan).collect();
        let glue_total: usize = inflights.iter().map(|i| i.plan.n_glue_tokens).sum();
        tracing::trace!(
            target: "candle_conversation::scheduler::reproject",
            n_slots = inflights.len(),
            glue_total,
            "gap-fill wave: batched multi-slot forward",
        );
        // Snapshot the expert-pipeline counters so the wave's cost can be
        // reported as a delta: how many expert activations this forward routed,
        // how many were VRAM-resident (hits) vs fetched by the expert GEMMs'
        // workers from pinned memory or the stager (misses), and how well
        // the Markov predictor pre-staged them. `None` on dense models.
        let expert_before = self.model.expert_stats();
        // With `--features profile`, drain the kernel-span accumulators built up
        // by decode waves since the last drain, so the post-forward snapshot
        // attributes spans to this glue forward alone.
        if cfg!(feature = "profile") {
            let decode_spans = self.model.snapshot_profiles();
            if !decode_spans.entries.is_empty() {
                tracing::debug!(
                    target: "candle_conversation::scheduler::reproject",
                    spans = %Self::format_profile_spans(&decode_spans),
                    "decode kernel spans since last wave",
                );
            }
        }
        let t_fire = std::time::Instant::now();
        if let Err(e) = super::super::projection_assembler::fire_gap_fill_batch(
            &mut self.session,
            &*self.model,
            &plan_refs,
        ) {
            tracing::warn!(
                target: "candle_conversation::scheduler::reproject",
                err = %e, n = inflights.len(),
                "batched gap-fill forward failed; aborting decode for all prepared views"
            );
            for inflight in inflights {
                // Its assembly recorded the glue pieces as placed; unfilled,
                // they are zero chunks no later rebuild may keep.
                if let Some(state) = self.slot_projection_state.get_mut(&inflight.plan.parent_id) {
                    state.placed_pieces.clear();
                }
                let _ =
                    inflight
                        .turn
                        .event_tx()
                        .send(TurnEvent::Error(ConversationError::Channel(format!(
                            "gap-fill wave forward failed: {e}"
                        ))));
            }
            return;
        }
        // The batched gap-fill forward time — for the wave it's shared across all
        // slots, so each view's reproject log reports the same `glue_ms`.
        let glue_ms = t_fire.elapsed().as_millis() as u64;
        super::super::REPROJ_GLUE_US.fetch_add(
            t_fire.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let (Some(before), Some(after)) = (expert_before, self.model.expert_stats()) {
            Self::log_glue_expert_telemetry(&before, &after, glue_ms, glue_total);
        }
        // Kernel-span breakdown of the glue forward — populated only when built
        // with `--features profile` (snapshot also clears the accumulators, so
        // each wave reports its own spans).
        if cfg!(feature = "profile") {
            let glue_spans = self.model.snapshot_profiles();
            if !glue_spans.entries.is_empty() {
                tracing::debug!(
                    target: "candle_conversation::scheduler::reproject",
                    glue_ms,
                    spans = %Self::format_profile_spans(&glue_spans),
                    "gap-fill wave: kernel spans",
                );
            }
        }

        // Phase 3 — complete each view: finish (deferred user + restore tail) +
        // carve the new view + re-key. Independent per view.
        for inflight in inflights {
            let view_id = inflight.view_id;
            if let Err(e) = self.reproject_view_complete(inflight, glue_ms) {
                tracing::warn!(
                    target: "candle_conversation::scheduler::reproject",
                    view_id = view_id.0, err = %e,
                    "reprojection complete failed; decode for this view ends"
                );
            }
        }
    }
}
