//! Runs one decode step across every active sequence as a single wave, and works
//! out which sequences ride it and at what draft depth.
//!
//! Every rule that removes a sequence from the wave is applied in one place, so
//! the width of a step can be explained from its terms. Bands share the step,
//! except that low-priority rows sit out three steps in four beside an urgent
//! decode. A wave carries one LoRA adapter, since the adapter alters projections
//! that every row flows through. Sequences held back in a step still have their
//! prefill and glue work driven in a decode-less wave, or that work would never
//! advance. After the forward, each block is walked, its rejected rows are rolled
//! back, and page cuts are taken from the accepted prefix.

use super::*;

impl Scheduler {
    // ── Decode ─────────────────────────────────────────────────────────

    /// Run one decode step for all active (non-finished) sequences.
    ///
    /// Each decoding sequence contributes its last generated token (and its
    /// drafts); a sequence holding a played static run contributes that run
    /// instead (`stencil_wave`). One wave carries them all.
    pub(in super::super) fn batch_decode_step(&mut self) {
        let mut stencil = StencilWave::default();
        self.decode_rows_step(&mut stencil);
        // After the step, so the wave's span is released before a heal can run
        // the next forward.
        self.settle_stencil_wave(stencil);
    }

    /// [`Self::batch_decode_step`]'s wave and the decode rows' commit. The
    /// static runs it takes go in `stencil`, and are marked written once the
    /// wave has written them.
    pub(super) fn decode_rows_step(&mut self, stencil: &mut StencilWave) {
        let t_select = std::time::Instant::now();
        self.step_rows = 0;
        // A slot with a pending deferred glue fire reprojects THIS wave (the
        // co-batched glue member rewrites its `[sealed | glue]` prefix), so its
        // logical offset and block table are mid-rewrite. It must not also run
        // a decode step this wave: the wave would then list the same slot in
        // both the decode and glue groups, and the per-unique-id cache borrow
        // (`caches_for_sequences_mut`) collapses the assembled context list,
        // desyncing every later member's varlen metadata from its slot headers.
        // Skip it here; its decode resumes next wave against the reprojected
        // backing (the glue fire drains in `take_wave_glue` during this step).
        let glue_pending: std::collections::HashSet<usize> = self
            .deferred_glue_fires
            .iter()
            .map(|p| p.parent_id.0)
            .collect();
        // A sequence holding a played static run rides this step as that run,
        // not as a decode row: its next token is sampled from the run's row,
        // and the constraint for the one after is the next play's.
        let mut seq_ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, _)| id)
            .filter(|id| !glue_pending.contains(&id.0))
            .filter(|&id| !self.holds_stencil_row(id))
            .collect();
        stencil.rows = self.take_stencil_rows(&glue_pending);

        // **Why this wave is as wide as it is, in the one place that decides.** A decode
        // forward's width is the whole of decode throughput, and it is not a cap — it is
        // whatever survives these filters. Reading it from the outside is guesswork: a
        // width of four could be four sequences active, or twenty with sixteen held back,
        // and those are unrelated problems. Every term that removed a row is named.
        tracing::trace!(
            target: "candle_conversation::scheduler::throttle",
            active = self.active_decodes.len(),
            finished = self
                .active_decodes
                .values()
                .filter(|s| s.finished)
                .count(),
            glue_pending = glue_pending.len(),
            selected = seq_ids.len(),
            "decode row selection",
        );

        // ── Every band's decode rides the step ──────────────────────────────
        // Decode is width-bound — the step's expert loads amortize across its
        // rows — so an ingest row beside a dialogue's costs the dialogue little
        // and is the ingest's whole progress. Bands share the step; only `Low`
        // beside an `Urgent` decode sits out, three steps in four (below,
        // `urgent_hold`). The highest band leads, so the adapter rule below
        // keeps the dialogue's. `decode_priority_or_high` treats an
        // unresolvable slot as `High`. Tenants are equal: the band alone ranks.
        seq_ids.sort_by_key(|&id| std::cmp::Reverse(rank(self.decode_priority_or_high(id))));

        // ── One adapter per wave ─────────────────────────────────────────────
        //
        // A LoRA alters the projections every row of the forward flows through
        // together — one set of matmuls runs over the whole batch — so a wave
        // carries one adapter or none. Keep the leading sequence's adapter and
        // defer the rest.
        //
        // This is the same kind of deferral as the ingest cap above: pure
        // scheduling, nothing dropped. Successive waves drain each adapter's
        // cohort in turn, which is why the engine loops waves by adapter rather
        // than trying to batch across them. Taking the *leading* sequence's
        // adapter matters — after the priority sort that is the interactive
        // dialogue, so a bulk adapted ingest never displaces a waiting turn.
        //
        // `BatchedInferenceSession::wave_adapter` refuses a mixed wave outright,
        // so without this filter an adapted conversation would fail the forward
        // rather than merely share it. A static run is a decode's row by
        // another name and leads by the same rank; one of another adapter
        // waits for a wave of its own.
        let lead = seq_ids
            .first()
            .copied()
            .into_iter()
            .chain(stencil.rows.iter().map(|r| r.seq))
            .min_by_key(|&id| std::cmp::Reverse(rank(self.decode_priority_or_high(id))));
        if let Some(lead) = lead {
            let wanted = self.session.sequence_adapter(lead.0).map(str::to_owned);
            seq_ids.retain(|id| self.session.sequence_adapter(id.0) == wanted.as_deref());
            let (carried, waiting): (Vec<StencilRow>, Vec<StencilRow>) =
                std::mem::take(&mut stencil.rows)
                    .into_iter()
                    .partition(|r| self.session.sequence_adapter(r.seq.0) == wanted.as_deref());
            stencil.rows = carried;
            self.stencil_rows.extend(waiting);
        }

        // ── A person waiting holds the background back ───────────────────────
        // Beside an `Urgent` decode or prefill, `Low` rows ride one step in
        // `LOW_RIDES_EVERY` (`urgent_hold`): a step bound by its rows' contexts
        // is the dialogue's latency, row for row. A held row is not dropped —
        // it waits, as an adapter's does, and its static run stays pending.
        let urgent = seq_ids
            .iter()
            .copied()
            .chain(stencil.rows.iter().map(|r| r.seq))
            .any(|id| self.decode_priority_or_high(id) == DecodePriority::Urgent)
            || self.an_urgent_prompt_prefills();
        self.urgent_hold.advance(urgent);
        // **A hold needs something to give the step to.** With the creep held
        // back by a debt and no row riding, holding them would leave a step
        // with nothing in it: done at once, its count spent, and the debt no
        // smaller.
        let rides =
            |s: &Self, id: SequenceId| s.urgent_hold.rides(id.0, s.decode_priority_or_high(id));
        let something_else_runs = !self.wave_cohort_advanced
            || seq_ids.iter().any(|&id| rides(self, id))
            || stencil.rows.iter().any(|r| rides(self, r.seq));
        if something_else_runs {
            seq_ids.retain(|&id| rides(self, id));
            let (carried, held): (Vec<StencilRow>, Vec<StencilRow>) =
                std::mem::take(&mut stencil.rows)
                    .into_iter()
                    .partition(|r| rides(self, r.seq));
            stencil.rows = carried;
            self.stencil_rows.extend(held);
        }
        if decode_diag::armed() {
            self.diag_step(&mut seq_ids, stencil);
        }
        self.step_rows = seq_ids.len() + stencil.rows.len();

        if seq_ids.is_empty() {
            // Every active decode is excluded this wave — deferred-glue-pending,
            // riding as a static run, or held back by the adapter rule — so
            // there is no decode row to run. But the work those rows would have
            // carried must still move: `run_prefill_until_budget` is gated
            // behind `decode_width() == 0`, which these excluded-but-live slots
            // hold `> 0`, so the prefill cohort, section chunks, deferred glue
            // and the static runs ride ONLY the decode sweep. Without this,
            // neither path advances them: a hard deadlock — 0 forwards, the
            // queue standing still, a dialogue's next-round prefill (1,856
            // tokens) never advancing. Drive a decode-less wave, exactly as
            // `run_prefill_until_budget` would, whenever any of the four has
            // work.
            if !self.deferred_glue_fires.is_empty()
                || self.prefill_width() > 0
                || self.section_ingest_width() > 0
                || !stencil.rows.is_empty()
            {
                let _g = profile::span("decode:step:rowless_wave");
                let seqs = stencil.seqs();
                let t_wave = std::time::Instant::now();
                let ran = stencil.inputs().and_then(|inputs| {
                    self.decode_forward_cobatched(&[], &[], &[], &[], &seqs, &inputs)
                });
                self.step_census.forward(t_wave.elapsed());
                match ran {
                    // `[stencil]`: nothing decodes and nothing verifies.
                    Ok(held) if !stencil.rows.is_empty() => {
                        if let Some(failures) = self.decode_retry.succeeded() {
                            tracing::info!(failures, "static run wave ran after failing to run");
                        }
                        self.stencil_rows_written(stencil, &held.rows);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("decode: decode-less wave for excluded rows failed: {e}");
                        if !stencil.rows.is_empty() {
                            if is_step_refusal(&e) {
                                self.retry_step(&stencil.ids(), "static run wave", &e);
                            } else {
                                self.fail_all_decodes(
                                    &stencil.ids(),
                                    &format!("static run wave failed: {e}"),
                                );
                            }
                        }
                    }
                }
            } else {
                // A step with nothing to run at all — every decode excluded and
                // no other work to carry. Counted, so a quantum that spins
                // through these reads as such rather than as decode time.
                profile::record("decode:step:no_rows", t_select.elapsed());
            }
            return;
        }
        profile::record("decode:step:select", t_select.elapsed());

        let _t_step = super::super::PhaseTimer::new("decode_batch_step");

        tracing::trace!(
            target: "sched",
            "decode batch={} prefill_active={}",
            seq_ids.len(),
            self.active_prefills.len()
        );

        // Build input tensors: each sequence's last generated token, shape [1, 1].
        // Capture the per-slot input tokens so we can mirror them
        // into the diagnostic log alongside `advance_sequence`.
        let input_tokens: Vec<u32> = seq_ids
            .iter()
            .map(|&id| *self.active_decodes[&id].generated_tokens.last().unwrap())
            .collect();
        // That token is now on its way out, so it stops being pending. Recorded
        // here rather than after the forward because the input is fixed at this
        // point and a failed wave re-forwards the same token — which is the
        // benign direction (a re-forward of a token still marked sent writes the
        // same slot again), where leaving it pending would have the NEXT
        // injection carry it a second time.
        for &id in seq_ids.iter() {
            if let Some(s) = self.active_decodes.get_mut(&id) {
                s.mark_forwarded();
            }
        }
        // Extract raw usize IDs for the forward_batched call into candle-transformers.
        let seq_ids_raw: Vec<usize> = seq_ids.iter().map(|id| id.0).collect();

        // Attended-KV length each decode step sweeps, summed over the batch
        // (captured before the forward advances the offsets). On the wave line
        // this is the per-step context the decode attends over.
        let kv_len: usize = seq_ids_raw
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .sum();

        // Where each sequence stands before this step, which is what its KV
        // rolls back to plus however many tokens it keeps.
        let poss: Vec<usize> = seq_ids_raw
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .collect();

        // ── Draft ────────────────────────────────────────────────────────────
        //
        // **Every decode runs this path, drafter or not.** A model without one
        // proposes nothing, every sequence takes an ordinary one-token row, and
        // the accept walk below commits exactly one token each — so there is no
        // second, plain decode path to drift out of step with this one.
        let t_fwd = std::time::Instant::now();
        // The KV the decodes have committed, sampled before this step writes:
        // what a creep cohort formed now will lose to them over its sweep
        // (`sweep_forecast`). Before, not after — the forward writes every block
        // whole and the rollback below takes the rejected rows off, so offsets
        // read after it would count drafts that never reach the KV.
        let offsets: Vec<(usize, usize)> = self
            .active_decodes
            .keys()
            .filter_map(|&id| self.session.sequence_offset(id.0).map(|at| (id.0, at)))
            .collect();
        self.sweep_forecast.sample_decodes(t_fwd, offsets);
        // Who drafts, and where each block stops — see `BlockGuard`. Free decode
        // drafts, and so does a grammar inside a free-text span (a think block,
        // a tool-call value): nothing is masked there, so the proposals are
        // judged under exactly the rules a plain step applies, and the block
        // stops at the token that leaves the span. A grammar at a branch, a
        // static run or its exit takes a plain row: its next token is
        // constrained or written, and there is nothing to speculate about.
        //
        // So does a replayed turn: its next token is its recording's, so there
        // is nothing to propose.
        let mut guards: Vec<Option<BlockGuard>> = seq_ids
            .iter()
            .map(|id| {
                // Every id here was read from `active_decodes` above.
                let s = &self.active_decodes[id];
                if !s.drafts() {
                    return None;
                }
                BlockGuard::for_step(
                    s.stencil.as_ref(),
                    s.pending_mask.as_ref(),
                    &s.triggers,
                    s.generated_tokens.last().copied(),
                )
            })
            .collect();
        // The model's own measured width ladder is the CEILING: how far it is
        // worth drafting shrinks as the wave widens, and where it stops paying
        // depends on the checkpoint's shape rather than on anything the
        // scheduler knows. Each turn drafts, under it, the depth its own
        // acceptance makes cheapest per token at the model's drafted-token cost
        // (`DraftDepth`), and the drafter walks the cohort to the deepest of them. A turn
        // its guard keeps to a plain row drafts nothing and must not set that
        // depth, or one such turn keeps the whole wave drafting at the ceiling
        // for proposals that are all thrown away.
        // A depth the partition refused since ground last freed bounds it
        // (`RefusedDepth`): a step that does not fit drafts shallower.
        self.settle_refused_width();
        let ceiling = self.draft_ceiling(seq_ids.len());
        let token_cost = self.model.draft_token_cost();
        let depths: Vec<usize> = seq_ids
            .iter()
            .zip(&guards)
            .map(|(id, guard)| match guard {
                Some(_) => self.active_decodes[id]
                    .draft_depth
                    .budget(ceiling, token_cost),
                None => 0,
            })
            .collect();
        let budget = depths.iter().copied().max().unwrap_or(0);
        let draft_span = profile::span("decode:draft");
        let mut drafts = match self.model.speculative_draft(
            &mut self.session,
            &seq_ids_raw,
            &input_tokens,
            budget,
        ) {
            Ok(d) => d,
            Err(e) if is_step_refusal(&e) => {
                if !is_expert_stall(&e) {
                    self.note_refused_depth(budget, &e);
                }
                self.retry_step(&seq_ids, "decode draft", &e);
                return;
            }
            // A draft is a proposal: the walk rolled back what it wrote, and
            // the step needs none of it. Every turn in the wave rides as a
            // plain row; a fault that is not the draft's fails the forward,
            // which is where a turn's own bytes are at stake.
            Err(e) => {
                tracing::error!(
                    sequences = seq_ids.len(),
                    "decode draft failed, so the step runs without one: {e}"
                );
                vec![Vec::new(); seq_ids.len()]
            }
        };
        if drafts.len() != seq_ids.len() {
            self.fail_all_decodes(
                &seq_ids,
                &format!(
                    "drafter returned {} proposal lists for {} sequences",
                    drafts.len(),
                    seq_ids.len()
                ),
            );
            return;
        }
        // The drafter walks the cohort to the deepest turn's depth; each turn
        // keeps only its own, and one its guard keeps to a plain row keeps none.
        for (draft, &depth) in drafts.iter_mut().zip(&depths) {
            draft.truncate(depth);
        }

        // A block is the sequence's committed token followed by its proposals.
        // A sequence that drafted nothing has a one-token block and rides the
        // wave as a plain decode row rather than a verify member.
        let blocks: Vec<Vec<u32>> = drafts
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut b = Vec::with_capacity(d.len() + 1);
                b.push(input_tokens[i]);
                b.extend_from_slice(d);
                b
            })
            .collect();
        let plain: Vec<(usize, u32)> = (0..blocks.len())
            .filter(|&i| blocks[i].len() == 1)
            .map(|i| (seq_ids_raw[i], input_tokens[i]))
            .collect();
        let spec_idx: Vec<usize> = (0..blocks.len()).filter(|&i| blocks[i].len() > 1).collect();
        let spec_seqs: Vec<usize> = spec_idx.iter().map(|&i| seq_ids_raw[i]).collect();
        let spec_blocks: Vec<Vec<u32>> = spec_idx.iter().map(|&i| blocks[i].clone()).collect();
        draft_span.end();
        let draft_elapsed = t_fwd.elapsed();
        self.step_census.host(HostPart::Draft, draft_elapsed);

        // ── The wave ─────────────────────────────────────────────────────────
        //
        // The model plans its rows; this owns the forward, so the verify blocks
        // ride the SAME continuous-fair wave as the creep group and the glue and
        // speculation costs no prefill throughput.
        let begin_span = profile::span("decode:begin_verify");
        let t_plan = std::time::Instant::now();
        let plan = match self.model.begin_verify(
            &mut self.session,
            &plain,
            &spec_seqs,
            &spec_blocks,
            budget,
        ) {
            Ok(Some(p)) => p,
            Ok(None) => {
                self.fail_all_decodes(
                    &seq_ids,
                    "this model drafted proposals but cannot verify them — it overrides \
                     `speculative_draft` without `begin_verify`",
                );
                return;
            }
            Err(e) => {
                // It may have armed some of what it arms before failing, and an
                // armed block or a capture left standing outlives the step it
                // belongs to — the next rewind would fold from it.
                self.model.abort_verify(&spec_seqs);
                self.fail_all_decodes(&seq_ids, &format!("verify setup failed: {e}"));
                return;
            }
        };
        begin_span.end();
        self.step_census.host(HostPart::Plan, t_plan.elapsed());
        let wave_rows = plan.rows;
        let forward_span = profile::span("decode:forward_cobatched");
        // The accept walk stacks each block position's rows beside the head's
        // logits on the forward's span, so the wave prices those stacks.
        self.session.set_accept_in_place(true);
        // So do the bitmaps of the rows a grammar holds to a set too wide to
        // name, each decided by now.
        let stencil_rows = seq_ids
            .iter()
            .filter_map(|id| self.active_decodes.get(id))
            .filter(
                |s| matches!(&s.pending_mask, Some(StepMask::Branch(set)) if set.bits().is_some()),
            )
            .count();
        self.session.set_stencil_rows(stencil_rows);
        let stencil_seqs = stencil.seqs();
        let t_wave = std::time::Instant::now();
        let forwarded = stencil.inputs().and_then(|stencil_inputs| {
            self.decode_forward_cobatched(
                &plan.decode_seqs,
                &plan.decode_inputs,
                &plan.verify_seqs,
                &plan.verify_inputs,
                &stencil_seqs,
                &stencil_inputs,
            )
        });
        self.session.set_accept_in_place(false);
        self.session.set_stencil_rows(0);
        self.step_census.forward(t_wave.elapsed());
        // Held until the rollback: the rows below, the walk's stacks and the
        // rows the commit checks are all views on the forward's span.
        let held = match forwarded {
            Ok(l) => l,
            Err(e) => {
                self.model.abort_verify(&spec_seqs);
                // The static runs rode this forward, so its fate is theirs: a
                // refusal holds their rows for the retry, a fault fails them.
                let rode: Vec<SequenceId> = seq_ids.iter().copied().chain(stencil.ids()).collect();
                if is_step_refusal(&e) {
                    if !is_expert_stall(&e) {
                        self.note_refused_depth(budget, &e);
                    }
                    self.retry_step(&rode, "decode forward", &e);
                } else {
                    self.fail_all_decodes(&rode, &format!("decode forward failed: {e}"));
                }
                return;
            }
        };
        forward_span.end();
        // The static runs' rows follow the plan's `[decode | verify]` prefix.
        // Taken now, while the span holds them; the walk reads only the prefix.
        if !stencil.rows.is_empty() {
            let at = wave_rows.min(held.rows.len());
            let own = &held.rows[at..(at + stencil.rows.len()).min(held.rows.len())];
            self.stencil_rows_written(stencil, own);
        }
        let scored = &held.rows[..wave_rows.min(held.rows.len())];
        if let Some(failures) = self.decode_retry.succeeded() {
            tracing::info!(failures, "decode step ran after failing to run");
        }
        let fwd_elapsed = t_fwd.elapsed();
        // The step's two halves by the wave's shape, so a pure decode step is
        // never averaged with one that also carried an ingest chunk.
        let (draft_name, verify_name) = if self.wave_cobatched {
            ("decode:cobatched:draft", "decode:cobatched:verify_forward")
        } else {
            (
                "decode:decode_only:draft",
                "decode:decode_only:verify_forward",
            )
        };
        profile::record(draft_name, draft_elapsed);
        profile::record(verify_name, fwd_elapsed.saturating_sub(draft_elapsed));
        let fwd_ms = fwd_elapsed.as_millis() as u64;
        // Charge the forward's FULL duration to every sequence in it — this is
        // decode-busy time per turn, the denominator of the reported tok/s.
        // Microseconds, so short forwards do not truncate to zero.
        let fwd_us = fwd_elapsed.as_micros() as u64;
        for &id in seq_ids.iter() {
            if let Some(st) = self.active_decodes.get_mut(&id) {
                st.decode_busy_us = st.decode_busy_us.saturating_add(fwd_us);
            }
        }
        super::super::record_phase(t_fwd, "decode_forward");
        // Rows, not sequences: a speculative step scores a block per drafted
        // sequence, and pricing it as one row each would understate the wave.
        self.wave_stats
            .record(false, seq_ids.len(), wave_rows, kv_len, fwd_ms);
        // What this step cost, taught to the planner. A bus-bound step says
        // nothing about the layer time and the model discards it; a
        // compute-bound one is the only thing that can move an estimate the
        // decode side was seeded 26x optimistic on. The hit coefficient rides
        // the same moment because it is read from the counters this forward just
        // moved.
        // Microseconds, not the truncated millisecond the stats line uses: the
        // truncation is a floor, so it teaches the model that every forward was
        // faster than it was — a 1.4 ms step reads as 1.0, and a sub-millisecond
        // one as zero and is dropped entirely. That bias lands straight on
        // `layer_secs`, which is the estimate this wiring exists to correct.
        //
        // **Only a step that carried nothing else.** A co-batched step's time is
        // the decode rows' and the creep slice or glue that rode them; taught as
        // a decode step, it priced every decode at the creep's cost too — the
        // published step, the trade's decode side, a burst's placement. On
        // BlackWorld (d7f850f79, 2026-10-20) 24 decodes priced at 225–575 ms
        // through heavy-read's co-batched steps.
        // A co-batched step teaches the fit with its creep as the third term
        // (`WaveRate::observe_decode_beside`), so a phase that co-batches every
        // step still says what its decodes cost; one carrying only glue, a cost
        // that term does not hold, teaches nothing.
        // The positions the decode rows attended, the fit's depth column.
        let positions: f64 = seq_ids
            .iter()
            .map(|&s| self.session.sequence_offset(s.0).unwrap_or(0) as f64)
            .sum();
        if !self.wave_cobatched {
            self.observe_decode_forward(seq_ids.len(), fwd_us, positions, wave_rows);
        } else {
            self.observe_decode_beside(
                seq_ids.len(),
                fwd_us,
                self.wave_step_creep_layer_rows,
                self.wave_step_creep_layer_pairs,
                positions,
                wave_rows,
            );
        }
        self.observe_expert_hit_rate();

        // Reads the scored rows back and advances each sequence by what the wave
        // actually wrote — the walk below rolls the rejected tail off again.
        let _accept_span = profile::span("decode:end_verify_and_sample");
        let t_read = std::time::Instant::now();
        let verified = {
            let _g = profile::span("decode:end_verify");
            self.model.end_verify(
                &mut self.session,
                &plain,
                &spec_seqs,
                &spec_blocks,
                scored.to_vec(),
            )
        };
        self.step_census.host(HostPart::Read, t_read.elapsed());
        let (plain_rows, spec_rows) = match verified {
            Ok(r) => r,
            Err(e) => {
                self.model.abort_verify(&spec_seqs);
                self.fail_all_decodes(&seq_ids, &format!("verify readback failed: {e}"));
                return;
            }
        };
        // Scored rows per sequence in cohort order: one for a plain row, one per
        // block position for a verify member.
        let mut rows_of: Vec<Vec<Tensor>> = vec![Vec::new(); blocks.len()];
        {
            let mut p = 0usize;
            for i in 0..blocks.len() {
                if blocks[i].len() == 1 {
                    match plain_rows.get(p) {
                        Some(r) => rows_of[i] = vec![r.clone()],
                        None => {
                            self.model.abort_verify(&spec_seqs);
                            self.fail_all_decodes(&seq_ids, "verify returned too few plain rows");
                            return;
                        }
                    }
                    p += 1;
                }
            }
            for (k, &i) in spec_idx.iter().enumerate() {
                match spec_rows.get(k) {
                    Some(r) if r.len() == blocks[i].len() => rows_of[i] = r.clone(),
                    _ => {
                        self.model.abort_verify(&spec_seqs);
                        self.fail_all_decodes(
                            &seq_ids,
                            "verify returned the wrong number of rows for a block",
                        );
                        return;
                    }
                }
            }
        }

        let configs = self.step_configs(&seq_ids);

        // Hand the persistent sampling states to the chooser for the duration of
        // the step: it needs them mutably (each sampled token is recorded into
        // its sequence's history, which is what prices the next block position),
        // and they go back into the map immediately after the walk.
        let removed_states = self.take_sampling_states(&seq_ids);

        // ── Accept walk ──────────────────────────────────────────────────────
        //
        // Every block position of every sequence is scored in ONE sampler
        // dispatch, each priced against the history its earlier positions'
        // drafts would leave, and the walk then commits position by position
        // from those picks (`SpecChooser`). Sampling each row — rather than
        // taking its argmax — is what draws each row from the distribution plain
        // decoding would: every drafter here proposes greedily, so the textbook
        // accept/reject rule collapses to "sample the row, accept the proposal
        // iff the sample agrees" (see `candle_transformers::models::
        // speculative_choice`). On top of that rule the chooser commits a draft
        // the sample missed when the model's typical-acceptance thresholds pass
        // it, inside the same dispatch — the one place a sampled step departs
        // from plain decoding.
        let t_sample = std::time::Instant::now();
        let (state_ids, states): (Vec<SequenceId>, Vec<SequenceSamplingState>) =
            removed_states.into_iter().unzip();
        let mut chooser = SpecChooser::new(
            &self.sampler,
            states,
            configs,
            self.model.typical_acceptance(),
        );
        let scored = {
            let _g = profile::span("decode:score");
            locate_rows(&in_head_order(&held.rows), &rows_of)
                .and_then(|(block, positions)| chooser.score(block, &positions, &blocks))
        };
        self.step_census.host(HostPart::Score, t_sample.elapsed());
        if let Err(e) = scored {
            for (id, state) in state_ids.into_iter().zip(chooser.into_states()) {
                self.sampling_states.insert(id, state);
            }
            self.model.abort_verify(&spec_seqs);
            self.describe_nonfinite_step("score", &seq_ids, &e);
            self.fail_all_decodes(&seq_ids, &format!("sampling failed: {e}"));
            return;
        }
        let mut emitted: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
        // How many more tokens each sequence may generate. The sinks apply only
        // this and EOS — the cheap half of the stop policy, which bounds how much
        // of a block reaches the KV; the full per-token path below can still stop
        // a sequence earlier, and what it keeps is rolled back to at the end.
        let room: Vec<usize> = seq_ids
            .iter()
            .map(|id| {
                self.active_decodes
                    .get(id)
                    .map_or(0, |s| s.max_tokens.saturating_sub(s.generated_tokens.len()))
            })
            .collect();
        let t_walk = std::time::Instant::now();
        let walked = {
            let _g = profile::span("decode:accept_walk");
            let eos = &self.eos_tokens;
            let breaks = &self.page_break_tokens;
            let tokenizer = &self.tokenizer;
            let mut walk = AcceptWalk::new(&blocks);
            let mut failure = None;
            // Whether each block was stopped by a rule (EOS, the turn's room,
            // its guard) rather than by a rejected proposal.
            let mut ruled = vec![false; blocks.len()];
            while !walk.finished() {
                let rows = walk.rows();
                let tokens = match chooser.commit(&rows) {
                    Ok(t) => t,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                // A sequence leaves the walk where its commit path will stop it,
                // so its sampling state never advances past what it commits. A
                // sequence with no guard drafted nothing and leaves after this
                // position anyway.
                if let Err(e) = walk.commit(&tokens, |i, t| {
                    emitted[i].push(t);
                    let goes_on = !eos.contains(&t)
                        && emitted[i].len() < room[i]
                        && guards[i].as_mut().is_some_and(|g| {
                            let bytes = || {
                                tokenizer
                                    .decode(&[t], false)
                                    .map(String::into_bytes)
                                    .unwrap_or_default()
                            };
                            g.continues_after(t, bytes, breaks)
                        });
                    ruled[i] |= !goes_on;
                    goes_on
                }) {
                    failure = Some(e);
                    break;
                }
            }
            match failure {
                Some(e) => Err(e),
                None => Ok((walk.finish(), ruled)),
            }
        };
        self.step_census.host(HostPart::Walk, t_walk.elapsed());
        let kept = match walked {
            // The walk's next-seed is not read here: the scheduler takes each
            // step's input from `generated_tokens.last()`, which the commit
            // below fills, so a second copy of it could only disagree.
            Ok(((_, kept), ruled)) => {
                // What the walk kept sets the depth the next step drafts at —
                // when it measured the drafter. A block a rule cut short (a
                // span's exit, a trigger, a page break, EOS, the turn's room)
                // says how far the grammar let it run, not how far the
                // proposals held, so it is recorded only if it kept everything.
                for (i, id) in seq_ids.iter().enumerate() {
                    if ruled[i] && kept[i] < blocks[i].len() {
                        continue;
                    }
                    if let Some(s) = self.active_decodes.get_mut(id) {
                        s.draft_depth.record(blocks[i].len() - 1, kept[i]);
                    }
                }
                kept
            }
            Err(e) => {
                for (id, state) in state_ids.into_iter().zip(chooser.into_states()) {
                    self.sampling_states.insert(id, state);
                }
                // The rollback below is never reached, so the rewind spans this
                // step's verify took are still held. Drop them: they name
                // offsets inside a block whose accept never happened.
                self.model.abort_verify(&spec_seqs);
                self.describe_nonfinite_step("choose", &seq_ids, &e);
                self.fail_all_decodes(&seq_ids, &format!("sampling failed: {e}"));
                return;
            }
        };
        // The row each token was drawn from, for the health checks, taken before
        // the states go back.
        let captured: Vec<Vec<Tensor>> = chooser.rows().to_vec();
        // Sequences whose turn the guard bar fails this step, described once the
        // rollback has settled: until then a verifying sequence's recurrent state
        // still has this step's rows pending, and reading it would describe a
        // state the step has not finished deciding.
        let mut degenerate: Vec<SequenceId> = Vec::new();
        for (id, state) in state_ids.into_iter().zip(chooser.into_states()) {
            // A sequence that has just started emitting token 0 gets its
            // recurrent state described, once.
            //
            // The logits are the far end of a forty-layer forward, so a NaN
            // there names nothing: every candidate explanation produces exactly
            // that observation. What separates them is whether the sequence's
            // PERSISTENT state is already non-finite — corruption that will
            // outlive this wave and that names the layer it entered at — or
            // whether the state is clean and only this wave's activations went
            // bad. Those are different bugs in different code, and the logits
            // cannot tell them apart.
            //
            // Fires at the guard bar, not on the first zero. Token 0 is `!` in
            // these vocabularies and ordinary prose contains it, so triggering
            // on a single one reports healthy turns as faults — measured, 16
            // such reports in a clean ingest. A RUN of them is the signal.
            //
            // Read from the sticky flag, never the run's length: a speculative
            // step commits several tokens, and one that commits the run's last
            // `!` and the guard's forced EOS together has reset the run by
            // now. Checking the length let exactly such a turn seal as an
            // answer of eight `!`.
            if state.degenerate && self.fail_degenerate(id) {
                degenerate.push(id);
            }
            self.sampling_states.insert(id, state);
        }

        let sample_ms = t_sample.elapsed().as_millis() as u64;
        super::super::record_phase(t_sample, "decode_sample");

        // Pre-decode the sampled tokens into a readable string for the timing
        // trace.  Only built when TRACE is active (gated by `tracing::enabled!`)
        // so the tokenizer call doesn't fire on the hot path under default
        // logging.  Multi-sequence batches join the per-sequence fragments
        // with `|`.
        let want_token_str = tracing::enabled!(target: "candle_conversation::scheduler::timing", tracing::Level::TRACE);
        let token_str: String = if want_token_str {
            let skip = !self.show_special_tokens;
            // Slot-labelled (`seq:token`): each fragment attributes to its slot
            // so a cross-slot row swap (one stream continuing another's text)
            // is directly visible in the trace instead of an anonymous mixture.
            emitted
                .iter()
                .zip(seq_ids.iter())
                .map(|(toks, id)| {
                    let frag = self
                        .tokenizer
                        .decode(toks, skip)
                        .unwrap_or_else(|_| "<?>".to_string());
                    format!("{}:{}", id.0, frag)
                })
                .collect::<Vec<_>>()
                .join("|")
        } else {
            String::new()
        };

        let accepted: usize = kept.iter().sum();
        tracing::trace!(
            target: "candle_conversation::scheduler::timing",
            batch = seq_ids.len(),
            rows = wave_rows,
            accepted,
            fwd_ms,
            sample_ms,
            token_str = %token_str,
            "decode_step",
        );

        // ── Commit, one position at a time ───────────────────────────────────
        //
        // Each token the walk accepted runs the full per-token path — stencil
        // advance, health, the stream event, reprojection triggers — exactly as
        // it would on a single-token step. A sequence that the walk carried
        // further than the per-token path will tolerate simply stops appearing.
        let deepest = emitted.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut committed = vec![0usize; blocks.len()];
        // Healed prefixes to forward once the rollback has settled — see
        // `HealPrefill`.
        let mut heal_prefills: Vec<HealPrefill> = Vec::new();
        let commit_span = profile::span("decode:commit");
        let t_commit = std::time::Instant::now();
        for p in 0..deepest {
            let at: Vec<usize> = (0..blocks.len())
                .filter(|&i| emitted[i].len() > p)
                .filter(|&i| {
                    self.active_decodes.get(&seq_ids[i]).is_some_and(|s| {
                        // Stopped by the per-token path — a health abort, a sink
                        // that closed.
                        !s.finished
                            // **Past the first position, only free text goes
                            // on.** The walk stops a block at the token that
                            // opens a grammar or leaves a free-text span
                            // (`BlockGuard`), so a later position here was
                            // sampled under the rules the sequence still
                            // decodes under: no grammar at all, or a grammar
                            // still inside the span it drafted in. This
                            // restates that rule where the tokens are
                            // committed. A position past a grammar's edge
                            // would feed text it was supposed to constrain to
                            // `driver.accept`, whose heal path then rewrites
                            // the token against a sequence whose KV already
                            // holds the rest of the block.
                            && (p == 0
                                || s.stencil.as_ref().is_none_or(StencilDriver::mid_free_span))
                            // **A page cut armed inside this block ends it too,
                            // for the same reason.** The cut is taken once per
                            // step, after the rollback, against whatever the K/V
                            // then holds — so every token committed after the
                            // arming one lands inside the page the cut was
                            // closing. One token a step that is exactly right;
                            // inside a drafted block it makes the page overshoot
                            // by the rest of the block.
                            //
                            // Measured: a turn whose reasoning span was
                            // `[26..67)` sealed page boundaries
                            // `[0, 26, 27, 68, …]` — the `</think>` cut armed at
                            // one block position and the next position carried
                            // grid 67 into the reasoning's page.
                            // `turn_sealed_without_thinking` then refused the
                            // whole turn, because a span that is not a union of
                            // whole pages cannot be windowed.
                            && !s.pending_page_cut
                    })
                })
                .collect();
            if at.is_empty() {
                break;
            }
            let ids: Vec<SequenceId> = at.iter().map(|&i| seq_ids[i]).collect();
            if p > 0 {
                for id in &ids {
                    if let Some(s) = self.active_decodes.get_mut(id) {
                        // The previous position's token is the block's next
                        // drafted token, so the wave already wrote its KV, and
                        // the rollback keeps it: this position commits too.
                        s.mark_forwarded();
                        // A plain step asks the walk for its action before every
                        // token it samples; inside the span that is `Free`, and
                        // asking keeps the walk's own account of the span in
                        // step with a plain decode's.
                        if let Some(driver) = s.stencil.as_mut() {
                            let action = driver.step();
                            debug_assert!(matches!(action, StepMask::Free { .. }));
                        }
                    }
                }
            }
            let mut toks: Vec<u32> = at.iter().map(|&i| emitted[i][p]).collect();
            let rows: Vec<Tensor> = at.iter().map(|&i| captured[i][p].clone()).collect();
            heal_prefills.extend(self.commit_decoded_tokens(&ids, &mut toks, &rows));
            for &i in &at {
                committed[i] += 1;
            }
        }
        commit_span.end();
        self.step_census.host(HostPart::Commit, t_commit.elapsed());

        // ── Roll each sequence back to what actually reached the turn ────────
        //
        // The wave wrote every block whole, rejected proposals included, so this
        // is where a partial accept becomes real. It runs **after** the commit
        // loop, not before it, and keys on `committed` rather than on the walk's
        // `kept`: the per-token path can stop a sequence short of what the walk
        // accepted, and those tokens must come off too.
        //
        // One call, not two. A rewind consumes the verifying sequence's armed
        // block — it is good for exactly one step — so a second
        // `truncate_sequences` in the same step finds no rewind point and
        // refuses, taking every other sequence's rollback down with it. The
        // earlier arrangement truncated to `kept` here and tried to correct to
        // `committed` afterwards, which could not work for that reason and left
        // rejected tokens in the KV whenever the per-token path stopped early.
        //
        // **A row that committed nothing is not rolled back with the others.**
        // Every live row commits at least its first token, so zero means the
        // sequence had already stopped — a degenerate row, a sink that closed —
        // and the model refuses a rewind to zero rows. Asked anyway, that one
        // refusal failed every decode in the step: on 2026-10-17 one
        // conversation's NaN state took another's turn down with it. It fails
        // alone instead, so its turn is discarded rather than sealed over a
        // block the K/V still holds.
        let (stopped, targets) = rollback_targets(&seq_ids_raw, &poss, &committed);
        for &i in &stopped {
            if let Some(state) = self.active_decodes.get_mut(&seq_ids[i]) {
                state.fail(ConversationError::Channel(
                    "the sequence stopped inside a speculative step before committing a token"
                        .to_string(),
                ));
            }
        }
        // Every read of the wave's rows is done; the rollback may run the next
        // forward, which reclaims their span.
        drop(captured);
        drop(rows_of);
        drop(held);
        let rollback_span = profile::span("decode:rollback");
        let t_rollback = std::time::Instant::now();
        if let Err(e) = self.model.truncate_sequences(&mut self.session, &targets) {
            // Every verifying sequence's recurrent rows go with the step: none of
            // them is kept, so none may stay pending into the next one.
            self.model.abort_verify(&spec_seqs);
            self.fail_all_decodes(&seq_ids, &format!("speculative rollback failed: {e}"));
            return;
        }
        // The rows that committed nothing keep none of their block either, so
        // their recurrent rows are dropped — the state stands as the step
        // entered it, and the turn is already failed.
        if !stopped.is_empty() {
            let stopped_seqs: Vec<usize> = stopped.iter().map(|&i| seq_ids_raw[i]).collect();
            self.model.abort_verify(&stopped_seqs);
        }
        rollback_span.end();
        self.step_census
            .host(HostPart::Rollback, t_rollback.elapsed());
        let t_settle = std::time::Instant::now();
        for id in degenerate {
            describe_recurrent_state(self.model.as_ref(), id);
        }
        // The healed prefixes, now that each sequence's KV ends at its last
        // committed position — the only place they belong.
        let heal_span = profile::span("decode:heal");
        for heal in heal_prefills {
            self.forward_heal_prefill(heal);
        }
        heal_span.end();
        // **Now the cuts.** The rollback has settled, so the live tail is the
        // accepted prefix rather than the drafted block — which is both the only
        // point a cut describes the right tokens, and the only point at which it
        // cannot be double-counted by a rewind whose snapshot does not cover
        // pages. See `DecodeState::pending_page_cut`.
        for &id in seq_ids.iter() {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                super::super::flush_page_cut(self.model.as_ref(), id, state);
            }
        }
        // The index-vs-K/V check that caught this lives at wave entry
        // (`qwen4exp index disagrees with this sequence's K/V`), which sees the
        // same divergence one step later and costs nothing per decode step. A
        // second check here would take a lock and walk every KV layer on the hot
        // path to report what that one already reports.
        //
        // Mirror what the KV really holds into the slot's diagnostic log: the
        // block prefix that survived the rollback.
        for (i, &id) in seq_ids.iter().enumerate() {
            let n = committed[i].min(blocks[i].len());
            super::super::Scheduler::record_slot_tokens(&mut self.slot_tokens, id, &blocks[i][..n]);
        }
        self.step_census.host(HostPart::Settle, t_settle.elapsed());
    }
}

/// Split a speculative step's rows for the rollback: the positions (into
/// `seqs`) of the rows that committed nothing, which are not rolled back, and
/// `(sequence, position after its committed tokens)` for every other row.
pub(super) fn rollback_targets(
    seqs: &[usize],
    poss: &[usize],
    committed: &[usize],
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut stopped = Vec::new();
    let mut targets = Vec::with_capacity(seqs.len());
    for (i, &seq) in seqs.iter().enumerate() {
        if committed[i] == 0 {
            stopped.push(i);
        } else {
            targets.push((seq, poss[i] + committed[i]));
        }
    }
    (stopped, targets)
}
