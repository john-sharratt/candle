//! Plays a tool-call stencil's static runs into the decode wave, and chooses the
//! sampling action for the next token.
//!
//! A static run that follows the pending token becomes that token's row in the
//! same wave, so an N-token run costs no forward of its own, and the token after
//! it is sampled from that row. Only the one token that has not yet been forwarded
//! is pending, because a dropped token is never committed and must not be sent
//! again. A close run ends with the assistant EOS, which terminates the turn and
//! seals it. A think-steer's injected close clears the block flag, so health
//! checks are correct for the answer that follows.

use super::*;

impl Scheduler {
    /// Note the steering finish as an event: a one-line summary of the path
    /// the stencil walk took (so a malformed tool call is diagnosable — e.g.
    /// `bailed=true`, or `free_tokens=0` where an argument value was expected).
    /// `reason` distinguishes a clean completion from a failsafe drop.
    pub(super) fn log_stencil_finish(seq_id: usize, driver: &StencilDriver, reason: &str) {
        let s = driver.stats();
        note("stencil steering finished", || {
            format!(
                "seq_id={seq_id} tree={} reason={reason} prefills={} prefill_tokens={} \
                 branch_tokens={} free_tokens={} heals={} dropped_delimiters={} bailed={}",
                driver.tree().label(),
                s.prefills,
                s.prefill_tokens,
                s.branch_tokens,
                s.free_tokens,
                s.heals,
                s.dropped_delimiters,
                s.bailed,
            )
        });
    }

    // ── Stencil static runs (Layer 3) ──────────────────────────────────

    /// Play any pending `Static` runs for active stencil sequences, then record
    /// the next decode action in `pending_mask`.  Runs once before
    /// `batch_decode_step` each decode iteration.
    ///
    /// A static run `R` following the sequence's pending token `Y` (the last
    /// generated, not-yet-forwarded token) becomes the row `[Y] ++ R`, which the
    /// decode step's own wave writes (`stencil_wave`); the token after it is
    /// sampled from that row under the stencil's next action
    /// ([`Self::commit_after_injection`]). So an N-token run costs no forward of
    /// its own: not N decode steps, not a prefill of its own, and not a decode
    /// step that re-forwards `R.last()` to read the same row. A run that ends the
    /// turn (its last token is EOS) leaves the EOS unforwarded, exactly as a
    /// sampled EOS is.
    pub(in super::super) fn play_stencil_runs(&mut self) {
        let ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished && s.stencil.is_some() && s.pending_mask.is_none())
            .map(|(&id, _)| id)
            .filter(|&id| !self.holds_stencil_row(id))
            .collect();
        let max_recent = self.sampler.max_recent_len();
        // Each sequence's row — its runs this iteration, in the order they were
        // played. One entry per sequence, so its last row is its last run's.
        let mut deferred: Vec<(SequenceId, Vec<u32>)> = Vec::new();
        // The sequences whose run closed the turn: they finish once the row is
        // written.
        let mut ends: HashSet<SequenceId> = HashSet::new();

        for id in ids {
            let mut guard = 0usize;
            loop {
                guard += 1;
                if guard > 100_000 {
                    tracing::error!(seq_id = id.0, "stencil prefill runaway — bailing");
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "runaway");
                        }
                        s.stencil = None;
                    }
                    break;
                }

                // Advance the driver one step (brief mutable borrow).
                let action = match self
                    .active_decodes
                    .get_mut(&id)
                    .and_then(|s| s.stencil.as_mut())
                {
                    Some(driver) => driver.step(),
                    None => break,
                };

                let StepMask::Prefill(run) = action else {
                    // Branch / Free / Done — the action for the next token. When
                    // runs were played above, that token is sampled from their
                    // row; otherwise the decode step samples it.
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        s.pending_mask = Some(action);
                    }
                    break;
                };

                // A replayed turn's stencil must play what the recording holds.
                // A run it does not hold would seal a turn that never happened,
                // so the turn fails here instead.
                let departed = self.active_decodes.get(&id).and_then(|s| {
                    let replay = s.recorded_reply.as_ref()?;
                    let at = s.generated_tokens.len();
                    if !replay.forces(at) {
                        return None;
                    }
                    departure(replay.ids(), at, &run[..], |t| self.eos_tokens.contains(&t))
                        .map(|k| at + k)
                });
                if let Some(at) = departed {
                    self.fail_all_decodes(
                        &[id],
                        &format!(
                            "the replay left its recording at reply id {at}: the stencil \
                             played a run the recorded turn does not hold"
                        ),
                    );
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        s.stencil = None;
                    }
                    break;
                }

                // Prefill `[pending] ++ run` — the whole run, so the forward's last
                // row is the next token's.
                //
                // **`pending` is what has not been forwarded, not simply the last
                // entry.** Both are usually the same single token — but the path
                // that re-drives this loop is a *drop*, and a dropped token is
                // never committed, so `generated_tokens.last()` then names the
                // token the sampling step already forwarded. Sending it again
                // costs nothing in K/V (the offset did not advance, so the write
                // lands on the same slot) and one phantom row in the index, which
                // surfaces as a turn whose pages over-cover it.
                // At most ONE token can be pending — the decode commits one per
                // step and forwards it on the next — so this asks whether there
                // is one, not how many. Taking the whole pending slice would
                // inherit any drift in the count; taking its last entry is the
                // old `generated_tokens.last()` exactly, minus the case that was
                // wrong: after a drop nothing is pending, and nothing is sent.
                // Asserted rather than assumed, because taking `.last()` of a
                // longer pending run would silently drop the earlier entries:
                // they stay in `generated_tokens` and so in the sealed grid,
                // while their K/V was never written. That is the same shape of
                // implicit invariant the page cut broke.
                if let Some(n) = self
                    .active_decodes
                    .get(&id)
                    .map(|s| s.pending_forward().len())
                {
                    if n > 1 {
                        tracing::warn!(
                            seq_id = id.0,
                            pending = n,
                            "more than one token is waiting to be forwarded; only the last \
                             will be carried, so the earlier ones would seal into the grid \
                             with no K/V behind them"
                        );
                    }
                }
                let pending: Option<u32> = self
                    .active_decodes
                    .get(&id)
                    .and_then(|s| s.pending_forward().last().copied());
                // A close run ends with the assistant EOS (`}}\n</tool_call>` +
                // `<|im_end|>`): a tool call is the whole assistant turn, so the
                // EOS terminates it.  Detect it here so the turn is sealed instead
                // of the model free-decoding a hallucinated answer past the call.
                let ends_turn = run.last().is_some_and(|&t| self.eos_tokens.contains(&t));

                let mut input = Vec::with_capacity(1 + run.len());
                input.extend(pending);
                // An EOS-terminated run's EOS is the turn terminator, whose KV is
                // never written (exactly as a model-sampled EOS), so it is left
                // out; every other run goes in whole.
                let forwarded = if ends_turn { run.len() - 1 } else { run.len() };
                input.extend_from_slice(&run[..forwarded]);

                // Empty only when nothing is pending and the run is a lone EOS.
                // Written by the decode step's wave, with every other
                // sequence's runs and the decode rows — not here, one sequence
                // at a time. Each was a whole-model forward of its own on the
                // decode loop: with eight templated ingest turns that was up to
                // eight serial forwards between decode steps, ~29 ms a step of
                // the 8-session decode. The bookkeeping below does not depend on
                // the forward having run, so the run is recorded now and the
                // wave writes it.
                if !input.is_empty() {
                    // One input per sequence: a second run this iteration extends
                    // the first, so the forward's last row follows the last run.
                    match deferred.iter_mut().find(|(s, _)| *s == id) {
                        Some((_, existing)) => existing.extend_from_slice(&input),
                        None => deferred.push((id, input)),
                    }
                    // Everything pending goes out with that row.
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        s.mark_forwarded();
                    }
                }

                // Append the run to the emitted output and stream it. The
                // trailing EOS of a close run is pushed to the buffer (it is part
                // of the sealed turn) but never streamed — matching the normal
                // decode path, which buffers EOS but does not emit it.
                let mut carries_eot = false;
                let think_close = self.think_close;
                let breaks = &self.page_break_tokens;
                if let Some(s) = self.active_decodes.get_mut(&id) {
                    for (k, &t) in run.iter().enumerate() {
                        // The page cut inside this run was already made by the
                        // prefill, which splits the forward at each boundary — the
                        // only place it can be made, since these tokens' rows are
                        // pooled by that pass. A break in the run's LAST slot has
                        // nothing after it in the pass, so its cut is the token
                        // after it: `commit_after_injection` commits that token and
                        // takes it.
                        if k < forwarded {
                            s.push_forwarded(t, think_close, breaks);
                        } else {
                            // The terminating EOS: committed, never forwarded.
                            s.push_pending(t, think_close, breaks);
                        }
                        if !(ends_turn && k + 1 == run.len()) {
                            let _ = s.event_tx.send(TurnEvent::Token(t));
                        }
                    }
                    // A think-steer tree always drops the model's own `</think>`
                    // (the close token is suppressed) and injects the closing tag
                    // as a static run instead, so the commit-loop flip that clears
                    // `inside_think_block` on a sampled `</think>` never fires for a
                    // think turn.  This injected closing tag is where the block
                    // actually ends, so clear the flag here when the prefilled run
                    // carries it — otherwise health stays relaxed forever after the
                    // first think block.
                    let eot = s.sampling_config.segment_close_token_id;
                    carries_eot = eot >= 0 && run.contains(&(eot as u32));
                    if carries_eot {
                        s.health.inside_think_block = false;
                    }
                }
                // Record the run in the repeat-penalty window so the model's
                // subsequent free decode sees the tool-call tokens as recent
                // context.  `record_context_tokens` touches only `recent_tokens`
                // — not `token_counts` or `current_len` — so it cannot skew
                // frequency/presence penalties or the EOS-length failsafe.
                if let Some(ss) = self.sampling_states.get_mut(&id) {
                    ss.record_context_tokens(&run, max_recent);
                    // The injected `</think>` is where a steered block actually ends;
                    // clear the sampler's `in_segment` here too (its commit-loop twin
                    // never sees the dropped close), so the EOT boost and the
                    // in-thinking temperature/DRY gates switch off for the answer.
                    if carries_eot {
                        ss.in_segment = false;
                    }
                }

                if ends_turn {
                    // The stencil emitted the tool call's closing EOS — the
                    // assistant turn is complete.  Stop steering; once the row
                    // is written the turn finishes and `cleanup_finished`
                    // finalizes and seals exactly as for a model-sampled EOS.
                    let written = deferred.iter().any(|(s, _)| *s == id);
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "completed");
                        }
                        s.stencil = None;
                        // A lone EOS with nothing pending has nothing to write.
                        if !written {
                            s.wrote_tool_call = true;
                            s.finished = true;
                        }
                    }
                    ends.insert(id);
                    break;
                }
                // Loop: the next action is the node after this static run.
            }
        }
        // Each row rides the decode steps a piece at a time — at most the
        // model's short span, each page break closing its page between two
        // (`stencil_wave`). No part of it is forwarded on its own.
        for (id, input) in deferred {
            let ends_turn = ends.contains(&id);
            self.stencil_rows
                .push(StencilRow::played(id, input, ends_turn));
        }
    }

    /// Each listed sequence's sampling config for this token, with its stencil's
    /// pending action folded in.
    ///
    /// The action was set by `play_stencil_runs`, which already played any
    /// static runs before it: a branch is its frontier (a tiny gather + sample),
    /// a free-text span clears the stencil (free decode through the kernel), and
    /// `Done` ends the walk (free decode + drop). The sampler resolves these per
    /// row in `sample_batch`, so a mask set here constrains only this sequence —
    /// other rows are unaffected, and constrained rows never run the full-vocab
    /// kernel.
    ///
    /// One function for every place a decoded token is sampled — the decode step
    /// and the token after an injected run — because two copies of this fold
    /// would be two opinions about what a stencil allows.
    pub(super) fn step_configs(&mut self, seq_ids: &[SequenceId]) -> Vec<SamplingConfig> {
        let mut configs: Vec<SamplingConfig> = seq_ids
            .iter()
            .map(|id| self.active_decodes[id].sampling_config.clone())
            .collect();
        for (i, &id) in seq_ids.iter().enumerate() {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                if state.stencil.is_none() {
                    continue;
                }
                match state.pending_mask.take() {
                    Some(StepMask::Branch(set)) => configs[i].constrain_to(&set),
                    Some(StepMask::Free { opens_value, .. }) => {
                        // A free-text span decodes normally.  A think-steer span's
                        // `</think>` and EOS are both intercepted by the session's
                        // `observe` (the suppressed close drops the token and
                        // prefills a continuation; the final span injects the
                        // closing tag), and tool-call value spans close on a byte
                        // delimiter — so the sampler runs free here, but for the
                        // first token of a call's value, where the model chooses
                        // how the value opens: an end token there is the turn
                        // ending and not the value, so it is barred, and the
                        // repetition penalties are lifted
                        // (`StencilSession::free_decode_action`). The config is
                        // this step's own copy, so neither goes further.
                        configs[i].stencil.clear();
                        if opens_value {
                            configs[i]
                                .banned_tokens
                                .extend(self.eos_tokens.iter().map(|&t| t as i32));
                            configs[i].unpenalized_step = true;
                        }
                    }
                    Some(StepMask::Done) => {
                        if let Some(d) = &state.stencil {
                            Self::log_stencil_finish(id.0, d, "completed");
                        }
                        state.stencil = None;
                        configs[i].stencil.clear();
                    }
                    // A `Prefill` is consumed by `play_stencil_runs`; `None`
                    // means the driver wasn't advanced — free-decode this step.
                    Some(StepMask::Prefill(_)) | None => configs[i].stencil.clear(),
                }
            }
        }
        configs
    }

    /// Take each listed sequence's persistent sampling state for one sample, with
    /// its steering span's close semantics synced for this token. The caller puts
    /// them back.
    ///
    /// The sync is only consulted inside a segment: the hard-cap closer script
    /// may play only in a TERMINAL free-text span (or an unsteered block, where
    /// there is no stencil). Everywhere else — a span that retires into further
    /// decoding, a tool-call value, a static prefill — a forced close stays bare.
    pub(super) fn take_sampling_states(
        &mut self,
        seq_ids: &[SequenceId],
    ) -> Vec<(SequenceId, SequenceSamplingState)> {
        seq_ids
            .iter()
            .map(|&id| {
                let mut state = self
                    .sampling_states
                    .remove(&id)
                    .expect("sampling state must exist for active sequence");
                if state.in_segment {
                    state.close_would_continue = self
                        .active_decodes
                        .get(&id)
                        .and_then(|s| s.stencil.as_ref())
                        .is_some_and(|d| !d.in_terminal_close_span());
                }
                (id, state)
            })
            .collect()
    }
}
