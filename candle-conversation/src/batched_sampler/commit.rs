//! Committing a scored row onto its sequence's state, and resolving the raw
//! sampled token into the token the sequence actually takes.
//!
//! A row is committed only onto the state it was scored against, since a
//! speculative draft's row was priced against the state its earlier drafts
//! leave; a mismatch is refused rather than committed. Resolution runs in one
//! place for both the CPU and CUDA paths, so the segment-close override, the
//! degenerate-decode abort and the EOS length failsafes hold whichever path the
//! device selects. The advance that prices a row past without committing it
//! records nothing and logs nothing.

use super::*;

/// Why [`BatchedSampler::resolve`] committed what it did, for the log line
/// the commit writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Resolution {
    Sampled,
    SegmentClose,
    CloseBeforeCall,
    Degenerate,
    WritingCall,
    ForcedEos,
    NoBoundaryTokens,
    GracefulEos,
}

impl BatchedSampler {
    /// Commit a scored row onto `state`: resolve a free row's pick (segment
    /// close, the EOS failsafes), record the token, take the row's RNG offset
    /// and track segment edges. `logits_row` of `logits2d` is the row it was
    /// scored from, described in the log if the commit crosses the degenerate
    /// bar.
    ///
    /// **The state must be the one the row was scored against.** A row of a
    /// speculative step was priced against the state its earlier positions'
    /// drafts would leave; committing it onto any other would hand the
    /// sequence a token drawn under rules it does not stand under, so a
    /// mismatch is refused rather than committed.
    pub fn commit_row(
        &self,
        logits2d: &Tensor,
        logits_row: usize,
        state: &mut SequenceSamplingState,
        config: &SamplingConfig,
        scored: Scored,
    ) -> candle::Result<u32> {
        let live = state.row_key();
        if live != scored.key {
            candle::bail!(
                "commit_row: row {logits_row} was scored against {:?} and commits onto {:?} — \
                 the sampler's advance through the step's drafts and the committed path \
                 disagree",
                scored.key,
                live
            );
        }
        let token = match scored.kind {
            RowKind::Constrained => scored.raw,
            RowKind::Free => {
                // One-shot: the dynamic EOS boost ramp begins as `current_len`
                // reaches `eos_ramp_start` (it increments by one, so this fires
                // exactly once per turn). After this point EOS pressure builds
                // toward the graceful/hard caps.
                if config.eos_boost != 0.0 && state.current_len == config.eos_ramp_start {
                    tracing::debug!(
                        target: "candle_conversation::eos",
                        row = logits_row,
                        current_len = state.current_len,
                        eos_ramp_start = config.eos_ramp_start,
                        eos_ramp_len = config.eos_ramp_len,
                        graceful_eos_after = config.graceful_eos_after,
                        forced_eos_after = config.forced_eos_after,
                        "EOS boost ramp entered",
                    );
                }
                let (token, why) = self.resolve(scored.raw, state, config);
                self.log_resolution(logits_row, why, state, config);
                token
            }
        };
        state.record_token(token, self.max_recent_len);
        state.rng_offset = scored.rng_after;
        if scored.kind == RowKind::Free {
            state.update_segment_state(
                token,
                config.segment_open_token_id,
                config.segment_close_token_id,
            );
        }
        // A row that has just crossed the degenerate bar gets its logits
        // described, once, in the log. The resolution can only say the row was
        // unusable — it never sees the logits — so without this the operator is
        // left with the guard's own guess ("all-equal or non-finite") and no way
        // to tell a dead forward from a NaN one. This reads one row off the
        // device and runs only on the step the bar is crossed, so it costs
        // nothing until something is already wrong.
        if state.degenerate_run == DEGENERATE_TOKEN_RUN {
            let stencil = &config.stencil;
            match describe_logit_row(logits2d, logits_row) {
                Ok(desc) => tracing::error!(
                    target: "candle_conversation::eos",
                    row = logits_row,
                    stencil_len = stencil.len(),
                    stencil_head = ?stencil.iter().take(4).collect::<Vec<_>>(),
                    "degenerate decode: {desc}"
                ),
                Err(e) => tracing::error!(
                    target: "candle_conversation::eos",
                    row = logits_row,
                    "degenerate decode: logits row unreadable ({e})"
                ),
            }
        }
        Ok(token)
    }

    /// Advance `state` as [`Self::commit_row`] would on committing a row whose
    /// pick was `token`. Answers with the token the advance recorded, and
    /// whether a commit could record `token` itself — which is what reaching
    /// the next row takes. Nothing is logged: the row is not being committed,
    /// only priced past.
    pub(super) fn advance(
        &self,
        state: &mut SequenceSamplingState,
        config: &SamplingConfig,
        token: u32,
    ) -> (u32, bool) {
        match config.stencil.as_slice() {
            [forced] => {
                let committed = *forced as u32;
                state.record_token(committed, self.max_recent_len);
                state.advance_rng();
                (committed, committed == token)
            }
            [_, _, ..] => {
                state.record_token(token, self.max_recent_len);
                state.advance_rng();
                // An allow-list draws only allowed tokens.
                (token, config.stencil.contains(&(token as i32)))
            }
            [] if config.allowed_bits.is_some() => {
                let step = self.free_rng_step(state, config);
                state.record_token(token, self.max_recent_len);
                state.rng_offset = state.rng_offset.wrapping_add(step);
                (token, true)
            }
            [] => {
                let step = self.free_rng_step(state, config);
                let (committed, _) = self.resolve(token, state, config);
                state.record_token(committed, self.max_recent_len);
                state.rng_offset = state.rng_offset.wrapping_add(step);
                state.update_segment_state(
                    committed,
                    config.segment_open_token_id,
                    config.segment_close_token_id,
                );
                (committed, committed == token)
            }
        }
    }

    /// How far a full-vocabulary row moves its sequence's RNG offset. The
    /// kernel draws, and advances the offset, only when the row samples — its
    /// temperature, raised inside a segment by `segment_temp_boost`, above
    /// zero — and returns the argmax untouched otherwise. The host sampler
    /// draws on every row.
    pub(super) fn free_rng_step(
        &self,
        state: &SequenceSamplingState,
        config: &SamplingConfig,
    ) -> u64 {
        if !matches!(self.device, Device::Cuda(_)) {
            return 1;
        }
        let boosted = state.in_segment && state.segment_len > 0;
        let temperature = config.temperature
            + if boosted {
                config.segment_temp_boost
            } else {
                0.0
            };
        u64::from(temperature > 0.0)
    }

    /// Resolve a raw sampled token into the token the sequence actually commits.
    ///
    /// Applies, in priority order: the segment-close override (authoritative for
    /// the step), the degenerate-decode abort, then the EOS length failsafes.
    /// `state` is taken by `&mut` for `segment_close_override`, which steps a
    /// closer script as it plays; nothing here records the committed token,
    /// which stays with the caller that owns the advance, and nothing is
    /// logged — [`Self::log_resolution`] does that for a row really committed,
    /// not one a speculative step is only priced past.
    ///
    /// Every row resolves through here because two copies of this logic
    /// drifted once already: the degenerate-decode abort was written into the
    /// CPU sampler alone, while every unconstrained row went to the kernel on a
    /// CUDA device — which is every deployment that matters. A forward
    /// producing unusable logits therefore ran to the length cap instead of
    /// stopping at [`DEGENERATE_TOKEN_RUN`], writing hundreds of `!` into the
    /// conversation and into the substrate, where the turn's signatures then
    /// polluted retrieval. One authority means a guard added here holds on
    /// whichever path the device selects.
    pub(super) fn resolve(
        &self,
        sampled: u32,
        state: &mut SequenceSamplingState,
        config: &SamplingConfig,
    ) -> (u32, Resolution) {
        let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);

        // Segment-close override: force the close token when the segment budget
        // is exhausted.  Authoritative for the step — the EOS failsafes must not
        // clobber the close token or a closer-script token (they fire on a later
        // step, once the segment is closed).
        if let Some(t) = segment_close_override(config, state) {
            return (t, Resolution::SegmentClose);
        }

        // A tool call opened inside the reasoning block closes the block first.
        if let Some(t) = close_before_call(config, state, sampled) {
            return (t, Resolution::CloseBeforeCall);
        }

        if state.degenerate_run >= DEGENERATE_TOKEN_RUN {
            // Degenerate decode: the forward is producing token 0 repeatedly,
            // which is what argmax returns from an all-equal or non-finite
            // logit row. Left alone this runs to the length cap and lands
            // hundreds of `!` in the conversation AND in the substrate, where
            // the turn's signatures then pollute retrieval. Stop at the first
            // sign of it — this is a fault, not an answer.
            return (eos_token_id, Resolution::Degenerate);
        }

        // A call being written is bounded by its grammar, not by the answer's
        // length budget — see `SequenceSamplingState::writing_call`.
        if state.writing_call {
            return (sampled, Resolution::WritingCall);
        }

        if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
            // Hard stop: unconditionally force EOS regardless of sentence position.
            return (eos_token_id, Resolution::ForcedEos);
        }

        if config.graceful_eos_after > 0 && state.current_len >= config.graceful_eos_after {
            if config.sentence_end_token_ids.is_empty() && config.line_end_token_ids.is_empty() {
                // No sentence-end tokens resolved (e.g. model loaded without
                // tokenizer resolution): hard stop at the graceful threshold.
                return (eos_token_id, Resolution::NoBoundaryTokens);
            }
            // Graceful stop: emit EOS only when the last token ended a sentence
            // (`.`, `!`, `?`) or a line.  This lets the current
            // sentence complete before termination, preventing mid-sentence
            // truncation.  `forced_eos_after` is the hard backstop if no boundary
            // is ever seen.
            if state.at_sentence_end(config) {
                return (eos_token_id, Resolution::GracefulEos);
            }
        }

        (sampled, Resolution::Sampled)
    }

    /// The log line for a committed row's resolution. `row` names its logits
    /// row; `state` is the sequence before the token is recorded.
    pub(super) fn log_resolution(
        &self,
        row: usize,
        why: Resolution,
        state: &SequenceSamplingState,
        config: &SamplingConfig,
    ) {
        match why {
            Resolution::Degenerate => tracing::error!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                run = state.degenerate_run,
                "degenerate decode: token 0 emitted {} times consecutively — \
                 forcing EOS. The forward pass produced unusable logits \
                 (all-equal or non-finite); the turn is truncated here.",
                state.degenerate_run,
            ),
            Resolution::ForcedEos => tracing::debug!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                forced_eos_after = config.forced_eos_after,
                "hard EOS forced (length cap)",
            ),
            Resolution::NoBoundaryTokens => tracing::debug!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                graceful_eos_after = config.graceful_eos_after,
                "hard EOS forced (no sentence-end tokens)",
            ),
            Resolution::GracefulEos => tracing::debug!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                graceful_eos_after = config.graceful_eos_after,
                "soft EOS forced (sentence boundary)",
            ),
            Resolution::Sampled
            | Resolution::SegmentClose
            | Resolution::CloseBeforeCall
            | Resolution::WritingCall => {}
        }
    }
}
