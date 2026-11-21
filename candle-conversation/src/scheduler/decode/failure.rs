//! Decides whether a failed decode step is retried or fails its turns, and
//! describes the recurrent state behind a non-finite result.
//!
//! A step refused for lack of device room, or stalled on an expert read, commits
//! nothing, so the same tokens are forwarded again after a backoff rather than
//! ending a live reply. Any other fault fails the turns that it carried. A run of
//! unusable logits long enough to mean the model has stopped fails the turn as
//! well, so nothing decoded over those logits is sealed. Where the persistent
//! state is non-finite, the report names the layer it entered at, because a NaN in
//! the logits alone cannot say whether the state or the wave went bad.

use super::*;

impl Scheduler {
    /// A decode step that did not run but may on the next attempt — the draft
    /// walk's or the verify forward's: its transient tier was refused, or the
    /// device gave a cold expert up on a stalled pack read ([`is_expert_stall`]).
    /// **The turns keep their place.** Either way nothing was committed — a
    /// segment that ran before the failed one is taken back by the step bracket
    /// (`decode_forward_cobatched`), and a faulted segment is rolled back as a
    /// failed sweep — and the next step re-forwards the same last token
    /// (`DecodeState::mark_forwarded`'s benign direction). Ending the turn
    /// instead threw away a reply mid-thought over ground that relief, a
    /// compaction or the weight side's between-wave concession hands back a step
    /// or two later — the weight side refuses to move only while a wave or an
    /// expert invocation is in flight — or over one slow read, which the next
    /// attempt waits for afresh. The step waits out the backoff ([`WaveRetry`])
    /// and runs again; a run of failures long enough to give up fails the turns
    /// with the error.
    pub(super) fn retry_step(&mut self, seq_ids: &[SequenceId], what: &str, err: &candle::Error) {
        match self.decode_retry.failed() {
            Next::Retry { wait, attempt } => {
                if WaveRetry::logs(attempt) {
                    tracing::warn!(
                        attempt,
                        wait_us = wait.as_micros() as u64,
                        "{what} did not run; the turns keep their place and it runs again: {err}"
                    );
                }
            }
            Next::GiveUp { attempts } => {
                tracing::error!(attempts, "{what} failed on every attempt: {err}");
                self.fail_all_decodes(
                    seq_ids,
                    &format!("{what} could not run in {attempts} attempts: {err}"),
                );
            }
        }
    }

    /// A sequence whose forward has produced unusable logits for a run of
    /// [`DEGENERATE_TOKEN_RUN`] tokens: fail the turn so what it decoded over
    /// those logits is never sealed. Answers whether it failed it now — the
    /// caller then says whether its carried recurrent state is to blame
    /// ([`describe_recurrent_state`]), once the step has settled that state.
    pub(super) fn fail_degenerate(&mut self, id: SequenceId) -> bool {
        // Once per turn: the flag stays set on every step after the bar.
        if self.active_decodes.get(&id).is_none_or(|s| s.faulted) {
            return false;
        }
        if let Some(state) = self.active_decodes.get_mut(&id) {
            state.fail(ConversationError::Channel(format!(
                "degenerate decode: the forward produced unusable logits for {} tokens \
                 running; the turn is discarded",
                DEGENERATE_TOKEN_RUN
            )));
        }
        true
    }

    /// A dispatch refused for non-finite logits ([`is_nonfinite_refusal`]):
    /// say which step it was and the sequences it carried, then whether each
    /// one's carried recurrent state is non-finite and from which layer
    /// ([`describe_recurrent_state`]). A refusal is otherwise only its rows'
    /// positions in the dispatch — no sequence, no layer, nothing that says
    /// whether the fault was carried in or made by this wave. Fault path only:
    /// nothing runs when the step is well.
    pub(super) fn describe_nonfinite_step(
        &self,
        step: &str,
        seqs: &[SequenceId],
        e: &candle::Error,
    ) {
        if !is_nonfinite_refusal(e) {
            return;
        }
        let ids: Vec<usize> = seqs.iter().map(|s| s.0).collect();
        tracing::error!(
            target: "candle_conversation::eos",
            step,
            seqs = ?ids,
            "non-finite logits refused a dispatch: {e}"
        );
        for &id in seqs {
            describe_recurrent_state(self.model.as_ref(), id);
        }
    }

    /// Fail every listed decode: each caller gets the error, and what each turn
    /// decoded is discarded rather than sealed ([`DecodeState::fail`]).
    pub(in super::super) fn fail_all_decodes(&mut self, seq_ids: &[SequenceId], msg: &str) {
        for &id in seq_ids {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                state.fail(ConversationError::Channel(msg.to_string()));
            }
        }
    }
}

/// Whether a step's forward did not run for want of something that comes back —
/// a refused tier, a full span, an exhausted arena, an expert still on its way —
/// so the step runs again with the same rows rather than failing them.
pub(super) fn is_step_refusal(e: &candle::Error) -> bool {
    is_tier_refusal(e) || is_span_full(e) || is_expert_stall(e) || is_arena_exhausted(e)
}

/// Whether `e` is the device having no room for the work right now — a refused
/// tier, a full span, the KV budget — rather than a fault: ground that comes
/// back as turns finish and the KV side relieves.
pub(super) fn is_ground_refusal(e: &ConversationError) -> bool {
    is_ground_refusal_message(&e.to_string())
}

/// Report whether a sequence's persistent recurrent state is finite, per layer.
///
/// Called when a sequence first emits token 0 — the point at which its logits
/// have gone unusable but nothing yet says why. A NaN in the logits is the far
/// end of the whole stack and is equally consistent with corrupted persistent
/// state and with one bad wave of activations; this separates them, and where
/// it is the state, names the first layer holding it.
///
/// Reports rather than fails: this runs on a fault path to make the fault
/// legible, and a probe that could itself error would just replace one
/// unexplained failure with another.
pub(super) fn describe_recurrent_state(model: &(dyn ManagedBatchedModel + Send), seq: SequenceId) {
    let exported = match model.export_recurrent(seq.0) {
        Ok(Some((_, layers))) => layers,
        // A stack with no recurrent state answers `None`; that is itself the
        // answer for this model, so say so rather than staying silent.
        Ok(None) => {
            tracing::error!(
                target: "candle_conversation::eos",
                seq = seq.0,
                "degenerate decode: model carries no recurrent state to inspect"
            );
            return;
        }
        Err(e) => {
            tracing::error!(
                target: "candle_conversation::eos",
                seq = seq.0,
                "degenerate decode: recurrent state unreadable ({e})"
            );
            return;
        }
    };

    let non_finite = |bytes: &[u8]| -> usize {
        bytes
            .as_chunks::<{ std::mem::size_of::<f32>() }>()
            .0
            .iter()
            .filter(|c| !f32::from_le_bytes(**c).is_finite())
            .count()
    };
    let mut bad_layers = 0usize;
    let mut first: Option<(u32, usize, usize)> = None;
    for l in &exported {
        let s = non_finite(&l.state);
        let tail = non_finite(&l.conv_tail);
        if s + tail > 0 {
            bad_layers += 1;
            if first.is_none() {
                first = Some((l.layer_index, s, tail));
            }
        }
    }
    match first {
        Some((layer, s, tail)) => tracing::error!(
            target: "candle_conversation::eos",
            seq = seq.0,
            layers = exported.len(),
            bad_layers,
            first_bad_layer = layer,
            state_non_finite = s,
            conv_tail_non_finite = tail,
            "degenerate decode: PERSISTENT recurrent state is non-finite"
        ),
        None => tracing::error!(
            target: "candle_conversation::eos",
            seq = seq.0,
            layers = exported.len(),
            "degenerate decode: recurrent state is entirely FINITE — the fault is in \
             this wave's activations, not in carried state"
        ),
    }
}
