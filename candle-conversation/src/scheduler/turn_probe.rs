//! The probe a boundary scores its turn by: the turn's own folded wide-Q, as
//! the reprojection and a request's first reading both read it
//! (`docs/hit_driven_projection.md` §7.4).
//!
//! The probe covers the turn's content only — from the block the turn began
//! at to the slot's end — never the projected prefix ahead of it, the same
//! domain the seal gathers for the belief gallery. A turn longer than the
//! policy's `max_probe_tokens` is probed by its head (the question, the
//! strongest intent signal) and its most recent chunks.

use super::*;

/// A turn's probe, and its pinned question window — the user half, trimmed to
/// the token boundary captured at turn open.
pub(super) struct TurnProbe {
    pub(super) probe: Vec<WideQSig>,
    pub(super) question: Vec<WideQSig>,
}

impl Scheduler {
    /// `view_id`'s probe, its turn beginning at block `turn_start` with
    /// `question_tokens` of user half, under a `max_probe_tokens` cap. `None`
    /// when the turn holds no block yet, or the gather covers no real token.
    pub(super) fn turn_probe(
        &self,
        view_id: SequenceId,
        turn_start: usize,
        question_tokens: usize,
        max_probe_tokens: usize,
    ) -> Option<TurnProbe> {
        let cur_chunks = self.session.sequence_block_count(view_id.0).unwrap_or(0);
        if cur_chunks <= turn_start {
            return None;
        }
        let cap = max_probe_tokens.max(1);
        let tail_lo = turn_start.max(cur_chunks.saturating_sub(cap.div_ceil(self.chunk_size)));
        let mut probe = Vec::new();
        if tail_lo > turn_start {
            let head_hi = (turn_start + QUERY_HEAD_CHUNKS).min(tail_lo);
            probe.extend(self.gather_wide_sigs(view_id, (turn_start, head_hi)));
        }
        probe.extend(self.gather_wide_sigs(view_id, (tail_lo, cur_chunks)));
        if probe.is_empty() {
            return None;
        }
        let q_tokens = question_tokens.min(cap);
        let q_hi = (turn_start + q_tokens.div_ceil(self.chunk_size)).min(cur_chunks);
        let mut question = self.gather_wide_sigs(view_id, (turn_start, q_hi));
        question.truncate(q_tokens);
        Some(TurnProbe { probe, question })
    }
}
