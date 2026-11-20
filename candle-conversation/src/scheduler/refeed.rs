//! The re-feed and the re-decode (`docs/reprojection_refeed.md`): a projection
//! that rebuilds a turn's slot takes the turn's last tokens back and reads them
//! again over it.
//!
//! A placed piece reaches the recurrent layers only through tokens forwarded
//! after it. So a rebuild leaves its turn's last tokens out — their index rows
//! dropped from the live tail, their K/V trimmed off the captured tail — and the
//! turn goes on from the cut:
//!
//! - **A decode decodes them again** (§2.6): the first of them is kept and is
//!   the next step's input, everything the turn wrote after it is dropped, and
//!   the step samples on over the rebuilt slot, so the reply can take another
//!   path from there. The caller never saw the dropped tokens: a turn streams
//!   only what no re-decode can reach (`DecodeState::stream_settled`).
//! - **Tokens the turn did not sample are forwarded again** — a stencil's
//!   run, a grammar's, a recorded reply's — as a row of the next wave ahead of
//!   the pending token, the way a stencil's static run rides it
//!   (`stencil_wave`); an opening prefills them again in its next stride.
//!
//! The attention layers then hold each position once, computed in the new
//! context. The recurrent state is not rewound: what it folded stays folded,
//! and what is forwarded or decoded next folds on top.
//!
//! The span is at most [`REFEED`] tokens, back to the newest index block
//! boundary within that reach, and never past the newest closed index page
//! (`ManagedBatchedModel::refeed_span`).

use super::stencil_wave::StencilRow;
use super::*;
use candle_nn::kv_cache::HELD_TAIL_TOKENS;

/// The most tokens a re-feed forwards again, or a re-decode takes back: the
/// recurrent layers' short span, and the tail the gap leaves native for it.
pub(super) const REFEED: usize = HELD_TAIL_TOKENS;

/// `opening` then `decoded`, less their last `n` tokens taken from the end.
pub(super) fn without_last<'a>(
    opening: &'a [u32],
    decoded: &'a [u32],
    n: usize,
) -> (&'a [u32], &'a [u32]) {
    match n.checked_sub(decoded.len()) {
        None => (opening, &decoded[..decoded.len() - n]),
        Some(rest) => (&opening[..opening.len().saturating_sub(rest)], &[]),
    }
}

/// The row a re-feed rides: the re-fed tokens, then what the turn has not
/// forwarded yet.
pub(super) fn refeed_row(refeed: &[u32], pending: &[u32]) -> Vec<u32> {
    let mut row = Vec::with_capacity(refeed.len() + pending.len());
    row.extend_from_slice(refeed);
    row.extend_from_slice(pending);
    row
}

/// Whether the tokens after `anchor` can be decoded again: every one of them
/// was sampled freely and none has been streamed, and none is a token whose
/// writing moved the turn's structure — a page break, the reasoning's close,
/// an end of turn — which `structural` names.
pub(super) fn redecodable(
    tokens: &[u32],
    anchor: usize,
    fixed_through: usize,
    streamed: usize,
    structural: impl Fn(u32) -> bool,
) -> bool {
    let from = anchor + 1;
    from <= tokens.len()
        && from >= fixed_through
        && from >= streamed
        && !tokens[from..].iter().any(|&t| structural(t))
}

impl Scheduler {
    /// The tokens `view_id`'s turn takes back after its rebuild: the model's
    /// re-feed span, when it lies inside what the turn itself has forwarded —
    /// its decoded tokens, or its opening's prefilled part — and nothing while
    /// a played run is still being written.
    pub(super) fn choose_refeed(&self, view_id: SequenceId) -> Result<usize, ConversationError> {
        if self.holds_stencil_row(view_id) {
            return Ok(0);
        }
        let forwarded = match self.active_decodes.get(&view_id) {
            Some(d) => d.forwarded_generated,
            None => match self.awaiting_opening(view_id) {
                Some(p) => p.offset,
                None => return Ok(0),
            },
        };
        let span = self
            .model
            .refeed_span(view_id.0, REFEED)
            .map_err(ConversationError::Model)?;
        Ok(if span <= forwarded { span } else { 0 })
    }

    /// Take back the last `n` forwarded tokens of `view_id`'s turn. A decode
    /// that sampled them decodes again from the first of them; one that did
    /// not forwards them again on its next wave; an opening prefills them again
    /// in its next stride.
    pub(super) fn hold_refeed(&mut self, view_id: SequenceId, n: usize) {
        if n == 0 {
            return;
        }
        if let Some(d) = self.active_decodes.get(&view_id) {
            let anchor = d.forwarded_generated - n;
            let free =
                d.stencil.is_none() && d.pending_mask.is_none() && d.recorded_reply.is_none();
            let structural = |t: u32| {
                self.page_break_tokens.contains(&t)
                    || Some(t) == self.think_close
                    || self.eos_tokens.contains(&t)
            };
            if free
                && redecodable(
                    &d.generated_tokens,
                    anchor,
                    d.fixed_through,
                    d.streamed,
                    structural,
                )
            {
                self.redecode(view_id, anchor);
                return;
            }
            let tokens = d.generated_tokens[anchor..d.forwarded_generated].to_vec();
            self.refeeds.insert(view_id, tokens);
            return;
        }
        if let Some(p) = self
            .active_prefills
            .iter_mut()
            .find(|p| p.work.sequence_id == view_id)
        {
            p.offset -= n;
        }
    }

    /// Decode `view_id`'s turn again from `anchor`, its sampler forgetting
    /// what the dropped tokens recorded.
    fn redecode(&mut self, view_id: SequenceId, anchor: usize) {
        let Some(d) = self.active_decodes.get_mut(&view_id) else {
            return;
        };
        let dropped = d.redecode_from(anchor);
        if let Some(s) = self.sampling_states.get_mut(&view_id) {
            s.forget(&dropped);
        }
        tracing::debug!(
            target: "candle_conversation::refeed",
            seq_id = view_id.0,
            anchor,
            dropped = dropped.len(),
            "re-decode: the turn decodes again from the cut",
        );
    }

    /// Put each held re-feed on the next wave: ahead of a run its stencil
    /// played this iteration, or as a row of its own ahead of the turn's
    /// pending token, which the token after it is then sampled from.
    pub(super) fn play_refeeds(&mut self) {
        for (id, tokens) in std::mem::take(&mut self.refeeds) {
            let Some(state) = self.active_decodes.get_mut(&id).filter(|s| !s.finished) else {
                continue;
            };
            if let Some(row) = self
                .stencil_rows
                .iter_mut()
                .find(|r| r.seq == id && r.piece == 0)
            {
                row.tokens.splice(0..0, tokens);
                continue;
            }
            let row = refeed_row(&tokens, state.pending_forward());
            state.mark_forwarded();
            self.stencil_rows.push(StencilRow::played(id, row, false));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::boundary_state;
    use super::*;

    /// **A re-feed takes its tokens from the end**: from the decoded tokens
    /// first, then from the opening when the decode holds fewer.
    #[test]
    fn the_last_tokens_come_off_the_end() {
        let opening = [1, 2, 3, 4];
        let decoded = [5, 6, 7];
        assert_eq!(
            without_last(&opening, &decoded, 2),
            (&opening[..], &decoded[..1])
        );
        assert_eq!(
            without_last(&opening, &decoded, 5),
            (&opening[..2], &[][..])
        );
        assert_eq!(
            without_last(&opening, &decoded, 0),
            (&opening[..], &decoded[..])
        );
    }

    /// **The row is the re-fed tokens, then the pending one**, so the token
    /// after the row is sampled from its last.
    #[test]
    fn a_refeed_row_ends_on_the_pending_token() {
        assert_eq!(refeed_row(&[7, 8, 9], &[10]), vec![7, 8, 9, 10]);
        assert_eq!(refeed_row(&[7, 8], &[]), vec![7, 8]);
    }

    /// The re-feed is the tail the gap leaves native.
    #[test]
    fn the_refeed_is_the_held_tail() {
        assert_eq!(REFEED, 8);
    }

    /// **Only freely sampled, unstreamed tokens are decoded again**, and none
    /// that moved the turn's structure.
    #[test]
    fn a_redecode_takes_back_only_free_unstreamed_tokens() {
        let tokens = [10, 11, 12, 13, 14, 15];
        let none = |_: u32| false;
        assert!(redecodable(&tokens, 2, 0, 0, none));
        // A token written by a grammar after the anchor.
        assert!(!redecodable(&tokens, 2, 4, 0, none));
        assert!(redecodable(&tokens, 2, 3, 0, none));
        // A token already sent to the caller.
        assert!(!redecodable(&tokens, 2, 0, 4, none));
        assert!(redecodable(&tokens, 2, 0, 3, none));
        // A page break, the reasoning's close or an end of turn after it.
        assert!(!redecodable(&tokens, 2, 0, 0, |t| t == 14));
        // The anchor itself is kept, so it may be structural.
        assert!(redecodable(&tokens, 2, 0, 0, |t| t == 12));
    }

    /// **The re-decode keeps the anchor as the next input and drops the
    /// rest**: the anchor is pending again, and the next token pushed follows
    /// it.
    #[test]
    fn a_redecode_keeps_the_anchor_pending_and_drops_what_followed() {
        let (mut s, _rx) = boundary_state();
        for t in [1, 2, 3, 4, 5, 6] {
            s.push_committed(t, None, &[]);
        }
        s.mark_forwarded();
        s.push_committed(7, None, &[]);
        s.last_projection_end = 7;
        assert_eq!(s.redecode_from(2), vec![4, 5, 6, 7]);
        assert_eq!(&s.generated_tokens[..], &[1, 2, 3]);
        assert_eq!(s.pending_forward(), &[3]);
        assert_eq!(s.last_projection_end, 3);
    }

    /// **The stream holds back what a re-decode can reach**: the newest
    /// [`REFEED`] tokens stay unsent until the turn's end, which sends them
    /// less a withheld end-of-turn token.
    #[test]
    fn the_stream_holds_back_the_reach_of_a_redecode() {
        let (mut s, rx) = boundary_state();
        for t in 1..=10 {
            s.push_committed(t, None, &[]);
            assert!(s.stream_settled());
        }
        let sent: Vec<u32> = rx
            .try_iter()
            .filter_map(|e| match e {
                TurnEvent::Token(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(sent, vec![1, 2]);
        s.push_committed(99, None, &[]);
        s.withholds_end = true;
        assert!(s.stream_rest());
        let rest: Vec<u32> = rx
            .try_iter()
            .filter_map(|e| match e {
                TurnEvent::Token(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(rest, vec![3, 4, 5, 6, 7, 8, 9, 10]);
    }

    /// **A token a grammar wrote is fixed**: the re-decode's reach stops after
    /// it.
    #[test]
    fn a_static_token_is_never_decoded_again() {
        let (mut s, _rx) = boundary_state();
        s.push_committed(1, None, &[]);
        s.push_forwarded(2, None, &[]);
        s.push_committed(3, None, &[]);
        assert_eq!(s.fixed_through, 2);
    }
}
