//! The re-feed (`docs/reprojection_refeed.md`): a projection that rebuilds a
//! turn's slot ends by forwarding the turn's last tokens again over it.
//!
//! A placed piece reaches the recurrent layers only through tokens forwarded
//! after it. So a rebuild leaves its turn's last tokens out — their index rows
//! dropped from the live tail, their K/V trimmed off the captured tail — and
//! the turn forwards them again before it samples on: a decode as a row of its
//! next wave, ahead of its pending token, as a stencil's static run rides it
//! (`stencil_wave`); an opening by prefilling them again in its next stride.
//! The attention layers then hold each position once, computed in the new
//! context. The recurrent state is not rewound: the re-fed tokens fold into it
//! on top of what it holds, the delta rule overwriting what their keys held.
//!
//! The span is at most [`REFEED`] tokens, back to the newest index block
//! boundary within that reach, and never past the newest closed index page
//! (`ManagedBatchedModel::refeed_span`).

use super::stencil_wave::StencilRow;
use super::*;
use candle_nn::kv_cache::HELD_TAIL_TOKENS;

/// The most tokens a re-feed forwards again: the recurrent layers' short span,
/// and the tail the gap leaves native for it.
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

impl Scheduler {
    /// The tokens `view_id`'s turn forwards again after its rebuild: the
    /// model's re-feed span, when it lies inside what the turn itself has
    /// forwarded — its decoded tokens, or its opening's prefilled part — and
    /// nothing while a played run is still being written.
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

    /// Hold `n` tokens of `view_id`'s turn to be forwarded again: a decode's
    /// last `n` forwarded tokens ride its next wave; an opening prefills them
    /// again in its next stride.
    pub(super) fn hold_refeed(&mut self, view_id: SequenceId, n: usize) {
        if n == 0 {
            return;
        }
        if let Some(d) = self.active_decodes.get(&view_id) {
            let to = d.forwarded_generated;
            let tokens = d.generated_tokens[to - n..to].to_vec();
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
}
