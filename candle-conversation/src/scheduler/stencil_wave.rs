//! A stencil's static runs, riding the decode step's wave.
//!
//! A static run `R` that follows a sequence's pending token `Y` is written as
//! `[Y] ++ R` by the decode step's own forward — a full-sweep member of the
//! wave's prefill slot, behind the verify blocks, scored at its last row — and
//! the token after it is sampled from that row. A run costs no forward of its
//! own. On a card that streams its experts a forward's price is mostly the
//! expert loads, and a separate pass for a handful of rows paid them a second
//! time for experts the decode rows beside it were about to load anyway.
//!
//! A played run is a [`StencilRow`] until waves have written it. A wave that
//! does not run — refused ground, a step that retries — leaves the row standing
//! for the next one, so the run's tokens, already recorded and streamed, are
//! never sealed without their K/V behind them.
//!
//! **A wave writes one piece of a row** ([`piece_len`]): at most the model's
//! short span (`ManagedBatchedModel::short_span_rows`), and never past a page
//! break with something after it. Within the short span a recurrent stack
//! writes the row into its timeline; past it the row would claim its
//! sequence's prefill scratch, a second copy of the whole recurrent state
//! (144 MiB a sequence on Flash-Next), which no cohort or admission has priced.
//! So a run longer than the span rides several steps, a piece each, and only
//! the piece that completes it is scored: the token after the run is sampled
//! from that row. A piece that ends on a break closes its page before the next
//! piece is written.

use std::collections::{HashMap, HashSet};

use candle::{Device, Tensor};
use candle_transformers::models::held_tensor::hold_row;

use super::{Scheduler, SequenceId};

/// One sequence's played run, waiting for waves to write it.
pub(super) struct StencilRow {
    pub seq: SequenceId,
    /// `[pending] ++ run`, past whatever of it waves have written already.
    pub tokens: Vec<u32>,
    /// The leading tokens of `tokens` the wave carrying the row writes — set
    /// when a step takes it ([`Scheduler::take_stencil_rows`]).
    pub piece: usize,
    /// The run closes the turn. Its EOS is committed and never forwarded, and
    /// the turn finishes once the row's last piece is written — not before, or
    /// the seal would take tokens whose K/V no forward has written.
    pub ends_turn: bool,
}

impl StencilRow {
    /// A played run, no piece of it taken yet.
    pub fn played(seq: SequenceId, tokens: Vec<u32>, ends_turn: bool) -> Self {
        Self {
            seq,
            tokens,
            piece: 0,
            ends_turn,
        }
    }

    /// The tokens the carrying wave writes.
    pub fn piece_tokens(&self) -> &[u32] {
        &self.tokens[..self.piece]
    }

    /// Whether the carrying wave writes the rest of the run.
    pub fn completes(&self) -> bool {
        self.piece == self.tokens.len()
    }
}

/// How many of a row's leading tokens one wave writes: at most `span`, and
/// never past a page break that has something after it, which closes its page
/// between this wave and the next. A break in the row's last slot ends no
/// piece: the token after the run closes that page (`commit_after_injection`).
pub(super) fn piece_len(tokens: &[u32], breaks: &[u32], span: usize) -> usize {
    let to_break = tokens[..tokens.len().saturating_sub(1)]
        .iter()
        .position(|t| breaks.contains(t))
        .map_or(tokens.len(), |i| i + 1);
    span.max(1).min(to_break)
}

/// The rows one decode step carries, and what its wave did with them.
#[derive(Default)]
pub(super) struct StencilWave {
    pub rows: Vec<StencilRow>,
    /// The completing rows' logits, copied off the forward's span. `Some` once
    /// the wave wrote every row's piece; `None` leaves the rows for the next
    /// step.
    written: Option<HashMap<SequenceId, Tensor>>,
    /// Rows the wave wrote a piece of, with the rest of their run still to go:
    /// they ride the next step, and nothing is sampled from them.
    continuing: Vec<StencilRow>,
    /// The pieces that ended on a page break, whose page closes once the wave's
    /// span is released and before the next piece is written.
    page_closes: Vec<(SequenceId, Vec<u32>)>,
}

impl StencilWave {
    pub fn seqs(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.seq.0).collect()
    }

    pub fn ids(&self) -> Vec<SequenceId> {
        self.rows.iter().map(|r| r.seq).collect()
    }

    /// Each row's piece, `[1, len]` on the host: the model lays a wave's ids on
    /// its forward span in one upload.
    pub fn inputs(&self) -> candle::Result<Vec<Tensor>> {
        self.rows
            .iter()
            .map(|r| Tensor::new(r.piece_tokens(), &Device::Cpu)?.unsqueeze(0))
            .collect()
    }
}

impl Scheduler {
    /// Whether `id` holds a played run no wave has finished writing.
    pub(super) fn holds_stencil_row(&self, id: SequenceId) -> bool {
        self.stencil_rows.iter().any(|r| r.seq == id)
    }

    /// The tokens `id`'s next step writes when it holds a played run: the next
    /// piece of it, as [`Self::take_stencil_rows`] will cut it.
    pub(super) fn stencil_row_len(&self, id: SequenceId) -> Option<usize> {
        let row = self.stencil_rows.iter().find(|r| r.seq == id)?;
        Some(piece_len(
            &row.tokens,
            self.prefill_breaks(),
            self.stencil_piece_span(),
        ))
    }

    /// The longest piece of a row one wave writes: the model's short span, or
    /// one forward's budget on a stack that has none.
    fn stencil_piece_span(&self) -> usize {
        self.model
            .short_span_rows()
            .unwrap_or_else(|| self.prefill_pass_budget())
    }

    /// Take the rows this step's wave can carry, each with the piece it writes
    /// ([`piece_len`]).
    ///
    /// A row whose turn ended while it waited — failed, or gone — is owed
    /// nothing and is dropped. A row whose slot reprojects this wave
    /// (`glue_pending`) stays for the next: the wave lists the slot as glue, and
    /// a slot listed twice desyncs every later member's varlen metadata.
    pub(super) fn take_stencil_rows(&mut self, glue_pending: &HashSet<usize>) -> Vec<StencilRow> {
        let span = self.stencil_piece_span();
        let breaks = self.prefill_breaks().to_vec();
        let mut take = Vec::new();
        for mut row in std::mem::take(&mut self.stencil_rows) {
            match self.active_decodes.get(&row.seq) {
                None => {}
                Some(s) if s.finished => {}
                Some(_) if glue_pending.contains(&row.seq.0) => self.stencil_rows.push(row),
                Some(_) => {
                    row.piece = piece_len(&row.tokens, &breaks, span);
                    take.push(row);
                }
            }
        }
        take
    }

    /// The wave wrote each of `wave`'s rows' pieces; `logits` are their scored
    /// rows, in order. Each sequence moves past its piece. A row whose piece
    /// completed its run has its logits copied off the span, which the next
    /// forward reclaims; one with more to write closes the page its piece ended
    /// on, if it did, and continues next step.
    pub(super) fn stencil_rows_written(&mut self, wave: &mut StencilWave, logits: &[Tensor]) {
        if logits.len() != wave.rows.len() {
            self.fail_all_decodes(
                &wave.ids(),
                &format!(
                    "the wave scored {} rows for {} static runs",
                    logits.len(),
                    wave.rows.len()
                ),
            );
            wave.rows.clear();
            return;
        }
        let breaks = self.prefill_breaks().to_vec();
        let mut kept = HashMap::new();
        let mut completed = Vec::with_capacity(wave.rows.len());
        for (mut row, logit) in std::mem::take(&mut wave.rows).into_iter().zip(logits) {
            if let Err(e) = self.session.advance_sequence(row.seq.0, row.piece) {
                self.fail_all_decodes(&[row.seq], &format!("static run advance failed: {e}"));
                continue;
            }
            Self::record_slot_tokens(&mut self.slot_tokens, row.seq, row.piece_tokens());
            if !row.completes() {
                // More of the run follows: the page its piece ended on closes
                // before the next piece is written, and nothing is sampled.
                if row
                    .piece_tokens()
                    .last()
                    .is_some_and(|t| breaks.contains(t))
                {
                    wave.page_closes
                        .push((row.seq, row.piece_tokens().to_vec()));
                }
                row.tokens.drain(..row.piece);
                row.piece = 0;
                wave.continuing.push(row);
                continue;
            }
            if !row.ends_turn {
                match hold_row(logit) {
                    Ok(t) => {
                        kept.insert(row.seq, t);
                    }
                    Err(e) => {
                        self.fail_all_decodes(&[row.seq], &format!("static run row: {e}"));
                    }
                }
            }
            completed.push(row);
        }
        wave.rows = completed;
        wave.written = Some(kept);
    }

    /// Settle a step's rows once its wave's span is released: a completed run
    /// that closed its turn finishes it, every other completed run's next token
    /// is sampled from its row, and a row with more of its run to write waits
    /// for the next step. Rows the wave did not write wait whole.
    pub(super) fn settle_stencil_wave(&mut self, wave: StencilWave) {
        let StencilWave {
            rows,
            written,
            continuing,
            page_closes,
        } = wave;
        for (seq, head) in page_closes {
            self.close_page_at_break(seq, &head);
        }
        self.stencil_rows.extend(continuing);
        let Some(logits) = written else {
            self.stencil_rows.extend(rows);
            return;
        };
        for row in rows.iter().filter(|r| r.ends_turn) {
            if let Some(s) = self.active_decodes.get_mut(&row.seq) {
                if !s.finished {
                    s.wrote_tool_call = true;
                    s.finished = true;
                }
            }
        }
        self.commit_after_injection(logits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BREAK: u32 = 9;

    /// The timeline's rows on the recurrent stacks.
    const SPAN: usize = 8;

    #[test]
    fn a_row_within_the_span_rides_whole() {
        assert_eq!(piece_len(&[1, 2, 3], &[BREAK], SPAN), 3);
        assert_eq!(piece_len(&[1, 2, 3, 4, 5, 6, 7, 8], &[], SPAN), 8);
    }

    #[test]
    fn a_row_past_the_span_writes_the_span() {
        assert_eq!(piece_len(&[1, 2, 3, 4, 5, 6, 7, 8, 10], &[], SPAN), 8);
    }

    #[test]
    fn a_piece_ends_on_a_break_with_something_after_it() {
        assert_eq!(piece_len(&[1, BREAK, 2, BREAK, 3, 4], &[BREAK], SPAN), 2);
        assert_eq!(piece_len(&[BREAK, 5], &[BREAK], SPAN), 1);
    }

    #[test]
    fn a_trailing_break_ends_no_piece() {
        assert_eq!(piece_len(&[1, 2, BREAK], &[BREAK], SPAN), 3);
        assert_eq!(piece_len(&[BREAK], &[BREAK], SPAN), 1);
    }

    #[test]
    fn a_model_without_pages_cuts_only_at_the_span() {
        assert_eq!(piece_len(&[1, BREAK, 2], &[], SPAN), 3);
        assert_eq!(piece_len(&[7], &[], 0), 1);
    }

    /// **A 25-token run plays as pieces of at most the span**, the last one
    /// completing it, each cut where `stencil_rows_written` cuts it.
    #[test]
    fn a_long_run_plays_in_pieces_and_completes_once() {
        let mut row = StencilRow::played(SequenceId(3), (100..125).collect(), false);
        let mut pieces = Vec::new();
        loop {
            row.piece = piece_len(&row.tokens, &[], SPAN);
            pieces.push(row.piece_tokens().to_vec());
            if row.completes() {
                break;
            }
            row.tokens.drain(..row.piece);
        }
        assert_eq!(
            pieces,
            vec![
                (100..108).collect::<Vec<u32>>(),
                (108..116).collect(),
                (116..124).collect(),
                vec![124],
            ]
        );
    }

    /// The row after a tool's name — its last token and `", "arguments": {"` —
    /// is nine rows on the live tokenizer (`zend/tests/stencil_tool_call.rs`,
    /// `the_rows_a_tool_call_plays_on_the_live_tokenizer`): eight, then one.
    #[test]
    fn the_nine_row_name_row_is_two_pieces() {
        let mut row = StencilRow::played(SequenceId(1), (1..=9).collect(), false);
        row.piece = piece_len(&row.tokens, &[BREAK], SPAN);
        assert_eq!(row.piece_tokens(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(!row.completes());
        row.tokens.drain(..row.piece);
        row.piece = piece_len(&row.tokens, &[BREAK], SPAN);
        assert_eq!(row.piece_tokens(), &[9]);
        assert!(row.completes());
    }
}
