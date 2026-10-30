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
//! A played run is a [`StencilRow`] until a wave writes it. A wave that does not
//! run — refused ground, a step that retries — leaves the row standing for the
//! next one, so the run's tokens, already recorded and streamed, are never
//! sealed without their K/V behind them.
//!
//! Only what the wave can carry rides it: a sequence named once, by its last
//! piece. A run that crosses a page break forwards its pieces up to the last
//! break ahead of the wave, where each break closes its page, and so does any
//! part of a run longer than one forward's budget ([`tail_start`]).

use std::collections::{HashMap, HashSet};

use candle::{Device, Tensor};
use candle_transformers::models::held_tensor::hold_row;

use super::{Scheduler, SequenceId};

/// One sequence's played run, waiting for a wave to write it.
pub(super) struct StencilRow {
    pub seq: SequenceId,
    /// `[pending] ++ run`, past whatever of it was forwarded ahead of the wave.
    pub tokens: Vec<u32>,
    /// The run closes the turn. Its EOS is committed and never forwarded, and
    /// the turn finishes once the row is written — not before, or the seal
    /// would take tokens whose K/V no forward has written.
    pub ends_turn: bool,
}

/// The rows one decode step carries, and what its wave did with them.
#[derive(Default)]
pub(super) struct StencilWave {
    pub rows: Vec<StencilRow>,
    /// The continuing rows' logits, copied off the forward's span. `Some` once
    /// the wave wrote every row; `None` leaves them for the next step.
    written: Option<HashMap<SequenceId, Tensor>>,
}

impl StencilWave {
    pub fn seqs(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.seq.0).collect()
    }

    pub fn ids(&self) -> Vec<SequenceId> {
        self.rows.iter().map(|r| r.seq).collect()
    }

    /// Each row's ids, `[1, len]` on the host: the model lays a wave's ids on
    /// its forward span in one upload.
    pub fn inputs(&self) -> candle::Result<Vec<Tensor>> {
        self.rows
            .iter()
            .map(|r| Tensor::new(r.tokens.as_slice(), &Device::Cpu)?.unsqueeze(0))
            .collect()
    }
}

/// Where a played run splits: the start of the tail that rides the wave. What
/// comes before it is forwarded ahead of the wave.
///
/// The tail is the run's last page piece — everything after the last break
/// token that has something after it, the cut [`super::prefill`]'s
/// `break_pieces` makes — no longer than `pass` tokens. A wave carries a
/// sequence once, so a piece before a break cannot ride it, and the break's
/// page closes between the two.
pub(super) fn tail_start(tokens: &[u32], breaks: &[u32], pass: usize) -> usize {
    let last_piece = tokens[..tokens.len().saturating_sub(1)]
        .iter()
        .rposition(|t| breaks.contains(t))
        .map_or(0, |i| i + 1);
    last_piece.max(tokens.len().saturating_sub(pass.max(1)))
}

impl Scheduler {
    /// Whether `id` holds a played run no wave has written yet.
    pub(super) fn holds_stencil_row(&self, id: SequenceId) -> bool {
        self.stencil_rows.iter().any(|r| r.seq == id)
    }

    /// The tokens `id`'s next step writes when it holds a played run.
    pub(super) fn stencil_row_len(&self, id: SequenceId) -> Option<usize> {
        self.stencil_rows
            .iter()
            .find(|r| r.seq == id)
            .map(|r| r.tokens.len())
    }

    /// Take the rows this step's wave can carry.
    ///
    /// A row whose turn ended while it waited — failed, or gone — is owed
    /// nothing and is dropped. A row whose slot reprojects this wave
    /// (`glue_pending`) stays for the next: the wave lists the slot as glue, and
    /// a slot listed twice desyncs every later member's varlen metadata.
    pub(super) fn take_stencil_rows(&mut self, glue_pending: &HashSet<usize>) -> Vec<StencilRow> {
        let mut take = Vec::new();
        for row in std::mem::take(&mut self.stencil_rows) {
            match self.active_decodes.get(&row.seq) {
                None => {}
                Some(s) if s.finished => {}
                Some(_) if glue_pending.contains(&row.seq.0) => self.stencil_rows.push(row),
                Some(_) => take.push(row),
            }
        }
        take
    }

    /// The wave wrote `wave`'s rows; `logits` are their scored rows, in order.
    /// Each sequence moves past what was written, and each continuing row's
    /// logits are copied off the span, which the next forward reclaims.
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
        let mut kept = HashMap::new();
        for (row, logit) in wave.rows.iter().zip(logits) {
            if let Err(e) = self.session.advance_sequence(row.seq.0, row.tokens.len()) {
                self.fail_all_decodes(&[row.seq], &format!("static run advance failed: {e}"));
                continue;
            }
            Self::record_slot_tokens(&mut self.slot_tokens, row.seq, &row.tokens);
            if row.ends_turn {
                continue;
            }
            match hold_row(logit) {
                Ok(t) => {
                    kept.insert(row.seq, t);
                }
                Err(e) => {
                    self.fail_all_decodes(&[row.seq], &format!("static run row: {e}"));
                }
            }
        }
        wave.written = Some(kept);
    }

    /// Settle a step's rows once its wave's span is released: a written run that
    /// closed its turn finishes it, and every other written run's next token is
    /// sampled from its row. Rows the wave did not write wait for the next step.
    pub(super) fn settle_stencil_wave(&mut self, wave: StencilWave) {
        let StencilWave { rows, written } = wave;
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

    #[test]
    fn a_run_without_a_break_rides_whole() {
        assert_eq!(tail_start(&[1, 2, 3], &[BREAK], 64), 0);
    }

    #[test]
    fn a_trailing_break_makes_no_cut() {
        assert_eq!(tail_start(&[1, 2, BREAK], &[BREAK], 64), 0);
    }

    #[test]
    fn the_tail_starts_after_the_last_break_with_something_after_it() {
        assert_eq!(tail_start(&[1, BREAK, 2, BREAK, 3, 4], &[BREAK], 64), 4);
        assert_eq!(tail_start(&[BREAK, 5], &[BREAK], 64), 1);
    }

    #[test]
    fn a_model_without_pages_never_cuts() {
        assert_eq!(tail_start(&[1, BREAK, 2], &[], 64), 0);
    }

    #[test]
    fn the_tail_is_no_longer_than_one_pass() {
        assert_eq!(tail_start(&[1, 2, 3, 4, 5], &[], 2), 3);
        assert_eq!(tail_start(&[1, BREAK, 2, 3, 4, 5], &[BREAK], 3), 3);
        assert_eq!(tail_start(&[1, BREAK, 2, 3], &[BREAK], 3), 2);
    }

    #[test]
    fn a_one_token_run_rides_whole() {
        assert_eq!(tail_start(&[BREAK], &[BREAK], 1), 0);
        assert_eq!(tail_start(&[7], &[], 0), 0);
    }
}
