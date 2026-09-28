//! Did a turn chain finish what it started?
//!
//! An ingest conversation — a file read, a directory summary — is shaped as a
//! tool round-trip: turn 0 asks and calls the tool, turn 1 carries the tool's
//! response and the answer. A chain can stop part-way, and **neither way leaves
//! the log corrupt**, which is what makes this module necessary: every record is
//! CRC-clean and every chunk count consistent, so `validate` reports the
//! substrate healthy while the chain sitting in it produced nothing.
//!
//! Two ways it stops:
//!
//! - **The response never arrived.** A [`TurnCoupling`] is written *before* the
//!   turn it points at, so `from_turn = n` promises a turn `n + 1`. When that
//!   turn was never sealed, the promise is the evidence
//!   ([`ChainBreak::ResponseMissing`]).
//! - **The turn produced no answer.** The decode budget ran out while the model
//!   was still reasoning, so the turn holds an unterminated `<think>` and
//!   nothing else: no tool call, hence no coupling, hence no next turn, hence no
//!   dangling promise to find ([`ChainBreak::NoAnswer`]). This is the one a
//!   coupling check alone cannot see.
//!
//! # Why this is judged on the ANSWER and not on the reasoning
//!
//! The obvious test — "the turn reasoned by design but holds no reasoning
//! segment" — reads the turn's shape, and the shape is dialect-dependent: one
//! family suppresses thinking with a `/no_think` marker the layout records
//! ([`TurnLayout::no_think`]), another by prefilling an already-closed
//! `<think></think>` block that is stripped from the body and leaves no marker
//! at all. On the second family a perfectly good turn has neither a marker nor a
//! reasoning segment, and that test would condemn every one of them.
//!
//! Asking what the turn *produced* holds on both. [`strip_think_blocks`] treats
//! an unterminated block as running to the end of the text, so a turn cut off
//! mid-thought strips to nothing, while a turn that reached its answer keeps it
//! whatever the dialect did with its reasoning.
//!
//! # It errs toward "finished"
//!
//! A chain being written *right now* can look unfinished for an instant — the
//! coupling lands before the response turn does. The caller's protection is that
//! an ingest pass snapshots its "already done" set once at the start and passes
//! are serialized, so no pass observes another's half-written chain. Where this
//! is uncertain it answers `None`: a missed break costs one stale chain, while a
//! false break costs re-ingesting a file that was fine, and re-ingesting in a
//! loop is the worse failure.
//!
//! [`TurnCoupling`]: crate::persistence::record::TurnCouplingPayload
//! [`TurnLayout::no_think`]: crate::turn_layout::TurnLayout::no_think

use crate::projection::{TimelineId, TurnIndex};
use crate::substrate::Substrate;
use crate::think_strip::strip_think_blocks;

/// How a chain stopped before it produced anything usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainBreak {
    /// A coupling promises the tool response for `after_turn` as the next turn,
    /// and that turn was never sealed.
    ResponseMissing { after_turn: u32 },
    /// The last turn emitted reasoning and never reached an answer — the decode
    /// was cut off mid-thought.
    NoAnswer { turn: u32 },
}

/// Whether an assistant body reached an answer: `false` when everything it
/// emitted was reasoning, or when it emitted nothing at all.
///
/// An unterminated `<think>` strips to the empty string (see
/// [`strip_think_blocks`]), which is exactly the cut-off case.
pub fn produced_no_answer(assistant_text: &str) -> bool {
    strip_think_blocks(assistant_text).is_empty()
}

/// The break in `timeline`'s chain, or `None` when it finished.
///
/// `None` for a timeline with no turns at all — there is no chain to be broken,
/// and a caller that has not ingested anything is not recovering from anything.
pub fn chain_break(substrate: &Substrate, timeline: TimelineId) -> Option<ChainBreak> {
    let last = substrate.turn_indices(timeline).map(|i| i.0).max()?;
    // A coupling naming the last turn (or beyond) promises a response that is
    // not there. Checked first: it is the cheaper question and the sharper
    // evidence, since the promise was recorded by the writer itself.
    if substrate
        .couplings_of(timeline)
        .iter()
        .any(|&from_turn| from_turn >= last)
    {
        return Some(ChainBreak::ResponseMissing { after_turn: last });
    }
    if produced_no_answer(&substrate.assistant_text_of(timeline, TurnIndex(last))) {
        return Some(ChainBreak::NoAnswer { turn: last });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cut-off turn this module exists for: the whole body is one
    /// unterminated `<think>`, ending mid-word, with no tool call after it.
    /// Taken from the shape of the README ingest that poisoned first turns.
    #[test]
    fn an_unterminated_think_block_produced_no_answer() {
        let cut_off = "<think>\nWe need answer user: read entire README.md. \
                       Could call 1-200 and 201-400 and 401-60";
        assert!(produced_no_answer(cut_off));
    }

    /// A turn that reasoned AND answered is finished, and so is one that
    /// answered without reasoning at all (a `no_think` ingest turn).
    #[test]
    fn a_turn_that_reached_its_answer_is_finished() {
        assert!(!produced_no_answer(
            "<think>weigh it up</think>The file documents the workspace layout."
        ));
        assert!(!produced_no_answer(
            "The file documents the workspace layout."
        ));
    }

    /// A body that is only a closed, EMPTY reasoning block — the prefilled
    /// suppression scaffolding — produced no answer, which is the truth: such a
    /// turn decoded nothing after its block.
    #[test]
    fn an_empty_closed_block_alone_produced_no_answer() {
        assert!(produced_no_answer("<think></think>"));
        assert!(produced_no_answer(""));
        assert!(produced_no_answer("   \n  "));
    }

    /// A turn whose visible output is a tool call counts as produced: the chain
    /// is mid-round, not cut off, and the dangling-coupling arm is what judges
    /// whether its response arrived.
    #[test]
    fn a_tool_call_counts_as_produced() {
        assert!(!produced_no_answer(
            "<think>read it</think><tool_call>\n{\"name\": \"file_read\"}\n</tool_call>"
        ));
    }

    /// Case-insensitive, because the stripper is: a capitalised unterminated
    /// tag must not read as an answer.
    #[test]
    fn an_unterminated_block_is_matched_case_insensitively() {
        assert!(produced_no_answer("<THINK>still deciding"));
    }
}
