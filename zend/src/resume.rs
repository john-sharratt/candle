//! Finishing a tool round the daemon died in the middle of.
//!
//! A turn that asks for tools becomes durable before those tools run: the call
//! turn seals, and only then is the dispatch made. A daemon that stops in
//! between leaves a sealed request whose results never existed anywhere but in
//! that process — so nobody outside it can finish the round, and the
//! conversation is stuck at a question it has visibly started answering.
//!
//! **Only the tool round is resumed.** A turn killed before its call turn
//! sealed ran nothing and recorded nothing worth keeping; whoever asked can ask
//! again, which is simpler than any machinery for it and leaves no half-turn
//! behind. The tool round is recovered precisely because it is the case that
//! cannot be recovered by hand.
//!
//! An assistant turn that asked for tools looks the same whether or not its
//! tools ever ran, so the fact has to be recorded: a **resume entry** is written
//! once the call turn is durable and before its tools are dispatched, naming
//! that turn, and cleared when the round ends.
//!
//! # The entry is a hint; the substrate is the authority
//!
//! The entry can outlive the round it describes — the process can die after the
//! round completes but before the clear reaches disk. So it is never acted on
//! by itself. On restart each entry is put to the substrate: if a turn arrived
//! after the call turn, the round finished and the entry is tombstoned;
//! otherwise it is dispatched. A stale entry therefore costs a lookup, never a
//! repeated round, and losing an entry to a crash costs one unresumed turn
//! rather than a wrong one.
//!
//! # Summary turns are not answers
//!
//! "A turn arrived after the call turn" cannot be asked as `turn_count >
//! call_turn + 1`. The async summariser seals `SummaryOfTurns` /
//! `SummaryOfSummaries` nodes into the same index space, so the turn sitting at
//! `call_turn + 1` may be a summary rather than the tool response — and reading
//! it as one would tombstone a round whose tools never ran, losing it silently.
//! [`answered`] skips summary nodes for that reason.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

/// The composer dials a resumed turn runs under when the conversation has
/// never recorded any — the same middles the GUI opens a new conversation at.
pub const DEFAULT_DIALS: (u8, u8, bool, u8) = (2, 2, true, 2);

/// The conversation-metadata key naming the call turn a resumed round finishes.
///
/// Its value is the call turn's own sealed index, in decimal. Cleared to the
/// empty string rather than removed — the bag is last-writer-wins by append, so
/// an empty value is how a key is retracted.
const CALL_TURN_KEY: &str = "resume_call_turn";

/// Record that `turn_index` is a sealed call turn whose tools are about to run.
///
/// Call this **before** the group-commit that makes the call turn durable, so
/// one commit covers both: an entry still sitting in the writer queue while the
/// tools run would be lost by the very crash it exists to survive.
pub fn mark_call_turn(engine: &ConversationEngine, timeline: TimelineId, turn_index: u32) {
    if let Err(e) =
        engine.set_conversation_metadata(timeline, CALL_TURN_KEY, &turn_index.to_string())
    {
        tracing::warn!(%turn_index, "resume: could not mark the call turn: {e}");
    }
}

/// Retract the entry — the round is over, however it ended.
pub fn clear_call_turn(engine: &ConversationEngine, timeline: TimelineId) {
    if let Err(e) = engine.set_conversation_metadata(timeline, CALL_TURN_KEY, "") {
        tracing::warn!("resume: could not clear the call turn: {e}");
    }
}

/// The call turn `timeline` has an entry for, if it has one that still names a
/// turn. An empty or unparseable value reads as no entry — the key is retracted
/// by writing empty, and a value that is not an index names nothing.
pub fn marked_call_turn(engine: &ConversationEngine, timeline: TimelineId) -> Option<u32> {
    engine
        .conversation_metadata(timeline)?
        .get(CALL_TURN_KEY)?
        .parse()
        .ok()
}

/// Whether a real turn was sealed after `call_turn`, given each turn as
/// `(index, is_summary)`.
///
/// Summary nodes do not count: they are the summariser's own, sealed into the
/// same index space, and one landing after a call turn says nothing about
/// whether that turn's tools ever produced a response.
fn answered(call_turn: u32, turns: impl Iterator<Item = (u32, bool)>) -> bool {
    turns
        .into_iter()
        .any(|(idx, is_summary)| idx > call_turn && !is_summary)
}

/// [`answered`], asked of the substrate.
pub fn round_answered(engine: &ConversationEngine, timeline: TimelineId, call_turn: u32) -> bool {
    let conv = engine.conversation();
    let view = conv.read();
    let turns: Vec<(u32, bool)> = view
        .turn_indices(timeline)
        .map(|idx| {
            let is_summary = view
                .tree_meta_of(timeline, idx)
                .is_some_and(|m| m.kind.is_summary());
            (idx.0, is_summary)
        })
        .collect();
    answered(call_turn, turns.into_iter())
}

#[cfg(test)]
mod tests {
    use super::answered;

    /// Nothing after the call turn: its tools never produced a response, so the
    /// round is the resume's to finish.
    #[test]
    fn a_call_turn_at_the_end_is_unanswered() {
        let turns = [(0, false), (1, false), (2, false)];
        assert!(!answered(2, turns.into_iter()));
    }

    /// The response turn is present, so the round completed and the entry is
    /// only an echo of it.
    #[test]
    fn a_turn_after_the_call_turn_answers_it() {
        let turns = [(0, false), (1, false), (2, false), (3, false)];
        assert!(answered(2, turns.into_iter()));
    }

    /// **The summariser does not answer a tool call.** A `SummaryOfTurns` node
    /// sealed after the call turn occupies the next index without being the
    /// response to anything — counting it would tombstone a round whose tools
    /// never ran, and the round would be lost with no trace.
    #[test]
    fn a_summary_sealed_after_the_call_turn_does_not_answer_it() {
        let turns = [(0, false), (1, false), (2, false), (3, true)];
        assert!(!answered(2, turns.into_iter()));
    }

    /// And when the summariser took the next index, the real response turn sits
    /// past it — still an answer.
    #[test]
    fn a_response_beyond_a_summary_still_answers() {
        let turns = [(2, false), (3, true), (4, false)];
        assert!(answered(2, turns.into_iter()));
    }

    /// Turns before the call turn are the conversation's history, not its
    /// answer.
    #[test]
    fn earlier_turns_do_not_answer_a_later_call() {
        let turns = [(0, false), (1, false), (5, false)];
        assert!(!answered(5, turns.into_iter()));
    }
}
