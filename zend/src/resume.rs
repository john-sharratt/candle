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
//! # Every round of a request is resumable
//!
//! A tool round's input — its call turn's results — is named by the resume
//! entry. A request's **first** round has no call turn behind it: its input is
//! the user's message, which reaches the substrate only when the round seals.
//! So the message is written down as the **pending message** before the round
//! is submitted and cleared in the commit of its seal; a turn cut off in its
//! first round resumes as that message asked again. A request whose final
//! response sealed carries neither entry and is never run again
//! (`docs/turn_handover.md` §3).
//!
//! # Handed on by a member that left
//!
//! A member that leaves stops its live turns where they stand and marks each
//! conversation **handed**, so the member that takes it resumes it at once
//! rather than waiting for its own restart (`docs/turn_handover.md` §4, §5).
//!
//! # Summary turns are not answers
//!
//! "A turn arrived after the call turn" cannot be asked as `turn_count >
//! call_turn + 1`. The async summariser seals `SummaryOfTurns` /
//! `SummaryOfSummaries` nodes into the same index space, so the turn sitting at
//! `call_turn + 1` may be a summary rather than the tool response — and reading
//! it as one would tombstone a round whose tools never ran, losing it silently.
//! [`answered`] skips summary nodes for that reason.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, RecoveredMessage, Role};

/// The conversations this member is resuming now. A second resume of one —
/// the boot scan's and the handed sweep's — would run its round twice: the
/// claim is this member's for both, so only this set keeps them apart.
#[derive(Default)]
pub struct Resuming {
    busy: Arc<Mutex<HashSet<TimelineId>>>,
}

/// One conversation being resumed here, until dropped.
pub struct Busy {
    busy: Arc<Mutex<HashSet<TimelineId>>>,
    timeline: TimelineId,
}

impl Resuming {
    /// Begin resuming `timeline` — `None` while this member already is.
    pub fn begin(&self, timeline: TimelineId) -> Option<Busy> {
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        busy.insert(timeline).then(|| Busy {
            busy: Arc::clone(&self.busy),
            timeline,
        })
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        self.busy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.timeline);
    }
}

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
        .conversation_metadata_value(timeline, CALL_TURN_KEY)?
        .parse()
        .ok()
}

/// The conversation-metadata key holding a request's first round's message
/// until that round seals ([`mark_message`]).
const MESSAGE_KEY: &str = "resume_message";

/// The conversation-metadata key a leaving member marks a conversation with,
/// the time it stopped its turn in Unix ms ([`mark_handed`]).
const HANDED_KEY: &str = "resume_handed";

/// Record `message` as the pending first round of the request `timeline`
/// is about to answer. Written before the round is submitted.
pub fn mark_message(engine: &ConversationEngine, timeline: TimelineId, message: &str) {
    if let Err(e) = engine.set_conversation_metadata(timeline, MESSAGE_KEY, message) {
        tracing::warn!("resume: could not record the pending message: {e}");
    }
}

/// Retract the pending message — its round sealed. Called before the
/// round's commit, so one commit makes the round and the retraction durable.
pub fn clear_message(engine: &ConversationEngine, timeline: TimelineId) {
    if let Err(e) = engine.set_conversation_metadata(timeline, MESSAGE_KEY, "") {
        tracing::warn!("resume: could not clear the pending message: {e}");
    }
}

/// The pending message of `timeline`'s request, if a first round is
/// unsealed.
pub fn marked_message(engine: &ConversationEngine, timeline: TimelineId) -> Option<String> {
    engine
        .conversation_metadata_value(timeline, MESSAGE_KEY)
        .filter(|m| !m.is_empty())
}

/// Mark `timeline` handed on: its turn was stopped at `at_ms` by a member
/// leaving, and the member that takes it resumes it.
pub fn mark_handed(engine: &ConversationEngine, timeline: TimelineId, at_ms: u64) {
    if let Err(e) = engine.set_conversation_metadata(timeline, HANDED_KEY, &at_ms.to_string()) {
        tracing::warn!("resume: could not mark the conversation handed: {e}");
    }
}

/// Retract the handed mark — the turn was resumed, or needs nothing.
pub fn clear_handed(engine: &ConversationEngine, timeline: TimelineId) {
    if let Err(e) = engine.set_conversation_metadata(timeline, HANDED_KEY, "") {
        tracing::warn!("resume: could not clear the handed mark: {e}");
    }
}

/// Whether a leaving member marked `timeline` handed on.
pub fn is_handed(engine: &ConversationEngine, timeline: TimelineId) -> bool {
    engine
        .conversation_metadata_value(timeline, HANDED_KEY)
        .is_some_and(|at| !at.is_empty())
}

/// Every conversation a leaving member handed on.
pub fn handed(engine: &ConversationEngine) -> Vec<TimelineId> {
    engine
        .conversations_with_metadata_key(HANDED_KEY)
        .into_iter()
        .filter(|(_, at)| !at.is_empty())
        .map(|(timeline, _)| timeline)
        .collect()
}

/// What a conversation's request left unfinished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unfinished {
    /// A tool round: call turn `k`'s tools ran, or not, and no turn answers it.
    Round(u32),
    /// The request's first round: this message, asked and not yet answered.
    Message(String),
}

/// What is unfinished, from the entries and whether a call turn is answered.
/// A round outranks the message: the message's round sealed before any call
/// turn could be named. A call turn answered is an entry that outlived its
/// round, and names nothing.
fn unfinished_of(
    call_turn: Option<u32>,
    answered: impl Fn(u32) -> bool,
    message: Option<String>,
) -> Option<Unfinished> {
    match call_turn {
        Some(k) if !answered(k) => Some(Unfinished::Round(k)),
        _ => message.map(Unfinished::Message),
    }
}

/// [`unfinished_of`], asked of the substrate.
pub fn unfinished(engine: &ConversationEngine, timeline: TimelineId) -> Option<Unfinished> {
    unfinished_of(
        marked_call_turn(engine, timeline),
        |k| round_answered(engine, timeline, k),
        marked_message(engine, timeline),
    )
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

/// Whether a conversation is held whole enough to resume over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wholeness {
    /// Every turn's content is here.
    Whole,
    /// A gap is still arriving — another member's turn catch-up will fill.
    Arriving,
    /// A turn is empty that no catch-up will fill: this member's own, its K/V
    /// lost with a process that died before writing it, or one retired in
    /// place. Resumed over, the round would answer as if it had never been
    /// said, so the request ends instead.
    Lost,
}

/// [`Wholeness`] from whether every turn holds its content and whether a gap
/// is among those that do not.
pub fn wholeness(whole: bool, gap: bool) -> Wholeness {
    match (whole, gap) {
        (true, _) => Wholeness::Whole,
        (false, true) => Wholeness::Arriving,
        (false, false) => Wholeness::Lost,
    }
}

/// [`wholeness`], asked of the substrate.
pub fn wholeness_of(engine: &ConversationEngine, timeline: TimelineId) -> Wholeness {
    wholeness(engine.holds_whole(timeline), engine.holds_gap(timeline))
}

/// End `timeline`'s unfinished request without running it: its entries —
/// the call turn, the pending message, a handed mark — are retracted, so no
/// member resumes it, and the conversation stands at its last sealed round
/// for the next message.
pub fn abandon(engine: &ConversationEngine, timeline: TimelineId) {
    clear_call_turn(engine, timeline);
    clear_message(engine, timeline);
    clear_handed(engine, timeline);
}

/// The message that opened the conversation's latest request: its last user
/// message that carries no tool results. `None` for a conversation with none.
pub fn opening_message(history: &[RecoveredMessage]) -> Option<String> {
    history
        .iter()
        .rev()
        .find(|m| m.role == Role::User && m.tool_tokens.is_empty() && !m.text.is_empty())
        .map(|m| m.text.clone())
}

/// The opening message of a request cut off mid-round with no entry left to
/// resume it by: the conversation's last message is the assistant's and made
/// tool calls (`made_calls`), or is a tool round's results — either way no
/// answer follows. `None` for a request that ended — its last answer called
/// nothing — or one with no opening message.
pub fn cut_off_request(
    history: &[RecoveredMessage],
    made_calls: impl Fn(&str) -> bool,
) -> Option<String> {
    let last = history.last()?;
    let cut = match last.role {
        Role::Assistant => made_calls(&last.text),
        Role::User => !last.tool_tokens.is_empty(),
        Role::System => false,
    };
    cut.then(|| opening_message(history)).flatten()
}

/// Ask `timeline`'s request again from its opening `message`: the pending
/// message is recorded and the call turn retracted, so the resume runs the
/// request as a first round, over the turns that are whole.
pub fn reask(engine: &ConversationEngine, timeline: TimelineId, message: &str) {
    mark_message(engine, timeline, message);
    clear_call_turn(engine, timeline);
}

#[cfg(test)]
mod tests {
    use super::{
        answered, cut_off_request, opening_message, unfinished_of, wholeness, Resuming, Unfinished,
        Wholeness,
    };
    use candle_conversation::projection::TimelineId;
    use candle_conversation::{RecoveredMessage, Role};

    fn msg(role: Role, text: &str, tool_results: bool) -> RecoveredMessage {
        RecoveredMessage {
            role,
            text: text.into(),
            turn: 0,
            no_think: false,
            thinking: None,
            tool_tokens: if tool_results { vec![10] } else { Vec::new() },
        }
    }

    /// **The opening message is the last one the user wrote**, never a tool
    /// round's results, which reach the model as a user message too.
    #[test]
    fn the_opening_message_is_the_users_not_a_tool_rounds() {
        let history = vec![
            msg(Role::User, "first question", false),
            msg(Role::Assistant, "answer", false),
            msg(Role::User, "Give me a tour", false),
            msg(Role::Assistant, "<tool_call>read</tool_call>", false),
            msg(Role::User, "<tool_response>…</tool_response>", true),
            msg(Role::Assistant, "<tool_call>list</tool_call>", false),
        ];
        assert_eq!(opening_message(&history).as_deref(), Some("Give me a tour"));
        assert_eq!(opening_message(&[]), None);
    }

    /// **A request whose last answer made calls no turn answers is cut off;
    /// one whose last answer called nothing ended**, and is never asked twice.
    #[test]
    fn a_request_is_cut_off_only_while_its_calls_are_unanswered() {
        let calls = |t: &str| t.contains("<tool_call>");
        let mut history = vec![
            msg(Role::User, "Give me a tour", false),
            msg(Role::Assistant, "<tool_call>read</tool_call>", false),
            msg(Role::User, "<tool_response>…</tool_response>", true),
            msg(Role::Assistant, "<tool_call>list</tool_call>", false),
        ];
        assert_eq!(
            cut_off_request(&history, calls).as_deref(),
            Some("Give me a tour")
        );
        history.push(msg(Role::User, "<tool_response>…</tool_response>", true));
        assert_eq!(
            cut_off_request(&history, calls).as_deref(),
            Some("Give me a tour"),
            "its results landed and no answer followed"
        );
        history.push(msg(Role::Assistant, "Here is the tour.", false));
        assert_eq!(cut_off_request(&history, calls), None, "the request ended");
    }

    /// **Only a gap is waited for.** A conversation whole resumes; one with a
    /// gap waits for catch-up; one with an empty turn and no gap holds a hole
    /// nothing will fill, and its request ends.
    #[test]
    fn only_a_gap_is_waited_for() {
        assert_eq!(wholeness(true, false), Wholeness::Whole);
        assert_eq!(wholeness(false, true), Wholeness::Arriving);
        assert_eq!(wholeness(false, false), Wholeness::Lost);
    }

    /// One conversation is resumed once at a time on a member, and free again
    /// once its resume ends.
    #[test]
    fn a_conversation_is_resumed_once_at_a_time() {
        let r = Resuming::default();
        let tl = TimelineId::from_raw(7).unwrap();
        let first = r.begin(tl).expect("free");
        assert!(r.begin(tl).is_none(), "already being resumed");
        assert!(r.begin(TimelineId::from_raw(8).unwrap()).is_some());
        drop(first);
        assert!(r.begin(tl).is_some(), "free again");
    }

    /// A tool round whose call turn no turn answers is what is unfinished,
    /// whatever message is pending.
    #[test]
    fn an_unanswered_round_is_unfinished() {
        assert_eq!(
            unfinished_of(Some(4), |_| false, None),
            Some(Unfinished::Round(4))
        );
        assert_eq!(
            unfinished_of(Some(4), |_| false, Some("hi".into())),
            Some(Unfinished::Round(4))
        );
    }

    /// A first round cut off before it sealed leaves only its message.
    #[test]
    fn a_pending_message_is_unfinished_with_no_round() {
        assert_eq!(
            unfinished_of(None, |_| false, Some("where is it?".into())),
            Some(Unfinished::Message("where is it?".into()))
        );
    }

    /// **A request whose final response sealed is finished**: its call turn
    /// is answered and its message cleared, so nothing runs again.
    #[test]
    fn a_finished_request_is_not_unfinished() {
        assert_eq!(unfinished_of(Some(4), |_| true, None), None);
        assert_eq!(unfinished_of(None, |_| true, None), None);
    }

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
