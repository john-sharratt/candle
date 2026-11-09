//! A conversation recovered as bubbles, from a given turn on — what a
//! sequence's `recovered_history` reads, and what a reader that has covered
//! the turns before one reads without rebuilding them.

use crate::projection::{TimelineId, TurnIndex};
use crate::recovered_message::{tool_response_lengths, RecoveredMessage, TOOL_RESPONSE_OPEN};
use crate::substrate::Substrate;
use crate::turn::Role;

/// `timeline`'s turns from `from_turn` on, as bubbles: one per non-empty half
/// of each turn, in order — the user's message, then the assistant's reply.
/// Ghost summary turns are skipped unless `include_ghost_summaries`. A turn
/// whose reasoning K/V was dropped has its reasoning length `estimate`d from
/// the prose; a tool-response turn's blocks are measured from its sealed ids,
/// each opening with `response_open`.
pub fn recovered_from(
    read: &Substrate,
    timeline: TimelineId,
    include_ghost_summaries: bool,
    from_turn: u32,
    estimate: impl Fn(&str) -> u32,
    response_open: &[u32],
) -> Vec<RecoveredMessage> {
    let mut out: Vec<RecoveredMessage> = Vec::new();
    for idx in read.turn_indices(timeline).filter(|i| i.0 >= from_turn) {
        if !include_ghost_summaries
            && read
                .tree_meta_of(timeline, idx)
                .is_some_and(|m| m.kind.is_summary())
        {
            continue;
        }
        push_turn(read, timeline, idx, &estimate, response_open, &mut out);
    }
    out
}

/// Turn `idx`'s bubbles, onto `out`.
fn push_turn(
    read: &Substrate,
    timeline: TimelineId,
    idx: TurnIndex,
    estimate: &impl Fn(&str) -> u32,
    response_open: &[u32],
    out: &mut Vec<RecoveredMessage>,
) {
    let user_text = read.user_text_of(timeline, idx);
    let assistant_text = read.assistant_text_of(timeline, idx);
    let no_think = read.turn_no_think(timeline, idx);
    if !user_text.is_empty() {
        let tool_tokens = if user_text.contains(TOOL_RESPONSE_OPEN) {
            let body = read.user_body_ids(timeline, idx).unwrap_or_default();
            tool_response_lengths(body, response_open)
        } else {
            Vec::new()
        };
        out.push(RecoveredMessage {
            role: Role::User,
            text: user_text,
            turn: idx.0,
            no_think,
            thinking: None,
            tool_tokens,
        });
    }
    if !assistant_text.is_empty() {
        let thinking = read.thinking_length_of(timeline, idx, estimate);
        out.push(RecoveredMessage {
            role: Role::Assistant,
            text: assistant_text,
            turn: idx.0,
            no_think: false,
            thinking,
            tool_tokens: Vec::new(),
        });
    }
}
