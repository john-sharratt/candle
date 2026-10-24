//! A dialogue's sealed turns as the parts a journal stretch is made of, and what
//! a claim citing each part rests on.
//!
//! A substrate turn holds both halves of one exchange — the user side and the
//! assistant's reply — and each half is cited on its own, because they support
//! a claim differently. Each half is a **part**, numbered from the turn's own
//! index so the numbering is stable across restarts and members:
//! `2·turn + 1` for the user side and `2·turn + 2` for the reply. Zero is no
//! part, so a journal that covers nothing covers part 0.
//!
//! | Half | Holds | A claim citing it is |
//! |---|---|---|
//! | user side, a person's message | what the user said | `Heard` |
//! | user side, `<tool_response>` blocks | what a tool returned | `Observed` |
//! | the reply | the call it made, or what it told the user | `Did` |
//!
//! The kind is read off which half it is and what the half carries, never off
//! what the words claim. A reply's reasoning block is left out: it is the model
//! thinking, not anything it did.

use candle_conversation::projection::TurnIndex;
use candle_conversation::{RecoveredMessage, Role};
use journal_core::entry::Kind;
use journal_core::span::SpanTurn;

/// The markup a tool result is returned in.
const TOOL_RESPONSE: &str = "<tool_response>";

/// The longest a part runs in the stretch a draft is shown, in characters. The
/// draft's fork already holds every turn in full; the stretch is there to
/// number them, and a tool result can run to thousands of lines.
pub const CLIP: usize = 400;

/// The part the user side of `turn` is cited as.
pub fn user_part(turn: u32) -> u64 {
    2 * u64::from(turn) + 1
}

/// The part the reply of `turn` is cited as.
pub fn reply_part(turn: u32) -> u64 {
    2 * u64::from(turn) + 2
}

/// The substrate turn part `part` belongs to. Part 0 is no part and belongs to
/// no turn.
pub fn turn_of(part: u64) -> Option<u32> {
    let turn = part.checked_sub(1)? / 2;
    u32::try_from(turn).ok()
}

/// `text` without its reasoning blocks.
fn without_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        match rest[open..].find("</think>") {
            Some(close) => rest = &rest[open + close + "</think>".len()..],
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// `text` on one line, at most [`CLIP`] characters, cut short with `…`.
fn clipped(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= CLIP {
        return flat;
    }
    let cut: String = flat.chars().take(CLIP).collect();
    format!("{}…", cut.trim_end())
}

/// The part `message` is, what it supports, and how it is introduced in the
/// stretch. A system bubble is no half of an exchange and no part.
fn classify(message: &RecoveredMessage) -> Option<(u64, Kind, &'static str)> {
    match message.role {
        Role::User if message.text.contains(TOOL_RESPONSE) => {
            Some((user_part(message.turn), Kind::Observed, "Tool results: "))
        }
        Role::User => Some((user_part(message.turn), Kind::Heard, "User: ")),
        Role::Assistant => Some((reply_part(message.turn), Kind::Did, "You: ")),
        Role::System => None,
    }
}

/// The parts of `history` past part `after`, oldest first, each at the instant
/// its turn was written (`written`, by turn index). A half with nothing left
/// once its reasoning is set aside is no part.
pub fn parts_after(
    history: &[RecoveredMessage],
    written: &[(TurnIndex, u64)],
    after: u64,
) -> Vec<SpanTurn> {
    let at = |turn: u32| {
        written
            .iter()
            .find(|(t, _)| t.0 == turn)
            .map_or(0, |(_, ms)| *ms)
    };
    history
        .iter()
        .filter_map(|message| {
            let (id, kind, who) = classify(message)?;
            if id <= after {
                return None;
            }
            let body = without_reasoning(&message.text);
            if body.trim().is_empty() {
                return None;
            }
            Some(SpanTurn {
                id,
                at_ms: at(message.turn),
                evidence: Some(kind),
                text: format!("{who}{}", clipped(&body)),
                full: format!("{who}{}", body.trim()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: Role, turn: u32, text: &str) -> RecoveredMessage {
        RecoveredMessage {
            role,
            text: text.to_string(),
            turn,
            no_think: false,
            thinking: None,
            tool_tokens: Vec::new(),
        }
    }

    fn history() -> Vec<RecoveredMessage> {
        vec![
            message(Role::User, 0, "Why does the scan stall on the 3090?"),
            message(
                Role::Assistant,
                0,
                "<think>Look at scan.rs.</think><tool_call>{\"name\": \"file_read\"}</tool_call>",
            ),
            message(
                Role::User,
                1,
                "<tool_response>fn scan() { … }</tool_response>",
            ),
            message(Role::Assistant, 1, "The loop never yields."),
        ]
    }

    fn written() -> Vec<(TurnIndex, u64)> {
        vec![(TurnIndex(0), 1_000), (TurnIndex(1), 2_000)]
    }

    #[test]
    fn each_half_of_a_turn_is_its_own_part_numbered_from_the_turn() {
        assert_eq!((user_part(0), reply_part(0)), (1, 2));
        assert_eq!((user_part(7), reply_part(7)), (15, 16));
        assert_eq!(turn_of(15), Some(7));
        assert_eq!(turn_of(16), Some(7));
        assert_eq!(turn_of(1), Some(0));
        assert_eq!(turn_of(0), None, "part 0 is no part");
    }

    #[test]
    fn what_the_user_said_was_heard_a_result_observed_and_a_reply_done() {
        let parts = parts_after(&history(), &written(), 0);
        let kinds: Vec<(u64, Option<Kind>)> = parts.iter().map(|p| (p.id, p.evidence)).collect();
        assert_eq!(
            kinds,
            [
                (1, Some(Kind::Heard)),
                (2, Some(Kind::Did)),
                (3, Some(Kind::Observed)),
                (4, Some(Kind::Did)),
            ]
        );
        assert_eq!(parts[0].text, "User: Why does the scan stall on the 3090?");
        assert_eq!(parts[3].text, "You: The loop never yields.");
        assert_eq!((parts[0].at_ms, parts[2].at_ms), (1_000, 2_000));
    }

    #[test]
    fn a_replys_reasoning_is_not_part_of_what_it_did() {
        let parts = parts_after(&history(), &written(), 0);
        assert_eq!(
            parts[1].text,
            "You: <tool_call>{\"name\": \"file_read\"}</tool_call>"
        );
        assert!(!without_reasoning("<think>a</think>b<think>c").contains('c'));
        assert_eq!(without_reasoning("x<think>a</think>y"), "xy");
    }

    #[test]
    fn a_reply_that_was_only_reasoning_is_no_part() {
        let h = vec![message(Role::Assistant, 3, "<think>hmm</think>  ")];
        assert!(parts_after(&h, &[], 0).is_empty());
    }

    #[test]
    fn only_the_parts_past_what_is_covered_are_taken() {
        let parts = parts_after(&history(), &written(), 2);
        assert_eq!(parts.iter().map(|p| p.id).collect::<Vec<_>>(), [3, 4]);
    }

    #[test]
    fn a_long_part_is_flattened_and_clipped() {
        let long = format!("<tool_response>{}</tool_response>", "line\n".repeat(400));
        let h = vec![message(Role::User, 0, &long)];
        let part = &parts_after(&h, &[], 0)[0];
        assert!(part.text.ends_with('…'), "{}", part.text);
        assert!(!part.text.contains('\n'));
        assert!(part.text.chars().count() <= "Tool results: ".len() + CLIP + 1);
    }

    /// The part is kept behind its entry whole, its lines as they ran; only
    /// its reasoning is left out.
    #[test]
    fn the_whole_part_is_kept_beside_the_clipped_one() {
        let long = format!("<tool_response>{}</tool_response>", "line\n".repeat(400));
        let h = vec![
            message(Role::User, 0, &long),
            message(Role::Assistant, 0, "<think>hm</think>\nDone.\n"),
        ];
        let parts = parts_after(&h, &[], 0);
        assert_eq!(parts[0].full, format!("Tool results: {}", long.trim()));
        assert_eq!(
            parts[0].full.lines().count(),
            401,
            "400 lines and the closing tag"
        );
        assert_eq!(parts[1].full, "You: Done.");
    }

    #[test]
    fn a_system_bubble_is_no_part() {
        let h = vec![message(Role::System, 0, "You are a coding assistant.")];
        assert!(parts_after(&h, &[], 0).is_empty());
    }

    #[test]
    fn a_turn_with_no_recorded_time_reads_as_the_epoch() {
        let parts = parts_after(&history(), &[], 0);
        assert!(parts.iter().all(|p| p.at_ms == 0));
    }
}
