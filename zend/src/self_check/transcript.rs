//! A stored conversation written out as a plain-text record, for the model to
//! judge.
//!
//! The self-check does not ask a conversation about itself through its own
//! projection: an ingest conversation's projection carries the priming chain's
//! turns as if they were its own, and a reader resumed onto a stored timeline
//! does not place that timeline's turns at all. Instead the conversation's
//! stored text is written into the question, so the model sees exactly the
//! conversation being judged and nothing else.
//!
//! The record keeps what a judgement needs and drops what would only cost
//! tokens: a tool result keeps its head (which names the file or folder that
//! came back), a reply keeps its opening, and reasoning blocks go. The chat
//! template's markers are written out as words, so the record can never be read
//! as live tool calls or turn boundaries.

/// Characters of a tool result the record keeps: enough for the header that
/// names what came back (`file=… page=…`, a listing's first entries, an error).
const TOOL_RESULT_CHARS: usize = 240;
/// Characters of a person's message the record keeps.
const MESSAGE_CHARS: usize = 800;
/// Characters of an assistant reply the record keeps: its tool calls, or the
/// opening of its answer, which is where a summary names its subject.
const REPLY_CHARS: usize = 1200;

const TOOL_RESPONSE_OPEN: &str = "<tool_response>";

/// The record of `turns`, each `(user text, assistant text)` in order.
pub fn render(turns: &[(String, String)]) -> String {
    let mut out = String::new();
    for (i, (user, assistant)) in turns.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("[turn {}]\n", i + 1));
        let user = user.trim();
        if let Some(result) = user.strip_prefix(TOOL_RESPONSE_OPEN) {
            out.push_str("TOOL RESULT: ");
            out.push_str(&clip(&plain(result), TOOL_RESULT_CHARS));
        } else {
            out.push_str("USER: ");
            out.push_str(&clip(&plain(user), MESSAGE_CHARS));
        }
        out.push('\n');
        out.push_str("ASSISTANT: ");
        out.push_str(&clip(&plain(&without_reasoning(assistant)), REPLY_CHARS));
        out.push('\n');
    }
    out
}

/// `text` with every `<think>…</think>` block removed; an unclosed block runs
/// to the end.
fn without_reasoning(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        rest = match rest[open..].find("</think>") {
            Some(close) => &rest[open + close + "</think>".len()..],
            None => "",
        };
    }
    out.push_str(rest);
    out
}

/// `text` with the chat template's markers written as words and its
/// whitespace runs collapsed to single spaces.
fn plain(text: &str) -> String {
    text.replace("<tool_call>", " [tool call] ")
        .replace("</tool_call>", " ")
        .replace(TOOL_RESPONSE_OPEN, " ")
        .replace("</tool_response>", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The first `max` characters of `text`, with how much was cut when anything
/// was.
fn clip(text: &str, max: usize) -> String {
    let total = text.chars().count();
    if total <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept} … [{} more characters]", total - max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(user: &str, assistant: &str) -> (String, String) {
        (user.to_string(), assistant.to_string())
    }

    #[test]
    fn a_file_reading_is_recorded_turn_by_turn() {
        let record = render(&[
            turn(
                "Read `a.rs`.",
                "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>",
            ),
            turn(
                "<tool_response>\n```rust file=r/a.rs page=0/1\n1 fn main() {}\n```\n</tool_response>",
                "**Purpose:** `a.rs` is the entry point.",
            ),
        ]);
        assert_eq!(
            record,
            "[turn 1]\n\
             USER: Read `a.rs`.\n\
             ASSISTANT: [tool call] {\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\"}}\n\
             \n\
             [turn 2]\n\
             TOOL RESULT: ```rust file=r/a.rs page=0/1 1 fn main() {} ```\n\
             ASSISTANT: **Purpose:** `a.rs` is the entry point.\n"
        );
    }

    #[test]
    fn reasoning_is_dropped_and_the_answer_kept() {
        assert_eq!(
            without_reasoning("<think>\nlet me see\n</think>\n\nIt is 10:53."),
            "\n\nIt is 10:53."
        );
        assert_eq!(without_reasoning("a<think>b</think>c<think>d"), "ac");
        assert_eq!(without_reasoning("no reasoning"), "no reasoning");
    }

    #[test]
    fn a_long_part_keeps_its_head_and_says_what_was_cut() {
        assert_eq!(clip("abcdef", 6), "abcdef");
        assert_eq!(clip("abcdefgh", 3), "abc … [5 more characters]");
        assert_eq!(clip("ééééé", 2), "éé … [3 more characters]");
    }

    #[test]
    fn a_stray_tool_result_is_recorded_as_one() {
        let record = render(&[turn(
            "<tool_response>{\"error\":\"call_cut_off\"}</tool_response>",
            "I'm not sure what this refers to.",
        )]);
        assert_eq!(
            record,
            "[turn 1]\n\
             TOOL RESULT: {\"error\":\"call_cut_off\"}\n\
             ASSISTANT: I'm not sure what this refers to.\n"
        );
    }
}
