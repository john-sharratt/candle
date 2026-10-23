//! Whether a stretch of the session is worth a journal entry at all.
//!
//! Asked before the write, against the journal the dialogue already carries in
//! its system prompt, as one `journal_gate` call: `yes` or `no`, and a reason. A
//! `no` is final for the stretch, so the journal does not fill with "nothing
//! new". A journal that holds no entry yet is not asked: there is nothing to
//! weigh the stretch against.

use journal_core::span::Span;
use journal_core::style::Style;
use journal_core::tools::arguments;
use serde_json::{json, Value};

use crate::journal::origin::turn_of;

/// The gate's call.
pub const GATE: &str = "journal_gate";

const YES: &str = "yes";
const NO: &str = "no";

/// The `journal_gate` call, in the flat `{name, params}` form.
pub fn spec() -> Value {
    json!({
        "name": GATE,
        "params": [
            { "name": "answer", "type": "string", "required": true, "enum": [YES, NO] },
            { "name": "reason", "type": "string", "required": true },
        ],
    })
}

/// The gate's question about `span`.
pub fn question(span: &Span, style: &Style) -> String {
    let mut turns: Vec<u32> = span.turns.iter().filter_map(|t| turn_of(t.id)).collect();
    turns.dedup();
    let turns = match turns.len() {
        1 => "1 more turn".to_string(),
        n => format!("{n} more turns"),
    };
    format!(
        "Your journal, in your instructions above, holds what you have written down so far. \
Since it was last brought up to date there have been {turns} of this session, \
{}. Is there anything in them worth a journal entry — a result read, a decision made, a \
change made, something left unfinished — that the journal does not already hold? Answer with \
journal_gate: `answer` yes or no, and `reason` in one short sentence.",
        (style.span)(span.from_ms, span.to_ms)
    )
}

/// What the gate answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub worth: bool,
    pub reason: String,
}

/// Read the gate's answer out of `decoded`. `Err` names what was wrong with it.
pub fn read(decoded: &str) -> Result<Verdict, String> {
    let (name, args) =
        arguments(decoded).ok_or_else(|| format!("no journal_gate call in: {}", decoded.trim()))?;
    if name != GATE {
        return Err(format!("the call was {name}, not {GATE}"));
    }
    let worth = match args.get("answer").and_then(Value::as_str).map(str::trim) {
        Some(YES) => true,
        Some(NO) => false,
        other => return Err(format!("`answer` must be yes or no, not {other:?}")),
    };
    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Ok(Verdict { worth, reason })
}

#[cfg(test)]
mod tests {
    use journal_core::entry::Kind;
    use journal_core::span::SpanTurn;

    use super::*;
    use crate::journal::style::CODING;

    #[test]
    fn a_yes_with_its_reason_is_read() {
        let decoded = r#"{"name": "journal_gate", "arguments": {"answer": "yes", "reason": "The test now passes."}}"#;
        assert_eq!(
            read(decoded),
            Ok(Verdict {
                worth: true,
                reason: "The test now passes.".into()
            })
        );
    }

    #[test]
    fn a_no_is_read_and_a_missing_reason_is_empty() {
        let decoded =
            r#"<tool_call>{"name": "journal_gate", "arguments": {"answer": "no"}}</tool_call>"#;
        assert_eq!(
            read(decoded),
            Ok(Verdict {
                worth: false,
                reason: String::new()
            })
        );
    }

    #[test]
    fn anything_else_is_refused_by_name() {
        assert!(read("I think so")
            .unwrap_err()
            .contains("no journal_gate call"));
        let other = r#"{"name": "journal_write", "arguments": {}}"#;
        assert!(read(other).unwrap_err().contains("journal_write"));
        let maybe = r#"{"name": "journal_gate", "arguments": {"answer": "maybe"}}"#;
        assert!(read(maybe).unwrap_err().contains("maybe"));
    }

    #[test]
    fn the_call_offers_exactly_yes_and_no_and_asks_a_reason() {
        let s = spec();
        assert_eq!(s["name"], GATE);
        assert_eq!(s["params"][0]["enum"], json!(["yes", "no"]));
        assert_eq!(s["params"][1]["name"], "reason");
        assert_eq!(s["params"][1]["required"], true);
    }

    #[test]
    fn the_question_names_the_stretch_and_the_call() {
        let span = Span::of(vec![
            SpanTurn {
                id: 31,
                at_ms: 0,
                evidence: Some(Kind::Heard),
                text: "User: x".into(),
            },
            SpanTurn {
                id: 32,
                at_ms: 60_000,
                evidence: Some(Kind::Did),
                text: "You: y".into(),
            },
        ])
        .unwrap();
        let q = question(&span, &CODING);
        assert!(q.contains("1 more turn of"), "both halves of turn 15: {q}");
        assert!(q.contains("[1970-01-01 00:00–00:01 UTC]"), "{q}");
        assert!(q.ends_with("one short sentence."), "{q}");
    }
}
