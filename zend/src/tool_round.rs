//! One round of tool calls, as the answer asked for them — including the ones
//! it asked for and could not be read.
//!
//! [`crate::tools::extract_tool_calls`] returns the calls that parse. A call
//! that does not — its JSON cut off by a length limit, or malformed — is not a
//! call to it, so on its own it would leave the turn looking like a final
//! answer: the loop ends, nothing runs, and neither the model nor the person
//! watching is told. A `write` of a design document ended exactly that way.
//!
//! A round is planned here instead. Every `<tool_call>` block in the answer is
//! either a [`Step::Run`] or a [`Step::Refuse`] carrying the error result it
//! gets in place of running, in the order the answer wrote them. The refusal is
//! an ordinary `{"error", "detail"}` result, so both readers already handle it:
//! the model reads it as a failed call and can issue the call again, and the
//! GUI pairs it with the call's card and shows it as an error.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Value};
use zend_tools::ToolContext;

use crate::tools::{answer_text, calls_in_answer, parse_call, run_tool, ToolCall, ToolResult};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// One call of a round.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// A call that parsed, to be run.
    Run(ToolCall),
    /// A call that could not be read. Nothing runs; this is its result.
    Refuse(ToolResult),
    /// A call already satisfied without running it — the fast path resolved it
    /// to content this conversation now carries (see [`crate::fast_path`]).
    /// Unlike [`Step::Refuse`] this is a success: the call is answered, it just
    /// costs no read.
    Served(ToolResult),
}

impl Step {
    /// The tool the step names — the name the GUI's card shows.
    pub fn name(&self) -> &str {
        match self {
            Step::Run(call) => &call.name,
            Step::Refuse(result) | Step::Served(result) => &result.call.name,
        }
    }
}

/// Why a `<tool_call>` block could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Problem {
    /// The call's JSON ends before the call does — the decode stopped partway.
    CutOff,
    /// The call's JSON is complete but not valid, or names no tool.
    Invalid(String),
}

/// The calls `response_text` makes, runnable and not, in the order it makes
/// them. Empty when it makes none — a final answer.
pub fn plan(response_text: &str) -> Vec<Step> {
    let answer = answer_text(response_text);
    let mut steps: Vec<(usize, Step)> = calls_in_answer(&answer)
        .into_iter()
        .map(|(at, call)| (at, Step::Run(call)))
        .collect();
    for (at, name, problem) in unreadable_blocks(&answer) {
        steps.push((at, Step::Refuse(refusal(name, &problem))));
    }
    steps.sort_by_key(|&(at, _)| at);
    steps.into_iter().map(|(_, step)| step).collect()
}

/// Whether the round has a call whose `</tool_call>` never arrived — the
/// stream then needs the closer written for it, or the GUI reads everything
/// after the opener as part of the call.
pub fn ends_inside_a_call(response_text: &str) -> bool {
    let answer = answer_text(response_text);
    answer
        .rfind(OPEN)
        .is_some_and(|open| !answer[open..].contains(CLOSE))
}

/// Run a planned round in order. A [`Step::Refuse`] contributes its result
/// without running anything.
pub fn run(ctx: &ToolContext, steps: Vec<Step>) -> Vec<ToolResult> {
    steps
        .into_iter()
        .map(|step| match step {
            Step::Run(call) => ToolResult {
                response: run_tool(ctx, &call),
                call,
            },
            Step::Refuse(result) => {
                tracing::warn!(
                    tool = %result.call.name,
                    error = %result.response["error"],
                    "tool call could not be read — answered with an error instead of run",
                );
                result
            }
            // Already answered by the fast path; running it would re-read a
            // file whose content this conversation is now carrying.
            Step::Served(result) => result,
        })
        .collect()
}

/// Every `<tool_call>` block in `answer` that does not parse as a call: its
/// offset, the tool it names when that much can be read, and why.
///
/// A block runs to its `</tool_call>`, or — when the answer ends or opens the
/// next call first — to that point, which is itself the sign it was cut off. A
/// block whose JSON parses is not reported, closed or not: the call scanners
/// already recover it.
fn unreadable_blocks(answer: &str) -> Vec<(usize, Option<String>, Problem)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = answer[from..].find(OPEN) {
        let at = from + rel;
        let body_start = at + OPEN.len();
        let rest = &answer[body_start..];
        let close = rest.find(CLOSE);
        let next_open = rest.find(OPEN);
        let (body_len, closed) = match (close, next_open) {
            (Some(c), Some(n)) if n < c => (n, false),
            (Some(c), _) => (c, true),
            (None, Some(n)) => (n, false),
            (None, None) => (rest.len(), false),
        };
        let body = rest[..body_len].trim();
        from = body_start + body_len;
        if body.is_empty() && !closed {
            // An opener with nothing after it: the turn ended on the marker,
            // before any call was written. There is no call to answer.
            continue;
        }
        let problem = match parse_call(body) {
            Ok(Some(_)) => continue,
            Ok(None) => Problem::Invalid("the call names no tool".to_string()),
            Err(e) if e.is_eof() || !closed => Problem::CutOff,
            Err(e) => Problem::Invalid(e.to_string()),
        };
        out.push((at, named_tool(body), problem));
    }
    out
}

/// The tool a call names, read from its text when the call itself does not
/// parse — `"name": "write"` survives a value cut off after it.
fn named_tool(body: &str) -> Option<String> {
    static NAME: OnceLock<Regex> = OnceLock::new();
    NAME.get_or_init(|| {
        Regex::new(r#""(?:name|function|tool)"\s*:\s*"([^"\\]+)""#).expect("static regex")
    })
    .captures(body)
    .map(|c| c[1].to_string())
}

/// The result an unreadable call gets: an ordinary tool error, saying that
/// nothing ran and what to do about it.
fn refusal(name: Option<String>, problem: &Problem) -> ToolResult {
    let label = name.as_deref().unwrap_or("tool");
    let response = match problem {
        Problem::CutOff => json!({
            "error": "call_cut_off",
            "detail": format!(
                "This {label} call was cut off before it was complete, so it was not \
                 run and nothing changed. If it carried a large value — a whole \
                 file's content — split the work across several smaller calls."
            ),
        }),
        Problem::Invalid(why) => json!({
            "error": "malformed_call",
            "detail": format!(
                "This {label} call could not be read ({why}), so it was not run and \
                 nothing changed. Issue it again as one JSON object with \"name\" \
                 and \"arguments\"."
            ),
        }),
    };
    ToolResult {
        call: ToolCall {
            name: name.unwrap_or_else(|| "tool_call".to_string()),
            arguments: Value::Null,
        },
        response,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(steps: &[Step]) -> Vec<&str> {
        steps.iter().map(Step::name).collect()
    }

    fn error_of(step: &Step) -> &str {
        match step {
            Step::Refuse(r) => r.response["error"].as_str().unwrap(),
            Step::Run(c) => panic!("{} was planned to run", c.name),
            Step::Served(r) => panic!("{} was served by the fast path", r.call.name),
        }
    }

    /// **The live failure: a `write` whose content was cut short.** The value
    /// ended on a backslash, so the grammar's closing quote became an escaped
    /// one and the JSON never ends. It is answered as cut off, under its own
    /// name, instead of vanishing.
    #[test]
    fn a_call_whose_value_never_closes_is_refused_as_cut_off() {
        let text = "<think>plan</think>\n\n<tool_call>\n{\"name\": \"write\", \"arguments\": \
                    {\"path\": \"docs/a.md\", \"content\": \"# A\\n\\n\\\"}}\n</tool_call>";
        let steps = plan(text);
        assert_eq!(runs(&steps), ["write"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
        assert!(!ends_inside_a_call(text), "the block itself is closed");
    }

    /// A call the turn stopped writing — no `</tool_call>` at all — is cut off
    /// too, and the stream is told it needs the closer.
    #[test]
    fn a_call_with_no_close_is_refused_and_flagged_open() {
        let text = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a\", \"content\": \"abc";
        let steps = plan(text);
        assert_eq!(runs(&steps), ["write"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
        assert!(ends_inside_a_call(text));
    }

    /// Complete but not JSON: refused as malformed, with the parser's reason.
    #[test]
    fn a_closed_call_that_is_not_json_is_refused_as_malformed() {
        let text = "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {path: x}}\n</tool_call>";
        let steps = plan(text);
        assert_eq!(runs(&steps), ["file_read"]);
        assert_eq!(error_of(&steps[0]), "malformed_call");
    }

    /// **Order is the answer's.** A refused call between two good ones keeps
    /// its place, so the n-th result still lands on the n-th card.
    #[test]
    fn a_refused_call_keeps_its_place_among_the_others() {
        let text = "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"file_read\", \"arguments\": {oops}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"datetime\", \"arguments\": {}}\n</tool_call>";
        let steps = plan(text);
        assert_eq!(runs(&steps), ["file_list", "file_read", "datetime"]);
        assert!(matches!(steps[0], Step::Run(_)));
        assert_eq!(error_of(&steps[1]), "malformed_call");
        assert!(matches!(steps[2], Step::Run(_)));
    }

    /// Good calls plan exactly as extraction finds them, and a final answer
    /// plans nothing.
    #[test]
    fn readable_calls_run_and_an_answer_plans_nothing() {
        let text = "<tool_call>\n{\"name\": \"datetime\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(
            plan(text),
            vec![Step::Run(ToolCall {
                name: "datetime".into(),
                arguments: json!({}),
            })]
        );
        assert!(plan("The answer is 4.").is_empty());
    }

    /// A call written inside the reasoning is deliberation — not run, and not
    /// refused either.
    #[test]
    fn a_broken_call_inside_the_reasoning_is_not_a_call() {
        assert!(plan("<think><tool_call>{\"name\": \"write\", \"argu</think>Done.").is_empty());
    }

    /// Nothing readable to name: the refusal still answers, as `tool_call`.
    #[test]
    fn a_nameless_cut_off_call_is_still_answered() {
        let steps = plan("<tool_call>\n{\"na");
        assert_eq!(runs(&steps), ["tool_call"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
    }
}
