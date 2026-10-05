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
//!
//! **A tag is markup only where the model wrote it as one.** The answer comes
//! as a [`TurnText`]: control tokens markup, ordinary tokens literal. A summary
//! that quotes `<tool_call>` from the file it describes spells the tag out in
//! ordinary tokens, and read as a string that quotation opens a call — six
//! `code_reading` summaries were refused as calls cut off at the opener that
//! way, and each model, told its summary was a broken call, went on to read and
//! summarise other files. The scanners read [`scan_text`] instead, in which a
//! quoted tag can no longer be matched.

use std::sync::OnceLock;

use candle_conversation::TurnText;
use regex::Regex;
use serde_json::{json, Value};
use zend_tools::ToolContext;

use crate::tools::{answer_text, calls_in_answer, parse_call, run_tool, ToolCall, ToolResult};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// What a quoted `<` becomes in [`scan_text`]: a private-use character no
/// scanner matches, mapped back to `<` in every argument a planned call
/// carries ([`unquote`]), so a call quoting a tag in its own value keeps it.
const QUOTED_LT: char = '\u{E000}';

/// The tags the scanners read. Inside a call's body a quotation of one of
/// these is the only `<` that has to be hidden: the body's own syntax — a
/// function block's `<function=…>` and `<parameter=…>` — is ordinary tokens
/// too, and must stay readable.
const SCANNED_TAGS: [&str; 4] = [OPEN, CLOSE, "<think>", "</think>"];

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

/// The calls `response` makes, runnable and not, in the order it makes them.
/// Empty when it makes none — a final answer.
pub fn plan(response: &TurnText) -> Vec<Step> {
    let answer = answer_text(&scan_text(response));
    let mut steps: Vec<(usize, Step)> = calls_in_answer(&answer)
        .into_iter()
        .map(|(at, mut call)| {
            unquote(&mut call.arguments);
            (at, Step::Run(call))
        })
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
pub fn ends_inside_a_call(response: &TurnText) -> bool {
    let answer = answer_text(&scan_text(response));
    answer
        .rfind(OPEN)
        .is_some_and(|open| !answer[open..].contains(CLOSE))
}

/// `response` as the call scanners read it: markup as written, and literal
/// text with its `<` written as [`QUOTED_LT`], so a tag the model only quoted
/// cannot open, close or bound anything.
///
/// Outside a call every literal `<` is quoted. Inside one — opened by markup
/// and not yet closed by it — only the `<` of a [`SCANNED_TAGS`] quotation is,
/// since the rest of the body is the call's own syntax.
fn scan_text(response: &TurnText) -> String {
    let mut out = String::new();
    let mut in_call = false;
    for piece in response.pieces() {
        if !piece.literal {
            out.push_str(&piece.text);
            in_call = match (piece.text.rfind(OPEN), piece.text.rfind(CLOSE)) {
                (Some(open), Some(close)) => open > close,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => in_call,
            };
        } else if in_call {
            let mut body = piece.text.clone();
            for tag in SCANNED_TAGS {
                body = body.replace(tag, &format!("{QUOTED_LT}{}", &tag[1..]));
            }
            out.push_str(&body);
        } else {
            out.push_str(&piece.text.replace('<', &QUOTED_LT.to_string()));
        }
    }
    out
}

/// Every string in `value` with [`QUOTED_LT`] written back as the `<` it
/// stands for.
fn unquote(value: &mut Value) {
    match value {
        Value::String(s) if s.contains(QUOTED_LT) => *s = s.replace(QUOTED_LT, "<"),
        Value::Array(items) => items.iter_mut().for_each(unquote),
        Value::Object(fields) => fields.values_mut().for_each(unquote),
        _ => {}
    }
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

    /// `text` with every tag in it written as a tag — the shape of a turn whose
    /// markers were all control tokens.
    fn tagged(text: &str) -> TurnText {
        TurnText::markup(text)
    }

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
        let steps = plan(&tagged(text));
        assert_eq!(runs(&steps), ["write"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
        assert!(
            !ends_inside_a_call(&tagged(text)),
            "the block itself is closed"
        );
    }

    /// A call the turn stopped writing — no `</tool_call>` at all — is cut off
    /// too, and the stream is told it needs the closer.
    #[test]
    fn a_call_with_no_close_is_refused_and_flagged_open() {
        let text = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a\", \"content\": \"abc";
        let steps = plan(&tagged(text));
        assert_eq!(runs(&steps), ["write"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
        assert!(ends_inside_a_call(&tagged(text)));
    }

    /// Complete but not JSON: refused as malformed, with the parser's reason.
    #[test]
    fn a_closed_call_that_is_not_json_is_refused_as_malformed() {
        let text = "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {path: x}}\n</tool_call>";
        let steps = plan(&tagged(text));
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
        let steps = plan(&tagged(text));
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
            plan(&tagged(text)),
            vec![Step::Run(ToolCall {
                name: "datetime".into(),
                arguments: json!({}),
            })]
        );
        assert!(plan(&tagged("The answer is 4.")).is_empty());
    }

    /// A call written inside the reasoning is deliberation — not run, and not
    /// refused either.
    #[test]
    fn a_broken_call_inside_the_reasoning_is_not_a_call() {
        let text = "<think><tool_call>{\"name\": \"write\", \"argu</think>Done.";
        assert!(plan(&tagged(text)).is_empty());
    }

    /// Nothing readable to name: the refusal still answers, as `tool_call`.
    #[test]
    fn a_nameless_cut_off_call_is_still_answered() {
        let steps = plan(&tagged("<tool_call>\n{\"na"));
        assert_eq!(runs(&steps), ["tool_call"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
    }

    /// **The live failure: a summary that quotes the tag.** The file it
    /// summarised documents `<tool_call>`, and the summary names it the same
    /// way — spelled out, in ordinary tokens. That is a final answer, not a
    /// call cut off at its opener.
    #[test]
    fn a_quoted_tag_in_the_answer_is_not_a_call() {
        let summary = TurnText::literal(
            "a ban that belongs to one session (`<tool_call>` on the answer that \
             closes a stuck tool loop) would otherwise reach the whole wave.",
        );
        assert!(plan(&summary).is_empty());
        assert!(!ends_inside_a_call(&summary));
        let both = TurnText::literal("`<tool_call>` and `</tool_call>` frame a call.");
        assert!(plan(&both).is_empty());
    }

    /// A real call after a quoted tag runs, and is the only step: the
    /// quotation neither opens a block of its own nor swallows the call.
    #[test]
    fn a_real_call_after_a_quoted_tag_runs_alone() {
        let text = TurnText::literal("I will read the file that defines `<tool_call>`.\n")
            .then_markup("<tool_call>")
            .then_literal("\n{\"name\": \"datetime\", \"arguments\": {}}\n")
            .then_markup("</tool_call>");
        assert_eq!(
            plan(&text),
            vec![Step::Run(ToolCall {
                name: "datetime".into(),
                arguments: json!({}),
            })]
        );
    }

    /// A call's body is read as written, quoted tags and all: a `write` whose
    /// content documents the tag keeps it in its argument.
    #[test]
    fn a_quoted_tag_inside_a_call_stays_in_its_argument() {
        let text = TurnText::markup("<tool_call>")
            .then_literal(
                "\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.md\", \
                 \"content\": \"calls open with <tool_call>\"}}\n",
            )
            .then_markup("</tool_call>");
        let steps = plan(&text);
        assert_eq!(runs(&steps), ["write"]);
        let Step::Run(call) = &steps[0] else {
            panic!("the write was not planned to run");
        };
        assert_eq!(call.arguments["content"], "calls open with <tool_call>");
    }

    #[test]
    fn literal_text_outside_a_call_has_its_tags_escaped() {
        let text = TurnText::literal("a <b> ")
            .then_markup("<tool_call>")
            .then_literal("<c>")
            .then_markup("</tool_call>")
            .then_literal(" <d>");
        assert_eq!(
            scan_text(&text),
            "a \u{E000}b> <tool_call><c></tool_call> \u{E000}d>"
        );
    }

    /// Inside a call only a quoted tag is hidden; the body's own `<` syntax
    /// stays readable.
    #[test]
    fn a_call_body_hides_only_quoted_tags() {
        let text = TurnText::markup("<tool_call>")
            .then_literal("<function=write><tool_call></think>")
            .then_markup("</tool_call>");
        assert_eq!(
            scan_text(&text),
            "<tool_call><function=write>\u{E000}tool_call>\u{E000}/think></tool_call>"
        );
    }

    #[test]
    fn unquoting_restores_every_nested_string() {
        let mut v = json!({"a": "x\u{E000}y", "b": ["\u{E000}", 3], "c": {"d": "\u{E000}e"}});
        unquote(&mut v);
        assert_eq!(v, json!({"a": "x<y", "b": ["<", 3], "c": {"d": "<e"}}));
    }
}
