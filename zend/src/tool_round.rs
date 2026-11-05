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

use candle_conversation::stencil::element_close::{value_end, Follows, PARAM_CLOSE, PARAM_TAG};
use candle_conversation::TurnText;
use regex::Regex;
use serde_json::{json, Value};
use zend_tools::ToolContext;

use crate::tools::{answer_text, calls_in_answer, parse_call, run_tool, ToolCall, ToolResult};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";

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
    /// A call a resumed turn does not run again, since doing so could repeat
    /// an effect that already happened; this is its result, saying so.
    Withheld(ToolResult),
}

impl Step {
    /// The tool the step names — the name the GUI's card shows.
    pub fn name(&self) -> &str {
        match self {
            Step::Run(call) => &call.name,
            Step::Refuse(result) | Step::Withheld(result) => &result.call.name,
        }
    }
}

/// Why a `<tool_call>` block could not be read, and where it ended: how many
/// characters of the call were written, and the parameter it was in.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Problem {
    /// The call's JSON ends before the call does — the decode stopped partway.
    CutOff(Ending),
    /// The call's JSON is complete but not valid, or names no tool.
    Invalid(String, Ending),
}

/// Where an unreadable call ended.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Ending {
    /// The call's length as written, in characters.
    chars: usize,
    /// The last parameter the call opened, when it opened one.
    parameter: Option<String>,
    /// Whether the call is a function block rather than a JSON object.
    block: bool,
}

impl Ending {
    fn of(body: &str) -> Self {
        Ending {
            chars: body.chars().count(),
            parameter: last_parameter(body),
            block: body.trim_start().starts_with(FN_OPEN),
        }
    }

    /// "It ended 556 characters in, inside parameter `content`."
    fn sentence(&self) -> String {
        match &self.parameter {
            Some(p) => format!(
                "It ended {} characters in, inside parameter `{p}`.",
                self.chars
            ),
            None => format!("It ended {} characters in.", self.chars),
        }
    }
}

/// The last parameter a call's text opens: a function block's last element,
/// found by walking its elements value by value (`element_close`) so a tag a
/// value quotes is not taken for one; or a JSON object's last `"key":` inside
/// `arguments`.
fn last_parameter(body: &str) -> Option<String> {
    let body = body.trim_start();
    if let Some(named) = body.strip_prefix(FN_OPEN) {
        let mut at = FN_OPEN.len() + named.find('>')? + 1;
        let mut last = None;
        loop {
            let rest = &body[at..];
            at += rest.len() - rest.trim_start().len();
            let Some(element) = body[at..].strip_prefix(PARAM_TAG) else {
                return last;
            };
            let key_len = element.find('>')?;
            let key = element[..key_len].trim().to_string();
            let value = at + PARAM_TAG.len() + key_len + 1;
            match value_end(body, value, Follows::Either) {
                Some(close) => at = close + PARAM_CLOSE.len(),
                None => return Some(key),
            }
            last = Some(key);
        }
    }
    static KEY: OnceLock<Regex> = OnceLock::new();
    let args = body.find("\"arguments\"")?;
    let key =
        KEY.get_or_init(|| Regex::new(r#""([A-Za-z_][A-Za-z0-9_]*)"\s*:"#).expect("static regex"));
    key.captures_iter(&body[args + "\"arguments\"".len()..])
        .last()
        .map(|c| c[1].to_string())
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

/// The tools that bring content from outside into the turn: the web, a
/// network session's reply, a remote shell's output, a remote file. A turn
/// that has run one is read-out, and what it writes to memory is checked by a
/// person first (`docs/zen_tool_reach.md` §6.2).
pub const READS_UNTRUSTED: [&str; 14] = [
    "web_fetch",
    "web_search",
    "weather",
    "http_request",
    "tcp_session_recv",
    "udp_session_recv",
    "tls_session_recv",
    "telnet_send",
    "ssh_session_exec",
    "ssh_session_exec_async",
    "ssh_session_poll",
    "remote_fs_session_get",
    "remote_fs_session_list_dir",
    "sql_session_query",
];

/// Mark `ctx`'s conversation read-out before `call` runs, when `call` reads
/// from outside.
fn note_reading(ctx: &ToolContext, call: &ToolCall) {
    if READS_UNTRUSTED.contains(&call.name.as_str()) {
        if let Some(runner) = &ctx.ops_runner {
            runner.reading_untrusted(ctx.lineage());
        }
    }
}

/// Run a planned round in order, results in the order of the calls. A
/// [`Step::Refuse`] contributes its result without running anything. Every
/// call runs alone and in turn, since a write and the read after it depend on
/// that order.
pub fn run(ctx: &ToolContext, steps: Vec<Step>) -> Vec<ToolResult> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            Step::Run(call) => {
                note_reading(ctx, &call);
                out.push(ToolResult {
                    response: run_tool(ctx, &call),
                    call,
                })
            }
            Step::Refuse(result) => {
                tracing::warn!(
                    tool = %result.call.name,
                    error = %result.response["error"],
                    "tool call could not be read — answered with an error instead of run",
                );
                out.push(result);
            }
            // A resumed turn's call whose effect may already have happened.
            Step::Withheld(result) => {
                tracing::info!(tool = %result.call.name, "resumed turn: call not re-issued");
                out.push(result);
            }
        }
    }
    out
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
            Ok(None) => Problem::Invalid("the call names no tool".to_string(), Ending::of(body)),
            Err(e) if e.is_eof() || !closed => Problem::CutOff(Ending::of(body)),
            Err(e) => Problem::Invalid(e.to_string(), Ending::of(body)),
        };
        out.push((at, named_tool(body), problem));
    }
    out
}

/// The tool a call names, read from its text when the call itself does not
/// parse — a function block's `<function=write>`, or `"name": "write"`, both
/// survive a value cut off after them.
fn named_tool(body: &str) -> Option<String> {
    if let Some(named) = body.trim_start().strip_prefix(FN_OPEN) {
        let name = named[..named.find('>')?].trim();
        return (!name.is_empty()).then(|| name.to_string());
    }
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
        Problem::CutOff(ending) => json!({
            "error": "call_cut_off",
            "detail": format!(
                "This {label} call was cut off before it was complete, so it was not \
                 run and nothing changed. {} If it carried a large value — a whole \
                 file's content — split the work across several smaller calls.",
                ending.sentence()
            ),
        }),
        Problem::Invalid(why, ending) => json!({
            "error": "malformed_call",
            "detail": format!(
                "This {label} call could not be read ({why}), so it was not run and \
                 nothing changed. {} Issue it again as {}.",
                ending.sentence(),
                match ending.block {
                    true => "one complete function block, each parameter closed and the \
                             function closed after the last",
                    false => "one JSON object with \"name\" and \"arguments\"",
                }
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
    use std::sync::{Arc, Mutex};

    use zend_tools::{registry, OpsRunner};
    use zend_vfs::RepoFiles;

    use super::*;

    /// **Every tool that reads from outside is a real tool**, so a rename
    /// cannot silently drop one from the read-out mark.
    #[test]
    fn every_tool_that_reads_from_outside_is_registered() {
        for name in READS_UNTRUSTED {
            assert!(registry::find(name).is_some(), "{name}");
        }
    }

    /// A runner that keeps which conversations it was told read from outside.
    #[derive(Default)]
    struct Marks(Mutex<Vec<String>>);

    impl OpsRunner for Marks {
        fn run(&self, _caller: &str, _tool: &str, _args: Value) -> Result<String, String> {
            Ok(String::new())
        }

        fn reading_untrusted(&self, caller: &str) {
            self.0.lock().unwrap().push(caller.to_string());
        }
    }

    /// **A call that reads from outside marks its conversation before it
    /// runs**, and one that does not marks nothing.
    #[test]
    fn a_call_that_reads_from_outside_marks_its_conversation() {
        let marks = Arc::new(Marks::default());
        let ctx = ToolContext::new()
            .with_ops_runner(Arc::clone(&marks) as Arc<dyn OpsRunner>)
            .with_files(Arc::new(RepoFiles::detached()), "c1");
        let call = |name: &str| ToolCall {
            name: name.into(),
            arguments: json!({"url": "http://127.0.0.1/"}),
        };
        note_reading(&ctx, &call("datetime"));
        assert!(marks.0.lock().unwrap().is_empty());
        note_reading(&ctx, &call("web_fetch"));
        assert_eq!(*marks.0.lock().unwrap(), ["c1"]);
    }

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
            Step::Withheld(r) => panic!("{} was withheld from a resumed turn", r.call.name),
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
        let Step::Refuse(r) = &steps[0] else {
            panic!("refused")
        };
        let detail = r.response["detail"].as_str().unwrap();
        assert!(detail.contains("inside parameter `content`"), "{detail}");
    }

    /// **A refusal says where the call ended**: how far it got, and the
    /// parameter it was in — a function block's element or a JSON key.
    #[test]
    fn a_refusal_names_where_the_call_ended() {
        let block = "<function=write>\n<parameter=path>\na.js</parameter>\n<parameter=content>\nx";
        assert_eq!(last_parameter(block).as_deref(), Some("content"));
        let json = r#"{"name": "write", "arguments": {"path": "a", "content": "ab"#;
        assert_eq!(last_parameter(json).as_deref(), Some("content"));
        assert_eq!(last_parameter(r#"{"name": "wri"#), None);
        // A tag the value quotes is content, not the parameter it ended in.
        let quoting =
            "<function=write>\n<parameter=content>\nsee <parameter=path> here</parameter> and";
        assert_eq!(last_parameter(quoting).as_deref(), Some("content"));
        assert_eq!(Ending::of("abc").sentence(), "It ended 3 characters in.");
        assert_eq!(
            Ending::of(block).sentence(),
            format!(
                "It ended {} characters in, inside parameter `content`.",
                block.chars().count()
            )
        );
    }

    /// **A function block cut short is answered under its own tool**, told
    /// to write a function block again, not a JSON object.
    #[test]
    fn a_function_block_cut_short_is_named_and_told_its_own_shape() {
        let text = "<tool_call>\n<function=file_write>\n<parameter=path>\na.md</parameter>\n<parameter=content>\nhalf";
        let steps = plan(&tagged(text));
        assert_eq!(runs(&steps), ["file_write"]);
        assert_eq!(error_of(&steps[0]), "call_cut_off");
        let block = "<function=file_write>\n<parameter=path>\na.md</parameter>\n</function";
        let r = refusal(
            named_tool(block),
            &Problem::Invalid("x".into(), Ending::of(block)),
        );
        let detail = r.response["detail"].as_str().unwrap();
        assert!(detail.starts_with("This file_write call"), "{detail}");
        assert!(detail.contains("one complete function block"), "{detail}");
        assert!(!detail.contains("JSON"), "{detail}");
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
