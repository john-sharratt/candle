//! A reply as its parts — prose, reasoning and calls — for the daemon's GUI
//! (`docs/zen_coding_session_fixes.md`, Cause 1b).
//!
//! The GUI once found calls by searching the streamed text for tags, and so
//! had the decoder's blind spot: a value holding `</tool_call>` or `<think>`
//! drew broken cards, and results paired with cards by position, so one wrong
//! split moved every later result. Here the reply is read once, on the server,
//! with the rule the grammar closes values on (`element_close`), and the GUI
//! is told what it holds:
//!
//! - prose, as the reply's `delta.content`, with no markup in it;
//! - [`PartEvent`]s for reasoning and calls — a part opens, its text or its
//!   arguments arrive as they decode, it closes, parsed or not;
//! - a call's result, by the call's id.
//!
//! [`PartSplit`] reads a reply as it streams; [`parts_of`] reads a stored one
//! into the same parts, so a reload draws what the stream drew. The raw text
//! keeps its markup — the model replays it — and only the GUI stops reading it.

use std::collections::VecDeque;

use candle_conversation::stencil::element_close::{
    follows_in, value_end, Follows, FUNCTION_TAG, PARAM_CLOSE, PARAM_TAG,
};
use candle_conversation::stencil::ToolSpec;
use serde::Serialize;
use serde_json::Value;

use crate::json_call_stream::{JsonCallStream, JsonOut};
use crate::reasoning_split::marker_start_len;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";

/// What a part is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Think,
    Call,
}

/// One step of a part, as the `part` event carries it. An id is `p<n>`, the
/// reply's n-th part, the same in the stream and in the history.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PartEvent {
    /// A part starts. A call opens at its `<tool_call>`, before its tool is
    /// written, so the GUI draws it from its first token.
    Open {
        id: String,
        kind: Kind,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
    },
    /// The tool a call names, once it is written.
    Tool { id: String, tool: String },
    /// The next chunk of a call's argument `name`, as decoded.
    Arg {
        id: String,
        name: String,
        text: String,
    },
    /// The next chunk of reasoning, and how many tokens it has run to.
    Text {
        id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<usize>,
    },
    /// The part ended: a call parsed, or not and why.
    Close {
        id: String,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// These calls have started running.
    Run { ids: Vec<String> },
    /// A call's result, and its length in tokens in the context.
    Result {
        id: String,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<usize>,
    },
}

/// What one fragment of a reply releases, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    Prose(String),
    Part(PartEvent),
}

/// Where a call stands.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    /// Before its syntax is known.
    Head,
    /// A function block, between its elements.
    Between,
    /// A function block's value for `name`; `started` once its framing
    /// newline is behind it.
    Value { name: String, started: bool },
    /// A function block whose `</function>` was written.
    Done,
    /// A JSON call, its name and arguments released as they decode; once its
    /// object ends, the `</tool_call>` after it.
    Json(JsonCallStream),
    /// Text that is no call, held to the close.
    Junk,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    Prose,
    Think(String),
    Call(String, Call),
}

/// Reads a reply into its parts as it streams. See the module docs.
pub struct PartSplit {
    /// The catalog the reply's grammar was compiled from: which tags may end
    /// each value (`element_close::follows_in`).
    tools: &'static [ToolSpec],
    pending: String,
    state: State,
    /// Parts opened so far: the next id.
    next: usize,
    /// The reasoning part last opened, which a token count is for.
    think: Option<String>,
    /// Calls closed and not yet answered, in order: what a round's results
    /// are for.
    unanswered: VecDeque<String>,
    /// The open function block's tool, and its arguments in the order
    /// written.
    block: (String, Vec<String>),
}

impl PartSplit {
    /// A splitter reading calls with `tools`, the catalog their grammar was
    /// compiled from.
    pub fn new(tools: &'static [ToolSpec]) -> Self {
        Self {
            tools,
            pending: String::new(),
            state: State::Prose,
            next: 0,
            think: None,
            unanswered: VecDeque::new(),
            block: (String::new(), Vec::new()),
        }
    }

    /// Feed the next fragment of the reply; returns what it releases.
    pub fn push(&mut self, text: &str) -> Vec<Out> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(&mut out) {}
        out
    }

    /// Release what is held once a turn of the reply ends: an open reasoning
    /// block closes, and an open call closes as cut short — the round answers
    /// it too, so it waits for its result like any call. The next turn starts
    /// in prose.
    pub fn finish(&mut self) -> Vec<Out> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        match std::mem::replace(&mut self.state, State::Prose) {
            State::Prose => prose(&mut out, rest),
            State::Think(id) => {
                if !rest.is_empty() {
                    out.push(Out::Part(PartEvent::Text {
                        id: id.clone(),
                        text: rest,
                        tokens: None,
                    }));
                }
                out.push(close(id, None));
            }
            State::Call(id, call) => {
                if let Call::Value { name, started } = call {
                    let text = if started {
                        rest
                    } else {
                        strip_lead(&rest).to_string()
                    };
                    if !text.is_empty() {
                        out.push(Out::Part(PartEvent::Arg {
                            id: id.clone(),
                            name,
                            text,
                        }));
                    }
                }
                self.closed_call(
                    &mut out,
                    id,
                    Some("the call was cut short before it was complete"),
                );
            }
        }
        out
    }

    /// The reasoning's running length: a count for the last reasoning part.
    pub fn think_tokens(&self, tokens: usize) -> Option<PartEvent> {
        Some(PartEvent::Text {
            id: self.think.clone()?,
            text: String::new(),
            tokens: Some(tokens),
        })
    }

    /// A round's calls have started running.
    pub fn run(&self) -> Option<PartEvent> {
        let ids: Vec<String> = self.unanswered.iter().cloned().collect();
        (!ids.is_empty()).then_some(PartEvent::Run { ids })
    }

    /// A round's results, in call order, each with its length in tokens:
    /// each answers the oldest call not yet answered.
    pub fn results(&mut self, results: &[Value], tokens: &[usize]) -> Vec<PartEvent> {
        let mut out = Vec::new();
        for (i, result) in results.iter().enumerate() {
            let Some(id) = self.unanswered.pop_front() else {
                break;
            };
            out.push(PartEvent::Result {
                id,
                result: result.clone(),
                tokens: tokens.get(i).copied(),
            });
        }
        out
    }

    fn open(&mut self, out: &mut Vec<Out>, kind: Kind, tool: Option<String>) -> String {
        let id = format!("p{}", self.next);
        self.next += 1;
        out.push(Out::Part(PartEvent::Open {
            id: id.clone(),
            kind,
            tool,
        }));
        id
    }

    /// One step over what is pending; `false` once nothing more can move.
    fn step(&mut self, out: &mut Vec<Out>) -> bool {
        match self.state.clone() {
            State::Prose => self.prose_step(out),
            State::Think(id) => self.think_step(out, id),
            State::Call(id, call) => self.call_step(out, id, call),
        }
    }

    fn prose_step(&mut self, out: &mut Vec<Out>) -> bool {
        let think = self.pending.find(THINK_OPEN);
        let call = self.pending.find(CALL_OPEN);
        let (at, marker) = match (think, call) {
            (Some(t), Some(c)) if t < c => (t, THINK_OPEN),
            (_, Some(c)) => (c, CALL_OPEN),
            (Some(t), None) => (t, THINK_OPEN),
            (None, None) => {
                let held = marker_start_len(&self.pending, THINK_OPEN)
                    .max(marker_start_len(&self.pending, CALL_OPEN));
                let upto = self.pending.len() - held;
                let text: String = self.pending.drain(..upto).collect();
                prose(out, text);
                return false;
            }
        };
        let text: String = self.pending.drain(..at).collect();
        prose(out, text);
        self.pending.drain(..marker.len());
        self.state = match marker {
            THINK_OPEN => {
                let id = self.open(out, Kind::Think, None);
                self.think = Some(id.clone());
                State::Think(id)
            }
            _ => {
                self.block = (String::new(), Vec::new());
                State::Call(self.open(out, Kind::Call, None), Call::Head)
            }
        };
        true
    }

    fn think_step(&mut self, out: &mut Vec<Out>, id: String) -> bool {
        let (upto, closed) = match self.pending.find(THINK_CLOSE) {
            Some(at) => (at, true),
            None => (
                self.pending.len() - marker_start_len(&self.pending, THINK_CLOSE),
                false,
            ),
        };
        let text: String = self.pending.drain(..upto).collect();
        if !text.is_empty() {
            out.push(Out::Part(PartEvent::Text {
                id: id.clone(),
                text,
                tokens: None,
            }));
        }
        if !closed {
            return false;
        }
        self.pending.drain(..THINK_CLOSE.len());
        out.push(close(id, None));
        self.state = State::Prose;
        true
    }

    fn call_step(&mut self, out: &mut Vec<Out>, id: String, call: Call) -> bool {
        match call {
            Call::Head => {
                let lead = self.pending.len() - self.pending.trim_start().len();
                let rest = &self.pending[lead..];
                if rest.starts_with(FN_OPEN) {
                    let Some(name_end) = rest.find('>') else {
                        return false;
                    };
                    let name = rest[FN_OPEN.len()..name_end].trim().to_string();
                    self.pending.drain(..lead + name_end + 1);
                    self.block = (name.clone(), Vec::new());
                    out.push(Out::Part(PartEvent::Tool {
                        id: id.clone(),
                        tool: name,
                    }));
                    self.state = State::Call(id, Call::Between);
                    true
                } else if rest.starts_with('{') {
                    self.pending.drain(..lead);
                    self.state = State::Call(id, Call::Json(JsonCallStream::default()));
                    true
                } else if rest.is_empty() || FN_OPEN.starts_with(rest) {
                    false
                } else {
                    self.state = State::Call(id, Call::Junk);
                    true
                }
            }
            Call::Between => {
                let lead = self.pending.len() - self.pending.trim_start().len();
                let rest = &self.pending[lead..];
                if let Some(named) = rest.strip_prefix(PARAM_TAG) {
                    let Some(name_len) = named.find('>') else {
                        return false;
                    };
                    let name = named[..name_len].trim().to_string();
                    self.pending.drain(..lead + PARAM_TAG.len() + name_len + 1);
                    self.block.1.push(name.clone());
                    self.state = State::Call(
                        id,
                        Call::Value {
                            name,
                            started: false,
                        },
                    );
                    true
                } else if rest.starts_with(FUNCTION_TAG) {
                    self.pending.drain(..lead + FUNCTION_TAG.len());
                    self.state = State::Call(id, Call::Done);
                    true
                } else if rest.is_empty()
                    || PARAM_TAG.starts_with(rest)
                    || FUNCTION_TAG.starts_with(rest)
                {
                    false
                } else {
                    self.state = State::Call(id, Call::Junk);
                    true
                }
            }
            Call::Value { name, started } => self.value_step(out, id, name, started),
            Call::Json(stream) => self.json_step(out, id, stream),
            Call::Done | Call::Junk => {
                let Some(at) = self.pending.find(CALL_CLOSE) else {
                    return false;
                };
                self.pending.drain(..at + CALL_CLOSE.len());
                match call {
                    Call::Done => self.closed_call(out, id, None),
                    _ => self.closed_call(out, id, Some("the call could not be read")),
                }
                self.state = State::Prose;
                true
            }
        }
    }

    /// A JSON call: its name and arguments as they decode, then — once its
    /// object ends — its `</tool_call>`. Text that is no JSON call is held to
    /// the close as junk.
    fn json_step(&mut self, out: &mut Vec<Out>, id: String, mut stream: JsonCallStream) -> bool {
        if !stream.is_done() {
            let (read, released) = stream.push(&self.pending);
            self.pending.drain(..read);
            for r in released {
                out.push(Out::Part(match r {
                    JsonOut::Name(tool) => {
                        self.block = (tool.clone(), Vec::new());
                        PartEvent::Tool {
                            id: id.clone(),
                            tool,
                        }
                    }
                    JsonOut::Arg { name, text } => PartEvent::Arg {
                        id: id.clone(),
                        name,
                        text,
                    },
                }));
            }
            if stream.is_bad() {
                self.state = State::Call(id, Call::Junk);
                return true;
            }
            if !stream.is_done() {
                self.state = State::Call(id, Call::Json(stream));
                return false;
            }
        }
        let lead = self.pending.len() - self.pending.trim_start().len();
        let rest = &self.pending[lead..];
        if rest.starts_with(CALL_CLOSE) {
            self.pending.drain(..lead + CALL_CLOSE.len());
            // A call that never named its tool is no call the round can run.
            let error = self.block.0.is_empty().then_some("the call names no tool");
            self.closed_call(out, id, error);
            self.state = State::Prose;
            return true;
        }
        if rest.is_empty() || CALL_CLOSE.starts_with(rest) {
            self.state = State::Call(id, Call::Json(stream));
            return false;
        }
        self.state = State::Call(id, Call::Junk);
        true
    }

    /// A function block's value: everything up to the `</parameter>` a tag the
    /// grammar offers there follows, released as it is decided. A
    /// `</parameter>` whose tail is still arriving, and a newline that may be
    /// the value's framing, are held.
    fn value_step(&mut self, out: &mut Vec<Out>, id: String, name: String, started: bool) -> bool {
        let (tool, written) = &self.block;
        let written: Vec<&str> = written.iter().map(String::as_str).collect();
        let follows = follows_in(self.tools, tool, &written);
        let mut started = started;
        if !started {
            if self.pending.is_empty() {
                return false;
            }
            if self.pending.starts_with('\n') {
                self.pending.drain(..1);
            }
            started = true;
        }
        let (upto, ended) = match value_end(&self.pending, 0, follows) {
            Some(close) => (close, true),
            None => (undecided_from(&self.pending, follows), false),
        };
        // The newline before the close is the value's framing, held until
        // the close decides it.
        let release = match self.pending[..upto].ends_with('\n') {
            true => upto - 1,
            false => upto,
        };
        let text: String = self.pending.drain(..release).collect();
        if !text.is_empty() {
            out.push(Out::Part(PartEvent::Arg {
                id: id.clone(),
                name: name.clone(),
                text,
            }));
        }
        if !ended {
            self.state = State::Call(id, Call::Value { name, started });
            return false;
        }
        let framing = upto - release;
        self.pending.drain(..framing + PARAM_CLOSE.len());
        self.state = State::Call(id, Call::Between);
        true
    }

    fn closed_call(&mut self, out: &mut Vec<Out>, id: String, error: Option<&str>) {
        self.unanswered.push_back(id.clone());
        out.push(close(id, error));
    }
}

/// Where the undecided tail of a value starts: at a `</parameter>` whose
/// tail is still a prefix of one of `follows`' that ends a value, or at a
/// partial `</parameter>` the next fragment may complete.
fn undecided_from(text: &str, follows: Follows) -> usize {
    let tails = follows.tails();
    let mut at = 0;
    while let Some(rel) = text[at..].find(PARAM_CLOSE) {
        let close = at + rel;
        let after = &text[close + PARAM_CLOSE.len()..];
        if tails.iter().any(|tail| tail.starts_with(after)) {
            return close;
        }
        at = close + PARAM_CLOSE.len();
    }
    text.len() - marker_start_len(text, PARAM_CLOSE)
}

fn strip_lead(text: &str) -> &str {
    text.strip_prefix('\n').unwrap_or(text)
}

fn prose(out: &mut Vec<Out>, text: String) {
    if !text.is_empty() {
        out.push(Out::Prose(text));
    }
}

fn close(id: String, error: Option<&str>) -> Out {
    Out::Part(PartEvent::Close {
        id,
        ok: error.is_none(),
        error: error.map(str::to_string),
    })
}

/// One part of a stored reply, whole — what the history carries.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Part {
    Prose {
        text: String,
    },
    Think {
        id: String,
        text: String,
    },
    Call {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        /// Arguments as written, by name, in the order they came.
        args: Vec<(String, String)>,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<usize>,
    },
}

/// A stored reply's parts, read as [`PartSplit`] reads a streaming one with
/// `tools`, with `results` — the reply's tool results in call order, each
/// with its length — on the calls they answer.
pub fn parts_of(
    text: &str,
    tools: &'static [ToolSpec],
    results: &[(Value, Option<usize>)],
) -> Vec<Part> {
    let mut split = PartSplit::new(tools);
    let mut events = split.push(text);
    events.extend(split.finish());
    let mut parts: Vec<Part> = Vec::new();
    let find = |parts: &mut Vec<Part>, want: &str| -> Option<usize> {
        parts.iter().position(|p| match p {
            Part::Think { id, .. } | Part::Call { id, .. } => id == want,
            Part::Prose { .. } => false,
        })
    };
    for event in events {
        match event {
            Out::Prose(text) => match parts.last_mut() {
                Some(Part::Prose { text: before }) => before.push_str(&text),
                _ => parts.push(Part::Prose { text }),
            },
            Out::Part(PartEvent::Open {
                id,
                kind: Kind::Think,
                ..
            }) => parts.push(Part::Think {
                id,
                text: String::new(),
            }),
            Out::Part(PartEvent::Open {
                id,
                kind: Kind::Call,
                tool,
            }) => parts.push(Part::Call {
                id,
                tool,
                args: Vec::new(),
                ok: false,
                error: None,
                result: None,
                tokens: None,
            }),
            Out::Part(PartEvent::Tool { id, tool: named }) => {
                if let Some(Part::Call { tool, .. }) = find(&mut parts, &id).map(|i| &mut parts[i])
                {
                    *tool = Some(named);
                }
            }
            Out::Part(PartEvent::Text { id, text, .. }) => {
                if let Some(Part::Think { text: body, .. }) =
                    find(&mut parts, &id).map(|i| &mut parts[i])
                {
                    body.push_str(&text);
                }
            }
            Out::Part(PartEvent::Arg { id, name, text }) => {
                if let Some(Part::Call { args, .. }) = find(&mut parts, &id).map(|i| &mut parts[i])
                {
                    match args.last_mut() {
                        Some((last, value)) if *last == name => value.push_str(&text),
                        _ => args.push((name, text)),
                    }
                }
            }
            Out::Part(PartEvent::Close {
                id,
                ok: closed_ok,
                error: why,
            }) => {
                if let Some(Part::Call { ok, error, .. }) =
                    find(&mut parts, &id).map(|i| &mut parts[i])
                {
                    *ok = closed_ok;
                    *error = why;
                }
            }
            Out::Part(PartEvent::Run { .. } | PartEvent::Result { .. }) => {}
        }
    }
    let mut answers = results.iter();
    for part in parts.iter_mut() {
        if let Part::Call { result, tokens, .. } = part {
            let Some((value, length)) = answers.next() else {
                break;
            };
            *result = Some(value.clone());
            *tokens = *length;
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::parse_tools;
    use serde_json::json;

    use super::*;

    /// Feed `fragments`, finish, and merge adjacent prose and argument chunks
    /// — what is asserted is what arrives, not how finely it was cut.
    fn run(fragments: &[&str]) -> Vec<Out> {
        run_with(&[], fragments)
    }

    fn run_with(tools: &'static [ToolSpec], fragments: &[&str]) -> Vec<Out> {
        let mut split = PartSplit::new(tools);
        let mut got = Vec::new();
        for f in fragments {
            got.extend(split.push(f));
        }
        got.extend(split.finish());
        let mut merged: Vec<Out> = Vec::new();
        for out in got {
            match (merged.last_mut(), &out) {
                (Some(Out::Prose(a)), Out::Prose(b)) => a.push_str(b),
                (
                    Some(Out::Part(PartEvent::Arg {
                        id: i,
                        name: n,
                        text: a,
                    })),
                    Out::Part(PartEvent::Arg { id, name, text }),
                ) if i == id && n == name => a.push_str(text),
                (
                    Some(Out::Part(PartEvent::Text { id: i, text: a, .. })),
                    Out::Part(PartEvent::Text {
                        id,
                        text,
                        tokens: None,
                    }),
                ) if i == id => a.push_str(text),
                _ => merged.push(out),
            }
        }
        merged
    }

    fn open(id: &str, kind: Kind, tool: Option<&str>) -> Out {
        Out::Part(PartEvent::Open {
            id: id.into(),
            kind,
            tool: tool.map(str::to_string),
        })
    }

    /// A call opening at its `<tool_call>`, then naming its tool.
    fn call(id: &str, tool: &str) -> [Out; 2] {
        [
            open(id, Kind::Call, None),
            Out::Part(PartEvent::Tool {
                id: id.into(),
                tool: tool.into(),
            }),
        ]
    }

    fn arg(id: &str, name: &str, text: &str) -> Out {
        Out::Part(PartEvent::Arg {
            id: id.into(),
            name: name.into(),
            text: text.into(),
        })
    }

    fn closed(id: &str) -> Out {
        Out::Part(PartEvent::Close {
            id: id.into(),
            ok: true,
            error: None,
        })
    }

    /// **Prose, reasoning and a call come apart**, with no markup left in the
    /// prose, whatever the token boundaries.
    #[test]
    fn a_reply_comes_apart_into_its_parts() {
        let reply = "<think>plan it</think>\nReading.\n<tool_call>\n<function=file_read>\n<parameter=path>\na.rs</parameter>\n</function>\n</tool_call>";
        let mut expected = vec![
            open("p0", Kind::Think, None),
            Out::Part(PartEvent::Text {
                id: "p0".into(),
                text: "plan it".into(),
                tokens: None,
            }),
            closed("p0"),
            Out::Prose("\nReading.\n".into()),
        ];
        expected.extend(call("p1", "file_read"));
        expected.extend([arg("p1", "path", "a.rs"), closed("p1")]);
        assert_eq!(run(&[reply]), expected);
        let bytes: Vec<String> = reply.chars().map(|c| c.to_string()).collect();
        let pieces: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(run(&pieces), expected, "byte at a time");
    }

    /// **A value holding the call's own tags keeps them**, and the call ends
    /// at its real close.
    #[test]
    fn a_value_holding_every_tag_streams_whole() {
        let content = "a</parameter> b</function> <tool_call> </tool_call> <think>x";
        let reply = format!(
            "<tool_call>\n<function=write>\n<parameter=content>\n{content}</parameter>\n</function>\n</tool_call>after"
        );
        let mut expected = call("p0", "write").to_vec();
        expected.extend([
            arg("p0", "content", content),
            closed("p0"),
            Out::Prose("after".into()),
        ]);
        assert_eq!(run(&[&reply]), expected);
    }

    /// **A note's text arrives as it decodes**, so the bubble fills while the
    /// call is still being written.
    /// **A last value holding another element's tags streams whole** when the
    /// catalog says nothing but the close may follow it, as the grammar did.
    #[test]
    fn a_last_value_closes_where_its_grammar_closed_it() {
        let tools = parse_tools(
            r#"[{"name":"file_write","params":[
                {"name":"path","type":"string","required":true},
                {"name":"content","type":"string","required":true}]}]"#,
        )
        .unwrap()
        .leak();
        let content = "x</parameter>\n<parameter=path>\ny";
        let reply = format!(
            "<tool_call>\n<function=file_write>\n<parameter=path>\na.md</parameter>\n<parameter=content>\n{content}</parameter>\n</function>\n</tool_call>"
        );
        let bytes: Vec<String> = reply.chars().map(|c| c.to_string()).collect();
        let pieces: Vec<&str> = bytes.iter().map(String::as_str).collect();
        let mut expected = call("p0", "file_write").to_vec();
        expected.extend([
            arg("p0", "path", "a.md"),
            arg("p0", "content", content),
            closed("p0"),
        ]);
        assert_eq!(run_with(tools, &pieces), expected);
    }

    /// **A call opens at its `<tool_call>`**, before a token of its tool is
    /// written, so the GUI draws it at once.
    #[test]
    fn a_call_opens_at_its_tag() {
        let mut split = PartSplit::new(&[]);
        assert_eq!(
            split.push("Next.<tool_call>"),
            vec![Out::Prose("Next.".into()), open("p0", Kind::Call, None)]
        );
    }

    /// **A JSON call streams as a function block does**: its tool once it is
    /// written, each string argument as it decodes — a document's text
    /// arrives while it is being written — and its close after its object.
    #[test]
    fn a_json_call_streams_its_arguments() {
        let reply = "<tool_call>\n{\"name\": \"file_write\", \"arguments\": {\"path\": \"a.md\", \"content\": \"# Title\\nbody </tool_call> \\\"q\\\"\", \"n\": 3}}\n</tool_call>after";
        let mut expected = call("p0", "file_write").to_vec();
        expected.extend([
            arg("p0", "path", "a.md"),
            arg("p0", "content", "# Title\nbody </tool_call> \"q\""),
            arg("p0", "n", "3"),
            closed("p0"),
            Out::Prose("after".into()),
        ]);
        assert_eq!(run(&[reply]), expected);
        let bytes: Vec<String> = reply.chars().map(|c| c.to_string()).collect();
        let pieces: Vec<&str> = bytes.iter().map(String::as_str).collect();
        assert_eq!(run(&pieces), expected, "byte at a time");
        let mut split = PartSplit::new(&[]);
        split.push("<tool_call>{\"name\": \"file_write\", \"arguments\": {\"content\": \"");
        assert_eq!(
            split.push("half a doc"),
            vec![arg("p0", "content", "half a doc")],
            "a value streams before it ends"
        );
    }

    /// **A JSON call that is not one, or names no tool, closes failed.**
    #[test]
    fn a_json_call_that_cannot_run_closes_failed() {
        let failed = |reply: &str| -> Option<String> {
            run(&[reply]).into_iter().find_map(|o| match o {
                Out::Part(PartEvent::Close {
                    ok: false, error, ..
                }) => error,
                _ => None,
            })
        };
        assert_eq!(
            failed("<tool_call>{\"arguments\": {}}</tool_call>").as_deref(),
            Some("the call names no tool")
        );
        assert_eq!(
            failed("<tool_call>{\"name\": \"a\"} junk</tool_call>").as_deref(),
            Some("the call could not be read")
        );
    }

    #[test]
    fn an_argument_streams_before_its_call_ends() {
        let mut split = PartSplit::new(&[]);
        split.push("<tool_call>\n<function=journal_note>\n<parameter=text>\n");
        let early = split.push("half a thought");
        assert_eq!(early, vec![arg("p0", "text", "half a thought")]);
    }

    /// **A call cut short closes as failed**, keeping what it wrote.
    #[test]
    fn a_call_cut_short_closes_failed() {
        let got = run(&["<tool_call>\n<function=write>\n<parameter=content>\nhalf"]);
        assert_eq!(got[2], arg("p0", "content", "half"));
        assert!(matches!(
            &got[3],
            Out::Part(PartEvent::Close {
                ok: false,
                error: Some(_),
                ..
            })
        ));
    }

    /// **Results land on their own calls by id**, oldest unanswered first.
    #[test]
    fn results_answer_the_calls_in_order() {
        let mut split = PartSplit::new(&[]);
        split.push("<tool_call>\n<function=a>\n</function>\n</tool_call><tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>");
        assert_eq!(
            split.run(),
            Some(PartEvent::Run {
                ids: vec!["p0".into(), "p1".into()]
            })
        );
        let got = split.results(&[json!("one"), json!("two")], &[3, 4]);
        assert_eq!(
            got,
            vec![
                PartEvent::Result {
                    id: "p0".into(),
                    result: json!("one"),
                    tokens: Some(3)
                },
                PartEvent::Result {
                    id: "p1".into(),
                    result: json!("two"),
                    tokens: Some(4)
                },
            ]
        );
        assert_eq!(split.run(), None);
    }

    /// **A stored reply reads into the same parts, with its results on its
    /// calls.**
    #[test]
    fn a_stored_reply_reads_back_as_its_parts() {
        let parts = parts_of(
            "Hi.<tool_call>\n<function=journal_note>\n<parameter=text>\nnoted it</parameter>\n</function>\n</tool_call>",
            &[],
            &[(json!("noted"), Some(2))],
        );
        assert_eq!(
            parts,
            vec![
                Part::Prose { text: "Hi.".into() },
                Part::Call {
                    id: "p0".into(),
                    tool: Some("journal_note".into()),
                    args: vec![("text".into(), "noted it".into())],
                    ok: true,
                    error: None,
                    result: Some(json!("noted")),
                    tokens: Some(2),
                },
            ]
        );
    }
}
