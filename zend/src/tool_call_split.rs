//! Pull the model's `<tool_call>` blocks out of a streamed answer.
//!
//! The answer's prose streams as it decodes. A call is held from its opening
//! marker to its close and then released whole and parsed
//! ([`parse_call`]), for the response to carry as a `tool_calls` entry rather
//! than as text. The opening marker arrives split across tokens (`<`, `tool`,
//! `_call>`), so a partial one is held until the next fragment settles it.
//! A block that turns out not to be a call, or never closes, goes out as the
//! text it is.

use candle_conversation::stencil::element_close::function_end;
use candle_conversation::stencil::ToolSpec;

use crate::openai_tools::{parse_call, Call};
use crate::reasoning_split::marker_start_len;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Where an open call's `</tool_call>` is in `body`, the text after its
/// opener: the first one after the call itself has ended — a function block
/// at its real `</function>`, read with the catalog its grammar was compiled
/// from (`element_close`), a JSON object at its closing brace — so a value
/// holding `</tool_call>` does not end the call. A body that is neither is the
/// model's text, ending at the first close. `None` while the call is still
/// arriving.
fn close_of(body: &str, specs: &[ToolSpec]) -> Option<usize> {
    let lead = body.len() - body.trim_start().len();
    let rest = &body[lead..];
    let ended = if rest.starts_with("<function=") {
        lead + function_end(rest, specs)?
    } else if rest.starts_with('{') {
        lead + object_end(rest)?
    } else {
        0
    };
    body[ended..].find(CLOSE).map(|at| ended + at)
}

/// The offset just past the JSON object opening `text`, its strings read
/// whole so a brace or a tag inside one is text. `None` while it is open.
pub(crate) fn object_end(text: &str) -> Option<usize> {
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for (i, b) in text.bytes().enumerate() {
        match (in_string, escaped, b) {
            (true, true, _) => escaped = false,
            (true, false, b'\\') => escaped = true,
            (true, false, b'"') => in_string = false,
            (true, false, _) => {}
            (false, _, b'"') => in_string = true,
            (false, _, b'{' | b'[') => depth += 1,
            (false, _, b'}' | b']') => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            (false, _, _) => {}
        }
    }
    None
}

/// What one fragment of the answer releases, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    Text(String),
    Call(Call),
}

/// Splits an answer fed to it in fragments. See the module docs.
pub struct ToolCallSplit {
    /// The client's tools, typing any function element the model writes.
    specs: Vec<ToolSpec>,
    /// Text not yet released: a marker still arriving, or an open call.
    pending: String,
    /// Whether `pending` is the body of an open call.
    in_call: bool,
}

impl ToolCallSplit {
    pub fn new(specs: Vec<ToolSpec>) -> Self {
        Self {
            specs,
            pending: String::new(),
            in_call: false,
        }
    }

    /// Feed the next fragment of the answer; returns what it releases.
    pub fn push(&mut self, text: &str) -> Vec<Out> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(close) = close_of(&self.pending, &self.specs) else {
                    return out;
                };
                let body: String = self.pending.drain(..close).collect();
                self.pending.drain(..CLOSE.len());
                self.in_call = false;
                match parse_call(&body, &self.specs) {
                    Some(call) => out.push(Out::Call(call)),
                    // Not a call after all: the model's text, as written.
                    None => push_text(&mut out, format!("{OPEN}{body}{CLOSE}")),
                }
            } else if let Some(open) = self.pending.find(OPEN) {
                let text: String = self.pending.drain(..open).collect();
                push_text(&mut out, text);
                self.pending.drain(..OPEN.len());
                self.in_call = true;
            } else {
                let upto = self.pending.len() - marker_start_len(&self.pending, OPEN);
                let text: String = self.pending.drain(..upto).collect();
                push_text(&mut out, text);
                return out;
            }
        }
    }

    /// Release what is held once the answer ends. An unclosed call — a reply
    /// cut off mid-call — goes out as text, never as a call whose arguments
    /// were invented for it.
    pub fn finish(&mut self) -> Vec<Out> {
        let rest = std::mem::take(&mut self.pending);
        let text = if std::mem::take(&mut self.in_call) {
            format!("{OPEN}{rest}")
        } else {
            rest
        };
        let mut out = Vec::new();
        push_text(&mut out, text);
        out
    }
}

fn push_text(out: &mut Vec<Out>, text: String) {
    if !text.is_empty() {
        out.push(Out::Text(text));
    }
}

#[cfg(test)]
mod tests {
    use super::{Out, ToolCallSplit};
    use crate::openai_tools::Call;
    use serde_json::json;

    /// Feed `fragments` one at a time, finish, and merge adjacent text — the
    /// assertions are about what is released, not how finely it was cut.
    fn run(fragments: &[&str]) -> Vec<Out> {
        let mut split = ToolCallSplit::new(Vec::new());
        let mut released = Vec::new();
        for f in fragments {
            released.extend(split.push(f));
        }
        released.extend(split.finish());
        let mut merged: Vec<Out> = Vec::new();
        for out in released {
            if let (Some(Out::Text(prev)), Out::Text(text)) = (merged.last_mut(), &out) {
                prev.push_str(text);
                continue;
            }
            merged.push(out);
        }
        merged
    }

    fn call(name: &str) -> Out {
        Out::Call(Call {
            name: name.to_string(),
            arguments: json!({}),
        })
    }

    /// The markers split across tokens, the way the stream delivers them.
    #[test]
    fn a_call_is_held_whole_and_released_parsed() {
        let out = run(&[
            "Reading.\n\n<",
            "tool",
            "_call>\n{\"name\": \"read_files\", ",
            "\"arguments\": {}}\n</tool",
            "_call>",
        ]);
        assert_eq!(
            out,
            vec![Out::Text("Reading.\n\n".into()), call("read_files")]
        );
    }

    /// **A value holding `</tool_call>` does not end the call**, in either
    /// syntax: the close counts only after the call itself has ended.
    #[test]
    fn a_value_holding_the_close_stays_inside_the_call() {
        let out = run(&[
            "<tool_call>{\"name\": \"write\", \"arguments\": {\"content\": \"x</tool_",
            "call>y\"}}</tool_call> done",
        ]);
        assert_eq!(
            out,
            vec![
                Out::Call(Call {
                    name: "write".into(),
                    arguments: json!({"content": "x</tool_call>y"}),
                }),
                Out::Text(" done".into())
            ]
        );
        let block = "\n<function=write>\n<parameter=content>\na</tool_call></function></parameter>\n</function>\n";
        let body = format!("{block}</tool_call> after");
        assert_eq!(super::close_of(&body, &[]), Some(block.len()));
        assert_eq!(
            super::close_of(
                "\n<function=write>\n<parameter=content>\na</tool_call>",
                &[]
            ),
            None
        );
        assert_eq!(super::close_of("nope</tool_call>", &[]), Some(4));
    }

    #[test]
    fn text_after_a_call_streams_as_text() {
        let out = run(&["<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call> done"]);
        assert_eq!(out, vec![call("a"), Out::Text(" done".into())]);
    }

    #[test]
    fn a_block_that_is_not_a_call_goes_out_as_written() {
        let text = "x <tool_call>nope</tool_call> y";
        assert_eq!(run(&[text]), vec![Out::Text(text.into())]);
    }

    #[test]
    fn a_call_cut_off_goes_out_as_text() {
        let text = "<tool_call>{\"name\": \"a\"";
        assert_eq!(run(&[text]), vec![Out::Text(text.into())]);
    }

    #[test]
    fn a_lone_angle_bracket_is_released_by_the_next_fragment() {
        assert_eq!(run(&["a <", "b"]), vec![Out::Text("a <b".into())]);
    }
}
