//! The frames of a streamed chat reply: each decoded token becomes zero or more
//! SSE `data:` frames, and the reply closes with a stop frame whose
//! `finish_reason` says how it ended.
//!
//! For the daemon's own replies a token is text, less a collapsed leading
//! `<think></think>` ([`gate_leading_think`]). For a passthrough reply the think
//! block goes out as `reasoning_content` ([`ReasoningSplit`]) and, when the
//! client sent tools, each `<tool_call>` block as a `tool_calls` entry
//! ([`ToolCallSplit`]) — the shapes an OpenAI-compatible client reads.

use std::sync::Mutex;

use axum::response::sse::Event;

use candle_conversation::stencil::ToolSpec;
use candle_conversation::FinishReason;

use crate::openai_tools::{wire_function, Call};
use crate::reasoning_split::{Piece, ReasoningSplit};
use crate::think_gate::{gate_leading_think, ThinkGate};
use crate::tool_call_split::{Out, ToolCallSplit};
use crate::types::{ChatCompletionChunk, ChunkChoice, Delta, DeltaToolCall, Usage};

/// How a reply's text is shaped for the client.
pub struct Framing {
    /// Send the think block as `reasoning_content` — the passthrough.
    pub split_reasoning: bool,
    /// The client's tools, when it runs its own: calls go out as `tool_calls`.
    pub tools: Option<Vec<ToolSpec>>,
}

/// The id a reply's `index`-th tool call goes out under.
pub fn call_id(reply_id: &str, index: u32) -> String {
    format!("{reply_id}-call-{index}")
}

/// Turns one reply's tokens into frames. Shared by the token stream and the
/// flush after it, hence the lock; nothing contends for it.
pub struct Framer {
    id: String,
    model: String,
    created: u64,
    state: Mutex<State>,
}

struct State {
    /// No frame has gone out yet — the next one carries `role`.
    first: bool,
    /// Tool calls sent so far; each takes the next `index`.
    calls: u32,
    /// The daemon's leading-think gate: the text held so far, and whether the
    /// gate has resolved.
    held: String,
    gate_open: bool,
    reasoning: Option<ReasoningSplit>,
    tools: Option<ToolCallSplit>,
    /// The reply's token counts, folded over its turns; none until one ends.
    usage: Option<Usage>,
    /// How the reply's last finished turn ended.
    finish: FinishReason,
}

/// The OpenAI `finish_reason` for a reply: `tool_calls` when it called any —
/// the client has calls to run whatever cut the text short — else `length` when
/// its last turn spent the whole response budget, else `stop`.
pub fn finish_reason(called_tools: bool, finish: FinishReason) -> &'static str {
    match (called_tools, finish) {
        (true, _) => "tool_calls",
        (false, FinishReason::Length) => "length",
        (false, FinishReason::Stop) => "stop",
    }
}

impl Framer {
    pub fn new(id: String, model: String, created: u64, framing: Framing) -> Self {
        Self {
            id,
            model,
            created,
            state: Mutex::new(State {
                first: true,
                calls: 0,
                held: String::new(),
                gate_open: false,
                reasoning: framing.split_reasoning.then(ReasoningSplit::default),
                tools: framing.tools.map(ToolCallSplit::new),
                usage: None,
                finish: FinishReason::Stop,
            }),
        }
    }

    /// The frames one decoded token releases — none while its text is held.
    pub fn token(&self, text: String) -> Vec<anyhow::Result<Event>> {
        let mut guard = self.state.lock().unwrap();
        let s = &mut *guard;
        let piece = match s.reasoning.as_mut() {
            Some(split) => split.push(&text),
            None => Piece {
                reasoning: String::new(),
                content: release_through_gate(&mut s.held, &mut s.gate_open, text),
            },
        };
        self.frames(s, piece)
    }

    /// Whatever is still held once the reply has ended — a reply cut off
    /// mid-thought, a marker's first bytes, an unclosed call.
    pub fn flush(&self) -> Vec<anyhow::Result<Event>> {
        let mut guard = self.state.lock().unwrap();
        let s = &mut *guard;
        let piece = s
            .reasoning
            .as_mut()
            .map(ReasoningSplit::finish)
            .unwrap_or_default();
        let mut frames = self.frames(s, piece);
        let outs = s
            .tools
            .as_mut()
            .map(ToolCallSplit::finish)
            .unwrap_or_default();
        for out in outs {
            frames.push(self.out_frame(s, out));
        }
        frames
    }

    /// The closing frame, carrying the reply's [`finish_reason`].
    pub fn stop(&self) -> anyhow::Result<Event> {
        let reason = self.reason();
        self.chunk(Delta::default(), Some(reason))
    }

    /// The reply's `finish_reason` as it stands — see [`finish_reason`].
    fn reason(&self) -> &'static str {
        let s = self.state.lock().unwrap();
        finish_reason(s.calls > 0, s.finish)
    }

    /// Fold in a finished turn: its token counts, and how it ended. The reply
    /// ends the way its last turn did.
    pub fn turn_end(&self, usage: Usage, finish: FinishReason) {
        let mut s = self.state.lock().unwrap();
        s.usage = Some(match s.usage {
            Some(before) => before.then(usage),
            None => usage,
        });
        s.finish = finish;
    }

    /// The last chunk before `[DONE]` for a client that asked for usage: no
    /// choices, just the reply's token counts. `None` when no turn reported
    /// any — a reply that failed before its turn finished.
    pub fn usage_chunk(&self) -> Option<anyhow::Result<Event>> {
        let usage = self.state.lock().unwrap().usage?;
        Some(encode(&ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: Vec::new(),
            usage: Some(usage),
        }))
    }

    fn frames(&self, s: &mut State, piece: Piece) -> Vec<anyhow::Result<Event>> {
        let mut frames = Vec::new();
        if !piece.reasoning.is_empty() {
            let delta = Delta {
                reasoning_content: Some(piece.reasoning),
                ..Delta::default()
            };
            frames.push(self.delta(s, delta));
        }
        let outs = match s.tools.as_mut() {
            Some(split) => split.push(&piece.content),
            None if piece.content.is_empty() => Vec::new(),
            None => vec![Out::Text(piece.content)],
        };
        for out in outs {
            frames.push(self.out_frame(s, out));
        }
        frames
    }

    fn out_frame(&self, s: &mut State, out: Out) -> anyhow::Result<Event> {
        let delta = match out {
            Out::Text(text) => Delta {
                content: Some(text),
                ..Delta::default()
            },
            Out::Call(call) => {
                let index = s.calls;
                s.calls += 1;
                Delta {
                    tool_calls: Some(vec![self.tool_call(index, &call)]),
                    ..Delta::default()
                }
            }
        };
        self.delta(s, delta)
    }

    fn tool_call(&self, index: u32, call: &Call) -> DeltaToolCall {
        DeltaToolCall {
            index,
            id: call_id(&self.id, index),
            kind: "function",
            function: wire_function(call),
        }
    }

    /// A content frame; the reply's first one also names the role.
    fn delta(&self, s: &mut State, mut delta: Delta) -> anyhow::Result<Event> {
        if std::mem::take(&mut s.first) {
            tracing::debug!("streaming first token");
            delta.role = Some("assistant");
        }
        self.chunk(delta, None)
    }

    fn chunk(&self, delta: Delta, finish_reason: Option<&'static str>) -> anyhow::Result<Event> {
        encode(&ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        })
    }
}

/// One chunk as an SSE `data:` frame.
fn encode(chunk: &ChatCompletionChunk) -> anyhow::Result<Event> {
    serde_json::to_string(chunk)
        .map_err(|e| anyhow::anyhow!(e))
        .map(|data| Event::default().data(data))
}

/// **Hold a leading empty think block off the wire** — the daemon's own
/// replies. The engine strips empty `<think></think>` from the text it STORES,
/// but the stream is a separate assembly of raw token events, so a collapsed
/// block would go out verbatim and render as leaked markup — the common case
/// once the `off` effort dial closes the block on the token after `<think>`.
///
/// Returns the text this token releases, empty while the gate holds. The hold
/// is bounded by CONTENT rather than by the closing marker — the first
/// non-blank token inside the block opens the gate for good — so a genuine
/// reasoning turn is never withheld waiting for its own `</think>`. Once open,
/// the gate never closes again for this turn: a `<think>` later in the answer
/// body is ordinary content.
fn release_through_gate(held: &mut String, gate_open: &mut bool, text: String) -> String {
    if *gate_open {
        return text;
    }
    held.push_str(&text);
    match gate_leading_think(held) {
        ThinkGate::Hold => String::new(),
        ThinkGate::Open(at) => {
            *gate_open = true;
            // Everything from the resolve point, which is the whole buffer
            // when no block was skipped.
            held[at..].to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framer() -> Framer {
        Framer::new(
            "c".to_string(),
            "m".to_string(),
            7,
            Framing {
                split_reasoning: false,
                tools: None,
            },
        )
    }

    /// A reply cut off by its response budget says so. Reported as `stop`, a
    /// client takes the truncated reply for a finished one.
    #[test]
    fn a_reply_its_budget_cut_off_finishes_with_length() {
        let f = framer();
        f.turn_end(Usage::for_turn(3249, 600), FinishReason::Length);
        assert_eq!(f.reason(), "length");
    }

    /// The daemon can answer one request over several turns (a tool call and
    /// the answer after it); the reply ends the way its last turn did.
    #[test]
    fn a_reply_ends_the_way_its_last_turn_did() {
        let f = framer();
        f.turn_end(Usage::for_turn(3000, 600), FinishReason::Length);
        f.turn_end(Usage::for_turn(3700, 40), FinishReason::Stop);
        assert_eq!(f.reason(), "stop");

        let g = framer();
        g.turn_end(Usage::for_turn(3000, 40), FinishReason::Stop);
        g.turn_end(Usage::for_turn(3700, 600), FinishReason::Length);
        assert_eq!(g.reason(), "length");
    }

    /// Before any turn has finished there is nothing to report but a stop.
    #[test]
    fn a_reply_no_turn_has_finished_reports_stop() {
        assert_eq!(framer().reason(), "stop");
    }

    /// Calls outrank how the text ended: the client has calls to run either way.
    #[test]
    fn tool_calls_outrank_how_the_text_ended() {
        assert_eq!(finish_reason(true, FinishReason::Length), "tool_calls");
        assert_eq!(finish_reason(true, FinishReason::Stop), "tool_calls");
        assert_eq!(finish_reason(false, FinishReason::Length), "length");
        assert_eq!(finish_reason(false, FinishReason::Stop), "stop");
    }
}
