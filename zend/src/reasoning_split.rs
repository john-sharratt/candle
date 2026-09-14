//! Carry a reply's leading `<think>` block as OpenAI `reasoning_content`.
//!
//! An OpenAI-compatible client (Cline, and most agent frameworks) renders
//! reasoning from a separate `reasoning_content` field and shows `content`
//! verbatim, so a think block left in `content` reaches the user as raw markup.
//! The passthrough's replies open with one; the split routes everything inside
//! the block to reasoning and everything after it to the answer.
//!
//! The stream is decoded a token at a time and the markers arrive split across
//! tokens (`<`, `th`, `ink>`), so a partial marker is held until the next
//! fragment settles it. Blank runs around the reasoning and at the start of the
//! answer — the template's newlines on either side of the block — are dropped,
//! so neither opens on a gap and a collapsed block yields no reasoning at all.
//!
//! A `<tool_call>` inside a block that is still open is held. If the block
//! closes after it, the call was deliberation and goes out as reasoning. If the
//! reply ends with the block still open, the model wrote its call without
//! closing its reasoning first — the call was the answer, and goes out as
//! content, where the client's tool calls are read.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";

/// Text one step of the split releases.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Piece {
    pub reasoning: String,
    pub content: String,
}

impl Piece {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Only blank text, or the first bytes of an opening marker, so far.
    #[default]
    Leading,
    Reasoning,
    /// Inside the block, from a `<tool_call>` on: held until the block closes
    /// (deliberation) or the reply ends (the answer).
    HeldCall,
    Answer,
}

/// Splits a reply fed to it in fragments. See the module docs.
#[derive(Debug, Default)]
pub struct ReasoningSplit {
    phase: Phase,
    /// Text not yet released: a marker still arriving, or trailing blanks.
    pending: String,
    /// Whether the current phase has released any text yet.
    started: bool,
}

impl ReasoningSplit {
    /// Feed the next decoded fragment; returns what it releases.
    pub fn push(&mut self, text: &str) -> Piece {
        self.pending.push_str(text);
        let mut out = Piece::default();
        loop {
            match self.phase {
                Phase::Leading => {
                    let head = self.pending.trim_start();
                    if let Some(rest) = head.strip_prefix(OPEN) {
                        self.pending = rest.to_string();
                        self.enter(Phase::Reasoning);
                    } else if OPEN.starts_with(head) {
                        // Blank, or an opening marker still arriving.
                        return out;
                    } else {
                        // No block: the whole reply is answer, released as sent.
                        self.enter(Phase::Answer);
                        self.started = true;
                    }
                }
                Phase::Reasoning => {
                    let close = self.pending.find(CLOSE);
                    let call = self.pending.find(CALL_OPEN);
                    if let Some(at) = close.filter(|&at| call.is_none_or(|c| at < c)) {
                        self.close_block(at, &mut out);
                    } else if let Some(at) = call {
                        // A call opened inside the block: release the reasoning
                        // before it and hold the rest, blank run included.
                        let upto = self.pending[..at].trim_end().len();
                        let reasoning: String = self.pending.drain(..upto).collect();
                        self.release(&reasoning, Into::Reasoning, &mut out);
                        self.phase = Phase::HeldCall;
                    } else {
                        // Hold what may begin either marker, and the blank run
                        // before it — trailing blanks belong to no one until
                        // text follows them.
                        let held = marker_start_len(&self.pending, CLOSE)
                            .max(marker_start_len(&self.pending, CALL_OPEN));
                        let body = self.pending.len() - held;
                        let upto = self.pending[..body].trim_end().len();
                        let reasoning: String = self.pending.drain(..upto).collect();
                        self.release(&reasoning, Into::Reasoning, &mut out);
                        return out;
                    }
                }
                Phase::HeldCall => match self.pending.find(CLOSE) {
                    // The block closed after all: the call was deliberation.
                    Some(at) => self.close_block(at, &mut out),
                    None => return out,
                },
                Phase::Answer => {
                    let answer = std::mem::take(&mut self.pending);
                    self.release(&answer, Into::Content, &mut out);
                    return out;
                }
            }
        }
    }

    /// Release what is still held once the reply has ended: an unclosed block —
    /// a reply cut off mid-thought — is reasoning, a call held inside one is the
    /// answer, and anything else is answer.
    pub fn finish(&mut self) -> Piece {
        let rest = std::mem::take(&mut self.pending);
        let mut out = Piece::default();
        match self.phase {
            Phase::Reasoning => self.release(rest.trim_end(), Into::Reasoning, &mut out),
            // The block never closed, so the call it held was the reply's answer.
            Phase::HeldCall => {
                self.enter(Phase::Answer);
                self.release(&rest, Into::Content, &mut out);
            }
            Phase::Leading | Phase::Answer => self.release(&rest, Into::Content, &mut out),
        }
        out
    }

    /// End the block at `at` in `pending`: what precedes it is reasoning, what
    /// follows the marker is answer.
    fn close_block(&mut self, at: usize, out: &mut Piece) {
        let reasoning = self.pending[..at].trim_end().to_string();
        self.release(&reasoning, Into::Reasoning, out);
        self.pending.drain(..at + CLOSE.len());
        self.enter(Phase::Answer);
    }

    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.started = false;
    }

    /// Append `text` to `out`, dropping the blank run a phase opens with.
    fn release(&mut self, text: &str, into: Into, out: &mut Piece) {
        let text = if self.started {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return;
        }
        self.started = true;
        match into {
            Into::Reasoning => out.reasoning.push_str(text),
            Into::Content => out.content.push_str(text),
        }
    }
}

#[derive(Clone, Copy)]
enum Into {
    Reasoning,
    Content,
}

/// Length of the longest proper prefix of `marker` that `s` ends with — bytes
/// the next fragment may complete into the marker.
pub(crate) fn marker_start_len(s: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&k| s.ends_with(&marker[..k]))
        .unwrap_or(0)
}

/// Split a whole reply at once — the non-streaming response.
pub fn split(text: &str) -> Piece {
    let mut split = ReasoningSplit::default();
    let mut out = split.push(text);
    let rest = split.finish();
    out.reasoning.push_str(&rest.reasoning);
    out.content.push_str(&rest.content);
    out
}

#[cfg(test)]
mod tests {
    use super::{split, Piece, ReasoningSplit};

    fn piece(reasoning: &str, content: &str) -> Piece {
        Piece {
            reasoning: reasoning.to_string(),
            content: content.to_string(),
        }
    }

    /// Feed `fragments` one at a time and concatenate everything released.
    fn streamed(fragments: &[&str]) -> Piece {
        let mut split = ReasoningSplit::default();
        let mut out = Piece::default();
        for f in fragments {
            let p = split.push(f);
            out.reasoning.push_str(&p.reasoning);
            out.content.push_str(&p.content);
        }
        let p = split.finish();
        out.reasoning.push_str(&p.reasoning);
        out.content.push_str(&p.content);
        out
    }

    /// The case this exists for, fed the way the stream feeds it: markers split
    /// across tokens, and the template's newlines on both sides of the block.
    #[test]
    fn a_streamed_reply_splits_at_its_markers() {
        let out = streamed(&[
            "<", "th", "ink>", "\n", "The user", " asks.", "\n", "</", "think>", "\n\n", "Rome",
            ".",
        ]);
        assert_eq!(out, piece("The user asks.", "Rome."));
    }

    /// A collapsed block carries no reasoning, and the answer opens on text.
    #[test]
    fn a_collapsed_block_yields_no_reasoning() {
        assert_eq!(split("<think>\n\n</think>\n\nRome."), piece("", "Rome."));
    }

    /// A reply that never opens a block is all answer, exactly as sent.
    #[test]
    fn a_reply_without_a_block_is_all_answer() {
        assert_eq!(
            split("Paris is the capital."),
            piece("", "Paris is the capital.")
        );
    }

    /// A reply cut off before `</think>` — a `max_tokens` stop — is reasoning.
    #[test]
    fn a_reply_cut_off_mid_thought_is_reasoning() {
        assert_eq!(
            split("<think>\nStill thinking\n"),
            piece("Still thinking", "")
        );
    }

    /// A `<` inside the reasoning is held only until the next fragment shows it
    /// is not the closing marker, and nothing of it is lost.
    #[test]
    fn a_lone_angle_bracket_is_released_by_the_next_fragment() {
        assert_eq!(streamed(&["<think>a <", "b</think>c"]), piece("a <b", "c"));
    }

    /// The reply that stalled Cline: the block opens and never closes, and the
    /// call written inside it is the answer — it reaches the client as content,
    /// where its tool calls are read, not as reasoning.
    #[test]
    fn a_call_in_a_block_that_never_closes_is_the_answer() {
        let out = streamed(&[
            "<think>\n\n```bash\nls\n```\n\nLet me look:\n\n<tool",
            "_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
        ]);
        assert_eq!(
            out,
            piece(
                "```bash\nls\n```\n\nLet me look:",
                "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>"
            )
        );
    }

    /// A call the model weighs inside a block it then closes is deliberation.
    #[test]
    fn a_call_deliberated_inside_a_closed_block_stays_reasoning() {
        assert_eq!(
            split(
                "<think>\nMaybe <tool_call>{\"name\": \"a\"}</tool_call>, no.\n</think>\n\nDone."
            ),
            piece(
                "Maybe <tool_call>{\"name\": \"a\"}</tool_call>, no.",
                "Done."
            )
        );
    }

    /// Streaming a character at a time releases exactly what splitting the
    /// whole reply does — interior blank runs included.
    #[test]
    fn streaming_and_splitting_agree() {
        for reply in [
            "<think>\nA b\n\nc\n</think>\n\nAns wer\n",
            "<think></think>x",
            "no block <think>here</think>",
            "<think>\nunclosed",
            "<think>\nplan\n<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>",
            "<think>\nx <tool_call>y</tool_call> z\n</think>\nw",
        ] {
            let chars: Vec<String> = reply.chars().map(String::from).collect();
            let fragments: Vec<&str> = chars.iter().map(String::as_str).collect();
            assert_eq!(streamed(&fragments), split(reply), "reply {reply:?}");
        }
    }
}
