//! Where a projection's boundaries fall in a token stream
//! (`docs/provenance_scoped_signatures.md` §2.1).
//!
//! One rule serves both sides of provenance. The live decode loop steps it
//! token by token and reprojects where it fires; the gallery runs it over a
//! stored turn and cuts its windows where it fires, so what ingest compares is
//! cut exactly as what a projection asks.
//!
//! A boundary fires after a token outside a tool call when:
//!
//! - the cadence's count of tokens has passed since the last boundary, or
//!   since the stream began — every boundary restarts it, so a boundary fired
//!   by a trigger is followed by a full cadence before the next;
//! - the token is a trigger (a line feed) and more than [`CONTENT_GATE`]
//!   non-trigger tokens have passed since the last boundary.
//!
//! None fires inside a tool call, its open and close tags included, and its
//! tokens count toward neither: the call is written against the slot it
//! opened on.
//!
//! The first token of a stream only enters a call when it opens one: the live
//! loop receives it from the prefill, not from a decode step.

use std::sync::Arc;

use tokenizers::Tokenizer;

/// Non-trigger tokens that must pass before a trigger fires a boundary, so
/// short lines and runs of line feeds do not each fire one.
pub const CONTENT_GATE: usize = 16;

/// The boundary rule a projection decodes under.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryRule {
    /// The trigger tokens (`reproject_trigger_texts`, tokenized).
    pub triggers: Arc<Vec<u32>>,
    /// A boundary once this many tokens have passed since the last one; 0 for
    /// none.
    pub cadence: usize,
    /// The `<tool_call>` token, when the tokenizer has it as one.
    pub tool_call_open: Option<u32>,
    /// The `</tool_call>` token, when the tokenizer has it as one.
    pub tool_call_close: Option<u32>,
}

impl BoundaryRule {
    /// The rule `tokenizer` gives `trigger_texts` (each text's tokens are all
    /// triggers) and `cadence`, with the tool-call gate on the tags the
    /// tokenizer holds as single tokens.
    pub fn from_tokenizer(tokenizer: &Tokenizer, trigger_texts: &[String], cadence: usize) -> Self {
        let mut triggers: Vec<u32> = Vec::new();
        for text in trigger_texts.iter().filter(|t| !t.is_empty()) {
            if let Ok(enc) = tokenizer.encode(text.as_str(), false) {
                for &id in enc.get_ids() {
                    if !triggers.contains(&id) {
                        triggers.push(id);
                    }
                }
            }
        }
        let single = |text: &str| match tokenizer.encode(text, false).ok()?.get_ids() {
            [id] => Some(*id),
            _ => None,
        };
        BoundaryRule {
            triggers: Arc::new(triggers),
            cadence,
            tool_call_open: single("<tool_call>"),
            tool_call_close: single("</tool_call>"),
        }
    }

    /// Whether the rule can fire at all.
    pub fn is_active(&self) -> bool {
        self.cadence > 0 || !self.triggers.is_empty()
    }
}

/// What one token did to the boundary state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// No boundary.
    Quiet,
    /// A boundary after this token.
    Boundary,
    /// The token opened a tool call: no boundary, and none fires until the
    /// call closes.
    OpenedToolCall,
}

impl Step {
    /// Whether a boundary falls after the token.
    pub fn fires(self) -> bool {
        self == Step::Boundary
    }
}

/// The state the rule carries between tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryState {
    /// Non-trigger tokens since the last boundary.
    since_boundary: usize,
    /// Tokens outside a call since the last boundary or the stream's start:
    /// the cadence's count.
    since_cadence: usize,
    /// Inside a tool call's body.
    in_tool_call: bool,
}

impl BoundaryState {
    /// The state after a stream's first token, which only enters a call.
    pub fn first(rule: &BoundaryRule, token: u32) -> BoundaryState {
        BoundaryState::starting(rule.tool_call_open == Some(token))
    }

    /// The state after a stream's first token, inside a tool call's body or
    /// not: a first token outside a call is the cadence's first.
    pub fn starting(in_tool_call: bool) -> BoundaryState {
        BoundaryState {
            since_boundary: 0,
            since_cadence: usize::from(!in_tool_call),
            in_tool_call,
        }
    }

    /// Whether the stream is inside a tool call's body.
    pub fn in_tool_call(&self) -> bool {
        self.in_tool_call
    }

    /// Take the stream's next `token`.
    pub fn step(&mut self, rule: &BoundaryRule, token: u32) -> Step {
        if rule.tool_call_open == Some(token) {
            self.since_boundary = 0;
            self.in_tool_call = true;
            return Step::OpenedToolCall;
        }
        if rule.tool_call_close == Some(token) {
            self.in_tool_call = false;
            self.since_boundary = 0;
            return Step::Quiet;
        }
        if self.in_tool_call {
            return Step::Quiet;
        }
        self.since_cadence += 1;
        let is_trigger = rule.triggers.contains(&token);
        let cadence = rule.cadence > 0 && self.since_cadence >= rule.cadence;
        if cadence || (is_trigger && self.since_boundary > CONTENT_GATE) {
            self.since_boundary = 0;
            self.since_cadence = 0;
            Step::Boundary
        } else {
            if !is_trigger {
                self.since_boundary += 1;
            }
            Step::Quiet
        }
    }
}

/// Where the boundaries fall in `tokens`: each the count of tokens before it,
/// so a boundary `b` closes the window `tokens[..b]`.
pub fn boundaries(tokens: &[u32], rule: &BoundaryRule) -> Vec<usize> {
    let Some((&first, rest)) = tokens.split_first() else {
        return Vec::new();
    };
    let mut state = BoundaryState::first(rule, first);
    let mut out = Vec::new();
    for (i, &token) in rest.iter().enumerate() {
        let count = i + 2;
        if state.step(rule, token).fires() {
            out.push(count);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LF: u32 = 10;
    const OPEN: u32 = 900;
    const CLOSE: u32 = 901;

    fn rule(cadence: usize) -> BoundaryRule {
        BoundaryRule {
            triggers: Arc::new(vec![LF]),
            cadence,
            tool_call_open: Some(OPEN),
            tool_call_close: Some(CLOSE),
        }
    }

    /// `n` non-trigger tokens.
    fn words(n: usize) -> Vec<u32> {
        (0..n).map(|i| 100 + i as u32 % 50).collect()
    }

    #[test]
    fn a_line_feed_fires_only_past_the_content_gate() {
        // 1 first token + 16 words: 16 since any boundary, not past the gate.
        let mut t = words(17);
        t.push(LF);
        assert!(boundaries(&t, &rule(0)).is_empty());
        // One more word: 17 > 16, and the line feed fires after itself.
        let mut t = words(18);
        t.push(LF);
        assert_eq!(boundaries(&t, &rule(0)), vec![19]);
    }

    #[test]
    fn a_run_of_line_feeds_fires_once() {
        let mut t = words(18);
        t.extend([LF, LF, LF]);
        assert_eq!(boundaries(&t, &rule(0)), vec![19]);
    }

    #[test]
    fn the_cadence_fires_every_cadence_tokens() {
        let t = words(130);
        assert_eq!(boundaries(&t, &rule(64)), vec![64, 128]);
    }

    /// **Every boundary restarts the cadence**: a line feed's boundary at 21
    /// puts the cadence's next at 21 + 64, not at 64.
    #[test]
    fn a_boundary_restarts_the_cadence() {
        let mut t = words(20);
        t.push(LF);
        t.extend(words(110));
        assert_eq!(boundaries(&t, &rule(64)), vec![21, 85]);
    }

    /// **No boundary inside a tool call, its tags included**: the call's
    /// tokens count toward neither the cadence nor the gate, and the content
    /// gate restarts at its close.
    #[test]
    fn a_tool_call_fires_nothing_until_it_closes() {
        let mut t = words(10);
        t.push(OPEN);
        t.extend(words(70));
        t.push(LF);
        t.push(CLOSE);
        t.extend(words(17));
        t.push(LF);
        t.extend(words(1));
        t.push(LF);
        assert_eq!(boundaries(&t, &rule(64)), vec![101]);
    }

    /// **The cadence counts only the tokens outside a call**: 40 before it and
    /// 24 after make the 64th.
    #[test]
    fn a_tool_call_pauses_the_cadence() {
        let mut t = words(40);
        t.push(OPEN);
        t.extend(words(100));
        t.push(CLOSE);
        t.extend(words(30));
        // Tokens 1..=40 count; the call is 41..=142; 143.. count from 41.
        assert_eq!(boundaries(&t, &rule(64)), vec![142 + 24]);
    }

    #[test]
    fn a_first_token_that_opens_a_call_starts_inside_it() {
        let mut t = vec![OPEN];
        t.extend(words(40));
        t.push(LF);
        assert!(boundaries(&t, &rule(0)).is_empty());
        let state = BoundaryState::first(&rule(0), OPEN);
        assert!(state.in_tool_call());
    }

    #[test]
    fn an_empty_or_single_token_stream_has_no_boundary() {
        assert!(boundaries(&[], &rule(64)).is_empty());
        assert!(boundaries(&[LF], &rule(1)).is_empty());
        assert!(!BoundaryRule::default().is_active());
    }
}
