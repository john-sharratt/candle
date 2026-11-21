//! The length of a turn's reasoning in tokens, reported as exact or estimated.
//!
//! Reasoning whose segment still holds K/V gives its exact length, which is the
//! decode's own count through `</think>`, recorded at seal. A segment whose K/V was
//! dropped keeps only its prose, and a decode, text and encode round trip need not
//! return the count it started from, so its length is an estimate and the result
//! says so. A turn that did not reason has no length at all.

use super::*;

/// How long a turn's reasoning was, in tokens.
///
/// `exact` when every reasoning segment is still real: its K/V span is the
/// decode's own count of the generated tokens through `</think>`, recorded at
/// seal. A segment whose K/V was dropped keeps only its prose, and a decode →
/// text → encode round trip need not give back the count it started from, so
/// that length is an estimate and says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ThinkingLength {
    pub tokens: u32,
    pub exact: bool,
}

impl TurnLayout {
    /// The turn's reasoning length: the summed K/V spans of its `Thinking`
    /// segments, with `estimate` counting the prose of any that are ethereal.
    /// `None` when the turn did not reason.
    pub fn thinking_length(&self, estimate: impl Fn(&str) -> u32) -> Option<ThinkingLength> {
        let mut total: Option<ThinkingLength> = None;
        for seg in &self.segments {
            let TurnSegment::Thinking { text, kv } = seg else {
                continue;
            };
            let (tokens, exact) = match kv {
                Some(kv) => (kv.len, true),
                None => (estimate(text), false),
            };
            let sum = total.get_or_insert(ThinkingLength {
                tokens: 0,
                exact: true,
            });
            sum.tokens += tokens;
            sum.exact &= exact;
        }
        total
    }
}
