//! Spans of a turn's user half whose recurrent state is captured or replayed
//! (`docs/recurrent_reply_calibration.md` §6.3, §7.2).
//!
//! A caller names a span by its bytes in the half's text ([`TextSpan`]); the
//! conversation turns that into token positions of the turn's prefill
//! ([`TokenSpan`]), and the scheduler cuts the prefill so the span runs as a
//! chunk of its own ([`snap_advance`]) — the model arms exactly that chunk.

use std::ops::Range;

pub use candle_transformers::models::delta_net::{ArmKind, ReplyRows};

/// A span of a user half, by its bytes in the half's text.
#[derive(Debug, Clone)]
pub struct TextSpan {
    pub bytes: Range<usize>,
    pub kind: ArmKind,
}

/// A span as positions in the turn's prefill tokens.
#[derive(Debug, Clone)]
pub struct TokenSpan {
    pub tokens: Range<usize>,
    pub kind: ArmKind,
}

/// How far a prefill at `offset` may advance, of the `advance` it was given,
/// so that every span runs as one chunk of its own: a chunk that would reach
/// into a span stops at its start, and one that starts on a span takes exactly
/// the span.
pub fn snap_advance(spans: &[TokenSpan], offset: usize, advance: usize) -> usize {
    for s in spans {
        if offset == s.tokens.start {
            return s.tokens.len();
        }
        if offset < s.tokens.start && offset + advance > s.tokens.start {
            return s.tokens.start - offset;
        }
    }
    advance
}

/// The span a chunk at `offset` runs, if it starts on one.
pub fn span_at(spans: &[TokenSpan], offset: usize) -> Option<&TokenSpan> {
    spans.iter().find(|s| s.tokens.start == offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans() -> Vec<TokenSpan> {
        [10..40, 60..90]
            .into_iter()
            .map(|tokens| TokenSpan {
                tokens,
                kind: ArmKind::Capture,
            })
            .collect()
    }

    #[test]
    fn a_chunk_stops_at_a_span_and_a_span_is_a_chunk_of_its_own() {
        let s = spans();
        assert_eq!(snap_advance(&s, 0, 512), 10, "stops at the first span");
        assert_eq!(snap_advance(&s, 0, 4), 4, "short of it, untouched");
        assert_eq!(snap_advance(&s, 10, 512), 30, "the span alone");
        assert_eq!(
            snap_advance(&s, 10, 5),
            30,
            "the span whole, past a smaller share"
        );
        assert_eq!(snap_advance(&s, 40, 512), 20, "stops at the next");
        assert_eq!(snap_advance(&s, 90, 512), 512, "past the last, untouched");
        assert!(span_at(&s, 60).is_some());
        assert!(span_at(&s, 61).is_none());
    }
}
