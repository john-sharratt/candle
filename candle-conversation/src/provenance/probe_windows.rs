//! A stored turn's gallery windows, each the probe a projection would have
//! built at one of its boundaries (`docs/provenance_scoped_signatures.md` §2.2).
//!
//! The live probe at a boundary is the turn's trailing chunks — at most
//! `max_probe_tokens`, chunk-aligned — and, once that tail has slid past them,
//! the turn's first [`QUERY_HEAD_CHUNKS`] chunks: the query head. A window here
//! is the same cut, in the turn's token offsets, at every boundary
//! [`boundaries`] finds in it. A turn the rule never fires in — shorter than
//! the content gate, say — is one window, the probe at its end.
//!
//! The cut needs only the turn's tokens, so a turn sealed before the rule
//! existed is windowed as one sealed after: nothing is re-ingested.

use std::iter::once;
use std::ops::Range;

use super::boundaries::{boundaries, BoundaryRule};

/// The query head's length in chunks, as the live probe keeps it.
pub const QUERY_HEAD_CHUNKS: usize = 2;

/// One window: the query head, when the tail has slid past it, and the tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeWindow {
    pub head: Option<Range<usize>>,
    pub tail: Range<usize>,
}

impl ProbeWindow {
    /// Its ranges, head first.
    pub fn ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.head.iter().cloned().chain(once(self.tail.clone()))
    }

    /// Its token count.
    pub fn len(&self) -> usize {
        self.ranges().map(|r| r.len()).sum()
    }

    /// Whether it holds no token.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The probe a boundary after `end` tokens builds: the chunks `end` reaches
/// into, at most `max_probe_tokens` of them, ending at `end`, and the head when
/// they start past it.
pub fn probe_at(end: usize, max_probe_tokens: usize, chunk: usize) -> ProbeWindow {
    let cur_chunks = end.div_ceil(chunk);
    let max_chunks = max_probe_tokens.max(1).div_ceil(chunk);
    let tail_lo = cur_chunks.saturating_sub(max_chunks);
    let head = (tail_lo > 0).then(|| 0..(QUERY_HEAD_CHUNKS.min(tail_lo) * chunk).min(end));
    ProbeWindow {
        head,
        tail: (tail_lo * chunk).min(end)..end,
    }
}

/// Every window of a turn of `tokens`: the probe at each boundary, and at the
/// turn's end when the last boundary falls short of it.
pub fn probe_windows(
    tokens: &[u32],
    rule: &BoundaryRule,
    max_probe_tokens: usize,
    chunk: usize,
) -> Vec<ProbeWindow> {
    let mut ends = boundaries(tokens, rule);
    if ends.last() != Some(&tokens.len()) && !tokens.is_empty() {
        ends.push(tokens.len());
    }
    ends.into_iter()
        .map(|end| probe_at(end, max_probe_tokens, chunk))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    const LF: u32 = 10;

    fn rule() -> BoundaryRule {
        BoundaryRule {
            triggers: Arc::new(vec![LF]),
            cadence: 64,
            tool_call_open: None,
            tool_call_close: None,
        }
    }

    #[test]
    fn a_short_probe_is_its_chunks_up_to_the_boundary() {
        assert_eq!(
            probe_at(40, 256, 32),
            ProbeWindow {
                head: None,
                tail: 0..40
            }
        );
    }

    /// Past the budget, the tail is the last eight chunks the boundary reaches
    /// into — the partial one included — and the two-chunk head comes back.
    #[test]
    fn a_long_probe_keeps_the_head_and_a_chunk_aligned_tail() {
        // 300 tokens reach into chunk 9 (of 0..10): tail chunks 2..10.
        assert_eq!(
            probe_at(300, 256, 32),
            ProbeWindow {
                head: Some(0..64),
                tail: 64..300
            }
        );
        // 1000 tokens: chunks 0..32, tail 24..32.
        let w = probe_at(1000, 256, 32);
        assert_eq!(w.head, Some(0..64));
        assert_eq!(w.tail, 768..1000);
        assert_eq!(w.len(), 64 + 232);
    }

    /// One chunk past the budget: the tail starts at chunk 1, so the head is
    /// that one chunk — never overlapping the tail.
    #[test]
    fn a_head_never_overlaps_the_tail() {
        let w = probe_at(260, 256, 32);
        assert_eq!(w.head, Some(0..32));
        assert_eq!(w.tail, 32..260);
    }

    /// A turn gets a window at every boundary and one at its end; one the rule
    /// never fires in is one window.
    #[test]
    fn every_boundary_and_the_end_get_a_window() {
        let tokens: Vec<u32> = (0..150).map(|i| 100 + i % 7).collect();
        let ends: Vec<usize> = probe_windows(&tokens, &rule(), 256, 32)
            .iter()
            .map(|w| w.tail.end)
            .collect();
        assert_eq!(ends, vec![64, 128, 150]);

        let short: Vec<u32> = (0..10).collect();
        assert_eq!(
            probe_windows(&short, &rule(), 256, 32),
            vec![ProbeWindow {
                head: None,
                tail: 0..10
            }]
        );
        assert!(probe_windows(&[], &rule(), 256, 32).is_empty());
    }

    #[test]
    fn a_boundary_at_the_end_is_not_doubled() {
        let tokens: Vec<u32> = (0..128).map(|i| 100 + i % 7).collect();
        let ends: Vec<usize> = probe_windows(&tokens, &rule(), 256, 32)
            .iter()
            .map(|w| w.tail.end)
            .collect();
        assert_eq!(ends, vec![64, 128]);
    }
}
