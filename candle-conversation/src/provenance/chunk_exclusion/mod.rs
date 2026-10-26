//! Anomalous-chunk exclusion: the gallery chunks that win votes on every
//! file's questions, cut from the scan so their votes go to what agrees next
//! best (`docs/provenance_scoped_signatures.md` §2.6).
//!
//! A chunk of 32 signature tokens is generic when it wins votes on questions
//! about other files — the request template, the tool-call JSON, a pager's
//! footer. Down-weighting such a chunk only shrinks its votes; cutting it from
//! the gallery hands each to the probe token's next-best agreement, and that
//! is what lifts recall. Which chunks go is learned from the files'
//! calibration questions, never from the gallery alone: a gallery statistic
//! (verbatim shared runs) lost at every threshold measured.

pub mod anomaly;

use std::collections::HashMap;
use std::ops::Range;

use crate::persistence::streams::StreamId;

/// The 32-token chunk exclusion is learned and cut at, as signature tokens.
pub const CHUNK: usize = 32;

/// The token ranges cut from each turn's gallery windows, and how many
/// calibration questions they were learned from. Held in memory; empty until
/// the first is learned, and learned again once the questions to learn from
/// have moved by a tenth.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkExclusion {
    by_stream: HashMap<StreamId, Vec<Range<usize>>>,
    questions: usize,
}

impl ChunkExclusion {
    /// This exclusion, as learned from `questions` calibration questions.
    pub fn learned_from(self, questions: usize) -> Self {
        ChunkExclusion { questions, ..self }
    }

    /// Whether this exclusion still stands for a sample of `questions`: it
    /// was learned from some, and the sample has moved by less than a tenth.
    pub fn stands_for(&self, questions: usize) -> bool {
        self.questions > 0 && questions.abs_diff(self.questions) * 10 < self.questions
    }

    /// An exclusion of `chunks`, each `(turn, chunk index, turn length)`.
    pub fn of_chunks(chunks: impl IntoIterator<Item = (StreamId, usize, usize)>) -> Self {
        let mut by_stream: HashMap<StreamId, Vec<Range<usize>>> = HashMap::new();
        for (sid, c, len) in chunks {
            let start = c * CHUNK;
            if start < len {
                by_stream
                    .entry(sid)
                    .or_default()
                    .push(start..((c + 1) * CHUNK).min(len));
            }
        }
        for ranges in by_stream.values_mut() {
            ranges.sort_by_key(|r| r.start);
        }
        ChunkExclusion {
            by_stream,
            questions: 0,
        }
    }

    /// The ranges cut from turn `sid`, ascending.
    pub fn of(&self, sid: StreamId) -> &[Range<usize>] {
        self.by_stream.get(&sid).map_or(&[], Vec::as_slice)
    }

    /// Chunks cut, over every turn.
    pub fn chunks(&self) -> usize {
        self.by_stream.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_stream.is_empty()
    }
}

/// `[start, end)` less every range in `cut`: what is left, in order.
pub fn uncut(start: usize, end: usize, cut: &[Range<usize>]) -> Vec<(usize, usize)> {
    let mut cuts: Vec<&Range<usize>> = cut
        .iter()
        .filter(|r| r.start < end && r.end > start)
        .collect();
    cuts.sort_by_key(|r| r.start);
    let mut left = Vec::new();
    let mut at = start;
    for r in cuts {
        if r.start > at {
            left.push((at, r.start));
        }
        at = at.max(r.end);
    }
    if at < end {
        left.push((at, end));
    }
    left
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    /// A chunk is its 32 tokens, the last one cut short by its turn's end; a
    /// chunk past the end is nothing.
    #[test]
    fn a_chunk_is_its_tokens_cut_at_the_turns_end() {
        let (a, b) = (StreamId(1), StreamId(2));
        let x = ChunkExclusion::of_chunks([(a, 2, 100), (a, 0, 100), (b, 1, 40), (b, 5, 40)]);
        assert_eq!(x.of(a), &[0..32, 64..96]);
        assert_eq!(x.of(b), &[32..40]);
        assert_eq!(x.of(StreamId(3)), &[] as &[Range<usize>]);
        assert_eq!(x.chunks(), 3);
    }

    /// **An exclusion stands until its questions move by a tenth**, either
    /// way; one learned from none never stands.
    #[test]
    fn an_exclusion_stands_until_its_questions_move_by_a_tenth() {
        let x = ChunkExclusion::default().learned_from(1000);
        assert!(x.stands_for(1000));
        assert!(x.stands_for(1099));
        assert!(!x.stands_for(1100));
        assert!(x.stands_for(901));
        assert!(!x.stands_for(900));
        assert!(!ChunkExclusion::default().stands_for(0));
        assert!(!ChunkExclusion::default().stands_for(10));
    }

    /// **A cut leaves what lies outside it, in order**: overlapping and
    /// out-of-order cuts, one past the end, one covering the start.
    #[test]
    fn a_window_less_its_cuts_keeps_what_lies_outside_them() {
        assert_eq!(uncut(0, 100, &[]), vec![(0, 100)]);
        assert_eq!(
            uncut(0, 100, &[64..96, 32..40, 36..48]),
            vec![(0, 32), (48, 64), (96, 100)]
        );
        assert_eq!(uncut(10, 50, &[0..20, 40..200]), vec![(20, 40)]);
        assert_eq!(uncut(10, 50, &[0..60]), Vec::<(usize, usize)>::new());
        assert_eq!(uncut(10, 50, &[60..70]), vec![(10, 50)]);
    }
}
