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
//!
//! It is learned incrementally (`docs/incremental_chunk_exclusion.md`): a
//! sample of the questions ([`sample`]), each sample token's statistics kept
//! as an exact mergeable summary ([`stats`]), in keyed records the substrate
//! replicates ([`records`]).

pub mod anomaly;
pub mod records;
pub mod sample;
pub mod stats;

use std::collections::HashMap;
use std::ops::Range;

use crate::persistence::streams::StreamId;

/// The 32-token chunk exclusion is learned and cut at, as signature tokens.
pub const CHUNK: usize = 32;

/// The token ranges cut from each turn's gallery windows. Installed from the
/// group's replicated cut record ([`records::CutRecord`]); empty until one is
/// written.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkExclusion {
    by_stream: HashMap<StreamId, Vec<Range<usize>>>,
}

impl ChunkExclusion {
    /// An exclusion of the `(turn, start, end)` token ranges a cut record
    /// holds.
    pub fn of_ranges(ranges: impl IntoIterator<Item = (StreamId, u32, u32)>) -> Self {
        let mut by_stream: HashMap<StreamId, Vec<Range<usize>>> = HashMap::new();
        for (sid, start, end) in ranges {
            if start < end {
                by_stream
                    .entry(sid)
                    .or_default()
                    .push(start as usize..end as usize);
            }
        }
        for ranges in by_stream.values_mut() {
            ranges.sort_by_key(|r| r.start);
        }
        ChunkExclusion { by_stream }
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

    /// **A cut record's ranges are the exclusion**, sorted per turn; an empty
    /// range is nothing, and a turn with none cuts nothing.
    #[test]
    fn a_cut_records_ranges_are_the_exclusion() {
        let (a, b) = (StreamId(1), StreamId(2));
        let x = ChunkExclusion::of_ranges([(a, 64, 96), (b, 32, 40), (a, 0, 32), (b, 5, 5)]);
        assert_eq!(x.of(a), &[0..32, 64..96]);
        assert_eq!(x.of(b), &[32..40]);
        assert_eq!(x.of(StreamId(3)), &[] as &[Range<usize>]);
        assert_eq!(x.chunks(), 3);
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
