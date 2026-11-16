//! Every probe token's statistics in one scan — what the statistics mode of
//! the paged scan hands back (`docs/incremental_chunk_exclusion.md` §4.5): per
//! probe token, per fold group the launch scanned, per segment, the leading and
//! runner-up cases with their best agreements and the agreement sums its vote
//! would have been made of.

use candle_kernels::provenance::BdpStats;

/// See the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenStats {
    /// One row per (probe token, launch group, segment), in that order: token
    /// major, then group, then segment.
    pub rows: Vec<BdpStats>,
    /// The full-width tokens each probe contributed, in probe order.
    pub per_probe: Vec<usize>,
    /// The fold groups the launch scanned, in row order.
    pub groups: Vec<usize>,
    /// The tokens each segment scanned — its `n`.
    pub seg_tokens: Vec<u64>,
}

impl TokenStats {
    /// Nothing scanned: no row for any of `per_probe`'s tokens.
    pub fn none(per_probe: Vec<usize>) -> Self {
        Self {
            rows: Vec::new(),
            per_probe,
            groups: Vec::new(),
            seg_tokens: Vec::new(),
        }
    }

    /// Segments per token and group.
    pub fn n_segments(&self) -> usize {
        self.seg_tokens.len()
    }

    /// The row of probe token `token` (counted over every probe), the
    /// `place`-th launch group, segment `segment`; `None` when nothing was
    /// scanned.
    pub fn row(&self, token: usize, place: usize, segment: usize) -> Option<&BdpStats> {
        let (ng, ns) = (self.groups.len(), self.n_segments());
        self.rows.get((token * ng + place) * ns + segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(a1: u32) -> BdpStats {
        BdpStats {
            a1,
            ..BdpStats::default()
        }
    }

    /// **Rows are token major, then group, then segment.** Two tokens, two
    /// groups, three segments: row `(1, 0, 2)` is the 9th.
    #[test]
    fn rows_are_token_then_group_then_segment() {
        let t = TokenStats {
            rows: (0..12).map(stats).collect(),
            per_probe: vec![2],
            groups: vec![0, 2],
            seg_tokens: vec![5, 6, 7],
        };
        assert_eq!(t.n_segments(), 3);
        assert_eq!(t.row(1, 0, 2).map(|r| r.a1), Some(8));
        assert_eq!(t.row(0, 1, 1).map(|r| r.a1), Some(4));
        assert_eq!(t.row(2, 0, 0), None);
        assert_eq!(TokenStats::none(vec![3]).row(0, 0, 0), None);
    }
}
