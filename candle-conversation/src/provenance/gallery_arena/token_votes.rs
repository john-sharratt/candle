//! A paged scan's raw output: every probe token's vote in every fold group and
//! segment, before the needle gate and the tally
//! ([`GalleryArena::scan_votes`](super::GalleryArena::scan_votes)).
//!
//! The live scan gates and tallies these on the host as soon as the launch
//! returns. An offline harness keeps them, and weighs, gates and folds them
//! under as many settings as it likes for the cost of one launch.

/// Every probe token's `(case, vote)` per fold group and segment, token-major:
/// row `(token · n_groups + group) · n_segments + segment`, the probes' tokens
/// laid end to end in probe order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenVotes {
    /// The winning case, global over the scan's segments; negative where the
    /// token voted for nothing.
    pub case: Vec<i32>,
    /// The `z × margin` vote the winner took.
    pub vote: Vec<f32>,
    /// Tokens each probe contributed (a token narrower than the gallery's fold
    /// is not scanned).
    pub per_probe: Vec<usize>,
    pub n_groups: usize,
    pub n_segments: usize,
}

impl TokenVotes {
    /// No probe token scored: `per_probe` tokens each, all of them empty.
    pub fn none(n_probes: usize, n_groups: usize, n_segments: usize) -> Self {
        TokenVotes {
            case: Vec::new(),
            vote: Vec::new(),
            per_probe: vec![0; n_probes],
            n_groups,
            n_segments,
        }
    }

    /// The first token of probe `p` in the end-to-end token order.
    pub fn first_token(&self, p: usize) -> usize {
        self.per_probe[..p].iter().sum()
    }

    /// Token `t` (end-to-end)'s winning case and vote in `group` and
    /// `segment`, or `None` where it voted for nothing.
    pub fn at(&self, t: usize, group: usize, segment: usize) -> Option<(usize, f32)> {
        let i = (t * self.n_groups + group) * self.n_segments + segment;
        let c = *self.case.get(i)?;
        (c >= 0).then(|| (c as usize, self.vote[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows are token-major, then group, then segment; a negative case is no
    /// vote; probes' tokens run end to end.
    #[test]
    fn a_row_is_token_then_group_then_segment() {
        let v = TokenVotes {
            // 3 tokens × 2 groups × 2 segments.
            case: vec![0, 2, -1, 3, 1, -1, 0, 2, 1, 3, -1, 2],
            vote: vec![1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 12.0],
            per_probe: vec![1, 2],
            n_groups: 2,
            n_segments: 2,
        };
        assert_eq!(v.at(0, 0, 1), Some((2, 2.0)));
        assert_eq!(v.at(0, 1, 0), None);
        assert_eq!(v.at(1, 1, 0), Some((0, 7.0)));
        assert_eq!(v.at(2, 1, 1), Some((2, 12.0)));
        assert_eq!(v.first_token(1), 1);
        assert_eq!(v.at(3, 0, 0), None, "past the end");
        assert_eq!(TokenVotes::none(2, 3, 1).per_probe, vec![0, 0]);
    }
}
