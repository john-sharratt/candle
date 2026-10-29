//! A paged scan's raw output: every probe token's vote in every fold group and
//! segment, before the needle gate and the tally
//! ([`GalleryArena::scan_votes`](super::GalleryArena::scan_votes)).
//!
//! The live scan gates and tallies these on the host as soon as the launch
//! returns. An offline harness keeps them, and weighs, gates and folds them
//! under as many settings as it likes for the cost of one launch.
//!
//! Rows always cover every fold group, though a launch scans only the groups
//! its segments weigh (`docs/gallery_residency.md` §4): a group it did not scan
//! voted for nothing, which is what weighing it zero tallies.

use super::groups::GroupSet;

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

/// A launch's rows over its `launch` groups, as rows over every one of
/// `n_groups`: each scanned group's row where it stands, every other group
/// voting for nothing (case `-1`, vote `0`). Rows are token-major, then group,
/// then segment, on both sides.
pub(super) fn widen_rows(
    case: &[i32],
    vote: &[f32],
    launch: GroupSet,
    n_groups: usize,
    n_segments: usize,
) -> (Vec<i32>, Vec<f32>) {
    let per_token = launch.len() * n_segments;
    if per_token == 0 {
        return (Vec::new(), Vec::new());
    }
    if launch == GroupSet::all(n_groups) {
        return (case.to_vec(), vote.to_vec());
    }
    let n_tokens = case.len() / per_token;
    let mut out_case = vec![-1i32; n_tokens * n_groups * n_segments];
    let mut out_vote = vec![0f32; out_case.len()];
    for t in 0..n_tokens {
        for (place, g) in launch.iter().enumerate() {
            let from = (t * launch.len() + place) * n_segments;
            let to = (t * n_groups + g) * n_segments;
            out_case[to..to + n_segments].copy_from_slice(&case[from..from + n_segments]);
            out_vote[to..to + n_segments].copy_from_slice(&vote[from..from + n_segments]);
        }
    }
    (out_case, out_vote)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A launch over groups 1 and 2 reads back as rows over all three**, group
    /// 0 voting for nothing: 2 tokens × 2 segments.
    #[test]
    fn a_narrow_launch_widens_to_every_group() {
        let launch = GroupSet::default().with(1).with(2);
        // token 0: g1 [s0 s1], g2 [s0 s1]; token 1 likewise.
        let case = [10, 11, 20, 21, 12, 13, 22, 23];
        let vote = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (c, v) = widen_rows(&case, &vote, launch, 3, 2);
        assert_eq!(c, [-1, -1, 10, 11, 20, 21, -1, -1, 12, 13, 22, 23]);
        assert_eq!(
            v,
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]
        );
        let full = GroupSet::all(3);
        assert_eq!(widen_rows(&c, &v, full, 3, 2), (c.clone(), v.clone()));
    }

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
