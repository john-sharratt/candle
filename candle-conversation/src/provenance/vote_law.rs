//! How a probe token's agreements become votes in one competition.
//!
//! A competition whose answer is one clean match — a tool call against the
//! tool's past calls, a question against a file's calibration questions — is
//! served by one vote per token: `z × margin` for the leading case. A
//! competition that is vote-starved — a question against a corpus of code,
//! where the right file is one of many partial matches — is served by each
//! token voting for its best `k` cases, case `r` by `z_r × (a_r − a_k)`, every
//! kept vote raised to a power below one so a few loud votes do not drown many
//! quiet ones. Measured both ways in `docs/results/provenance_scoring_lab_2026-11-16.md`:
//! top-8 at power 0.5 lifts code retrieval's MRR from 0.235 to 0.663 and drops
//! tool selection's Top-1 from 80.0% to 63.7%, so the law is set per group.

/// See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoteLaw {
    /// Cases a probe token votes for, `1..=MAX_TOP_K`.
    pub top_k: usize,
    /// The power each kept vote is raised to before it is summed.
    pub power: f32,
}

impl VoteLaw {
    /// The most cases a token votes for — the kernels' `BDP_MAX_TOP_K`.
    pub const MAX_TOP_K: usize = 8;

    /// One `z × margin` vote per token, summed as it is.
    pub const SINGLE: VoteLaw = VoteLaw {
        top_k: 1,
        power: 1.0,
    };

    /// `v` as this law sums it. At power 1 it is `v` itself, not `v.powf(1)`,
    /// so a single-vote tally is the arithmetic it always was.
    #[inline]
    pub fn raise(&self, v: f32) -> f32 {
        if self.power == 1.0 {
            v
        } else {
            v.signum() * v.abs().powf(self.power)
        }
    }

    /// Whether the law is one a scan can run: `top_k` in `1..=MAX_TOP_K`, a
    /// finite positive power.
    pub fn is_valid(&self) -> bool {
        (1..=Self::MAX_TOP_K).contains(&self.top_k) && self.power.is_finite() && self.power > 0.0
    }
}

impl Default for VoteLaw {
    fn default() -> Self {
        Self::SINGLE
    }
}

/// A token's votes in one fold group under `law`: `case_max[c]` its best
/// agreement with case `c`, `mean` and `std` over every gallery token it was
/// compared with. The best `law.top_k` cases, ties to the lower case id, each
/// vote `z_r × (a_r − a_k)` with `a_k` the next case's agreement (0 when fewer
/// cases scored). A case at agreement 0 never places. At `top_k` 1 this is the
/// leader's `z × margin` over the runner-up.
pub fn law_votes(case_max: &[u32], mean: f32, std: f32, top_k: usize) -> Vec<(usize, f32)> {
    let k = top_k.clamp(1, VoteLaw::MAX_TOP_K);
    // The best k + 1, best first; a strictly greater agreement displaces, and
    // cases arrive ascending, so ties keep the lower id ahead.
    let mut best: Vec<(u32, usize)> = Vec::with_capacity(k + 1);
    for (c, &m) in case_max.iter().enumerate() {
        if m == 0 || (best.len() == k + 1 && m <= best[k].0) {
            continue;
        }
        let at = best.iter().position(|&(a, _)| m > a).unwrap_or(best.len());
        best.insert(at, (m, c));
        best.truncate(k + 1);
    }
    let floor = best.get(k).map_or(0, |b| b.0);
    best.iter()
        .take(k)
        .map(|&(a, c)| {
            let z = ((a as f32 - mean) / std).max(0.0);
            (c, z * a.saturating_sub(floor) as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A single vote is the leader's `z × margin`**: cases at 90, 70, 80
    /// over mean 50, spread 10 — case 0 by z 4 × (90 − 80).
    #[test]
    fn a_single_vote_is_the_leaders_margin_over_the_runner_up() {
        assert_eq!(law_votes(&[90, 70, 80], 50.0, 10.0, 1), vec![(0, 40.0)]);
    }

    /// **At k, each of the best k votes over the (k+1)-th**: case 0 by
    /// 4 × (90 − 70), case 2 by 3 × (80 − 70); case 1, the floor, by nothing.
    #[test]
    fn each_of_the_best_k_votes_over_the_next() {
        assert_eq!(
            law_votes(&[90, 70, 80], 50.0, 10.0, 2),
            vec![(0, 80.0), (2, 30.0)]
        );
    }

    /// **Fewer cases than k vote over a floor of 0**, and a case at 0 never
    /// places.
    #[test]
    fn fewer_scoring_cases_than_k_vote_over_zero() {
        assert_eq!(
            law_votes(&[0, 60, 0], 50.0, 10.0, 3),
            vec![(1, 60.0)],
            "z 1 × (60 − 0)"
        );
        assert!(law_votes(&[0, 0], 50.0, 10.0, 2).is_empty());
    }

    /// **Ties rank by case id**, as the kernels' ascending strict pass does:
    /// cases 1 and 2 tie at 80, so 1 leads and 2 is its runner-up, margin 0.
    #[test]
    fn ties_rank_by_case_id() {
        assert_eq!(law_votes(&[10, 80, 80], 50.0, 10.0, 1), vec![(1, 0.0)]);
        assert_eq!(
            law_votes(&[10, 80, 80], 50.0, 10.0, 2),
            vec![(1, 210.0), (2, 210.0)],
            "both over case 0's 10: z 3 × 70"
        );
    }

    /// **Power 1 sums a vote as it is; below 1 it flattens**, keeping the
    /// sign.
    #[test]
    fn a_law_raises_its_votes() {
        assert_eq!(VoteLaw::SINGLE.raise(9.0), 9.0);
        let root = VoteLaw {
            top_k: 8,
            power: 0.5,
        };
        assert_eq!(root.raise(9.0), 3.0);
        assert_eq!(root.raise(-4.0), -2.0);
        assert!(root.is_valid() && VoteLaw::SINGLE.is_valid());
        assert!(!VoteLaw {
            top_k: 9,
            power: 0.5
        }
        .is_valid());
        assert!(!VoteLaw {
            top_k: 0,
            power: 0.5
        }
        .is_valid());
        assert!(!VoteLaw {
            top_k: 2,
            power: 0.0
        }
        .is_valid());
    }
}
