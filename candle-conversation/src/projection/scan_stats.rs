//! The shape of one group's raw scan scores for one probe — what a layer's
//! need for the projection is read from (`docs/provenance_adaptive_projection.md`
//! Concept B).
//!
//! Everything here is over the RAW ungated scores, before hit-level
//! normalization: normalization rescales every group's best member toward the
//! same band, which is exactly what hides whether a group has anything worth
//! carrying. And everything is an order statistic, never a sum, so a group
//! with ten thousand members reads the same as one with fifty when what stands
//! out of them is the same.

use serde::{Deserialize, Serialize};

/// Order statistics of one group's raw scores under one probe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScanStats {
    /// How many members were scored.
    pub members: u32,
    /// The best member's score.
    pub top: f32,
    /// The score 90 % of members are at or below (nearest rank).
    pub p90: f32,
    /// The typical member's score: the median (nearest rank, lower middle).
    pub median: f32,
}

/// What one group's scan looked like on this projection: the trailing window's
/// probe, and the turn's question when the group pins it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GroupScan {
    pub tail: ScanStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<ScanStats>,
}

/// [`ScanStats`] of `raw`; `None` when nothing was scored. Non-finite scores
/// are dropped rather than ordered.
pub fn scan_stats(raw: &[f32]) -> Option<ScanStats> {
    let mut sorted: Vec<f32> = raw.iter().copied().filter(|s| s.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    // Nearest rank: the smallest value with at least `q` of the members at or
    // below it.
    let rank = |q: f32| sorted[((q * n as f32).ceil() as usize).clamp(1, n) - 1];
    Some(ScanStats {
        members: n as u32,
        top: sorted[n - 1],
        p90: rank(0.9),
        median: rank(0.5),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_scored_has_no_stats() {
        assert_eq!(scan_stats(&[]), None);
        assert_eq!(scan_stats(&[f32::NAN]), None);
    }

    #[test]
    fn one_member_is_its_own_top_and_median() {
        assert_eq!(
            scan_stats(&[7.0]),
            Some(ScanStats {
                members: 1,
                top: 7.0,
                p90: 7.0,
                median: 7.0
            })
        );
    }

    /// Ten members 1..=10 in any order: nearest-rank p90 is the 9th, the
    /// median the 5th.
    #[test]
    fn order_statistics_use_nearest_rank() {
        let raw = [3.0, 10.0, 1.0, 8.0, 5.0, 2.0, 9.0, 4.0, 7.0, 6.0];
        assert_eq!(
            scan_stats(&raw),
            Some(ScanStats {
                members: 10,
                top: 10.0,
                p90: 9.0,
                median: 5.0
            })
        );
    }

    /// The statistics are order statistics, so more of the same members move
    /// nothing: a group read twice over reads the same.
    #[test]
    fn volume_does_not_move_the_statistics() {
        let once = [1.0, 2.0, 3.0, 4.0, 100.0];
        let twice: Vec<f32> = once.iter().chain(once.iter()).copied().collect();
        let (a, b) = (scan_stats(&once).unwrap(), scan_stats(&twice).unwrap());
        assert_eq!((a.top, a.p90, a.median), (b.top, b.p90, b.median));
        assert_eq!(b.members, 10);
    }

    #[test]
    fn non_finite_scores_are_dropped() {
        let s = scan_stats(&[1.0, f32::INFINITY, 3.0, f32::NAN]).unwrap();
        assert_eq!(s.members, 2);
        assert_eq!(s.top, 3.0);
    }
}
