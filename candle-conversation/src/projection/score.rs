//! [`ScoreFormula::aggregate`] — turn scores → single group score.
//!
//! Each layer declares one [`ScoreFormula`]; the formula is applied to the
//! turn scores of each group inside that layer to produce a derived
//! per-group score. That derived score drives:
//!
//! 1. **Layer-level threshold gating** — groups whose derived score is
//!    below their layer's `score_threshold` are dropped wholesale.
//! 2. **Emission ordering** — within a layer, groups are sorted by
//!    ascending derived score so higher-scored groups appear LAST in the
//!    emitted projection (closer to the model's recency bias).
//!
//! # Choosing a formula
//!
//! ```text
//!   max         — robust to noise; one strong turn elevates the group
//!                 (the natural default for substrate-style salience).
//!   sum         — large groups dominate; rewards depth.
//!   mean        — penalises noisy groups (low-scoring tail drags the mean).
//!   top_k_mean  — peak-aware but smoothed against single outliers.
//!   count       — score-independent; pure "how big is this group".
//! ```
//!
//! All formulas return `0.0` for an empty input slice (a group with no
//! eligible turns has no meaningful aggregated score).

use super::schema::ScoreFormula;

impl ScoreFormula {
    /// Aggregate a slice of turn scores into a single group score.
    ///
    /// Empty slice returns `0.0` for all formulas (a group with no eligible
    /// turns has no meaningful score). For `TopKMean`, `k` is silently
    /// clamped to `scores.len()` if larger.
    pub fn aggregate(&self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        match self {
            ScoreFormula::Max => scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            ScoreFormula::Sum => scores.iter().copied().sum(),
            ScoreFormula::Mean => scores.iter().copied().sum::<f32>() / scores.len() as f32,
            ScoreFormula::TopKMean { k } => {
                let mut sorted = scores.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let take = (*k).min(sorted.len());
                sorted[..take].iter().copied().sum::<f32>() / take as f32
            }
            ScoreFormula::Count => scores.len() as f32,
            // At the group level, span of a turn list = max per-turn span score.
            // The actual span computation happens inside the Aggregator per turn;
            // here we just pick the most span-relevant turn to represent the group.
            ScoreFormula::Span { .. } => scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            // Like Span: the per-turn pertok_excess is computed in the
            // Aggregator; the group is represented by its strongest turn.
            ScoreFormula::PerTokenExcess => scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_basic() {
        assert_eq!(ScoreFormula::Max.aggregate(&[0.3, 0.9, 0.5]), 0.9);
    }

    #[test]
    fn max_empty() {
        assert_eq!(ScoreFormula::Max.aggregate(&[]), 0.0);
    }

    #[test]
    fn sum_basic() {
        let s = ScoreFormula::Sum.aggregate(&[0.2, 0.3, 0.5]);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_basic() {
        let s = ScoreFormula::Mean.aggregate(&[0.0, 1.0]);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mean_empty() {
        assert_eq!(ScoreFormula::Mean.aggregate(&[]), 0.0);
    }

    #[test]
    fn top_k_mean_k_gt_len() {
        // k=10 but only 3 scores → mean of all 3
        let s = ScoreFormula::TopKMean { k: 10 }.aggregate(&[0.9, 0.5, 0.1]);
        assert!((s - (0.9 + 0.5 + 0.1) / 3.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_mean_picks_top() {
        let s = ScoreFormula::TopKMean { k: 2 }.aggregate(&[0.1, 0.9, 0.5]);
        assert!((s - (0.9 + 0.5) / 2.0).abs() < 1e-5);
    }

    #[test]
    fn count_ignores_values() {
        assert_eq!(ScoreFormula::Count.aggregate(&[0.0, 0.0, 1.0, 2.0]), 4.0);
    }

    #[test]
    fn count_empty() {
        assert_eq!(ScoreFormula::Count.aggregate(&[]), 0.0);
    }
}
