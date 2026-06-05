//! Recency score for right-edge leaves (§8.2).
//!
//! ```text
//!   recency_score(node) =
//!       +∞           if node ∈ last `hard_anchor` right-edge leaves
//!       d^(k - hard_anchor)  if node is the kᵗʰ-most-recent leaf,
//!                            k > hard_anchor
//!       0            if node is not a SummaryOfTurns binary leaf
//! ```
//!
//! Default `hard_anchor = 3`, `decay = 0.8`.
//!
//! Composes with the provenance score via
//! `effective_score = max(provenance, recency)` — the provenance scan
//! is still authoritative for old high-relevance turns, while recent
//! turns are guaranteed a slot regardless of provenance.

use super::tree::NodeId;

/// Tunables for the recency-decay scorer.  Defaults match the design
/// document's resolved §12 parameters.
#[derive(Debug, Clone, Copy)]
pub struct RecencyConfig {
    /// Number of rightmost binary leaves that are hard-anchored
    /// (effective score `+∞`).  Default 3.
    pub hard_anchor: usize,
    /// Multiplicative decay rate `d` applied to leaves beyond the hard
    /// anchor: `recency_score(k) = d^(k − hard_anchor)`.  Must be in
    /// `(0, 1)`.  Default 0.8.
    pub decay: f32,
}

impl Default for RecencyConfig {
    fn default() -> Self {
        Self {
            hard_anchor: 3,
            decay: 0.8,
        }
    }
}

/// Compute the recency score for `node` given a chronological list of
/// `SummaryOfTurns` binary leaves (`chrono_leaves`) where the **last**
/// element is the newest.
///
/// Returns `f32::INFINITY` for hard-anchored leaves, the decayed score
/// for older right-edge leaves, and `0.0` for everything else
/// (including non-leaf nodes and unknown ids).
pub fn recency_score(node: NodeId, chrono_leaves: &[NodeId], cfg: RecencyConfig) -> f32 {
    let len = chrono_leaves.len();
    // Distance from the end: 0 = newest, 1 = second-newest, …
    let dist_from_end = match chrono_leaves.iter().rposition(|n| *n == node) {
        Some(idx) => len - 1 - idx,
        None => return 0.0,
    };
    if dist_from_end < cfg.hard_anchor {
        f32::INFINITY
    } else {
        let k = dist_from_end - cfg.hard_anchor + 1; // 1, 2, 3, …
        cfg.decay.powi(k as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(x: u32) -> NodeId {
        NodeId(x)
    }

    #[test]
    fn unknown_node_scores_zero() {
        let chrono = vec![nid(1), nid(2), nid(3)];
        let cfg = RecencyConfig::default();
        assert_eq!(recency_score(nid(99), &chrono, cfg), 0.0);
    }

    #[test]
    fn last_three_are_hard_anchored() {
        let chrono = vec![nid(1), nid(2), nid(3), nid(4), nid(5)];
        let cfg = RecencyConfig::default();
        assert!(recency_score(nid(5), &chrono, cfg).is_infinite());
        assert!(recency_score(nid(4), &chrono, cfg).is_infinite());
        assert!(recency_score(nid(3), &chrono, cfg).is_infinite());
    }

    #[test]
    fn beyond_anchor_decays_geometrically() {
        let chrono = vec![nid(1), nid(2), nid(3), nid(4), nid(5), nid(6), nid(7)];
        let cfg = RecencyConfig::default();
        // 7 is index 6 (dist 0), 6 dist 1, 5 dist 2 — those are hard.
        // 4 (dist 3) → k=1 → 0.8^1 = 0.8
        // 3 (dist 4) → k=2 → 0.8^2 = 0.64
        // 2 (dist 5) → k=3 → 0.8^3 = 0.512
        // 1 (dist 6) → k=4 → 0.8^4 = 0.4096
        assert!((recency_score(nid(4), &chrono, cfg) - 0.8).abs() < 1e-5);
        assert!((recency_score(nid(3), &chrono, cfg) - 0.64).abs() < 1e-5);
        assert!((recency_score(nid(2), &chrono, cfg) - 0.512).abs() < 1e-5);
        assert!((recency_score(nid(1), &chrono, cfg) - 0.4096).abs() < 1e-4);
    }

    #[test]
    fn smaller_anchor_works() {
        let chrono = vec![nid(1), nid(2), nid(3)];
        let cfg = RecencyConfig {
            hard_anchor: 1,
            decay: 0.5,
        };
        // Only nid(3) is hard.
        assert!(recency_score(nid(3), &chrono, cfg).is_infinite());
        // nid(2) at dist 1, k=1 → 0.5.
        assert!((recency_score(nid(2), &chrono, cfg) - 0.5).abs() < 1e-6);
        // nid(1) at dist 2, k=2 → 0.25.
        assert!((recency_score(nid(1), &chrono, cfg) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn anchor_larger_than_leaf_count_anchors_all() {
        let chrono = vec![nid(1), nid(2)];
        let cfg = RecencyConfig::default(); // anchor = 3
        assert!(recency_score(nid(1), &chrono, cfg).is_infinite());
        assert!(recency_score(nid(2), &chrono, cfg).is_infinite());
    }

    #[test]
    fn empty_chrono_is_safe() {
        let chrono: Vec<NodeId> = Vec::new();
        let cfg = RecencyConfig::default();
        assert_eq!(recency_score(nid(1), &chrono, cfg), 0.0);
    }
}
