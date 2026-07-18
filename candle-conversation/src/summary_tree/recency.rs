//! Recency score for the window's right edge (`docs/immutable_summary_forest.md`,
//! *Window of attention*).
//!
//! ```text
//!   recency_score(node) =
//!       +∞                   if node is one of the last `hard_anchor` Normal
//!                            turns — the recent-raw tail, injected verbatim
//!       +∞                   if node is one of the last `hard_anchor`
//!                            SummaryOfTurns leaves
//!       d^(k - hard_anchor)  if node is the kᵗʰ-most-recent SoT leaf,
//!                            k > hard_anchor
//!       0                    otherwise — older Normal turns, SoS internals,
//!                            unknown ids
//! ```
//!
//! Default `hard_anchor = 3`, `decay = 0.8`.
//!
//! Anchoring the newest **Normal turns** — not only the SoT leaves that
//! summarise them — is what makes recent context **verbatim**. The anchored
//! turns win a slot in [`select_dense`](super::select::select_dense)'s greedy
//! fit, and its redundant-ancestor step then drops their now-covered SoT
//! parents. Score only the leaves and the newest turns fall to `0.0`, lose the
//! fit, and the coverage-gap step papers over them with the very summaries that
//! compress them — so the live conversation never reaches the model at all.
//!
//! Composes with the provenance score via
//! `effective_score = max(provenance, recency)` — the provenance scan stays
//! authoritative for old high-relevance turns, while recent turns are
//! guaranteed a slot regardless of provenance.

use super::exchange::{self, Couplings};
use super::tree::NodeId;

/// Tunables for the recency-decay scorer.
#[derive(Debug, Clone, Copy)]
pub struct RecencyConfig {
    /// Number of rightmost Normal turns (and SoT leaves) that are hard-anchored
    /// (effective score `+∞`).  Default 3.
    pub hard_anchor: usize,
    /// Multiplicative decay rate `d` applied to SoT leaves beyond the hard
    /// anchor: `recency_score(k) = d^(k − hard_anchor)`.  Must be in `(0, 1)`.
    /// Default 0.8.
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

/// The first `chrono_normals` position inside the recent-raw tail: every Normal
/// at or after it belongs to one of the newest `hard_anchor` **exchanges**.
///
/// Counting turns instead of exchanges cuts tool round-trips in half at the
/// boundary — anchoring a `<tool_response>` whose call falls outside the window,
/// or a call whose result does. An exchange is one unit of recent context, so the
/// tail is measured in exchanges and its members are anchored together.
///
/// Computed once per selection and compared with `>=`, so scoring stays O(1) per
/// node rather than re-deriving the grouping for each.
///
/// `SummaryOfTurns` leaves need no equivalent: a leaf now covers exactly one
/// exchange, so the newest `hard_anchor` leaves *are* the newest `hard_anchor`
/// exchanges.
pub fn anchor_start(couplings: &Couplings, n_normals: usize, cfg: RecencyConfig) -> usize {
    if cfg.hard_anchor == 0 {
        // Nothing is anchored — no position can be >= this.
        return n_normals;
    }
    let groups = exchange::exchanges(couplings, n_normals);
    if groups.len() <= cfg.hard_anchor {
        // Fewer exchanges than the anchor: all of them are the recent tail.
        return 0;
    }
    groups[groups.len() - cfg.hard_anchor].start
}

/// Compute the recency score for `node`.
///
/// `chrono_normals` is the chronological list of `Normal` turn sub-leaves and
/// `chrono_leaves` the chronological list of `SummaryOfTurns` binary leaves; in
/// both the **last** element is the newest.
///
/// The newest `hard_anchor` Normal turns score `+∞` (the verbatim recent tail).
/// Older Normal turns score `0.0` — provenance decides whether they return, and
/// the coverage gap-fill otherwise covers them with their summaries. SoT leaves
/// keep the anchor-then-decay curve so mid-history summaries outrank ancient
/// ones when provenance is silent. Everything else (SoS internals, unknown ids)
/// scores `0.0`.
pub fn recency_score(
    node: NodeId,
    chrono_normals: &[NodeId],
    chrono_leaves: &[NodeId],
    anchor_start: usize,
    cfg: RecencyConfig,
) -> f32 {
    // The recent-raw tail — anchored so the live conversation is injected
    // verbatim instead of as its own summaries. `anchor_start` is measured in
    // whole exchanges (see `anchor_start`), so a tool round-trip is anchored as
    // one unit rather than being cut at the `hard_anchor`-th turn.
    if let Some(idx) = chrono_normals.iter().rposition(|n| *n == node) {
        return if idx >= anchor_start {
            f32::INFINITY
        } else {
            0.0
        };
    }
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

    fn couplings(from: &[u32]) -> Couplings {
        from.iter().copied().collect()
    }

    /// Score with no couplings — every turn is its own exchange, so the tail is
    /// measured in turns exactly as it was before exchanges existed.
    fn score(node: NodeId, normals: &[NodeId], leaves: &[NodeId], cfg: RecencyConfig) -> f32 {
        let start = anchor_start(&couplings(&[]), normals.len(), cfg);
        recency_score(node, normals, leaves, start, cfg)
    }

    /// No normals in play — the leaf curve is unchanged.
    fn leaves_only(node: NodeId, chrono: &[NodeId], cfg: RecencyConfig) -> f32 {
        recency_score(node, &[], chrono, 0, cfg)
    }

    #[test]
    fn unknown_node_scores_zero() {
        let chrono = vec![nid(1), nid(2), nid(3)];
        let cfg = RecencyConfig::default();
        assert_eq!(leaves_only(nid(99), &chrono, cfg), 0.0);
    }

    #[test]
    fn last_three_are_hard_anchored() {
        let chrono = vec![nid(1), nid(2), nid(3), nid(4), nid(5)];
        let cfg = RecencyConfig::default();
        assert!(leaves_only(nid(5), &chrono, cfg).is_infinite());
        assert!(leaves_only(nid(4), &chrono, cfg).is_infinite());
        assert!(leaves_only(nid(3), &chrono, cfg).is_infinite());
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
        assert!((leaves_only(nid(4), &chrono, cfg) - 0.8).abs() < 1e-5);
        assert!((leaves_only(nid(3), &chrono, cfg) - 0.64).abs() < 1e-5);
        assert!((leaves_only(nid(2), &chrono, cfg) - 0.512).abs() < 1e-5);
        assert!((leaves_only(nid(1), &chrono, cfg) - 0.4096).abs() < 1e-4);
    }

    #[test]
    fn smaller_anchor_works() {
        let chrono = vec![nid(1), nid(2), nid(3)];
        let cfg = RecencyConfig {
            hard_anchor: 1,
            decay: 0.5,
        };
        // Only nid(3) is hard.
        assert!(leaves_only(nid(3), &chrono, cfg).is_infinite());
        // nid(2) at dist 1, k=1 → 0.5.
        assert!((leaves_only(nid(2), &chrono, cfg) - 0.5).abs() < 1e-6);
        // nid(1) at dist 2, k=2 → 0.25.
        assert!((leaves_only(nid(1), &chrono, cfg) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn anchor_larger_than_leaf_count_anchors_all() {
        let chrono = vec![nid(1), nid(2)];
        let cfg = RecencyConfig::default(); // anchor = 3
        assert!(leaves_only(nid(1), &chrono, cfg).is_infinite());
        assert!(leaves_only(nid(2), &chrono, cfg).is_infinite());
    }

    #[test]
    fn empty_chrono_is_safe() {
        let chrono: Vec<NodeId> = Vec::new();
        let cfg = RecencyConfig::default();
        assert_eq!(leaves_only(nid(1), &chrono, cfg), 0.0);
    }

    /// The recent-raw tail: the newest `hard_anchor` Normal turns are anchored
    /// so the live conversation is selected verbatim rather than as summaries.
    /// This is the regression guard for "the window had no recent conversation
    /// in it at all — it was replaced entirely by summaries".
    #[test]
    fn newest_normal_turns_are_hard_anchored() {
        let normals = vec![nid(10), nid(20), nid(30), nid(40), nid(50)];
        let leaves = vec![nid(11), nid(21), nid(31), nid(41), nid(51)];
        let cfg = RecencyConfig::default(); // anchor = 3
                                            // The three newest turns are anchored at full fidelity.
        assert!(score(nid(50), &normals, &leaves, cfg).is_infinite());
        assert!(score(nid(40), &normals, &leaves, cfg).is_infinite());
        assert!(score(nid(30), &normals, &leaves, cfg).is_infinite());
        // Older turns fall back to provenance / the coarse peak cover.
        assert_eq!(score(nid(20), &normals, &leaves, cfg), 0.0);
        assert_eq!(score(nid(10), &normals, &leaves, cfg), 0.0);
    }

    /// The bug this fixes. Five turns where 1↔2 and 3↔4 are tool round-trips, so
    /// the exchanges are `[0] [1,2] [3,4]`. Counting *turns*, an anchor of 3 pins
    /// turns 2,3,4 — cutting exchange `[1,2]` in half and anchoring turn 2's
    /// `<tool_response>` while the call that requested it (turn 1) falls outside
    /// the tail. Counting *exchanges*, all three are pinned whole.
    #[test]
    fn the_tail_is_measured_in_whole_exchanges() {
        let normals: Vec<NodeId> = (0..5).map(nid).collect();
        let c = couplings(&[1, 3]);
        let cfg = RecencyConfig::default(); // anchor = 3 exchanges
        let start = anchor_start(&c, normals.len(), cfg);
        assert_eq!(start, 0, "3 exchanges exist, so all of them are the tail");
        for n in &normals {
            assert!(
                recency_score(*n, &normals, &[], start, cfg).is_infinite(),
                "{n:?} should be anchored — no round-trip may be cut in half"
            );
        }
    }

    /// With more exchanges than the anchor, the boundary lands on an exchange
    /// edge — never mid-round-trip.
    #[test]
    fn the_tail_boundary_lands_on_an_exchange_edge() {
        // Exchanges: [0] [1,2] [3,4] [5,6] — anchor 3 keeps the last three.
        let normals: Vec<NodeId> = (0..7).map(nid).collect();
        let c = couplings(&[1, 3, 5]);
        let cfg = RecencyConfig::default();
        let start = anchor_start(&c, normals.len(), cfg);
        assert_eq!(start, 1, "tail starts at the head of exchange [1,2]");
        // Turn 0 is outside; turns 1..6 are the three whole exchanges.
        assert_eq!(recency_score(nid(0), &normals, &[], start, cfg), 0.0);
        for i in 1..7u32 {
            assert!(recency_score(nid(i), &normals, &[], start, cfg).is_infinite());
        }
    }

    /// A zero anchor pins nothing, even with couplings present.
    #[test]
    fn a_zero_anchor_anchors_nothing() {
        let cfg = RecencyConfig {
            hard_anchor: 0,
            decay: 0.8,
        };
        let start = anchor_start(&couplings(&[0]), 3, cfg);
        assert_eq!(start, 3, "no position can be >= the turn count");
        let normals: Vec<NodeId> = (0..3).map(nid).collect();
        for n in &normals {
            assert_eq!(recency_score(*n, &normals, &[], start, cfg), 0.0);
        }
    }

    /// A Normal turn is scored as a turn, never mistaken for a leaf: membership
    /// in `chrono_normals` wins even though the leaf list is also consulted.
    #[test]
    fn normals_are_scored_before_leaves() {
        let normals = vec![nid(1), nid(2)];
        let leaves = vec![nid(3), nid(4)];
        let cfg = RecencyConfig::default();
        // Both normals are within the anchor → +∞.
        assert!(score(nid(1), &normals, &leaves, cfg).is_infinite());
        assert!(score(nid(2), &normals, &leaves, cfg).is_infinite());
        // Leaves keep their own curve.
        assert!(score(nid(4), &normals, &leaves, cfg).is_infinite());
    }
}
