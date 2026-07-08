//! Per-turn selection diagnostics (§10.8.4).
//!
//! Returned in `TurnResponse.selection_diagnostics` for every assistant
//! turn so the recall-stress harness can answer:
//!
//! 1. Which nodes did the score-density selection put into the slot?
//! 2. Why was each one chosen?  (hard anchor / recency decay / provenance
//!    score / coverage fill / refill)
//!
//! Both axes are needed to triage recall failures (§10.8.3): if the
//! planted turn made it in but the model couldn't use it, that's a
//! quantize / prompt issue; if the planted turn missed the selection,
//! the score-density rules are wrong.

use ahash::AHashMap;

use super::tree::NodeId;

/// How a node entered the selected set.
///
/// The first group is produced by the score-density selection over the summary
/// forest (`select_dense`, §8); the last two (`Recent` / `Historical`) by the
/// plain rule-based `conversation` selection used when a timeline has no summary
/// tree yet. Surfacing the tag per projected turn is what makes "why is this in
/// my context?" answerable from the GUI / persisted projection record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionOrigin {
    /// Not yet absorbed into the tree by the async summariser.  Lives
    /// in the foreground's `pending` queue and is injected verbatim
    /// into the slot.
    Pending,
    /// One of the last 3 binary leaves; included unconditionally
    /// regardless of provenance score (§8.2 hard anchor).
    HardAnchor,
    /// Provenance score was below the recency-decay floor, but the
    /// recency decay still made it competitive enough to win a slot.
    RecencyDecay,
    /// Provenance score won the slot — i.e. the provenance scan ruled this
    /// node relevant to the current probe Q.
    ProvenanceScore,
    /// Greedy fit left a coverage gap; the smallest covering node was
    /// added by step 4 of `select_dense` (§8.4).
    CoverageFill,
    /// Added by the step-5 multi-pass refill loop after some redundant
    /// ancestor was eliminated.
    Refill,
    /// Rule-based path (no summary tree): inside the inviolate recency
    /// window (`conversation.recent`) — always shown, regardless of score.
    Recent,
    /// Rule-based path (no summary tree): an older turn pulled back by
    /// relevance score (`conversation.historical_top_k`).
    Historical,
}

/// Per-turn selection diagnostics, attached to `TurnResponse`.
///
/// Always populated.  Cost is small (a few KB per turn even for
/// budget-of-thousands selections), so no feature gate.
#[derive(Debug, Clone, Default)]
pub struct SelectionDiagnostics {
    /// Every node the score-density selection placed in the slot, in
    /// chronological order (oldest first, recent last).
    pub selected_nodes: Vec<NodeId>,
    /// Per-node origin tag matching `selected_nodes` index-by-index.
    pub origins: Vec<SelectionOrigin>,
    /// Effective score (max of provenance and recency) that won the
    /// node its slot.  Useful for debugging tie-breaks.
    pub effective_scores: AHashMap<NodeId, f32>,
    /// Number of pending turns at the moment of selection.  Bigger
    /// pending → smaller selection region (§9 backpressure).
    pub pending_count: usize,
    /// Total token cost of the selected set (excludes pending).
    pub selected_tokens: u32,
    /// Layer window budget used (for diagnostics — informational only).
    pub budget: u32,
}

impl SelectionDiagnostics {
    pub fn new(budget: u32) -> Self {
        Self {
            budget,
            ..Self::default()
        }
    }

    /// Record one selected node.
    pub fn push(&mut self, id: NodeId, origin: SelectionOrigin, effective_score: f32, tokens: u32) {
        self.selected_nodes.push(id);
        self.origins.push(origin);
        self.effective_scores.insert(id, effective_score);
        self.selected_tokens = self.selected_tokens.saturating_add(tokens);
    }

    /// True iff the diagnostics record contains `id` in its selected
    /// set.
    pub fn contains(&self, id: NodeId) -> bool {
        self.selected_nodes.contains(&id)
    }

    /// Origin tag for a previously-recorded node, if present.
    pub fn origin_of(&self, id: NodeId) -> Option<SelectionOrigin> {
        self.selected_nodes
            .iter()
            .position(|n| *n == id)
            .map(|i| self.origins[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_diagnostics_default() {
        let d = SelectionDiagnostics::new(8000);
        assert_eq!(d.budget, 8000);
        assert!(d.selected_nodes.is_empty());
        assert_eq!(d.selected_tokens, 0);
        assert!(!d.contains(NodeId(1)));
        assert_eq!(d.origin_of(NodeId(1)), None);
    }

    #[test]
    fn push_records_in_order() {
        let mut d = SelectionDiagnostics::new(8000);
        d.push(NodeId(1), SelectionOrigin::HardAnchor, f32::INFINITY, 20);
        d.push(NodeId(2), SelectionOrigin::ProvenanceScore, 0.8, 30);
        d.push(NodeId(3), SelectionOrigin::CoverageFill, 0.1, 20);
        assert_eq!(d.selected_nodes, vec![NodeId(1), NodeId(2), NodeId(3)]);
        assert_eq!(
            d.origins,
            vec![
                SelectionOrigin::HardAnchor,
                SelectionOrigin::ProvenanceScore,
                SelectionOrigin::CoverageFill,
            ]
        );
        assert_eq!(d.selected_tokens, 70);
        assert_eq!(
            d.origin_of(NodeId(2)),
            Some(SelectionOrigin::ProvenanceScore)
        );
    }
}
