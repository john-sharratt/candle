//! Selection policy — the belief-update + eviction + budget knobs a projection
//! node uses to turn per-candidate provenance scores into a selected set.
//!
//! Replaces the old three-band `depth_weights` combine. A node (layer, section
//! collection, or turn group) carries a [`SelectionPolicy`]: a [`PolicyConfig`]
//! (the RelLeak belief + hysteresis + budget from the `SectionSelector`) plus an
//! optional gather-scope **tag filter**. Nodes inherit their parent's policy when
//! they declare none; a schema-level default covers the root.
//!
//! The concrete belief/selection mechanism lives in
//! [`crate::provenance::selection`]; this module is the *schema-side* description
//! that the runtime converts into per-slot `SectionPolicy` + `GroupBudget`.
//!
//! See `docs/tool_selection_provenance_results.md` §24.

/// A named, tuned policy preset (the §24.6 recommendations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyPreset {
    /// Stable, correct, small tool scope: β0.40, min 1000 / evict 750, budget
    /// 1..3. Thresholds are on the `z × margin` hybrid scorer (§24.7) with a
    /// 256-token probe window (`reproject_max_probe_tokens`) and the needle gate
    /// (§24.8); the §80.2 sweep derives them for **100% recall** — the true tool is
    /// always Top-3, and `min 1000` sits in the 100%-recall band (a wide margin
    /// below the ~1100 recall floor) while trimming the set to ~0.4 false
    /// positives, `evict 750` below that floor so a correct pick is never evicted.
    /// A committed tool is held stable across its `<tool_call>` block by the
    /// scheduler suppressing reprojection there, not by pinning the selection.
    /// Default for the `tools` collection.
    CommittedToolScope,
    /// Recall over set size: β0.40, min 40 / evict 20, budget 1..5. ~99.7% recall
    /// with the weak tail pruned to ~4 members.
    HighRecallScope,
    /// One pick: β0.40, no threshold, budget 1..1.
    SinglePick,
}

impl PolicyPreset {
    /// The tuned configuration for this preset.
    pub fn config(self) -> PolicyConfig {
        match self {
            PolicyPreset::CommittedToolScope => PolicyConfig {
                beta: 0.40,
                min_score: 1000.0,
                evict_score: 750.0,
                budget_min: 1,
                budget_max: 3,
            },
            PolicyPreset::HighRecallScope => PolicyConfig {
                beta: 0.40,
                min_score: 40.0,
                evict_score: 20.0,
                budget_min: 1,
                budget_max: 5,
            },
            PolicyPreset::SinglePick => PolicyConfig {
                beta: 0.40,
                min_score: 0.0,
                evict_score: 0.0,
                budget_min: 1,
                budget_max: 1,
            },
        }
    }

    /// Parse a snake_case preset name.
    pub fn from_name(name: &str) -> Option<PolicyPreset> {
        match name {
            "committed_tool_scope" => Some(PolicyPreset::CommittedToolScope),
            "high_recall_scope" => Some(PolicyPreset::HighRecallScope),
            "single_pick" => Some(PolicyPreset::SinglePick),
            _ => None,
        }
    }
}

use crate::provenance::{GroupBudget, SectionPolicy};

/// The concrete belief/selection knobs, resolved from a preset ± overrides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyConfig {
    /// RelLeak leak fraction (per-section decay rate).
    pub beta: f32,
    /// Confidence a candidate must reach to be selected.
    pub min_score: f32,
    /// Confidence below which a selected candidate is evicted; `≤ min_score` for
    /// a stable hysteresis band.
    pub evict_score: f32,
    /// Minimum members this node contributes (force-fills from the top).
    pub budget_min: usize,
    /// Maximum members this node contributes.
    pub budget_max: usize,
}

impl PolicyConfig {
    /// The provenance-side per-slot policy for a belief selector, on `group`.
    pub fn section_policy(&self, group: usize) -> SectionPolicy {
        SectionPolicy {
            group,
            beta: self.beta,
            min_score: self.min_score,
            evict_score: self.evict_score,
        }
    }

    /// The provenance-side budget for a belief selector.
    pub fn budget(&self) -> GroupBudget {
        GroupBudget {
            min: self.budget_min,
            max: self.budget_max,
        }
    }
}

/// A node's full selection policy: the [`PolicyConfig`] plus the gather-scope tag
/// filter. An empty `tags` list means "all projections in scope" (the
/// self-reinforcing default); a non-empty list restricts the gallery to
/// projections whose source turn carries one of the tags.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionPolicy {
    pub config: PolicyConfig,
    pub tags: Vec<String>,
}

impl SelectionPolicy {
    /// The schema-wide default when no node up the inheritance chain declares a
    /// policy: the committed tool scope, unrestricted tag scope.
    pub fn default_policy() -> SelectionPolicy {
        SelectionPolicy {
            config: PolicyPreset::CommittedToolScope.config(),
            tags: Vec::new(),
        }
    }

    /// A policy from a preset with unrestricted scope.
    pub fn from_preset(preset: PolicyPreset) -> SelectionPolicy {
        SelectionPolicy {
            config: preset.config(),
            tags: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_names_round_trip() {
        for (name, preset) in [
            ("committed_tool_scope", PolicyPreset::CommittedToolScope),
            ("high_recall_scope", PolicyPreset::HighRecallScope),
            ("single_pick", PolicyPreset::SinglePick),
        ] {
            assert_eq!(PolicyPreset::from_name(name), Some(preset));
        }
        assert_eq!(PolicyPreset::from_name("nope"), None);
    }

    #[test]
    fn committed_tool_scope_matches_locked_values() {
        let c = PolicyPreset::CommittedToolScope.config();
        assert_eq!(c.beta, 0.40);
        assert_eq!(c.min_score, 1000.0);
        assert_eq!(c.evict_score, 750.0);
        assert_eq!((c.budget_min, c.budget_max), (1, 3));
    }

    #[test]
    fn single_pick_is_single_slot() {
        let c = PolicyPreset::SinglePick.config();
        assert_eq!(c.budget_max, 1);
    }

    #[test]
    fn default_policy_is_committed_scope_unrestricted() {
        let p = SelectionPolicy::default_policy();
        assert_eq!(p.config, PolicyPreset::CommittedToolScope.config());
        assert!(p.tags.is_empty());
    }
}
