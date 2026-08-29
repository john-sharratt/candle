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
    /// Stable, correct, small tool scope on the **normalized 0–1000 hit-level
    /// band** (Concept A, `docs/provenance_adaptive_projection.md` §3): β0.40,
    /// min 112 / evict 60, budget 1..4, plus an early-decode grace window
    /// (35/26 through the first cadence reprojection). The thresholds come
    /// from the full-corpus normalized sweep (results doc §25.8: 745 turns ×
    /// 93 tools — true-tool normalized scores sit at p25 ≈ 949 / p50 ≈ 1394;
    /// min 112 at budget 4 holds ~99.1 % recall at ~65 % exact-1 / 0.5 mean
    /// FP; the budget-3 recall ceiling on the grown corpus is 99.2 %, budget 4
    /// recovers 99.6 %). A committed tool is held stable across its
    /// `<tool_call>` block by the scheduler suppressing reprojection there,
    /// not by pinning the selection. Default for the `tools` collection.
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
                // min 800 / evict 600 on the normalized 0–1000 band. A genuine
                // sustained tool match sits near 1000 by construction, so 800 is
                // a STRICT gate that admits only strong matches — which is what
                // tool selection needs. (The earlier migration to min 112 was
                // measured too permissive live: weak content bleed — e.g. an
                // `hmac_compute` tool scoring ~156 — cleared the gate for a code
                // query it had no business answering. 800 restores strictness;
                // it maps cleanly to the band because a real hit ≈ 1000 here just
                // as the raw hits clustered near 1000 before normalization.)
                min_score: 800.0,
                evict_score: 600.0,
                budget_min: 1,
                budget_max: 3,
                // Early-decode grace: for the first 64 generated tokens hold the
                // bar at 250 (evict 187.5, same 0.75 hysteresis ratio) so a
                // correct pick whose decode-Q is still building stays in scope
                // until its signal crosses.
                early_window_tokens: 64,
                early_min_score: 250.0,
                early_evict_score: 187.5,
            },
            PolicyPreset::HighRecallScope => PolicyConfig {
                beta: 0.40,
                min_score: 40.0,
                evict_score: 20.0,
                budget_min: 1,
                budget_max: 5,
                // Already a low bar — no separate early window (mirror the
                // steady-state band so `windowed` is a no-op).
                early_window_tokens: 0,
                early_min_score: 40.0,
                early_evict_score: 20.0,
            },
            PolicyPreset::SinglePick => PolicyConfig {
                beta: 0.40,
                min_score: 0.0,
                evict_score: 0.0,
                budget_min: 1,
                budget_max: 1,
                early_window_tokens: 0,
                early_min_score: 0.0,
                early_evict_score: 0.0,
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

use super::adaptive::ScanPolicy;
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
    /// Last generated-token position (inclusive) covered by the early-decode
    /// grace window; the window spans decode positions `1..=early_window_tokens`
    /// (submit / position 0 is excluded — see [`Self::windowed`]). Inside it the
    /// selection bar drops to `early_min_score`/`early_evict_score` and a carried
    /// belief is floored at `early_min_score`. The right tool's decode-Q
    /// signal only builds over roughly this many tokens (the call↔definition
    /// domain gap makes the query's prefill-Q ≈ 0 for it), so the steady-state
    /// `min_score` would evict a correct-but-still-accruing pick before it ever
    /// crosses. The bound is **inclusive** so it covers the first cadence
    /// reprojection, which fires at exactly this position (`reproject_every_n`) —
    /// often the earliest reprojection whose decode-Q has any signal. `0`
    /// disables the window.
    pub early_window_tokens: usize,
    /// Selection threshold used inside the early-decode window. `≤ min_score`.
    pub early_min_score: f32,
    /// Eviction threshold used inside the early-decode window. `≤ early_min_score`.
    pub early_evict_score: f32,
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

    /// Resolve the effective config **and** the carried-belief floor at a decode
    /// position. `decode_pos` is the count of tokens generated so far this turn;
    /// `None` means "not an early-decode selection" (a stateless/unit projection).
    ///
    /// The window covers decode positions `1..=early_window_tokens` — actual decode
    /// tokens, through the first cadence reprojection (which fires at exactly
    /// `early_window_tokens`). Inside it the band is lowered to
    /// `early_min_score`/`early_evict_score` and the floor is `Some(early_min_score)`
    /// — [`crate::provenance::belief_step`] raises every *carried, already-selected*
    /// member to at least that floor after the RelLeak decay, so a decaying lock-on
    /// can drop *relative to fresher rivals* but not out of scope while its own
    /// decode-Q is still accruing.
    ///
    /// Position `0` (the submit / opening projection) is deliberately **excluded**:
    /// there the "prior" is the *previous* turn's decayed belief, not a this-turn
    /// pick, so lowering the bar or flooring there would pin a stale cross-turn tool
    /// into the opening instead of protecting this turn's own signal. Submit and
    /// everything past the window use the steady band with no floor.
    pub fn windowed(&self, decode_pos: Option<usize>) -> (PolicyConfig, Option<f32>) {
        match decode_pos {
            Some(pos)
                if self.early_window_tokens > 0
                    && (1..=self.early_window_tokens).contains(&pos) =>
            {
                let mut c = *self;
                c.min_score = self.early_min_score;
                c.evict_score = self.early_evict_score;
                (c, Some(self.early_min_score))
            }
            _ => (*self, None),
        }
    }
}

/// A node's full selection policy: the [`PolicyConfig`] plus the gather-scope tag
/// filter.
///
/// # What an empty list means
///
/// An empty `tags` list admits **only turns that are themselves untagged** — it
/// is not "everything". A non-empty list admits only turns carrying one of the
/// named tags, and excludes untagged ones. See the gather in
/// [`super::resolver`], and `substrate::tests::an_empty_tag_filter_admits_only_untagged_turns`
/// which pins both halves.
///
/// This comment previously read "all projections in scope", which is the
/// opposite of what the code does, and the difference is not cosmetic: it is
/// what lets one corpus serve many scopes. Content ingested untagged is shared
/// by every node declaring no filter; content ingested with a tag is reachable
/// only from a node naming it. Were "empty" to mean everything, every tagged
/// turn would be visible from every unfiltered node — silently, since nothing
/// about the result would look wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionPolicy {
    pub config: PolicyConfig,
    pub tags: Vec<String>,
    /// Per-layer-group weights on the belief scorer's `z × margin` vote
    /// (`layer_weights[g]` scales fold-group `g`; missing / empty ⇒ uniform 1.0).
    /// The fold is `[46, 1, 1]` (§ `wide_sig`): group 0 = L0–45, group 1 = L46,
    /// group 2 = L47. Tools keep uniform weighting (their identity is spread across
    /// the upper groups); repo_map peaks on group 1 (L46), where cluster identity
    /// concentrates — see `docs/tool_selection_provenance_results.md` §83. Set via
    /// the node's `policy.layer_weights` in the YAML.
    pub layer_weights: Vec<f32>,
    /// Scan-side knobs (`docs/provenance_adaptive_projection.md`): fusion mode
    /// (Concept G), question pinning (Concept F), mass constants (Concept B),
    /// level-prior constants (Concept A.4). Defaults = today's behavior.
    pub scan: ScanPolicy,
}

impl SelectionPolicy {
    /// The schema-wide default when no node up the inheritance chain declares a
    /// policy: the committed tool scope, unrestricted tag scope, uniform layers.
    pub fn default_policy() -> SelectionPolicy {
        SelectionPolicy {
            config: PolicyPreset::CommittedToolScope.config(),
            tags: Vec::new(),
            layer_weights: Vec::new(),
            scan: ScanPolicy::default(),
        }
    }

    /// A policy from a preset with unrestricted scope and uniform layer weights.
    pub fn from_preset(preset: PolicyPreset) -> SelectionPolicy {
        SelectionPolicy {
            config: preset.config(),
            tags: Vec::new(),
            layer_weights: Vec::new(),
            scan: ScanPolicy::default(),
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
        // Strict gate on the normalized band (min 800 ≈ near the 1000 hit
        // ceiling). The min-112 migration was reverted after live testing showed
        // it admitted weak tools into unrelated queries.
        let c = PolicyPreset::CommittedToolScope.config();
        assert_eq!(c.beta, 0.40);
        assert_eq!(c.min_score, 800.0);
        assert_eq!(c.evict_score, 600.0);
        assert_eq!((c.budget_min, c.budget_max), (1, 3));
        assert_eq!(c.early_window_tokens, 64);
        assert_eq!(c.early_min_score, 250.0);
        assert_eq!(c.early_evict_score, 187.5);
    }

    #[test]
    fn windowed_lowers_band_and_floors_inside_window() {
        let c = PolicyPreset::CommittedToolScope.config();
        // Inside the window — decode positions 1..=64, INCLUDING the boundary 64
        // (the first cadence reprojection): lowered band + a floor at the early min.
        for pos in [Some(1), Some(63), Some(64)] {
            let (w, floor) = c.windowed(pos);
            assert_eq!(w.min_score, 250.0, "pos={pos:?}");
            assert_eq!(w.evict_score, 187.5, "pos={pos:?}");
            assert_eq!(floor, Some(250.0), "pos={pos:?}");
            // Untouched knobs carry through.
            assert_eq!(w.beta, c.beta);
            assert_eq!((w.budget_min, w.budget_max), (c.budget_min, c.budget_max));
        }
        // Submit (pos 0), past the window (65+), and the stateless `None` case:
        // steady-state band, no floor. Position 0's "prior" is the previous turn's
        // decayed belief, so the window must not lower the bar or floor there.
        for pos in [None, Some(0), Some(65), Some(129), Some(10_000)] {
            let (w, floor) = c.windowed(pos);
            assert_eq!(w.min_score, 800.0, "pos={pos:?}");
            assert_eq!(w.evict_score, 600.0, "pos={pos:?}");
            assert_eq!(floor, None, "pos={pos:?}");
        }
    }

    #[test]
    fn windowed_is_noop_when_disabled() {
        // `early_window_tokens == 0` disables the window regardless of position.
        let c = PolicyPreset::SinglePick.config();
        let (w, floor) = c.windowed(Some(0));
        assert_eq!(w, c);
        assert_eq!(floor, None);
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
