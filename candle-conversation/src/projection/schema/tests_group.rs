//! Tests for the group rules: how a stamp is read, how entries fade, which files
//! a group draws, and which score source gates belief.
//!
//! Each test pins a property that selection depends on. The belief-gate tests
//! cover the precedence order: an explicit zero disables the gate, a declared
//! band outranks a declared threshold, and an inherited default band never
//! becomes a gate. The checks assert exact instants and exact values rather than
//! tolerances.

use super::*;

#[cfg(test)]
pub(super) mod recency_tests {
    use super::{days_from_civil, stamp_ms, Recency};

    /// **A stamp is read to the minute it names**, and a name that does not
    /// open with one has none — exact instants, not tolerances.
    #[test]
    fn a_stamp_names_its_minute() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(2026, 11, 8), 20_765);
        assert_eq!(stamp_ms("1970-01-01-0001-ab12.yaml"), Some(60_000));
        assert_eq!(
            stamp_ms("2026-11-08-0130-7c1e.yaml"),
            Some(20_765 * 86_400_000 + 90 * 60_000)
        );
        assert_eq!(stamp_ms("README.md"), None);
        assert_eq!(stamp_ms("2026-13-08-0130-x.yaml"), None);
        assert_eq!(stamp_ms("2026-11-08T0130.yaml"), None);
    }

    /// **An entry's score halves each half-life**; a guide, whose name has no
    /// stamp, is not scaled, and neither is an entry dated after now.
    #[test]
    fn an_entry_fades_by_its_half_life() {
        let r = Recency {
            half_life_days: 30.0,
            dated_by: "path".into(),
        };
        let at = stamp_ms("2026-10-09-0000-a.yaml").unwrap();
        let day = 86_400_000;
        let path = "journals/groups/zen/2026-10-09-0000-a.yaml";
        assert_eq!(r.factor(path, at), 1.0);
        assert_eq!(r.factor(path, at + 30 * day), 0.5);
        assert_eq!(r.factor(path, at + 60 * day), 0.25);
        assert_eq!(
            r.factor("candle/mcp/src/guides/deploy.md", at + 60 * day),
            1.0
        );
        assert_eq!(r.factor(path, at - day), 1.0);
    }
}

#[cfg(test)]
pub(super) mod draws_tests {
    use super::Draws;
    use crate::projection::ids::GroupId;

    /// **A hub file under a drawn path is not drawn**; an entry beside it is,
    /// and a file of the same name outside the paths was never drawn.
    #[test]
    fn a_hub_file_under_a_drawn_path_is_not_drawn() {
        let d = Draws {
            from: GroupId::new(1),
            paths: vec!["journals/".into()],
            except: vec!["README.md".into(), "group.yaml".into()],
        };
        assert!(d.takes("journals/groups/zen/2026-11-01-0900-a1b2.yaml"));
        assert!(!d.takes("journals/groups/zen/README.md"));
        assert!(!d.takes("journals/groups/guest/group.yaml"));
        assert!(!d.takes("candle/README.md"));
        assert!(d.takes("journals/groups/zen/README.md.yaml"));
    }
}

#[cfg(test)]
pub(super) mod belief_config_tests {
    use super::{Budget, GroupSchema, SelectionPolicy, SelectionRule};
    use crate::projection::ids::GroupId;
    use crate::projection::policy::PolicyPreset;
    use crate::projection::swap_cap::DEFAULT_MAX_SWAPS;
    use crate::projection::ScanPolicy;

    fn group(score_threshold: Option<f32>, min: f32, evict: f32) -> GroupSchema {
        group_with(score_threshold, min, evict, true)
    }

    fn group_with(
        score_threshold: Option<f32>,
        min: f32,
        evict: f32,
        policy_band_declared: bool,
    ) -> GroupSchema {
        let mut config = PolicyPreset::CommittedToolScope.config();
        config.min_score = min;
        config.evict_score = evict;
        GroupSchema {
            id: GroupId::new(1),
            name: "structure".into(),
            selection: SelectionRule::TopK { k: 3 },
            score_threshold,
            policy_band_declared,
            policy: SelectionPolicy {
                config,
                tags: Vec::new(),
                layer_weights: Vec::new(),
                calibration_layer_weights: Vec::new(),
                scan: ScanPolicy::default(),
            },
            budget: Budget::default(),
            budget_adaptive: None,
            locality: None,
            anchor: None,
            max_swaps: DEFAULT_MAX_SWAPS,
            draws: None,
            recency: None,
        }
    }

    /// A declared threshold IS the belief gate for a group with NO band of its
    /// own — which is how most groups configure selection, and must keep
    /// working. (The band values passed here are an inherited preset, flagged
    /// undeclared; this test previously used the `group` helper, which hardcodes
    /// `policy_band_declared = true`, and so was silently asserting the
    /// both-declared case its own description does not describe.)
    #[test]
    fn a_declared_threshold_governs_the_belief_gate() {
        let g = group_with(Some(250.0), 600.0, 400.0, false);
        let cfg = g.belief_config(20);
        assert_eq!(cfg.min_score, 250.0);
        assert_eq!(cfg.evict_score, 250.0);
    }

    /// A group that declares BOTH gives the band the belief gate: the threshold
    /// governs selection on fresh scores, the band governs accumulated belief.
    /// `repo_map/structure` is the bundled schema's only such group and relies
    /// on exactly this.
    #[test]
    fn a_declared_band_outranks_a_threshold() {
        let g = group_with(Some(1.0), 250.0, 250.0, true);
        let cfg = g.belief_config(20);
        assert_eq!(cfg.min_score, 250.0, "the declared band is the belief gate");
        assert_eq!(cfg.evict_score, 250.0);
    }

    /// An explicit `0.0` still means "no gate" — it is a real choice, not an
    /// absence, and must not silently fall through to the policy band.
    #[test]
    fn an_explicit_zero_threshold_still_disables_the_gate() {
        let g = group(Some(0.0), 600.0, 400.0);
        let cfg = g.belief_config(20);
        assert_eq!(cfg.min_score, 0.0);
        assert_eq!(cfg.evict_score, 0.0);
    }

    /// The regression: with no declared threshold the policy's band survives.
    /// Previously the `0.0` default overwrote it, so every candidate was
    /// eligible at zero evidence and selection fell through to the score-tie
    /// order — the first turns in the group, presented as the most relevant
    /// content in the corpus.
    #[test]
    fn an_absent_threshold_leaves_the_policy_band_in_force() {
        let g = group(None, 600.0, 400.0);
        let cfg = g.belief_config(20);
        assert_eq!(cfg.min_score, 600.0, "policy min_score must survive");
        assert_eq!(cfg.evict_score, 400.0, "policy evict_score must survive");
    }

    /// The safety property: a group that declares NEITHER a threshold nor a
    /// policy band gets no gate. Its inherited band is the schema default — a
    /// tool-scope preset whose 800 bar would silently swallow every turn.
    #[test]
    fn an_undeclared_band_does_not_become_a_gate() {
        let g = group_with(None, 800.0, 600.0, false);
        let cfg = g.belief_config(20);
        assert_eq!(cfg.min_score, 0.0);
        assert_eq!(cfg.evict_score, 0.0);
    }

    /// Budget still comes from the selection rule either way.
    #[test]
    fn budget_always_comes_from_the_selection_rule() {
        for threshold in [None, Some(0.0), Some(600.0)] {
            let cfg = group(threshold, 600.0, 400.0).belief_config(20);
            assert_eq!(cfg.budget_max, 3, "top_k(3)");
            assert_eq!(cfg.budget_min, 0);
        }
        let mut g = group(None, 600.0, 400.0);
        g.selection = SelectionRule::Single;
        assert_eq!(g.belief_config(20).budget_max, 1);
        g.selection = SelectionRule::AlwaysVisible;
        assert_eq!(g.belief_config(20).budget_max, 20);
    }
}
