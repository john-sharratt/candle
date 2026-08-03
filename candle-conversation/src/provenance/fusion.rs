//! Cross-group fusion of per-fold-group provenance scores — Concept G of
//! `docs/provenance_adaptive_projection.md` (§9).
//!
//! The scan scores each fold layer-group independently (see
//! [`super::scan::score_provenance_late_fusion_grouped`]); this module is the
//! law that combines the per-group per-case tallies into one score per case.
//! The mode is a **policy value** — the same code runs on every axis, and the
//! measured split (results doc §25.5/§25.8) is expressed purely in
//! configuration: tools keep [`FusionMode::Additive`] (tool identity lives in
//! the id-groups with no content agreement — gating collapses Top-1
//! 97.3 % → 32.9 %), content axes use [`FusionMode::ContentGated`] (identity
//! votes count only when the gate group agrees at all, which kills pure
//! id-spikes like a call site outscoring a definition 2004 → 0 while
//! preserving additive magnitude and recall).

use serde::{Deserialize, Serialize};

/// How per-group scores combine into one score per case.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum FusionMode {
    /// Sum the groups — the shipped scorer. This is the only mode whose needle
    /// gate spans groups (per-token total magnitude); it is bit-identical to
    /// [`super::score_provenance_late_fusion_weighted`].
    #[default]
    Additive,
    /// Identity confirms, content decides: the per-group tallies sum, but only
    /// when the gate group's own tally is positive — a case with zero
    /// gate-group agreement scores zero regardless of its id-group spikes.
    /// [`Self::fuse`] over the grouped tallies IS the law (a full-additive
    /// variant gated by a one-hot scan was measured and rejected — the true
    /// target collapsed; results doc §25).
    ContentGated {
        #[serde(default)]
        gate_group: usize,
    },
    /// Per-case minimum across groups. Measured recall-fragile at pool scale
    /// (a case must lead every group simultaneously) — kept for measurement,
    /// not recommended as a production value.
    ConsensusMin,
    /// Per-case geometric mean across groups.
    ConsensusGeo,
}

impl FusionMode {
    /// Fuse per-group per-case tallies (`grouped[g][case]`) into one score per
    /// case. `grouped` must be non-empty and rectangular. [`FusionMode::Additive`]
    /// here sums the per-group-gated tallies — callers wanting the shipped
    /// cross-group needle gate dispatch to the single-pass scorer instead (the
    /// scan entry points do this; see `score_slots_fused`).
    pub fn fuse(&self, grouped: &[Vec<f32>]) -> Vec<f32> {
        let n_groups = grouped.len();
        let n_cases = grouped.first().map(Vec::len).unwrap_or(0);
        (0..n_cases)
            .map(|c| match self {
                FusionMode::Additive => grouped.iter().map(|g| g[c]).sum(),
                FusionMode::ContentGated { gate_group } => {
                    let gate = grouped.get(*gate_group).map(|g| g[c]).unwrap_or(0.0);
                    if gate > 0.0 {
                        grouped.iter().map(|g| g[c]).sum()
                    } else {
                        0.0
                    }
                }
                FusionMode::ConsensusMin => grouped
                    .iter()
                    .map(|g| g[c])
                    .fold(f32::INFINITY, f32::min)
                    .max(0.0),
                FusionMode::ConsensusGeo => {
                    let prod: f32 = grouped.iter().map(|g| g[c].max(0.0)).product();
                    prod.powf(1.0 / n_groups as f32)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grouped() -> Vec<Vec<f32>> {
        // case 0 = pure id-spike (zero gate-group agreement, huge id groups);
        // case 1 = balanced true match; case 2 = gate-only match.
        vec![
            vec![0.0, 448.0, 30.0],  // group 0 (gate / content)
            vec![1160.0, 57.0, 0.0], // group 1 (id)
            vec![861.0, 58.0, 0.0],  // group 2 (id)
        ]
    }

    #[test]
    fn additive_sums_groups() {
        let f = FusionMode::Additive.fuse(&grouped());
        assert_eq!(f, vec![2021.0, 563.0, 30.0]);
    }

    #[test]
    fn content_gated_kills_id_spikes_and_keeps_additive_magnitude() {
        let f = FusionMode::ContentGated { gate_group: 0 }.fuse(&grouped());
        assert_eq!(
            f,
            vec![0.0, 563.0, 30.0],
            "the spike must score zero; gated cases keep the full additive sum"
        );
    }

    #[test]
    fn gate_group_is_config_not_convention() {
        // Gating on group 2 instead: case 2 (zero in group 2) dies, the spike
        // survives — the gate index is a knob, not a hardcoded fold.
        let f = FusionMode::ContentGated { gate_group: 2 }.fuse(&grouped());
        assert_eq!(f, vec![2021.0, 563.0, 0.0]);
    }

    #[test]
    fn consensus_min_requires_every_group() {
        let f = FusionMode::ConsensusMin.fuse(&grouped());
        assert_eq!(
            f,
            vec![0.0, 57.0, 0.0],
            "min keeps only cases with votes in every group"
        );
    }

    #[test]
    fn consensus_geo_is_the_cube_root_of_the_product() {
        let f = FusionMode::ConsensusGeo.fuse(&grouped());
        assert_eq!(f[0], 0.0);
        assert!((f[1] - (448.0f32 * 57.0 * 58.0).powf(1.0 / 3.0)).abs() < 1e-3);
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn missing_gate_group_scores_zero() {
        let f = FusionMode::ContentGated { gate_group: 9 }.fuse(&grouped());
        assert_eq!(f, vec![0.0, 0.0, 0.0]);
    }
}
