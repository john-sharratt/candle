//! Provenance-adaptive projection mechanisms — the configuration types and
//! pure algorithms of `docs/provenance_adaptive_projection.md` Concepts A.4
//! (level prior), B (attention mass → adaptive budgets), C (turn locality),
//! D (file-head anchor), F (question-anchored probing), and G (fusion mode).
//!
//! Everything here is layer-agnostic by construction (the design's hard
//! generalization invariant): each type is a knob any policy, group, or layer
//! can carry, and each algorithm is pure over plain floats so it unit-tests in
//! isolation. The wiring — which scores flow in, which selections flow out —
//! lives in `resolver.rs` / `project.rs`.

use serde::{Deserialize, Serialize};

use crate::provenance::FusionMode;

/// Reference window for the Concept A.4 size ratio — the live probe cap
/// (`reproject_max_probe_tokens`).
pub const LEVEL_PRIOR_T_REF: usize = 256;

/// Scan-side policy knobs (per selection-policy node): Concept G's fusion
/// mode, Concept F's question pinning, Concept B's mass constants, and
/// Concept A.4's level-prior constants. Defaults are today's behavior — an
/// absent block changes nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanPolicy {
    /// Concept G: how the fold groups' votes combine. `Additive` = the shipped
    /// scorer; content axes configure `ContentGated` (measured: kills id-group
    /// spikes while preserving recall; tools MUST stay additive — results doc
    /// §25.5/§25.8).
    pub fusion: FusionMode,
    /// Concept F: scan the turn's pinned question window as a second probe and
    /// take the per-slot max of the two normalized scans.
    pub question_pin: bool,
    /// Concept B: top-k share used as the concentration factor of mass.
    /// Measured `k = 1` (results doc §25: the top-1 share is the strong
    /// discriminator between a genuine hit and a diffuse probe).
    pub mass_top_k: usize,
    /// Concept B: exponent on the concentration factor (`0` = plain gated
    /// sum). Measured `ρ = 2`: squaring the top-1 share is what flips the
    /// raw-sum inversion (code 3470×0.579² = 1163 vs recall 5591×0.368² = 757).
    pub mass_rho: f32,
    /// Concept A.4: base of the size-aware level-prior floor, in raw-score
    /// units (the learned-level scale). `0` disables the size prior.
    pub level_prior_base: f32,
    /// Concept A.4: maximum size-scaling multiple for tiny children.
    pub level_prior_cap: f32,
}

impl Default for ScanPolicy {
    fn default() -> Self {
        ScanPolicy {
            fusion: FusionMode::Additive,
            question_pin: false,
            mass_top_k: 1,
            mass_rho: 2.0,
            level_prior_base: 0.0,
            level_prior_cap: 8.0,
        }
    }
}

impl ScanPolicy {
    /// The Concept A.4 denominator floor for a child of `tokens` real tokens:
    /// `base × clamp(t_ref / tokens, 1, cap)` — small children get
    /// proportionally higher floors. `0.0` when the prior is disabled or the
    /// size is unknown.
    pub fn level_floor(&self, tokens: usize, t_ref: usize) -> f32 {
        if self.level_prior_base <= 0.0 || tokens == 0 || t_ref == 0 {
            return 0.0;
        }
        self.level_prior_base
            * (t_ref as f32 / tokens as f32).clamp(1.0, self.level_prior_cap.max(1.0))
    }
}

/// Concept B — attention mass of one score competition (group / collection):
/// the gated sum of normalized scores, scaled by concentration so a diffuse
/// probe (mass smeared over many weak candidates) self-mutes while a genuine
/// hit (mass concentrated on few) carries. Measured basis: raw sum-mass is
/// INVERTED on real probes; concentration widens the corrected margins
/// (results doc §25.6, F4/F10).
///
/// ```text
/// S       = { s_i : s_i ≥ min_score }
/// sum     = Σ min(s_i, band)
/// conc    = Σ_{top-k} min(s_i, band) / sum
/// mass    = sum × conc^ρ
/// ```
pub fn attention_mass(scores: &[f32], min_score: f32, band: f32, k: usize, rho: f32) -> f32 {
    let mut gated: Vec<f32> = scores
        .iter()
        .filter(|&&s| s >= min_score && s > 0.0)
        .map(|&s| s.min(band))
        .collect();
    if gated.is_empty() {
        return 0.0;
    }
    gated.sort_by(|a, b| b.total_cmp(a));
    let sum: f32 = gated.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let topk: f32 = gated.iter().take(k.max(1)).sum();
    sum * (topk / sum).powf(rho.max(0.0))
}

/// Concept B — a layer/group token budget's adaptive rail: the flexbox
/// priority is scaled by attention mass, clamped by the declared percents
/// (which stay the outer authority).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetAdaptive {
    /// Priority multiplier per `band` (1000) of mass:
    /// `effective_priority = priority × (1 + gain × mass / band)`.
    pub gain: f32,
    /// Adaptive ceiling (percent of parent) that overrides the static
    /// `max_percent` when mass demands more room. `None` = keep the static
    /// ceiling.
    pub max_percent: Option<f32>,
}

impl BudgetAdaptive {
    /// The mass-modulated flexbox priority.
    pub fn effective_priority(&self, priority: f32, mass: f32, band: f32) -> f32 {
        priority * (1.0 + self.gain.max(0.0) * (mass / band.max(1.0)))
    }
}

/// Concept B — a belief group's member-budget extension: extra members granted
/// per `per_extra` of attention mass, capped at `absolute_max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemberBudgetAdaptive {
    /// Mass required per additional member above the base budget.
    pub per_extra: f32,
    /// Hard cap on the extended budget.
    pub absolute_max: usize,
}

impl MemberBudgetAdaptive {
    /// `clamp(base + floor(mass / per_extra), base, absolute_max)`.
    pub fn effective_max(&self, base_max: usize, mass: f32) -> usize {
        if self.per_extra <= 0.0 {
            return base_max;
        }
        let extra = (mass / self.per_extra).floor() as usize;
        (base_max + extra).clamp(base_max, self.absolute_max.max(base_max))
    }
}

/// Concept C — turn locality: a hit on an exchange drags its neighbors into
/// contention, radius growing with attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalityConfig {
    /// Normalized score at which an exchange starts dragging neighbors.
    pub seed_threshold: f32,
    /// Per-step score falloff (`score × decay^distance`), in `(0, 1]`.
    pub decay: f32,
    /// Radius granted to any seed at the threshold.
    pub base_radius: usize,
    /// Additional radius per this much score above the threshold.
    pub extend_per: f32,
    /// Cap on the additional radius.
    pub extra_radius_max: usize,
}

impl LocalityConfig {
    /// The drag radius for a seed of score `s` (0 below the threshold).
    pub fn radius(&self, s: f32) -> usize {
        if s < self.seed_threshold {
            return 0;
        }
        let extra = if self.extend_per > 0.0 {
            (((s - self.seed_threshold) / self.extend_per).floor() as usize)
                .min(self.extra_radius_max)
        } else {
            0
        };
        self.base_radius + extra
    }

    /// Apply locality over one timeline's ordered exchange scores, returning
    /// the boosted scores and, per exchange, whether the boost RAISED it (the
    /// [`super::SelectionOrigin::Locality`] stamp for members selected only
    /// through their neighbor). `max`, not sum: two adjacent hits never
    /// double-count, and an exchange that scored higher on its own merits
    /// keeps its own score.
    pub fn apply(&self, scores: &[f32]) -> (Vec<f32>, Vec<bool>) {
        let mut out = scores.to_vec();
        let mut boosted = vec![false; scores.len()];
        for (e, &s) in scores.iter().enumerate() {
            let radius = self.radius(s);
            for d in 1..=radius {
                let drag = s * self.decay.powi(d as i32);
                for n in [e.checked_sub(d), e.checked_add(d)] {
                    let Some(n) = n else { continue };
                    if n >= out.len() {
                        continue;
                    }
                    if drag > out[n] {
                        out[n] = drag;
                        boosted[n] = true;
                    }
                }
            }
        }
        (out, boosted)
    }
}

/// Concept D — which member of a timeline rides along whenever any of its
/// exchanges is selected. `First` = the timeline's first exchange (for a code
/// file's conversation: the `FileHeader` scope — module doc + imports).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorMember {
    First,
}

/// Concept D — a group's anchor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorConfig {
    pub member: AnchorMember,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Concept B: attention mass ────────────────────────────────────────────

    #[test]
    fn mass_is_zero_when_nothing_clears_the_gate() {
        assert_eq!(attention_mass(&[10.0, 39.9], 40.0, 1000.0, 3, 1.0), 0.0);
        assert_eq!(attention_mass(&[], 0.0, 1000.0, 3, 1.0), 0.0);
    }

    #[test]
    fn mass_caps_each_score_at_the_band() {
        // One score far above the band contributes exactly `band`.
        let m = attention_mass(&[5000.0], 0.0, 1000.0, 3, 0.0);
        assert_eq!(m, 1000.0);
    }

    #[test]
    fn concentration_separates_peaked_from_smeared() {
        // Same gated sum (600): peaked = one 600; smeared = twelve 50s.
        let peaked = attention_mass(&[600.0], 0.0, 1000.0, 1, 1.0);
        let smeared = attention_mass(&[50.0; 12], 0.0, 1000.0, 1, 1.0);
        assert_eq!(peaked, 600.0); // conc = 1
        assert!((smeared - 600.0 * (50.0 / 600.0)).abs() < 1e-3);
        assert!(peaked > 10.0 * smeared);
        // ρ = 0 disables the factor: both mass to the plain sum.
        assert_eq!(attention_mass(&[50.0; 12], 0.0, 1000.0, 1, 0.0), 600.0);
    }

    #[test]
    fn budget_adaptive_scales_priority_from_mass() {
        let a = BudgetAdaptive {
            gain: 2.0,
            max_percent: Some(25.0),
        };
        assert_eq!(a.effective_priority(5.0, 0.0, 1000.0), 5.0);
        assert_eq!(a.effective_priority(5.0, 1000.0, 1000.0), 15.0);
    }

    #[test]
    fn member_budget_extends_exactly_at_per_extra_boundaries() {
        let a = MemberBudgetAdaptive {
            per_extra: 800.0,
            absolute_max: 8,
        };
        assert_eq!(a.effective_max(4, 0.0), 4);
        assert_eq!(a.effective_max(4, 799.9), 4);
        assert_eq!(a.effective_max(4, 800.0), 5);
        assert_eq!(a.effective_max(4, 3200.0), 8);
        assert_eq!(a.effective_max(4, 99999.0), 8, "absolute_max caps");
        // Degenerate config is inert.
        let off = MemberBudgetAdaptive {
            per_extra: 0.0,
            absolute_max: 8,
        };
        assert_eq!(off.effective_max(4, 5000.0), 4);
    }

    // ── Concept C: locality (table-driven per the design's §11.3) ────────────

    fn loc() -> LocalityConfig {
        LocalityConfig {
            seed_threshold: 600.0,
            decay: 0.5,
            base_radius: 1,
            extend_per: 200.0,
            extra_radius_max: 2,
        }
    }

    #[test]
    fn radius_grows_with_attention_and_caps() {
        let l = loc();
        assert_eq!(l.radius(599.9), 0);
        assert_eq!(l.radius(600.0), 1);
        assert_eq!(l.radius(799.9), 1);
        assert_eq!(l.radius(800.0), 2);
        assert_eq!(l.radius(1000.0), 3);
        assert_eq!(l.radius(9999.0), 3, "extra_radius_max caps");
    }

    #[test]
    fn drag_is_max_not_sum_and_respects_timeline_bounds() {
        let l = loc();
        // Seed 1000 at index 2 (radius 3): drags 500 at ±1, 250 at ±2, 125 at ±3.
        let scores = [0.0, 0.0, 1000.0, 700.0, 0.0, 0.0];
        let (out, boosted) = l.apply(&scores);
        assert_eq!(out[1], 500.0);
        assert_eq!(out[0], 250.0);
        // Index 3 scored 700 on its own merits (> the 500 drag): keeps its own,
        // NOT boosted — and is itself a seed (radius 1) dragging 350 to index 4;
        // the 1000-seed's 250 drag at distance 2 loses to it.
        assert_eq!(out[3], 700.0);
        assert!(!boosted[3]);
        assert_eq!(out[4], 350.0);
        assert!(boosted[4]);
        // Distance 3 from the 1000-seed reaches index 5: max(125, nothing) = 175?
        // No — index 5 is distance 3 from seed 2 (125) and distance 2 from seed 3
        // (700 × 0.25 = 175, but seed 3's radius is only 1). So 125.
        assert_eq!(out[5], 125.0);
        assert!(boosted[0] && boosted[1] && boosted[5]);
    }

    #[test]
    fn no_seed_no_drag() {
        let l = loc();
        let scores = [100.0, 599.0, 0.0];
        let (out, boosted) = l.apply(&scores);
        assert_eq!(out, scores.to_vec());
        assert!(boosted.iter().all(|b| !b));
    }

    // ── Concept A.4: level floor ─────────────────────────────────────────────

    #[test]
    fn level_floor_scales_inverse_to_size_and_caps() {
        let p = ScanPolicy {
            level_prior_base: 100.0,
            level_prior_cap: 8.0,
            ..ScanPolicy::default()
        };
        assert_eq!(p.level_floor(256, 256), 100.0); // full window → base
        assert_eq!(p.level_floor(512, 256), 100.0); // big window → clamped at 1×
        assert_eq!(p.level_floor(64, 256), 400.0); // quarter window → 4×
        assert_eq!(p.level_floor(8, 256), 800.0); // tiny fragment → capped 8×
        assert_eq!(p.level_floor(0, 256), 0.0);
        assert_eq!(
            ScanPolicy::default().level_floor(8, 256),
            0.0,
            "off by default"
        );
    }
}
