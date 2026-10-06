//! What the wave rate planner needs to know about the weight side.
//!
//! The planner's whole question is whether a wave's rows earn back the expert
//! residency they dislodge, and that needs two things the KV side cannot
//! observe: the **geometry** a routed expert is priced in, and the **range** the
//! weight zone may move in.
//!
//! Neither is derivable from the other side of the boundary.
//! `request_kv_ground` reports bytes conceded *after the fact*, and the zone
//! grows back — so capacity and frontier read exactly as they did at load while
//! the conceded slots hold something else. A figure inferred from concessions is
//! therefore the same whether or not the ground came back — the trap hot-path
//! invariant 7 records as "never infer 'nothing moved' from the geometry".
//!
//! So the zone publishes it. Every field is a gauge on
//! [`PipelineStats`](super::PipelineStats), refreshed where the numbers are
//! already in scope, and this is the shape they are read back in.

/// The weight side as the wave rate planner prices it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightPlan {
    /// MoE layers a decode step walks.
    pub moe_layers: usize,
    /// Experts one layer holds — the cap on a layer's copy, however many rows
    /// route through it.
    pub experts_per_layer: usize,
    /// Bytes one expert slot occupies: the grain every figure here is a
    /// multiple of.
    pub slot_bytes: u64,
    /// Expert bytes resident right now.
    pub resident_bytes: u64,
    /// The zone as it stands.
    pub zone_bytes: u64,
    /// The residency the zone may not go under, and the most it could reach.
    pub zone_min_bytes: u64,
    pub zone_max_bytes: u64,
}

/// What a gauge set says about whether the rate planner can be armed.
///
/// **Three outcomes, not two, because the third is a defect and the other two are
/// not.** `Option<WeightPlan>` could only say "plan" or "do not plan", so the
/// scheduler had to treat a dense stack and a routed model with broken gauges
/// identically — and its handling for both was to admit one prefill and return. That
/// is right for the dense stack and catastrophic for the routed one: no plan means no
/// rate model, which means every wave is one row wide, which on the 30B-A3B measured
/// 32 t/s against a batched ceiling of 518. It logged nothing, because from the
/// scheduler's side there was nothing to distinguish.
///
/// So the distinction is in the type, and the caller has to handle each case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightPlanning {
    /// No expert cache at all — a dense stack. There is genuinely no residency to
    /// trade a wave's rows against, so not planning is the correct answer and the
    /// width backstop is the only bound. Expected, and not a problem.
    Dense,
    /// A complete gauge set.
    Ready(WeightPlan),
    /// **A routed model whose gauges do not describe it.** The cache says it has MoE
    /// layers and experts, so it knows it is routed; a zero slot size or zone limit in
    /// that state is not a state the engine can be in legitimately. Carries the field
    /// that was zero, because the fix depends on which.
    Incomplete { field: &'static str },
}

impl WeightPlan {
    /// Read a plan from the cache's published gauges.
    ///
    /// **All-or-nothing on the plan itself.** A partial plan would let the planner
    /// price a decode's copy against a zero layer count or a zero slot size, and both
    /// make a routed expert free — which is the one direction the model must never be
    /// wrong in, because it spends residency it cannot get back.
    ///
    /// But "cannot plan" is not one condition, and [`WeightPlanning`] is where that is
    /// separated. A dense stack reports no MoE geometry and is `Dense`; a cache that
    /// reports geometry and then zeroes for its zone is `Incomplete`, which is a bug.
    pub fn from_stats(s: &super::PipelineStats) -> WeightPlanning {
        // No routed geometry at all: a dense stack, or a non-CUDA build. Nothing to
        // plan and nothing wrong.
        if s.moe_layers == 0 || s.total_experts == 0 {
            return WeightPlanning::Dense;
        }
        // From here the cache has told us it is routed, so every remaining zero is a
        // gauge that was never published rather than a model that has none.
        let experts_per_layer = s.total_experts / s.moe_layers;
        if experts_per_layer == 0 {
            return WeightPlanning::Incomplete {
                field: "total_experts / moe_layers rounded to zero",
            };
        }
        if s.expert_slot_bytes == 0 {
            return WeightPlanning::Incomplete {
                field: "expert_slot_bytes",
            };
        }
        if s.zone_max_bytes == 0 {
            return WeightPlanning::Incomplete {
                field: "zone_max_bytes",
            };
        }
        WeightPlanning::Ready(Self {
            moe_layers: s.moe_layers,
            experts_per_layer,
            slot_bytes: s.expert_slot_bytes as u64,
            resident_bytes: s.resident_vram_bytes as u64,
            zone_bytes: s.zone_bytes as u64,
            zone_min_bytes: s.zone_min_bytes as u64,
            zone_max_bytes: s.zone_max_bytes as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::PipelineStats;

    fn full() -> PipelineStats {
        PipelineStats {
            moe_layers: 41,
            total_experts: 41 * 256,
            expert_slot_bytes: 1_933_312,
            resident_vram_bytes: 7 << 30,
            zone_bytes: 8 << 30,
            zone_min_bytes: 4 << 30,
            zone_max_bytes: 10 << 30,
            ..Default::default()
        }
    }

    /// Unwrap a `Ready`, or say which other verdict came back.
    fn ready(s: &PipelineStats) -> WeightPlan {
        match WeightPlan::from_stats(s) {
            WeightPlanning::Ready(p) => p,
            other => panic!("expected a complete gauge set to plan, got {other:?}"),
        }
    }

    #[test]
    fn a_full_gauge_set_yields_the_geometry_and_the_range() {
        let p = ready(&full());
        assert_eq!(p.moe_layers, 41);
        assert_eq!(p.experts_per_layer, 256);
        assert_eq!(p.slot_bytes, 1_933_312);
        assert_eq!(p.zone_min_bytes, 4 << 30);
        assert_eq!(p.zone_max_bytes, 10 << 30);
    }

    /// **No routed geometry is `Dense`, and it is not a defect.**
    ///
    /// A stack with no MoE layers or no experts has no residency to trade a wave's rows
    /// against, so the scheduler is right to plan nothing and fall back to its width
    /// backstop.
    #[test]
    fn a_stack_with_no_routing_is_dense() {
        for (name, s) in [
            (
                "no MoE layers",
                PipelineStats {
                    moe_layers: 0,
                    ..full()
                },
            ),
            (
                "no experts",
                PipelineStats {
                    total_experts: 0,
                    ..full()
                },
            ),
        ] {
            assert_eq!(
                WeightPlan::from_stats(&s),
                WeightPlanning::Dense,
                "{name} is a dense stack, not a broken gauge set"
            );
        }
    }

    /// **A routed model with a missing gauge is `Incomplete`, and names the field.**
    ///
    /// Each of these would otherwise make a routed expert cost nothing — a zero slot size,
    /// or a zone with no range to move in — and the planner would spend residency on rows
    /// that never repay it. But the cache has already said it *is* routed, so a zero here
    /// is a gauge nobody published rather than a model that has none, and the two must not
    /// share a verdict: an absent plan collapses every wave to one row, which is correct
    /// for a dense stack and an order of magnitude for this one.
    #[test]
    fn a_routed_model_with_a_missing_gauge_names_the_field() {
        for (field, s) in [
            (
                "expert_slot_bytes",
                PipelineStats {
                    expert_slot_bytes: 0,
                    ..full()
                },
            ),
            (
                "zone_max_bytes",
                PipelineStats {
                    zone_max_bytes: 0,
                    ..full()
                },
            ),
        ] {
            assert_eq!(
                WeightPlan::from_stats(&s),
                WeightPlanning::Incomplete { field },
                "a routed model missing {field} must say so rather than read as dense"
            );
        }

        // Fewer experts than layers rounds the per-layer count to zero, which is the same
        // class of defect reached by a different arithmetic route.
        assert!(matches!(
            WeightPlan::from_stats(&PipelineStats {
                total_experts: 3,
                moe_layers: 41,
                ..full()
            }),
            WeightPlanning::Incomplete { .. }
        ));
    }
}
