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
//! therefore the same whether or not the ground came back, which is the trap
//! hot-path invariant 7 records for cached slot addresses, one tenant over.
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

impl WeightPlan {
    /// Read a plan from the cache's published gauges, or `None` when they do not
    /// describe a MoE model with a live zone.
    ///
    /// **All-or-nothing on purpose.** A partial plan would let the planner price
    /// a decode's copy against a zero layer count or a zero slot size, and both
    /// make a routed expert free — which is the one direction the model must
    /// never be wrong in, because it spends residency it cannot get back. A
    /// dense model, a cache that has not run a classify yet, and a non-CUDA
    /// build all land here and all mean the same thing: do not plan on this.
    pub fn from_stats(s: &super::PipelineStats) -> Option<Self> {
        if s.moe_layers == 0 || s.total_experts == 0 || s.expert_slot_bytes == 0 {
            return None;
        }
        let experts_per_layer = s.total_experts / s.moe_layers;
        if experts_per_layer == 0 || s.zone_max_bytes == 0 {
            return None;
        }
        Some(Self {
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

    #[test]
    fn a_full_gauge_set_yields_the_geometry_and_the_range() {
        let p = WeightPlan::from_stats(&full()).expect("a complete gauge set plans");
        assert_eq!(p.moe_layers, 41);
        assert_eq!(p.experts_per_layer, 256);
        assert_eq!(p.slot_bytes, 1_933_312);
        assert_eq!(p.zone_min_bytes, 4 << 30);
        assert_eq!(p.zone_max_bytes, 10 << 30);
    }

    /// **Any missing gauge means no plan at all.**
    ///
    /// Each of these would otherwise make a routed expert cost nothing — a zero
    /// layer count, a zero slot size, or a zone with no range to move in — and
    /// the planner would then spend residency on rows that never repay it.
    #[test]
    fn a_partial_gauge_set_plans_nothing() {
        for (name, s) in [
            (
                "dense model",
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
            (
                "no slot size",
                PipelineStats {
                    expert_slot_bytes: 0,
                    ..full()
                },
            ),
            (
                "zone never published",
                PipelineStats {
                    zone_max_bytes: 0,
                    ..full()
                },
            ),
            (
                "fewer experts than layers",
                PipelineStats {
                    total_experts: 3,
                    moe_layers: 41,
                    ..full()
                },
            ),
        ] {
            assert!(
                WeightPlan::from_stats(&s).is_none(),
                "{name} must not produce a plan"
            );
        }
    }
}
