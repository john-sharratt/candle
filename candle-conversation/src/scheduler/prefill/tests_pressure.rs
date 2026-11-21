//! Tests the free-region setpoint, the settle loop that packs the pools, and the
//! warm-budget slack.
//!
//! The setpoint scales with the span, so the same constants hold on a small card
//! and a large one, and decode asks for less than a load does. A span too small
//! for its floors clamps to half rather than demanding the floor, because a
//! setpoint the card cannot hold would mean permanent pressure. The settle loop
//! follows a clipped pass that freed ground and stops on a pass that gained
//! nothing, was refused, or only churned its own arenas. The warm slack keeps a
//! healthy drain pipeline from reading as over budget.

use super::*;

#[cfg(test)]
pub(super) mod setpoint_tests {
    use super::{setpoint_regions, VramPhase};

    /// The setpoint scales with the span so the same constants hold on this
    /// card's 226-region KV side and on the workstation's, and decode always
    /// insists on less than load — KV grows a chunk per sequence per 32 steps
    /// there, so evicting defensively would just cost reloads.
    #[test]
    fn the_setpoint_scales_with_the_span_and_decode_asks_for_less() {
        let load = setpoint_regions(VramPhase::Load, 800);
        let decode = setpoint_regions(VramPhase::Decode, 800);
        assert_eq!(load, 100, "load is span/8 once the span clears the floor");
        assert_eq!(decode, 50, "decode is span/16");
        assert!(decode < load);
    }

    /// On a span too small for the floors, the setpoint stops at half the span.
    /// Asking for more would mean permanent pressure: every wave would run a
    /// relief pass that cannot possibly reach a setpoint the card can't hold.
    #[test]
    fn a_small_span_clamps_to_half_rather_than_demanding_the_floor() {
        assert_eq!(setpoint_regions(VramPhase::Load, 32), 16);
        assert_eq!(setpoint_regions(VramPhase::Decode, 8), 4);
        assert_eq!(
            setpoint_regions(VramPhase::Load, 0),
            0,
            "no span, no demand"
        );
    }
}

#[cfg(test)]
pub(super) mod settle_passes_tests {
    use super::{settle_passes, PackPass};

    fn run(passes: &[Option<PackPass>]) -> (usize, usize) {
        let mut it = passes.iter().copied();
        let mut calls = 0;
        let released = settle_passes(|| {
            calls += 1;
            it.next()
                .expect("the loop asked for a pass past the script")
        });
        (released, calls)
    }

    fn capped(released: usize) -> Option<PackPass> {
        Some(PackPass {
            released,
            clipped: true,
            gained: released > 0,
        })
    }

    /// The 30B probe's burst: ~13 arenas a capped pass, 120 regions of air. Every
    /// pass is followed until one finishes its plan — nine passes, not four.
    #[test]
    fn a_burst_is_packed_until_a_pass_finishes_its_plan() {
        let mut script = vec![capped(13); 8];
        script.push(Some(PackPass {
            released: 16,
            clipped: false,
            gained: true,
        }));
        assert_eq!(run(&script), (8 * 13 + 16, 9));
    }

    /// A clipped pass that released nothing ends the loop: the next would plan the
    /// same moves from the same ground.
    #[test]
    fn a_clipped_pass_that_frees_nothing_ends_it() {
        assert_eq!(run(&[capped(12), capped(0)]), (12, 2));
    }

    /// A pass that released only the fresh arenas it created itself gained no
    /// ground, whatever its `released` count: the loop stops there rather than
    /// running the same fruitless pass again.
    #[test]
    fn a_pass_that_only_churns_its_own_arenas_ends_it() {
        let churn = Some(PackPass {
            released: 22,
            clipped: true,
            gained: false,
        });
        assert_eq!(run(&[capped(9), churn, capped(30)]), (9 + 22, 2));
    }

    /// Turned away by a forward or a migrate: nothing more to do now.
    #[test]
    fn a_refused_pass_ends_it() {
        assert_eq!(run(&[capped(5), None]), (5, 2));
        assert_eq!(run(&[None]), (0, 1));
    }
}

#[cfg(test)]
pub(super) mod warm_budget_tests {
    use super::WARM_PIPELINE_SLACK_BYTES;

    /// The slack exists so a zero-budget machine's transient drain traffic never
    /// reads as over-budget — a healthy pipeline peaked ~0.7 GiB.
    #[test]
    fn the_slack_clears_a_healthy_drain_pipeline() {
        assert_eq!(WARM_PIPELINE_SLACK_BYTES, 1 << 30);
    }
}
