//! Checks how a phase's chains combine, and how the two layer phases share a
//! region beside the forward span.
//!
//! A prelude runs ahead of a phase's mixer rather than instead of it, and a
//! hybrid's two mixers are a maximum, never a sum. The two layer phases share one
//! region while the forward span stands beside it, and phase plans round to the
//! span quantum so the footprint covers the layout exactly. A wave costs the
//! larger of its phases rather than their sum, because the generations never run
//! at once.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{gated_partial_rotary, hyper_moe, moe, w};

    /// A prelude runs AHEAD of the phase's mixer, not instead of it: the phase
    /// walks the pre-mix and then the largest alternative on from its end.
    #[test]
    fn a_phase_with_a_prelude_walks_it_before_the_mixer() {
        let p = WavePlan::new(hyper_moe());
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(
            p.phase_bytes(LayerPhase::Ffn, wide),
            p.walk_chain(10_731_776, Chain::Ffn, wide)
        );
        let attn = p
            .walk_chain(10_731_776, Chain::Attention, wide)
            .max(p.walk_chain(10_731_776, Chain::DeltaNet, wide));
        assert_eq!(p.phase_bytes(LayerPhase::Attention, wide), attn);
        // And a stack with no prelude prices exactly as it did.
        let plain = WavePlan::new(moe());
        assert_eq!(
            plain.phase_bytes(LayerPhase::Ffn, wide),
            plain.chain_bytes(Chain::Ffn, wide)
        );
    }

    /// **The two layer phases share one region; the forward span stands beside
    /// it.** The footprint and the layout, as raw figures.
    #[test]
    fn the_layer_phases_share_a_region_beside_the_forward_span() {
        assert_eq!(tier_footprint([300, 200, 50]), 350);
        assert_eq!(tier_footprint([200, 300, 50]), 350);
        assert_eq!(tier_footprint([0, 0, 50]), 50);
        assert_eq!(phase_offsets([300, 200, 50]), [0, 0, 300]);
        assert_eq!(phase_offsets([200, 300, 50]), [0, 0, 300]);
    }

    /// Plans are rounded to the span quantum, so the layout keeps every span
    /// base on the boundary and the footprint covers the layout exactly.
    #[test]
    fn phase_plans_round_to_the_span_quantum() {
        let plan = align_phase_plan([1, WAVE_SPAN_ALIGN, WAVE_SPAN_ALIGN + 1]);
        assert_eq!(
            plan,
            [WAVE_SPAN_ALIGN, WAVE_SPAN_ALIGN, 2 * WAVE_SPAN_ALIGN]
        );
        assert_eq!(align_phase_plan([0, 0, 0]), [0, 0, 0]);
        // Already-rounded plans pass through unchanged.
        let exact = [4 * WAVE_SPAN_ALIGN, 512 * WAVE_SPAN_ALIGN, WAVE_SPAN_ALIGN];
        assert_eq!(align_phase_plan(exact), exact);
    }

    /// **A hybrid's two mixers are a max, never a sum.** A layer runs one of
    /// them, so charging both prices a wave for a layer that does not exist —
    /// and charging neither under-sizes the arena the other one carves from.
    #[test]
    fn the_attention_phase_takes_the_larger_mixer_chain() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let rows = WaveWidth::prefill(2100, 4);
        let attn = plan.chain_bytes(Chain::Attention, rows);
        let dn = plan.chain_bytes(Chain::DeltaNet, rows);
        assert!(dn > 0, "a hybrid geometry must price its mixer chain");
        assert_eq!(
            plan.phase_bytes(LayerPhase::Attention, rows),
            attn.max(dn),
            "the phase must be the larger chain, not their sum ({attn} + {dn})"
        );
        // And an all-attention stack prices its mixer chain at nothing at all,
        // alignment included, so no such model's span moves.
        let plain = WavePlan::new(moe());
        assert_eq!(plain.chain_bytes(Chain::DeltaNet, rows), 0);
        assert_eq!(
            plain.phase_bytes(LayerPhase::Attention, rows),
            plain.chain_bytes(Chain::Attention, rows)
        );
    }

    /// A wave costs the larger phase, never the sum: the two generations do not
    /// overlap, so summing would price twice what a half holds at once.
    #[test]
    fn a_wave_costs_the_larger_phase_not_the_sum() {
        let plan = WavePlan::new(moe());
        let sum = plan.phase_bytes(LayerPhase::Attention, w(64))
            + plan.phase_bytes(LayerPhase::Ffn, w(64));
        assert!(plan.wave_bytes(w(64)) < sum);
        assert!(plan.wave_bytes(w(64)) >= plan.phase_bytes(LayerPhase::Ffn, w(64)));
    }
}
