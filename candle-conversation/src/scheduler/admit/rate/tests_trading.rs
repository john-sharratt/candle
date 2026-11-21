//! Checks the trade between prefill and decode, and the widths a cohort holds beside
//! running decodes.
//!
//! A charge is work already paid for, so it is taken whatever it costs and never
//! closes the wave, and charged decodes become the baseline the next offer is judged
//! against. A held cohort narrows as decodes join it, down to a least width, and a
//! prompt placed beside decodes is priced at the width the trade gives it rather than
//! at the dedicated forward's rate.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        approx, decode, offer, planner_learned, planner_with_layer, prefill, BW_RUN15, CAP,
        DECODE_CAP, DECODE_DISLODGE, FLOOR_16GB, GIB, LINK_4090_MOBILE, MIB, RESIDENT_RUN15,
    };

    // ── work the caller has already paid for ───────────────────────────────

    /// A charge is not a decision: it is taken whatever it costs — past the
    /// cap, under the floor, past the bus — and it never latches the wave full.
    #[test]
    fn a_charge_is_never_refused_and_never_closes_the_wave() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(FLOOR_16GB, FLOOR_16GB, FLOOR_16GB, 100, 2);
        // Over the cap and at the floor.
        p.charge(prefill(5_000), FLOOR_16GB - GIB);
        // Past the decode cap, and past the bus at that residency.
        for _ in 0..8 {
            p.charge(decode(2), FLOOR_16GB - GIB);
        }
        assert_eq!(p.tokens(), 5_000);
        assert_eq!(p.decodes(), 8);
        assert_eq!(p.routed_per_layer(), 24 * 8);
        assert_eq!(p.resident_now(), FLOOR_16GB - GIB);
        assert!(!p.is_full(), "a charge is not a refusal");
    }

    /// **A charged wave is not an empty one.** The head waiver exists so a wave
    /// carries something; a wave already carrying a held creep group or a
    /// stepping decode carries something, so the next offer is judged.
    #[test]
    fn a_charged_wave_judges_its_next_offer_instead_of_taking_it_as_the_head() {
        // Uncharged: the head is taken even though it crosses the floor.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(
            FLOOR_16GB + 100 * MIB,
            FLOOR_16GB,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        assert!(offer(&mut p, prefill(250), GIB).is_admitted());
        assert!(p.resident_now() < FLOOR_16GB);

        // Charged with a held creep first: the same offer is judged, and the
        // floor refuses it.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(
            FLOOR_16GB + 100 * MIB,
            FLOOR_16GB,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        p.charge(prefill(300), FLOOR_16GB + 100 * MIB);
        assert!(matches!(
            offer(&mut p, prefill(250), GIB),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert_eq!(p.tokens(), 300, "the charged rows stand, the offer did not");
    }

    /// A charged decode counts against the offers that follow it: the wave's
    /// decode rate, its routed set and its residency all move, so the next
    /// decode is judged against the wave as it will actually run.
    #[test]
    fn charged_decodes_are_the_baseline_the_next_offer_is_judged_against() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let mut resident = RESIDENT_RUN15;
        for n in 1..=9 {
            resident -= DECODE_DISLODGE;
            let projected = p.charge(decode(2), resident);
            assert!(approx(projected, p.decode_rate(n, 24 * n, resident), 1e-9));
        }
        assert_eq!(p.decodes(), 9);
        // The tenth is the one `decodes_stop_when_another_would_lower_the_
        // decode_rate` refuses — and it is refused here too, because the nine
        // ahead of it were charged rather than ignored.
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Refused(Refusal::Worse { .. })
        ));
    }

    /// A charge into an empty wave still leaves the projection readable, and a
    /// charge of nothing is a no-op on the counts.
    #[test]
    fn charging_reports_the_waves_projected_rate() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let projected = p.charge(prefill(474), RESIDENT_RUN15);
        assert!(approx(projected, 472.0, 0.02), "{projected}");
        assert_eq!(p.charge(prefill(0), RESIDENT_RUN15), projected);
        assert_eq!(p.tokens(), 474);
    }

    /// A planner whose decode side has learned a 20 ms layer, and a cohort whose
    /// every row's tier dislodges a MiB of weights from 12 GiB resident — so a
    /// wide cohort costs running decodes real stalls.
    pub(in super::super) fn traded() -> (WaveRate, impl Fn(usize) -> u64) {
        let mut rate = WaveRate::with_link_rate(
            25e9,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        );
        for _ in 0..WaveRate::MIN_DECODE_SAMPLES {
            rate.observe_decode(1, 0, u64::MAX, 20e-3);
        }
        let weights_at = |rows: usize| (12u64 << 30).saturating_sub((rows as u64) << 20);
        (rate, weights_at)
    }

    pub(in super::super) const FULL: usize = 8192;

    pub(in super::super) const LEAST: usize = 128;

    pub(in super::super) const CHUNK: usize = 32;

    /// **The held cohort's width, re-traded.** With nothing decoding it keeps
    /// its whole width; decodes narrow it, the more the narrower, never below
    /// the least advance; and where it stops is where the trade turns.
    #[test]
    fn a_held_cohort_narrows_with_the_decodes_beside_it() {
        let (rate, weights_at) = traded();
        let width = |decodes: usize| {
            rate.traded_width(
                FULL,
                LEAST,
                CHUNK,
                1.0,
                decodes,
                rate.routed_for(decodes),
                &weights_at,
            )
        };
        assert_eq!(width(0), FULL, "nothing decoding, nothing to trade");
        let (one, four, many) = (width(1), width(4), width(24));
        assert!(
            one >= four && four >= many,
            "monotone: {one} ≥ {four} ≥ {many}"
        );
        assert!(many < FULL, "24 decodes narrow a 8,192-row cohort: {many}");
        assert!(many >= LEAST, "never below the least advance: {many}");
        // The boundary is the trade's own: the last chunk in pays, the next out
        // does not.
        let routed = rate.routed_for(24);
        let net = |before: usize, after: usize| {
            rate.trade_net(
                before,
                after,
                weights_at(before),
                weights_at(after),
                1.0,
                24.0,
                24,
                routed,
            )
        };
        if many > LEAST {
            assert!(net(many - CHUNK, many) > 0.0, "the last chunk kept pays");
        }
        if many + CHUNK <= FULL {
            assert!(net(many, many + CHUNK) <= 1e-12, "the next chunk would not");
        }
    }

    /// **What a member advertises is the step a placed decode would get.** A
    /// wide cohort held with nothing decoding publishes the step at the width
    /// one placed decode would narrow it to — not at the width it stands at,
    /// which no placement would ever come to narrow.
    #[test]
    fn a_wide_cohort_advertises_the_step_at_its_traded_width() {
        let (rate, weights_at) = traded();
        let placed = rate.placed_decode_step_ms(0, FULL, LEAST, CHUNK, 1.0, &weights_at);
        let routed = rate.routed_for(1);
        let narrowed = rate.traded_width(FULL, LEAST, CHUNK, 1.0, 1, routed, &weights_at);
        let at = |rows: usize| rate.decode_step_secs(routed, weights_at(rows)) * 1e3;
        assert!(
            (placed - at(narrowed)).abs() < 1e-9,
            "{placed} against {}",
            at(narrowed)
        );
        assert!(
            placed <= at(FULL),
            "never slower than the cohort as it stands"
        );
    }

    /// **A prompt placed beside decodes is priced at the width the trade gives
    /// it**, in a forward carrying the decodes' rows — not at the dedicated
    /// forward's rate. On BlackWorld (a965e3a07) the pass ceiling's 2,390 tok/s
    /// drew a pattern's opening burst to a member draining a few hundred.
    #[test]
    fn a_prompt_placed_beside_decodes_is_priced_at_its_traded_width() {
        let (rate, weights_at) = traded();
        let alone = rate.placed_prefill_tok_s(FULL, LEAST, CHUNK, 1.0, 0, 0, 0.0, 0.0, &weights_at);
        let wide = FULL as f64 / rate.forward_secs(FULL, weights_at(FULL));
        assert!(
            (alone - wide).abs() < 1e-6 * wide,
            "with no decode the wide rate stands: {alone} against {wide}"
        );
        let decodes = 24;
        let beside = rate.placed_prefill_tok_s(
            FULL,
            LEAST,
            CHUNK,
            1.0,
            decodes,
            decodes,
            0.0,
            0.0,
            &weights_at,
        );
        let width = rate.traded_width(
            FULL,
            LEAST,
            CHUNK,
            1.0,
            decodes,
            rate.routed_for(decodes),
            &weights_at,
        );
        let expect = width as f64 / rate.forward_secs(width + decodes, weights_at(width));
        assert!(
            (beside - expect).abs() < 1e-6 * expect,
            "{beside} against {expect} at {width} rows"
        );
        assert!(
            beside < alone,
            "decodes beside it cost the prompt: {beside} vs {alone}"
        );
    }
}
