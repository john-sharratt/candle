//! Checks the prefill admission rules against the cards and wave sizes they were
//! measured on.
//!
//! Each test reads one rule: the floor, the cap, the first offer's head waiver, the
//! refusal on a worse rate, the refusal on a saturated one, and the joint budget that
//! interleaved offers share. The 16 GB card, the 72 GB Blackwell and the 3090 are the
//! cases that fixed the current behaviour, and each refusal is checked to name its
//! reason, so a wave that stops for the floor says so.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        approx, decode, fill_with, offer, planner, planner_learned, planner_with_layer, prefill,
        BW_RUN15, CAP, DECODE_CAP, DECODE_DISLODGE, EXEMPLAR, EXEMPLAR_DISLODGE, FLOOR_16GB, GIB,
        LINK_3090, LINK_4090_MOBILE, LINK_BLACKWELL, MIB, RESIDENT_RUN15,
    };

    /// The model's answer for the 16 GB card as run 15 stood, with exemplars
    /// dislodging a MiB a row: 2,379 tokens a wave at ~1,210 tok/s, against
    /// 472 today.
    #[test]
    fn the_16gb_card_widens_to_the_floor_at_twelve_hundred_tokens_a_second() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let projected = p.rate(2_379, FLOOR_16GB);
        assert!(approx(projected, 1_210.0, 0.01), "{projected}");
    }

    // ── the prefill budget on the 16 GB card ───────────────────────────────

    /// Reset, then admit exemplars until the floor stops it: on run 15's
    /// residency that is nine 250-token exemplars, the refusal names the floor
    /// with the residency the tenth would have left, and the wave is full.
    #[test]
    fn prefill_admissions_stop_at_the_floor_and_say_so() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        assert_eq!(
            admitted, 9,
            "2,379 MiB of room at a MiB a row is nine 250-row exemplars"
        );
        assert_eq!(p.tokens(), 2_250);
        match refusal {
            Refusal::Floor {
                resident_after,
                floor,
            } => {
                assert_eq!(floor, FLOOR_16GB);
                assert_eq!(resident_after, RESIDENT_RUN15 - 2_500 * MIB);
            }
            other => panic!("expected the floor, got {other:?}"),
        }
        assert!(p.is_full());
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15 - 2_250 * MIB,
            "a refusal moves nothing"
        );
    }

    /// The caller's dislodge is what the decision reads: the same tokens with
    /// no dislodge at all widen to the cap, and with a heavy dislodge stop on
    /// the rate, well above the floor.
    #[test]
    fn the_callers_dislodge_drives_the_decision() {
        // Nothing dislodged: the rate only climbs, the floor is never nearer.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, 2_000, DECODE_CAP);
        let (n, refusal) = fill_with(&mut p, prefill(EXEMPLAR), 0);
        assert_eq!(n, 8);
        assert!(matches!(refusal, Refusal::Cap { max_tokens: 2_000 }));
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
        // A GiB dislodged for 32 tokens: 70 ms of copy against 13 ms of
        // compute, on a wave already at 817 tok/s. Refused as worse, with two
        // GiB still above the floor.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 0).is_admitted());
        let (n, refusal) = fill_with(&mut p, prefill(32), GIB);
        assert_eq!(n, 0);
        assert!(
            matches!(refusal, Refusal::Worse { before, after } if after < before),
            "{refusal:?}"
        );
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15,
            "stopped on the rate, not the floor"
        );
    }

    /// An admission whose dislodged weights cost more copy than its tokens
    /// repay is refused as worse, with both rates.
    #[test]
    fn an_admission_that_costs_more_copy_than_it_repays_is_refused_as_worse() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 0).is_admitted());
        let current = p.rate(1_000, RESIDENT_RUN15);
        // 64 tokens that dislodge 2 GiB: 27 ms of tokens against 140 ms of copy.
        match offer(&mut p, prefill(64), 2 * GIB) {
            Admit::Refused(Refusal::Worse { before, after }) => {
                assert!(approx(before, current, 1e-9));
                assert!(after < before);
            }
            other => panic!("expected worse, got {other:?}"),
        }
        assert!(p.is_full());
    }

    /// A dislodge in proportion to the rows — the tier at a MiB a row — never
    /// makes widening worse: `W / (F + k·W)` only climbs. The stop is then the
    /// floor or saturation.
    #[test]
    fn a_proportional_dislodge_never_makes_widening_worse() {
        let mut p = planner(2e6).with_min_gain(0.01);
        let resident = ExpertGeometry::QWEN36_35B_A3B.total_bytes() - 64 * MIB;
        p.reset(resident, 0, 0, 1 << 20, DECODE_CAP);
        let mut last = 0.0;
        let refusal = loop {
            match offer(&mut p, prefill(64), 64 * MIB) {
                Admit::Admitted { projected } => {
                    assert!(projected > last, "monotone");
                    last = projected;
                }
                Admit::Refused(r) => break r,
            }
        };
        assert!(matches!(refusal, Refusal::Saturated { .. }), "{refusal:?}");
        assert!(
            p.tokens() >= 512 && p.tokens() <= 1_024,
            "tokens {}",
            p.tokens()
        );
    }

    /// The first offer after a reset is taken whatever it costs — under the
    /// floor, over the cap, past the bus — because a wave has to carry its
    /// head. The second is judged.
    #[test]
    fn the_first_admission_after_a_reset_is_always_taken() {
        // A prefill that would dislodge past the floor and is over the cap.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB + GIB, FLOOR_16GB, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, prefill(5_000), 2 * GIB).is_admitted());
        assert!(!p.is_full());
        assert!(
            p.resident_now() < FLOOR_16GB,
            "the head took the weights under the floor"
        );
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Cap { .. })
        ));
        // A decode whose copy alone exceeds the layer time at the seed.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(
            p.decode_copy_secs(24, RESIDENT_RUN15) > p.layer_secs(),
            "the seed cannot hide one drafted decode"
        );
        assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        assert_eq!(p.decodes(), 1);
        // A decode into a wave whose cap is zero.
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, 0);
        assert!(offer(&mut p, decode(0), DECODE_DISLODGE).is_admitted());
        assert!(matches!(
            offer(&mut p, decode(0), DECODE_DISLODGE),
            Admit::Refused(Refusal::DecodeCap { max_decodes: 0 })
        ));
    }

    /// Barely resident: the weight side stands one exemplar above the floor.
    /// The first fits, the second would cross, and the refusal is the floor.
    #[test]
    fn a_card_at_the_floor_admits_what_fits_and_refuses_the_rest() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(
            FLOOR_16GB + 300 * MIB,
            FLOOR_16GB,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        assert!(offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert!(p.is_full(), "the floor latched the wave full");
    }

    /// Exactly to the floor is allowed; one byte past it is not.
    #[test]
    fn the_floor_itself_is_reachable_and_not_crossable() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(
            FLOOR_16GB + 300 * MIB,
            FLOOR_16GB,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        assert!(offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE).is_admitted());
        assert!(
            offer(&mut p, prefill(50), 50 * MIB).is_admitted(),
            "exactly to the floor"
        );
        assert_eq!(p.resident_now(), FLOOR_16GB);
        assert!(matches!(
            offer(&mut p, prefill(1), 1),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// Under the floor already: the head is still taken (the caller's rule),
    /// and nothing else is — even an offer that dislodges nothing, because the
    /// weights after it are still under the floor.
    #[test]
    fn a_card_under_the_floor_takes_only_its_head() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(FLOOR_16GB - MIB, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(200), 0).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// The cap is the engine's widest forward, and it holds even with residency
    /// to spare.
    #[test]
    fn the_cap_bounds_the_wave_whatever_the_rate_says() {
        let mut p = planner(LINK_4090_MOBILE).with_min_gain(0.0);
        p.reset(9_600 * MIB, FLOOR_16GB, FLOOR_16GB, 1_024, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(25), 25 * MIB),
            Admit::Refused(Refusal::Cap { max_tokens: 1_024 })
        ));
        assert!(p.is_full());
    }

    /// Reset forgets the wave — tokens, decodes, residency, `full` — and keeps
    /// the learning.
    #[test]
    fn reset_clears_the_budget_and_keeps_the_estimates() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        let (bw, layer) = (p.effective_bytes_per_s(), p.layer_secs());
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        assert!(offer(&mut p, prefill(500), 500 * MIB).is_admitted());
        let _ = fill_with(&mut p, prefill(500), 500 * MIB);
        assert!(p.is_full());
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert_eq!(p.tokens(), 0);
        assert_eq!(p.decodes(), 0);
        assert_eq!(p.routed_per_layer(), 0);
        assert!(!p.is_full());
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
        assert_eq!(p.effective_bytes_per_s(), bw);
        assert_eq!(p.layer_secs(), layer);
    }

    /// When the copy is small (most of the experts resident) and the wave is
    /// already wide, another admission buys almost nothing, and the planner
    /// stops widening rather than spend residency for a fraction of a percent.
    #[test]
    fn a_saturated_wave_refuses_to_spend_residency_for_nothing() {
        // Blackwell: 20.3 GB of experts, 19.9 GB resident, fast bus.
        let mut p = planner(LINK_BLACKWELL).with_min_gain(0.01);
        let resident = 19_000 * MIB;
        p.reset(resident, 8 * GIB, 8 * GIB, 65_536, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, prefill(512), 512 * MIB);
        assert!(
            matches!(refusal, Refusal::Saturated { gain } if (0.0..0.01).contains(&gain)),
            "{refusal:?}"
        );
        assert!(p.resident_now() > 8 * GIB + GIB, "well short of the floor");
        assert!(p.tokens() < 65_536, "well short of the cap");
        assert!(admitted >= 2, "at least the first couple always pay");
    }

    /// **A prefill that spends no residency a forward must copy is never refused as
    /// saturated.** Every expert resident with room to spare — the 72 GiB card under
    /// the 35B-A3B — so each offer's dislodge leaves the copy at zero: the rate is
    /// flat, the gain is zero, and only the cap ends the wave. Refusing at the second
    /// offer would defer queued prompts into later waves that pay every per-wave cost
    /// again.
    #[test]
    fn a_prefill_that_spends_no_copied_residency_runs_to_the_cap() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_BLACKWELL).with_min_gain(0.01);
        // Far more resident than the experts need: dislodging 256 MiB an offer never
        // brings a single expert byte back onto the bus.
        let resident = total + 40 * GIB;
        p.reset(resident, 8 * GIB, 8 * GIB, 4096, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, prefill(512), 256 * MIB);
        assert!(
            matches!(refusal, Refusal::Cap { max_tokens: 4096 }),
            "{refusal:?}"
        );
        assert_eq!(admitted, 8, "4096 rows at 512 a prompt");
        assert!(p.non_resident_bytes(p.resident_now()) == 0);
    }

    /// With no minimum gain, only the floor or the cap ends a prefill wave
    /// whose dislodge is proportional.
    #[test]
    fn with_no_minimum_gain_only_the_floor_or_cap_ends_the_wave() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(128), 128 * MIB);
        assert!(matches!(refusal, Refusal::Floor { .. }), "{refusal:?}");
    }

    // ── one budget, one `full` ─────────────────────────────────────────────

    /// Decodes first, then prefills: the decodes reach their limit, and that
    /// refusal closes the wave to the prefills behind them.
    #[test]
    fn a_decode_refusal_closes_the_wave_to_later_prefills() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (nine, refusal) = fill_with(&mut p, decode(2), DECODE_DISLODGE);
        assert_eq!(nine, 9);
        assert!(matches!(refusal, Refusal::Worse { .. }));
        assert!(p.is_full());
        assert!(matches!(
            offer(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE),
            Admit::Refused(Refusal::Full)
        ));
        assert_eq!(p.tokens(), 0, "no prefill got in");
    }

    /// Decodes under their limit leave the wave open, and prefills then widen
    /// it on their own rule until the floor — from the residency the decodes'
    /// stores left, not from the reset figure.
    #[test]
    fn decodes_under_the_limit_leave_room_for_prefills() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        for _ in 0..4 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        }
        assert!(!p.is_full());
        let after_stores = RESIDENT_RUN15 - 4 * DECODE_DISLODGE;
        assert_eq!(p.resident_now(), after_stores);
        let (admitted, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        // 2,379 − 704 = 1,675 MiB of room: six exemplars, not nine.
        assert_eq!(admitted, 6);
        assert!(matches!(refusal, Refusal::Floor { .. }));
        assert_eq!(p.decodes(), 4);
        assert_eq!(p.tokens(), 1_500);
    }

    /// Prefills first, to the floor: the floor closes the wave to the decodes
    /// behind them — the caller offered the prefills first, so the wave is
    /// theirs.
    #[test]
    fn a_prefill_floor_closes_the_wave_to_later_decodes() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(EXEMPLAR), EXEMPLAR_DISLODGE);
        assert!(matches!(refusal, Refusal::Floor { .. }));
        assert!(matches!(
            offer(&mut p, decode(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        assert_eq!(p.decodes(), 0);
    }

    /// Full is full for everything, including offers that dislodge nothing.
    #[test]
    fn full_refuses_every_kind_until_reset() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, prefill(100), 100 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), 0),
            Admit::Refused(Refusal::Cap { .. })
        ));
        assert!(matches!(
            offer(&mut p, prefill(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        assert!(matches!(
            offer(&mut p, decode(0), 0),
            Admit::Refused(Refusal::Full)
        ));
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, 100, DECODE_CAP);
        assert!(offer(&mut p, decode(0), DECODE_DISLODGE).is_admitted());
    }

    /// Interleaved offers in the caller's priority order behave as one budget:
    /// each is judged on its own model against the wave so far, and every
    /// admission's weights carry into the next — the prefills' tiers count
    /// against the decodes' floor and the decodes' stores against the
    /// prefills'.
    #[test]
    fn interleaved_offers_share_one_budget() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        for _ in 0..3 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
            assert!(offer(&mut p, prefill(300), 300 * MIB).is_admitted());
        }
        assert_eq!((p.decodes(), p.tokens()), (3, 900));
        assert_eq!(
            p.resident_now(),
            RESIDENT_RUN15 - 3 * DECODE_DISLODGE - 900 * MIB
        );
        // 951 MiB above the floor: five more stores fit, and the floor — not
        // the rate — refuses the sixth.
        for _ in 0..5 {
            assert!(offer(&mut p, decode(2), DECODE_DISLODGE).is_admitted());
        }
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Refused(Refusal::Floor { .. })
        ));
        assert!(p.is_full());
    }

    /// The caller's `weights_before` is what the current rate is read at, so a
    /// caller whose residency moved between offers (a purchase, a give-back) is
    /// judged against the truth, not against the module's last figure: the
    /// same tokens to the same `weights_after` are admitted from one baseline
    /// and refused from another.
    #[test]
    fn the_callers_weights_before_is_the_baseline() {
        let resident = RESIDENT_RUN15 - 1_000 * MIB;
        // The weight side gave a GiB back since the last offer, and this
        // 32-token admission takes it again: 70 ms of copy for 13 ms of
        // compute, worse than the wave stood.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert_eq!(p.resident_now(), resident);
        match p.try_admit(prefill(32), resident + GIB, resident) {
            Admit::Refused(Refusal::Worse { before, after }) => {
                assert!(approx(before, p.rate(1_000, resident + GIB), 1e-9));
                assert!(after < before);
            }
            other => panic!("{other:?}"),
        }
        // Nothing moved since: the same 32 tokens to the same weights dislodge
        // nothing, and are admitted.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_min_gain(0.0);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(1_000), 1_000 * MIB).is_admitted());
        assert!(p.try_admit(prefill(32), resident, resident).is_admitted());
    }

    // ── other cards ────────────────────────────────────────────────────────

    /// The 3090: half the link, 12 GiB resident of 20.3 GB. The same exemplar
    /// costs longer and widening buys ~4×.
    #[test]
    fn the_3090_pays_the_slower_bus_and_gains_from_width() {
        let p = planner(LINK_3090); // seeded at 9.6 GB/s
        let r = 12 * GIB;
        let one = p.rate(250, r);
        let eight = p.rate(2_000, r - 1_750 * MIB);
        assert!(p.forward_secs(250, r) > 0.7 && p.forward_secs(250, r) < 1.1);
        assert!(
            eight / one > 3.5 && eight / one < 4.5,
            "ratio {}",
            eight / one
        );
    }

    /// Fully resident (the 72 GB Blackwell holds every expert) and nothing
    /// dislodged: there is no copy to amortise, the forward is pure compute,
    /// the rate is at the ceiling from the first row, and only the cap ends the
    /// prefill wave.
    #[test]
    fn a_fully_resident_card_is_compute_bound_and_only_the_cap_stops_it() {
        let mut p = planner(LINK_BLACKWELL).with_min_gain(0.0);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        assert_eq!(p.non_resident_bytes(total), 0);
        assert!(approx(p.forward_secs(1_000, total), 0.42, 1e-9));
        p.reset(total, 8 * GIB, 8 * GIB, CAP, DECODE_CAP);
        let (_, refusal) = fill_with(&mut p, prefill(1_024), 0);
        assert!(
            matches!(refusal, Refusal::Cap { max_tokens: CAP }),
            "{refusal:?}"
        );
        assert_eq!(p.tokens(), CAP);
    }

    /// A small card whose whole zone is under the floor the caller asks for:
    /// the head is taken, nothing else.
    #[test]
    fn a_card_too_small_for_the_floor_takes_only_its_head() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(2 * GIB, 3 * GIB, 3 * GIB, CAP, DECODE_CAP);
        assert!(offer(&mut p, prefill(64), 64 * MIB).is_admitted());
        assert!(matches!(
            offer(&mut p, prefill(1), MIB),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }
}
