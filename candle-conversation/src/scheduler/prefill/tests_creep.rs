//! Tests that a creep survives its members leaving and pausing, and that the
//! decodes beside it set its pace.
//!
//! A creep is a group sweeping layers together, so a member leaving mid-sweep
//! must take only its own rows and leave the survivors' sweep intact. Otherwise
//! the layers they already ran would be replayed into their state. The throttle
//! tests cover yielding at a decode's ratio, the layer budget, fitting a cohort
//! to its shortfall, and bounding a decode-less step to a slice. Each case runs
//! on a host arena of known rows, so the rows that remain can be checked in order.

use super::*;

#[cfg(test)]
pub(super) mod creep_member_drop_tests {
    use candle::DType;
    use candle_transformers::models::residual_rows::{copy_rows, RowCopy, SpanRows};

    use super::super::super::tests::make_test_scheduler;
    use super::super::super::*;
    use super::CreepResidual;

    fn section(seq_id: usize, advance: usize) -> WaveMember {
        WaveMember::Section { seq_id, advance }
    }

    /// A creep arena of `rows` rows of width two on the host, holding
    /// `0..2·rows` in order.
    fn held(rows: usize) -> CreepResidual {
        let mut c = CreepResidual::claim(&Device::Cpu, rows, 2, DType::F32).unwrap();
        let values: Vec<f32> = (0..rows * 2).map(|v| v as f32).collect();
        let src = Tensor::from_vec(values, (rows, 2), &Device::Cpu).unwrap();
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: rows,
        }];
        copy_rows(&SpanRows::new(&src, 2).unwrap(), c.pending(), &plan).unwrap();
        c.commit();
        c
    }

    /// The values a creep arena holds, in caller order.
    fn held_values(c: &CreepResidual) -> Vec<f32> {
        c.committed()
            .to_tensor()
            .unwrap()
            .narrow(0, 0, c.rows())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// **A member leaving mid-sweep takes only its own rows.** The survivors'
    /// rows stay, in order, and the sweep keeps its cursor — it is not
    /// restarted, which would replay layers into state that already holds them.
    #[test]
    fn a_member_leaving_mid_sweep_takes_only_its_rows() {
        let (mut sched, _tx) = make_test_scheduler();
        sched.wave_prefill_members = vec![section(1, 2), section(2, 3), section(3, 1)];
        sched.wave_prefill_rows = vec![2, 3, 1];
        sched.wave_prefill_cursor = 7;
        sched.wave_prefill_residual = Some(held(6));

        sched.drop_wave_members(&[true, false, true]).unwrap();

        let res = sched.wave_prefill_residual.as_ref().expect("still held");
        assert_eq!(res.rows(), 3);
        assert_eq!(
            held_values(res),
            vec![0., 1., 2., 3., 10., 11.],
            "member 1's two rows, then member 3's one"
        );
        let ids: Vec<usize> = sched
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(ids, vec![1, 3]);
        assert_eq!(sched.wave_prefill_rows, vec![2, 1]);
        assert_eq!(sched.wave_prefill_cursor, 7, "the sweep goes on");
    }

    /// With every member gone there is nothing to carry: the group is over.
    #[test]
    fn a_sweep_whose_members_all_leave_ends() {
        let (mut sched, _tx) = make_test_scheduler();
        sched.wave_prefill_members = vec![section(1, 2)];
        sched.wave_prefill_rows = vec![2];
        sched.wave_prefill_cursor = 7;
        sched.wave_prefill_residual = Some(held(2));

        sched.drop_wave_members(&[false]).unwrap();

        assert!(sched.wave_prefill_residual.is_none());
        assert!(sched.wave_prefill_members.is_empty());
        assert_eq!(sched.wave_prefill_cursor, 0);
    }

    /// **Freeing a member's slot keeps the other members' sweep.** It used to
    /// abandon the whole group, and the survivors re-ran their finished layers.
    #[test]
    fn freeing_a_members_slot_keeps_the_others_sweep() {
        let (mut sched, _tx) = make_test_scheduler();
        sched.wave_prefill_members = vec![section(1, 2), section(2, 2)];
        sched.wave_prefill_rows = vec![2, 2];
        sched.wave_prefill_cursor = 5;
        sched.wave_prefill_residual = Some(held(4));

        sched.purge_freed_slot_scheduling_state(SequenceId(1));

        let ids: Vec<usize> = sched
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(ids, vec![2]);
        assert_eq!(sched.wave_prefill_cursor, 5);
        let res = sched.wave_prefill_residual.as_ref().unwrap();
        assert_eq!(held_values(res), vec![4., 5., 6., 7.], "member 2's rows");
    }

    /// **A paused creep resumes exactly where it stopped**: the same residual,
    /// layer, members and rows — and while paused nothing is in flight.
    #[test]
    fn a_paused_creep_resumes_where_it_stopped() {
        let (mut sched, _tx) = make_test_scheduler();
        sched.wave_prefill_members = vec![section(1, 2), section(2, 3)];
        sched.wave_prefill_rows = vec![2, 3];
        sched.wave_prefill_cursor = 9;
        sched.wave_prefill_residual = Some(held(5));

        sched.pause_creep();
        assert!(sched.wave_prefill_members.is_empty());
        assert_eq!(sched.wave_prefill_cursor, 0);
        assert!(sched.wave_prefill_residual.is_none());
        assert_eq!(
            sched.wave_group_section_seqs(),
            [1, 2].into_iter().collect(),
            "a paused section is still in flight for the standalone pass"
        );

        assert!(sched.resume_creep(), "nothing waits above it");
        let ids: Vec<usize> = sched
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(sched.wave_prefill_rows, vec![2, 3]);
        assert_eq!(sched.wave_prefill_cursor, 9);
        assert_eq!(
            held_values(sched.wave_prefill_residual.as_ref().unwrap()),
            (0..10).map(|v| v as f32).collect::<Vec<_>>(),
            "the same rows it paused with"
        );
        assert!(sched.paused_creep.is_none());
    }

    /// Freeing a paused member's slot takes it out of the paused creep, and
    /// the last one leaving ends it.
    #[test]
    fn freeing_a_paused_member_takes_it_out_of_the_paused_creep() {
        let (mut sched, _tx) = make_test_scheduler();
        sched.wave_prefill_members = vec![section(1, 2), section(2, 2)];
        sched.wave_prefill_rows = vec![2, 2];
        sched.wave_prefill_cursor = 4;
        sched.wave_prefill_residual = Some(held(4));
        sched.pause_creep();

        sched.purge_freed_slot_scheduling_state(SequenceId(1));
        let paused = sched.paused_creep.as_ref().expect("one member left");
        assert_eq!(paused.rows, vec![2]);
        assert_eq!(
            held_values(paused.residual.as_ref().unwrap()),
            vec![4., 5., 6., 7.]
        );

        sched.purge_freed_slot_scheduling_state(SequenceId(2));
        assert!(sched.paused_creep.is_none());
    }
}

#[cfg(test)]
pub(super) mod creep_throttle_tests {
    use super::{creep_layer_budget, throttle_ratio, DECODE_LESS_SLICE, HIGH_BESIDE_HIGH_RATIO};
    use crate::projection::DecodePriority::{High, Low, Normal};
    use std::time::Duration;

    /// Work yields to a decode that outranks it at that decode's ratio, and
    /// `High` work to a `High` decode at [`HIGH_BESIDE_HIGH_RATIO`] — two
    /// conversations share the device, the decoding one keeping two-thirds; it
    /// runs whole beside none.
    #[test]
    fn work_yields_to_an_outranking_decode_and_high_to_high() {
        assert_eq!(throttle_ratio(High, []), 1, "nothing decoding");
        assert_eq!(HIGH_BESIDE_HIGH_RATIO, 2);
        assert_eq!(
            throttle_ratio(High, [High]),
            2,
            "a second conversation's prompt beside the first's decode"
        );
        assert_eq!(throttle_ratio(High, [High, Normal, Low]), 2);
        assert_eq!(throttle_ratio(High, [Normal, Low]), 1, "outranks them all");
        assert_eq!(
            throttle_ratio(Low, [Low]),
            1,
            "an ingest beside its own turns"
        );
        assert_eq!(
            throttle_ratio(Normal, [Normal]),
            1,
            "below High, a band's own decodes never hold its work back"
        );
        assert_eq!(throttle_ratio(Low, [High]), 64);
        assert_eq!(throttle_ratio(Low, [Normal]), 16);
        assert_eq!(throttle_ratio(Low, [Normal, High]), 64, "the highest ratio");
        assert_eq!(throttle_ratio(Normal, [High, Low]), 64);
        assert_eq!(throttle_ratio(Normal, [Normal, Low]), 1);
    }

    /// `Urgent` is paced as `High` is beside its own band, and outranks it:
    /// a live dialogue's decode holds a Maker's table prompt back at the full
    /// ratio, while a Maker's decode never holds an urgent prompt back.
    #[test]
    fn urgent_paces_as_high_and_outranks_it() {
        use crate::projection::DecodePriority::Urgent;
        assert_eq!(throttle_ratio(Urgent, [Urgent]), HIGH_BESIDE_HIGH_RATIO);
        assert_eq!(throttle_ratio(High, [Urgent]), 64);
        assert_eq!(throttle_ratio(Urgent, [High, Normal, Low]), 1);
        assert_eq!(throttle_ratio(Low, [Urgent]), 64);
    }

    /// `ceil(n / R)` layers a wave, never zero.
    #[test]
    fn the_layer_budget_is_the_layers_over_the_ratio_rounded_up() {
        assert_eq!(creep_layer_budget(48, 1), 48);
        assert_eq!(creep_layer_budget(48, 16), 3);
        assert_eq!(creep_layer_budget(40, 16), 3);
        assert_eq!(creep_layer_budget(48, 64), 1);
        assert_eq!(creep_layer_budget(1, 64), 1);
        assert_eq!(creep_layer_budget(48, 0), 48, "a zero ratio reads as 1");
    }

    /// A cohort that does not fit its ground shrinks by the shortfall, with a
    /// margin, to fewer rows and never under a chunk; one that fits, or is a
    /// chunk already, is left alone. The measured case: 8,192 rows needing
    /// 7,264 + 200 MiB against 7,000 MiB of ground.
    #[test]
    fn a_cohort_shrinks_by_its_shortfall_to_fit() {
        use super::{fitted_rows, CHUNK_SIZE};
        let mib = 1usize << 20;
        assert_eq!(fitted_rows(8_192, 7_000 * mib, 7_000 * mib), None, "fits");
        let rows = fitted_rows(8_192, 7_464 * mib, 7_000 * mib).expect("shrinks");
        assert!((CHUNK_SIZE..8_192).contains(&rows));
        assert_eq!(rows, (8_192.0 * 7_000.0 / 7_464.0 * 0.97) as usize);
        assert_eq!(
            fitted_rows(CHUNK_SIZE, 9_000 * mib, mib),
            None,
            "a chunk is the floor"
        );
        assert_eq!(
            fitted_rows(CHUNK_SIZE + 1, 9_000 * mib, mib),
            Some(CHUNK_SIZE),
            "never under a chunk"
        );
    }

    /// **A decode-less step is a slice, not the sweep.** The measured case: an
    /// 8,190-row cohort at ~6.6 µs a layer-row (2.6 s over 48 layers) runs a
    /// handful of layers a step within `DECODE_LESS_SLICE`, so a prompt arriving
    /// between two of a chat's turns waits ~a quarter second, not 2.6 s; a
    /// small cohort still finishes its sweep in one step; unmeasured, one layer.
    #[test]
    fn a_decode_less_step_is_bounded_by_its_slice() {
        use super::super::super::creep_slice::time_bounded_layers;
        let per_layer_row = Duration::from_nanos(6_600);
        let layers = time_bounded_layers(
            creep_layer_budget(48, 1),
            8_190,
            Some(per_layer_row),
            None,
            Some(DECODE_LESS_SLICE),
        );
        assert_eq!(layers, 4, "250 ms / (8,190 × 6.6 µs) = 4.6 → 4 layers");
        let step = per_layer_row.mul_f64((8_190 * layers) as f64);
        assert!(step <= DECODE_LESS_SLICE, "a step of {step:?}");
        let small = time_bounded_layers(48, 19, Some(per_layer_row), None, Some(DECODE_LESS_SLICE));
        assert_eq!(small, 48, "a 19-row prompt sweeps whole");
        assert_eq!(
            time_bounded_layers(48, 8_190, None, None, Some(DECODE_LESS_SLICE)),
            1,
            "unmeasured, one layer"
        );
    }
}
