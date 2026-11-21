//! Tests the wave rules on a single layer: a short span is folded or discarded, a
//! pending layer refuses a second span, and a rollback drops only its own rows.
//!
//! A short span leaves the state as it entered until a fold lands it. A long span
//! needs the scratch, which is released once the sequence stops prefilling. Commit
//! and rollback keep both buffers of a half together, so a state is never a wave
//! ahead of its tail.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::delta_net::mix::DeltaNetOut;

    #[cfg(feature = "cuda")]
    use candle_nn::kv_cache::slot_stride;

    /// The wire bytes are each float's IEEE-754 little-endian image, in order.
    #[test]
    fn f32_le_bytes_is_each_floats_little_endian_image_in_order() {
        assert_eq!(
            f32_le_bytes(&[1.0, -2.5, 0.0]),
            vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x20, 0xC0, 0x00, 0x00, 0x00, 0x00]
        );
        assert!(f32_le_bytes(&[]).is_empty());
    }

    pub(in super::super) fn dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        }
    }

    pub(in super::super) fn kinds() -> Vec<LayerKind> {
        vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
            LayerKind::DeltaNet,
        ]
    }

    /// One long span's writes into a layer's scratch: `live + 1` in both
    /// buffers, which is what the scan kernels do to their two write pointers.
    pub(in super::super) fn bump_into(live: &DeltaNetState, out: &DeltaNetOut) {
        let one = |src: &Tensor, dst: &Tensor| {
            let ones = Tensor::ones(src.shape(), src.dtype(), src.device()).unwrap();
            dst.slice_set(&src.add(&ones).unwrap(), 0, 0).unwrap();
        };
        one(&live.s, &out.s);
        one(&live.conv_tail, &out.conv_tail);
    }

    /// Run a long span over `layer` of `store`: take its write and bump.
    pub(in super::super) fn long_span(store: &mut RecurrentStateStore, layer: usize) {
        let (live, write) = store.layer_span_mut(layer, TIMELINE_ROWS + 1).unwrap();
        let SpanWrite::State(out) = write else {
            panic!("a long span writes a state");
        };
        bump_into(live, &out);
    }

    /// Run a one-row span over `layer` of `store`, writing its timeline with a
    /// known generator: G = 0, k = 1, v_new = `v`, x = `x`. Folded, every element
    /// of S moves by `v` and the tail shifts `x` in.
    pub(in super::super) fn short_span(
        store: &mut RecurrentStateStore,
        layer: usize,
        v: f32,
        x: f32,
    ) {
        let d = dims();
        let lay = TimelineLayout::of(&d);
        let (_, write) = store.layer_span_mut(layer, 1).unwrap();
        let SpanWrite::Timeline(tl) = write else {
            panic!("a one-row span writes a timeline");
        };
        let dev = tl.device().clone();
        let full = |shape: &[usize], v: f32| Tensor::full(v, shape, &dev).unwrap();
        lay.write_rows(
            &tl,
            &full(&[1, d.n_v_heads], 0.0),
            &full(&[1, d.n_k_heads, d.head_dim], 1.0),
            &full(&[1, d.n_v_heads, d.head_dim], v),
            &full(&[1, d.conv_dim()], x),
        )
        .unwrap();
    }

    pub(in super::super) fn filled_store() -> RecurrentStateStore {
        filled_store_on(&Device::Cpu)
    }

    pub(in super::super) fn filled_store_on(dev: &Device) -> RecurrentStateStore {
        let dev = dev.clone();
        let d = dims();
        let mut store = RecurrentStateStore::new(&kinds(), &d, &dev).unwrap();
        for (i, li) in [0usize, 1, 3].iter().enumerate() {
            let n = d.state_elems();
            let s: Vec<f32> = (0..n).map(|j| (i * 1000 + j) as f32 * 0.01).collect();
            let tn = d.conv_state_elems();
            let t: Vec<f32> = (0..tn).map(|j| (i * 100 + j) as f32 * 0.1).collect();
            let live = store.layer_state_mut(*li).unwrap();
            live.copy_from(&DeltaNetState {
                s: Tensor::from_vec(s, (d.n_v_heads, d.head_dim, d.head_dim), &dev).unwrap(),
                conv_tail: Tensor::from_vec(t, (d.conv_dim(), d.conv_kernel - 1), &dev).unwrap(),
            })
            .unwrap();
        }
        store
    }

    /// **A short span leaves the state as it entered; a fold lands it.** The span
    /// writes its timeline and the layer is pending; the live state does not move
    /// until the fold, which applies exactly the timeline's rows.
    #[test]
    fn a_short_span_writes_its_timeline_and_a_fold_lands_it() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        store.commit_wave();
        assert!(store.has_pending());
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
        assert_eq!(
            store
                .layer_state(0)
                .unwrap()
                .s
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            {
                let e: Vec<f32> = entry[0]
                    .state
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                e
            },
            "a short span wrote the live state"
        );

        store.fold(None).unwrap();
        assert!(!store.has_pending());
        let after = store.export().unwrap();
        // S·e^0 + 2·1 on every element of layer 0; layers 1 and 3 untouched.
        let s0: Vec<f32> = entry[0]
            .state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) + 2.0)
            .collect();
        assert_eq!(after[0].state, f32_le_bytes(&s0));
        // The tail shifted the row's raw input in: [t0 t1] → [t1 7].
        let t: Vec<f32> = entry[0]
            .conv_tail
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let shifted: Vec<f32> = t.chunks(2).flat_map(|c| [c[1], 7.0]).collect();
        assert_eq!(after[0].conv_tail, f32_le_bytes(&shifted));
        assert_eq!(after[1], entry[1]);
        assert_eq!(after[2], entry[2]);
    }

    /// A span with no wave open is refused: its writes would belong to nothing
    /// that could roll them back.
    #[test]
    fn a_span_outside_a_wave_is_refused() {
        let mut store = filled_store();
        let err = store.layer_span_mut(0, 1).unwrap_err();
        assert!(err.to_string().contains("no wave open"), "{err}");
    }

    /// A discard drops the rows: the state is as the span found it.
    #[test]
    fn a_discard_leaves_the_state_as_it_entered() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 5.0, 1.0);
        store.commit_wave();
        store.discard_pending();
        assert!(!store.has_pending());
        assert_eq!(store.export().unwrap(), entry);
    }

    /// A fold keeping fewer rows than the span wrote applies only those; one
    /// keeping none applies nothing.
    #[test]
    fn a_fold_keeps_the_rows_it_is_told_to() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 3.0, 1.0);
        store.commit_wave();
        store.fold(Some(0)).unwrap();
        assert!(!store.has_pending(), "a fold of nothing still ends pending");
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a fold of none applied rows"
        );
    }

    /// **A pending layer refuses a second span**, and every whole-store reader
    /// refuses a pending store: its live state is not where the sequence stands.
    #[test]
    fn a_pending_layer_refuses_a_second_span_and_the_store_refuses_its_readers() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();

        store.begin_wave().unwrap();
        let err = store.layer_span_mut(0, 1).unwrap_err();
        assert!(err.to_string().contains("pending"), "{err}");
        assert!(
            store.layer_span_mut(1, 1).is_ok(),
            "another layer runs freely"
        );
        store.rollback_wave().unwrap();

        assert!(store.export().is_err(), "export with rows pending");
        assert!(store.fork_from().is_err(), "fork with rows pending");
        store.fold(None).unwrap();
        assert!(store.export().is_ok());
        assert!(store.fork_from().is_ok());
    }

    /// **A rollback drops what its own wave wrote and nothing else.** A step's
    /// first segment commits with layer 0 pending; its second segment runs
    /// layer 1 and fails: layer 1's rows go, layer 0's stay for the reconcile.
    #[test]
    fn a_rollback_drops_its_own_waves_rows_and_keeps_an_earlier_segments() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.rollback_wave().unwrap();
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
    }

    /// A long span needs the scratch, claimed between forwards.
    #[test]
    fn a_long_span_needs_the_prefill_scratch() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        let err = store.layer_span_mut(0, TIMELINE_ROWS + 1).unwrap_err();
        assert!(err.to_string().contains("scratch"), "{err}");
        assert!(store.ensure_scratch().is_err(), "claimed mid-wave");
        store.rollback_wave().unwrap();
        store.ensure_scratch().unwrap();
        assert!(store.holds_scratch());
    }

    #[test]
    fn wave_rollback_restores_entry_state_and_commit_keeps_writes() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "rollback must restore the wave-entry state exactly"
        );

        // A successful wave: mutate, commit — the write stands, in BOTH
        // buffers. Asserting only on `s` would pass while the conv tail was
        // left behind in the half the exchange filed away, which is precisely
        // the failure a partial exchange produces: a state one wave ahead of
        // its tail.
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();
        let committed = store.export().unwrap();
        assert_ne!(
            committed[0].state, entry[0].state,
            "commit must install `s`"
        );
        assert_ne!(
            committed[0].conv_tail, entry[0].conv_tail,
            "commit must install the advanced conv tail, not just `s`"
        );
        // Layer 1 never ran, so its slot keeps its state as it was.
        assert_eq!(
            committed[1], entry[1],
            "an unrun layer must not be exchanged"
        );
    }

    /// **A wave never writes the state it read, so an entering alias is never
    /// disturbed by a wave that fails.** Short and long spans alike: the short
    /// one writes its timeline, the long one the scratch.
    #[test]
    fn a_wave_leaves_the_entering_buffer_untouched() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        // Shares storage with the slot's entering state — the same view the
        // decode kernel's pointer table holds for this wave.
        let alias = store.layer_state(0).unwrap().s.clone();
        let entry: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        short_span(&mut store, 1, 1.0, 1.0);
        let during: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            during, entry,
            "the wave wrote into the buffer it was reading — the entering state \
             is gone and a rollback has nothing to return to"
        );
        store.rollback_wave().unwrap();
        let after: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(after, entry, "rollback must leave the entering state alone");
    }

    /// **The scratch is held while the sequence prefills and no longer.** A
    /// commit that ran a long span keeps it; the first commit that ran none —
    /// the sequence decoding — gives it back.
    #[test]
    fn the_scratch_goes_back_after_the_first_wave_with_no_long_span() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();
        assert!(
            store.holds_scratch(),
            "a prefilling sequence keeps its scratch"
        );

        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        assert!(
            !store.holds_scratch(),
            "a decoding sequence holds no scratch"
        );
    }

    /// **A segment that reaches no recurrent layer keeps the scratch.** A
    /// prefill chunk split into layer windows can run a window of attention
    /// layers only; its commit says nothing about whether the sequence still
    /// prefills, and dropping the scratch there would refuse the chunk's next
    /// window.
    #[test]
    fn a_wave_that_runs_no_recurrent_layer_keeps_the_scratch() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();

        store.begin_wave().unwrap();
        store.commit_wave();
        assert!(
            store.holds_scratch(),
            "an attention-only window dropped the scratch"
        );

        store.begin_wave().unwrap();
        long_span(&mut store, 3);
        store.commit_wave();
        assert!(store.holds_scratch());
    }

    /// A reset store reads exactly as a fresh one — every layer's `s` and conv
    /// tail back to zero, nothing pending, no scratch, and no longer seeded —
    /// even when a wave was left open on it.
    #[test]
    fn reset_returns_the_store_to_the_sequence_start_state() {
        let mut store = filled_store();
        store.mark_seeded();
        store.ensure_scratch().unwrap();
        let fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert_ne!(store.export().unwrap(), fresh.export().unwrap());

        // Rows left pending by a committed segment, then a wave left open.
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.commit_wave();
        store.begin_wave().unwrap();

        store.reset().unwrap();
        assert!(!store.has_pending());
        assert!(!store.holds_scratch());
        assert_eq!(
            store.export().unwrap(),
            fresh.export().unwrap(),
            "a reset store must hold the sequence-start state byte for byte"
        );
        assert!(!store.is_seeded());

        // A span on the reset store starts from zero: one row of v = 1 reads as ones.
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        store.fold(None).unwrap();
        let s: Vec<f32> = store
            .layer_state(0)
            .unwrap()
            .s
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(s.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn wave_sequencing_is_enforced() {
        let mut store = filled_store();
        assert!(store.rollback_wave().is_err(), "rollback with no wave open");
        store.begin_wave().unwrap();
        assert!(store.begin_wave().is_err(), "overlapping wave");
        assert!(store.export().is_err(), "export mid-wave");
        assert!(store.fold(None).is_err(), "fold mid-wave");
        store.commit_wave();
        assert!(store.export().is_ok());
    }

    /// The fixture's slot: a 256 B `s`, a 256 B conv tail on the next 256-byte
    /// boundary, and a 1,920 B timeline (8 rows of 4 + 8 + 16 + 32 floats) on
    /// the next — a 2,432 B block in a 2,560 B slot.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_slot_is_state_then_tail_then_timeline() {
        assert_eq!(DeltaNetState::byte_sizes(&dims()), (256, 256));
        assert_eq!(
            state_block(&dims()),
            StateBlock {
                conv_off: 256,
                state_end: 512,
                timeline_off: 512,
                bytes: 2432,
            }
        );
        assert_eq!(slot_stride(state_block(&dims()).bytes), 2560);
    }
}
