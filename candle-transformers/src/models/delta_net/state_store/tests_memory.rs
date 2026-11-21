//! Tests that a store moves with its pending rows, is priced from its geometry
//! alone, and releases its pinned readback buffer.
//!
//! A decoding store moves with its unfolded rows, which wait in the slot between
//! steps. Its price comes from the geometry, so admission does not depend on which
//! stores happen to exist. An empty conv tail is refused, and a release frees the
//! readback buffer even after a seal that panicked while holding it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    use crate::models::gpu_test_lock::gpu_serial;

    use super::super::tests_layers::tests::{dims, filled_store_on, kinds};

    /// **A decoding store moves with its timeline rows pending.** They wait in the
    /// slot from step to step until a fold, so a store that refused to move while
    /// any were pending pinned the frontier under every running decode; moved,
    /// the destination holds the state, its tail and the unfolded rows, and the
    /// rows are still pending.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_store_relocates_with_its_timeline_rows_pending() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let mut store = filled_store_on(&dev);
        // A two-row span over layer 1 leaves its rows pending once committed.
        store.begin_wave()?;
        let (_, write) = store.layer_span_mut(1, 2)?;
        let SpanWrite::Timeline(timeline) = write else {
            unreachable!("a two-row span writes the timeline")
        };
        let n = timeline.elem_count();
        let rows: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        timeline.slice_set(
            &Tensor::from_vec(rows.clone(), timeline.shape(), &dev)?,
            0,
            0,
        )?;
        store.commit_wave();
        assert!(store.has_pending(), "the rows wait for a fold");
        let state = |store: &RecurrentStateStore| -> Result<Vec<Vec<f32>>> {
            store
                .slots
                .iter()
                .flat_map(|s| [&s.live.state.s, &s.live.state.conv_tail])
                .map(|t| t.flatten_all()?.to_vec1())
                .collect()
        };
        let before = state(&store)?;

        // Every live half onto a fresh slot of its stride.
        let stride = state_block(&dims()).bytes;
        let fresh = claim_arena_slots(&dev, SlotTenant::RecurrentState, stride, 3)?;
        let mut moves: HashMap<u64, ArenaSlot> = store
            .slots
            .iter()
            .filter_map(|s| s.live.held.as_ref().map(|h| h.ptr()))
            .zip(fresh)
            .collect();
        assert_eq!(store.relocate(&mut moves)?, 3, "every live half moved");
        assert!(moves.is_empty(), "every destination taken");

        assert_eq!(state(&store)?, before, "state and tail moved whole");
        let at = store.slots.iter().position(|s| s.layer_index == 1).unwrap();
        let moved: Vec<f32> = store.slots[at].live.timeline.flatten_all()?.to_vec1()?;
        assert_eq!(moved, rows, "the pending rows moved with it");
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(1, 2)]);
        Ok(())
    }

    /// The whole point of the geometry price: it answers the same for a process
    /// holding no stores as for one holding a hundred, because it never looks at
    /// them. Three DeltaNet layers, one 2,560 B slot each: 6,553 slots to a
    /// region, so each claims 2,561 B of it and the store 7,683 B — and the
    /// scratch a prefilling sequence adds is as much again.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_store_is_priced_from_geometry_not_from_what_is_resident() {
        let priced = RecurrentStateStore::reserved_bytes_for(&kinds(), &dims());
        assert_eq!(priced, 3 * 2561, "three slots at their share of a region");
        assert_eq!(
            RecurrentStateStore::scratch_bytes_for(&kinds(), &dims()),
            3 * 2561
        );

        // Nothing in the call depends on a store existing, so building some
        // cannot move it. This is the property the scheduler relies on.
        let _live: Vec<_> = (0..4)
            .map(|_| RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap())
            .collect();
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&kinds(), &dims()),
            priced,
            "four live stores must not change what one store costs"
        );
        // An attention-only stack carries no recurrent state and costs nothing.
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&[LayerKind::Attention], &dims()),
            0,
            "no DeltaNet layers, no reservation"
        );
    }

    /// An empty conv tail has no address of its own, so the store refuses the
    /// geometry rather than leasing a tail at the next region's base.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_empty_conv_tail_is_refused() -> Result<()> {
        let _gpu = gpu_serial();
        let d = DeltaNetDims {
            head_dim: 64,
            n_k_heads: 1,
            n_v_heads: 4,
            conv_kernel: 1,
        };
        let err = match RecurrentStateStore::new(&[LayerKind::DeltaNet], &d, &Device::new_cuda(0)?)
        {
            Ok(_) => panic!("an empty conv tail must be refused"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("empty half"), "{err}");
        Ok(())
    }

    /// **A release empties the readback slot**, so the buffer a shut-down engine's
    /// seals pinned is freed rather than held for the life of the process — and an
    /// empty slot releases as a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_release_empties_the_readback_slot() {
        let slot = Mutex::new(Some(PinnedBuf::Host {
            data: vec![0u8; 64],
        }));
        release_readback_in(&slot);
        assert!(slot.lock().unwrap().is_none(), "the buffer was released");
        release_readback_in(&slot);
        assert!(slot.lock().unwrap().is_none(), "an empty slot stays empty");
    }

    /// **A poisoned slot still releases.** A seal that panicked while holding the
    /// lock must not keep the buffer pinned past its engine's shutdown.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_poisoned_readback_slot_still_releases() {
        let slot = Arc::new(Mutex::new(Some(PinnedBuf::Host {
            data: vec![0u8; 64],
        })));
        let poisoner = Arc::clone(&slot);
        let _ = std::thread::spawn(move || {
            let _held = poisoner.lock().unwrap();
            panic!("a seal panicked holding the readback lock");
        })
        .join();
        assert!(slot.is_poisoned());
        release_readback_in(&slot);
        assert!(
            slot.lock().unwrap_or_else(|e| e.into_inner()).is_none(),
            "the buffer was released through the poison"
        );
    }
}
