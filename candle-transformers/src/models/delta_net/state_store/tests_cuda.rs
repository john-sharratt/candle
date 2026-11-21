//! Tests on the device that resets, readbacks, folds and restores match the CPU
//! store byte for byte.
//!
//! A reset zeroes the state in place without a new claim, and the pinned readback
//! yields the bytes the CPU export does. The device fold lands what the CPU fold
//! lands, and releasing the readback buffer unpins exactly what the seal pinned. A
//! restore's uploads come from a pooled buffer and land the bytes the snapshot
//! holds.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    use crate::models::gpu_test_lock::gpu_serial;

    use super::super::tests_layers::tests::{dims, filled_store, filled_store_on, kinds};

    /// **A reset zeroes the state on the slots it already holds.** Same decode
    /// addresses before and after — no claim, no release — and the export reads
    /// as a fresh store's.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_reset_zeroes_in_place_without_a_new_claim() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let mut store = filled_store_on(&dev);
        let before: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        store.reset()?;
        let after: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(after, before, "a reset must keep the store's slots");
        let fresh = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        assert_eq!(store.export()?, fresh.export()?);
        Ok(())
    }

    /// **The pinned readback is the same bytes the per-tensor one was.** A CUDA
    /// store filled with the fixture's values exports byte-for-byte what the CPU
    /// store does — every layer, state and conv tail, in slot order.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_export_is_byte_identical_to_the_cpu_one() -> Result<()> {
        let _gpu = gpu_serial();
        let cuda = filled_store_on(&Device::new_cuda(0)?).export()?;
        let cpu = filled_store().export()?;
        assert_eq!(cuda, cpu);
        // Twice, so the second export reads through the buffer the first sized.
        let again = filled_store_on(&Device::new_cuda(0)?).export()?;
        assert_eq!(again, cpu);
        Ok(())
    }

    /// **A CUDA fold lands what the CPU fold lands**, bit for bit on these
    /// exactly representable rows: the store's one-launch fold over its slots.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_store_fold_is_the_cpu_ones() -> Result<()> {
        let _gpu = gpu_serial();
        // The fold kernel runs at d = 128.
        let d = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 4,
        };
        let one = vec![LayerKind::DeltaNet, LayerKind::DeltaNet];
        let run = |dev: &Device| -> Result<Vec<ExportedLayerState>> {
            let mut store = RecurrentStateStore::new(&one, &d, dev)?;
            store.begin_wave()?;
            for li in 0..2 {
                let (_, write) = store.layer_span_mut(li, 1)?;
                let SpanWrite::Timeline(tl) = write else {
                    candle::bail!("a one-row span writes a timeline");
                };
                let full = |shape: &[usize], v: f32| Tensor::full(v, shape, dev);
                TimelineLayout::of(&d).write_rows(
                    &tl,
                    &full(&[1, 2], 0.0)?,
                    &full(&[1, 1, 128], 0.5)?,
                    &full(&[1, 2, 128], 4.0 + li as f32)?,
                    &full(&[1, d.conv_dim()], 3.0)?,
                )?;
            }
            store.commit_wave();
            store.fold(None)?;
            store.export()
        };
        assert_eq!(run(&Device::new_cuda(0)?)?, run(&Device::Cpu)?);
        Ok(())
    }

    /// **A released readback buffer unpins exactly what the seal pinned.** The
    /// fixture's three recurrent layers each read back 256 bytes of state and
    /// 256 of conv tail, so one export pins a 1,536-byte owned staging buffer;
    /// releasing it empties the slot, and an owned buffer unpins on drop — what
    /// an engine's shutdown relies on to leave no pinned memory behind.
    ///
    /// Read from the slot, not from the process's pinned-staging total: every
    /// test that stages moves that total, and they run in parallel.
    #[cfg(feature = "cuda")]
    #[test]
    fn releasing_the_host_buffers_unpins_the_readback() -> Result<()> {
        let _gpu = gpu_serial();
        let held = || {
            READBACK
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map(|b| (matches!(b, PinnedBuf::Owned { .. }), b.len()))
        };
        let store = filled_store_on(&Device::new_cuda(0)?);
        release_recurrent_host_buffers();
        assert_eq!(held(), None);
        store.export()?;
        assert_eq!(held(), Some((true, 1_536)), "one owned, pinned buffer");
        release_recurrent_host_buffers();
        assert_eq!(held(), None, "the release dropped it");
        // The next seal allocates it afresh and reads the same bytes.
        assert_eq!(store.export()?, filled_store().export()?);
        release_recurrent_host_buffers();
        Ok(())
    }

    /// **A restore on CUDA uploads from a pooled pinned buffer** and lands the
    /// bytes the snapshot holds: a store exported, imported into a fresh store
    /// and exported again reads back the same bytes, through an owned, pinned
    /// buffer of the store's size. A restore whose predecessor's uploads have
    /// completed takes that buffer again; one whose predecessor's are still
    /// queued takes another rather than wait.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_restore_uploads_from_a_pooled_pinned_buffer() -> Result<()> {
        let _gpu = gpu_serial();
        let held = || {
            UPLOAD
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|u| (matches!(u.buf, PinnedBuf::Owned { .. }), u.buf.len()))
                .collect::<Vec<_>>()
        };
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!("a CUDA device")
        };
        let source = filled_store_on(&dev);
        let hash = source.schedule_hash();
        let snapshot = source.export()?;
        release_recurrent_host_buffers();
        let mut restored = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        restored.import(hash, &snapshot)?;
        assert_eq!(held(), [(true, 1_536)], "one owned, pinned buffer");
        assert_eq!(restored.export()?, snapshot);
        assert_eq!(restored.export()?, filled_store().export()?);
        // The export synchronised the stream, so the first restore's uploads
        // have completed and the second takes its buffer.
        let mut again = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        again.import(hash, &snapshot)?;
        assert_eq!(
            held(),
            [(true, 1_536)],
            "the completed buffer is taken again"
        );
        // Two restores back to back with nothing waited between them may each
        // take a buffer; none is waited on, and both land.
        let mut third = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        third.import(hash, &snapshot)?;
        let pooled = held();
        assert!(
            (1..=2).contains(&pooled.len()) && pooled.iter().all(|&b| b == (true, 1_536)),
            "{pooled:?}"
        );
        cuda.cuda_stream()
            .synchronize()
            .map_err(|e| Error::Msg(format!("{e:?}")))?;
        assert_eq!(again.export()?, snapshot);
        assert_eq!(third.export()?, snapshot);
        release_recurrent_host_buffers();
        assert!(held().is_empty(), "the release dropped them");
        Ok(())
    }

    /// The decode table's addresses are the live half's own — read off the
    /// slot, they must name exactly the buffers the tensors view — and a
    /// commit that exchanged a layer with its scratch moves that layer's, and
    /// no other's.
    ///
    /// The fixture's block puts the conv tail 256 bytes past the state and the
    /// timeline 512 past it.
    #[cfg(feature = "cuda")]
    #[test]
    fn decode_addresses_are_the_live_halfs_and_follow_an_exchange() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!("new_cuda answered a CUDA device")
        };
        let views = |store: &RecurrentStateStore| -> Result<Vec<[u64; 3]>> {
            store
                .slots
                .iter()
                .map(|s| {
                    Ok([
                        tensor_device_ptr(cuda, &s.live.state.conv_tail)?,
                        tensor_device_ptr(cuda, &s.live.state.s)?,
                        tensor_device_ptr(cuda, &s.live.timeline)?,
                    ])
                })
                .collect()
        };
        let mut store = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        let before: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(before.len(), 3, "one row per DeltaNet layer: 0, 1 and 3");
        assert_eq!(before, views(&store)?);
        for [tail, s, tl] in &before {
            assert_eq!((*tail, *tl), (s + 256, s + 512));
        }

        // A long span over layer 3 only — the third slot — committed.
        store.ensure_scratch()?;
        store.begin_wave()?;
        store.layer_span_mut(3, TIMELINE_ROWS + 1)?;
        store.commit_wave();
        let after: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(after, views(&store)?);
        assert_eq!(
            &after[..2],
            &before[..2],
            "layers that did not run keep their slots"
        );
        assert_ne!(after[2], before[2], "layer 3 exchanged with its scratch");
        Ok(())
    }
}
