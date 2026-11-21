//! Tests that the layer cache puts every layer's bytes in the right slot, including
//! under concurrent cold reads.
//!
//! A sweep runs two forwards over a zone of a chosen size and checks byte identity
//! at every layer, covering both a zone that streams most of the model and one that
//! holds all of it. A dedicated test runs a cold-only pass without synchronising,
//! because the copy stream must stay deep for the bug it exercises: concurrent cold
//! reads sharing one staging buffer overwrite each other's bytes. Every layer's
//! payload is non-zero, so a slot that was never written is distinguishable.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    // ── The stack, end to end, on the real device ────────────────────────
    //
    // Synthetic layers at a few hundred KB each rather than a checkpoint's
    // 240 MB, so the whole 64-layer sweep costs megabytes and seconds. What it
    // exercises is everything except the model loader: images, the pack's
    // write/read/checksum path, the warm tier's fill and promotion, slot
    // uploads on the copy stream, the per-layer fences, and the residency
    // policy driving all of it.
    //
    // The assertion is byte identity: after the wave reaches each layer, the
    // slot holds *that* layer's payload and not its predecessor's. A fence
    // that is joined too late, a slot handed to two layers, or an eviction of
    // something still in flight all show up here as the wrong bytes.

    use super::super::super::pack::{header_for, LayerPack, PackWriter};

    use crate::models::layer_stream::descriptor::{
        layer_image, FfnForm, LayerTensor, MixKind, Projection,
    };

    use candle::cuda_backend::cudarc::driver::CudaSlice;

    use candle::quantized::GgmlDType;

    use std::fs::File;

    use std::io::{BufWriter, Write};

    use std::path::PathBuf;

    pub(in super::super) const TEST_LAYERS: usize = 64;

    pub(in super::super) const TEST_PINNED: usize = 2;

    /// Small enough to be quick, large enough to cross a sector.
    pub(in super::super) const PROJ_BYTES: usize = 8192;

    pub(in super::super) fn test_image(kind: MixKind) -> LayerImage {
        let p = |role, rows, cols| Projection {
            role,
            shape: [rows, cols],
            // The KO twin, not the source dtype — a slot holds repacked bytes.
            dtype: GgmlDType::Q4_KO,
            payload: PROJ_BYTES,
            extent: PROJ_BYTES,
        };
        let mut projs: Vec<Projection> = match kind {
            MixKind::DeltaNet => vec![
                p(LayerTensor::Wqkv, 10240, 5120),
                p(LayerTensor::Wz, 6144, 5120),
                p(LayerTensor::WOut, 5120, 6144),
            ],
            MixKind::Attention => vec![
                p(LayerTensor::Wq, 12288, 5120),
                p(LayerTensor::Wk, 1024, 5120),
                p(LayerTensor::Wv, 1024, 5120),
                p(LayerTensor::Wo, 5120, 6144),
            ],
        };
        projs.extend([
            p(LayerTensor::FfnGateUp, 34816, 5120),
            p(LayerTensor::FfnDown, 5120, 17408),
        ]);
        layer_image(kind, FfnForm::Fused, &projs).unwrap()
    }

    /// The lineage's 3:1 interleave: three DeltaNet layers then an attention
    /// one, so both kinds and both image widths are in the same run.
    pub(in super::super) fn test_images() -> Vec<LayerImage> {
        (0..TEST_LAYERS)
            .map(|i| {
                test_image(if i % 4 == 3 {
                    MixKind::Attention
                } else {
                    MixKind::DeltaNet
                })
            })
            .collect()
    }

    /// A payload unique to `(layer, projection)`, so a slot holding the wrong
    /// layer is visible in its first byte.
    ///
    /// Never zero: a freshly allocated slot reads as zeros, so a zero-valued
    /// tag would make "this layer's bytes arrived" and "nothing was ever
    /// written here" indistinguishable — which is exactly the failure this test
    /// exists to catch.
    pub(in super::super) fn payload(layer: usize, idx: usize, len: usize) -> Vec<u8> {
        let tag = 1 + ((layer * 7 + idx) % 254) as u8;
        vec![tag; len]
    }

    /// A layer section holding every fixture layer, the pinned head included,
    /// written into a temp file at a sector offset as a model pack holds it.
    /// Returns the file to remove afterwards, and the open section.
    pub(in super::super) fn test_pack(tag: &str, images: &[LayerImage]) -> (PathBuf, LayerPack) {
        let path = std::env::temp_dir().join(format!(
            "candle_layer_{tag}_{}.pack.gguf",
            std::process::id()
        ));
        let mut out = BufWriter::new(File::create(&path).unwrap());
        out.write_all(&[0u8; 4096]).unwrap();
        let header = header_for(
            images,
            &|_, _| GgmlDType::Q4_K,
            Int8Mode::Performance as u32,
            &|_, _| 0xFEED,
        );
        let mut w = PackWriter::new(&mut out, header).unwrap();
        for (li, img) in images.iter().enumerate() {
            let bufs: Vec<Vec<u8>> = img
                .placements
                .iter()
                .enumerate()
                .map(|(i, p)| payload(li, i, p.bytes))
                .collect();
            let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
            w.write_layer(li, &refs).unwrap();
        }
        w.finish().unwrap();
        out.flush().unwrap();
        drop(out);
        let pack = LayerPack::open_section(&path, 4096, |_, _| 0xFEED).unwrap();
        (path, pack)
    }

    #[test]
    #[ignore = "needs a CUDA device; allocates a few hundred MB of VRAM and writes a \
                temporary pack. Run with: cargo test -p candle-transformers \
                --features cuda --lib layer_stream::cache::tests::a_wave_streams_every_layer \
                -- --ignored --nocapture"]
    fn a_wave_streams_every_layer_through_a_small_ring() {
        // Both regimes §7 says must be one code path: a zone far smaller than
        // the model, which streams most of it, and a zone that holds it, where
        // "it fits" is the degenerate case and nothing ever moves after the
        // first pass.
        let images = test_images();
        let whole: usize = images.iter().map(|i| i.total).sum();
        let tight = run_sweep(crate::models::layer_stream::slot_bytes_for_layers(&images) * 10);
        let roomy = run_sweep(whole);
        let (Some(tight), Some(roomy)) = (tight, roomy) else {
            return; // no device; run_sweep already said so
        };

        assert_eq!(tight.abandoned, 0, "an opportunistic load failed");
        assert!(
            tight.evictions > 0,
            "a ring of 10 over {TEST_LAYERS} layers must evict"
        );
        assert!(
            tight.warm_hits > 0,
            "the warm tier was never promoted from — its membership is unused"
        );

        // The model fits: after the first pass every layer is resident, so the
        // second pass is pure hits and moves no bytes at all.
        assert_eq!(
            roomy.evictions, 0,
            "a ring holding the whole model must never evict"
        );
        assert_eq!(
            roomy.warm_hits + roomy.cold_reads,
            TEST_LAYERS - TEST_PINNED,
            "a layer was loaded more than once despite fitting"
        );
        assert!(
            roomy.hits > tight.hits,
            "residency bought nothing: roomy {} vs tight {}",
            roomy.hits,
            tight.hits
        );
    }

    /// **Several cold reads in one plan must not overwrite each other's staging.**
    ///
    /// A cold read fills a pinned host buffer and then uploads out of it
    /// *asynchronously*, so the buffer is still under DMA when `issue` returns.
    /// With a single staging buffer the next cold read in the same plan refills
    /// it mid-flight and the earlier layer lands holding this one's bytes.
    ///
    /// The sweep above cannot see that: it synchronizes and verifies after every
    /// layer, which drains the copy stream between reads and serialises exactly
    /// the overlap the bug needs. So this one warms **nothing** — every read is
    /// cold — runs a whole pass with no intermediate synchronize, and checks the
    /// slots once at the end.
    ///
    /// It is a real bug this reproduces, not a hypothetical: the 27B ran clean
    /// for as long as its warm tier happened to cover every streamed layer, and
    /// began emitting garbage logits within three configs of the tier being
    /// sized honestly against host RAM.
    #[test]
    #[ignore = "needs a CUDA device; allocates a few hundred MB of VRAM and writes a \
                temporary pack. Run with: cargo test -p candle-transformers \
                --features cuda --lib \
                layer_stream::cache::tests::concurrent_cold_reads_keep_their_own_bytes \
                -- --ignored --nocapture"]
    fn concurrent_cold_reads_keep_their_own_bytes() {
        let Some(bad) = run_cold_only(false) else {
            return; // no device
        };
        assert_eq!(
            bad, 0,
            "{bad} slots held another layer's bytes after a pass of cold reads — \
             the staging ring is being reused before its upload has landed"
        );
    }

    /// **A recording wave streams layers in order.** The same cold-only pass,
    /// inside a wave capture, with each layer's slot copied out by a recorded
    /// device-to-device copy between `ensure` and `prefetch` — the place a
    /// layer's GEMMs sit. A join left on the capture stream, or an eviction
    /// ordered only behind the compute stream's eager work, lets a recorded
    /// read see another layer's bytes.
    #[test]
    #[ignore = "needs a CUDA device; allocates a few hundred MB of VRAM and writes a \
                temporary pack. Run with: cargo test -p candle-transformers \
                --features cuda --lib \
                layer_stream::cache::tests::a_recording_wave_reads_every_layer_it_streams \
                -- --ignored --nocapture"]
    fn a_recording_wave_reads_every_layer_it_streams() {
        let Some(bad) = run_cold_only(true) else {
            return; // no device
        };
        assert_eq!(
            bad, 0,
            "{bad} layers were read back holding another layer's bytes inside a \
             recording wave"
        );
    }

    /// One pass with an empty warm tier and no per-layer readback; answers with
    /// the number of resident slots holding the wrong layer's bytes. `recorded`
    /// runs the pass inside a wave capture and also counts every layer whose
    /// recorded copy-out read the wrong bytes.
    pub(in super::super) fn run_cold_only(recorded: bool) -> Option<usize> {
        use candle::Device;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return None;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda yields a cuda device")
        };

        let images = test_images();
        let slot_bytes = crate::models::layer_stream::slot_bytes_for_layers(&images);
        let tag = if recorded { "cold_recorded" } else { "cold" };
        let (path, pack) = test_pack(tag, &images);

        // One contiguous arena, as the span is: the plan lays layers down inside
        // it at their own sizes, so the addresses under test are the ones
        // production computes rather than a hand-built table.
        let (arena, base, plan) = arena_for(cuda, &images, slot_bytes * 12);

        // **Warm budget zero**, so every promotion is a pack read and the
        // staging ring is the only path bytes take.
        let mut cache = LayerCache::new(
            cuda,
            images.clone(),
            Int8Mode::Performance,
            pack,
            &plan,
            TEST_PINNED,
            0,
            |view, _layer| Ok(view),
        )
        .unwrap();

        // Allocated before the wave opens, so the pass's only eager device
        // calls are the cache's own.
        let mut copies: Vec<CudaSlice<u8>> = (0..TEST_LAYERS)
            .map(|li| unsafe { cuda.alloc::<u8>(images[li].total).unwrap() })
            .collect();
        let wave = recorded.then(|| {
            let wave = cuda.begin_wave_capture().unwrap();
            cuda.record_launches().unwrap();
            wave
        });
        // One pass, no synchronize inside it: the copy stream stays deep, which
        // is the condition the bug needs.
        for li in 0..TEST_LAYERS {
            cache.ensure(li).unwrap();
            if recorded && li >= TEST_PINNED {
                let off = (cache.slot_base_of(li).unwrap() - base) as usize;
                cuda.cuda_stream()
                    .memcpy_dtod(&arena.slice(off..off + images[li].total), &mut copies[li])
                    .unwrap();
            }
            cache.prefetch().unwrap();
        }
        if let Some(wave) = wave {
            wave.finish().unwrap();
        }
        cuda.cuda_stream().synchronize().unwrap();
        let mut wrong = 0usize;
        if recorded {
            for layer in TEST_PINNED..TEST_LAYERS {
                let got = cuda.cuda_stream().memcpy_dtov(&copies[layer]).unwrap();
                for (i, p) in images[layer].placements.iter().enumerate() {
                    if got[p.offset..p.offset + p.bytes] != payload(layer, i, p.bytes)[..] {
                        eprintln!("  L{layer} projection {i} was read holding foreign bytes");
                        wrong += 1;
                        break;
                    }
                }
            }
        }

        let s = cache.stats();
        eprintln!(
            "[layer-stream] cold-only pass: warm {} cold {} evictions {}",
            s.warm_hits, s.cold_reads, s.evictions
        );
        assert!(
            s.cold_reads > STAGING_SLOTS,
            "the pass must issue more cold reads than the ring has buffers, or \
             a stale buffer can never be reused"
        );

        // Every slot still holding a layer must hold *that* layer's bytes — the
        // pinned head included, which the cache filled from the pack at load.
        for layer in 0..TEST_LAYERS {
            if !cache.residency().residence(layer).is_readable() {
                continue;
            }
            let got = read_layer(cuda, &arena, base, &cache, &images, layer);
            for (i, p) in images[layer].placements.iter().enumerate() {
                let want = payload(layer, i, p.bytes);
                if got[p.offset..p.offset + p.bytes] != want[..] {
                    eprintln!("  L{layer} projection {i} holds foreign bytes");
                    wrong += 1;
                    break;
                }
            }
        }
        drop(cache);
        std::fs::remove_file(&path).ok();
        Some(wrong)
    }

    /// A contiguous device arena and the [`ZonePlan`] laid out inside it.
    ///
    /// The plan's addresses are absolute, so the arena's own base is where `end`
    /// is taken from — exactly as the span's right edge serves in production.
    pub(in super::super) fn arena_for(
        cuda: &CudaDevice,
        images: &[LayerImage],
        budget: usize,
    ) -> (
        CudaSlice<u8>,
        u64,
        crate::models::layer_stream::zone::ZonePlan,
    ) {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let whole: usize = images.iter().map(|i| i.total).sum();
        let budget = budget.min(whole);
        let arena = unsafe { cuda.alloc::<u8>(budget).unwrap() };
        let stream = cuda.cuda_stream();
        // The guard is dropped before the arena moves out; the address stays
        // valid because the caller holds the allocation for the whole test.
        let base = {
            let (p, _g) = arena.device_ptr(&stream);
            p
        };
        let plan = crate::models::layer_stream::zone::plan_zone(
            images,
            TEST_PINNED,
            base + budget as u64,
            budget,
        )
        .expect("the fixture budget must clear the model's floor");
        (arena, base, plan)
    }

    /// A layer's bytes, read back from wherever the cache put it.
    pub(in super::super) fn read_layer<T, A: SlotAssembler<T>>(
        cuda: &CudaDevice,
        arena: &CudaSlice<u8>,
        base: u64,
        cache: &LayerCache<T, A>,
        images: &[LayerImage],
        layer: usize,
    ) -> Vec<u8> {
        let at = cache.slot_base_of(layer).unwrap();
        let off = (at - base) as usize;
        let n = images[layer].total;
        cuda.cuda_stream()
            .memcpy_dtov(&arena.slice(off..off + n))
            .unwrap()
    }

    /// Two forwards over a zone of `budget` bytes, checking byte identity at
    /// every layer. `None` when there is no device.
    pub(in super::super) fn run_sweep(budget: usize) -> Option<LayerCacheStats> {
        use candle::Device;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return None;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda yields a cuda device")
        };

        let images = test_images();

        // ── the cold tier ──
        let (path, pack) = test_pack(&format!("sweep_{budget}"), &images);

        // ── the hot tier ──
        let (arena, base, plan) = arena_for(cuda, &images, budget);

        // The identity assembler: this test is about bytes and residency, not
        // about any model's layer type, so a slot is presented as its own
        // views. That the cache can be exercised with no model at all is the
        // point of it being generic over the payload.
        let mut cache = LayerCache::new(
            cuda,
            images.clone(),
            // The fixture's images are KO twins, so the mode must be an int8 one
            // for `from_qtensor_view` to accept them.
            Int8Mode::Performance,
            pack,
            &plan,
            TEST_PINNED,
            20,
            |view, _layer| Ok(view),
        )
        .unwrap();

        // ── two full forwards, so the wrap is exercised as well as the sweep ──
        let mut checked = 0usize;
        for pass in 0..2 {
            for (li, img) in images.iter().enumerate().take(TEST_LAYERS) {
                cache.ensure(li).unwrap();
                cache.prefetch().unwrap();

                // Read the ground back and check it is this layer's bytes.
                cuda.cuda_stream().synchronize().unwrap();
                let got = read_layer(cuda, &arena, base, &cache, &images, li);
                for (i, p) in img.placements.iter().enumerate() {
                    let want = payload(li, i, p.bytes);
                    assert_eq!(
                        &got[p.offset..p.offset + p.bytes],
                        want.as_slice(),
                        "pass {pass} L{li} projection {i} ({:?}) holds the wrong layer's bytes",
                        p.role
                    );
                }
                checked += 1;
            }
        }

        let s = cache.stats();
        eprintln!(
            "[layer-stream] {} of {TEST_LAYERS} homed: {checked} layer reads over 2 forwards",
            cache.residency().homed()
        );
        eprintln!(
            "[layer-stream]   hits {} warm {} cold {} joins {} evictions {} abandoned {}",
            s.hits, s.warm_hits, s.cold_reads, s.fence_joins, s.evictions, s.abandoned
        );
        assert_eq!(
            checked,
            TEST_LAYERS * 2,
            "a layer was skipped rather than checked"
        );

        drop(cache);
        std::fs::remove_file(&path).ok();
        Some(s)
    }
}
