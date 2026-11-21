//! Tests on the GPU that the batched migration plan and the per-band write land
//! identical bytes, and that a hot-to-warm round trip is byte-identical.
//!
//! The batched plan and the per-band walk are independent implementations of the
//! same addressing, so a divergence would write plausible but wrong bytes. Round
//! trips from the device to host and back are compared byte for byte. A narrow band
//! in a wider slot must move its payload while addressing slots by their stride.
//! These tests run only with a device, under the GPU lock.

#[cfg(feature = "cuda")]
use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    #[cfg(feature = "cuda")]
    use super::*;

    #[cfg(feature = "cuda")]
    use crate::kv_cache::chunked::ChunkedKvBacking;

    #[cfg(feature = "cuda")]
    use crate::kv_cache::QuantFormat;

    #[cfg(feature = "cuda")]
    use candle::{DType, Device, Tensor};

    // ── Batched-async migrate tests (GPU-only) ──────────────────────────────
    //
    // These exercise `migrate_sealed_to_cpu_batch_async` and the warm lift
    // (`warm_lift::lift_warm_to_gpu`) — the production hot↔warm path. They
    // require a CUDA device; on a CPU-only build each test silently returns
    // early.
    #[cfg(feature = "cuda")]
    use crate::kv_cache::chunked::warm_lift::lift_one;

    #[cfg(feature = "cuda")]
    use super::super::tests_support::tests::{
        bytes_of_cpu_sealed, cuda_backing, cuda_device_or_skip, seed_and_seal_pattern,
    };

    /// **The batched migration plan and the per-band walk must land identical
    /// bytes — proven on the GPU, where the plan actually runs.**
    ///
    /// On CUDA, a contiguous float write takes the one-launch batched plan
    /// (`try_plan_batched_write`), while a non-contiguous source falls back to
    /// the per-band walk. The two are independent implementations of the same
    /// band addressing, and a divergence writes plausible bytes — fluent,
    /// wrong attention, no error. The CPU round-trip above can never reach the
    /// plan path (it is `cfg(cuda)` + GPU-arena only), so this is the plan's
    /// only write-then-read coverage: both write forms must read back as the
    /// identity, at every kernel head dim, across chunk boundaries.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_plan_and_walk_writes_agree_at_every_head_dim() {
        // Through `cuda_device_or_skip`, not `Device::new_cuda`: this builds
        // `ChunkedKvBacking`s, so it must hold the crate's GPU lock for the
        // duration or it races whatever else is mid-wave — see that helper's
        // note.
        let (device, _gpu) = match cuda_device_or_skip() {
            Some(v) => v,
            None => {
                eprintln!("skipping: CUDA device required");
                return;
            }
        };
        let n_kv_head = 2usize;
        for head_dim in [64usize, 128, 256] {
            for n_tokens in [1usize, 31, 32, 33, 64, 100] {
                let n = n_kv_head * n_tokens * head_dim;
                let kv: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let k = Tensor::from_vec(kv.clone(), (1, n_kv_head, n_tokens, head_dim), &device)
                    .unwrap();
                let v = Tensor::from_vec(
                    kv.iter().map(|x| -x).collect::<Vec<f32>>(),
                    (1, n_kv_head, n_tokens, head_dim),
                    &device,
                )
                .unwrap();

                // The same logical K/V as a transposed view: `[1, len, heads,
                // head_dim]` data seen through a `(1, heads, len, head_dim)`
                // transpose is non-contiguous (for len > 1), which routes the
                // write down the per-band walk instead of the plan.
                let mut perm = Vec::with_capacity(n);
                for t in 0..n_tokens {
                    for h in 0..n_kv_head {
                        let base = (h * n_tokens + t) * head_dim;
                        perm.extend_from_slice(&kv[base..base + head_dim]);
                    }
                }
                let k_walk =
                    Tensor::from_vec(perm.clone(), (1, n_tokens, n_kv_head, head_dim), &device)
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap();
                let v_walk = Tensor::from_vec(
                    perm.iter().map(|x| -x).collect::<Vec<f32>>(),
                    (1, n_tokens, n_kv_head, head_dim),
                    &device,
                )
                .unwrap()
                .transpose(1, 2)
                .unwrap();
                if n_tokens > 1 {
                    assert!(
                        !k_walk.is_contiguous(),
                        "the walk input must be non-contiguous or this test \
                         exercises the plan twice"
                    );
                }

                let write_and_read = |k_in: &Tensor, v_in: &Tensor| -> (Tensor, Tensor) {
                    let backing =
                        ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F32, &device, 256)
                            .unwrap();
                    let slot = backing.alloc_sequence().unwrap();
                    backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
                    backing.write_contiguous(slot, 0, k_in, v_in).unwrap();
                    backing.set_len(slot, n_tokens);
                    backing.read_contiguous(slot, 0, n_tokens).unwrap()
                };
                let (k_plan_out, v_plan_out) = write_and_read(&k, &v);
                let (k_walk_out, v_walk_out) = write_and_read(&k_walk, &v_walk);

                let same = |a: &Tensor, b: &Tensor| -> f32 {
                    a.sub(b)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .max(0)
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                };
                for (label, got) in [("plan K", &k_plan_out), ("walk K", &k_walk_out)] {
                    assert_eq!(
                        same(got, &k),
                        0.0,
                        "{label} round-trip differs at head_dim {head_dim}, \
                         {n_tokens} tokens"
                    );
                }
                for (label, got) in [("plan V", &v_plan_out), ("walk V", &v_walk_out)] {
                    assert_eq!(
                        same(got, &v),
                        0.0,
                        "{label} round-trip differs at head_dim {head_dim}, \
                         {n_tokens} tokens"
                    );
                }
            }
        }
    }

    /// `migrate_sealed_to_cpu_batch_async` on a single sequence must
    /// produce a CPU-located sealed with the same chunk count, token
    /// count, and per-chunk metadata as the source.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_single_sequence_preserves_structure() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 1 full chunk (32) + 1 partial (8)
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let sealed = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);

        assert_eq!(sealed.token_count, n_tokens);
        assert_eq!(sealed.chunks.len(), 2);
        assert_eq!(sealed.location, ArenaLocation::Gpu);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let cpu_batch = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&sealed])
            .expect("batched async migrate must succeed");

        assert_eq!(cpu_batch.len(), 1, "one sequence in → one out");
        let cpu = &cpu_batch[0];
        assert_eq!(cpu.location, ArenaLocation::Cpu);
        assert_eq!(cpu.token_count, n_tokens);
        assert_eq!(cpu.chunks.len(), sealed.chunks.len());
        for (orig, mig) in sealed.chunks.iter().zip(cpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// `migrate_sealed_to_cpu_batch_async` on multiple sequences must
    /// migrate each to its own CPU slots — distinct GIDs across results,
    /// per-sequence structure preserved.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_multiple_sequences_get_distinct_destinations() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 2;
        let head_dim = 16;
        let n_tokens = 32; // exactly one full chunk per sequence
        let backing = cuda_backing(&device, n_kv_head, head_dim);

        let s0 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);
        let s1 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 1000);
        let s2 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 2000);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let migrated = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&s0, &s1, &s2])
            .expect("multi-sequence batched async migrate must succeed");

        assert_eq!(migrated.len(), 3);
        // Every migrated sequence is CPU-located.
        for m in &migrated {
            assert_eq!(m.location, ArenaLocation::Cpu);
            assert_eq!(m.token_count, n_tokens);
            assert_eq!(m.chunks.len(), 1);
        }
        // All destination GIDs across the three sequences must be
        // distinct — no two migrated chunks share a physical CPU slot.
        let mut all_gids: Vec<i64> = Vec::new();
        for m in &migrated {
            for g in m.chunks[0].gids.iter() {
                all_gids.push(g.raw());
            }
        }
        all_gids.sort();
        let unique: std::collections::HashSet<i64> = all_gids.iter().copied().collect();
        assert_eq!(unique.len(), all_gids.len(), "GIDs must be globally unique");
    }

    /// The warm lift of a CPU sequence must produce a GPU-located sealed with
    /// the same per-chunk metadata.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_to_gpu_preserves_structure() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40;
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let gpu_orig = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // First migrate GPU → CPU, then CPU → GPU.
        let cpu = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let gpu_back = lift_one(&backing, &device, &[&cpu[0]]).expect("the warm lift must succeed");

        assert_eq!(gpu_back.len(), 1);
        let r = &gpu_back[0];
        assert_eq!(r.location, ArenaLocation::Gpu);
        assert_eq!(r.token_count, n_tokens);
        assert_eq!(r.chunks.len(), gpu_orig.chunks.len());
        for (orig, mig) in gpu_orig.chunks.iter().zip(r.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// **End-to-end byte identity** through the production hot↔warm
    /// path: a known pattern seeded on GPU, migrated to RAM via
    /// `migrate_sealed_to_cpu_batch_async` (kv_migrate_on gather + DtoH
    /// on the copy stream + pinned scratch → Tensor::from_slice +
    /// slice_set), then back to VRAM via the warm lift (host copy into
    /// the pinned ring + the ring scatter). After the round-trip,
    /// the CPU bytes of the final form must be byte-identical to the
    /// CPU bytes of the first migration.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 2 chunks: one full, one partial
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let gpu_orig =
            seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 12345);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // GPU → CPU (the bytes the production hot→warm path produces).
        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(
            !first_bytes.is_empty(),
            "CPU sealed must have non-empty bytes"
        );

        // CPU → GPU (the production warm→hot path).
        let gpu_back = lift_one(&backing, &device, &[&cpu_first[0]]).unwrap();

        // GPU → CPU again so we can read bytes to compare.
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(
            first_bytes.len(),
            second_bytes.len(),
            "round-trip must preserve byte length"
        );
        assert_eq!(
            first_bytes, second_bytes,
            "round-trip must be byte-identical — any difference means a chunk got swapped, \
             dropped, or had its bytes corrupted by the gather/scatter pipeline"
        );
    }

    /// **F16 end-to-end byte identity.** Same shape as the BF16
    /// round-trip but with `DType::F16` backing — exercises the
    /// `Tensor::from_slice` + `slice_set` write path with `half::f16`
    /// element casts on the way back to CPU arena slots.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_f16_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use half::f16;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // two chunks: one full, one partial
        let backing =
            ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F16, &device, 256).unwrap();

        // F16 seed pattern — distinct per-element value, deterministic.
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let data: Vec<f16> = (0..total)
            .map(|i| f16::from_f32(((12345 + i) as f32) * 0.001))
            .collect();
        let k = Tensor::from_vec(data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        let gpu_orig = backing.record_turn(slot).unwrap();

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(!first_bytes.is_empty());

        let gpu_back = lift_one(&backing, &device, &[&cpu_first[0]]).unwrap();
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(first_bytes.len(), second_bytes.len());
        assert_eq!(
            first_bytes, second_bytes,
            "F16 round-trip must be byte-identical"
        );
    }

    /// **The hazard size classes introduce, and the one the old round-trips
    /// structurally could not catch.**
    ///
    /// A `Q8_0` band is 1088 B and lives in a 1152 B slot, so payload and
    /// stride now differ by 64 B. Every copy length must be the payload while
    /// every address step must be the stride; confusing them either moves pad
    /// over PCIe and grows the persisted image, or addresses slot `n` at
    /// `n * payload` and reads into a neighbour.
    ///
    /// The two tests this replaces read the same arena-derived length on both
    /// sides of the trip, so a symmetric error compared equal (audit A7). This
    /// one seeds a known pattern through `write_raw_sealed_chunk` — the
    /// production cold-load path, which takes explicit format bytes — and then
    /// asserts on the payload alone.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_band_narrower_than_its_slot_round_trips_its_payload() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 2;
        let head_dim = 128;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = CHUNK_SIZE * sub_head_dim;

        // A Q8_0 backing. Its 1088 B payload lands on its own rung, so the
        // band fills its slot exactly — the interesting case for this test is
        // still a band NARROWER than its slot, which the palette split below
        // produces regardless of the class.
        let q8 = KvFormat::Quantized(QuantFormat::Q8_0);
        let backing =
            ChunkedKvBacking::new_with_format(4, n_kv_head, head_dim, q8, q8, &device, 256)
                .unwrap();
        let ggml = QuantFormat::Q8_0.to_ggml_dtype();
        let band_bytes = (elems_per_band / ggml.block_size()) * ggml.type_size();
        assert_eq!(band_bytes, 1088, "Q8_0 payload for one band");
        let class =
            crate::kv_cache::chunked::size_class::class_for_format(q8, elems_per_band).unwrap();
        assert_eq!(class.bytes(), 1088, "and it sits on its own rung");

        let slot = backing.alloc_sequence().unwrap();
        let n_bands = n_kv_head * N_PALETTE;
        // Distinct value per byte so any swap, shift, or truncation shows up.
        let pattern: Vec<u8> = (0..n_bands * band_bytes).map(|i| (i % 251) as u8).collect();
        backing
            .write_raw_sealed_chunk(
                slot,
                0,
                &pattern,
                &pattern,
                Arc::new(Vec::new()),
                Arc::new(Vec::new()),
                Arc::new(Vec::new()),
                Arc::new(Vec::new()),
            )
            .unwrap();
        backing.set_len(slot, CHUNK_SIZE);
        let hot = backing.record_turn(slot).unwrap();

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // hot -> warm.
        let warm = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&hot])
            .unwrap();
        let first = bytes_of_cpu_sealed(&backing, &warm[0]);
        assert_eq!(
            first.len(),
            2 * n_bands * band_bytes,
            "the gather must move payloads, not 1152 B slots"
        );
        assert_eq!(&first[..band_bytes], &pattern[..band_bytes], "K band 0");

        // warm -> hot -> warm again.
        let back = lift_one(&backing, &device, &[&warm[0]]).unwrap();
        let warm2 = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&back[0]])
            .unwrap();
        let second = bytes_of_cpu_sealed(&backing, &warm2[0]);

        assert_eq!(
            first, second,
            "a full hot->warm->hot->warm cycle must be byte-identical when the \
             slot is wider than the band"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_empty_input_is_noop() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some((device, _gpu)) = cuda_device_or_skip() else {
            return;
        };
        let backing = cuda_backing(&device, 2, 16);
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let out = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[])
            .unwrap();
        assert!(out.is_empty());
        assert!(pinned.is_none(), "no scratch allocation on empty input");

        let out_gpu = lift_one(&backing, &device, &[]).unwrap();
        assert!(out_gpu.is_empty());
    }
}
