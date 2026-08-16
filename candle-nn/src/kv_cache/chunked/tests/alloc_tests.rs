//! Tests for allocation methods in ChunkedKvBacking.

use candle::{DType, Device, Tensor};

use crate::kv_cache::chunked::ChunkedKvBacking;

/// Test helper: snapshot K-side GIDs as `[batch][block]` with -1 for unallocated.
fn k_gid_snapshot(backing: &ChunkedKvBacking) -> Vec<Vec<i64>> {
    let state = backing.state.read().expect("lock");
    let mb = state.max_blocks;
    state
        .sequences
        .iter()
        .map(|slot| {
            let mut row = vec![-1i64; mb];
            if let Some(s) = slot {
                for (i, cw) in s.chunks_slice().iter().enumerate() {
                    row[i] = cw.gids.k_gid(0).raw();
                }
            }
            row
        })
        .collect()
}

/// The refusal→hint→creation round trip, on a real device.
///
/// A sealing pass turned away mid-wave is only useful if the refusal *says what
/// it wanted*: without that, the next pass rediscovers the same need at the same
/// depth, after redoing the work that led there, and is refused again if a wave
/// happens to be running. With it, the gap between forwards creates the arena
/// and the next pass only ever fills it — which is legal at any point in a wave.
#[cfg(all(test, feature = "cuda"))]
mod deferred_arena_tests {
    use crate::kv_cache::arena_table::ArenaLocation;
    use crate::kv_cache::chunked::arena::ArenaKey;
    use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
    use crate::kv_cache::chunked::size_class::SizeClass;
    use crate::kv_cache::chunked::{begin_wave, ChunkedKvBacking, KV_ARENA_MID_WAVE};
    use crate::kv_cache::LayerPhase;
    use candle::{DType, Device, Result};

    #[test]
    fn a_refused_arena_is_remembered_and_made_in_the_gap() -> Result<()> {
        let _serial = gpu_serial();
        let Ok(device @ Device::Cuda(_)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &device, 256)?;
        let Device::Cuda(cd) = &device else {
            unreachable!()
        };
        let stream = cd.cuda_stream();

        // A class the backing has no arena for yet, so serving it needs a
        // creation rather than a free slot.
        let key = ArenaKey::new(SizeClass::at(SizeClass::COUNT - 1), ArenaLocation::Gpu);

        // Inside a wave, that creation is refused — and the refusal is marked
        // retryable so the sealing pass knows to come back.
        let guard = begin_wave(&stream, LayerPhase::Attention)?;
        let err = backing
            .alloc_chunk_for_key(key)
            .err()
            .expect("a class with no arena cannot be served mid-wave")
            .to_string();
        assert!(
            err.contains(KV_ARENA_MID_WAVE),
            "the refusal must be marked retryable: {err}"
        );
        assert_eq!(
            backing.create_deferred_arenas()?,
            0,
            "the demand must not be satisfied while the wave is still running"
        );
        drop(guard);

        // In the gap it is created — from the record, with nothing having to
        // rediscover the need.
        assert_eq!(
            backing.create_deferred_arenas()?,
            1,
            "the refusal recorded the class it wanted; the gap must act on it"
        );
        assert_eq!(
            backing.create_deferred_arenas()?,
            0,
            "the record is drained, not replayed — a second gap must not \
             create a second arena for the same class"
        );

        // And the allocation the wave refused now succeeds against the arena
        // that already exists, which is the point: filling is always legal.
        backing
            .alloc_chunk_for_key(key)
            .expect("the class has an arena now, so no creation is needed");
        Ok(())
    }

    /// **A deferral must never be laundered into scarcity.**
    ///
    /// `alloc_chunk_run_for_key` answers a failed stamp by promoting to a wider
    /// class and retrying four times, then reports `VRAM exhaustion on arena
    /// creation`. That is the right answer when no region can be had and the
    /// wrong one when a region can be had *next wave* — and the wrong one is
    /// what shipped: the `1088 B` class froze at 49 arenas with 75 regions
    /// claimable, the hot→warm drain stalled at 634 MiB, and the log blamed VRAM
    /// while a fifth of the reservation stood free.
    #[test]
    fn a_run_refused_mid_wave_reports_the_deferral_not_exhaustion() -> Result<()> {
        let _serial = gpu_serial();
        let Ok(device @ Device::Cuda(_)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &device, 256)?;
        let Device::Cuda(cd) = &device else {
            unreachable!()
        };
        let stream = cd.cuda_stream();
        let key = ArenaKey::new(SizeClass::at(SizeClass::COUNT - 2), ArenaLocation::Gpu);

        let guard = begin_wave(&stream, LayerPhase::Attention)?;
        let err = backing
            .alloc_chunk_run_for_key(key, 4)
            .err()
            .expect("a run needing a new arena cannot be served mid-wave")
            .to_string();
        drop(guard);

        assert!(
            err.contains(KV_ARENA_MID_WAVE),
            "the run allocator must hand the deferral back as itself: {err}"
        );
        assert!(
            !err.contains("VRAM exhaustion"),
            "a deferral reported as exhaustion sends the next investigation \
             looking for a KV leak that is not there: {err}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_backing() -> ChunkedKvBacking {
        ChunkedKvBacking::new(
            4,  // initial_batch
            4,  // n_kv_head
            32, // head_dim
            DType::BF16,
            &Device::Cpu,
            256,
        )
        .unwrap()
    }

    // ==================== ensure_max_blocks Tests ====================

    mod ensure_max_blocks_tests {
        use super::*;

        #[test]
        fn test_ensure_max_blocks_no_growth_needed() {
            let backing = create_test_backing();

            // Initial max_blocks should be at least 1
            let initial_dims = {
                let s = k_gid_snapshot(&backing);
                vec![s.len(), s[0].len()]
            };

            // Ensure 1 block (no growth needed)
            backing.ensure_max_blocks(1).ok();

            let new_dims = {
                let s = k_gid_snapshot(&backing);
                vec![s.len(), s[0].len()]
            };
            assert_eq!(initial_dims[1], new_dims[1]);
        }

        #[test]
        fn test_ensure_max_blocks_growth() {
            let backing = ChunkedKvBacking::new(
                4,
                4,
                32,
                DType::BF16,
                &Device::Cpu,
                64, // Small initial: ceil(64/32) = 2 blocks
            )
            .unwrap();

            let initial_max = k_gid_snapshot(&backing)[0].len();

            // Force growth to 10 blocks
            backing.ensure_max_blocks(10).ok();

            let new_max = k_gid_snapshot(&backing)[0].len();
            assert!(new_max >= 10);
            assert!(new_max > initial_max);
        }

        #[test]
        fn test_ensure_max_blocks_preserves_data() {
            let backing = create_test_backing();

            // First allocate some chunks
            backing.ensure_for_offset(0, 0, 32).unwrap();

            let chunk_id_before = k_gid_snapshot(&backing)[0][0];

            // Grow max_blocks
            backing.ensure_max_blocks(100).ok();

            // Verify existing allocation preserved
            let chunk_id_after = k_gid_snapshot(&backing)[0][0];

            assert_eq!(chunk_id_before, chunk_id_after);
        }
    }

    // ==================== Arena Creation Tests ====================

    mod arena_creation_tests {
        use super::*;

        #[test]
        fn test_arena_created_on_allocation() {
            let backing = create_test_backing();

            assert_eq!(backing.arena_count().unwrap(), 0);

            // Allocate a chunk - should create an arena
            backing.ensure_for_offset(0, 0, 32).unwrap();

            // Should have at least one arena now
            assert!(backing.arena_count().unwrap() >= 1);
        }

        #[test]
        fn test_arena_growth_with_more_chunks() {
            let backing = ChunkedKvBacking::new(4, 4, 32, DType::BF16, &Device::Cpu, 256).unwrap();

            // Allocate enough chunks to need multiple arenas
            for i in 0..4 {
                backing.ensure_for_offset(i, 0, 32).unwrap();
            }

            let arena_count_after_4 = backing.arena_count().unwrap();

            // Allocate more - should trigger another arena
            for i in 0..4 {
                backing.ensure_for_offset(i, 32, 32).unwrap();
            }

            let arena_count_after_8 = backing.arena_count().unwrap();

            // Should have more arenas now
            assert!(arena_count_after_8 >= arena_count_after_4);
        }

        /// Writing a sequence materialises arenas, and each band's slot is a
        /// run of the class's bytes. There is no "float arena" to ask for any
        /// more — an arena holds whatever fits its stride.
        #[test]
        fn writing_a_sequence_materialises_class_arenas() {
            let backing = create_test_backing();

            backing.ensure_for_offset(0, 0, 32).unwrap();

            let (count, strides) = backing
                .with_arenas(|arenas| {
                    let mut strides: Vec<usize> =
                        arenas.values().map(|a| a.slot_stride()).collect();
                    strides.sort_unstable();
                    strides.dedup();
                    (arenas.len(), strides)
                })
                .unwrap();

            assert!(count > 0, "a written sequence must have arenas behind it");
            for stride in strides {
                assert!(
                    crate::kv_cache::chunked::LADDER.contains(&stride),
                    "every arena's stride must be a rung of the ladder, got {stride}"
                );
            }
        }

        #[test]
        fn test_sequential_chunk_allocation() {
            let backing = create_test_backing();

            // First allocation should get chunk 0
            backing.ensure_for_offset(0, 0, 32).unwrap();
            let chunk0 = k_gid_snapshot(&backing)[0][0];
            assert_eq!(chunk0, 0);

            // Second allocation: each block uses GIDS_PER_HEAD*n_kv_head = 8*4 = 32 GIDs,
            // so seq 1's K head 0 gets GID 32 (the next block's first K slot).
            backing.ensure_for_offset(1, 0, 32).unwrap();
            let chunk1 = k_gid_snapshot(&backing)[1][0];
            assert_eq!(chunk1, 32);
        }

        #[test]
        fn test_multi_block_allocation() {
            let backing = create_test_backing();

            // Allocate 3 blocks worth (CHUNK_SIZE=32, so 96 tokens = 3 blocks)
            backing.ensure_for_offset(0, 0, 96).unwrap();

            let row = k_gid_snapshot(&backing)[0].clone();

            // First 3 blocks should be allocated
            assert!(row[0] >= 0);
            assert!(row[1] >= 0);
            assert!(row[2] >= 0);
        }

        #[test]
        fn test_incremental_allocation() {
            let backing = create_test_backing();

            // First allocate 1 block
            backing.ensure_for_offset(0, 0, 32).unwrap();
            let row = k_gid_snapshot(&backing)[0].clone();
            let first_chunk = row[0];
            assert!(first_chunk >= 0);

            // Now extend to need 2 blocks
            backing.ensure_for_offset(0, 32, 32).unwrap();
            let row = k_gid_snapshot(&backing)[0].clone();

            // First chunk should be unchanged
            assert_eq!(row[0], first_chunk);
            // Second chunk should now be allocated
            assert!(row[1] >= 0);
        }

        #[test]
        fn test_no_double_allocation() {
            let backing = create_test_backing();

            // Allocate 2 blocks
            backing.ensure_for_offset(0, 0, 64).unwrap();
            let row = k_gid_snapshot(&backing)[0].clone();
            let (c0, c1) = (row[0], row[1]);

            // Re-request the same range
            backing.ensure_for_offset(0, 0, 64).unwrap();
            let row = k_gid_snapshot(&backing)[0].clone();

            // Should still have the same chunks
            assert_eq!(row[0], c0);
            assert_eq!(row[1], c1);
        }
    }

    // ==================== Free List Tests ====================

    mod free_list_tests {
        use super::*;
        use crate::kv_cache::chunked::ChunkedKvBacking;

        fn backing_with_sequence() -> ChunkedKvBacking {
            let backing = create_test_backing();
            // Allocate a sequence and write data to allocate chunks
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
            backing
        }

        #[test]
        fn test_free_returns_chunks_to_pool() {
            let backing = backing_with_sequence();

            // Get the chunk IDs
            let row = k_gid_snapshot(&backing)[0].clone();
            let c0 = row[0];
            assert!(c0 >= 0);

            // Free the sequence
            backing.free_sequence(0).unwrap();

            // Allocate a new sequence and write data
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(1, 0, &k, &v).unwrap();

            let new_row = k_gid_snapshot(&backing)[1].clone();

            // The new sequence should have gotten the freed chunk
            // (free list prefers lower IDs)
            assert!(new_row[0] >= 0);
        }

        #[test]
        fn test_prefer_lower_chunk_ids() {
            let backing = ChunkedKvBacking::new(8, 4, 32, DType::BF16, &Device::Cpu, 256).unwrap();

            // Allocate several sequences with data
            for i in 0..4 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // Sequences 0-3 have chunks 0-3
            // Free sequence 1 and 3 (chunks 1 and 3 go to free list)
            backing.free_sequence(1).unwrap();
            backing.free_sequence(3).unwrap();

            // Allocate new sequence - should prefer chunk 1 (lowest)
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(4, 0, &k, &v).unwrap();

            let row = k_gid_snapshot(&backing)[4].clone();

            // Writer-block band slots come from CONTIGUOUS high-water runs, never
            // the recycle stack: the run allocator prefers a fresh high-water run
            // (a locality optimization for the select/QREL walk — correctness is
            // per-band, see `resolve_band_source`). The freed slots from seqs 1
            // and 3 remain for singleton allocs; this new block's bands take the
            // next fresh run at/above the prior high-water mark and stay
            // consecutive.
            assert!(
                row[0] >= 128,
                "run-allocated K head-0 GID must come from the fresh high-water \
                 tail (>= 128), got {}",
                row[0]
            );
        }
    }

    // ==================== Quantized Allocation Tests (conditional) ====================

    // Note: Quantized tests require the cuda feature for GPU quantization kernels
    // They are skipped in CPU-only builds

    // ==================== Pool-Driven Allocation Integration Tests ====================

    /// Tests that verify the GID pool is the single source of truth for chunk
    /// allocation (no backdoors like mint()).  All paths go through
    /// alloc_chunk_for_key → ChunkGidPool::allocate_for / register_arena.
    mod pool_integration_tests {
        use super::*;

        /// After allocating N chunks they must have contiguous GIDs 0..N-1
        /// (min-heap packs into lowest arenas first).
        #[test]
        fn test_pool_allocates_contiguous_gids_from_zero() {
            let backing = create_test_backing();

            // Allocate 4 sequences each needing 1 block (32 tokens)
            for i in 0..4 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            let snap = k_gid_snapshot(&backing);
            // Each block uses GIDS_PER_HEAD*n_kv_head = 8*4 = 32 GIDs.
            for i in 0..4usize {
                let row = snap[i].clone();
                assert_eq!(
                    row[0],
                    (i * 32) as i64,
                    "sequence {i} should hold GID {}",
                    i * 32
                );
            }
        }

        /// Freeing a sequence returns its GIDs to the pool; the very next
        /// allocation picks up the lowest freed GID.
        #[test]
        fn test_free_sequence_recycles_gids_via_pool() {
            // Need batch capacity > 4 to allocate 5th sequence (index 4)
            let backing = ChunkedKvBacking::new(8, 4, 32, DType::BF16, &Device::Cpu, 256).unwrap();

            for i in 0..4 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // Free seq 1 (GID 1) and seq 3 (GID 3)
            backing.free_sequence(1).unwrap();
            backing.free_sequence(3).unwrap();

            // New allocation should get GID 1 (the lowest returned GID)
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(4, 0, &k, &v).unwrap();

            let row = k_gid_snapshot(&backing)[4].clone();
            // The 4 initial seqs took contiguous slots 0..128 (high-water mark).
            // Freeing seqs 1 and 3 pushes their slots onto the recycle stack —
            // available to singleton allocs — but writer-block band slots come
            // from CONTIGUOUS high-water runs (the select/QREL layout contract),
            // so the new block's gids land at/above the prior high-water mark.
            assert!(
                row[0] >= 128,
                "run-allocated slot must come from the fresh high-water tail \
                 (>= 128), got {}",
                row[0]
            );
        }

        // ── Scarcity-only class promotion ────────────────────────────────
        //
        // A chunk may occupy a slot of the next class up when its own class
        // has no free slot AND no region can be had for it. Strictly
        // scarcity-gated: under any region availability a class gets its own
        // (`docs/archived/arena_unification.md` §3.4).

        /// **The gate.** Under normal conditions — regions freely available —
        /// nothing promotes, however many chunks are claimed. If this ever
        /// fails, promotion has become a background mixing vector and the
        /// per-class occupancy numbers stop meaning anything.
        #[test]
        fn ordinary_allocation_never_promotes() {
            use crate::kv_cache::chunked::class_promotion_count;

            let before = class_promotion_count();
            let backing = create_test_backing();
            for i in 0..3 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }
            assert_eq!(
                class_promotion_count(),
                before,
                "a well-fed allocator must never widen a chunk's class"
            );
        }

        /// A promoted chunk is still addressed correctly, because reads take
        /// their extent from the band's *format* bytes and never from the
        /// slot stride. The wider slot changes only the pad.
        #[test]
        fn a_wider_slot_still_round_trips_its_band() {
            use crate::kv_cache::chunked::size_class::SizeClass;

            // Write a BF16 band's worth of bytes into a slot two rungs above
            // the one it would normally take, and read exactly its payload
            // back out.
            let class = SizeClass::at(6);
            let bytes = class.chunks_per_region() * class.bytes();
            let data = Tensor::zeros(bytes, DType::U8, &Device::Cpu).unwrap();
            let mut arena = crate::kv_cache::chunked::Arena::new(
                data,
                class,
                crate::kv_cache::arena_table::ArenaLocation::Cpu,
                0,
            );

            let payload: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();
            let src = Tensor::from_slice(&payload, payload.len(), &Device::Cpu).unwrap();
            arena.write_slot_bytes(3, &src).unwrap();

            let back = arena
                .slot_bytes(3, payload.len())
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            assert_eq!(back, payload, "the band reads back exactly");

            // And the pad past it is untouched — it belongs to no chunk.
            let pad = arena
                .slot_bytes(3, class.bytes())
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            assert!(
                pad[payload.len()..].iter().all(|&b| b == 0),
                "the pad past the payload must stay zero"
            );
        }

        /// Promotion walks **one rung at a time** and gives up at the top
        /// rather than wrapping. The top class failing is the honest answer:
        /// there is nowhere wider to go.
        #[test]
        fn promotion_walks_the_ladder_and_stops() {
            use crate::kv_cache::chunked::size_class::SizeClass;

            let mut class = SizeClass::at(0);
            let mut widths = vec![class.bytes()];
            while let Some(next) = class.promote() {
                assert!(
                    next.bytes() > class.bytes(),
                    "a promotion must always widen"
                );
                class = next;
                widths.push(class.bytes());
            }
            assert_eq!(widths.len(), SizeClass::COUNT);
            assert!(
                class.promote().is_none(),
                "the top class has nowhere to promote to"
            );
        }

        /// GIDs encode arena_idx and chunk_idx. Verify the encoding is correct
        /// for the first two arenas.
        #[test]
        fn test_gid_encodes_arena_and_chunk_indices_correctly() {
            // Capacity is the SIZE CLASS's slot count, not a per-format one.
            // This backing's BF16 band is 32 tokens x 8 dims x 2 B = 512 B,
            // which lands on the 640 B rung.
            let class = crate::kv_cache::chunked::class_for_format(
                crate::kv_cache::KvFormat::Float(DType::BF16),
                crate::CHUNK_SIZE * (32 / crate::kv_cache::arena_table::N_PALETTE),
            )
            .expect("BF16 is covered by the ladder");
            let arena_chunks = class.chunks_per_region();
            let gid_stride = crate::GID_STRIDE;
            // Each block uses GIDS_PER_HEAD*n_kv_head = 8*4 = 32 GIDs.
            // To spill into arena 1 we need more than one arena worth of K-head slots.
            // Enough sequences that the pool must register a second arena:
            // each block claims GIDS_PER_HEAD * n_kv_head = 32 slots, and a run
            // never straddles an arena, so round UP before adding one.
            let seqs_per_arena = arena_chunks.div_ceil(32);
            let n_seqs = seqs_per_arena + 1;
            let backing =
                ChunkedKvBacking::new(n_seqs + 1, 4, 32, DType::BF16, &Device::Cpu, 32).unwrap();

            for i in 0..n_seqs {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // The last sequence's K head-0 GID must have spilled into arena 1,
            // and must decode as `(raw / GID_STRIDE, raw % GID_STRIDE)` — the
            // shift/mask split the CUDA side performs.
            let last_gid = k_gid_snapshot(&backing)[seqs_per_arena][0] as usize;

            assert_eq!(
                last_gid / gid_stride,
                1,
                "GID {last_gid} should have spilled into arena 1                  (arena 0 holds {arena_chunks} slots)"
            );
            assert!(
                last_gid % gid_stride < arena_chunks,
                "the chunk index must be inside the arena's capacity"
            );
        }

        /// Allocate multiple blocks per sequence. Verify all GIDs are unique
        /// (no double-allocation via pool path).
        #[test]
        fn test_multi_block_sequences_gids_unique() {
            let backing = create_test_backing();

            // 3 sequences × 3 blocks = 9 GIDs (3×32 = 96 tokens each)
            for i in 0..3 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            let snap = k_gid_snapshot(&backing);
            let mut seen = std::collections::HashSet::new();
            for i in 0..3usize {
                let row = snap[i].clone();
                for &gid in row.iter().take(3) {
                    assert!(gid >= 0, "unallocated GID in row {i}");
                    assert!(seen.insert(gid), "duplicate GID {gid} across sequences");
                }
            }
        }

        /// Free a multi-block sequence; the freed GIDs must be re-used bottom-up.
        #[test]
        fn test_free_multi_block_sequence_recycles_all_gids() {
            let backing = create_test_backing();

            // Allocate two sequences of 2 blocks each (64 tokens)
            for i in 0..2 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // Seq 0 holds K0 GIDs {0,32} for blocks 0-1; seq 1 holds K0 GIDs {64,96}
            backing.free_sequence(0).unwrap();

            // New allocation gets the freed GIDs back (lowest first)
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(2, 0, &k, &v).unwrap();

            let row = k_gid_snapshot(&backing)[2].clone();
            // Seq 0 took slots [0,64) (two blocks), seq 1 took [64,128). Freeing
            // seq 0 pushes [0,64) onto the recycle stack — but writer-block band
            // slots come from CONTIGUOUS high-water runs (the select/QREL layout
            // contract; see `alloc_chunk_run_for_key`), never recycled
            // singletons. Seq 2's blocks therefore take fresh runs at/above the
            // prior high-water mark; the freed slots stay available for
            // singleton allocs.
            assert!(
                row[0] >= 128 && row[1] >= 128,
                "run-allocated blocks must come from the fresh high-water tail \
                 (>= 128), got {row:?}"
            );
        }

        /// migrate_chunk must return a GID from the pool (not a raw mint), and
        /// the returned GID must be distinct from the source.
        /// The CPU key a gid's slot relocates into: same size class, warm tier.
        fn cpu_key_of(backing: &ChunkedKvBacking, raw: i64) -> crate::kv_cache::chunked::ArenaKey {
            use crate::kv_cache::arena_table::ArenaLocation;
            use crate::kv_cache::chunked::{ArenaKey, GID_STRIDE};
            let arena_idx = (raw as usize) / GID_STRIDE;
            let key = backing
                .with_arenas(|a| a.get(&arena_idx).map(|a| a.arena_key()))
                .unwrap()
                .expect("source arena exists");
            ArenaKey::new(key.class, ArenaLocation::Cpu)
        }

        #[test]
        fn test_migrate_chunk_gid_comes_from_pool_not_stolen() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            let src_gid = k_gid_snapshot(&backing)[0][0];

            // Relocate to CPU: same class, different location — the shape
            // every production migrate has (a slot move never converts).
            let target = cpu_key_of(&backing, src_gid);
            let new_gid = backing.migrate_chunk(src_gid, target).unwrap();

            // The new GID must be distinct from the source
            assert_ne!(new_gid.raw(), src_gid, "migrate must yield a new GID");
            assert!(new_gid.raw() >= 0, "new GID must be non-negative");
        }

        /// After migrate_chunk (same format copy), the GID comes from the pool
        /// and is placed in the same arena format (no new arena needed if one exists).
        #[test]
        fn test_migrate_chunk_same_format_stays_in_same_arena_pool() {
            let backing = ChunkedKvBacking::new(4, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            // Allocate 2 sequences
            for i in 0..2 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            let before = backing.arena_count().unwrap();

            let src = k_gid_snapshot(&backing)[0][0];
            // Relocate within the same class — the copy path.
            let target = cpu_key_of(&backing, src);
            let new_gid = backing.migrate_chunk(src, target).unwrap();

            let after = backing.arena_count().unwrap();
            // Same format: no new arena needed (existing arena has spare capacity)
            assert_eq!(
                after, before,
                "same-format migrate should not create a new arena"
            );
            // But the new GID must differ from the source
            assert_ne!(new_gid.raw(), src);
        }

        /// Migrating N chunks to the same-format target reuses one arena pool --
        /// all N GIDs come from pool allocations, none are double-allocated.
        #[test]
        fn test_migrate_multiple_chunks_unique_gids() {
            let backing = create_test_backing();

            // Allocate 3 sequences
            for i in 0..3 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            let src_gids: Vec<i64> = {
                let snap = k_gid_snapshot(&backing);
                (0..3).map(|i| snap[i][0]).collect()
            };
            // Keep the returned ChunkGids alive so RAII doesn't immediately return them
            let held_gids: Vec<_> = src_gids
                .iter()
                .map(|src| {
                    let target = cpu_key_of(&backing, *src);
                    backing.migrate_chunk(*src, target).unwrap()
                })
                .collect();
            let new_gids: Vec<i64> = held_gids.iter().map(|g| g.raw()).collect();

            // All new GIDs must be unique (pool handed out distinct slots)
            let unique: std::collections::HashSet<i64> = new_gids.iter().copied().collect();
            assert_eq!(
                unique.len(),
                3,
                "pool must yield 3 distinct GIDs for 3 migrations"
            );

            // None of the new GIDs should equal any source GID
            for &ng in &new_gids {
                for &sg in &src_gids {
                    assert_ne!(
                        ng, sg,
                        "migrated GID {} must not equal source GID {}",
                        ng, sg
                    );
                }
            }
        }
    }

    // ==================== reserve_glue_gap_chunk Tests ====================
    //
    // The in-place glue gap is the load-bearing primitive of the interleaved-
    // glue design: each gap is a real chunk with a real `usage` at its logical
    // position, so the cumulative-usage `rope_base` of every later chunk equals
    // its true sequence position — the single convention decode + glue read via
    // `slice_rope`. These tests pin that convention at the lowest layer.
    mod glue_gap_tests {
        use super::*;
        use crate::kv_cache::chunked::CHUNK_SIZE;

        /// Sum of `usage` over chunks `[0, blk)` — exactly the `rope_base` the
        /// decode kernel derives for block `blk` (`types.rs` `rebuild_decode`).
        fn rope_base_of(backing: &ChunkedKvBacking, batch_idx: usize, blk: usize) -> u32 {
            let state = backing.state.read().expect("lock");
            let seq = state.sequences[batch_idx].as_ref().expect("slot");
            seq.chunks_slice()[..blk].iter().map(|c| c.usage).sum()
        }

        fn read_chunk(backing: &ChunkedKvBacking, batch_idx: usize, blk: usize) -> (u32, u16) {
            let state = backing.state.read().expect("lock");
            let seq = state.sequences[batch_idx].as_ref().expect("slot");
            let c = &seq.chunks_slice()[blk];
            (c.usage, c.offset)
        }

        fn writer_start(backing: &ChunkedKvBacking, batch_idx: usize) -> usize {
            let state = backing.state.read().expect("lock");
            state.sequences[batch_idx]
                .as_ref()
                .expect("slot")
                .writer_start_idx()
        }

        fn block_count(backing: &ChunkedKvBacking, batch_idx: usize) -> usize {
            let state = backing.state.read().expect("lock");
            state.sequences[batch_idx]
                .as_ref()
                .expect("slot")
                .block_count()
        }

        #[test]
        fn reserves_full_chunk_with_tail_window() {
            // Full-by-construction: a 5-token gap is allocated `usage=5`,
            // `offset = CHUNK_SIZE - 5 = 27`, so `offset + usage == CHUNK_SIZE`.
            // The chunk reads as FULL to every writer-region scan (so no prefill
            // can extend into it), while `usage` stays 5 for the cumulative
            // `rope_base` convention. Valid window is the tail `[27, 32)`.
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let (gap, in_blk_base) = backing.reserve_glue_gap_chunk(0, 5).unwrap();
            assert_eq!(gap, 0, "first gap is block 0");
            assert_eq!(
                in_blk_base,
                CHUNK_SIZE as u32 - 5,
                "scatter base == offset == 27 (tail window [27, 32))"
            );
            assert_eq!(
                read_chunk(&backing, 0, 0),
                (5, CHUNK_SIZE as u16 - 5),
                "usage=5, offset=27 (full: offset+usage==CHUNK_SIZE)"
            );
            // Gap + the empty writer chunk pushed after it (the immutable gap
            // must never be the slot's writable tail — see
            // `reserve_glue_gap_chunk`).
            assert_eq!(block_count(&backing, 0), 2);
        }

        #[test]
        fn gap_sits_below_writer_boundary() {
            // The gap must NOT be the active writer: `writer_start` is advanced
            // past it so decode/prefill never auto-select it (only the glue
            // forward fills it, by explicit target).
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let (gap, _) = backing.reserve_glue_gap_chunk(0, 4).unwrap();
            assert_eq!(
                writer_start(&backing, 0),
                gap + 1,
                "writer_start advanced past the gap"
            );
        }

        #[test]
        fn cumulative_usage_equals_logical_position() {
            // THE convention: reserve three gaps of different sizes; each later
            // chunk's rope_base (Σ preceding usage) is its logical start.
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            // Each reserve pushes [gap, empty writer]: gaps land at even block
            // indices, zero-usage writer chunks between them (they don't move
            // the cumulative rope base).
            backing.reserve_glue_gap_chunk(0, 3).unwrap(); // logical [0,3) at blk 0
            backing.reserve_glue_gap_chunk(0, 7).unwrap(); // logical [3,10) at blk 2
            backing.reserve_glue_gap_chunk(0, 2).unwrap(); // logical [10,12) at blk 4
            assert_eq!(rope_base_of(&backing, 0, 0), 0, "gap0 base");
            assert_eq!(rope_base_of(&backing, 0, 2), 3, "gap1 base = Σ(3)");
            assert_eq!(rope_base_of(&backing, 0, 4), 10, "gap2 base = Σ(3,7)");
            assert_eq!(rope_base_of(&backing, 0, 5), 12, "end = Σ(3,7,2)");
        }

        #[test]
        fn gaps_get_unique_gids() {
            // Each gap is writer-owned with its own GIDs — never Arc-shared, so
            // the glue's explicit write can't corrupt another holder and truncate
            // frees them by refcount.
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            backing.reserve_glue_gap_chunk(0, 4).unwrap();
            backing.reserve_glue_gap_chunk(0, 4).unwrap();
            let snap = k_gid_snapshot(&backing);
            assert_ne!(snap[0][0], snap[0][1], "the two gaps hold distinct GIDs");
            assert_ne!(snap[0][0], -1, "gap0 allocated");
            assert_ne!(snap[0][1], -1, "gap1 allocated");
        }

        #[test]
        fn rejects_zero_and_oversize() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            assert!(
                backing.reserve_glue_gap_chunk(0, 0).is_err(),
                "0-token gap rejected"
            );
            assert!(
                backing
                    .reserve_glue_gap_chunk(0, CHUNK_SIZE as u32 + 1)
                    .is_err(),
                "gap larger than CHUNK_SIZE rejected (one island = one chunk)"
            );
            assert!(
                backing.reserve_glue_gap_chunk(0, CHUNK_SIZE as u32).is_ok(),
                "exactly CHUNK_SIZE is allowed"
            );
        }

        #[test]
        fn out_of_range_batch_idx_errors() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            assert!(backing.reserve_glue_gap_chunk(999, 4).is_err());
        }
    }

    // ==================== Cross-Layer Capacity Tests ====================

    /// `ensure_for_batch_entries_all` speaks for every layer, so a layer whose
    /// block structure has drifted from layer 0's must still be asked.
    mod layer_capacity_tests {
        use super::*;
        use crate::kv_cache::chunked::CHUNK_SIZE;

        /// One layer's backing holding a single sequence of exactly 32 tokens —
        /// a tail chunk that is precisely full, with nothing writable behind it.
        fn layer_with_full_tail() -> ChunkedKvBacking {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
            // `write_contiguous` places bytes; the window's valid token count is
            // the writer's to advance, exactly as the prefill path does.
            backing.set_len(0, 32);
            backing
        }

        #[test]
        fn every_layer_is_asked_not_only_layer_zero() {
            // The skew a windowed creep prefill leaves behind: the resumed layer
            // already carries an empty writer chunk for the next window, the
            // layers still pending resume carry the exactly-full tail. Asking
            // layer 0 alone reads "writable tail, nothing to allocate" and
            // suppresses the allocation every other layer needs.
            let layers = vec![layer_with_full_tail(), layer_with_full_tail()];
            layers[0].push_empty_writer_chunk(0).unwrap();

            let entries = [(0usize, 32usize)];
            assert!(
                layers[0].validate_decode_batch_state(&entries).is_ok(),
                "precondition: the skewed layer already has a writable tail"
            );
            assert!(
                layers[1].validate_decode_batch_state(&entries).is_err(),
                "precondition: the lagging layer's tail is exactly full"
            );

            ChunkedKvBacking::ensure_for_batch_entries_all(&layers, &entries, 1).unwrap();

            for (li, backing) in layers.iter().enumerate() {
                backing
                    .validate_decode_batch_state(&entries)
                    .unwrap_or_else(|e| panic!("layer {li} was left without a writable tail: {e}"));
            }
        }

        /// One layer's backing holding a single sequence of `tokens` tokens.
        fn layer_with(tokens: usize) -> ChunkedKvBacking {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
            backing.set_len(0, tokens);
            backing
        }

        /// `(block count, index of the chunk the write slot resolves to,
        /// total tokens)` — the part of a layer's structure the shared decode
        /// position map is built from, read the way the metadata builder reads
        /// it.
        fn observed_layout(backing: &ChunkedKvBacking) -> (usize, usize, usize) {
            let state = backing.state.read().expect("lock");
            let slot = state.sequences[0].as_ref().expect("slot");
            let chunks = slot.chunks_slice();
            let start = slot.writer_start_idx().min(chunks.len().saturating_sub(1));
            let writer = (start..chunks.len())
                .find(|&i| (chunks[i].offset as usize + chunks[i].usage as usize) < CHUNK_SIZE)
                .unwrap_or(chunks.len().saturating_sub(1));
            let tokens = chunks.iter().map(|c| c.usage as usize).sum();
            (chunks.len(), writer, tokens)
        }

        #[test]
        fn diverged_writer_index_is_reconciled() {
            // The hazard the reconciliation exists for: both layers have a
            // WRITABLE tail, so neither needs an allocation, but they resolve
            // the write slot to different chunks. The kernel scatters through
            // the per-layer slice while attention reads through the one shared
            // position map, so this divergence corrupts silently — there is no
            // fault and no wrong-looking number to notice.
            let layers = vec![layer_with(20), layer_with(20)];
            layers[0].push_empty_writer_chunk(0).unwrap();

            assert_eq!(observed_layout(&layers[0]), (2, 1, 20));
            assert_eq!(
                observed_layout(&layers[1]),
                (1, 0, 20),
                "precondition: the layers disagree on the write slot"
            );

            let entries = [(0usize, 20usize)];
            ChunkedKvBacking::ensure_for_batch_entries_all(&layers, &entries, 1).unwrap();

            let unified = observed_layout(&layers[0]);
            assert_eq!(
                observed_layout(&layers[1]),
                unified,
                "every layer must resolve the write slot to the same chunk"
            );
            assert_eq!(unified.2, 20, "reconciliation must not shift any position");
            for (li, backing) in layers.iter().enumerate() {
                backing
                    .validate_decode_batch_state(&entries)
                    .unwrap_or_else(|e| panic!("layer {li} was left without a writable tail: {e}"));
            }
        }

        #[test]
        fn diverged_token_windows_are_an_error_not_a_repair() {
            // Appending a common writer chunk equalises trailing structure, and
            // the tail heal may drop up to one chunk's worth of undelivered
            // tokens. A spread wider than a chunk is more than any single
            // failed operation leaves behind, so no repair may claim it.
            let layers = vec![layer_with(20), layer_with(2 * CHUNK_SIZE)];
            let entries = [(0usize, 20usize)];

            let err = ChunkedKvBacking::ensure_for_batch_entries_all(&layers, &entries, 1)
                .expect_err("a spread past one chunk cannot share one position map");
            let msg = err.to_string();
            assert!(
                msg.contains("could not be reconciled or healed"),
                "the error must name both repairs it outlived, got: {msg}"
            );
        }

        /// **A failed wave's skew heals instead of bricking the sequence.**
        ///
        /// The decode sweep advances each layer's usage as that layer completes,
        /// so a wave that dies mid-sweep leaves the early layers one token ahead
        /// — undelivered tokens, since the wave never retired. The forward path
        /// now rolls that back at the failure; this covers the same state
        /// arriving from before the rollback existed, or through a substrate
        /// reload of a sealed turn that captured it. The next decode truncates
        /// every layer to the shortest, reconciles the trailing structure, and
        /// proceeds — the alternative was refusing this sequence forever.
        #[test]
        fn a_failed_waves_tail_skew_is_healed_not_refused() {
            // Layer 0 one token ahead of layer 1: the exact signature of a wave
            // that died between layer 0's advance and layer 1's.
            let layers = vec![layer_with(33), layer_with(32)];
            let entries = [(0usize, 32usize)];

            ChunkedKvBacking::ensure_for_batch_entries_all(&layers, &entries, 1)
                .expect("a one-token tail skew is healable and must not refuse the decode");

            // Both layers agree on every data window afterwards, and both are
            // writable — the healed sequence decodes like any other.
            let layouts: Vec<_> = layers.iter().map(observed_layout).collect();
            assert_eq!(
                layouts[0], layouts[1],
                "healing must leave the layers identical, got {layouts:?}"
            );
            assert_eq!(
                layouts[0].2, 32,
                "the undelivered token is dropped, the delivered ones stay"
            );
            for (li, backing) in layers.iter().enumerate() {
                backing
                    .validate_decode_batch_state(&entries)
                    .unwrap_or_else(|e| panic!("layer {li} not decodable after heal: {e}"));
            }
        }

        #[test]
        fn uniform_layers_still_allocate_once_each() {
            // The common case must be unchanged: every layer's tail is full, so
            // every layer takes the allocation path and ends up writable.
            let layers = vec![layer_with_full_tail(), layer_with_full_tail()];
            let entries = [(0usize, 32usize)];

            ChunkedKvBacking::ensure_for_batch_entries_all(&layers, &entries, 1).unwrap();

            let counts: Vec<usize> = layers
                .iter()
                .map(|b| b.sequence_block_count(0).expect("slot"))
                .collect();
            assert_eq!(counts, vec![2, 2], "one fresh writer chunk per layer");
            for (li, backing) in layers.iter().enumerate() {
                backing
                    .validate_decode_batch_state(&entries)
                    .unwrap_or_else(|e| panic!("layer {li} was left without a writable tail: {e}"));
            }
        }
    }
}
