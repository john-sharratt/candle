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

        #[test]
        fn test_float_arena_kv_access() {
            let backing = create_test_backing();

            backing.ensure_for_offset(0, 0, 32).unwrap();

            let k_arenas = backing.k_arenas();
            let v_arenas = backing.v_arenas();

            assert_eq!(k_arenas.len(), 1);
            assert_eq!(v_arenas.len(), 1);

            // Palette-split arenas store one head, one side, one palette band.
            // With head_dim=32 and N_PALETTE=4 this becomes (8192, 32, 8).
            assert_eq!(k_arenas[0].dims(), &[8192, 32, 8]);
            assert_eq!(v_arenas[0].dims(), &[8192, 32, 8]);
        }
    }

    // ==================== Chunk Allocation Tests ====================

    mod chunk_allocation_tests {
        use super::*;

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

            // Should recycle the lowest freed K head-0 GID.
            // With palette-split blocks, K0 strides by 32: 0, 32, 64, 96, ...
            // Freed seqs 1 and 3 therefore return K0 gids 32 and 96.
            assert_eq!(row[0], 32);
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
            // Freed seqs 1 (K0=32) and 3 (K0=96). Lowest freed K head-0 GID is 32.
            assert_eq!(
                row[0], 32,
                "recycled slot should be GID 32 (lowest freed K head-0)"
            );
        }

        /// GIDs encode arena_idx and chunk_idx. Verify the encoding is correct
        /// for the first two arenas.
        #[test]
        fn test_gid_encodes_arena_and_chunk_indices_correctly() {
            let arena_chunks =
                crate::arena_chunks_for_format(crate::kv_cache::KvFormat::Float(DType::BF16));
            let gid_stride = crate::arena_gid_stride();
            // Each block uses GIDS_PER_HEAD*n_kv_head = 8*4 = 32 GIDs.
            // To spill into arena 1 we need more than one arena worth of K-head slots.
            let seqs_per_arena = arena_chunks / 32;
            let n_seqs = seqs_per_arena + 1;
            let backing =
                ChunkedKvBacking::new(n_seqs + 1, 4, 32, DType::BF16, &Device::Cpu, 32).unwrap();

            for i in 0..n_seqs {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // The last sequence's K head-0 GID should be in arena 1
            let last_gid = k_gid_snapshot(&backing)[seqs_per_arena][0];

            assert_eq!(
                last_gid as usize, gid_stride,
                "GID {} should be the base of arena 1 = {}",
                last_gid, gid_stride
            );
            assert_eq!(last_gid as usize / gid_stride, 1, "arena_idx should be 1");
            assert_eq!(last_gid as usize % gid_stride, 0, "chunk_idx should be 0");
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
            assert_eq!(row[0], 0, "first recycled K head-0 GID should be 0");
            assert_eq!(row[1], 32, "second recycled K head-0 GID should be 32");
        }

        /// migrate_chunk must return a GID from the pool (not a raw mint), and
        /// the returned GID must be distinct from the source.
        #[test]
        fn test_migrate_chunk_gid_comes_from_pool_not_stolen() {
            use crate::kv_cache::chunked::ArenaKey;
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            let src_gid = k_gid_snapshot(&backing)[0][0];

            // Migrate to CPU (same dtype, different location — triggers alloc_chunk_for_key)
            let target = ArenaKey::cpu_float(DType::BF16);
            let new_gid = backing.migrate_chunk(src_gid, target).unwrap();

            // The new GID must be distinct from the source
            assert_ne!(new_gid.raw(), src_gid, "migrate must yield a new GID");
            assert!(new_gid.raw() >= 0, "new GID must be non-negative");
        }

        /// After migrate_chunk (same format copy), the GID comes from the pool
        /// and is placed in the same arena format (no new arena needed if one exists).
        #[test]
        fn test_migrate_chunk_same_format_stays_in_same_arena_pool() {
            use crate::kv_cache::chunked::ArenaKey;
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
            // Migrate to same format (cpu-float BF16 → cpu-float BF16 copy)
            let target = ArenaKey::cpu_float(DType::BF16);
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
            use crate::kv_cache::chunked::ArenaKey;
            let backing = create_test_backing();

            // Allocate 3 sequences
            for i in 0..3 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            let target = ArenaKey::cpu_float(DType::BF16);

            let src_gids: Vec<i64> = {
                let snap = k_gid_snapshot(&backing);
                (0..3).map(|i| snap[i][0]).collect()
            };
            // Keep the returned ChunkGids alive so RAII doesn't immediately return them
            let held_gids: Vec<_> = src_gids
                .iter()
                .map(|src| backing.migrate_chunk(*src, target.clone()).unwrap())
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
            assert_eq!(block_count(&backing, 0), 1);
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
            backing.reserve_glue_gap_chunk(0, 3).unwrap(); // logical [0,3)
            backing.reserve_glue_gap_chunk(0, 7).unwrap(); // logical [3,10)
            backing.reserve_glue_gap_chunk(0, 2).unwrap(); // logical [10,12)
            assert_eq!(rope_base_of(&backing, 0, 0), 0, "gap0 base");
            assert_eq!(rope_base_of(&backing, 0, 1), 3, "gap1 base = Σ(3)");
            assert_eq!(rope_base_of(&backing, 0, 2), 10, "gap2 base = Σ(3,7)");
            assert_eq!(rope_base_of(&backing, 0, 3), 12, "end = Σ(3,7,2)");
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
}
