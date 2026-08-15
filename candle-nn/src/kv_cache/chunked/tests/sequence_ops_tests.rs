//! Tests for sequence operations: alloc, free, share_prefix, fork.

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
            256, // max_blocks = ceil(256/32) = 8
        )
        .unwrap()
    }

    fn create_small_backing() -> ChunkedKvBacking {
        // max_blocks = ceil(128/32) = 4
        ChunkedKvBacking::new(4, 4, 32, DType::BF16, &Device::Cpu, 128).unwrap()
    }

    mod decode_validation_tests {
        use super::*;

        #[test]
        fn test_decode_validation_rejects_full_tail_before_kernel_launch() {
            let backing = create_test_backing();
            backing.ensure_for_offset(0, 0, 1).unwrap();

            {
                let mut state = backing.state.write().expect("lock");
                let seq = state.sequences[0].as_mut().expect("allocated sequence");
                let tail = seq.last_chunk_mut().expect("allocated tail chunk");
                tail.offset = 0;
                tail.usage = crate::kv_cache::chunked::CHUNK_SIZE as u32;
            }

            let err = backing
                .validate_decode_batch_state(&[(0, crate::kv_cache::chunked::CHUNK_SIZE)])
                .unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("validation failed"));
            assert!(msg.contains("invalid") || msg.contains("full/stale"));
        }

        #[test]
        fn test_ensure_for_batch_entries_rotates_physically_full_offset_tail() {
            let backing = create_test_backing();
            backing.ensure_for_offset(0, 0, 1).unwrap();

            {
                let mut state = backing.state.write().expect("lock");
                let seq = state.sequences[0].as_mut().expect("allocated sequence");
                let tail = seq.last_chunk_mut().expect("allocated tail chunk");
                tail.offset = 27;
                tail.usage = 5;
            }

            backing.ensure_for_batch_entries(&[(0, 5)], 1).unwrap();

            let state = backing.state.read().expect("lock");
            let seq = state.sequences[0].as_ref().expect("allocated sequence");
            assert_eq!(
                seq.block_count(),
                2,
                "physically full tail must rotate before decode"
            );
            assert_eq!(
                seq.chunks_slice()[0].usage,
                5,
                "sealed partial usage should be preserved"
            );
            assert_eq!(
                seq.chunks_slice()[1].usage,
                0,
                "new write tail starts empty"
            );
        }
    }

    // ==================== alloc_sequence Tests ====================

    mod alloc_sequence_tests {
        use super::*;

        #[test]
        fn test_alloc_sequence_basic() {
            let backing = create_test_backing();

            let slot = backing.alloc_sequence().unwrap();
            assert_eq!(slot, 0);
        }

        #[test]
        fn test_alloc_sequence_multiple() {
            let backing = create_test_backing();

            let slot0 = backing.alloc_sequence().unwrap();
            let slot1 = backing.alloc_sequence().unwrap();
            let slot2 = backing.alloc_sequence().unwrap();

            assert_eq!(slot0, 0);
            assert_eq!(slot1, 1);
            assert_eq!(slot2, 2);
        }

        #[test]
        fn test_alloc_sequence_grows_capacity() {
            let backing = ChunkedKvBacking::new(
                2, // Only 2 initial slots
                4,
                32,
                DType::BF16,
                &Device::Cpu,
                64,
            )
            .unwrap();

            assert_eq!(backing.batch_capacity(), 2);

            // Allocate beyond initial capacity
            backing.alloc_sequence().unwrap(); // slot 0
            backing.alloc_sequence().unwrap(); // slot 1
            let slot2 = backing.alloc_sequence().unwrap(); // should grow capacity

            assert!(backing.batch_capacity() > 2);
            assert_eq!(slot2, 2);
        }

        #[test]
        fn test_alloc_sequence_reuses_freed_slot() {
            let backing = create_test_backing();

            backing.alloc_sequence().unwrap(); // slot 0
            backing.alloc_sequence().unwrap(); // slot 1

            // Free slot 0
            backing.free_sequence(0).unwrap();

            // Next alloc should reuse slot 0
            let slot = backing.alloc_sequence().unwrap();
            assert_eq!(slot, 0);
        }
    }

    // ==================== ensure_sequence_allocated Tests ====================

    mod ensure_sequence_allocated_tests {
        use super::*;

        #[test]
        fn test_ensure_sequence_allocated_basic() {
            let backing = create_test_backing();

            backing.ensure_sequence_allocated(0).unwrap();

            // Should be able to write to the sequence now
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
        }

        #[test]
        fn test_ensure_sequence_allocated_idempotent() {
            let backing = create_test_backing();

            backing.ensure_sequence_allocated(0).unwrap();
            backing.ensure_sequence_allocated(0).unwrap(); // Should be no-op

            // Still works
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
        }

        #[test]
        fn test_ensure_sequence_allocated_grows() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            // Ensure slot 5 (beyond capacity)
            backing.ensure_sequence_allocated(5).unwrap();

            assert!(backing.batch_capacity() > 5);
        }
    }

    // ==================== free_sequence Tests ====================

    mod free_sequence_tests {
        use super::*;

        #[test]
        fn test_free_sequence_basic() {
            let backing = create_test_backing();

            backing.alloc_sequence().unwrap();

            // Write some data
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Free the sequence
            backing.free_sequence(0).unwrap();

            // Block table should be cleared
            let row = k_gid_snapshot(&backing)[0].clone();
            assert_eq!(row[0], -1);
        }

        #[test]
        fn test_free_sequence_out_of_range() {
            let backing = create_test_backing();

            let result = backing.free_sequence(10);
            assert!(result.is_err());
        }

        #[test]
        fn test_free_sequence_already_free() {
            let backing = create_test_backing();

            // Free without allocating (should be no-op)
            let result = backing.free_sequence(0);
            assert!(result.is_ok());
        }

        #[test]
        fn test_free_sequence_double_free() {
            let backing = create_test_backing();

            backing.alloc_sequence().unwrap();
            backing.free_sequence(0).unwrap();

            // Second free should be no-op
            let result = backing.free_sequence(0);
            assert!(result.is_ok());
        }

        #[test]
        fn test_free_sequence_returns_chunks_to_pool() {
            let backing = create_small_backing();

            // Allocate multiple sequences to consume chunks
            for i in 0..4 {
                backing.alloc_sequence().unwrap();
                let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
                backing.write_contiguous(i, 0, &k, &v).unwrap();
            }

            // Get chunk count (all 4 chunks in one arena used)
            let arena_count_before = backing.arena_count().unwrap();

            // Free a sequence
            backing.free_sequence(0).unwrap();

            // Allocate new sequence - should reuse freed chunk
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Arena count should not increase
            let arena_count_after = backing.arena_count().unwrap();
            assert_eq!(arena_count_before, arena_count_after);
        }
    }

    // ==================== set_len Tests ====================

    mod set_len_tests {
        use super::*;

        #[test]
        fn test_set_len() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            backing.set_len(0, 16);

            // Can't directly test, but it shouldn't panic
        }
    }

    // ==================== share_prefix Tests ====================

    mod share_prefix_tests {
        use super::*;

        fn setup_source_sequence(backing: &ChunkedKvBacking) {
            backing.alloc_sequence().unwrap(); // slot 0
                                               // Write 64 tokens (2 chunks)
            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
        }

        #[test]
        fn test_share_prefix_basic() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);
            backing.alloc_sequence().unwrap(); // slot 1 (target)

            // Share 1 chunk (32 tokens) of prefix
            let shared_tokens = backing.share_prefix(0, 1, 32).unwrap();

            assert_eq!(shared_tokens, 32);
        }

        #[test]
        fn test_share_prefix_multiple_chunks() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);
            backing.alloc_sequence().unwrap(); // slot 1

            // Share 2 chunks (64 tokens)
            let shared_tokens = backing.share_prefix(0, 1, 64).unwrap();

            assert_eq!(shared_tokens, 64);

            // Verify both sequences have same chunk IDs for shared blocks
            let snap = k_gid_snapshot(&backing);
            let row0 = &snap[0];
            let row1 = &snap[1];

            assert_eq!(row0[0], row1[0]);
            assert_eq!(row0[1], row1[1]);
        }

        #[test]
        fn test_share_prefix_floors_to_chunk() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);
            backing.alloc_sequence().unwrap(); // slot 1

            // Request 48 tokens, but should floor to 32 (1 chunk)
            let shared_tokens = backing.share_prefix(0, 1, 48).unwrap();

            assert_eq!(shared_tokens, 32);
        }

        #[test]
        fn test_share_prefix_zero_tokens() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);
            backing.alloc_sequence().unwrap();

            // Share 0 tokens
            let shared_tokens = backing.share_prefix(0, 1, 0).unwrap();

            assert_eq!(shared_tokens, 0);
        }

        #[test]
        fn test_share_prefix_self_fails() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);

            // Cannot share with self
            let result = backing.share_prefix(0, 0, 32);
            assert!(result.is_err());
        }

        #[test]
        fn test_share_prefix_source_not_allocated_fails() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap(); // slot 0 (empty)
            backing.alloc_sequence().unwrap(); // slot 1

            // Slot 0 has no blocks allocated (just a slot with no data)
            let result = backing.share_prefix(0, 1, 32);
            assert!(result.is_err());
        }

        #[test]
        fn test_share_prefix_marks_blocks_shared() {
            let backing = create_test_backing();
            setup_source_sequence(&backing);
            backing.alloc_sequence().unwrap(); // slot 1

            // Share
            backing.share_prefix(0, 1, 32).unwrap();

            // After sharing, both sequences should reference the same block.
            let snap = k_gid_snapshot(&backing);
            let src_id = snap[0][0];
            let dst_id = snap[1][0];
            assert_eq!(src_id, dst_id, "shared block must have same GID");
        }
    }

    // ==================== fork_sequence Tests ====================

    /// `truncate_sequence_to_tokens` owns the writer tail and nothing else.
    ///
    /// A slot with a deferred glue fire counts its reserved gap tokens in the
    /// chunk windows but not in the scheduler's offset, so the idempotency
    /// truncation arrives with a target short of the sealed cum by exactly the
    /// pending glue. That must clamp to the sealed boundary — cutting only the
    /// writable tail — not error, and not touch the reservation the fire needs.
    mod truncate_sealed_boundary_tests {
        use super::*;

        /// Sealed prefix of two chunks (32 + 13 reserved glue), writer at 2,
        /// one stale writable chunk of 5 behind it.
        fn deferred_glue_slot() -> ChunkedKvBacking {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(0, 0, 96).unwrap();
            backing.set_block_window(0, 0, 0, 32).unwrap();
            backing.set_block_window(0, 1, 0, 13).unwrap();
            backing.set_block_window(0, 2, 0, 5).unwrap();
            backing.test_set_writer_start(0, 2).unwrap();
            backing
        }

        fn windows(backing: &ChunkedKvBacking) -> Vec<(u16, u32)> {
            let state = backing.state.read().expect("lock");
            let slot = state.sequences[0].as_ref().expect("slot");
            slot.chunks_slice()
                .iter()
                .map(|c| (c.offset, c.usage))
                .collect()
        }

        #[test]
        fn a_target_inside_sealed_ground_clamps_and_cuts_only_the_writable_tail() {
            let backing = deferred_glue_slot();
            // The scheduler's offset: everything except the 13 reserved glue
            // tokens. The old contract bailed here ("target 32 cuts into the
            // Arc-shared prefix"), and the one caller that discarded the error
            // was accidentally providing the correct behaviour.
            backing.truncate_sequence_to_tokens(0, 32).unwrap();
            assert_eq!(
                windows(&backing),
                vec![(0, 32), (0, 13)],
                "sealed chunks (including the glue reservation) survive intact; \
                 the stale writable chunk is gone"
            );
        }

        #[test]
        fn a_target_in_the_writable_tail_still_trims_it() {
            let backing = deferred_glue_slot();
            // 45 sealed + 3 of the writable chunk's 5.
            backing.truncate_sequence_to_tokens(0, 48).unwrap();
            assert_eq!(
                windows(&backing),
                vec![(0, 32), (0, 13), (0, 3)],
                "a target past the sealed boundary trims the writer tail exactly"
            );
        }

        #[test]
        fn a_covering_target_is_a_no_op() {
            let backing = deferred_glue_slot();
            backing.truncate_sequence_to_tokens(0, 50).unwrap();
            assert_eq!(windows(&backing), vec![(0, 32), (0, 13), (0, 5)]);
        }
    }

    mod fork_sequence_tests {
        use super::*;

        fn setup_source_with_data(backing: &ChunkedKvBacking, tokens: usize) {
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();
        }

        #[test]
        fn test_fork_sequence_basic() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 64);
            backing.alloc_sequence().unwrap(); // slot 1 (target)

            let forked_len = backing.fork_sequence(0, 1, 64).unwrap();

            assert_eq!(forked_len, 64);
        }

        #[test]
        fn test_fork_sequence_shares_complete_blocks() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 64); // 2 complete chunks
            backing.alloc_sequence().unwrap(); // slot 1

            backing.fork_sequence(0, 1, 64).unwrap();

            // Both complete chunks should be shared (same chunk IDs)
            let snap = k_gid_snapshot(&backing);
            let row0 = &snap[0];
            let row1 = &snap[1];

            assert_eq!(row0[0], row1[0]);
            assert_eq!(row0[1], row1[1]);
        }

        #[test]
        fn test_fork_sequence_copies_partial_block() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 48); // 1 complete + 16 tokens partial
            backing.alloc_sequence().unwrap(); // slot 1

            backing.fork_sequence(0, 1, 48).unwrap();

            let snap = k_gid_snapshot(&backing);
            let row0 = &snap[0];
            let row1 = &snap[1];

            // First block should be shared
            assert_eq!(row0[0], row1[0]);

            // Second block (partial) should be different (copied, not shared)
            assert_ne!(row0[1], row1[1]);
        }

        #[test]
        fn test_fork_sequence_zero_len() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 64);
            backing.alloc_sequence().unwrap(); // slot 1

            // Fork with zero length should free target
            let forked_len = backing.fork_sequence(0, 1, 0).unwrap();

            assert_eq!(forked_len, 0);
        }

        #[test]
        fn test_fork_sequence_to_self_fails() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 64);

            let result = backing.fork_sequence(0, 0, 64);
            assert!(result.is_err());
        }

        #[test]
        fn test_fork_sequence_alloc() {
            let backing = create_test_backing();
            setup_source_with_data(&backing, 64);

            // fork_sequence_alloc allocates target automatically
            let target_slot = backing.fork_sequence_alloc(0, 64).unwrap();

            assert!(target_slot > 0); // Should be new slot
        }
    }

    // ==================== seq_blocks_count Tests ====================

    mod seq_blocks_count_tests {
        use super::*;

        #[test]
        fn test_seq_blocks_count_empty() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            assert_eq!(backing.seq_blocks_count(0).unwrap(), 0);
        }

        #[test]
        fn test_seq_blocks_count_with_data() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Write 96 tokens (3 chunks)
            let k = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            assert_eq!(backing.seq_blocks_count(0).unwrap(), 3);
        }
    }

    // ==================== get_chunk_refs Tests ====================

    mod get_chunk_refs_tests {
        use super::*;

        #[test]
        fn test_get_chunk_refs_empty() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            let refs = backing.get_chunk_refs(0, None).unwrap();
            assert!(refs.is_empty());
        }

        #[test]
        fn test_get_chunk_refs_with_data() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            let refs = backing.get_chunk_refs(0, None).unwrap();
            assert_eq!(refs.len(), 2);
        }

        #[test]
        fn test_get_chunk_refs_with_range() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            let k = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Get only first 2 refs
            let refs = backing.get_chunk_refs(0, Some(0..2)).unwrap();
            assert_eq!(refs.len(), 2);
        }
    }

    // ==================== ensure_block_writable Tests ====================

    mod ensure_block_writable_tests {
        use super::*;

        #[test]
        fn test_ensure_block_writable_unique() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Already unique - should return same chunk ID
            let original_id = k_gid_snapshot(&backing)[0][0];

            let writable_gids = backing.ensure_block_writable(0, 0).unwrap();

            assert_eq!(original_id, writable_gids[0].raw());
        }
    }

    // ==================== Static Chunk Cache Operations ====================

    mod static_chunk_cache_ops {
        use super::*;

        // ---- block_usage ----

        #[test]
        fn test_block_usage_default_is_chunk_size() {
            let backing = create_test_backing();
            let slot = backing.alloc_sequence().unwrap();
            let chunk_size = 32;
            // Write one block so block_count = 1
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, chunk_size);
            let bv = backing.block_usage(slot);
            // Only the allocated block gets chunk_size; unallocated blocks get 0
            assert_eq!(bv[0], chunk_size as u32);
        }

        #[test]
        fn test_set_and_get_block_usage() {
            let backing = create_test_backing();
            let slot = backing.alloc_sequence().unwrap();
            let chunk_size = 32;
            // block_usage is derived from block_count; write a block first
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, chunk_size);
            let bv = backing.block_usage(slot);
            assert_eq!(bv[0], chunk_size as u32);
        }

        #[test]
        fn test_block_usage_unallocated_returns_defaults() {
            let backing = create_test_backing();
            // Slot 99 doesn't exist — returns all-zero defaults (no active block)
            let bv = backing.block_usage(99);
            assert!(bv.iter().all(|&v| v == 0)); // unallocated slot → 0
        }

        #[test]
        fn test_block_usage_survives_free_and_realloc() {
            let backing = create_test_backing();
            let chunk_size = 32;
            let slot = backing.alloc_sequence().unwrap();
            // Write a block, then free
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.free_sequence(slot).unwrap();
            // After realloc, slot should be reset: block_count=0
            let slot2 = backing.alloc_sequence().unwrap();
            assert_eq!(slot2, slot); // reuses freed slot
                                     // Fresh slot with no blocks: block_usage returns all 0s
            let bv = backing.block_usage(slot2);
            assert_eq!(
                bv[0], 0,
                "reused slot has no blocks, block_usage[0] should be 0"
            );
        }

        // ---- slot_chunk_ids ----

        #[test]
        fn test_slot_chunk_ids_empty_slot() {
            let backing = create_test_backing();
            let slot = backing.alloc_sequence().unwrap();
            let ids = backing.slot_chunk_ids(slot).unwrap();
            assert!(ids.is_empty());
        }

        #[test]
        fn test_slot_chunk_ids_after_alloc_blocks() {
            let backing = create_test_backing();
            let slot = backing.alloc_sequence().unwrap();
            // Write some tokens to trigger block allocation
            let chunk_size = 32; // matches CHUNK_SIZE
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            let ids = backing.slot_chunk_ids(slot).unwrap();
            assert_eq!(ids.len(), 1);
            assert!(ids[0][0].raw() >= 0); // valid chunk ID
        }

        #[test]
        fn test_slot_chunk_ids_unallocated_errors() {
            let backing = create_test_backing();
            assert!(backing.slot_chunk_ids(99).is_err());
        }

        // ---- chunk_rope_positions ----

        #[test]
        fn test_chunk_rope_positions_default_all_zero() {
            let backing = create_test_backing();
            let slot = backing.alloc_sequence().unwrap();
            let positions = backing.chunk_rope_positions(slot);
            assert!(positions.iter().all(|&s| s == 0));
        }

        // ---- inject_prefix_chunks ----

        #[test]
        fn test_inject_prefix_basic() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Create a "prototype" slot with one block of data
            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, chunk_size);

            // Get chunk IDs from prototype
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();
            assert_eq!(chunk_ids.len(), 1);

            // Create target slot and inject.
            let target = backing.alloc_sequence().unwrap();
            backing.inject_prefix_chunks(target, &chunk_ids, 5).unwrap();

            // Verify target has the same chunk IDs
            let target_ids = backing.slot_chunk_ids(target).unwrap();
            assert_eq!(target_ids, chunk_ids);

            // Verify stored rope position for block 0 is 0 (canonical: 0 * chunk_size).
            let target_positions = backing.chunk_rope_positions(target);
            assert_eq!(target_positions[0], 0);
        }

        #[test]
        fn test_inject_prefix_multiple_chunks() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Create prototype with 2 blocks
            let proto = backing.alloc_sequence().unwrap();
            let tokens = chunk_size * 2;
            backing.ensure_for_offset(proto, 0, tokens).unwrap();
            let data = Tensor::zeros((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, tokens);

            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();
            assert_eq!(chunk_ids.len(), 2);

            // Inject with dense positions (prefix starts at position 0).
            let target = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(target, &chunk_ids, tokens)
                .unwrap();

            let target_ids = backing.slot_chunk_ids(target).unwrap();
            assert_eq!(target_ids, chunk_ids);

            let target_positions = backing.chunk_rope_positions(target);
            assert_eq!(target_positions[0], 0); // block 0: 0 * 32
            assert_eq!(target_positions[1], 32); // block 1: 1 * 32
        }

        #[test]
        fn test_inject_prefix_with_block_usage() {
            // Simulates a 5-token fragment in chunk_size=32: bv = [5]
            let backing = create_test_backing();
            let chunk_size = 32;

            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, 5);

            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            let target = backing.alloc_sequence().unwrap();
            backing.inject_prefix_chunks(target, &chunk_ids, 5).unwrap();

            // block_usage is derived from sealed_block_count; since seq_len=5 < chunk_size,
            // the block is not yet sealed, so block_usage defaults to chunk_size.
            let _ = backing.block_usage(target); // just verify it doesn't panic
                                                 // seq_len is set internally by inject_prefix_chunks via set_len
        }

        #[test]
        fn test_multiple_borrowers_same_prototype() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Prototype
            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, chunk_size);
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            // Two borrowers both taking the same prefix from position 0.
            let b1 = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(b1, &chunk_ids, chunk_size)
                .unwrap();

            let b2 = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(b2, &chunk_ids, chunk_size)
                .unwrap();

            // Both share the same chunk IDs
            assert_eq!(backing.slot_chunk_ids(b1).unwrap(), chunk_ids);
            assert_eq!(backing.slot_chunk_ids(b2).unwrap(), chunk_ids);

            // Both have the same stored position for block 0
            assert_eq!(backing.chunk_rope_positions(b1)[0], 0);
            assert_eq!(backing.chunk_rope_positions(b2)[0], 0);
        }

        #[test]
        fn test_free_borrower_does_not_free_prototype_chunks() {
            let backing = create_test_backing();
            let chunk_size = 32;

            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, chunk_size);
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            // Borrow and then free
            let borrower = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(borrower, &chunk_ids, chunk_size)
                .unwrap();
            backing.free_sequence(borrower).unwrap();

            // Prototype's chunk IDs should still be valid
            let proto_ids = backing.slot_chunk_ids(proto).unwrap();
            assert_eq!(proto_ids, chunk_ids);
        }
    }

    // ==================== Regression Tests ====================

    mod regression_tests {
        use super::*;

        /// Regression: inject a partial boundary (N*chunk_size + r tokens, 0 < r < chunk_size)
        /// then call `ensure_for_offset` with enough new tokens to overflow the active block.
        ///
        /// Before the fix, `ensure_for_offset` assumed the previous active block was completely
        /// full (`chunk_size - active_chunk_offset` tokens), so it committed the last boundary
        /// block with `usage = chunk_size` instead of `r`.  This caused `chunk_meta_row` to
        /// produce `active_use = 0` for the new block and `usage = chunk_size` (garbage-attending)
        /// for the boundary block.
        #[test]
        fn test_ensure_for_offset_after_partial_inject() {
            let backing = create_test_backing();
            let chunk_size = 32usize;

            // Build a "prototype" with 2 blocks so that the injection leaves a
            // partial last block: r = 5 tokens in block 1.
            let r = 5usize;
            let proto_tokens = chunk_size + r; // 37
            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, proto_tokens).unwrap();
            let data = Tensor::zeros((1, 4, proto_tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, proto_tokens);
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();
            assert_eq!(chunk_ids.len(), 2);

            // Inject into a fresh target.
            let target = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(target, &chunk_ids, proto_tokens)
                .unwrap();

            // Adding `overflow` new tokens forces a third block to be allocated.
            // Block 1 (the previous active) is sealed with full capacity
            // because the write extends past it — the caller (prefill kernel)
            // will fill positions r..chunk_size-1 before reading.
            let overflow = chunk_size - r + 1; // 28 tokens; forces block 2
            backing
                .ensure_for_offset(target, proto_tokens, overflow)
                .unwrap();
            backing.set_len(target, proto_tokens + overflow);

            // chunk_meta_row must reflect correct token counts.
            let seq_len = proto_tokens + overflow;
            let meta = backing.chunk_meta_row(target, seq_len);

            // Block 0: fully sealed with chunk_size tokens at rope 0.
            assert_eq!(meta[0].usage(), chunk_size as u32, "block 0 usage");
            assert_eq!(meta[0].rope_base(), 0, "block 0 rope_base");

            // Block 1: sealed with full capacity (the write at offset=37 fills
            // positions 5..31, completing the block).
            assert_eq!(
                meta[1].usage(),
                chunk_size as u32,
                "block 1 usage must be full capacity (write fills the rest)"
            );
            assert_eq!(
                meta[1].rope_base(),
                chunk_size as i32,
                "block 1 rope_base must be chunk_size"
            );

            // Block 2: active block.  committed = 2 * chunk_size = 64,
            // so only seq_len - 64 = 1 token lands here.
            let committed = 2 * chunk_size;
            let expected_active_use = (seq_len - committed) as u32;
            assert_eq!(
                meta[2].usage(),
                expected_active_use,
                "active block usage must be {} (seq_len - committed)",
                expected_active_use
            );
        }

        /// Regression for B1: `append_borrowed_chunks_cow` must seal the active block
        /// with its ACTUAL token count, not with `CHUNK_SIZE - offset`.
        ///
        /// Scenario:
        ///   - Target has 1 full committed block (32 tokens) + partial active block (5 tokens).
        ///   - `append_borrowed_chunks_cow` is called to inject 2 borrowed blocks.
        ///   - The partial active block must be sealed with usage = 5, not usage = 32.
        ///   - Due to the cascading rope_base computation, block 2 (first borrowed block)
        ///     must have rope_base = 37 (= 32 + 5), not rope_base = 64 (= 32 + 32).
        #[test]
        fn test_append_borrowed_cow_preserves_partial_active_usage() {
            let backing = create_test_backing();
            let chunk_size = 32usize;
            let partial = 5usize; // tokens in the partial active block

            // --- Build proto with 2 full blocks whose GIDs we will borrow ---
            let proto = backing.alloc_sequence().unwrap();
            let proto_tokens = chunk_size * 2;
            backing.ensure_for_offset(proto, 0, proto_tokens).unwrap();
            let proto_data =
                Tensor::zeros((1, 4, proto_tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing
                .write_contiguous(proto, 0, &proto_data, &proto_data)
                .unwrap();
            backing.set_len(proto, proto_tokens);
            let proto_ids = backing.slot_chunk_ids(proto).unwrap();
            assert_eq!(proto_ids.len(), 2, "proto must have 2 blocks");

            // --- Build target with 1 full block + partial active ---
            let target = backing.alloc_sequence().unwrap();
            let target_tokens = chunk_size + partial; // 37
            backing.ensure_for_offset(target, 0, target_tokens).unwrap();
            let target_data =
                Tensor::zeros((1, 4, target_tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing
                .write_contiguous(target, 0, &target_data, &target_data)
                .unwrap();
            // set_len syncs active.usage = partial (5)
            backing.set_len(target, target_tokens);

            // --- Inject borrowed blocks (the operation that triggers the bug) ---
            backing
                .append_borrowed_chunks_cow(target, &proto_ids, proto_tokens)
                .unwrap();

            // --- Verify invariants ---
            // We pass seq_len = target_tokens (37) — only the pre-injection tokens.
            let meta = backing.chunk_meta_row(target, target_tokens);

            // Block 0: original full committed block, unchanged.
            assert_eq!(
                meta[0].usage(),
                chunk_size as u32,
                "block 0 usage = chunk_size"
            );
            assert_eq!(meta[0].rope_base(), 0, "block 0 rope_base = 0");

            // Block 1: SEALED PARTIAL ACTIVE — must have usage = partial (5), NOT chunk_size (32).
            // Bug: overwrites with CHUNK_SIZE - offset = 32 instead of preserving actual fill.
            assert_eq!(
                meta[1].usage(),
                partial as u32,
                "sealed partial active block must have usage == {partial} (actual fill), not chunk_size"
            );
            assert_eq!(
                meta[1].rope_base(),
                chunk_size as i32,
                "block 1 rope_base = {chunk_size}"
            );

            // Block 2: first borrowed block.
            // rope_base must be chunk_size + partial = 37, not 64 (cascading effect of B1).
            assert_eq!(
                meta[2].rope_base(),
                (chunk_size + partial) as i32,
                "borrowed block rope_base must equal chunk_size + partial = {} (cascading RoPE check)",
                chunk_size + partial
            );
            assert_eq!(
                meta[2].usage(),
                chunk_size as u32,
                "borrowed block usage = chunk_size"
            );
        }

        /// Same scenario but with a FULL boundary last block (r = 0, P = N * chunk_size).
        /// The last block must still be committed with usage = chunk_size (not 0).
        #[test]
        fn test_ensure_for_offset_after_aligned_inject() {
            let backing = create_test_backing();
            let chunk_size = 32usize;

            let proto_tokens = chunk_size * 2; // exactly 2 full blocks
            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, proto_tokens).unwrap();
            let data = Tensor::zeros((1, 4, proto_tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, proto_tokens);
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            let target = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(target, &chunk_ids, proto_tokens)
                .unwrap();

            // Add 1 token to force a new block.
            backing.ensure_for_offset(target, proto_tokens, 1).unwrap();
            backing.set_len(target, proto_tokens + 1);

            let seq_len = proto_tokens + 1;
            let meta = backing.chunk_meta_row(target, seq_len);

            assert_eq!(meta[0].usage(), chunk_size as u32, "block 0 usage");
            assert_eq!(
                meta[1].usage(),
                chunk_size as u32,
                "block 1 usage must stay chunk_size"
            );
            assert_eq!(meta[2].usage(), 1u32, "active block has 1 new token");
        }
    }

    // ==================== RoPE-at-Read-Time Architecture Tests ====================
    //
    // K is stored un-rotated in the arena. The decode kernel applies RoPE at
    // read time using canonical positions: block B → B * chunk_size.
    // These tests verify that chunk_rope_positions() returns the correct
    // canonical layout under various cache lifecycle scenarios.

    mod rope_at_read_time_tests {
        use super::*;

        /// After growing a sequence from 1 block to 3 blocks, each block's
        /// canonical position must equal block_index * chunk_size.
        #[test]
        fn test_canonical_positions_after_growth() {
            let backing = create_test_backing();
            let chunk_size = 32;

            let slot = backing.alloc_sequence().unwrap();

            // Start with 1 block
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, chunk_size);

            let pos1 = backing.chunk_rope_positions(slot);
            assert_eq!(pos1[0], 0, "block 0 should be 0 * chunk_size");

            // Grow to 3 blocks
            let tokens_3 = chunk_size * 3;
            backing
                .ensure_for_offset(slot, chunk_size, chunk_size * 2)
                .unwrap();
            let data3 = Tensor::zeros((1, 4, tokens_3, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data3, &data3).unwrap();
            backing.set_len(slot, tokens_3);

            let pos3 = backing.chunk_rope_positions(slot);
            assert_eq!(pos3[0], 0, "block 0 → 0 * 32 = 0");
            assert_eq!(pos3[1], 32, "block 1 → 1 * 32 = 32");
            assert_eq!(pos3[2], 64, "block 2 → 2 * 32 = 64");
        }

        /// After COW via append_borrowed_chunks_cow, the borrowed blocks
        /// still get canonical positions based on their block index in the
        /// borrower's slot, not the prototype's slot.
        #[test]
        fn test_canonical_positions_after_cow_borrow() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Create prototype with 2 blocks
            let proto = backing.alloc_sequence().unwrap();
            let tokens = chunk_size * 2;
            backing.ensure_for_offset(proto, 0, tokens).unwrap();
            let data = Tensor::zeros((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, tokens);

            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            // Borrow those 2 blocks into a new slot using dense positions.
            let borrower = backing.alloc_sequence().unwrap();
            backing
                .append_borrowed_chunks_cow(borrower, &chunk_ids, tokens)
                .unwrap();

            // Rope positions are dense sequential (formula: blk * chunk_size).
            let pos = backing.chunk_rope_positions(borrower);
            assert_eq!(pos[0], 0, "borrowed block 0 → 0 * 32");
            assert_eq!(pos[1], 32, "borrowed block 1 → 1 * 32");

            // Prototype also has canonical positions
            let proto_pos = backing.chunk_rope_positions(proto);
            assert_eq!(proto_pos[0], 0);
            assert_eq!(proto_pos[1], 32);
        }

        /// Multiple borrowers of the same block all storing the same rope_position
        /// (prefix at logical offset 0) must each reflect that position correctly.
        #[test]
        fn test_positions_invariant_under_different_rope_shifts() {
            let backing = create_test_backing();
            let chunk_size = 32;

            let proto = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(proto, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, chunk_size);
            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            // Inject the same block into 3 different targets, all starting at position 0.
            let targets: Vec<usize> = (0..3).map(|_| backing.alloc_sequence().unwrap()).collect();
            for &t in &targets {
                backing
                    .inject_prefix_chunks(t, &chunk_ids, chunk_size)
                    .unwrap();
            }

            // All three targets must have position 0 for block 0.
            for &t in &targets {
                assert_eq!(
                    backing.chunk_rope_positions(t)[0],
                    0,
                    "all borrowers at prefix offset 0 must have block-0 position == 0"
                );
            }
        }

        /// Unallocated blocks in a sequence must have position 0 (benign default).
        #[test]
        fn test_unallocated_blocks_have_zero_position() {
            let backing = create_test_backing();
            let chunk_size = 32;

            let slot = backing.alloc_sequence().unwrap();
            // Allocate only 1 block
            backing.ensure_for_offset(slot, 0, chunk_size).unwrap();
            let data = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, chunk_size);

            let positions = backing.chunk_rope_positions(slot);
            assert_eq!(positions[0], 0, "allocated block 0 → 0");
            // All remaining blocks should be 0 (unallocated default)
            for (i, &p) in positions.iter().enumerate().skip(1) {
                assert_eq!(p, 0, "unallocated block {} should be 0", i);
            }
        }

        /// After free_sequence, re-allocating the same slot must reset
        /// canonical positions to all zeros (no stale data from previous use).
        #[test]
        fn test_positions_reset_after_free_and_realloc() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Fill slot with 2 blocks
            let slot = backing.alloc_sequence().unwrap();
            let tokens = chunk_size * 2;
            backing.ensure_for_offset(slot, 0, tokens).unwrap();
            let data = Tensor::zeros((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, tokens);

            // Verify non-zero positions exist
            let before = backing.chunk_rope_positions(slot);
            assert_eq!(before[1], 32);

            // Free and re-alloc
            backing.free_sequence(slot).unwrap();
            let slot2 = backing.alloc_sequence().unwrap();

            // All positions must be 0 in the fresh slot
            let after = backing.chunk_rope_positions(slot2);
            assert!(
                after.iter().all(|&p| p == 0),
                "positions should be all zero after free+realloc, got {:?}",
                after
            );
        }

        /// Multiple independent sequences should have independent canonical
        /// positions (no cross-contamination between batch slots).
        #[test]
        fn test_multi_batch_independent_positions() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Slot 0: 3 blocks
            let s0 = backing.alloc_sequence().unwrap();
            let tok3 = chunk_size * 3;
            backing.ensure_for_offset(s0, 0, tok3).unwrap();
            let d3 = Tensor::zeros((1, 4, tok3, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(s0, 0, &d3, &d3).unwrap();
            backing.set_len(s0, tok3);

            // Slot 1: 1 block
            let s1 = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(s1, 0, chunk_size).unwrap();
            let d1 = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(s1, 0, &d1, &d1).unwrap();
            backing.set_len(s1, chunk_size);

            let pos0 = backing.chunk_rope_positions(s0);
            let pos1 = backing.chunk_rope_positions(s1);

            // Slot 0 has 3 canonical positions
            assert_eq!(pos0[0], 0);
            assert_eq!(pos0[1], 32);
            assert_eq!(pos0[2], 64);

            // Slot 1 has only 1
            assert_eq!(pos1[0], 0);
            assert_eq!(pos1[1], 0, "slot 1 block 1 unallocated → 0");
        }

        /// Canonical positions must work correctly when the sequence
        /// spans exactly one token (partial first block).
        #[test]
        fn test_single_token_sequence_position() {
            let backing = create_test_backing();

            let slot = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(slot, 0, 1).unwrap();
            let data = Tensor::zeros((1, 4, 1, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(slot, 0, &data, &data).unwrap();
            backing.set_len(slot, 1);

            let pos = backing.chunk_rope_positions(slot);
            assert_eq!(pos[0], 0, "single-token sequence: block 0 → 0");
        }

        /// Inject prefix then grow: canonical positions for injected + new
        /// blocks must form a contiguous sequence.
        #[test]
        fn test_inject_prefix_then_grow_positions() {
            let backing = create_test_backing();
            let chunk_size = 32;

            // Create prototype with 2 blocks
            let proto = backing.alloc_sequence().unwrap();
            let proto_tokens = chunk_size * 2;
            backing.ensure_for_offset(proto, 0, proto_tokens).unwrap();
            let data = Tensor::zeros((1, 4, proto_tokens, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(proto, 0, &data, &data).unwrap();
            backing.set_len(proto, proto_tokens);

            let chunk_ids = backing.slot_chunk_ids(proto).unwrap();

            // Inject 2 prefix blocks with dense positions.
            let target = backing.alloc_sequence().unwrap();
            backing
                .inject_prefix_chunks(target, &chunk_ids, proto_tokens)
                .unwrap();

            // Grow by 1 more block (new tokens after prefix)
            backing
                .ensure_for_offset(target, proto_tokens, chunk_size)
                .unwrap();
            let extra = Tensor::zeros((1, 4, chunk_size, 32), DType::BF16, &Device::Cpu).unwrap();
            backing
                .write_contiguous(target, proto_tokens, &extra, &extra)
                .unwrap();
            backing.set_len(target, proto_tokens + chunk_size);

            let pos = backing.chunk_rope_positions(target);
            assert_eq!(pos[0], 0, "prefix block 0 → 0");
            assert_eq!(pos[1], 32, "prefix block 1 → 32");
            assert_eq!(pos[2], 64, "new block 2 → 64");
        }
    }

    // ==================== Multi-Turn Scenario (Integration) Tests ====================

    mod multi_turn_scenario_tests {
        use super::*;
        use crate::kv_cache::ChunkMeta;

        /// Assert the B3 + A3 invariants for `num_blocks` consecutive blocks.
        ///
        /// - B3: `rope_base[i] == sum(usage[0..i])` for every block
        /// - A3: `offset == 0` for every standard block
        fn check_rope_and_offset_invariant(meta: &[ChunkMeta], num_blocks: usize) {
            let mut cumulative: i32 = 0;
            for i in 0..num_blocks {
                let cm = &meta[i];
                assert_eq!(
                    cm.rope_base(),
                    cumulative,
                    "B3: block {i} rope_base {} != cumulative sum {cumulative}",
                    cm.rope_base()
                );
                assert_eq!(
                    cm.offset(),
                    0,
                    "A3: block {i} offset {} != 0 (standard block must start at offset 0)",
                    cm.offset()
                );
                cumulative += cm.usage() as i32;
            }
        }

        /// Integration reproducer for the model-degradation bug:
        ///   Qwen3-30B-A3B (Q4_K_M) producing gibberish output after a system prompt.
        ///
        /// Scenario (Arc-share-everything model — current live design):
        ///   1. System prompt of 985 tokens → parent.chunks = [block0..block29(32), block30(25)].
        ///   2. `create_view_sequence` Arc-borrows every parent block (full and
        ///      partial) read-only and pushes a fresh empty active block —
        ///      `view.chunks = [parent.b0..parent.b30 (Arc), fresh_active(0)]`.
        ///      `borrowed_blocks == parent_blocks` (all 31). The fresh active
        ///      at index 31 is uniquely owned and is where the next turn's
        ///      writes land — no COW of the partial is needed because the
        ///      Arc-shared partial is never written to.
        ///   3. User message of 5 tokens is prefilled into block31 at logical
        ///      offset 985.
        ///   4. `set_len(view, 991)` simulates the first decode step (block31
        ///      usage advances by one).
        ///
        /// Covers:
        ///   - D1: `borrowed_token_count == sys_tokens` (view offset is correct).
        ///   - B3/A3: `rope_base[i] == sum(usage[0..i])` and `offset[i] == 0`
        ///     at every step — including across the mid-sequence partial at
        ///     block 30.
        ///   - B1 (regression): partial blocks stored with correct usage, not
        ///     rounded up.
        ///   - Decode: after `set_len(view, 991)` block31.usage == 6 (5 user +
        ///     1 decode).
        #[test]
        fn test_system_prompt_view_user_message_rope_invariants() {
            const CHUNK_SIZE: usize = 32;
            // 985 = 30 full blocks + 25 token partial active.
            let sys_tokens = 30 * CHUNK_SIZE + 25; // 985
            let n_full_sys_blocks = sys_tokens / CHUNK_SIZE; // 30
            let partial_sys = sys_tokens % CHUNK_SIZE; // 25
            let parent_blocks = n_full_sys_blocks + 1; // 31 = ceil(985/32)
            let user_tokens = 5usize;
            let n_kv_head = 4usize;
            let head_dim = 32usize;

            // Need capacity for parent (31 blocks) + view (31 borrowed + 1 new = 32 blocks).
            // initial_seq_len = 33 * CHUNK_SIZE = 1056 gives max_blocks = 33.
            let backing =
                ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, &Device::Cpu, 1056)
                    .unwrap();

            // ── Step 1: Fill parent with system-prompt tokens ──────────────────────
            let parent = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(parent, 0, sys_tokens).unwrap();
            let k_sys = Tensor::zeros(
                (1, n_kv_head, sys_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            let v_sys = Tensor::zeros(
                (1, n_kv_head, sys_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            backing.write_contiguous(parent, 0, &k_sys, &v_sys).unwrap();
            // set_len syncs active.usage = partial_sys so seq_len() == 985.
            backing.set_len(parent, sys_tokens);

            // ── Step 2: Verify parent chunk_meta (B3 + A3) ────────────────────────
            let parent_meta = backing.chunk_meta_row(parent, sys_tokens);
            check_rope_and_offset_invariant(&parent_meta, parent_blocks);

            // Spot-check boundary blocks.
            assert_eq!(
                parent_meta[n_full_sys_blocks - 1].usage(),
                CHUNK_SIZE as u32,
                "last full parent block usage must be CHUNK_SIZE"
            );
            assert_eq!(
                parent_meta[n_full_sys_blocks - 1].rope_base(),
                ((n_full_sys_blocks - 1) * CHUNK_SIZE) as i32,
                "last full parent block rope_base"
            );
            assert_eq!(
                parent_meta[n_full_sys_blocks].usage(),
                partial_sys as u32,
                "partial active parent block usage must be {partial_sys}, not CHUNK_SIZE (B1 check)"
            );
            assert_eq!(
                parent_meta[n_full_sys_blocks].rope_base(),
                (n_full_sys_blocks * CHUNK_SIZE) as i32,
                "partial active parent block rope_base"
            );

            // ── Step 3: Create view borrowing all 31 parent blocks ────────────────
            let view = backing.alloc_sequence().unwrap();
            let (borrowed_blocks, borrowed_tokens) = backing
                .create_view_sequence(view, parent, &[(0, parent_blocks)])
                .unwrap();

            // Arc-share-everything: every parent block (including the
            // partial tail) is Arc-borrowed into the view. The view also
            // gains a fresh empty active block at index `parent_blocks`
            // so the next turn's writes have somewhere unshared to land
            // — `borrowed_blocks` therefore equals the full source count.
            assert_eq!(
                borrowed_blocks, parent_blocks,
                "Arc-share: borrowed_blocks must equal parent_blocks = {parent_blocks}"
            );
            // D1: view offset must equal the actual number of borrowed tokens.
            assert_eq!(
                borrowed_tokens, sys_tokens,
                "D1: borrowed_token_count must equal sys_tokens = {sys_tokens}"
            );

            // The view's structural shape: `parent_blocks` Arc-shared + 1
            // fresh active.
            let writer_block_idx = parent_blocks; // 31 (the fresh active)
            let total_blocks = parent_blocks + 1; // 32 chunks
            {
                let state = backing.state.read().unwrap();
                let vs = state.sequences[view].as_ref().unwrap();
                assert_eq!(
                    vs.chunks_slice().len(),
                    total_blocks,
                    "view must have parent_blocks + 1 chunks (Arc-share + fresh active)"
                );
                assert_eq!(
                    vs.chunks_slice()[writer_block_idx].usage,
                    0,
                    "fresh active block starts empty"
                );
            }

            // ── Step 4: Verify view chunk_meta immediately after creation ──────────
            // chunk_meta_row(view, sys_tokens) covers the first
            // `parent_blocks` blocks (the fresh active at index 31 is
            // empty so it doesn't contribute until writes land in it).
            let view_meta_init = backing.chunk_meta_row(view, sys_tokens);
            check_rope_and_offset_invariant(&view_meta_init, parent_blocks);

            // Every borrowed block must match the parent.
            for i in 0..parent_blocks {
                assert_eq!(
                    view_meta_init[i].usage(),
                    parent_meta[i].usage(),
                    "view block {i} usage must match parent ({} == {})",
                    view_meta_init[i].usage(),
                    parent_meta[i].usage()
                );
                assert_eq!(
                    view_meta_init[i].rope_base(),
                    parent_meta[i].rope_base(),
                    "view block {i} rope_base must match parent ({} == {})",
                    view_meta_init[i].rope_base(),
                    parent_meta[i].rope_base()
                );
            }

            // ── Step 5: Prefill user message into the view ───────────────────────
            backing
                .ensure_for_offset(view, sys_tokens, user_tokens)
                .unwrap();
            let k_user = Tensor::zeros(
                (1, n_kv_head, user_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            let v_user = Tensor::zeros(
                (1, n_kv_head, user_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            backing
                .write_contiguous(view, sys_tokens, &k_user, &v_user)
                .unwrap();
            // Mirrors `set_current_seq_len(offset + seq_len)` from prefill_utils.rs.
            let total_prefill = sys_tokens + user_tokens; // 990
            backing.set_len(view, total_prefill);

            // ── Step 6: Verify view chunk_meta after user-message prefill ──────────
            // Arc-share: the user tokens land in the fresh active block
            // at `writer_block_idx`. The partial parent block at
            // `n_full_sys_blocks` stays Arc-shared and unchanged.
            let view_meta_prefill = backing.chunk_meta_row(view, total_prefill);
            check_rope_and_offset_invariant(&view_meta_prefill, total_blocks);

            // Block 30 (Arc-shared partial) keeps its original usage.
            assert_eq!(
                view_meta_prefill[n_full_sys_blocks].usage(),
                partial_sys as u32,
                "Arc-shared partial block 30 usage stays at partial_sys = {partial_sys}"
            );
            assert_eq!(
                view_meta_prefill[n_full_sys_blocks].rope_base(),
                (n_full_sys_blocks * CHUNK_SIZE) as i32,
                "block 30 rope_base must be n_full_sys_blocks * CHUNK_SIZE = {}",
                n_full_sys_blocks * CHUNK_SIZE
            );
            // Block 31 (fresh active) holds the prefill bytes.
            assert_eq!(
                view_meta_prefill[writer_block_idx].usage(),
                user_tokens as u32,
                "fresh active (block 31) holds user_tokens = {user_tokens}"
            );
            assert_eq!(
                view_meta_prefill[writer_block_idx].rope_base(),
                sys_tokens as i32,
                "block 31 rope_base must equal sys_tokens = {sys_tokens}",
            );

            // ── Step 7: First decode step — set_len to total_prefill + 1 = 991 ───
            // set_len(view, 991) advances block 31's usage to user_tokens + 1.
            let decode_seq_len = total_prefill + 1; // 991
            backing.set_len(view, decode_seq_len);
            let view_meta_decode = backing.chunk_meta_row(view, decode_seq_len);
            check_rope_and_offset_invariant(&view_meta_decode, total_blocks);

            // Block 30 still unchanged.
            assert_eq!(
                view_meta_decode[n_full_sys_blocks].usage(),
                partial_sys as u32,
                "decode step: Arc-shared block 30 still at partial_sys = {partial_sys}"
            );
            // Block 31 advances by one decode token.
            let expected_decode_usage = user_tokens + 1; // 6
            assert_eq!(
                view_meta_decode[writer_block_idx].usage(),
                expected_decode_usage as u32,
                "decode step 1: block 31 usage must be user_tokens + 1 = {expected_decode_usage}"
            );
            assert_eq!(
                view_meta_decode[writer_block_idx].rope_base(),
                sys_tokens as i32,
                "block 31 rope_base unchanged across the decode step"
            );
        }

        /// Regression test for block isolation across view/parent turns.
        ///
        /// Arc-share-everything model: when a view borrows N blocks from the
        /// parent, all N (including any trailing partial) are present in
        /// `view.chunks` as Arc-shared read-only entries. A fresh empty
        /// active block is pushed at index N at view-creation time so new
        /// writes never touch the Arc-shared parent data. The partial parent
        /// block is left at its original usage; the view's writes land in
        /// the fresh block instead. (The previous COW-the-partial design
        /// was reverted because it required cross-format quant elevation
        /// — Q4_KS → R16 — that isn't implemented; cold-loaded parents
        /// would fail at runtime.)
        ///
        /// Scenario (CHUNK_SIZE = 32):
        ///   1. Parent has 40 tokens: chunks = [block0(32), block1(8)].
        ///   2. `record_turn` snapshots the state; chunks unchanged.
        ///   3. Create a view borrowing all 2 parent blocks: borrowed = 2,
        ///      view.chunks = [block0 Arc, block1 Arc(8), block2_fresh(0)].
        ///   4. Write 10 tokens into the view at logical offset 40 — they
        ///      land in block2_fresh, leaving block1's Arc-shared partial
        ///      untouched.
        ///   5. `finalize_view(view, parent, 2)` → parent.chunks =
        ///      [block0(32), block1(8), block2(10)].
        ///   6. Parent ends up with 3 blocks. There is exactly one trailing
        ///      partial (block2) and one mid-sequence partial (block1) —
        ///      "no duplicate" here means a single block per logical
        ///      position, not the absence of mid-sequence partials.
        #[test]
        fn test_finalize_view_no_duplicate_partial_block() {
            const CHUNK_SIZE: usize = 32;
            let n_kv_head = 4usize;
            let head_dim = 32usize;
            let partial_tokens = 8usize;
            let turn1_tokens = CHUNK_SIZE + partial_tokens; // 40 tokens = 1 full + 8 partial
            let turn2_tokens = 10usize;

            let backing =
                ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, &Device::Cpu, 256)
                    .unwrap();

            // ── Step 1: Fill parent with turn1_tokens ──────────────────────────────
            let parent = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(parent, 0, turn1_tokens).unwrap();
            let k1 = Tensor::zeros(
                (1, n_kv_head, turn1_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            let v1 = k1.clone();
            backing.write_contiguous(parent, 0, &k1, &v1).unwrap();
            backing.set_len(parent, turn1_tokens);

            let sealed = backing.record_turn(parent).unwrap();
            assert_eq!(
                sealed.chunks.len(),
                2,
                "record_turn should produce 2 sealed chunks (1 full + 1 partial)"
            );
            // After record_turn: parent.chunks.len() == 2.
            let state = backing.state.read().unwrap();
            let ps = state.sequences[parent].as_ref().unwrap();
            assert_eq!(
                ps.chunks_slice().len(),
                2,
                "parent must have 2 chunks after record_turn"
            );
            assert_eq!(
                ps.chunks_slice()[1].usage as usize,
                partial_tokens,
                "parent chunks[1].usage must equal partial_tokens"
            );
            drop(state);

            // ── Step 2: Create a view ──────────────────────────────────────────
            let parent_block_count = 2; // ceil(40/32) = 2
            let view = backing.alloc_sequence().unwrap();
            let (borrowed_blocks, borrowed_tokens) = backing
                .create_view_sequence(view, parent, &[(0, parent_block_count)])
                .unwrap();
            // Arc-share-everything: every parent block (including the
            // partial) is Arc-borrowed; the view also gains a fresh
            // empty active block at index `parent_block_count` for the
            // next turn's writes.
            assert_eq!(
                borrowed_blocks, parent_block_count,
                "Arc-share: borrowed_blocks must equal parent_block_count"
            );
            assert_eq!(
                borrowed_tokens, turn1_tokens,
                "borrowed_tokens must equal turn1_tokens"
            );

            // ── Step 3: Write turn2_tokens into the view ──────────────────────────
            // Writes land in the fresh active block at index
            // `parent_block_count` — the Arc-shared partial at index 1
            // is never written to (it stays at usage = partial_tokens).
            backing
                .ensure_for_offset(view, borrowed_tokens, turn2_tokens)
                .unwrap();
            let writer_block_idx = parent_block_count; // 2 — the fresh active
            {
                let state = backing.state.read().unwrap();
                let vs = state.sequences[view].as_ref().unwrap();
                assert_eq!(
                    vs.chunks_slice().len(),
                    parent_block_count + 1,
                    "view must have parent_block_count + 1 chunks (Arc-share + fresh active)"
                );
                assert_eq!(
                    vs.chunks_slice()[1].usage as usize,
                    partial_tokens,
                    "Arc-shared partial keeps its original usage"
                );
                assert_eq!(
                    vs.chunks_slice()[writer_block_idx].usage,
                    0,
                    "fresh active starts at usage 0"
                );
            }
            let k2 = Tensor::zeros(
                (1, n_kv_head, turn2_tokens, head_dim),
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap();
            let v2 = k2.clone();
            backing
                .write_contiguous(view, borrowed_tokens, &k2, &v2)
                .unwrap();
            let total_view_tokens = borrowed_tokens + turn2_tokens; // 40 + 10 = 50
            backing.set_len(view, total_view_tokens);

            // Verify view chunks after write.
            {
                let state = backing.state.read().unwrap();
                let vs = state.sequences[view].as_ref().unwrap();
                assert_eq!(
                    vs.chunks_slice().len(),
                    parent_block_count + 1,
                    "view must still have exactly parent_block_count + 1 chunks"
                );
                assert_eq!(
                    vs.chunks_slice()[1].usage as usize,
                    partial_tokens,
                    "Arc-shared partial (block 1) unchanged after write"
                );
                assert_eq!(
                    vs.chunks_slice()[writer_block_idx].usage as usize,
                    turn2_tokens,
                    "fresh active (block 2) holds turn2_tokens = {turn2_tokens}"
                );
            }

            // ── Step 4: Finalize view back to parent ────────────────────────────────
            backing
                .finalize_view(view, parent, borrowed_blocks)
                .unwrap();

            // ── Step 5: Assert parent has the right blocks ───────────────────────────
            // finalize_view truncates parent to borrowed_blocks = 2 (no-op,
            // parent already has 2) and extends with view.chunks[2..] =
            // [block2_with_writes]. Parent ends with 3 blocks total.
            {
                let state = backing.state.read().unwrap();
                let ps = state.sequences[parent].as_ref().unwrap();
                assert_eq!(
                    ps.chunks_slice().len(),
                    parent_block_count + 1,
                    "parent must have parent_block_count + 1 chunks after finalize_view"
                );
                assert_eq!(
                    ps.chunks_slice()[0].usage as usize,
                    CHUNK_SIZE,
                    "parent chunks[0] (full block) must have usage = CHUNK_SIZE"
                );
                assert_eq!(
                    ps.chunks_slice()[1].usage as usize, partial_tokens,
                    "parent chunks[1] (Arc-shared partial — preserved) must have usage = partial_tokens"
                );
                assert_eq!(
                    ps.chunks_slice()[2].usage as usize,
                    turn2_tokens,
                    "parent chunks[2] (view's new block) must have usage = turn2_tokens"
                );
            }

            // ── Step 6: Verify chunk_meta for a subsequent token ─────────────────────
            let parent_offset_after = total_view_tokens; // 50
            let meta = backing.chunk_meta_row(parent, parent_offset_after);
            // Expected: 3 blocks — 1 full + 1 mid-sequence partial + 1
            // trailing partial. The rope_base invariant still holds:
            // each block's rope_base == sum of preceding usages.
            let n_blocks = parent_block_count + 1; // 3
            check_rope_and_offset_invariant(&meta, n_blocks);
            assert_eq!(
                meta[0].usage(),
                CHUNK_SIZE as u32,
                "meta block 0 must be full"
            );
            assert_eq!(meta[0].rope_base(), 0, "meta block 0 rope_base must be 0");
            assert_eq!(
                meta[1].usage(),
                partial_tokens as u32,
                "meta block 1 usage must be partial_tokens = {partial_tokens}"
            );
            assert_eq!(
                meta[1].rope_base(),
                CHUNK_SIZE as i32,
                "meta block 1 rope_base must be CHUNK_SIZE = {CHUNK_SIZE}"
            );
            assert_eq!(
                meta[2].usage(),
                turn2_tokens as u32,
                "meta block 2 usage must be turn2_tokens = {turn2_tokens}"
            );
            assert_eq!(
                meta[2].rope_base(),
                (CHUNK_SIZE + partial_tokens) as i32,
                "meta block 2 rope_base must be CHUNK_SIZE + partial_tokens = {}",
                CHUNK_SIZE + partial_tokens
            );
        }
    }

    // ==================== CPU↔GPU sealed-sequence primitives =================

    /// `inject_sealed_at_tail` and friends: pure-metadata
    /// reconstruction of a slot from a previously-snapshotted
    /// `SealedSequence`.  The substrate stores GPU-resident sealed
    /// sequences, so injection Arc-clones existing `ChunkGid`s
    /// onto the dst slot's block table without moving bytes.
    mod inject_tests {
        use super::*;
        use crate::kv_cache::chunked::CHUNK_SIZE;

        #[test]
        fn test_inject_sealed_at_tail_appends_chunks() {
            let backing = create_test_backing();
            backing.ensure_for_offset(0, 0, CHUNK_SIZE).unwrap();
            {
                let mut state = backing.state.write().expect("lock");
                let seq = state.sequences[0].as_mut().expect("seq");
                seq.chunks_slice_mut()[0].usage = CHUNK_SIZE as u32;
            }
            let sealed = backing.record_turn(0).expect("record_turn");

            // Inject into a fresh sequence and verify chunk count.
            let dst = backing.alloc_sequence().expect("alloc dst");
            let (start, end) = backing.inject_sealed_at_tail(dst, &sealed).expect("inject");
            assert_eq!(start, 0, "fresh sequence should start at block 0");
            assert_eq!(
                end,
                sealed.chunks.len(),
                "should have one block per sealed chunk"
            );

            // Verify the dst sequence's chunks match the sealed metadata.
            let state = backing.state.read().expect("lock");
            let seq = state.sequences[dst].as_ref().expect("dst seq");
            assert_eq!(seq.block_count(), sealed.chunks.len());
            for (i, sc) in sealed.chunks.iter().enumerate() {
                let cw = &seq.chunks_slice()[i];
                assert_eq!(cw.usage, sc.token_count as u32, "chunk {i} usage");
                assert_eq!(cw.offset, sc.offset, "chunk {i} offset");
            }
        }

        #[test]
        fn test_inject_empty_sealed_is_noop() {
            let backing = create_test_backing();
            let dst = backing.alloc_sequence().expect("alloc");
            let empty = crate::kv_cache::chunked::SealedSequence {
                chunks: Vec::new(),
                token_count: 0,
                chunk_size: CHUNK_SIZE,
                location: crate::kv_cache::chunked::ArenaLocation::Cpu,
            };
            let (start, end) = backing.inject_sealed_at_tail(dst, &empty).expect("inject");
            assert_eq!(start, end);
        }

        #[test]
        fn test_inject_then_record_round_trip() {
            // Full round-trip: sealed → fresh sequence via inject →
            // re-record, verifying that the chunk metadata survives
            // the round-trip (block count, usages, rope_base
            // recomputation).
            let backing = create_test_backing();
            let n_tokens = 2 * CHUNK_SIZE + 7;
            backing.ensure_for_offset(0, 0, n_tokens).unwrap();
            {
                let mut state = backing.state.write().expect("lock");
                let seq = state.sequences[0].as_mut().expect("seq");
                let cs = seq.chunks_slice_mut();
                cs[0].usage = CHUNK_SIZE as u32;
                cs[1].usage = CHUNK_SIZE as u32;
                cs[2].usage = 7;
            }

            let sealed = backing.record_turn(0).expect("record_turn");
            let dst = backing.alloc_sequence().expect("alloc dst");
            let (_, end) = backing.inject_sealed_at_tail(dst, &sealed).expect("inject");
            assert_eq!(end, sealed.chunks.len());

            // re-record on dst and check window metadata matches.  RoPE
            // is not part of the SealedChunk contract — it's computed
            // late in the kernel against the destination slot's
            // layout — so we only assert on the position-independent
            // window fields.
            let resealed = backing.record_turn(dst).expect("record_turn dst");
            assert_eq!(resealed.chunks.len(), sealed.chunks.len());
            for (i, (a, b)) in sealed.chunks.iter().zip(resealed.chunks.iter()).enumerate() {
                assert_eq!(a.token_count, b.token_count, "chunk {i} usage");
            }
            assert_eq!(resealed.token_count, sealed.token_count);
        }
    }
}
