//! Tests for chunked KV cache types: ChunkGid, SlotState, ChunkedState.

use crate::kv_cache::chunked::BlockTableState;
use crate::kv_cache::chunked::ChunkMeta;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::gid_pool::{ChunkGid, ChunkGidPool};

    /// Helper: build a ChunkWindow with a single uniform GID (n_kv_head=1).
    #[cfg(not(feature = "cuda"))]
    fn chunk_window_uniform(
        gid: ChunkGid,
        usage: u32,
        offset: u16,
    ) -> crate::kv_cache::chunked::ChunkWindow {
        // These test ChunkWindows are never GPU-serialized; use empty palettes
        // to avoid needing a head_dim in this non-GPU test helper.
        crate::kv_cache::chunked::ChunkWindow {
            gids: crate::kv_cache::chunked::head_gids::HeadGids::uniform(gid, 1),
            usage,
            offset,
            k_pal: std::sync::Arc::new(Vec::new()),
            v_pal: std::sync::Arc::new(Vec::new()),
            k_scale: std::sync::Arc::new(Vec::new()),
            v_scale: std::sync::Arc::new(Vec::new()),
        }
    }

    fn create_test_state(batch: usize, max_blocks: usize) -> BlockTableState {
        BlockTableState {
            layer_idx: 0,
            max_blocks,
            sequences: vec![None; batch],
        }
    }

    fn bytes_per_chunk_for_format(format: crate::kv_cache::KvFormat) -> usize {
        let elems_per_chunk =
            crate::kv_cache::chunked::CHUNK_SIZE * crate::kv_cache::chunked::CHUNK_SIZE;
        match format {
            crate::kv_cache::KvFormat::Float(dtype) => elems_per_chunk * dtype.size_in_bytes(),
            crate::kv_cache::KvFormat::Quantized(qf) => {
                let ggml = qf.to_ggml_dtype();
                (elems_per_chunk / ggml.block_size()) * ggml.type_size()
            }
        }
    }

    mod arena_sizing_tests {
        use crate::kv_cache::chunked::types::{arena_chunks_for_format, TARGET_ARENA_BYTES};
        use crate::kv_cache::{KvFormat, QuantFormat};
        use candle::DType;

        #[test]
        fn test_f16_formula_matches_legacy_arena_chunks() {
            assert_eq!(arena_chunks_for_format(KvFormat::Float(DType::F16)), 8192);
        }

        #[test]
        fn test_quant_formula_stays_within_16mib_budget() {
            let format = KvFormat::Quantized(QuantFormat::Q8_0);
            let n = arena_chunks_for_format(format);
            let bytes_per_chunk = super::bytes_per_chunk_for_format(format);
            assert!(n * bytes_per_chunk <= TARGET_ARENA_BYTES);
            assert!((n + 1) * bytes_per_chunk > TARGET_ARENA_BYTES);
        }
    }

    // ==================== ChunkGid Tests ====================

    mod chunk_gid_tests {
        use super::*;

        fn create_test_handle(id: i64) -> ChunkGid {
            ChunkGid::detached(id)
        }

        #[test]
        fn test_chunk_gid_new_basic() {
            let handle = create_test_handle(0);
            let chunk_ref = handle;

            assert_eq!(chunk_ref.raw(), 0);
            assert_eq!(chunk_ref.arena_idx(), 0);
            assert_eq!(chunk_ref.chunk_idx(), 0);
        }

        #[test]
        fn test_chunk_gid_arena_addressing() {
            let arena_stride = crate::arena_gid_stride();

            let handle0 = create_test_handle(0);
            let ref0 = handle0;
            assert_eq!(ref0.arena_idx(), 0);
            assert_eq!(ref0.chunk_idx(), 0);

            let handle63 = create_test_handle(63);
            let ref63 = handle63;
            assert_eq!(ref63.arena_idx(), 0);
            assert_eq!(ref63.chunk_idx(), 63);

            let handle_base1 = create_test_handle(arena_stride as i64);
            let ref_base1 = handle_base1;
            assert_eq!(ref_base1.arena_idx(), 1);
            assert_eq!(ref_base1.chunk_idx(), 0);

            let handle_base1_63 = create_test_handle(arena_stride as i64 + 63);
            let ref_base1_63 = handle_base1_63;
            assert_eq!(ref_base1_63.arena_idx(), 1);
            assert_eq!(ref_base1_63.chunk_idx(), 63);
        }

        #[test]
        fn test_chunk_gid_with_rope_shift() {
            // rope_position is stored in ChunkMeta, not ChunkGid.
            // Verify ChunkGid addressing is correct when gid is non-trivial.
            let handle = create_test_handle(42);
            let chunk_ref = handle;
            assert_eq!(chunk_ref.raw(), 42);
            assert_eq!(chunk_ref.arena_idx(), 0); // 42 / 512 = 0
            assert_eq!(chunk_ref.chunk_idx(), 42); // 42 % 512 = 42
        }

        #[test]
        fn test_chunk_gid_negative_rope_shift() {
            // Rope position lives in ChunkMeta. This test verified that
            // ChunkGid can be created from any valid handle.
            let handle = create_test_handle(5);
            let chunk_ref = handle;
            assert_eq!(chunk_ref.raw(), 5);
        }

        #[test]
        fn test_chunk_gid_is_shared() {
            let gid = create_test_handle(42);
            let ref1 = gid.clone();
            let ref2 = gid;

            // ref1 + ref2 share the Arc -> strong_count > 1 -> shared
            assert!(ref1.is_shared());
            assert!(ref2.is_shared());
            assert!(!ref1.is_unique());
            assert!(!ref2.is_unique());
        }

        #[test]
        fn test_chunk_gid_is_unique() {
            let gid = create_test_handle(42);
            let chunk_ref = gid;

            // Only chunk_ref holds the Arc -> strong_count 1 -> unique
            assert!(chunk_ref.is_unique());
            assert!(!chunk_ref.is_shared());
        }

        #[test]
        fn test_chunk_gid_with_shifted_rope() {
            // rope_position is no longer part of ChunkGid (moved to ChunkMeta).
            // Verify that cloning a ChunkGid shares the gid (COW semantics).
            let gid = create_test_handle(42);
            let ref1 = gid;
            let ref2 = ref1.clone();

            assert_eq!(ref1.raw(), ref2.raw());
            assert_eq!(ref1.arena_idx(), ref2.arena_idx());
            assert_eq!(ref1.chunk_idx(), ref2.chunk_idx());

            // ref1 + ref2 share the Arc -> strong_count > 1 -> shared
            assert!(ref1.is_shared());
            assert!(ref2.is_shared());
        }

        #[test]
        fn test_chunk_gid_new_allocated() {
            let chunk_ref = ChunkGid::detached(100);

            assert_eq!(chunk_ref.raw(), 100);
            assert_eq!(chunk_ref.arena_idx(), 0); // 100 / 512 = 0
            assert_eq!(chunk_ref.chunk_idx(), 100); // 100 % 512 = 100
        }

        #[test]
        fn test_chunk_gid_clone() {
            let gid = create_test_handle(42);
            let ref1 = gid.clone();
            let ref2 = ref1.clone();

            assert_eq!(ref1.raw(), ref2.raw());
            assert_eq!(ref1.arena_idx(), ref2.arena_idx());
            assert_eq!(ref1.chunk_idx(), ref2.chunk_idx());

            // gid + ref1 + ref2 -> strong_count 3 > 2 -> shared
            assert!(ref1.is_shared());
            assert!(ref2.is_shared());
        }
    }

    // ==================== SlotState Tests ====================

    #[cfg(not(feature = "cuda"))]
    mod slot_state_tests {
        use super::*;

        #[test]
        fn test_slot_state_new() {
            let slot = crate::kv_cache::chunked::SequenceState::new();

            assert_eq!(slot.block_count(), 0);
            assert_eq!(slot.block_count(), 0);
        }

        #[test]
        fn test_slot_state_with_chunks() {
            let mut slot = crate::kv_cache::chunked::SequenceState::new();

            // Add a committed chunk and an active chunk
            let ref0 = ChunkGid::detached(0);
            let ref1 = ChunkGid::detached(1);

            slot.push_chunk(chunk_window_uniform(ref0, 1, 0));
            slot.push_chunk(chunk_window_uniform(ref1, 1, 0));
            // block_count() = chunks.len() = 2.
            assert_eq!(slot.block_count(), 2);
            assert!(slot.chunk_at(0).is_some());
            assert!(slot.chunk_at(1).is_some());
            assert!(slot.chunk_at(2).is_none());
        }

        #[test]
        fn test_slot_state_clone() {
            let mut slot = crate::kv_cache::chunked::SequenceState::new();
            let ref0 = ChunkGid::detached(42);
            slot.push_chunk(chunk_window_uniform(ref0, 1, 0));
            // block_count() = 1.

            let cloned = slot.clone();

            assert_eq!(cloned.block_count(), 1);
            assert_eq!(cloned.block_count(), 1);
            assert!(cloned.chunk_at(0).is_some());
            assert_eq!(cloned.chunk_at(0).unwrap().gids[0].raw(), 42);
        }
    }

    // ==================== ChunkedState Tests ====================

    mod chunked_state_tests {
        use super::*;

        #[test]
        fn test_chunked_state_creation() {
            let state = create_test_state(4, 8);

            assert_eq!(state.sequences.len(), 4);
            assert_eq!(state.max_blocks, 8);
            assert!(state.sequences.iter().all(|s| s.is_none()));
        }

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn test_chunked_state_slot_gid_via_chunks() {
            let mut state = create_test_state(4, 8);

            // No slots allocated — sequences should be None
            assert!(state.sequences[0].is_none());
            assert!(state.sequences[2].is_none());

            // Allocate slot 2, push 4 committed chunks (indices 0..3), then check block 3
            state.sequences[2] = Some(crate::kv_cache::chunked::SequenceState::new());
            let slot = state.sequences[2].as_mut().unwrap();
            for i in 0..4 {
                slot.push_chunk(chunk_window_uniform(ChunkGid::detached(i * 10), 1, 0));
            }
            // Replace GID at block 3
            let gid = ChunkGid::detached(42);
            slot.set_block_gids(
                3,
                crate::kv_cache::chunked::head_gids::HeadGids::uniform(gid, 1),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            );

            assert_eq!(
                state.sequences[2]
                    .as_ref()
                    .unwrap()
                    .chunk_at(3)
                    .unwrap()
                    .gids[0]
                    .raw(),
                42
            );
            // Other blocks should still have their original GIDs
            assert!(state.sequences[2].as_ref().unwrap().chunk_at(2).is_some());
            // Block beyond allocated range should be None
            assert!(state.sequences[2].as_ref().unwrap().chunk_at(5).is_none());
        }

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn test_chunked_state_slot_allocation() {
            let mut state = create_test_state(4, 8);

            // Allocate slot 1
            state.sequences[1] = Some(crate::kv_cache::chunked::SequenceState::new());

            assert!(state.sequences[0].is_none());
            assert!(state.sequences[1].is_some());
            assert!(state.sequences[2].is_none());
            assert!(state.sequences[3].is_none());
        }

        #[test]
        fn test_chunked_state_global_allocation() {
            let _state = create_test_state(2, 4);
            let pool = ChunkGidPool::new();

            // Simulate allocating chunks using GidPool convenience method
            let gid0 = pool.allocate();
            let gid1 = pool.allocate();

            assert_eq!(gid0.raw(), 0);
            assert_eq!(gid1.raw(), 1);
        }

        #[test]
        fn test_chunked_state_global_free_list() {
            use crate::kv_cache::chunked::arena::ArenaKey;

            let _state = create_test_state(2, 4);
            let pool = ChunkGidPool::new();
            let key = ArenaKey::gpu_float(candle::DType::BF16);

            // Register arena and inject a few extra raw IDs beyond the first arena.
            pool.register_arena(key.clone());
            let base = crate::kv_cache::arena_gid_stride();
            let initial_free = pool.free_list_len_for(key.clone());

            // Free 3 additional IDs beyond the arena range
            pool.free(key.clone(), (base + 10) as i64);
            pool.free(key.clone(), (base + 5) as i64);
            pool.free(key.clone(), (base + 15) as i64);

            assert_eq!(pool.free_list_len_for(key.clone()), initial_free + 3);
            // allocate_for should return lowest first (0 from registered arena)
            let gid = pool.allocate_for(key.clone()).unwrap();
            assert_eq!(gid.raw(), 0);
        }

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn test_chunked_state_full_workflow() {
            let mut state = create_test_state(2, 4);
            let pool = ChunkGidPool::new();

            // Allocate slot 0
            state.sequences[0] = Some(crate::kv_cache::chunked::SequenceState::new());

            // Allocate 2 chunks for slot 0
            let gid0 = pool.allocate();
            let gid1 = pool.allocate();

            // Push first as committed, second as active
            let slot = state.sequences[0].as_mut().unwrap();
            slot.push_chunk(chunk_window_uniform(gid0, 1, 0));
            slot.push_chunk(chunk_window_uniform(gid1, 1, 0));

            // Verify via chunk_at
            let slot = state.sequences[0].as_ref().unwrap();
            assert_eq!(slot.chunk_at(0).unwrap().gids[0].raw(), 0);
            assert_eq!(slot.chunk_at(1).unwrap().gids[0].raw(), 1);
            assert!(slot.chunk_at(2).is_none());
            assert_eq!(slot.block_count(), 2);
        }
    }

    // ==================== ChunkMeta rope_pos Tests ====================
    // rope_position was moved from ChunkGid to ChunkMeta (the packed metadata
    // struct passed to CUDA kernels). These tests verify ChunkMeta correctness.

    mod chunk_gid_rope_shift {
        use super::*;

        #[test]
        fn test_chunk_meta_zero_rope_pos() {
            let meta = ChunkMeta::new(8, 0, 0u16);
            assert_eq!(meta.rope_base(), 0);
        }

        #[test]
        fn test_chunk_meta_positive_rope_pos() {
            let meta = ChunkMeta::new(8, 100, 0u16);
            assert_eq!(meta.rope_base(), 100);
        }

        #[test]
        fn test_chunk_meta_negative_rope_pos() {
            let meta = ChunkMeta::new(8, -42, 0u16);
            assert_eq!(meta.rope_base(), -42);
        }

        #[test]
        fn test_chunk_meta_zero_constructor() {
            let meta = ChunkMeta::new(8, 0, 0u16);
            assert_eq!(meta.rope_base(), 0);
        }
    }
}
