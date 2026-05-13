//! Tests for ChunkedKvBacking and BackingInner.

use candle::{DType, Device};

use crate::kv_cache::chunked::ChunkedKvBacking;
use crate::kv_cache::KvFormat;

/// Test helper: snapshot K-side GIDs as `[batch][block]` with -1 for unallocated.
fn k_gid_snapshot(backing: &ChunkedKvBacking) -> Vec<Vec<i64>> {
    let state = backing.state.read().expect("lock");
    let mb = state.max_blocks;
    state.sequences.iter().map(|slot| {
        let mut row = vec![-1i64; mb];
        if let Some(s) = slot {
            for (i, cw) in s.chunks_slice().iter().enumerate() {
                row[i] = cw.gids.k_gid(0).raw();
            }
        }
        row
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ChunkedKvBacking Constructor Tests ====================

    mod constructor_tests {
        use super::*;

        #[test]
        fn test_new_basic() {
            let backing = ChunkedKvBacking::new(
                4,  // initial_batch
                8,  // n_kv_head
                64, // head_dim
                DType::BF16,
                &Device::Cpu,
                256, // initial_max_seq_len
            )
            .unwrap();

            assert_eq!(backing.batch_capacity(), 4);
            assert_eq!(backing.n_kv_head(), 8);
            assert_eq!(backing.head_dim(), 64);
            assert!(!backing.is_quantized());
        }

        #[test]
        fn test_new_with_format_float() {
            let backing = ChunkedKvBacking::new_with_format(
                2,
                4,
                32,
                KvFormat::Float(DType::F32),
                KvFormat::Float(DType::F32),
                &Device::Cpu,
                64,
            )
            .unwrap();

            assert_eq!(backing.k_format(), KvFormat::Float(DType::F32));
            assert!(!backing.is_quantized());
        }

        #[test]
        fn test_new_with_format_quantized() {
            use crate::kv_cache::QuantFormat;

            // For Q4_0, block_size=32, so head_dim and chunk_size must be multiples of 32
            let result = ChunkedKvBacking::new_with_format(
                2,
                4,
                64, // head_dim divisible by 32
                KvFormat::Quantized(QuantFormat::Q4_0),
                KvFormat::Quantized(QuantFormat::Q4_0),
                &Device::Cpu,
                64,
            );

            assert!(result.is_ok());
            let backing = result.unwrap();
            assert!(backing.is_quantized());
            assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q4_0));
        }

        #[test]
        fn test_new_zero_batch_fails() {
            let result = ChunkedKvBacking::new(
                0, // zero batch should fail
                8,
                64,
                DType::BF16,
                &Device::Cpu,
                256,
            );

            assert!(result.is_err());
        }

        #[test]
        fn test_new_quantized_misaligned_head_dim_fails() {
            use crate::kv_cache::QuantFormat;

            // head_dim=48 is not divisible by block_size=32
            let result = ChunkedKvBacking::new_with_format(
                2,
                4,
                48, // not divisible by 32
                KvFormat::Quantized(QuantFormat::Q4_0),
                KvFormat::Quantized(QuantFormat::Q4_0),
                &Device::Cpu,
                64,
            );

            assert!(result.is_err());
        }
    }

    // ==================== Batch Capacity Tests ====================

    mod batch_capacity_tests {
        use super::*;

        #[test]
        fn test_grow_batch_capacity() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            assert_eq!(backing.batch_capacity(), 2);

            backing.grow_batch_capacity(8).unwrap();
            assert_eq!(backing.batch_capacity(), 8);
        }

        #[test]
        fn test_grow_batch_capacity_no_shrink() {
            let backing = ChunkedKvBacking::new(8, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            assert_eq!(backing.batch_capacity(), 8);

            // Growing to smaller size should be a no-op
            backing.grow_batch_capacity(4).unwrap();
            assert_eq!(backing.batch_capacity(), 8);
        }

        #[test]
        fn test_grow_batch_capacity_same_size() {
            let backing = ChunkedKvBacking::new(4, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            backing.grow_batch_capacity(4).unwrap();
            assert_eq!(backing.batch_capacity(), 4);
        }
    }

    // ==================== Arena Access Tests ====================

    mod arena_tests {
        use super::*;

        #[test]
        fn test_k_arenas_empty_initially() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            let k_arenas = backing.k_arenas();
            assert!(k_arenas.is_empty());
        }

        #[test]
        fn test_v_arenas_empty_initially() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            let v_arenas = backing.v_arenas();
            assert!(v_arenas.is_empty());
        }

        #[test]
        fn test_float_arenas_empty_initially() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            let float_arenas = backing.float_arenas();
            assert!(float_arenas.is_some());
            let (k, v) = float_arenas.unwrap();
            assert!(k.is_empty());
            assert!(v.is_empty());
        }

        #[test]
        fn test_quantized_arenas_returns_none_for_float() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            assert!(backing.quantized_arenas().is_none());
        }

        #[test]
        fn test_arena_count_initially_zero() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            assert_eq!(backing.arena_count().unwrap(), 0);
        }
    }

    // ==================== Block Table Tests ====================

    mod block_table_tests {
        use super::*;

        #[test]
        fn test_block_table_initial_shape() {
            let backing = ChunkedKvBacking::new(
                4, // batch
                8,
                64,
                DType::BF16,
                &Device::Cpu,
                256, // initial_max_seq_len
            )
            .unwrap();

            let snap = k_gid_snapshot(&backing);
            assert_eq!(snap.len(), 4); // batch
            // max_blocks = ceil(256/32) = 8
            assert_eq!(snap[0].len(), 8);
        }

        #[test]
        fn test_block_table_initial_values() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 32).unwrap();

            let snap = k_gid_snapshot(&backing);
            // All entries should be -1 initially (unallocated)
            for row in &snap {
                for &val in row {
                    assert_eq!(val, -1);
                }
            }
        }
    }

    // ==================== Compact Tests ====================

    mod compact_tests {
        use super::*;

        #[test]
        fn test_compact_empty() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            // Compacting when empty should free 0
            let freed = backing.compact().unwrap();
            assert_eq!(freed, 0);
        }
    }

    // ==================== Clone/Share Tests ====================

    mod clone_tests {
        use super::*;

        #[test]
        fn test_backing_clone_shares_inner() {
            let backing1 = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            let backing2 = backing1.clone();

            // Both should share the same inner state
            assert_eq!(backing1.batch_capacity(), backing2.batch_capacity());

            // Growing one should affect the other
            backing1.grow_batch_capacity(8).unwrap();
            assert_eq!(backing2.batch_capacity(), 8);
        }
    }

    // ==================== Ensure For Offset Tests ====================

    mod ensure_offset_tests {
        use super::*;

        fn create_test_backing() -> ChunkedKvBacking {
            ChunkedKvBacking::new(
                4,  // batch
                4,  // n_kv_head
                32, // head_dim
                DType::BF16,
                &Device::Cpu,
                256, // max_seq_len → max_blocks = ceil(256/32) = 8
            )
            .unwrap()
        }

        #[test]
        fn test_ensure_for_offset_zero_add() {
            let backing = create_test_backing();

            // Adding 0 tokens should be a no-op
            backing.ensure_for_offset(0, 0, 0).unwrap();

            let snap = k_gid_snapshot(&backing);
            // All should still be -1
            for &val in &snap[0] {
                assert_eq!(val, -1);
            }
        }

        #[test]
        fn test_ensure_for_offset_out_of_range() {
            let backing = create_test_backing();

            // batch_idx=10 is out of range for capacity=4
            let result = backing.ensure_for_offset(10, 0, 16);
            assert!(result.is_err());
        }

        #[test]
        fn test_ensure_for_offset_allocates_chunks() {
            let backing = create_test_backing();

            // chunk_size=32, so 64 tokens needs 2 chunks
            backing.ensure_for_offset(0, 0, 64).unwrap();

            let row0 = k_gid_snapshot(&backing)[0].clone();

            // First 2 chunks should be allocated
            assert!(row0[0] >= 0);
            assert!(row0[1] >= 0);
        }

        #[test]
        fn test_ensure_for_offsets_wrong_length() {
            let backing = create_test_backing();

            // Providing wrong number of offsets should fail
            let offsets = vec![0, 0]; // Only 2, but batch=4
            let result = backing.ensure_for_offsets(&offsets, 8);
            assert!(result.is_err());
        }

        #[test]
        fn test_ensure_for_offsets_correct_length() {
            let backing = create_test_backing();

            let offsets = vec![0, 0, 0, 0]; // Correct: 4 offsets for batch=4
            backing.ensure_for_offsets(&offsets, 0).unwrap();
        }
    }
}
