//! Tests for chunk operations: migrate, copy, convert, prepare, reconcile.

use candle::{DType, Device, Tensor};

use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::chunked::{ArenaKey, ChunkedKvBacking, StoragePolicy};
use crate::kv_cache::KvFormat;

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
            64,
        )
        .unwrap()
    }

    fn setup_sequence_with_data(backing: &ChunkedKvBacking, tokens: usize) {
        backing.alloc_sequence().unwrap();
        let k = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
        let v = Tensor::ones((1, 4, tokens, 32), DType::BF16, &Device::Cpu).unwrap();
        backing.write_contiguous(0, 0, &k, &v).unwrap();
    }

    // ==================== migrate_chunk Tests ====================

    mod migrate_chunk_tests {
        use super::*;

        #[test]
        fn test_migrate_chunk_same_format() {
            let backing = create_test_backing();
            setup_sequence_with_data(&backing, 8);

            // Get chunk ID from block table
            let source_gid = k_gid_snapshot(&backing)[0][0];

            // Migrate to same format (should copy)
            let target_key = ArenaKey::cpu_float(DType::BF16);
            let new_gid = backing.migrate_chunk(source_gid, target_key).unwrap();

            // Should be a different chunk
            assert_ne!(source_gid, new_gid.raw());
        }

        #[test]
        fn test_migrate_chunk_invalid_id() {
            let backing = create_test_backing();

            let result = backing.migrate_chunk(-1, ArenaKey::cpu_float(DType::BF16));
            assert!(result.is_err());
        }

        #[test]
        fn test_migrate_chunk_out_of_range() {
            let backing = create_test_backing();
            setup_sequence_with_data(&backing, 8);

            // Try to migrate non-existent chunk
            let result = backing.migrate_chunk(9999, ArenaKey::cpu_float(DType::BF16));
            assert!(result.is_err());
        }
    }

    // ==================== StoragePolicy Tests ====================

    mod storage_policy_tests {
        use super::*;

        #[test]
        fn test_storage_policy_to_arena_key_gpu_float() {
            let policy = StoragePolicy::GpuFloat(DType::BF16);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Float(DType::BF16));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }

        #[test]
        fn test_storage_policy_to_arena_key_cpu_float() {
            let policy = StoragePolicy::CpuFloat(DType::F32);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Float(DType::F32));
            assert_eq!(key.location, ArenaLocation::Cpu);
        }

        #[test]
        fn test_storage_policy_active_dtype() {
            // Float policies return their dtype
            assert_eq!(
                StoragePolicy::GpuFloat(DType::BF16).active_dtype(),
                DType::BF16
            );
            assert_eq!(
                StoragePolicy::CpuFloat(DType::F32).active_dtype(),
                DType::F32
            );
        }
    }

    // ==================== reconcile Tests ====================

    // ==================== ArenaKey Tests ====================

    mod arena_key_tests {
        use super::*;

        #[test]
        fn test_arena_key_constructors() {
            let gpu_float = ArenaKey::gpu_float(DType::BF16);
            assert!(gpu_float.is_gpu());
            assert!(!gpu_float.is_quantized());

            let cpu_float = ArenaKey::cpu_float(DType::F32);
            assert!(!cpu_float.is_gpu());
            assert!(!cpu_float.is_quantized());
        }

        #[test]
        fn test_arena_key_equality() {
            let key1 = ArenaKey::cpu_float(DType::BF16);
            let key2 = ArenaKey::cpu_float(DType::BF16);
            let key3 = ArenaKey::cpu_float(DType::F32);
            let key4 = ArenaKey::gpu_float(DType::BF16);

            assert_eq!(key1, key2);
            assert_ne!(key1, key3); // Different dtype
            assert_ne!(key1, key4); // Different location
        }

        #[test]
        fn test_arena_key_new() {
            let key = ArenaKey::uniform(KvFormat::Float(DType::F16), ArenaLocation::Gpu);

            assert_eq!(key.format, KvFormat::Float(DType::F16));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }
    }

    // ==================== Integration Tests ====================

    mod integration_tests {
        use super::*;

        #[test]
        fn test_migrate_preserves_data_integrity() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Write specific test data
            let k = Tensor::arange(0.0f32, 1024.0, &Device::Cpu)
                .unwrap()
                .reshape((1, 4, 8, 32))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let v = Tensor::arange(1024.0f32, 2048.0, &Device::Cpu)
                .unwrap()
                .reshape((1, 4, 8, 32))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            backing.write_contiguous(0, 0, &k, &v).unwrap();
            backing.set_len(0, 8);

            // Read data before migration
            let (k_before, _v_before) = backing.read_contiguous(0, 0, 8).unwrap();

            // Get chunk ID and migrate
            let source_gid = k_gid_snapshot(&backing)[0][0];

            // Migrate (same format, should copy)
            let _new_gid = backing
                .migrate_chunk(source_gid, ArenaKey::cpu_float(DType::BF16))
                .unwrap();

            // Data from original chunk should still be readable
            let (k_after, _v_after) = backing.read_contiguous(0, 0, 8).unwrap();

            // Convert to F32 for comparison
            let k_before_f32 = k_before.to_dtype(DType::F32).unwrap();
            let k_after_f32 = k_after.to_dtype(DType::F32).unwrap();

            let diff = (&k_before_f32 - &k_after_f32)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();

            assert!(diff < 0.01, "Data changed after migration: diff={}", diff);
        }
    }
}
