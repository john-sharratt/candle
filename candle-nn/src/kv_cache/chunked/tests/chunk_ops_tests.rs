//! Tests for chunk operations: migrate, copy, convert, prepare, reconcile.

use candle::{DType, Device, Tensor};

use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::chunked::{ArenaKey, ChunkedKvBacking};

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

        /// The CPU key a gid's slot relocates into: same size class, warm tier.
        ///
        /// A relocation never changes a chunk's class — that is what makes the
        /// byte-verbatim copy safe — so the target is always derived from the
        /// source rather than named by a format.
        pub(super) fn cpu_key_of(backing: &ChunkedKvBacking, raw: i64) -> ArenaKey {
            use crate::kv_cache::chunked::GID_STRIDE;
            let key = backing
                .with_arenas(|a| a.get(&((raw as usize) / GID_STRIDE)).map(|a| a.arena_key()))
                .unwrap()
                .expect("source arena exists");
            ArenaKey::new(key.class, ArenaLocation::Cpu)
        }

        #[test]
        fn test_migrate_chunk_relocates_within_its_class() {
            let backing = create_test_backing();
            setup_sequence_with_data(&backing, 8);

            // Get chunk ID from block table
            let source_gid = k_gid_snapshot(&backing)[0][0];

            // Relocate to the warm tier — same class, so a byte copy.
            let target_key = cpu_key_of(&backing, source_gid);
            let new_gid = backing.migrate_chunk(source_gid, target_key).unwrap();

            // Should be a different chunk
            assert_ne!(source_gid, new_gid.raw());
        }

        #[test]
        fn test_migrate_chunk_invalid_id() {
            let backing = create_test_backing();

            let any = ArenaKey::new(
                crate::kv_cache::chunked::SizeClass::at(0),
                ArenaLocation::Cpu,
            );
            let result = backing.migrate_chunk(-1, any);
            assert!(result.is_err());
        }

        #[test]
        fn test_migrate_chunk_out_of_range() {
            let backing = create_test_backing();
            setup_sequence_with_data(&backing, 8);

            // Try to migrate non-existent chunk
            let any = ArenaKey::new(
                crate::kv_cache::chunked::SizeClass::at(0),
                ArenaLocation::Cpu,
            );
            let result = backing.migrate_chunk(9999, any);
            assert!(result.is_err());
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
            let target = migrate_chunk_tests::cpu_key_of(&backing, source_gid);
            let _new_gid = backing.migrate_chunk(source_gid, target).unwrap();

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
