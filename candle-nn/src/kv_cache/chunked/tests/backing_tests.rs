//! Tests for ChunkedKvBacking and BackingInner.

use candle::{DType, Device};

use crate::kv_cache::chunked::ChunkedKvBacking;
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

        /// **Audit A12's decision input.** The histogram must count every live
        /// band into the class its *format* maps to, and separately count the
        /// ones narrower than the smallest rung — those are the only slots a
        /// {64, 160, 320} split would save anything on.
        #[test]
        fn class_histogram_counts_bands_by_their_format() {
            use candle::Tensor;

            let backing = ChunkedKvBacking::new(2, 4, 128, DType::BF16, &Device::Cpu, 64).unwrap();
            backing.alloc_sequence().unwrap();
            let k = Tensor::ones((1, 4, 32, 128), DType::BF16, &Device::Cpu).unwrap();
            let v = k.clone();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            let (per_class, narrow) = backing.class_histogram(0);
            let total: usize = per_class.iter().sum();
            assert_eq!(
                total,
                4 * crate::kv_cache::arena_table::N_PALETTE * 2,
                "every band of the written chunk must be counted exactly once"
            );
            // A CPU backing writes float bands: BF16 is 2048 B at this
            // geometry, so everything lands on the 2048 rung and nothing is
            // narrower than the 320 B floor. The rung is looked up rather than
            // written as an index, so a ladder edit moves it automatically.
            // head_dim 128 / N_PALETTE 4 = 32 dims per band, x CHUNK_SIZE 32.
            let bf16 = crate::kv_cache::chunked::size_class::class_for_format(
                KvFormat::Float(DType::BF16),
                32 * 32,
            )
            .unwrap();
            assert_eq!(bf16.bytes(), 2048);
            assert_eq!(per_class[bf16.index()], total);
            assert_eq!(narrow, 0, "float bands are never sub-320");
        }

        /// An unallocated slot has no bands, and asking is not an error.
        #[test]
        fn class_histogram_of_an_empty_slot_is_empty() {
            let backing = ChunkedKvBacking::new(2, 4, 128, DType::BF16, &Device::Cpu, 64).unwrap();
            let (per_class, narrow) = backing.class_histogram(0);
            assert_eq!(per_class.iter().sum::<usize>(), 0);
            assert_eq!(narrow, 0);
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

    // ==================== Arena Sweep Tests ====================

    mod sweep_tests {
        use super::*;

        #[test]
        fn sweeping_an_empty_backing_releases_nothing() {
            let backing = ChunkedKvBacking::new(2, 4, 32, DType::BF16, &Device::Cpu, 64).unwrap();

            let freed = backing.release_empty_arenas().unwrap();
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
            let result = backing.ensure_for_offsets(&offsets, &[8, 8]);
            assert!(result.is_err());
        }

        #[test]
        fn test_ensure_for_offsets_correct_length() {
            let backing = create_test_backing();

            let offsets = vec![0, 0, 0, 0]; // Correct: 4 offsets for batch=4
            backing.ensure_for_offsets(&offsets, &[0, 0, 0, 0]).unwrap();
        }
    }
    // ============ Band-format tag propagation through window mutation ========

    /// A `ChunkWindow`'s `k_fmt`/`v_fmt` describe how to read the bytes its
    /// gids point at, so any mutation that re-points the gids must replace the
    /// tags with them.
    ///
    /// This is the cold-load / warm-elevate hazard: a window created by
    /// `alloc_block_chunks` carries the *active* formats (R16 K, F16 V on GPU),
    /// and `set_block_gids` then re-points it at sealed chunks in whatever
    /// formats were persisted. Leaving the old tags in place makes every reader
    /// decode those chunks as raw floats. The model gate cannot catch it —
    /// audit A2 lists cold load as one of its blind spots.
    mod band_tag_propagation_tests {
        use super::*;
        use crate::kv_cache::arena_table::N_PALETTE;
        use crate::kv_cache::QuantFormat;

        #[test]
        fn alloc_sealed_block_stamps_the_destination_formats_on_the_window() {
            let backing = ChunkedKvBacking::new(1, 2, 32, DType::BF16, &Device::Cpu, 64).unwrap();
            backing.alloc_sequence().unwrap();
            let n_kv_head = backing.n_kv_head();
            let want = n_kv_head * N_PALETTE;

            // Deliberately NOT the active format: if the window keeps what
            // `alloc_block_chunks` stamped, these assertions fail.
            let k_formats: Vec<KvFormat> = (0..want)
                .map(|i| {
                    KvFormat::Quantized(if i % 2 == 0 {
                        QuantFormat::Q8_0
                    } else {
                        QuantFormat::Q4_0
                    })
                })
                .collect();
            let v_formats: Vec<KvFormat> = (0..want)
                .map(|_| KvFormat::Quantized(QuantFormat::Q4_0))
                .collect();

            backing
                .alloc_sealed_block(
                    0,
                    0,
                    &k_formats,
                    &v_formats,
                    std::sync::Arc::new(Vec::new()),
                    std::sync::Arc::new(Vec::new()),
                    std::sync::Arc::new(Vec::new()),
                    std::sync::Arc::new(Vec::new()),
                )
                .unwrap();

            let sealed = backing
                .live_chunks_as_sealed(0)
                .expect("slot should hold the sealed block");
            let chunk = sealed.first().expect("one block");
            let want_k: Vec<u8> = k_formats.iter().map(|f| f.to_tag()).collect();
            let want_v: Vec<u8> = v_formats.iter().map(|f| f.to_tag()).collect();
            assert_eq!(
                chunk.k_fmt.as_slice(),
                want_k.as_slice(),
                "K tags must be the destination formats, not the active ones"
            );
            assert_eq!(chunk.v_fmt.as_slice(), want_v.as_slice());

            // And the tags are exactly what the persist path will write.
            let (k, v) = chunk.format_tags().expect("tags recorded");
            assert_eq!(k, want_k.as_slice());
            assert_eq!(v, want_v.as_slice());
        }
    }
}
