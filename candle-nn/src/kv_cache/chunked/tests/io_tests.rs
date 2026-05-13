//! Tests for I/O operations: read_contiguous and write_contiguous.

use candle::{DType, Device, Tensor};

use crate::kv_cache::chunked::ChunkedKvBacking;

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

    fn create_test_backing() -> ChunkedKvBacking {
        ChunkedKvBacking::new(
            4,  // initial_batch
            4,  // n_kv_head
            32, // head_dim
            DType::BF16,
            &Device::Cpu,
            256, // max_seq_len → 256 / CHUNK_SIZE(32) = 8 max_blocks
        )
        .unwrap()
    }

    // ==================== read_contiguous Tests ====================

    mod read_contiguous_tests {
        use super::*;

        #[test]
        fn test_read_contiguous_after_write() {
            let backing = create_test_backing();

            // Allocate sequence and write some data
            backing.alloc_sequence().unwrap();

            let n_kv_head = 4;
            let head_dim = 32;
            let len = 32;

            // Create test K/V data: shape (1, n_kv_head, len, head_dim)
            let k = Tensor::randn(0.0f32, 1.0, (1, n_kv_head, len, head_dim), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let v = Tensor::randn(0.0f32, 1.0, (1, n_kv_head, len, head_dim), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            // Write at offset 0
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Read back
            let (k_read, v_read) = backing.read_contiguous(0, 0, len).unwrap();

            // Shape should match
            assert_eq!(k_read.dims(), &[1, n_kv_head, len, head_dim]);
            assert_eq!(v_read.dims(), &[1, n_kv_head, len, head_dim]);
        }

        #[test]
        fn test_read_contiguous_partial() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Write 64 tokens (2 full chunks)
            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Read only 32 tokens (1 chunk)
            let (k_read, v_read) = backing.read_contiguous(0, 0, 32).unwrap();

            assert_eq!(k_read.dims(), &[1, 4, 32, 32]);
            assert_eq!(v_read.dims(), &[1, 4, 32, 32]);
        }

        #[test]
        fn test_read_contiguous_with_offset() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Write 64 tokens (2 full chunks)
            let k = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Read 16 tokens starting at offset 16
            let (k_read, v_read) = backing.read_contiguous(0, 16, 16).unwrap();

            assert_eq!(k_read.dims(), &[1, 4, 16, 32]);
            assert_eq!(v_read.dims(), &[1, 4, 16, 32]);
        }

        #[test]
        fn test_read_contiguous_unallocated_fails() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Try to read without writing (no chunks allocated)
            let result = backing.read_contiguous(0, 0, 32);
            assert!(result.is_err());
        }
    }

    // ==================== write_contiguous Tests ====================

    mod write_contiguous_tests {
        use super::*;

        #[test]
        fn test_write_contiguous_basic() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_ok());
        }

        #[test]
        fn test_write_contiguous_zero_length() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Zero-length write should be a no-op
            let k = Tensor::ones((1, 4, 0, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 0, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_ok());
        }

        #[test]
        fn test_write_contiguous_out_of_range() {
            let backing = create_test_backing();

            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();

            // Batch idx 10 is out of range
            let result = backing.write_contiguous(10, 0, &k, &v);
            assert!(result.is_err());
        }

        #[test]
        fn test_write_contiguous_wrong_batch_dim() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Wrong batch dim (2 instead of 1)
            let k = Tensor::ones((2, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((2, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_err());
        }

        #[test]
        fn test_write_contiguous_wrong_head_count() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Wrong head count (8 instead of 4)
            let k = Tensor::ones((1, 8, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 8, 32, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_err());
        }

        #[test]
        fn test_write_contiguous_wrong_head_dim() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Wrong head dim (64 instead of 32)
            let k = Tensor::ones((1, 4, 32, 64), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 64), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_err());
        }

        #[test]
        fn test_write_contiguous_kv_shape_mismatch() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // K and V have different lengths
            let k = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 64, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_err());
        }

        #[test]
        fn test_write_contiguous_dtype_conversion() {
            let backing = create_test_backing(); // BF16 storage
            backing.alloc_sequence().unwrap();

            // Write F32 data - should be converted to BF16
            let k = Tensor::ones((1, 4, 32, 32), DType::F32, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 32, 32), DType::F32, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_ok());
        }

        #[test]
        fn test_write_contiguous_multi_chunk() {
            let backing = create_test_backing(); // CHUNK_SIZE=32
            backing.alloc_sequence().unwrap();

            // Write 96 tokens (spans 3 chunks of 32)
            let k = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();
            let v = Tensor::ones((1, 4, 96, 32), DType::BF16, &Device::Cpu).unwrap();

            let result = backing.write_contiguous(0, 0, &k, &v);
            assert!(result.is_ok());

            // Verify block table has 3 chunks allocated
            let row = k_gid_snapshot(&backing)[0].clone();
            assert!(row[0] >= 0);
            assert!(row[1] >= 0);
            assert!(row[2] >= 0);
        }

        #[test]
        fn test_write_contiguous_incremental() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Write first 32 tokens (1 full chunk)
            let k1 = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            let v1 = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap();
            backing.write_contiguous(0, 0, &k1, &v1).unwrap();

            // Write next 32 tokens at offset 32 (fills second chunk)
            let k2 = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap() * 2.0;
            let v2 = Tensor::ones((1, 4, 32, 32), DType::BF16, &Device::Cpu).unwrap() * 2.0;
            backing
                .write_contiguous(0, 32, &k2.unwrap(), &v2.unwrap())
                .unwrap();

            // Should have 2 chunks now
            let row = k_gid_snapshot(&backing)[0].clone();
            assert!(row[0] >= 0);
            assert!(row[1] >= 0);
        }
    }

    // ==================== Roundtrip Tests ====================

    mod roundtrip_tests {
        use super::*;

        #[test]
        fn test_write_read_roundtrip() {
            let backing = create_test_backing();
            backing.alloc_sequence().unwrap();

            // Create deterministic test data
            // 1 * 4 * 32 * 32 = 4096 elements
            let k = Tensor::arange(0.0f32, 4096.0, &Device::Cpu)
                .unwrap()
                .reshape((1, 4, 32, 32))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let v = Tensor::arange(4096.0f32, 8192.0, &Device::Cpu)
                .unwrap()
                .reshape((1, 4, 32, 32))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            // Write
            backing.write_contiguous(0, 0, &k, &v).unwrap();

            // Read back
            let (k_read, _v_read) = backing.read_contiguous(0, 0, 32).unwrap();

            // Convert to F32 for comparison and flatten
            let k_f32 = k.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
            let k_read_f32 = k_read.to_dtype(DType::F32).unwrap().flatten_all().unwrap();

            // Check values are approximately equal (BF16 has limited precision)
            let diff = (&k_f32 - &k_read_f32)
                .unwrap()
                .abs()
                .unwrap()
                .max(0) // max over the single flattened dimension
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();

            // Allow small tolerance for BF16 precision
            assert!(diff < 0.1, "K roundtrip error too large: {}", diff);
        }
    }
}
