//! Tests forking sequences in the chunked and scattered caches, sharing their full
//! blocks and copying their partial tails.
//!
//! A fork shares every full block by id, so the prefix is stored once, and copies
//! the partial remainder so the two sequences can diverge without touching each
//! other's data. The tests check the block counts and ids for full, partial and
//! zero-length forks, that appends to one fork leave the other unchanged, that a fork
//! into an occupied target replaces what was there, and that forking to self or past
//! the source's length is refused. Multiple forks of one source all share the same
//! full block.

use super::*;

#[test]
pub(super) fn test_kvcache_fork_chunked() -> Result<()> {
    use crate::kv_cache::cache::CacheStorage;

    let backing = ChunkedKvBacking::new(4, 4, 64, DType::F32, &Device::Cpu, 256)?;

    let cache1 = make_chunked_kvcache(&backing)?;

    // Set up seq len (simulating data was written)
    cache1.k_cache().set_chunked_len(50);
    cache1.v_cache().set_chunked_len(50);

    // Ensure capacity
    backing.ensure_for_offset(0, 0, 50)?;

    // Fork
    let cache2 = cache1.fork()?;

    // Should have different batch indices
    let idx1 = match &cache1.k_cache().storage {
        CacheStorage::Chunked(c) => c.batch_idx,
        _ => panic!("expected chunked"),
    };
    let idx2 = match &cache2.k_cache().storage {
        CacheStorage::Chunked(c) => c.batch_idx,
        _ => panic!("expected chunked"),
    };
    assert_ne!(idx1, idx2);

    Ok(())
}

#[test]
pub(super) fn test_kvcache_fork_contiguous() -> Result<()> {
    let mut cache1 = KvCache::new(2, 256);
    let device = Device::Cpu;

    // Append some data
    let k = Tensor::ones((1, 4, 10, 64), DType::F32, &device)?;
    let v = Tensor::ones((1, 4, 10, 64), DType::F32, &device)?;
    cache1.append(&k, &v)?;

    assert_eq!(cache1.current_seq_len(), 10);

    // Fork
    let cache2 = cache1.fork()?;

    // Both should have same seq len
    assert_eq!(cache2.current_seq_len(), 10);

    // Append more to cache2 shouldn't affect cache1
    let k2 = Tensor::full(2.0f32, (1, 4, 5, 64), &device)?;
    let v2 = Tensor::full(2.0f32, (1, 4, 5, 64), &device)?;
    let mut cache2 = cache2;
    cache2.append(&k2, &v2)?;

    assert_eq!(cache1.current_seq_len(), 10);
    assert_eq!(cache2.current_seq_len(), 15);

    Ok(())
}

#[test]
pub(super) fn test_cache_fork_chunked_append() -> Result<()> {
    let backing = ChunkedKvBacking::new(4, 4, 64, DType::F32, &Device::Cpu, 256)?;

    // Allocate and write to first sequence
    let batch_idx = backing.alloc_sequence()?;
    backing.ensure_for_offset(batch_idx, 0, 64)?;

    let k = Tensor::full(1.0f32, (1, 4, 64, 64), &Device::Cpu)?;
    let v = Tensor::full(1.0f32, (1, 4, 64, 64), &Device::Cpu)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    // Fork using backing API (64 tokens = 2 full blocks, no partial tail)
    let batch_idx2 = backing.fork_sequence_alloc(batch_idx, 64)?;

    // Both sequences should share the same block IDs
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[batch_idx][0], table[batch_idx2][0]);
    assert_eq!(table[batch_idx][1], table[batch_idx2][1]);

    // Append new tokens to forked sequence (at offset 64, after shared prefix)
    let k2 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    let v2 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    backing.write_contiguous(batch_idx2, 64, &k2, &v2)?;

    // Forked seq should now have 3 blocks (2 shared + 1 new tail)
    assert_eq!(backing.seq_blocks_count(batch_idx2)?, 3);

    // Prefix blocks still shared
    let table2 = k_gid_snapshot(&backing);
    assert_eq!(table2[batch_idx][0], table2[batch_idx2][0]);
    assert_eq!(table2[batch_idx][1], table2[batch_idx2][1]);

    Ok(())
}

#[test]
pub(super) fn test_backing_alloc_free_sequence() -> Result<()> {
    let backing = ChunkedKvBacking::new(4, 4, 64, DType::F32, &Device::Cpu, 256)?;

    // Allocate sequences
    let idx1 = backing.alloc_sequence()?;
    let idx2 = backing.alloc_sequence()?;
    let idx3 = backing.alloc_sequence()?;

    assert_eq!(idx1, 0);
    assert_eq!(idx2, 1);
    assert_eq!(idx3, 2);

    // Free middle one
    backing.free_sequence(idx2)?;

    // Next alloc should reuse it
    let idx4 = backing.alloc_sequence()?;
    assert_eq!(idx4, 1);

    // Allocate one more - this should succeed (idx 3 is free)
    let idx5 = backing.alloc_sequence()?;
    assert_eq!(idx5, 3);

    // Capacity grows dynamically - allocate more to verify
    let idx6 = backing.alloc_sequence()?;
    assert!(idx6 >= 4); // Should have grown capacity
    assert_eq!(backing.batch_capacity(), 8); // Doubled from 4

    Ok(())
}

// ==================== Fork Sequence Tests ====================

#[test]
pub(super) fn test_fork_sequence_basic() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 has 50 tokens = 1 full block (32) + 18 remainder
    backing.ensure_for_offset(0, 0, 50)?;

    // Write some data
    let k = Tensor::full(1.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    let v = Tensor::full(1.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k, &v)?;

    // Fork to seq 1
    let forked_len = backing.fork_sequence(0, 1, 50)?;
    assert_eq!(forked_len, 50);

    // Both should have 2 blocks
    assert_eq!(backing.seq_blocks_count(0)?, 2);
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    // First block (full) should be shared — same GID
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[1][0], "first block should be shared");

    // Second block (partial) should NOT be shared — it was copied at fork time
    assert_ne!(table[0][1], table[1][1], "second block should be copied");

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_full_blocks_only() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 has exactly 64 tokens = 2 full blocks, no remainder
    backing.ensure_for_offset(0, 0, 64)?;

    // Fork to seq 1
    backing.fork_sequence(0, 1, 64)?;

    // Both blocks should be shared (no partial block to copy)
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[1][0]);
    assert_eq!(table[0][1], table[1][1]);

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_single_partial_block() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 has only 16 tokens = 0 full blocks + 16 remainder
    backing.ensure_for_offset(0, 0, 16)?;

    // Fork to seq 1
    backing.fork_sequence(0, 1, 16)?;

    // Single block should be copied (not shared) since it's partial
    let table = k_gid_snapshot(&backing);
    assert_ne!(
        table[0][0], table[1][0],
        "partial block should be copied, not shared"
    );

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_preserves_data() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Write distinct data pattern to seq 0
    backing.ensure_for_offset(0, 0, 50)?;
    let k0 = Tensor::full(42.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    let v0 = Tensor::full(42.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k0, &v0)?;

    // Fork to seq 1
    backing.fork_sequence(0, 1, 50)?;

    // Read back from seq 1's partial block and verify data
    let table = k_gid_snapshot(&backing);

    // Check the copied partial block (block 1)
    let first_val = band_f32(&backing, table[1][1] as usize, 1)?[0];
    assert!((first_val - 42.0).abs() < 0.01, "data should be preserved");

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_append_independence() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 has 50 tokens (1 full + 18 partial)
    backing.ensure_for_offset(0, 0, 50)?;
    let k0 = Tensor::full(1.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    let v0 = Tensor::full(1.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k0, &v0)?;

    // Fork to seq 1 (full block shared, partial tail copied)
    backing.fork_sequence(0, 1, 50)?;
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[1][0], "full block should be shared");
    assert_ne!(table[0][1], table[1][1], "partial tail should be copied");

    // Append to seq 1 — should not affect seq 0's data
    let k1 = Tensor::full(99.0f32, (1, 4, 10, 64), &Device::Cpu)?;
    let v1 = Tensor::full(99.0f32, (1, 4, 10, 64), &Device::Cpu)?;
    backing.write_contiguous(1, 50, &k1, &v1)?;

    // Verify seq 0's partial block data unchanged
    let table = k_gid_snapshot(&backing);
    let first_val = band_f32(&backing, table[0][1] as usize, 1)?[0];
    assert!(
        (first_val - 1.0).abs() < 0.01,
        "seq 0 data should be unchanged"
    );

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_overwrites_target() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Both sequences have independent data
    backing.ensure_for_offset(0, 0, 50)?;
    backing.ensure_for_offset(1, 0, 100)?;

    // Seq 1 had 4 blocks before
    assert_eq!(backing.seq_blocks_count(1)?, 4);

    // Fork seq 0 to seq 1
    backing.fork_sequence(0, 1, 50)?;

    // Seq 1 now has 2 blocks (matching seq 0's 50 tokens)
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_zero_len() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 1 has some data
    backing.ensure_for_offset(1, 0, 64)?;
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    // Fork with zero length should clear target
    backing.fork_sequence(0, 1, 0)?;

    assert_eq!(backing.seq_blocks_count(1)?, 0);

    Ok(())
}

#[test]
pub(super) fn test_fork_sequence_errors() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Cannot fork to self
    backing.ensure_for_offset(0, 0, 64)?;
    assert!(backing.fork_sequence(0, 0, 64).is_err());

    // Cannot fork more than source has
    assert!(backing.fork_sequence(0, 1, 100).is_err()); // source only has 64

    // Out of range batch indices
    assert!(backing.fork_sequence(10, 1, 32).is_err());
    assert!(backing.fork_sequence(0, 10, 32).is_err());

    Ok(())
}

#[test]
pub(super) fn test_fork_multiple_times() -> Result<()> {
    let backing = make_test_backing(4)?;

    // Seq 0 is source with 50 tokens
    backing.ensure_for_offset(0, 0, 50)?;

    // Fork to multiple targets
    backing.fork_sequence(0, 1, 50)?;
    backing.fork_sequence(0, 2, 50)?;
    backing.fork_sequence(0, 3, 50)?;

    // First block should be shared by all 4
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[1][0]);
    assert_eq!(table[0][0], table[2][0]);
    assert_eq!(table[0][0], table[3][0]);

    // Second (partial) block should all be different
    assert_ne!(table[0][1], table[1][1]);
    assert_ne!(table[0][1], table[2][1]);
    assert_ne!(table[0][1], table[3][1]);
    assert_ne!(table[1][1], table[2][1]);
    assert_ne!(table[1][1], table[3][1]);
    assert_ne!(table[2][1], table[3][1]);

    Ok(())
}

// ==================== Original ScatteredKvCache Tests ====================

#[test]
pub(super) fn test_scattered_kv_cache() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = ScatteredCacheBuilder::new(2, 5, DType::F32, &device)?;
    let inf = f32::INFINITY;

    let iam = cache.indices_and_mask(1, &[true, false])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[0], [0]]);
    assert_eq!(
        mask,
        [[[0.0, -inf, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
    );

    let iam = cache.indices_and_mask(1, &[true, false])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[1], [0]]);
    assert_eq!(
        mask,
        [[[0.0, 0.0, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
    );

    let iam = cache.indices_and_mask(3, &[false, true])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[2, 2, 2], [0, 1, 2]]);
    assert_eq!(
        mask,
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [0.0, -inf, -inf, -inf, -inf],
                [0.0, 0.0, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, -inf, -inf]
            ]
        ]
    );

    let iam = cache.indices_and_mask(3, &[true, true])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[2, 3, 4], [3, 4, 0]]);
    assert_eq!(
        mask,
        [
            [
                [0.0, 0.0, 0.0, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [-inf, 0.0, 0.0, 0.0, -inf],
                [-inf, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ]
    );

    let iam = cache.indices_and_mask(1, &[true, false])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[0], [1]]);
    assert_eq!(
        mask,
        [[[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
    );

    let iam = cache.indices_and_mask(2, &[true, false])?;
    let mask = iam.mask().i((.., 0))?.to_vec3::<f32>()?;
    assert_eq!(iam.indices().to_vec2::<u32>()?, [[1, 2], [1, 1]]);
    assert_eq!(
        mask,
        [
            [[0.0, 0.0, -inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]
    );

    Ok(())
}
