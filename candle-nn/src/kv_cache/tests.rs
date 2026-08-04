//! Unit tests for KV cache types.

use super::*;
use candle::{DType, Device, IndexOp, Result, Tensor};

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

// ==================== KvCache Fork Tests ====================

fn make_chunked_kvcache(backing: &ChunkedKvBacking) -> Result<KvCache> {
    let batch_idx = backing.alloc_sequence()?;
    let mut kv = KvCache::new(2, 256); // dim=2 (sequence dimension)
    kv.set_chunked_backing(backing, batch_idx, None)?;
    Ok(kv)
}

#[test]
fn test_kvcache_fork_chunked() -> Result<()> {
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
fn test_kvcache_fork_contiguous() -> Result<()> {
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
fn test_cache_fork_chunked_append() -> Result<()> {
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
fn test_backing_alloc_free_sequence() -> Result<()> {
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

// ==================== ChunkedKvBacking Prefix Sharing Tests ====================

fn make_test_backing(batch: usize) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new(
        batch,
        4,  // n_kv_head
        64, // head_dim
        DType::F32,
        &Device::Cpu,
        256, // initial_max_seq_len
    )
}

#[test]
fn test_share_prefix_basic() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Allocate blocks for sequence 0
    backing.ensure_for_offset(0, 0, 100)?; // needs ceil(100/32) = 4 blocks

    assert_eq!(backing.seq_blocks_count(0)?, 4);
    assert_eq!(backing.seq_blocks_count(1)?, 0);

    // Share 64 tokens (2 blocks) from seq 0 to seq 1
    let shared = backing.share_prefix(0, 1, 64)?;
    assert_eq!(shared, 64); // 2 blocks * 32 tokens

    // Both sequences should now have the same block IDs for blocks 0-1
    let table_data = k_gid_snapshot(&backing);
    assert_eq!(table_data[0][0], table_data[1][0]); // Block 0 shared
    assert_eq!(table_data[0][1], table_data[1][1]); // Block 1 shared

    // Seq 1 should have 2 blocks now
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    Ok(())
}

#[test]
fn test_share_prefix_edge_cases() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Allocate for seq 0
    backing.ensure_for_offset(0, 0, 64)?; // 2 blocks

    // Edge case: prefix_tokens < chunk_size (should share 0 blocks)
    let shared = backing.share_prefix(0, 1, 31)?;
    assert_eq!(shared, 0);
    assert_eq!(backing.seq_blocks_count(1)?, 0);

    // Edge case: prefix_tokens = 0
    let shared = backing.share_prefix(0, 1, 0)?;
    assert_eq!(shared, 0);

    Ok(())
}

#[test]
fn test_share_prefix_errors() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Allocate for seq 0
    backing.ensure_for_offset(0, 0, 32)?; // 1 block

    // Error: share with self
    let result = backing.share_prefix(0, 0, 32);
    assert!(result.is_err());

    // Error: source out of range
    let result = backing.share_prefix(99, 1, 32);
    assert!(result.is_err());

    // Error: target out of range
    let result = backing.share_prefix(0, 99, 32);
    assert!(result.is_err());

    // Error: source doesn't have enough blocks
    let result = backing.share_prefix(0, 1, 128); // Needs 4 blocks, seq 0 has 1
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_free_sequence_with_shared_blocks() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Allocate and share
    backing.ensure_for_offset(0, 0, 64)?;
    backing.share_prefix(0, 1, 64)?;

    // Free seq 0 - shared blocks should NOT be returned to pool
    backing.free_sequence(0)?;

    assert_eq!(backing.seq_blocks_count(0)?, 0);
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    // Free seq 1 - now blocks should be freed
    backing.free_sequence(1)?;
    assert_eq!(backing.seq_blocks_count(1)?, 0);

    Ok(())
}

#[test]
fn test_append_after_share_prefix() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Setup: allocate seq 0 and write 32 tokens (1 full block)
    let k = Tensor::ones((1, 4, 32, 64), DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 4, 32, 64), DType::F32, &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k, &v)?;

    let table_before = k_gid_snapshot(&backing);
    let seq0_block0_before = table_before[0][0];

    // Share prefix
    backing.share_prefix(0, 1, 32)?;

    // Append new tokens to seq 1 (at offset 32, after shared block)
    let k2 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    let v2 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    backing.write_contiguous(1, 32, &k2, &v2)?;

    // Seq 0's block should be unchanged
    let table_after = k_gid_snapshot(&backing);
    assert_eq!(seq0_block0_before, table_after[0][0]);

    // Seq 1 should have 2 blocks: 1 shared + 1 new tail
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    Ok(())
}

#[test]
fn test_share_preserves_source_data() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Write distinct data to seq 0
    let k0 = Tensor::full(1.0f32, (1, 4, 32, 64), &Device::Cpu)?;
    let v0 = Tensor::full(1.0f32, (1, 4, 32, 64), &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k0, &v0)?;

    // Share prefix
    backing.share_prefix(0, 1, 32)?;

    // Append different data to seq 1 (after shared prefix)
    let k1 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    let v1 = Tensor::full(2.0f32, (1, 4, 16, 64), &Device::Cpu)?;
    backing.write_contiguous(1, 32, &k1, &v1)?;

    // Read back and verify seq 0's data is unchanged
    let k_arenas = backing.k_arenas();
    let table = k_gid_snapshot(&backing);

    let seq0_gid = table[0][0] as usize;
    let arena_idx = seq0_gid / arena_gid_stride();
    let chunk_idx = seq0_gid % arena_gid_stride();

    let chunk = k_arenas[arena_idx].narrow(0, chunk_idx, 1)?;
    let values = chunk.flatten_all()?.to_vec1::<f32>()?;

    // Seq 0's chunk should still be all 1.0
    assert!(values.iter().all(|&x| (x - 1.0).abs() < 1e-6));

    Ok(())
}

#[test]
fn test_ensure_block_writable_sole_owner() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Allocate without sharing
    backing.ensure_for_offset(0, 0, 32)?;

    let original_id = k_gid_snapshot(&backing)[0][0];
    let returned_gids = backing.ensure_block_writable(0, 0)?;

    // Should return same block (no COW needed)
    assert_eq!(original_id, returned_gids[0].raw());

    Ok(())
}

#[test]
fn test_share_prefix_overwrites_target_blocks() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Both sequences allocate independently
    backing.ensure_for_offset(0, 0, 64)?; // 2 blocks
    backing.ensure_for_offset(1, 0, 64)?; // 2 blocks

    // Get seq 1's original block IDs
    let table_before = k_gid_snapshot(&backing);
    let seq1_blocks_before = (table_before[1][0], table_before[1][1]);

    // Share from seq 0 to seq 1 (should free seq 1's old blocks)
    backing.share_prefix(0, 1, 64)?;

    let table_after = k_gid_snapshot(&backing);

    // Seq 1 now has seq 0's block IDs
    assert_eq!(table_after[1][0], table_after[0][0]);
    assert_eq!(table_after[1][1], table_after[0][1]);

    // Old blocks should be different
    assert_ne!(table_after[1][0], seq1_blocks_before.0);

    Ok(())
}

#[test]
fn test_multiple_sequences_share_same_prefix() -> Result<()> {
    let backing = make_test_backing(4)?;

    // Seq 0 is the source
    backing.ensure_for_offset(0, 0, 64)?;

    // Share with seqs 1, 2, 3
    backing.share_prefix(0, 1, 64)?;
    backing.share_prefix(0, 2, 64)?;
    backing.share_prefix(0, 3, 64)?;

    // All should have same block IDs
    let table = k_gid_snapshot(&backing);
    for seq in 1..4 {
        assert_eq!(table[seq][0], table[0][0]);
        assert_eq!(table[seq][1], table[0][1]);
    }

    // Free seq 0 - blocks still referenced by 1,2,3
    backing.free_sequence(0)?;

    // Free seq 1
    backing.free_sequence(1)?;

    // Free seq 2
    backing.free_sequence(2)?;

    // After freeing 0,1,2 seq 3 is sole owner — verify it still has 2 blocks.
    assert_eq!(backing.seq_blocks_count(3)?, 2);

    Ok(())
}

// ==================== Fork Sequence Tests ====================

#[test]
fn test_fork_sequence_basic() -> Result<()> {
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
fn test_fork_sequence_full_blocks_only() -> Result<()> {
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
fn test_fork_sequence_single_partial_block() -> Result<()> {
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
fn test_fork_sequence_preserves_data() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Write distinct data pattern to seq 0
    backing.ensure_for_offset(0, 0, 50)?;
    let k0 = Tensor::full(42.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    let v0 = Tensor::full(42.0f32, (1, 4, 50, 64), &Device::Cpu)?;
    backing.write_contiguous(0, 0, &k0, &v0)?;

    // Fork to seq 1
    backing.fork_sequence(0, 1, 50)?;

    // Read back from seq 1's partial block and verify data
    let k_arenas = backing.k_arenas();
    let table = k_gid_snapshot(&backing);

    // Check the copied partial block (block 1)
    let seq1_gid = table[1][1] as usize;
    let arena_idx = seq1_gid / arena_gid_stride();
    let chunk_idx = seq1_gid % arena_gid_stride();

    let chunk = k_arenas[arena_idx].narrow(0, chunk_idx, 1)?;
    let first_val = chunk.flatten_all()?.to_vec1::<f32>()?[0];
    assert!((first_val - 42.0).abs() < 0.01, "data should be preserved");

    Ok(())
}

#[test]
fn test_fork_sequence_append_independence() -> Result<()> {
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
    let k_arenas = backing.k_arenas();
    let table = k_gid_snapshot(&backing);

    let seq0_gid = table[0][1] as usize;
    let arena_idx = seq0_gid / arena_gid_stride();
    let chunk_idx = seq0_gid % arena_gid_stride();

    let chunk = k_arenas[arena_idx].narrow(0, chunk_idx, 1)?;
    let first_val = chunk.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (first_val - 1.0).abs() < 0.01,
        "seq 0 data should be unchanged"
    );

    Ok(())
}

#[test]
fn test_fork_sequence_overwrites_target() -> Result<()> {
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
fn test_fork_sequence_zero_len() -> Result<()> {
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
fn test_fork_sequence_errors() -> Result<()> {
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
fn test_fork_multiple_times() -> Result<()> {
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

// ==================== Concurrency & Memory Safety Tests ====================

#[test]
fn test_concurrent_share_prefix() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let backing = Arc::new(make_test_backing(8)?);

    // Seq 0 is the source with 4 blocks
    backing.ensure_for_offset(0, 0, 128)?;

    // Spawn threads that each share prefix to a different target
    let handles: Vec<_> = (1..8)
        .map(|target_seq| {
            let backing = Arc::clone(&backing);
            thread::spawn(move || {
                backing.share_prefix(0, target_seq, 64) // Share 2 blocks
            })
        })
        .collect();

    // Wait for all threads
    for h in handles {
        h.join().unwrap()?;
    }

    // Verify all sequences share the same blocks
    let table = k_gid_snapshot(&backing);
    for seq in 1..8 {
        assert_eq!(table[seq][0], table[0][0], "seq {} block 0 mismatch", seq);
        assert_eq!(table[seq][1], table[0][1], "seq {} block 1 mismatch", seq);
    }

    // All shared blocks should reference the same GID
    let table = k_gid_snapshot(&backing);
    for seq in 1..8 {
        assert_eq!(table[seq][0], table[0][0], "seq {} block 0 mismatch", seq);
        assert_eq!(table[seq][1], table[0][1], "seq {} block 1 mismatch", seq);
    }

    Ok(())
}

#[test]
fn test_concurrent_append_after_share() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let backing = Arc::new(make_test_backing(4)?);
    let device = Device::Cpu;

    // Seq 0 allocates and writes initial data (1 full block)
    backing.ensure_for_offset(0, 0, 32)?;
    let k = Tensor::ones((1, 4, 32, 64), DType::F32, &device)?;
    let v = Tensor::ones((1, 4, 32, 64), DType::F32, &device)?;
    backing.write_contiguous(0, 0, &k, &v)?;

    // Share with seqs 1, 2, 3
    for target in 1..4 {
        backing.share_prefix(0, target, 32)?;
    }

    // All should reference the same block
    let table = k_gid_snapshot(&backing);
    for seq in 1..4 {
        assert_eq!(table[seq][0], table[0][0]);
    }

    // Spawn threads that each APPEND new tokens to their sequence
    // (not overwriting shared prefix — that's the valid pattern)
    let handles: Vec<_> = (1..4)
        .map(|seq| {
            let backing = Arc::clone(&backing);
            let device = device.clone();
            thread::spawn(move || -> Result<()> {
                let k = Tensor::full(seq as f32, (1, 4, 16, 64), &device)?;
                let v = Tensor::full(seq as f32, (1, 4, 16, 64), &device)?;
                backing.write_contiguous(seq, 32, &k, &v)?;
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap()?;
    }

    // Each seq 1-3 should now have 2 blocks (1 shared prefix + 1 own tail)
    for seq in 1..4 {
        assert_eq!(backing.seq_blocks_count(seq)?, 2);
    }

    Ok(())
}

#[test]
fn test_concurrent_free_and_share() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    let backing = Arc::new(make_test_backing(4)?);

    // Setup: seq 0 has blocks, shared with seq 1
    backing.ensure_for_offset(0, 0, 64)?;
    backing.share_prefix(0, 1, 64)?;

    // Also setup seq 2 with its own blocks
    backing.ensure_for_offset(2, 0, 64)?;

    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: Free seq 0
    let backing1 = Arc::clone(&backing);
    let barrier1 = Arc::clone(&barrier);
    let h1 = thread::spawn(move || -> Result<()> {
        barrier1.wait();
        backing1.free_sequence(0)?;
        Ok(())
    });

    // Thread 2: Share from seq 2 to seq 3
    let backing2 = Arc::clone(&backing);
    let barrier2 = Arc::clone(&barrier);
    let h2 = thread::spawn(move || -> Result<()> {
        barrier2.wait();
        backing2.share_prefix(2, 3, 64)?;
        Ok(())
    });

    h1.join().unwrap()?;
    h2.join().unwrap()?;

    // Seq 0 should be freed (0 blocks)
    assert_eq!(backing.seq_blocks_count(0)?, 0);

    // Seq 1 should still have its blocks (was sharing with seq 0)
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    // Seq 2 and 3 should share blocks
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[2][0], table[3][0]);
    assert_eq!(table[2][1], table[3][1]);

    Ok(())
}

#[test]
fn test_memory_not_leaked_on_overwrite() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 allocates 2 blocks
    backing.ensure_for_offset(0, 0, 64)?;

    // Seq 1 also allocates 2 blocks
    backing.ensure_for_offset(1, 0, 64)?;

    // Get initial block IDs
    let table_before = k_gid_snapshot(&backing);
    let _seq1_old = (table_before[1][0], table_before[1][1]);

    // Share from seq 0 to seq 1 (overwrites seq 1's blocks)
    backing.share_prefix(0, 1, 64)?;

    // Free seq 0 and seq 1
    backing.free_sequence(0)?;
    backing.free_sequence(1)?;

    // Now allocate for seq 0 again - should get the old seq1 blocks back from free list
    backing.ensure_for_offset(0, 0, 64)?;

    let table_after = k_gid_snapshot(&backing);

    // The freed blocks should have been reused (not leaked)
    // We can't guarantee exact block IDs due to allocation order, but
    // we should be able to allocate without running out of memory
    assert!(table_after[0][0] >= 0);
    assert!(table_after[0][1] >= 0);

    Ok(())
}

#[test]
fn test_memory_not_leaked_on_partial_overwrite() -> Result<()> {
    let backing = make_test_backing(2)?;

    // Seq 0 allocates 4 blocks
    backing.ensure_for_offset(0, 0, 128)?;

    // Seq 1 also allocates 4 blocks
    backing.ensure_for_offset(1, 0, 128)?;

    // Share only 2 blocks from seq 0 to seq 1 (partial overwrite)
    backing.share_prefix(0, 1, 64)?;

    // Seq 1 should have:
    // - 2 shared blocks from seq 0
    // - 2 original blocks that were NOT overwritten
    assert_eq!(backing.seq_blocks_count(1)?, 4);

    // Free both sequences
    backing.free_sequence(0)?;
    backing.free_sequence(1)?;

    // All blocks should be freed and reusable
    assert_eq!(backing.seq_blocks_count(0)?, 0);
    assert_eq!(backing.seq_blocks_count(1)?, 0);

    // Allocate 6 blocks - should work if nothing leaked
    backing.ensure_for_offset(0, 0, 128)?;
    backing.ensure_for_offset(1, 0, 64)?;

    assert_eq!(backing.seq_blocks_count(0)?, 4);
    assert_eq!(backing.seq_blocks_count(1)?, 2);

    Ok(())
}

#[test]
fn test_arc_refcount_integrity() -> Result<()> {
    let backing = make_test_backing(3)?;

    // Seq 0 allocates 1 block
    backing.ensure_for_offset(0, 0, 32)?;

    // Share with seq 1
    backing.share_prefix(0, 1, 32)?;
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[1][0], "blocks should be shared");

    // Share with seq 2
    backing.share_prefix(0, 2, 32)?;
    let table = k_gid_snapshot(&backing);
    assert_eq!(table[0][0], table[2][0], "blocks should be shared");

    // Free seq 0
    backing.free_sequence(0)?;

    // Free seq 1
    backing.free_sequence(1)?;

    // Free seq 2 — block returned to free list
    backing.free_sequence(2)?;

    // Allocate new sequence — should get the freed block
    backing.ensure_for_offset(0, 0, 32)?;
    assert_eq!(backing.seq_blocks_count(0)?, 1);

    Ok(())
}

#[test]
fn test_no_use_after_free() -> Result<()> {
    let backing = make_test_backing(2)?;
    let device = Device::Cpu;

    // Seq 0 writes data
    backing.ensure_for_offset(0, 0, 32)?;
    let k = Tensor::full(42.0f32, (1, 4, 32, 64), &device)?;
    let v = Tensor::full(42.0f32, (1, 4, 32, 64), &device)?;
    backing.write_contiguous(0, 0, &k, &v)?;

    // Share with seq 1
    backing.share_prefix(0, 1, 32)?;

    // Free seq 0 - block should NOT be freed (still used by seq 1)
    backing.free_sequence(0)?;

    // Seq 1 should still be able to read the data
    // (If block was freed, this could crash or return garbage)
    let table = k_gid_snapshot(&backing);
    assert!(table[1][0] >= 0, "seq 1's block should still be valid");

    // Allocate new data for seq 0
    backing.ensure_for_offset(0, 0, 32)?;
    let k2 = Tensor::full(99.0f32, (1, 4, 32, 64), &device)?;
    let v2 = Tensor::full(99.0f32, (1, 4, 32, 64), &device)?;
    backing.write_contiguous(0, 0, &k2, &v2)?;

    // Seq 1's data should still be intact (different block)
    let table2 = k_gid_snapshot(&backing);
    assert_ne!(
        table2[0][0], table2[1][0],
        "seq 0 should have different block than seq 1"
    );

    Ok(())
}

#[test]
fn test_concurrent_reads_during_cow() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    let backing = Arc::new(make_test_backing(3)?);
    let device = Device::Cpu;

    // Setup: seq 0 has data, shared with seq 1 and 2
    backing.ensure_for_offset(0, 0, 32)?;
    let k = Tensor::ones((1, 4, 32, 64), DType::F32, &device)?;
    let v = Tensor::ones((1, 4, 32, 64), DType::F32, &device)?;
    backing.write_contiguous(0, 0, &k, &v)?;

    backing.share_prefix(0, 1, 32)?;
    backing.share_prefix(0, 2, 32)?;

    let barrier = Arc::new(Barrier::new(3));

    // Thread 1: Append to seq 0 (at offset 32, not overwriting shared block)
    let backing1 = Arc::clone(&backing);
    let barrier1 = Arc::clone(&barrier);
    let device1 = device.clone();
    let h1 = thread::spawn(move || -> Result<()> {
        barrier1.wait();
        let k = Tensor::full(1.0f32, (1, 4, 16, 64), &device1)?;
        let v = Tensor::full(1.0f32, (1, 4, 16, 64), &device1)?;
        backing1.write_contiguous(0, 32, &k, &v)?;
        Ok(())
    });

    // Thread 2: Read seq 1's block table
    let backing2 = Arc::clone(&backing);
    let barrier2 = Arc::clone(&barrier);
    let h2 = thread::spawn(move || -> Result<i64> {
        barrier2.wait();
        let table = k_gid_snapshot(&backing2);
        Ok(table[1][0])
    });

    // Thread 3: Read seq 2's block count
    let backing3 = Arc::clone(&backing);
    let barrier3 = Arc::clone(&barrier);
    let h3 = thread::spawn(move || -> Result<usize> {
        barrier3.wait();
        backing3.seq_blocks_count(2)
    });

    h1.join().unwrap()?;
    let block_id = h2.join().unwrap()?;
    let block_count = h3.join().unwrap()?;

    // Results should be valid (no panics, no corruption)
    assert!(block_id >= 0, "block ID should be valid");
    assert_eq!(block_count, 1, "seq 2 should have 1 block");

    Ok(())
}

#[test]
fn test_stress_share_and_free_cycles() -> Result<()> {
    let backing = make_test_backing(4)?;

    // Run multiple cycles of share and free
    for cycle in 0..10 {
        // Allocate for seq 0
        backing.ensure_for_offset(0, 0, 64)?;

        // Share with others
        for target in 1..4 {
            backing.share_prefix(0, target, 32)?;
        }

        // Verify sharing via block table
        let table = k_gid_snapshot(&backing);
        for seq in 1..4 {
            assert_eq!(
                table[seq][0], table[0][0],
                "cycle {} seq {} block 0 not shared",
                cycle, seq
            );
        }

        // Free all in random order
        let order = match cycle % 4 {
            0 => [0, 1, 2, 3],
            1 => [3, 2, 1, 0],
            2 => [1, 3, 0, 2],
            _ => [2, 0, 3, 1],
        };

        for seq in order {
            backing.free_sequence(seq)?;
        }

        // Verify all freed
        for seq in 0..4 {
            assert_eq!(
                backing.seq_blocks_count(seq)?,
                0,
                "cycle {} seq {} not freed",
                cycle,
                seq
            );
        }
    }

    // After all cycles, memory should not be fragmented/leaked
    // Allocate max capacity
    backing.ensure_for_offset(0, 0, 128)?;
    backing.ensure_for_offset(1, 0, 128)?;

    assert_eq!(backing.seq_blocks_count(0)?, 4);
    assert_eq!(backing.seq_blocks_count(1)?, 4);

    Ok(())
}

// ==================== Original ScatteredKvCache Tests ====================

#[test]
fn test_scattered_kv_cache() -> Result<()> {
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

// ==================== Phase 1: Quantization Format Tests ====================

#[test]
fn test_quant_format_q4_0_properties() {
    use crate::kv_cache::QuantFormat;

    let fmt = QuantFormat::Q4_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 18); // 16 data + 2 scale
    assert!((fmt.bytes_per_elem() - 0.5625).abs() < 0.001); // 18/32
}

#[test]
fn test_quant_format_q8_0_properties() {
    use crate::kv_cache::QuantFormat;

    let fmt = QuantFormat::Q8_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 34); // 32 data + 2 scale
    assert!((fmt.bytes_per_elem() - 1.0625).abs() < 0.001); // 34/32
}

#[test]
fn test_quant_format_equality() {
    use crate::kv_cache::QuantFormat;

    let q4a = QuantFormat::Q4_0;
    let q4b = QuantFormat::Q4_0;
    let q8 = QuantFormat::Q8_0;

    assert_eq!(q4a, q4b);
    assert_ne!(q4a, q8);
}

#[test]
fn test_quant_format_to_ggml_dtype() {
    use crate::kv_cache::QuantFormat;
    use candle::quantized::GgmlDType;

    // Test internal conversion (accessible in tests within crate)
    assert_eq!(QuantFormat::Q4_0.to_ggml_dtype(), GgmlDType::Q4_0);
    assert_eq!(QuantFormat::Q8_0.to_ggml_dtype(), GgmlDType::Q8_0);
}

#[test]
fn test_kv_format_float_variants() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let f32_fmt = KvFormat::Float(DType::F32);
    let f16_fmt = KvFormat::Float(DType::F16);
    let bf16_fmt = KvFormat::Float(DType::BF16);

    assert!(!f32_fmt.is_quantized());
    assert!(!f16_fmt.is_quantized());
    assert!(!bf16_fmt.is_quantized());

    assert_eq!(f32_fmt.bytes_per_elem(), 4.0);
    assert_eq!(f16_fmt.bytes_per_elem(), 2.0);
    assert_eq!(bf16_fmt.bytes_per_elem(), 2.0);
}

/// `bytes_per_block` is the exact integer counterpart of `bytes_per_elem`, and
/// must agree with the block layouts byte for byte — VRAM accounting rounds to
/// whole blocks, and a quantized block's size does not divide its element count.
#[test]
fn test_kv_format_bytes_per_block() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Float formats: dtype width across all 32 elements.
    assert_eq!(KvFormat::Float(DType::F32).bytes_per_block(), 128);
    assert_eq!(KvFormat::Float(DType::F16).bytes_per_block(), 64);
    assert_eq!(KvFormat::Float(DType::BF16).bytes_per_block(), 64);

    // Quantized formats delegate to the block layout: d:f16 + 32 nibbles = 18,
    // d:f16 + 32 i8 = 34, and R16's per-element d:f16 + q:u16 = 128.
    assert_eq!(KvFormat::Quantized(QuantFormat::Q4_0).bytes_per_block(), 18);
    assert_eq!(KvFormat::Quantized(QuantFormat::Q8_0).bytes_per_block(), 34);
    assert_eq!(KvFormat::Quantized(QuantFormat::R16).bytes_per_block(), 128);

    // Every format agrees with its own float ratio — the two accessors are the
    // same quantity, so they must not be able to drift apart.
    use strum::IntoEnumIterator;
    let all = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3]
        .into_iter()
        .map(KvFormat::Float)
        .chain(QuantFormat::iter().map(KvFormat::Quantized));
    for fmt in all {
        let ratio = fmt.bytes_per_block() as f32 / CHUNK_SIZE as f32;
        assert!(
            (ratio - fmt.bytes_per_elem()).abs() < 1e-6,
            "{fmt:?}: {ratio} != {}",
            fmt.bytes_per_elem()
        );
    }
}

#[test]
fn test_kv_format_quantized_variants() {
    use crate::kv_cache::{KvFormat, QuantFormat};

    let q4_fmt = KvFormat::Quantized(QuantFormat::Q4_0);
    let q8_fmt = KvFormat::Quantized(QuantFormat::Q8_0);

    assert!(q4_fmt.is_quantized());
    assert!(q8_fmt.is_quantized());

    assert!((q4_fmt.bytes_per_elem() - 0.5625).abs() < 0.001);
    assert!((q8_fmt.bytes_per_elem() - 1.0625).abs() < 0.001);
}

#[test]
fn test_kv_format_default() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let default_fmt = KvFormat::default();
    assert_eq!(default_fmt, KvFormat::Float(DType::BF16));
}

#[test]
fn test_kv_format_from_dtype() {
    use crate::kv_cache::KvFormat;
    use candle::DType;

    let fmt: KvFormat = DType::F32.into();
    assert_eq!(fmt, KvFormat::Float(DType::F32));

    let fmt: KvFormat = DType::F16.into();
    assert_eq!(fmt, KvFormat::Float(DType::F16));
}

#[test]
fn test_kv_format_from_quant_format() {
    use crate::kv_cache::{KvFormat, QuantFormat};

    let fmt: KvFormat = QuantFormat::Q4_0.into();
    assert_eq!(fmt, KvFormat::Quantized(QuantFormat::Q4_0));

    let fmt: KvFormat = QuantFormat::Q8_0.into();
    assert_eq!(fmt, KvFormat::Quantized(QuantFormat::Q8_0));
}

#[test]
fn test_kv_format_equality() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    let f32a = KvFormat::Float(DType::F32);
    let f32b = KvFormat::Float(DType::F32);
    let f16 = KvFormat::Float(DType::F16);
    let q4 = KvFormat::Quantized(QuantFormat::Q4_0);

    assert_eq!(f32a, f32b);
    assert_ne!(f32a, f16);
    assert_ne!(f32a, q4);
}

#[test]
fn test_kv_format_memory_comparison() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Memory savings comparison
    let f32_bpe = KvFormat::Float(DType::F32).bytes_per_elem();
    let f16_bpe = KvFormat::Float(DType::F16).bytes_per_elem();
    let q8_bpe = KvFormat::Quantized(QuantFormat::Q8_0).bytes_per_elem();
    let q4_bpe = KvFormat::Quantized(QuantFormat::Q4_0).bytes_per_elem();

    // Verify expected savings: Q4 < Q8 < F16 < F32
    assert!(q4_bpe < q8_bpe);
    assert!(q8_bpe < f16_bpe);
    assert!(f16_bpe < f32_bpe);

    // Q4_0 is ~7x smaller than F32
    let q4_savings = f32_bpe / q4_bpe;
    assert!(q4_savings > 7.0);
    assert!(q4_savings < 7.2);

    // Q8_0 is ~3.75x smaller than F32
    let q8_savings = f32_bpe / q8_bpe;
    assert!(q8_savings > 3.7);
    assert!(q8_savings < 3.8);
}

// ==================== Phase 4: Quantized KV Write Tests ====================

#[test]
fn test_quantized_backing_creation_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        4,  // initial_batch
        4,  // n_kv_head
        64, // head_dim (must be divisible by 32)
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256, // initial_max_seq_len
    )?;

    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q8_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
fn test_quantized_backing_creation_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        4,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
        &device,
        256,
    )?;

    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q4_0));
    Ok(())
}

#[test]
fn test_quantized_backing_rejects_bad_head_dim() {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    // head_dim=48 is not divisible by 32
    let result = ChunkedKvBacking::new_with_format(
        4,
        4,
        48,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    );

    assert!(result.is_err());
}

#[test]
fn test_quantized_write_single_token_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    // Allocate sequence
    let batch_idx = backing.alloc_sequence()?;

    // Create K/V tensors (batch=1, n_kv_head, len=1, head_dim)
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;

    // Write at position 0
    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    // If we get here without error, the write succeeded
    Ok(())
}

#[test]
fn test_quantized_write_multiple_tokens_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 10;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write multiple tokens at once
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

#[test]
fn test_quantized_write_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

#[test]
fn test_quantized_write_cross_chunk_boundary() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write tokens that span chunk boundary (CHUNK_SIZE=32, so positions 30-35 span chunks 0 and 1)
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 6, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 6, head_dim), &device)?;

    // Start at position 30, write 6 tokens (30-35)
    backing.write_contiguous(batch_idx, 30, &k, &v)?;

    Ok(())
}

#[test]
fn test_quantized_write_sequential_appends() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Simulate sequential token generation
    for pos in 0..20 {
        let k = Tensor::full(pos as f32, (1, n_kv_head, 1, head_dim), &device)?;
        let v = Tensor::full((pos as f32) * 2.0, (1, n_kv_head, 1, head_dim), &device)?;
        backing.write_contiguous(batch_idx, pos, &k, &v)?;
    }

    Ok(())
}

#[test]
fn test_quantized_write_multiple_sequences() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        4,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    // Allocate multiple sequences
    let idx0 = backing.alloc_sequence()?;
    let idx1 = backing.alloc_sequence()?;
    let idx2 = backing.alloc_sequence()?;

    // Write different amounts to each
    let k0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    let v0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    backing.write_contiguous(idx0, 0, &k0, &v0)?;

    let k1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 50, head_dim), &device)?;
    let v1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 50, head_dim), &device)?;
    backing.write_contiguous(idx1, 0, &k1, &v1)?;

    let k2 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v2 = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(idx2, 0, &k2, &v2)?;

    Ok(())
}

#[test]
fn test_quantized_write_larger_head_dim() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 8;
    let head_dim = 128; // Larger head_dim (still divisible by 32)

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        512,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 20, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 20, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}

#[test]
fn test_quantized_float_regression() -> Result<()> {
    // Ensure float path still works after changes
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F32, &device, 256)?;

    assert!(!backing.is_quantized());

    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 10, head_dim), &device)?;

    backing.write_contiguous(batch_idx, 0, &k, &v)?;

    Ok(())
}
// ==================== Phase 5: Quantized KV Read Tests ====================

#[test]
fn test_quantized_read_single_token() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write a single token
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read it back
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;

    assert_eq!(k_read.dims(), &[1, n_kv_head, 1, head_dim]);
    assert_eq!(v_read.dims(), &[1, n_kv_head, 1, head_dim]);

    Ok(())
}

#[test]
fn test_quantized_read_multiple_tokens() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 10;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write multiple tokens
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, seq_len);

    // Read all back
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, seq_len)?;

    assert_eq!(k_read.dims(), &[1, n_kv_head, seq_len, head_dim]);
    assert_eq!(v_read.dims(), &[1, n_kv_head, seq_len, head_dim]);

    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q8_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write known values
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read back and compare (cast to F32 for comparison since backing may use F16)
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;

    // Q8_0 should have reasonable accuracy
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // MSE should be small for Q8_0 (8-bit quantization)
    assert!(k_diff < 0.01, "K MSE too high: {}", k_diff);
    assert!(v_diff < 0.01, "V MSE too high: {}", v_diff);

    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q4_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;

    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_0),
        KvFormat::Quantized(QuantFormat::Q4_0),
        &device,
        256,
    )?;

    let batch_idx = backing.alloc_sequence()?;

    // Write known values
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);

    // Read back and compare (cast to F32 for comparison since backing may use F16)
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;

    // Q4_0 has more quantization error than Q8_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // MSE should be reasonable for Q4_0 (4-bit quantization has more error)
    assert!(k_diff < 0.1, "K MSE too high for Q4_0: {}", k_diff);
    assert!(v_diff < 0.1, "V MSE too high for Q4_0: {}", v_diff);

    Ok(())
}

#[test]
fn test_float_read_contiguous() -> Result<()> {
    // Test that float read path also works
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let seq_len = 5;

    let backing = ChunkedKvBacking::new(1, n_kv_head, head_dim, DType::F32, &device, 256)?;

    let batch_idx = backing.alloc_sequence()?;

    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, seq_len);

    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, seq_len)?;

    // Float should be exact
    let k_diff = (&k - &k_read)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.abs()?.max_all()?.to_scalar::<f32>()?;

    assert!(k_diff < 1e-5, "K diff too high: {}", k_diff);
    assert!(v_diff < 1e-5, "V diff too high: {}", v_diff);

    Ok(())
}

// ==================== PagedKvArenas Trait Tests ====================

#[test]
fn test_paged_kv_arenas_trait_float() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new(1, n_kv_head, head_dim, DType::F32, &device, 256)?;

    // Test trait methods
    assert_eq!(backing.n_kv_head(), n_kv_head);
    assert_eq!(backing.head_dim(), head_dim);
    assert_eq!(backing.k_format(), KvFormat::Float(DType::F32));
    assert!(!backing.is_quantized());

    // Float arenas should be available
    assert!(backing.float_arenas().is_some());
    assert!(backing.quantized_arenas().is_none());

    Ok(())
}

#[test]
fn test_paged_kv_arenas_trait_quantized() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};

    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_0),
        KvFormat::Quantized(QuantFormat::Q8_0),
        &device,
        256,
    )?;

    // Test trait methods
    assert_eq!(backing.n_kv_head(), n_kv_head);
    assert_eq!(backing.head_dim(), head_dim);
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q8_0));
    assert!(backing.is_quantized());

    // Quantized arenas should be available
    assert!(backing.float_arenas().is_none());
    assert!(backing.quantized_arenas().is_some());

    Ok(())
}

#[test]
fn test_kv_format_dtype_method() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    // Float formats return Some(dtype)
    assert_eq!(KvFormat::Float(DType::F32).dtype(), Some(DType::F32));
    assert_eq!(KvFormat::Float(DType::F16).dtype(), Some(DType::F16));
    assert_eq!(KvFormat::Float(DType::BF16).dtype(), Some(DType::BF16));

    // Quantized formats return None
    assert_eq!(KvFormat::Quantized(QuantFormat::Q4_0).dtype(), None);
    assert_eq!(KvFormat::Quantized(QuantFormat::Q8_0).dtype(), None);
}

// ==================== Phase 6: New Quant Format Tests (Q4_KS, Q8_KS, Q2_0, Q3_0) ====================

#[test]
fn test_quant_format_q4_ks_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q4_KS;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 20); // 4-bit nibbles + 2 byte scale + 2 sub-block scales
    assert!((fmt.bytes_per_elem() - 20.0 / 32.0).abs() < 0.001);
}

#[test]
fn test_quant_format_q8_ks_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q8_KS;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 36); // 32 bytes int8 + 2 byte scale + 2 sub-block scales
    assert!((fmt.bytes_per_elem() - 36.0 / 32.0).abs() < 0.001);
}

#[test]
fn test_quant_format_q2_0_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q2_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 10); // 2-bit × 32 + 2 byte scale
    assert!((fmt.bytes_per_elem() - 10.0 / 32.0).abs() < 0.001);
}

#[test]
fn test_quant_format_q3_0_properties() {
    use crate::kv_cache::QuantFormat;
    let fmt = QuantFormat::Q3_0;
    assert_eq!(fmt.block_size(), 32);
    assert_eq!(fmt.bytes_per_block(), 14); // 3-bit × 32 + 2 byte scale
    assert!((fmt.bytes_per_elem() - 14.0 / 32.0).abs() < 0.001);
}

#[test]
fn test_quant_format_to_ggml_dtype_new_types() {
    use crate::kv_cache::QuantFormat;
    use candle::quantized::GgmlDType;

    assert_eq!(QuantFormat::Q4_KS.to_ggml_dtype(), GgmlDType::Q4_KS);
    assert_eq!(QuantFormat::Q8_KS.to_ggml_dtype(), GgmlDType::Q8_KS);
    assert_eq!(QuantFormat::Q2_0.to_ggml_dtype(), GgmlDType::Q2_0);
    assert_eq!(QuantFormat::Q3_0.to_ggml_dtype(), GgmlDType::Q3_0);
}

#[test]
fn test_new_quant_formats_memory_ordering() {
    use crate::kv_cache::QuantFormat;

    // Ordering by bytes per element (lowest to highest):
    // Q2_0(10/32) < Q3_0(14/32) < Q4_0(18/32) < Q4_KS(20/32) < Q8_0(34/32) < Q8_KS(36/32)
    // Note: Q4_KS has slightly more bytes than Q4_0 due to sub-block scale overhead
    let q2_bpe = QuantFormat::Q2_0.bytes_per_elem();
    let q3_bpe = QuantFormat::Q3_0.bytes_per_elem();
    let q4_bpe = QuantFormat::Q4_0.bytes_per_elem();
    let q4ks_bpe = QuantFormat::Q4_KS.bytes_per_elem();
    let q8_bpe = QuantFormat::Q8_0.bytes_per_elem();

    assert!(q2_bpe < q3_bpe);
    assert!(q3_bpe < q4_bpe);
    // Q4_KS uses 20 bytes vs Q4_0's 18 bytes (sub-block scales add overhead)
    assert!(q4_bpe < q4ks_bpe);
    assert!(q4ks_bpe < q8_bpe);
}

#[test]
fn test_quantized_backing_creation_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q4_KS));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
fn test_quantized_backing_creation_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q8_KS));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
fn test_quantized_backing_creation_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q2_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
fn test_quantized_backing_creation_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        4,
        64,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    assert_eq!(backing.k_format(), KvFormat::Quantized(QuantFormat::Q3_0));
    assert!(backing.is_quantized());
    Ok(())
}

#[test]
fn test_quantized_write_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
fn test_quantized_write_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
fn test_quantized_write_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
fn test_quantized_write_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 5, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q4_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q4_KS),
        KvFormat::Quantized(QuantFormat::Q4_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q4_KS uses sub-block scales for better 4-bit accuracy than plain Q4_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.1, "K MSE too high for Q4_KS: {}", k_diff);
    assert!(v_diff < 0.1, "V MSE too high for Q4_KS: {}", v_diff);
    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q8_ks() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q8_KS),
        KvFormat::Quantized(QuantFormat::Q8_KS),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q8_KS has sub-block scales: similar accuracy to Q8_0
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.01, "K MSE too high for Q8_KS: {}", k_diff);
    assert!(v_diff < 0.01, "V MSE too high for Q8_KS: {}", v_diff);
    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q2_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q2_0),
        KvFormat::Quantized(QuantFormat::Q2_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q2_0 has high quantization error (2 bits), use loose tolerance
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 1.0, "K MSE too high for Q2_0: {}", k_diff);
    assert!(v_diff < 1.0, "V MSE too high for Q2_0: {}", v_diff);
    Ok(())
}

#[test]
fn test_quantized_roundtrip_accuracy_q3_0() -> Result<()> {
    use crate::kv_cache::{ChunkedKvBacking, KvFormat, QuantFormat};
    let device = Device::Cpu;
    let n_kv_head = 4;
    let head_dim = 64;
    let backing = ChunkedKvBacking::new_with_format(
        1,
        n_kv_head,
        head_dim,
        KvFormat::Quantized(QuantFormat::Q3_0),
        KvFormat::Quantized(QuantFormat::Q3_0),
        &device,
        256,
    )?;
    let batch_idx = backing.alloc_sequence()?;
    let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, 1, head_dim), &device)?;
    backing.write_contiguous(batch_idx, 0, &k, &v)?;
    backing.set_len(batch_idx, 1);
    let (k_read, v_read) = backing.read_contiguous(batch_idx, 0, 1)?;
    let k_read = k_read.to_dtype(candle::DType::F32)?;
    let v_read = v_read.to_dtype(candle::DType::F32)?;
    // Q3_0 has moderate quantization error (3 bits)
    let k_diff = (&k - &k_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let v_diff = (&v - &v_read)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(k_diff < 0.5, "K MSE too high for Q3_0: {}", k_diff);
    assert!(v_diff < 0.5, "V MSE too high for Q3_0: {}", v_diff);
    Ok(())
}

// ==================== KvFormat tag round-trip ====================

/// `to_tag` / `from_tag` must round-trip for every format
/// `ArenaFormatTag::from_kv_format` accepts — the persistence codec relies on
/// it. `from_tag` derives its answer from `to_tag`, so this also guards the
/// forward mapping against silently producing colliding tags.
#[test]
fn kv_format_tag_round_trips() {
    use crate::kv_cache::{KvFormat, QuantFormat};
    use strum::IntoEnumIterator;

    let mut formats: Vec<KvFormat> = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3]
        .into_iter()
        .map(KvFormat::Float)
        .collect();
    formats.extend(QuantFormat::iter().map(KvFormat::Quantized));

    let mut seen_tags = std::collections::HashSet::new();
    for fmt in formats {
        let tag = fmt.to_tag();
        assert!(
            seen_tags.insert(tag),
            "duplicate tag {tag} for {fmt:?} — from_kv_format is not injective"
        );
        assert_eq!(
            KvFormat::from_tag(tag),
            Some(fmt),
            "tag {tag} did not round-trip back to {fmt:?}"
        );
    }
    // An unknown tag decodes to None, not a wrong format.
    assert_eq!(KvFormat::from_tag(254), None);
}
