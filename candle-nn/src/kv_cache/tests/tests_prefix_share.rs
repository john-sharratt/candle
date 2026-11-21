//! Tests sharing a prefix from one sequence to another in the chunked backing, and
//! the ownership rules that govern the shared blocks.
//!
//! Sharing whole blocks gives the target the same block ids as the source, and a
//! prefix shorter than a block shares nothing. Invalid requests, such as sharing with
//! oneself, from or to an out-of-range sequence, or from a source too short for the
//! prefix, are refused. A shared block is released only when its last owner frees it.
//! An append after a share goes to a new tail block, and a write through a sole owner
//! does not copy. Sharing into a target replaces that target's own blocks.

use super::*;

#[test]
pub(super) fn test_share_prefix_basic() -> Result<()> {
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
pub(super) fn test_share_prefix_edge_cases() -> Result<()> {
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
pub(super) fn test_share_prefix_errors() -> Result<()> {
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
pub(super) fn test_free_sequence_with_shared_blocks() -> Result<()> {
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
pub(super) fn test_append_after_share_prefix() -> Result<()> {
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
pub(super) fn test_share_preserves_source_data() -> Result<()> {
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
    let table = k_gid_snapshot(&backing);
    let values = band_f32(&backing, table[0][0] as usize, CHUNK_SIZE)?;

    // Seq 0's chunk should still be all 1.0
    assert!(values.iter().all(|&x| (x - 1.0).abs() < 1e-6));

    Ok(())
}

#[test]
pub(super) fn test_ensure_block_writable_sole_owner() -> Result<()> {
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
pub(super) fn test_share_prefix_overwrites_target_blocks() -> Result<()> {
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
pub(super) fn test_multiple_sequences_share_same_prefix() -> Result<()> {
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
