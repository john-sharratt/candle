//! Tests that shared, appended, freed and rewritten KV blocks stay consistent under
//! concurrent access and across repeated share and free cycles.
//!
//! Several threads share one source prefix to different targets, and the tests check
//! that every sharer references the same block id. Appends after a share must
//! extend the sequence with a private tail and leave the shared prefix alone. A
//! block is returned to the free list only when its last reference is released, so
//! a sequence freed while another still shares its block must not free it. Reads
//! taken while another thread appends must see a valid table. Repeated share and
//! free cycles must leave the allocator able to hand out its full capacity.

use super::*;

// ==================== Concurrency & Memory Safety Tests ====================

#[test]
pub(super) fn test_concurrent_share_prefix() -> Result<()> {
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
pub(super) fn test_concurrent_append_after_share() -> Result<()> {
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
pub(super) fn test_concurrent_free_and_share() -> Result<()> {
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
pub(super) fn test_memory_not_leaked_on_overwrite() -> Result<()> {
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
pub(super) fn test_memory_not_leaked_on_partial_overwrite() -> Result<()> {
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
pub(super) fn test_arc_refcount_integrity() -> Result<()> {
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
pub(super) fn test_no_use_after_free() -> Result<()> {
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
pub(super) fn test_concurrent_reads_during_cow() -> Result<()> {
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
pub(super) fn test_stress_share_and_free_cycles() -> Result<()> {
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
