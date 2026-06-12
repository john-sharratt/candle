//! GPU integration tests for the arena compaction kernels.
//!
//! Tests the copy kernel (format-agnostic byte copy) and the patch kernel
//! (binary-search GID rewrite in a GPU block table).
#![cfg(feature = "cuda")]

use candle_core::quantized::{arena_compact_copy, arena_compact_patch, CompactMove};
use candle_core::{Device, Result};
use cudarc::driver::DevicePtr;

fn get_cuda_dev() -> Result<(Device, candle_core::CudaDevice)> {
    let dev = Device::new_cuda(0)?;
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    Ok((dev, cuda_dev))
}

// =============================================================================
// Copy kernel tests
// =============================================================================

#[test]
fn arena_compact_copy_single_move() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    // 256 bytes = 16 uint4s — one block of blockDim=16 handles it exactly
    let stride: usize = 256;
    let src_data: Vec<u8> = (0..stride).map(|i| (i % 251) as u8).collect();
    let dst_data: Vec<u8> = vec![0u8; stride];

    let src_gpu = cuda_dev.memcpy_stod(&src_data)?;
    let dst_gpu = cuda_dev.memcpy_stod(&dst_data)?;

    let src_ptr = src_gpu.device_ptr(&stream).0 as u64;
    let dst_ptr = dst_gpu.device_ptr(&stream).0 as u64;

    let moves = vec![CompactMove {
        dst: dst_ptr,
        src: src_ptr,
        stride_bytes: stride as u32,
        _pad: 0,
    }];
    let moves_gpu = cuda_dev.memcpy_stod(&moves)?;

    arena_compact_copy(&moves_gpu, 1, 16, &stream)?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&dst_gpu)?;
    assert_eq!(result, src_data, "copy output must match source exactly");
    Ok(())
}

#[test]
fn arena_compact_copy_multiple_moves() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let stride: usize = 2048; // large tier — blockDim=128
    let num_moves = 4;

    // Create distinct source buffers
    let mut src_gpus = Vec::new();
    let mut dst_gpus = Vec::new();
    let mut src_datas = Vec::new();
    for i in 0..num_moves {
        let data: Vec<u8> = (0..stride).map(|j| ((i * 37 + j) % 253) as u8).collect();
        src_gpus.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpus.push(cuda_dev.memcpy_stod(&vec![0u8; stride])?);
        src_datas.push(data);
    }

    let moves: Vec<CompactMove> = (0..num_moves)
        .map(|i| CompactMove {
            dst: dst_gpus[i].device_ptr(&stream).0 as u64,
            src: src_gpus[i].device_ptr(&stream).0 as u64,
            stride_bytes: stride as u32,
            _pad: 0,
        })
        .collect();
    let moves_gpu = cuda_dev.memcpy_stod(&moves)?;

    arena_compact_copy(&moves_gpu, num_moves, 128, &stream)?;
    dev.synchronize()?;

    for i in 0..num_moves {
        let result = cuda_dev.memcpy_dtov(&dst_gpus[i])?;
        assert_eq!(
            result, src_datas[i],
            "move {i}: copy output must match source"
        );
    }
    Ok(())
}

#[test]
fn arena_compact_copy_small_stride() -> Result<()> {
    // Q0 stride = 32 bytes — smallest tier, blockDim = 32/16 = 2
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let stride: usize = 32;
    let block_dim = stride / 16; // = 2
    let src_data: Vec<u8> = (0..stride).map(|i| (i * 7 + 3) as u8).collect();

    let src_gpu = cuda_dev.memcpy_stod(&src_data)?;
    let dst_gpu = cuda_dev.memcpy_stod(&vec![0u8; stride])?;

    let moves = vec![CompactMove {
        dst: dst_gpu.device_ptr(&stream).0 as u64,
        src: src_gpu.device_ptr(&stream).0 as u64,
        stride_bytes: stride as u32,
        _pad: 0,
    }];
    let moves_gpu = cuda_dev.memcpy_stod(&moves)?;

    arena_compact_copy(&moves_gpu, 1, block_dim, &stream)?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&dst_gpu)?;
    assert_eq!(result, src_data);
    Ok(())
}

// =============================================================================
// Patch kernel tests
// =============================================================================

#[test]
fn arena_compact_patch_basic() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    // block_table: [10, 20, -1, 30, 40, 50]
    // moves: 20→200, 40→400
    // expected: [10, 200, -1, 30, 400, 50]
    let block_table = vec![10i32, 20, -1, 30, 40, 50];
    let src_gids = vec![20i32, 40]; // sorted
    let dst_gids = vec![200i32, 400];

    let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
    let src_gpu = cuda_dev.memcpy_stod(&src_gids)?;
    let dst_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

    arena_compact_patch(
        &mut bt_gpu,
        block_table.len(),
        &src_gpu,
        &dst_gpu,
        src_gids.len(),
        &stream,
    )?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&bt_gpu)?;
    let expected = vec![10i32, 200, -1, 30, 400, 50];
    assert_eq!(result, expected, "patch must rewrite matched GIDs");
    Ok(())
}

#[test]
fn arena_compact_patch_no_matches() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let block_table = vec![1i32, 2, 3, 4, 5];
    let src_gids = vec![100i32, 200];
    let dst_gids = vec![999i32, 888];

    let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
    let src_gpu = cuda_dev.memcpy_stod(&src_gids)?;
    let dst_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

    arena_compact_patch(
        &mut bt_gpu,
        block_table.len(),
        &src_gpu,
        &dst_gpu,
        src_gids.len(),
        &stream,
    )?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&bt_gpu)?;
    assert_eq!(result, block_table, "no matches → table unchanged");
    Ok(())
}

#[test]
fn arena_compact_patch_all_entries_match() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let block_table = vec![5i32, 10, 15];
    let src_gids = vec![5i32, 10, 15]; // sorted
    let dst_gids = vec![50i32, 100, 150];

    let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
    let src_gpu = cuda_dev.memcpy_stod(&src_gids)?;
    let dst_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

    arena_compact_patch(
        &mut bt_gpu,
        block_table.len(),
        &src_gpu,
        &dst_gpu,
        src_gids.len(),
        &stream,
    )?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&bt_gpu)?;
    assert_eq!(result, vec![50i32, 100, 150]);
    Ok(())
}

#[test]
fn arena_compact_roundtrip_copy_then_patch() -> Result<()> {
    // End-to-end: copy data, then patch a block table.
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let stride: usize = 512;
    let num_moves = 2;

    // Allocate "arenas": 2 src, 2 dst
    let src_data_0: Vec<u8> = (0..stride).map(|i| (i % 200) as u8).collect();
    let src_data_1: Vec<u8> = (0..stride).map(|i| ((i + 100) % 200) as u8).collect();

    let src_gpu_0 = cuda_dev.memcpy_stod(&src_data_0)?;
    let src_gpu_1 = cuda_dev.memcpy_stod(&src_data_1)?;
    let dst_gpu_0 = cuda_dev.memcpy_stod(&vec![0u8; stride])?;
    let dst_gpu_1 = cuda_dev.memcpy_stod(&vec![0u8; stride])?;

    let moves = vec![
        CompactMove {
            dst: dst_gpu_0.device_ptr(&stream).0 as u64,
            src: src_gpu_0.device_ptr(&stream).0 as u64,
            stride_bytes: stride as u32,
            _pad: 0,
        },
        CompactMove {
            dst: dst_gpu_1.device_ptr(&stream).0 as u64,
            src: src_gpu_1.device_ptr(&stream).0 as u64,
            stride_bytes: stride as u32,
            _pad: 0,
        },
    ];
    let moves_gpu = cuda_dev.memcpy_stod(&moves)?;

    // Copy
    arena_compact_copy(&moves_gpu, num_moves, 128, &stream)?;

    // Patch: GIDs 100→1000, 200→2000
    let block_table = vec![50i32, 100, -1, 200, 300];
    let src_gids = vec![100i32, 200];
    let dst_gids = vec![1000i32, 2000];

    let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
    let src_gids_gpu = cuda_dev.memcpy_stod(&src_gids)?;
    let dst_gids_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

    arena_compact_patch(
        &mut bt_gpu,
        block_table.len(),
        &src_gids_gpu,
        &dst_gids_gpu,
        src_gids.len(),
        &stream,
    )?;
    dev.synchronize()?;

    // Verify copy
    let r0 = cuda_dev.memcpy_dtov(&dst_gpu_0)?;
    let r1 = cuda_dev.memcpy_dtov(&dst_gpu_1)?;
    assert_eq!(r0, src_data_0, "copy move 0");
    assert_eq!(r1, src_data_1, "copy move 1");

    // Verify patch
    let bt_result = cuda_dev.memcpy_dtov(&bt_gpu)?;
    assert_eq!(bt_result, vec![50, 1000, -1, 2000, 300], "patch result");
    Ok(())
}

#[test]
fn arena_compact_copy_mixed_strides() -> Result<()> {
    // Single launch with mixed strides: one move is 2048 bytes, another is 256.
    // The greedy loop handles both — threads beyond the stride just skip.
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let stride_large: usize = 2048;
    let stride_small: usize = 256;

    let src_large: Vec<u8> = (0..stride_large).map(|i| (i % 199) as u8).collect();
    let src_small: Vec<u8> = (0..stride_small).map(|i| (i % 131) as u8).collect();

    let src_gpu_l = cuda_dev.memcpy_stod(&src_large)?;
    let src_gpu_s = cuda_dev.memcpy_stod(&src_small)?;
    // Dst buffers must be at least as large as the stride
    let dst_gpu_l = cuda_dev.memcpy_stod(&vec![0u8; stride_large])?;
    let dst_gpu_s = cuda_dev.memcpy_stod(&vec![0u8; stride_small])?;

    let moves = vec![
        CompactMove {
            dst: dst_gpu_l.device_ptr(&stream).0 as u64,
            src: src_gpu_l.device_ptr(&stream).0 as u64,
            stride_bytes: stride_large as u32,
            _pad: 0,
        },
        CompactMove {
            dst: dst_gpu_s.device_ptr(&stream).0 as u64,
            src: src_gpu_s.device_ptr(&stream).0 as u64,
            stride_bytes: stride_small as u32,
            _pad: 0,
        },
    ];
    let moves_gpu = cuda_dev.memcpy_stod(&moves)?;

    // blockDim=128, blocks_per_move=1 — large move uses full block, small move uses 1 thread
    arena_compact_copy(&moves_gpu, 2, 128, &stream)?;
    dev.synchronize()?;

    let r_large = cuda_dev.memcpy_dtov(&dst_gpu_l)?;
    let r_small = cuda_dev.memcpy_dtov(&dst_gpu_s)?;
    assert_eq!(r_large, src_large, "large stride move");
    assert_eq!(r_small, src_small, "small stride move");
    Ok(())
}

// =============================================================================
// Async copy test
// =============================================================================

#[test]
fn arena_compact_copy_async_basic() -> Result<()> {
    use candle_core::quantized::arena_compact_copy_async;
    use candle_core::quantized::pinned_staging::PinnedStager;

    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let stager = PinnedStager::new(&cuda_dev);

    let stride: usize = 2048;
    let num_moves = 3;

    let mut src_gpus = Vec::new();
    let mut dst_gpus = Vec::new();
    let mut src_datas = Vec::new();
    for i in 0..num_moves {
        let data: Vec<u8> = (0..stride).map(|j| ((i * 41 + j) % 251) as u8).collect();
        src_gpus.push(cuda_dev.memcpy_stod(&data)?);
        dst_gpus.push(cuda_dev.memcpy_stod(&vec![0u8; stride])?);
        src_datas.push(data);
    }

    let moves: Vec<CompactMove> = (0..num_moves)
        .map(|i| CompactMove {
            dst: dst_gpus[i].device_ptr(&stream).0 as u64,
            src: src_gpus[i].device_ptr(&stream).0 as u64,
            stride_bytes: stride as u32,
            _pad: 0,
        })
        .collect();

    // Use async path — host slice, no pre-upload
    arena_compact_copy_async(&moves, 128, &stream, &stager)?;
    dev.synchronize()?;

    for i in 0..num_moves {
        let result = cuda_dev.memcpy_dtov(&dst_gpus[i])?;
        assert_eq!(result, src_datas[i], "async move {i} mismatch");
    }
    stager.flush()?;
    Ok(())
}

// =============================================================================
// Performance benchmarks (run with --ignored --nocapture)
// =============================================================================

/// Benchmark: measure kernel launch overhead + throughput for the async copy kernel.
/// Reports: calls/sec (single-move launches) and blocks/sec (batched launches).
/// The PinnedStager is kept alive for the duration so pinned buffers are reused
/// via the deferred-cleanup queue rather than allocated/freed per call.
#[test]
#[ignore]
fn perf_arena_compact_copy() -> Result<()> {
    use candle_core::quantized::arena_compact_copy_async;
    use candle_core::quantized::pinned_staging::PinnedStager;

    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let stager = PinnedStager::new(&cuda_dev);

    // Allocate a pool of src/dst buffers to avoid measuring allocation
    let stride: usize = 2048; // F16 stride — common case
    let pool_size = 4096; // max moves in one launch
    let buf_bytes = pool_size * stride;

    // One big src buffer, one big dst buffer — moves index into them
    let src_host: Vec<u8> = (0..buf_bytes).map(|i| (i % 251) as u8).collect();
    let src_gpu = cuda_dev.memcpy_stod(&src_host)?;
    let dst_gpu = cuda_dev.memcpy_stod(&vec![0xFFu8; buf_bytes])?;

    let src_base = src_gpu.device_ptr(&stream).0 as u64;
    let dst_base = dst_gpu.device_ptr(&stream).0 as u64;

    // Pre-build move arrays for various batch sizes
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 4000];
    let warmup = 50;
    let iters = 500;

    println!("\n=== arena_compact_copy_async perf (stride={stride}, blockDim=128) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "moves", "calls/sec", "moves/sec", "GB/s"
    );

    for &n_moves in &batch_sizes {
        let moves: Vec<CompactMove> = (0..n_moves)
            .map(|i| CompactMove {
                dst: dst_base + (i * stride) as u64,
                src: src_base + (i * stride) as u64,
                stride_bytes: stride as u32,
                _pad: 0,
            })
            .collect();

        // Warmup
        for _ in 0..warmup {
            arena_compact_copy_async(&moves, 128, &stream, &stager)?;
        }
        dev.synchronize()?;
        stager.flush()?;

        // Timed
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            arena_compact_copy_async(&moves, 128, &stream, &stager)?;
        }
        dev.synchronize()?;
        stager.flush()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let calls_per_sec = iters as f64 / elapsed;
        let moves_per_sec = (iters * n_moves) as f64 / elapsed;
        let bytes_per_sec = moves_per_sec * stride as f64;
        let gb_per_sec = bytes_per_sec / 1e9;

        println!(
            "{:>8} {:>12.0} {:>12.0} {:>10.2}",
            n_moves, calls_per_sec, moves_per_sec, gb_per_sec
        );
    }

    // Validate one batch to make sure perf loop didn't corrupt anything
    let n_check = 100;
    // Reset dst to zeros
    let dst_gpu = cuda_dev.memcpy_stod(&vec![0u8; buf_bytes])?;
    let dst_base = dst_gpu.device_ptr(&stream).0 as u64;
    let moves: Vec<CompactMove> = (0..n_check)
        .map(|i| CompactMove {
            dst: dst_base + (i * stride) as u64,
            src: src_base + (i * stride) as u64,
            stride_bytes: stride as u32,
            _pad: 0,
        })
        .collect();
    arena_compact_copy_async(&moves, 128, &stream, &stager)?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&dst_gpu)?;
    for i in 0..n_check {
        let off = i * stride;
        assert_eq!(
            &result[off..off + stride],
            &src_host[off..off + stride],
            "validation failed at move {i}"
        );
    }
    println!("Validation: {n_check} moves verified byte-exact.");

    Ok(())
}

/// Benchmark: measure patch kernel throughput.
/// Reports: calls/sec and entries/sec for various block_table sizes and move counts.
#[test]
#[ignore]
fn perf_arena_compact_patch() -> Result<()> {
    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();

    let warmup = 50;
    let iters = 500;

    // Build a block table with sequential GIDs and a move table that remaps some
    let configs: Vec<(usize, usize)> = vec![
        (1_000, 100),
        (10_000, 500),
        (100_000, 1_000),
        (262_144, 2_000), // Llama-3-8B scale
        (1_000_000, 4_000),
    ];

    println!("\n=== arena_compact_patch perf ===");
    println!(
        "{:>10} {:>8} {:>12} {:>14}",
        "entries", "moves", "calls/sec", "entries/sec"
    );

    for &(num_entries, num_moves) in &configs {
        // Block table: 0..num_entries, moves remap every (num_entries/num_moves)-th entry
        let block_table: Vec<i32> = (0..num_entries as i32).collect();
        let step = (num_entries / num_moves).max(1);
        let mut src_gids: Vec<i32> = (0..num_moves).map(|i| (i * step) as i32).collect();
        src_gids.sort(); // already sorted but be explicit
        let dst_gids: Vec<i32> = src_gids.iter().map(|&g| g + 1_000_000).collect();

        let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
        let src_gpu = cuda_dev.memcpy_stod(&src_gids)?;
        let dst_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

        // Warmup
        for _ in 0..warmup {
            arena_compact_patch(
                &mut bt_gpu,
                num_entries,
                &src_gpu,
                &dst_gpu,
                num_moves,
                &stream,
            )?;
        }
        dev.synchronize()?;

        // Timed — we don't reset between iterations since the patch is idempotent
        // for measurement purposes (we just want to time the kernel)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            arena_compact_patch(
                &mut bt_gpu,
                num_entries,
                &src_gpu,
                &dst_gpu,
                num_moves,
                &stream,
            )?;
        }
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let calls_per_sec = iters as f64 / elapsed;
        let entries_per_sec = (iters * num_entries) as f64 / elapsed;

        println!(
            "{:>10} {:>8} {:>12.0} {:>14.0}",
            num_entries, num_moves, calls_per_sec, entries_per_sec
        );
    }

    // Validate the last config
    let (num_entries, num_moves) = *configs.last().unwrap();
    let block_table: Vec<i32> = (0..num_entries as i32).collect();
    let step = (num_entries / num_moves).max(1);
    let mut src_gids: Vec<i32> = (0..num_moves).map(|i| (i * step) as i32).collect();
    src_gids.sort();
    let dst_gids: Vec<i32> = src_gids.iter().map(|&g| g + 1_000_000).collect();

    let mut bt_gpu = cuda_dev.memcpy_stod(&block_table)?;
    let src_gpu = cuda_dev.memcpy_stod(&src_gids)?;
    let dst_gpu = cuda_dev.memcpy_stod(&dst_gids)?;

    arena_compact_patch(
        &mut bt_gpu,
        num_entries,
        &src_gpu,
        &dst_gpu,
        num_moves,
        &stream,
    )?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&bt_gpu)?;
    let src_set: std::collections::HashSet<i32> = src_gids.iter().copied().collect();
    let mut validated = 0;
    for i in 0..num_entries {
        let original = i as i32;
        if src_set.contains(&original) {
            assert_eq!(
                result[i],
                original + 1_000_000,
                "entry {i}: expected patched value"
            );
            validated += 1;
        } else {
            assert_eq!(result[i], original, "entry {i}: should be unchanged");
        }
    }
    println!("Validation: {validated}/{num_moves} patched entries verified.");

    Ok(())
}

/// Benchmark: mixed-stride async copy — simulates realistic compaction with
/// different format sizes in a single launch.
#[test]
#[ignore]
fn perf_arena_compact_copy_mixed() -> Result<()> {
    use candle_core::quantized::arena_compact_copy_async;
    use candle_core::quantized::pinned_staging::PinnedStager;

    let (dev, cuda_dev) = match get_cuda_dev() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let stream = cuda_dev.cuda_stream();
    let stager = PinnedStager::new(&cuda_dev);

    // Mix of strides matching real format distribution:
    // 40% F16 (2048), 30% Q8_0 (1088), 20% Q4_0 (576), 10% Q2_0 (160)
    let strides = [2048usize, 1088, 576, 160];
    let weights = [40, 30, 20, 10]; // out of 100
    let total_moves = 2000;

    // Build the stride list
    let mut move_strides = Vec::with_capacity(total_moves);
    for (&s, &w) in strides.iter().zip(weights.iter()) {
        let count = total_moves * w / 100;
        for _ in 0..count {
            move_strides.push(s);
        }
    }
    // Pad to exactly total_moves
    while move_strides.len() < total_moves {
        move_strides.push(strides[0]);
    }

    // Allocate one big buffer for all moves
    let max_stride = *strides.iter().max().unwrap();
    let buf_bytes = total_moves * max_stride;
    let src_host: Vec<u8> = (0..buf_bytes).map(|i| (i % 239) as u8).collect();
    let src_gpu = cuda_dev.memcpy_stod(&src_host)?;
    let dst_gpu = cuda_dev.memcpy_stod(&vec![0u8; buf_bytes])?;
    let src_base = src_gpu.device_ptr(&stream).0 as u64;
    let dst_base = dst_gpu.device_ptr(&stream).0 as u64;

    let moves: Vec<CompactMove> = move_strides
        .iter()
        .enumerate()
        .map(|(i, &s)| CompactMove {
            dst: dst_base + (i * max_stride) as u64,
            src: src_base + (i * max_stride) as u64,
            stride_bytes: s as u32,
            _pad: 0,
        })
        .collect();

    let warmup = 50;
    let iters = 500;

    // Warmup
    for _ in 0..warmup {
        arena_compact_copy_async(&moves, 128, &stream, &stager)?;
    }
    dev.synchronize()?;
    stager.flush()?;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        arena_compact_copy_async(&moves, 128, &stream, &stager)?;
    }
    dev.synchronize()?;
    stager.flush()?;
    let elapsed = t0.elapsed().as_secs_f64();

    let calls_per_sec = iters as f64 / elapsed;
    let moves_per_sec = (iters * total_moves) as f64 / elapsed;
    let total_bytes_per_call: usize = move_strides.iter().sum();
    let gb_per_sec = (total_bytes_per_call as f64 * iters as f64) / elapsed / 1e9;

    println!("\n=== arena_compact_copy_async mixed-stride perf ===");
    println!("  {total_moves} moves/call, distribution:");
    for (&s, &w) in strides.iter().zip(weights.iter()) {
        println!("    stride={s:>5}: {w}%");
    }
    println!("  calls/sec:  {calls_per_sec:.0}");
    println!("  moves/sec:  {moves_per_sec:.0}");
    println!("  throughput: {gb_per_sec:.2} GB/s");
    println!("  avg µs/call: {:.1}", elapsed * 1e6 / iters as f64);

    // Validate
    let dst_gpu = cuda_dev.memcpy_stod(&vec![0u8; buf_bytes])?;
    let dst_base = dst_gpu.device_ptr(&stream).0 as u64;
    let moves: Vec<CompactMove> = move_strides
        .iter()
        .enumerate()
        .map(|(i, &s)| CompactMove {
            dst: dst_base + (i * max_stride) as u64,
            src: src_base + (i * max_stride) as u64,
            stride_bytes: s as u32,
            _pad: 0,
        })
        .collect();
    arena_compact_copy_async(&moves, 128, &stream, &stager)?;
    dev.synchronize()?;

    let result = cuda_dev.memcpy_dtov(&dst_gpu)?;
    for (i, &s) in move_strides.iter().enumerate() {
        let off = i * max_stride;
        assert_eq!(
            &result[off..off + s],
            &src_host[off..off + s],
            "mixed move {i} (stride={s}) validation failed"
        );
    }
    println!("Validation: {total_moves} mixed-stride moves verified byte-exact.");

    Ok(())
}
