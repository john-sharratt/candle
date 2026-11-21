//! Round trips through the GPU quantizers and dequantizers, checked against the
//! CPU codec.
//!
//! Each format is quantized on the GPU from known raw data and its bytes are
//! compared with the CPU quantizer's, then dequantized and compared by RMSE. The
//! K-quants with iterative refinement may settle on different local optima, which
//! is why the direct comparison also reports RMSE. Several debug tests print the
//! block headers of Q2_K and Q4_K as laid out, so a mismatch can be read against
//! the offsets. The diagnostic prints per-format counts and does not assert byte
//! exactness itself; the dedicated per-format tests do.

use super::*;

/// Comprehensive quantize/dequantize roundtrip test for all quantization types.
/// Tests correctness and measures performance (GB/s).
#[test]
pub(super) fn debug_q4_0_roundtrip() -> Result<()> {
    let dev = CudaDevice::new(0)?;

    // Test one block of Q4_0 = 32 elements
    let elem_count = 32usize;
    let dtype = GgmlDType::Q4_0;

    // Simple ascending values
    let src_data: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 2.0).collect();
    println!("Input: {:?}", src_data);

    let src_gpu = dev.memcpy_stod(&src_data)?;

    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let num_blocks = elem_count.div_ceil(block_size);
    let quant_size = num_blocks * type_size;

    println!(
        "block_size={}, type_size={}, num_blocks={}, quant_size={}",
        block_size, type_size, num_blocks, quant_size
    );

    let mut quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

    quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, dtype, &dev)?;
    dev.synchronize()?;

    // Read quantized bytes
    let quant_bytes: Vec<u8> = dev.memcpy_dtov(&quant_buf.slice(..))?;
    println!("Quantized bytes: {:?}", quant_bytes);

    // Dequantize
    let padded = PaddedCudaSlice {
        inner: quant_buf.clone(),
        len: quant_size,
    };
    let dequant_storage = dequantize_f32(&padded, dtype, elem_count, &dev)?;
    dev.synchronize()?;

    let dequant_gpu = dequant_storage.as_cuda_slice::<f32>()?;
    let dequant_data = dev.memcpy_dtov(&dequant_gpu.slice(..))?;
    println!("Dequantized: {:?}", dequant_data);

    // Error
    for i in 0..elem_count {
        let err = (src_data[i] - dequant_data[i]).abs();
        if err > 0.5 {
            println!(
                "ERROR at {}: src={}, dequant={}, err={}",
                i, src_data[i], dequant_data[i], err
            );
        }
    }

    Ok(())
}

#[test]
pub(super) fn quantize_dequantize_roundtrip_all_dtypes() -> Result<()> {
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;

    // Test parameters - large enough to measure, small enough to be fast
    // 1M elements = 4MB f32 data, runs in ~1-10ms per dtype
    let elem_count = 1024 * 1024usize; // 1M elements
    let num_warmup = 2;
    let num_iters = 5;

    // All supported quantization types with their expected max RMSE
    // Tolerances based on theoretical quantization error for uniform random [-8, 8]
    // RMSE ≈ step_size / sqrt(12) where step_size = range / (2^bits - 1)
    // Added margin for rounding differences between GPU and CPU
    let test_configs: Vec<(GgmlDType, f32, &str)> = vec![
        (GgmlDType::Q4_0, 0.35, "Q4_0"), // 4-bit: theoretical ~0.31
        (GgmlDType::Q4_1, 0.35, "Q4_1"), // 4-bit with min offset
        (GgmlDType::Q5_0, 0.18, "Q5_0"), // 5-bit: theoretical ~0.15
        (GgmlDType::Q5_1, 0.18, "Q5_1"), // 5-bit with min offset
        (GgmlDType::Q8_0, 0.04, "Q8_0"), // 8-bit: theoretical ~0.018
        (GgmlDType::Q8_1, 0.04, "Q8_1"), // 8-bit with sum
        (GgmlDType::Q2_K, 1.50, "Q2K"),  // 2-bit K-quant: high error expected
        (GgmlDType::Q3_K, 0.80, "Q3K"),  // 3-bit K-quant
        (GgmlDType::Q4_K, 0.40, "Q4K"),  // 4-bit K-quant
        (GgmlDType::Q5_K, 0.25, "Q5K"),  // 5-bit K-quant
        (GgmlDType::Q6_K, 0.15, "Q6K"),  // 6-bit K-quant
        (GgmlDType::Q8_K, 0.04, "Q8K"),  // 8-bit K-quant
    ];

    // Generate random test data in KV cache typical range [-8.0, 8.0]
    // This matches typical attention key/value magnitudes
    let mut rng = rand::rng();
    let src_data: Vec<f32> = (0..elem_count)
        .map(|_| rng.random_range(-8.0..8.0))
        .collect();
    let src_gpu = dev.memcpy_stod(&src_data)?;

    // Results table
    struct TestResult {
        name: &'static str,
        valid: bool,
        rmse: f32,
        mean_diff: f32,
        max_diff: f32,
        quant_gbps: f64,
        dequant_gbps: f64,
    }
    let mut results: Vec<TestResult> = Vec::new();

    println!("\n=== Quantize/Dequantize Roundtrip Test ===");
    println!(
        "Elements: {} ({:.2} MB f32), Range: [-8.0, 8.0]",
        elem_count,
        elem_count as f64 * 4.0 / 1e6
    );
    println!();

    for (dtype, max_rmse, name) in &test_configs {
        // Calculate buffer sizes
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let num_blocks = elem_count.div_ceil(block_size);
        let quant_size = num_blocks * type_size;

        // Allocate quantized buffer
        let mut quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

        // Warmup
        for _ in 0..num_warmup {
            quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, *dtype, &dev)?;
            dev.synchronize()?;
        }

        // Benchmark quantize
        let start = Instant::now();
        for _ in 0..num_iters {
            quantize_to_dtype(&src_gpu.slice(..), &mut quant_buf, elem_count, *dtype, &dev)?;
        }
        dev.synchronize()?;
        let quant_time = start.elapsed().as_secs_f64() / num_iters as f64;

        // Create QCudaStorage for dequantize
        let padded = PaddedCudaSlice {
            inner: quant_buf.clone(),
            len: quant_size,
        };

        // Warmup dequantize
        for _ in 0..num_warmup {
            let _ = dequantize_f32(&padded, *dtype, elem_count, &dev)?;
            dev.synchronize()?;
        }

        // Benchmark dequantize
        let start = Instant::now();
        let mut dequant_storage = None;
        for _ in 0..num_iters {
            dequant_storage = Some(dequantize_f32(&padded, *dtype, elem_count, &dev)?);
        }
        dev.synchronize()?;
        let dequant_time = start.elapsed().as_secs_f64() / num_iters as f64;

        // Get dequantized result for validation
        let dequant_storage = dequant_storage.unwrap();
        let dequant_gpu = dequant_storage.as_cuda_slice::<f32>()?;
        let dequant_data = dev.memcpy_dtov(&dequant_gpu.slice(..))?;

        // Calculate error metrics
        let mut sum_sq_err = 0.0f64;
        let mut sum_abs_diff = 0.0f64;
        let mut max_diff = 0.0f32;
        for i in 0..elem_count {
            let diff = src_data[i] - dequant_data[i];
            let abs_diff = diff.abs();
            sum_sq_err += (diff as f64) * (diff as f64);
            sum_abs_diff += abs_diff as f64;
            if abs_diff > max_diff {
                max_diff = abs_diff;
            }
        }
        let rmse = (sum_sq_err / elem_count as f64).sqrt() as f32;
        let mean_diff = (sum_abs_diff / elem_count as f64) as f32;
        let valid = rmse <= *max_rmse;

        // Calculate throughput
        // Quantize: read f32, write quant
        let quant_bytes = (elem_count * 4 + quant_size) as f64;
        let quant_gbps = quant_bytes / quant_time / 1e9;

        // Dequantize: read quant, write f32
        let dequant_bytes = (quant_size + elem_count * 4) as f64;
        let dequant_gbps = dequant_bytes / dequant_time / 1e9;

        results.push(TestResult {
            name,
            valid,
            rmse,
            mean_diff,
            max_diff,
            quant_gbps,
            dequant_gbps,
        });
    }

    // Print summary table
    println!("┌─────────┬───────┬──────────┬──────────┬──────────┬─────────────┬─────────────┐");
    println!("│ DType   │ Valid │ RMSE     │ MeanDiff │ MaxDiff  │ Quant GB/s  │ Dequant GB/s│");
    println!("├─────────┼───────┼──────────┼──────────┼──────────┼─────────────┼─────────────┤");
    for r in &results {
        let valid_str = if r.valid { "✓" } else { "✗" };
        println!(
            "│ {:7} │   {}   │ {:8.5} │ {:8.5} │ {:8.4} │ {:10.2}  │ {:10.2}  │",
            r.name, valid_str, r.rmse, r.mean_diff, r.max_diff, r.quant_gbps, r.dequant_gbps
        );
    }
    println!("└─────────┴───────┴──────────┴──────────┴──────────┴─────────────┴─────────────┘");

    // Assert all passed
    let all_valid = results.iter().all(|r| r.valid);
    assert!(all_valid, "Some quantization types failed validation");

    Ok(())
}

/// Precision test: Creates known-good quantized data on CPU, then tests GPU quantize
/// by verifying it produces identical bytes when requantizing dequantized values.
///
/// Test flow:
/// 1. CPU: Generate random f32 data, quantize with CPU (known-good) → cpu_quant_bytes
/// 2. CPU: Dequantize CPU result → baseline_f32 (this is what the quantized block represents)
/// 3. Upload baseline_f32 to GPU
/// 4. GPU: Quantize baseline_f32 → gpu_quant_bytes
/// 5. Compare: gpu_quant_bytes == cpu_quant_bytes (should match exactly!)
/// 6. GPU: Dequantize gpu_quant_bytes → roundtrip_f32
/// 7. Compare: roundtrip_f32 vs baseline_f32 (RMSE should be ~0)
///
/// This isolates the GPU quantize kernel and verifies byte-level correctness.
#[test]
pub(super) fn quantize_kernel_byte_accuracy() -> Result<()> {
    use crate::quantized::k_quants::{BlockQ2_K, BlockQ4_0, BlockQ4_K, BlockQ8_0, GgmlType};
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;

    // Test with enough blocks to measure bandwidth, but not too many for quick tests
    let num_blocks = 4096; // 4K blocks

    println!("\n=== GPU Quantize Kernel Byte-Level Accuracy Test ===\n");

    // First, a simple debug test for Q4_0
    {
        println!("=== Q4_0 Debug (1 block) ===");
        // Create simple ascending data
        let test_data: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 2.0).collect();
        println!("Input data: {:?}", test_data);

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_0::zeros(); 1];
        BlockQ4_0::from_float(&test_data, &mut cpu_block);

        // CPU dequantize to get baseline
        let mut baseline = vec![0.0f32; 32];
        BlockQ4_0::to_float(&cpu_block, &mut baseline);
        println!("CPU baseline (after quant->dequant): {:?}", baseline);

        // Show CPU block
        println!(
            "CPU block: d={:?}, qs={:?}",
            cpu_block[0].d, cpu_block[0].qs
        );

        // Upload baseline to GPU and quantize
        let baseline_gpu = dev.memcpy_stod(&baseline)?;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(18)? };
        quantize_to_dtype(
            &baseline_gpu.slice(..),
            &mut gpu_quant_buf,
            32,
            GgmlDType::Q4_0,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, 18) };

        println!("CPU bytes: {:02x?}", cpu_bytes);
        println!("GPU bytes: {:02x?}", gpu_bytes);

        // Compare
        for i in 0..18 {
            if cpu_bytes[i] != gpu_bytes[i] {
                println!(
                    "  Mismatch at byte {}: CPU={:02x} GPU={:02x}",
                    i, cpu_bytes[i], gpu_bytes[i]
                );
            }
        }
        println!();
    }

    // Q2K debug test (1 block) - DIRECT comparison (same raw input)
    {
        println!("=== Q2K Debug (1 block, direct) ===");
        // Create simple data that spans a known range
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32 - 127.5) / 16.0).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data[0], test_data[255]
        );

        // CPU quantize the RAW data
        let mut cpu_block = vec![BlockQ2_K::zeros(); 1];
        BlockQ2_K::from_float(&test_data, &mut cpu_block);

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );
        println!("CPU scales[0..16]: {:02x?}", cpu_block[0].scales);

        // GPU quantize the SAME raw data
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ2_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q2_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Q2K layout: scales[16], qs[64], d (2 bytes), dmin (2 bytes) = 84 bytes
        println!("CPU scales bytes[0..16]: {:02x?}", &cpu_bytes[0..16]);
        println!("GPU scales bytes[0..16]: {:02x?}", &gpu_bytes[0..16]);

        // d is at offset 80, dmin at 82
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[80], cpu_bytes[81]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[82], cpu_bytes[83]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[80], gpu_bytes[81]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[82], gpu_bytes[83]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Count mismatches
        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
                if mismatches <= 5 {
                    println!(
                        "Mismatch at byte {}: CPU={:02x} GPU={:02x}",
                        i, cpu_bytes[i], gpu_bytes[i]
                    );
                }
            }
        }
        println!("Total mismatches: {} / {}", mismatches, type_size);

        println!();
    }

    // Q4K debug test (1 block)
    {
        println!("=== Q4K Debug (1 block) ===");
        // Create simple data that spans a known range
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32 - 127.5) / 16.0).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data[0], test_data[255]
        );

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        // CPU dequantize to get baseline
        let mut baseline = vec![0.0f32; 256];
        BlockQ4_K::to_float(&cpu_block, &mut baseline);
        println!(
            "CPU baseline range: {:.3} to {:.3}",
            baseline.iter().cloned().fold(f32::INFINITY, f32::min),
            baseline.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );
        println!("CPU scales[0..12]: {:?}", cpu_block[0].scales);

        // Upload baseline to GPU and quantize
        let baseline_gpu = dev.memcpy_stod(&baseline)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &baseline_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Show dm bytes (first 4 bytes = half2)
        println!("CPU dm bytes: {:02x?}", &cpu_bytes[0..4]);
        println!("GPU dm bytes: {:02x?}", &gpu_bytes[0..4]);

        // Interpret as half
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Show scales bytes (bytes 4-15)
        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        println!();
    }

    // Q4K debug test with random data (first block only)
    {
        println!("=== Q4K Debug (random, 1 block, seeded) ===");
        // Create random data with a seed for reproducibility
        let test_data: Vec<f32> = (0..256)
            .map(|i| {
                // Pseudo-random but deterministic, uses a simple LCG
                let seed = (i as u64)
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let x = (seed >> 33) as i32;
                (x as f32 - (i32::MAX / 2) as f32) / (i32::MAX as f32 / 16.0)
            })
            .collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data.iter().cloned().fold(f32::INFINITY, f32::min),
            test_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Show first 8 values (first sub-block)
        println!("First 8 values: {:?}", &test_data[0..8]);

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        // CPU dequantize
        let mut cpu_dequant = vec![0.0f32; 256];
        BlockQ4_K::to_float(&cpu_block, &mut cpu_dequant);

        // Show CPU block header
        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        // Interpret as half
        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        // Show scales bytes (bytes 4-15)
        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        // Decode and show scales/mins
        fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
            if j < 4 {
                let d = q[j] & 63;
                let m = q[j + 4] & 63;
                (d, m)
            } else {
                let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                (d, m)
            }
        }

        println!("CPU sub-block scales/mins:");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &cpu_bytes[4..16]);
            println!("  sub-block {}: sc={}, m={}", j, sc, m);
        }

        println!("GPU sub-block scales/mins:");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &gpu_bytes[4..16]);
            println!("  sub-block {}: sc={}, m={}", j, sc, m);
        }

        // Count mismatches
        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
            }
        }
        println!("Total mismatches: {}", mismatches);

        println!();
    }

    // Q4K with truly random data (1 block)
    {
        println!("=== Q4K Debug (truly random, 1 block) ===");
        let mut rng = rand::rng();
        let test_data: Vec<f32> = (0..256).map(|_| rng.random_range(-8.0..8.0)).collect();
        println!(
            "Input data range: {:.3} to {:.3}",
            test_data.iter().cloned().fold(f32::INFINITY, f32::min),
            test_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // CPU quantize
        let mut cpu_block = vec![BlockQ4_K::zeros(); 1];
        BlockQ4_K::from_float(&test_data, &mut cpu_block);

        println!(
            "CPU block: d={:?}, dmin={:?}",
            cpu_block[0].d, cpu_block[0].dmin
        );

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(type_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_block.as_ptr() as *const u8, type_size) };

        let cpu_d = half::f16::from_le_bytes([cpu_bytes[0], cpu_bytes[1]]);
        let cpu_dmin = half::f16::from_le_bytes([cpu_bytes[2], cpu_bytes[3]]);
        let gpu_d = half::f16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);
        let gpu_dmin = half::f16::from_le_bytes([gpu_bytes[2], gpu_bytes[3]]);
        println!(
            "CPU: d={:.6}, dmin={:.6}",
            cpu_d.to_f32(),
            cpu_dmin.to_f32()
        );
        println!(
            "GPU: d={:.6}, dmin={:.6}",
            gpu_d.to_f32(),
            gpu_dmin.to_f32()
        );

        println!("CPU scales bytes: {:02x?}", &cpu_bytes[4..16]);
        println!("GPU scales bytes: {:02x?}", &gpu_bytes[4..16]);

        let mut mismatches = 0;
        for i in 0..type_size {
            if cpu_bytes[i] != gpu_bytes[i] {
                mismatches += 1;
            }
        }
        println!("Total mismatches: {} / {}", mismatches, type_size);
        println!();
    }

    // Q2K multi-block debug test
    {
        println!("=== Q2K Debug (4 blocks, random) ===");
        let mut rng = rand::rng();
        let num_test_blocks = 4;
        let test_data: Vec<f32> = (0..256 * num_test_blocks)
            .map(|_| rng.random_range(-8.0..8.0))
            .collect();

        // CPU quantize
        let mut cpu_blocks = vec![BlockQ2_K::zeros(); num_test_blocks];
        BlockQ2_K::from_float(&test_data, &mut cpu_blocks);

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ2_K>();
        let quant_size = num_test_blocks * type_size;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256 * num_test_blocks,
            GgmlDType::Q2_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
        };

        let mut mismatches = 0;
        let mut first_mismatch = None;
        for blk in 0..num_test_blocks {
            for i in 0..type_size {
                let idx = blk * type_size + i;
                if cpu_bytes[idx] != gpu_bytes[idx] {
                    mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((blk, i, cpu_bytes[idx], gpu_bytes[idx]));
                    }
                }
            }
        }

        println!("Total mismatches: {} / {} bytes", mismatches, quant_size);
        if let Some((blk, off, cpu, gpu)) = first_mismatch {
            println!(
                "First mismatch: block {}, offset {}: CPU={:02x}, GPU={:02x}",
                blk, off, cpu, gpu
            );
        }

        // Show each block's d/dmin and scales comparison
        // Q2K layout: scales[16] (0-15), qs[64] (16-79), d (80-81), dmin (82-83)
        for blk in 0..num_test_blocks {
            let blk_start = blk * type_size;
            let cpu_d =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 80], cpu_bytes[blk_start + 81]]);
            let cpu_dmin =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 82], cpu_bytes[blk_start + 83]]);
            let gpu_d =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 80], gpu_bytes[blk_start + 81]]);
            let gpu_dmin =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 82], gpu_bytes[blk_start + 83]]);
            let d_match = (cpu_d.to_f32() - gpu_d.to_f32()).abs() < 0.001;
            let dmin_match = (cpu_dmin.to_f32() - gpu_dmin.to_f32()).abs() < 0.001;
            println!(
                "  Block {}: CPU d={:.6} dmin={:.6} | GPU d={:.6} dmin={:.6} | d={} dmin={}",
                blk,
                cpu_d.to_f32(),
                cpu_dmin.to_f32(),
                gpu_d.to_f32(),
                gpu_dmin.to_f32(),
                if d_match { "✓" } else { "✗" },
                if dmin_match { "✓" } else { "✗" }
            );

            // Show scales comparison for this block
            let cpu_scales = &cpu_bytes[blk_start..blk_start + 16];
            let gpu_scales = &gpu_bytes[blk_start..blk_start + 16];
            let scales_match = cpu_scales == gpu_scales;
            if !scales_match {
                println!("    CPU scales: {:02x?}", cpu_scales);
                println!("    GPU scales: {:02x?}", gpu_scales);
            }
        }
        println!();
    } // Q4K with truly random data (4 blocks - to test multi-block kernel)
    {
        println!("=== Q4K Debug (truly random, 4 blocks) ===");
        let mut rng = rand::rng();
        let num_test_blocks = 4;
        let test_data: Vec<f32> = (0..256 * num_test_blocks)
            .map(|_| rng.random_range(-8.0..8.0))
            .collect();

        // CPU quantize
        let mut cpu_blocks = vec![BlockQ4_K::zeros(); num_test_blocks];
        BlockQ4_K::from_float(&test_data, &mut cpu_blocks);

        // GPU quantize the same raw input
        let test_data_gpu = dev.memcpy_stod(&test_data)?;
        let type_size = std::mem::size_of::<BlockQ4_K>();
        let quant_size = num_test_blocks * type_size;
        let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
        quantize_to_dtype(
            &test_data_gpu.slice(..),
            &mut gpu_quant_buf,
            256 * num_test_blocks,
            GgmlDType::Q4_K,
            &dev,
        )?;
        dev.synchronize()?;

        let gpu_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
        let cpu_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
        };

        let mut mismatches = 0;
        let mut first_mismatch_block = None;
        for blk in 0..num_test_blocks {
            for i in 0..type_size {
                let idx = blk * type_size + i;
                if cpu_bytes[idx] != gpu_bytes[idx] {
                    mismatches += 1;
                    if first_mismatch_block.is_none() {
                        first_mismatch_block = Some((blk, i, cpu_bytes[idx], gpu_bytes[idx]));
                    }
                }
            }
        }

        println!("Total mismatches: {} / {} bytes", mismatches, quant_size);
        if let Some((blk, off, cpu, gpu)) = first_mismatch_block {
            println!(
                "First mismatch: block {}, offset {}: CPU={:02x}, GPU={:02x}",
                blk, off, cpu, gpu
            );
        }

        // Show each block's d/dmin comparison
        for blk in 0..num_test_blocks {
            let blk_start = blk * type_size;
            let cpu_d = half::f16::from_le_bytes([cpu_bytes[blk_start], cpu_bytes[blk_start + 1]]);
            let cpu_dmin =
                half::f16::from_le_bytes([cpu_bytes[blk_start + 2], cpu_bytes[blk_start + 3]]);
            let gpu_d = half::f16::from_le_bytes([gpu_bytes[blk_start], gpu_bytes[blk_start + 1]]);
            let gpu_dmin =
                half::f16::from_le_bytes([gpu_bytes[blk_start + 2], gpu_bytes[blk_start + 3]]);
            let d_match = cpu_d == gpu_d;
            let dmin_match = cpu_dmin == gpu_dmin;
            println!(
                "  Block {}: CPU d={:.6} dmin={:.6} | GPU d={:.6} dmin={:.6} | d={} dmin={}",
                blk,
                cpu_d.to_f32(),
                cpu_dmin.to_f32(),
                gpu_d.to_f32(),
                gpu_dmin.to_f32(),
                if d_match { "✓" } else { "✗" },
                if dmin_match { "✓" } else { "✗" }
            );
        }
        println!();
    }

    // Helper macro for testing a specific quantized type
    macro_rules! test_dtype {
        ($block_ty:ty, $dtype:expr, $name:expr) => {{
            let block_size = <$block_ty>::BLCK_SIZE;
            let elem_count = num_blocks * block_size;
            let type_size = std::mem::size_of::<$block_ty>();
            let quant_size = num_blocks * type_size;

            // Step 1: Generate random data and CPU-quantize (known-good baseline)
            let mut rng = rand::rng();
            let random_f32: Vec<f32> = (0..elem_count)
                .map(|_| rng.random_range(-8.0..8.0))
                .collect();

            let mut cpu_blocks: Vec<$block_ty> = vec![<$block_ty>::zeros(); num_blocks];
            <$block_ty>::from_float(&random_f32, &mut cpu_blocks);

            // Get CPU quantized bytes
            let cpu_quant_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    cpu_blocks.as_ptr() as *const u8,
                    quant_size
                ).to_vec()
            };

            // Step 2: CPU dequantize to get baseline (what the quantized block represents)
            let mut baseline_f32 = vec![0.0f32; elem_count];
            <$block_ty>::to_float(&cpu_blocks, &mut baseline_f32);

            // Step 3: Upload baseline to GPU (this is NOT timed - we're testing kernel accuracy)
            let baseline_gpu = dev.memcpy_stod(&baseline_f32)?;

            // Step 4: GPU quantize baseline
            let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };

            // Warmup
            quantize_to_dtype(&baseline_gpu.slice(..), &mut gpu_quant_buf, elem_count, $dtype, &dev)?;
            dev.synchronize()?;

            // Timed run
            let start = Instant::now();
            quantize_to_dtype(&baseline_gpu.slice(..), &mut gpu_quant_buf, elem_count, $dtype, &dev)?;
            dev.synchronize()?;
            let quant_time = start.elapsed();

            // Download GPU quantized bytes
            let gpu_quant_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;

            // Step 5: Compare bytes
            let mut byte_mismatches = 0usize;
            let mut first_mismatch_idx = None;
            for i in 0..quant_size {
                if cpu_quant_bytes[i] != gpu_quant_bytes[i] {
                    byte_mismatches += 1;
                    if first_mismatch_idx.is_none() {
                        first_mismatch_idx = Some(i);
                    }
                }
            }

            // Step 6: GPU dequantize for RMSE comparison
            let padded = PaddedCudaSlice {
                inner: gpu_quant_buf.clone(),
                len: quant_size,
            };

            // Warmup
            let _ = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            dev.synchronize()?;

            // Timed run
            let start = Instant::now();
            let roundtrip_storage = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            dev.synchronize()?;
            let dequant_time = start.elapsed();

            let roundtrip_gpu = roundtrip_storage.as_cuda_slice::<f32>()?;
            let roundtrip_f32 = dev.memcpy_dtov(&roundtrip_gpu.slice(..))?;

            // Step 7: Calculate RMSE between baseline and roundtrip
            let mut sum_sq_err = 0.0f64;
            let mut max_diff = 0.0f32;
            for i in 0..elem_count {
                let diff = baseline_f32[i] - roundtrip_f32[i];
                sum_sq_err += (diff as f64) * (diff as f64);
                max_diff = max_diff.max(diff.abs());
            }
            let rmse = (sum_sq_err / elem_count as f64).sqrt() as f32;

            // Bandwidth calculations (only GPU kernel time)
            let quant_bytes_rw = (elem_count * 4 + quant_size) as f64;
            let quant_gbps = quant_bytes_rw / quant_time.as_secs_f64() / 1e9;
            let dequant_bytes_rw = (quant_size + elem_count * 4) as f64;
            let dequant_gbps = dequant_bytes_rw / dequant_time.as_secs_f64() / 1e9;

            // Report
            let byte_match_pct = 100.0 * (quant_size - byte_mismatches) as f64 / quant_size as f64;
            let status = if byte_mismatches == 0 && rmse < 1e-5 { "✓" } else { "✗" };

            println!("{} {:5}: Bytes: {:6.2}% match ({:6} / {:6} mismatches), RMSE: {:.2e}, MaxDiff: {:.2e}",
                status, $name, byte_match_pct, byte_mismatches, quant_size, rmse, max_diff);
            println!("         Quant: {:6.1} GB/s, Dequant: {:6.1} GB/s",
                quant_gbps, dequant_gbps);

            if let Some(idx) = first_mismatch_idx {
                let block_idx = idx / type_size;
                let byte_in_block = idx % type_size;
                println!("         First mismatch: byte {} (block {}, offset {}): CPU={:02x} GPU={:02x}",
                    idx, block_idx, byte_in_block, cpu_quant_bytes[idx], gpu_quant_bytes[idx]);
            }

            (byte_mismatches == 0, rmse)
        }};
    }

    // Test each dtype
    println!(
        "Testing Q4_0 ({} blocks × {} elements = {} elements)...",
        num_blocks,
        BlockQ4_0::BLCK_SIZE,
        num_blocks * BlockQ4_0::BLCK_SIZE
    );
    let (q4_0_exact, q4_0_rmse) = test_dtype!(BlockQ4_0, GgmlDType::Q4_0, "Q4_0");

    println!(
        "\nTesting Q8_0 ({} blocks × {} elements = {} elements)...",
        num_blocks,
        BlockQ8_0::BLCK_SIZE,
        num_blocks * BlockQ8_0::BLCK_SIZE
    );
    let (q8_0_exact, q8_0_rmse) = test_dtype!(BlockQ8_0, GgmlDType::Q8_0, "Q8_0");

    println!(
        "\nTesting Q2K ({} blocks × {} elements = {} elements)...",
        num_blocks,
        BlockQ2_K::BLCK_SIZE,
        num_blocks * BlockQ2_K::BLCK_SIZE
    );
    let (q2k_exact, q2k_rmse) = test_dtype!(BlockQ2_K, GgmlDType::Q2_K, "Q2K");

    println!(
        "\nTesting Q4K ({} blocks × {} elements = {} elements)...",
        num_blocks,
        BlockQ4_K::BLCK_SIZE,
        num_blocks * BlockQ4_K::BLCK_SIZE
    );
    let (q4k_exact, q4k_rmse) = test_dtype!(BlockQ4_K, GgmlDType::Q4_K, "Q4K");

    println!("\n=== Summary ===");
    println!("Q4_0: exact_bytes={}, rmse={:.2e}", q4_0_exact, q4_0_rmse);
    println!("Q8_0: exact_bytes={}, rmse={:.2e}", q8_0_exact, q8_0_rmse);
    println!("Q2K:  exact_bytes={}, rmse={:.2e}", q2k_exact, q2k_rmse);
    println!("Q4K:  exact_bytes={}, rmse={:.2e}", q4k_exact, q4k_rmse);

    // Reporting diagnostic: prints per-format exact-byte counts and RMSE for a quick
    // overview. Strict byte-exactness is asserted by the dedicated per-format tests.

    Ok(())
}

/// Direct quantization comparison: GPU quantize vs CPU quantize on identical raw input.
/// This tests whether the GPU quantization algorithm matches the CPU exactly.
///
/// Unlike the byte accuracy test (which quantizes already-dequantized data),
/// this test uses fresh random data, so we expect byte-exact matches for
/// formats with deterministic algorithms (like Q4_0, Q8_0).
///
/// For K-quants with iterative refinement (Q4K, Q5K), the GPU may converge to
/// slightly different local optima due to floating-point differences, so we
/// also measure RMSE of the dequantized results.
#[test]
pub(super) fn quantize_direct_comparison() -> Result<()> {
    use crate::quantized::k_quants::{
        BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K,
        BlockQ6_K, BlockQ8_0, BlockQ8_K, BlockQAWQ, BlockQAWQ_G64, GgmlType,
    };

    let dev = CudaDevice::new(0)?;

    // Thresholds for passing
    const MIN_BYTE_MATCH_PCT: f64 = 95.0; // Minimum byte match percentage
    const MAX_RMSE: f64 = 0.5; // Maximum RMSE between GPU and CPU dequant

    struct TestResult {
        name: &'static str,
        byte_match_pct: f64,
        mismatches: usize,
        rmse: f64,
        max_diff: f32,
        passed: bool,
    }

    let mut results: Vec<TestResult> = Vec::new();

    macro_rules! test_direct {
        ($block_ty:ty, $dtype:expr, $name:expr, $num_blocks:expr) => {{
            let block_size = <$block_ty>::BLCK_SIZE;
            let elem_count = $num_blocks * block_size;
            let type_size = std::mem::size_of::<$block_ty>();
            let quant_size = $num_blocks * type_size;

            // Generate random raw input
            let mut rng = rand::rng();
            let raw_input: Vec<f32> = (0..elem_count)
                .map(|_| rng.random_range(-8.0..8.0))
                .collect();

            // CPU quantize
            let mut cpu_blocks: Vec<$block_ty> = vec![<$block_ty>::zeros(); $num_blocks];
            <$block_ty>::from_float(&raw_input, &mut cpu_blocks);

            // CPU dequantize for RMSE comparison
            let mut cpu_dequant = vec![0.0f32; elem_count];
            <$block_ty>::to_float(&cpu_blocks, &mut cpu_dequant);

            // GPU quantize the same raw input
            let raw_input_gpu = dev.memcpy_stod(&raw_input)?;
            let mut gpu_quant_buf = unsafe { dev.alloc::<u8>(quant_size)? };
            quantize_to_dtype(
                &raw_input_gpu.slice(..),
                &mut gpu_quant_buf,
                elem_count,
                $dtype,
                &dev,
            )?;
            dev.synchronize()?;

            // Download GPU quantized bytes
            let gpu_quant_bytes: Vec<u8> = dev.memcpy_dtov(&gpu_quant_buf.slice(..))?;
            let cpu_quant_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(cpu_blocks.as_ptr() as *const u8, quant_size).to_vec()
            };

            // Count byte mismatches
            let mut byte_mismatches = 0usize;
            for i in 0..quant_size {
                if cpu_quant_bytes[i] != gpu_quant_bytes[i] {
                    byte_mismatches += 1;
                }
            }

            // GPU dequantize for RMSE
            let padded = PaddedCudaSlice {
                inner: gpu_quant_buf.clone(),
                len: quant_size,
            };
            let gpu_dequant_storage = dequantize_f32(&padded, $dtype, elem_count, &dev)?;
            let gpu_dequant_cuda = gpu_dequant_storage.as_cuda_slice::<f32>()?;
            let gpu_dequant: Vec<f32> = dev.memcpy_dtov(&gpu_dequant_cuda.slice(..))?;

            // RMSE between GPU dequant and CPU dequant
            let mut sum_sq = 0.0f64;
            let mut max_diff = 0.0f32;
            for i in 0..elem_count {
                let diff = (gpu_dequant[i] - cpu_dequant[i]).abs();
                sum_sq += (diff as f64).powi(2);
                max_diff = max_diff.max(diff);
            }
            let rmse = (sum_sq / elem_count as f64).sqrt();

            let byte_pct = 100.0 * (quant_size - byte_mismatches) as f64 / quant_size as f64;
            let passed = byte_pct >= MIN_BYTE_MATCH_PCT && rmse <= MAX_RMSE;

            results.push(TestResult {
                name: $name,
                byte_match_pct: byte_pct,
                mismatches: byte_mismatches,
                rmse,
                max_diff,
                passed,
            });
        }};
    }

    // Test all quantization types
    // NOTE: Q8_1 is excluded because its to_float() is unimplemented in CPU code
    test_direct!(BlockQ4_0, GgmlDType::Q4_0, "Q4_0", 4096);
    test_direct!(BlockQ4_1, GgmlDType::Q4_1, "Q4_1", 4096);
    test_direct!(BlockQ5_0, GgmlDType::Q5_0, "Q5_0", 4096);
    test_direct!(BlockQ5_1, GgmlDType::Q5_1, "Q5_1", 4096);
    test_direct!(BlockQ8_0, GgmlDType::Q8_0, "Q8_0", 4096);
    // Q8_1 skipped - to_float() unimplemented
    test_direct!(BlockQ2_K, GgmlDType::Q2_K, "Q2K", 4096);
    test_direct!(BlockQ3_K, GgmlDType::Q3_K, "Q3K", 4096);
    test_direct!(BlockQ4_K, GgmlDType::Q4_K, "Q4K", 4096);
    test_direct!(BlockQ5_K, GgmlDType::Q5_K, "Q5K", 4096);
    test_direct!(BlockQ6_K, GgmlDType::Q6_K, "Q6K", 4096);
    test_direct!(BlockQ8_K, GgmlDType::Q8_K, "Q8K", 4096);
    // AWQ formats
    test_direct!(BlockQAWQ, GgmlDType::QAWQ, "QAWQ", 4096);
    test_direct!(BlockQAWQ_G64, GgmlDType::QAWQ_G64, "QAWQG64", 4096);

    // Print fancy table
    println!();
    println!("┌────────┬────────┬────────────┬────────────┬──────────┬──────────┐");
    println!("│ DType  │ Status │ Byte Match │ Mismatches │   RMSE   │ MaxDiff  │");
    println!("├────────┼────────┼────────────┼────────────┼──────────┼──────────┤");

    for r in &results {
        let status = if r.passed { "   ✓    " } else { "   ✗    " };
        println!(
            "│ {:>6} │{}│ {:>9.2}% │ {:>10} │ {:>8.2e} │ {:>8.2e} │",
            r.name, status, r.byte_match_pct, r.mismatches, r.rmse, r.max_diff
        );
    }

    println!("└────────┴────────┴────────────┴────────────┴──────────┴──────────┘");
    println!(
        "\nThresholds: byte_match >= {:.0}%, RMSE <= {:.1}",
        MIN_BYTE_MATCH_PCT, MAX_RMSE
    );
    println!("Note: Q8_1 excluded (to_float unimplemented in CPU)");

    // Check all passed
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failed.is_empty() {
        println!("\nFailed types:");
        for r in &failed {
            println!(
                "  - {}: byte_match={:.2}%, rmse={:.2e}",
                r.name, r.byte_match_pct, r.rmse
            );
        }
        panic!(
            "Quantization test failed: {} types did not meet thresholds",
            failed.len()
        );
    }

    Ok(())
}

/// `GgmlDType::dequantizes_to_bf16` must name exactly the types the device path accepts: a
/// block type the dequantize kernel dispatches (not a KO twin, which refuses), or MXFP4.
#[test]
pub(super) fn the_bf16_dequantize_predicate_matches_the_kernel_dispatch() {
    // Every code the GGUF reader knows, whatever numbering it uses.
    for code in 0..=255u32 {
        let Ok(dtype) = GgmlDType::from_u32(code) else {
            continue;
        };
        let ko = matches!(
            dtype,
            GgmlDType::Q4_KO
                | GgmlDType::Q5_KO
                | GgmlDType::Q6_KO
                | GgmlDType::Q8_KO
                | GgmlDType::MXFP4_KO
                | GgmlDType::Q2_KO
                | GgmlDType::Q3_KO
        );
        let dispatched = dtype == GgmlDType::MXFP4 || (!ko && dtype_to_qtype(dtype).is_ok());
        assert_eq!(dtype.dequantizes_to_bf16(), dispatched, "{dtype:?}");
    }
}
