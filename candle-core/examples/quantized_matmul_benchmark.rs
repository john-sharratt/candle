//! Benchmark for quantized matmul operations.
//!
//! This benchmark measures the performance of quantized matrix multiplication
//! with various batch sizes to help identify performance bottlenecks.
//!
//! Tests both Q4_K and Q6_K formats using F16 activations.
//! Also tests FP8 (E4M3) activations with Q4_K to verify tc32 kernel path.
//! Validates correctness by comparing against dequantized reference matmul.
//!
//! Uses K/128 blocks with embedded scales - no external scale extraction needed.
//!
//! REALISTIC MODE: L2 cache is flushed via CUDA kernel before each batch
//! measurement to simulate real LLM inference where different matrices
//! evict each other from cache. Matrix sizes match actual LLaMA and Qwen3 dimensions.

use anyhow::Result;
use candle_core::quantized::{cuda_flush_l2, get_dispatch_info, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::CudaSlice;
use std::time::Instant;

/// Flush L2 cache using fast CUDA kernel
/// This simulates realistic LLM inference where different matrices evict each other
fn flush_l2_cache_cuda(flush_buffer: &CudaSlice<u8>, device: &Device) {
    let cuda_dev = device.as_cuda_device().unwrap();
    cuda_flush_l2(flush_buffer, cuda_dev);
}

/// Generate weights with realistic distribution scaled to fit quantization limits
/// K-quants work best with weights in roughly [-1, 1] range with std ~0.3-0.5
fn generate_weights(n_rows: usize, n_cols: usize, device: &Device) -> Result<Tensor> {
    // Target std ~0.4 to give range roughly [-1.5, 1.5] (covers 99.7% of values)
    // This fits well within K-quant block-wise quantization limits
    let std_dev = 0.4f32;

    // Generate pseudo-random normal distribution using Box-Muller transform
    let total = n_rows * n_cols;
    let mut data = Vec::with_capacity(total);

    for i in 0..total {
        // Use deterministic pseudo-random for reproducibility
        let u1 = ((i * 1103515245 + 12345) % (1 << 31)) as f64 / (1u64 << 31) as f64;
        let u2 = ((i * 1103515245 + 12345 + 1) % (1 << 31)) as f64 / (1u64 << 31) as f64;

        // Box-Muller transform
        let u1_safe = u1.max(1e-10); // Avoid log(0)
        let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push((z * std_dev as f64) as f32);
    }

    Ok(Tensor::from_vec(data, (n_rows, n_cols), device)?)
}

/// Generate input activations with reasonable distribution
fn generate_activations(batch_size: usize, n_cols: usize, device: &Device) -> Result<Tensor> {
    let total = batch_size * n_cols;
    let mut data = Vec::with_capacity(total);

    // Generate values roughly in [-1, 1] range (typical for normalized activations)
    for i in 0..total {
        let u1 = ((i * 48271 + 12345) % (1 << 31)) as f64 / (1u64 << 31) as f64;
        let u2 = ((i * 48271 + 12346) % (1 << 31)) as f64 / (1u64 << 31) as f64;
        let u1_safe = u1.max(1e-10);
        let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push((z * 0.5) as f32); // Scale to reasonable activation range
    }

    Ok(Tensor::from_vec(data, (batch_size, n_cols), device)?)
}

/// Run benchmark for a specific quantization type
/// flush_buffer: Large CUDA buffer (>L2 size) used to evict L2 cache before each measurement
fn run_benchmark(
    qtype: GgmlDType,
    qtype_name: &str,
    weights_f32: &Tensor,
    n_rows: usize,
    n_cols: usize,
    device: &Device,
    flush_buffer: &CudaSlice<u8>,
) -> Result<bool> {
    // Quantize weights
    let qtensor = QTensor::quantize(weights_f32, qtype)?;
    let orig_bits = qtensor.storage_size_in_bytes() as f64 * 8.0 / (n_rows * n_cols) as f64;

    // Dequantize for reference comparison
    let weights_dequant = qtensor.dequantize(device)?;

    // Check quantization error
    let diff = (weights_f32 - &weights_dequant)?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let quant_rmse = mse.sqrt();

    // Repack to K/128 format with embedded scales
    let qtensor_repacked = qtensor.repack_gemx()?;
    let repacked_size_bytes = qtensor_repacked.storage_size_in_bytes();
    let repacked_bits = repacked_size_bytes as f64 * 8.0 / (n_rows * n_cols) as f64;

    // Create QMatMul wrapper
    let qmatmul = QMatMul::from_qtensor(qtensor_repacked)?;

    // Convert dequantized weights to F16 for reference matmul
    let weights_f16 = weights_dequant.to_dtype(DType::F16)?;

    // Batch sizes to benchmark (1-32 for detailed kernel analysis, then key merge points)
    let batch_sizes = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 47, 48, 49, 63, 64, 128, 256, 511, 512, 768, 1024, 1280, 1536,
        1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
    ];

    println!(
        "\n╔═════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ {:^87} ║",
        format!(
            "{} │ {}x{} │ {:.1}b→{:.1}b │ err={:.4}",
            qtype_name, n_rows, n_cols, orig_bits, repacked_bits, quant_rmse
        )
    );
    println!(
        "╠═════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ Batch │   Time(ms) │   GB/s │ GFLOP/s │ Variant% │ Status │ Kernels                     ║"
    );
    println!(
        "╠═════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mut all_pass = true;

    for &batch_size in &batch_sizes {
        // Create F16 input tensor
        let input_f32 = generate_activations(batch_size, n_cols, device)?;
        let input = input_f32.to_dtype(DType::F16)?;

        // Compute reference result (dequantized matmul)
        let reference = input.matmul(&weights_f16.t()?)?;

        // Compute quantized result (K/128 format with embedded scales)
        let quantized_result = qmatmul.forward_via_gemx(&input)?;

        // Calculate variant (relative error vs baseline)
        let result_diff =
            (&reference.to_dtype(DType::F32)? - &quantized_result.to_dtype(DType::F32)?)?;
        let result_mse = result_diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let result_rmse = result_mse.sqrt();
        let ref_abs_mean = reference
            .to_dtype(DType::F32)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let variant_pct = result_rmse / ref_abs_mean * 100.0;

        let (status, pass) = if variant_pct.is_nan() || variant_pct > 2.0 {
            ("FAIL", false)
        } else {
            ("PASS", true)
        };
        if !pass {
            all_pass = false;
        }

        // Warmup (no L2 flush - just get GPU warmed up)
        for _ in 0..3 {
            let _ = qmatmul.forward_via_gemx(&input)?;
        }
        device.synchronize()?;

        // Benchmark with L2 flush before each iteration (simulates real inference)
        // We time only the matmul, not the flush itself
        let n_iters = if batch_size <= 16 {
            50
        } else if batch_size <= 256 {
            25
        } else {
            10
        };
        let mut total_matmul_time = std::time::Duration::ZERO;
        for _ in 0..n_iters {
            // Flush L2 cache to simulate matrix switching in real inference
            flush_l2_cache_cuda(flush_buffer, device);
            // Time only the matmul operation (after flush completes)
            let matmul_start = Instant::now();
            let _ = qmatmul.forward_via_gemx(&input)?;
            device.synchronize()?;
            total_matmul_time += matmul_start.elapsed();
        }

        let time_per_op = total_matmul_time.as_secs_f64() / n_iters as f64;
        let time_ms = time_per_op * 1000.0;

        // Calculate metrics
        let weight_bytes = repacked_size_bytes as f64;
        let input_bytes = (batch_size * n_cols * 2) as f64; // F16 = 2 bytes
        let output_bytes = (batch_size * n_rows * 2) as f64; // F16 = 2 bytes
        let total_bytes = weight_bytes + input_bytes + output_bytes;
        let gb_per_s = total_bytes / time_per_op / 1e9;

        let flops = 2.0 * (n_rows as f64) * (n_cols as f64) * (batch_size as f64);
        let gflop_per_s = flops / time_per_op / 1e9;

        // Get kernel dispatch info
        let kernel_path = get_dispatch_info(batch_size as i32, repacked_size_bytes);

        println!(
            "║ {:>5} │ {:>10.3} │ {:>6.1} │ {:>7.1} │ {:>7.3}% │  {:^4}  │ {:<27} ║",
            batch_size, time_ms, gb_per_s, gflop_per_s, variant_pct, status, kernel_path
        );
    }

    println!(
        "╚═════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!(
        "  {} Result: {}",
        qtype_name,
        if all_pass {
            "✓ PASS"
        } else {
            "✗ FAIL (variant >= 2%)"
        }
    );

    Ok(all_pass)
}

/// Run FP8 benchmark for a specific quantization type
/// Tests tc32 kernel path (FP8-only) with key batch sizes
fn run_fp8_benchmark(
    qtype: GgmlDType,
    qtype_name: &str,
    weights_f32: &Tensor,
    n_rows: usize,
    n_cols: usize,
    device: &Device,
    flush_buffer: &CudaSlice<u8>,
) -> Result<bool> {
    // Quantize weights
    let qtensor = QTensor::quantize(weights_f32, qtype)?;
    let orig_bits = qtensor.storage_size_in_bytes() as f64 * 8.0 / (n_rows * n_cols) as f64;

    // Dequantize for reference comparison
    let weights_dequant = qtensor.dequantize(device)?;

    // Check quantization error
    let diff = (weights_f32 - &weights_dequant)?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let quant_rmse = mse.sqrt();

    // Repack to K/128 format with embedded scales
    let qtensor_repacked = qtensor.repack_gemx()?;
    let repacked_size_bytes = qtensor_repacked.storage_size_in_bytes();
    let repacked_bits = repacked_size_bytes as f64 * 8.0 / (n_rows * n_cols) as f64;

    // Create QMatMul wrapper
    let qmatmul = QMatMul::from_qtensor(qtensor_repacked)?;

    // Convert dequantized weights to F16 for reference matmul
    let weights_f16 = weights_dequant.to_dtype(DType::F16)?;

    // FP8 test: Key batch sizes that exercise tc32 path (32, 64, etc)
    let batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    println!(
        "\n╔═════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ {:^87} ║",
        format!(
            "{} (FP8) │ {}x{} │ {:.1}b→{:.1}b │ err={:.4}",
            qtype_name, n_rows, n_cols, orig_bits, repacked_bits, quant_rmse
        )
    );
    println!(
        "╠═════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ Batch │   Time(ms) │   GB/s │ GFLOP/s │ Variant% │ Status │ Kernels                     ║"
    );
    println!(
        "╠═════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mut all_pass = true;

    for &batch_size in &batch_sizes {
        // Create FP8 input tensor (convert from F32 via clamp to FP8 range)
        let input_f32 = generate_activations(batch_size, n_cols, device)?;
        // FP8 E4M3 has range [-448, 448], but we keep activations in reasonable range
        let input_fp8 = input_f32.to_dtype(DType::F8E4M3)?;

        // Compute reference result (F16 dequantized matmul)
        let input_f16 = input_f32.to_dtype(DType::F16)?;
        let reference = input_f16.matmul(&weights_f16.t()?)?;

        // Compute quantized result with FP8 activations
        let quantized_result = qmatmul.forward_via_gemx(&input_fp8)?;

        // Calculate variant (relative error vs baseline)
        // FP8 has lower precision, so we expect slightly higher error
        let result_diff =
            (&reference.to_dtype(DType::F32)? - &quantized_result.to_dtype(DType::F32)?)?;
        let result_mse = result_diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let result_rmse = result_mse.sqrt();
        let ref_abs_mean = reference
            .to_dtype(DType::F32)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let variant_pct = result_rmse / ref_abs_mean * 100.0;

        // FP8 has lower precision, allow up to 5% variant
        let (status, pass) = if variant_pct.is_nan() || variant_pct > 5.0 {
            ("FAIL", false)
        } else {
            ("PASS", true)
        };
        if !pass {
            all_pass = false;
        }

        // Warmup (no L2 flush - just get GPU warmed up)
        for _ in 0..3 {
            let _ = qmatmul.forward_via_gemx(&input_fp8)?;
        }
        device.synchronize()?;

        // Benchmark with L2 flush before each iteration (simulates real inference)
        let n_iters = if batch_size <= 16 {
            50
        } else if batch_size <= 256 {
            25
        } else {
            10
        };
        let mut total_matmul_time = std::time::Duration::ZERO;
        for _ in 0..n_iters {
            // Flush L2 cache to simulate matrix switching in real inference
            flush_l2_cache_cuda(flush_buffer, device);
            // Time only the matmul operation (after flush completes)
            let matmul_start = Instant::now();
            let _ = qmatmul.forward_via_gemx(&input_fp8)?;
            device.synchronize()?;
            total_matmul_time += matmul_start.elapsed();
        }

        let time_per_op = total_matmul_time.as_secs_f64() / n_iters as f64;
        let time_ms = time_per_op * 1000.0;

        // Calculate metrics
        let weight_bytes = repacked_size_bytes as f64;
        let input_bytes = (batch_size * n_cols) as f64; // FP8 = 1 byte
        let output_bytes = (batch_size * n_rows) as f64; // FP8 output = 1 byte
        let total_bytes = weight_bytes + input_bytes + output_bytes;
        let gb_per_s = total_bytes / time_per_op / 1e9;

        let flops = 2.0 * (n_rows as f64) * (n_cols as f64) * (batch_size as f64);
        let gflop_per_s = flops / time_per_op / 1e9;

        // Get kernel dispatch info (note: this shows F16 path, FP8 uses tc32)
        let kernel_path = get_dispatch_info(batch_size as i32, repacked_size_bytes);
        // For FP8, tc32 is used instead of tc16, but dispatch_info doesn't know ytype
        let kernel_path_fp8 = kernel_path.replace("tc16", "tc32*");

        println!(
            "║ {:>5} │ {:>10.3} │ {:>6.1} │ {:>7.1} │ {:>7.3}% │  {:^4}  │ {:<27} ║",
            batch_size, time_ms, gb_per_s, gflop_per_s, variant_pct, status, kernel_path_fp8
        );
    }

    println!(
        "╚═════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!(
        "  {} (FP8) Result: {}",
        qtype_name,
        if all_pass {
            "✓ PASS"
        } else {
            "✗ FAIL (variant >= 5%)"
        }
    );

    Ok(all_pass)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Query L2 cache size from device
    let l2_cache_bytes = device.l2_cache_size()?;
    let l2_mb = l2_cache_bytes as f64 / 1024.0 / 1024.0;

    println!("GPU L2 Cache: {:.0} MB", l2_mb);
    println!("Mode: REALISTIC (CUDA L2 flush before each batch to simulate real LLM inference)\n");

    // Create L2 flush buffer directly on CUDA device (2x L2 size to ensure full eviction)
    let flush_size = l2_cache_bytes * 2;
    let cuda_dev = device.as_cuda_device()?;
    let flush_buffer: CudaSlice<u8> = cuda_dev.alloc_zeros(flush_size)?;

    // =========================================================================
    // Use realistic LLaMA 3.2-3B matrix dimensions
    // =========================================================================
    // FFN matrices are the largest and most performance-critical:
    //   ffn_gate/up: 8192×3072 = 25.2M elements (~16MB Q4_K repacked)
    //   ffn_down:    3072×8192 = 25.2M elements (~20MB Q6_K repacked)
    // Attention matrices:
    //   attn_q/out:  3072×3072 = 9.4M elements (~6MB Q4_K repacked)
    //   attn_k/v:    1024×3072 = 3.1M elements (~2MB repacked)

    let bytes_per_q4k = 0.625; // Repacked Q4_K
    let bytes_per_q6k = 0.8125; // Repacked Q6_K

    // =========================================================================
    // RUN 1: Q4_K FFN gate/up matrix (largest Q4_K in LLaMA)
    // =========================================================================
    let n_rows_ffn = 8192; // intermediate_size
    let n_cols_ffn = 3072; // hidden_size
    let ffn_size_mb = (n_rows_ffn * n_cols_ffn) as f64 * bytes_per_q4k / 1e6;
    println!(
        "Quantized Matmul Benchmark │ {}x{} ({:.1}M, ~{:.0}MB Q4_K) │ LLaMA ffn_gate/up │ {:.0}% of {:.0}MB L2",
        n_rows_ffn,
        n_cols_ffn,
        (n_rows_ffn * n_cols_ffn) as f64 / 1e6,
        ffn_size_mb,
        ffn_size_mb / l2_mb * 100.0,
        l2_mb
    );

    let weights_f32_ffn = generate_weights(n_rows_ffn, n_cols_ffn, &device)?;

    let q4k_ffn_pass = run_benchmark(
        GgmlDType::Q4_K,
        "Q4_K",
        &weights_f32_ffn,
        n_rows_ffn,
        n_cols_ffn,
        &device,
        &flush_buffer,
    )?;

    // =========================================================================
    // RUN 2: Q4_K attention q/output matrix
    // =========================================================================
    let n_rows_attn = 3072; // hidden_size
    let n_cols_attn = 3072; // hidden_size
    let attn_size_mb = (n_rows_attn * n_cols_attn) as f64 * bytes_per_q4k / 1e6;
    println!(
        "\nQuantized Matmul Benchmark │ {}x{} ({:.1}M, ~{:.0}MB Q4_K) │ LLaMA attn_q/out │ {:.0}% of {:.0}MB L2",
        n_rows_attn,
        n_cols_attn,
        (n_rows_attn * n_cols_attn) as f64 / 1e6,
        attn_size_mb,
        attn_size_mb / l2_mb * 100.0,
        l2_mb
    );

    let weights_f32_attn = generate_weights(n_rows_attn, n_cols_attn, &device)?;

    let q4k_attn_pass = run_benchmark(
        GgmlDType::Q4_K,
        "Q4_K",
        &weights_f32_attn,
        n_rows_attn,
        n_cols_attn,
        &device,
        &flush_buffer,
    )?;

    // =========================================================================
    // RUN 3: Q4_K FP8 activations with FFN matrix (tests tc32 kernel path)
    // =========================================================================
    println!(
        "\nQuantized Matmul Benchmark │ {}x{} ({:.1}M, ~{:.0}MB Q4_K) │ LLaMA ffn FP8 (tc32 path)",
        n_rows_ffn,
        n_cols_ffn,
        (n_rows_ffn * n_cols_ffn) as f64 / 1e6,
        ffn_size_mb,
    );

    let q4k_fp8_pass = run_fp8_benchmark(
        GgmlDType::Q4_K,
        "Q4_K",
        &weights_f32_ffn,
        n_rows_ffn,
        n_cols_ffn,
        &device,
        &flush_buffer,
    )?;

    // =========================================================================
    // RUN 4: Q6_K FFN down matrix (uses Q6_K in Q4_K_M quantization)
    // =========================================================================
    let n_rows_down = 3072; // hidden_size
    let n_cols_down = 8192; // intermediate_size
    let down_size_mb = (n_rows_down * n_cols_down) as f64 * bytes_per_q6k / 1e6;
    println!(
        "\nQuantized Matmul Benchmark │ {}x{} ({:.1}M, ~{:.0}MB Q6_K) │ LLaMA ffn_down │ {:.0}% of {:.0}MB L2",
        n_rows_down,
        n_cols_down,
        (n_rows_down * n_cols_down) as f64 / 1e6,
        down_size_mb,
        down_size_mb / l2_mb * 100.0,
        l2_mb
    );

    let weights_f32_down = generate_weights(n_rows_down, n_cols_down, &device)?;

    let q6k_pass = run_benchmark(
        GgmlDType::Q6_K,
        "Q6_K",
        &weights_f32_down,
        n_rows_down,
        n_cols_down,
        &device,
        &flush_buffer,
    )?;

    // =========================================================================
    // RUN 5: Qwen3-32B FFN gate/up matrix (WORST CASE - largest common matrix)
    // =========================================================================
    // Qwen3-32B: hidden=5120, intermediate=27648
    // FFN gate/up: 27648×5120 = 141.6M elements (~88MB Q4_K repacked)
    // This exceeds L2 cache significantly and represents a worst-case scenario
    let n_rows_qwen = 27648; // Qwen3-32B intermediate_size
    let n_cols_qwen = 5120; // Qwen3-32B hidden_size
    let qwen_size_mb = (n_rows_qwen * n_cols_qwen) as f64 * bytes_per_q4k / 1e6;
    println!(
        "\nQuantized Matmul Benchmark │ {}x{} ({:.1}M, ~{:.0}MB Q4_K) │ Qwen3-32B ffn_gate/up │ {:.0}% of {:.0}MB L2",
        n_rows_qwen,
        n_cols_qwen,
        (n_rows_qwen * n_cols_qwen) as f64 / 1e6,
        qwen_size_mb,
        qwen_size_mb / l2_mb * 100.0,
        l2_mb
    );

    let weights_f32_qwen = generate_weights(n_rows_qwen, n_cols_qwen, &device)?;

    let qwen_pass = run_benchmark(
        GgmlDType::Q4_K,
        "Q4_K",
        &weights_f32_qwen,
        n_rows_qwen,
        n_cols_qwen,
        &device,
        &flush_buffer,
    )?;

    println!(
        "\nOverall: Q4_K(ffn)={}, Q4_K(attn)={}, Q4_K(FP8)={}, Q6_K(ffn_down)={}, Qwen3-32B={}",
        if q4k_ffn_pass { "PASS" } else { "FAIL" },
        if q4k_attn_pass { "PASS" } else { "FAIL" },
        if q4k_fp8_pass { "PASS" } else { "FAIL" },
        if q6k_pass { "PASS" } else { "FAIL" },
        if qwen_pass { "PASS" } else { "FAIL" },
    );

    if q4k_ffn_pass && q4k_attn_pass && q4k_fp8_pass && q6k_pass && qwen_pass {
        Ok(())
    } else {
        anyhow::bail!("Some tests failed")
    }
}
