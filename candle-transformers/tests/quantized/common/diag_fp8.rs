//! Diagnostics for the FP8 activation path, which the batch-64 kernel serves.
//!
//! The FP8 result is compared with an F32 baseline. Indicator values then show
//! which batch positions receive the wrong activations: an identity per batch,
//! a single probe column, a batch permutation check, the thread-group pattern of
//! the m16n8k32 instruction, and the sensitivity to which K positions are active.

use super::*;

/// FP8 Activation Diagnostic: Test with FP8 input dtype at various batch sizes
/// This specifically tests the FP8 m16n8k32 MMA path which fails at batch >= 64
#[cfg(feature = "cuda")]
pub fn run_fp8_activation_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[FP8 Diagnostic] Testing FP8 activation dtype at batch={}...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Create simple weights
    let weights_f32 = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights_f32, ggml_dtype)?;

    // Create FP8 input
    let input_f32 = create_test_input(batch, 1, ncols, DType::F32, device)?;
    let input_fp8 = input_f32.to_dtype(DType::F8E4M3)?;

    // Compute baseline (using F32 for precision)
    let baseline = compute_baseline(&qtensor, &input_f32, batch, 1, nrows, ncols, device)?;
    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;

    // Compute FP8 result
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;
    let fp8_result = qmatmul.forward(&input_fp8)?;
    let fp8_result_f32 = fp8_result.to_dtype(DType::F32)?;
    let fp8_vec: Vec<f32> = fp8_result_f32.flatten_all()?.to_vec1()?;

    // Compare first batch element
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut mismatches = 0;
    let threshold = 0.15; // FP8 tolerance

    for i in 0..nrows {
        let diff = (baseline_vec[i] - fp8_vec[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        if diff > threshold {
            mismatches += 1;
        }
    }

    let status = if max_diff < threshold { "OK" } else { "FAIL" };

    println!(
        "  batch={}: max_diff={:.6} at idx={}, mismatches={}/{} [{}]",
        batch, max_diff, max_diff_idx, mismatches, nrows, status
    );

    if max_diff >= threshold {
        // Show more detail for failures
        println!("  First few mismatches:");
        let mut shown = 0;
        for i in 0..nrows {
            let diff = (baseline_vec[i] - fp8_vec[i]).abs();
            if diff > threshold && shown < 5 {
                println!(
                    "    idx={}: baseline={:.4}, fp8={:.4}, diff={:.4}",
                    i, baseline_vec[i], fp8_vec[i], diff
                );
                shown += 1;
            }
        }

        // Check if it's a systematic error (e.g., all zeros, 2x, etc.)
        let mut zero_count = 0;
        let mut ratio_sum = 0.0f32;
        let mut ratio_count = 0;
        for i in 0..nrows {
            if fp8_vec[i].abs() < 1e-6 {
                zero_count += 1;
            }
            if baseline_vec[i].abs() > 0.01 {
                ratio_sum += fp8_vec[i] / baseline_vec[i];
                ratio_count += 1;
            }
        }
        let avg_ratio = if ratio_count > 0 {
            ratio_sum / ratio_count as f32
        } else {
            0.0
        };
        println!(
            "  Zero outputs: {}/{}, avg ratio (fp8/baseline): {:.3}x",
            zero_count, nrows, avg_ratio
        );
    }

    Ok(())
}

/// FP8 MMA Indicator Tracing - Use specific indicator values to trace exact data flow
///
/// This test sets up carefully chosen weight and activation values to determine:
/// 1. Whether the FP8 path produces different results than F32
/// 2. Which batch positions are affected
/// 3. What pattern the errors follow
#[cfg(feature = "cuda")]
pub fn run_fp8_mma_indicator_tracing(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[FP8 MMA Trace] Reverse-engineering FP8 m16n8k32 batch mapping...");

    let batch = 32; // Uses s32_tc kernel (m16n8k32 MMA for FP8)
    let nrows = 256; // N dimension
    let ncols = 256; // K dimension

    // TEST 1: BATCH IDENTITY TEST
    // Set A[b, k] = (b+1) for all k (each batch has its own identifier)
    // Set W[k, n] = 1/ncols for all k,n (average pooling effect)
    // Expected: C[b, n] ≈ (b+1) for all n
    // This reveals which batch's activations are actually being used
    println!("\n  TEST 1: Batch Identity Mapping");
    println!(
        "  A[b,k] = (b+1), W[k,n] = 1/{} → Expected C[b,n] ≈ (b+1)",
        ncols
    );

    let scale = 1.0 / ncols as f32;
    let weights_uniform: Vec<f32> = vec![scale; ncols * nrows];
    let weights = Tensor::from_vec(weights_uniform, (ncols, nrows), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;

    // Each batch b has value (b+1) at all K positions
    let mut act_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        for k in 0..ncols {
            act_data[b * ncols + k] = (b + 1) as f32;
        }
    }
    let act_f32 = Tensor::from_vec(act_data.clone(), (batch, 1, ncols), device)?;
    let act_fp8 = act_f32.to_dtype(DType::F8E4M3)?;

    let result_f32 = qmatmul.forward(&act_f32)?;
    let result_fp8 = qmatmul.forward(&act_fp8)?;

    let vec_f32: Vec<f32> = result_f32.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let vec_fp8: Vec<f32> = result_fp8.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    println!("\n  batch | expected | F32 result | FP8 result | FP8 implies batch | status");
    println!("  ------|----------|------------|------------|-------------------|--------");

    for b in 0..batch {
        let expected = (b + 1) as f32;
        let f32_val = vec_f32[b * nrows]; // First output column
        let fp8_val = vec_fp8[b * nrows];

        // Reverse-engineer: which batch index does the FP8 result imply?
        let implied_batch = (fp8_val.round() as i32 - 1).max(0) as usize;
        let status = if (implied_batch as i32 - b as i32).abs() <= 1 {
            "OK"
        } else {
            "WRONG"
        };

        println!(
            "  {:5} | {:8.1} | {:10.2} | {:10.2} | {:17} | {}",
            b, expected, f32_val, fp8_val, implied_batch, status
        );
    }

    // TEST 2: SINGLE BATCH PROBE
    // For each target batch, set A[target,0] = 1.0, all else = 0
    // W[0,0] = 1.0, all else = 0
    // Expected: C[target,0] = 1.0, all other C[b,0] = 0
    println!("\n  TEST 2: Single Batch Isolation (which output positions light up?)");
    println!("  For each target batch: A[target,0]=1, W[0,0]=1");

    let mut weights_single = vec![0.0f32; ncols * nrows];
    weights_single[0] = 1.0; // W[k=0, n=0] = 1.0
    let weights2 = Tensor::from_vec(weights_single, (ncols, nrows), device)?;
    let qtensor2 = QTensor::quantize(&weights2, ggml_dtype)?;
    let qmatmul2 = QMatMulWrapper::from_qtensor(qtensor2)?;

    println!("\n  target | F32 output positions with value | FP8 output positions with value");
    println!("  -------|----------------------------------|----------------------------------");

    for target in 0..batch {
        let mut act_probe = vec![0.0f32; batch * ncols];
        act_probe[target * ncols + 0] = 1.0; // Only A[target, k=0] = 1.0

        let act_f32_2 = Tensor::from_vec(act_probe.clone(), (batch, 1, ncols), device)?;
        let act_fp8_2 = act_f32_2.to_dtype(DType::F8E4M3)?;

        let res_f32_2 = qmatmul2.forward(&act_f32_2)?;
        let res_fp8_2 = qmatmul2.forward(&act_fp8_2)?;

        let v_f32: Vec<f32> = res_f32_2.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let v_fp8: Vec<f32> = res_fp8_2.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // Find which batch positions have non-zero output
        let f32_active: Vec<usize> = (0..batch)
            .filter(|&b| v_f32[b * nrows].abs() > 0.1)
            .collect();
        let fp8_active: Vec<usize> = (0..batch)
            .filter(|&b| v_fp8[b * nrows].abs() > 0.1)
            .collect();

        let f32_str = if f32_active.is_empty() {
            "none".to_string()
        } else {
            f32_active
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let fp8_str = if fp8_active.is_empty() {
            "none".to_string()
        } else {
            fp8_active
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };

        let status = if f32_active == fp8_active {
            ""
        } else {
            " ← MISMATCH"
        };
        println!("  {:6} | {:32} | {}{}", target, f32_str, fp8_str, status);
    }

    // TEST 3: BATCH PERMUTATION DETECTION
    // A[b,k] = b (batch index), W[k,n] = 1 (sum over K)
    // Analyze which batch indices appear in FP8 outputs
    println!("\n  TEST 3: Full Permutation Analysis");
    println!("  A[b,k] = b, W[k,n] = 1/K → C[b,n] should be b");

    // Use same K as previous tests (must match ncols for consistent block alignment)
    let small_k = ncols;
    let small_scale = 1.0 / small_k as f32;
    let weights3_data: Vec<f32> = vec![small_scale; small_k * nrows];
    let weights3 = Tensor::from_vec(weights3_data, (small_k, nrows), device)?;
    let qtensor3 = QTensor::quantize(&weights3, ggml_dtype)?;
    let qmatmul3 = QMatMulWrapper::from_qtensor(qtensor3)?;

    let mut act3_data = vec![0.0f32; batch * small_k];
    for b in 0..batch {
        for k in 0..small_k {
            act3_data[b * small_k + k] = b as f32;
        }
    }
    let act3_f32 = Tensor::from_vec(act3_data.clone(), (batch, 1, small_k), device)?;
    let act3_fp8 = act3_f32.to_dtype(DType::F8E4M3)?;

    let res3_f32 = qmatmul3.forward(&act3_f32)?;
    let res3_fp8 = qmatmul3.forward(&act3_fp8)?;

    let v3_f32: Vec<f32> = res3_f32.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let v3_fp8: Vec<f32> = res3_fp8.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    println!("\n  Detected batch mapping (FP8 actual → expected):");
    println!("  output_batch | F32_val (=expected) | FP8_val | FP8 reads from batch | diff");
    println!("  -------------|---------------------|---------|----------------------|-----");

    let mut permutation: Vec<(usize, usize)> = Vec::new();
    for b in 0..batch {
        let f32_val = v3_f32[b * nrows];
        let fp8_val = v3_fp8[b * nrows];
        let fp8_source = fp8_val.round() as i32;
        let diff = (fp8_source - b as i32).abs();

        if diff > 0 {
            permutation.push((b, fp8_source as usize));
        }

        let mark = if diff > 0 { "←" } else { "" };
        println!(
            "  {:12} | {:19.1} | {:7.1} | {:20} | {:3} {}",
            b, f32_val, fp8_val, fp8_source, diff, mark
        );
    }

    // TEST 4: THREAD/GROUP PATTERN ANALYSIS
    // For m16n8k32: groupID = lane/4 (0-7), tid = lane%4 (0-3)
    // Each groupID handles rows groupID and groupID+8
    // Let's see if errors follow this pattern
    println!("\n  TEST 4: MMA Thread Group Pattern Analysis");
    println!("  Grouping batches by MMA thread structure:");
    println!("  (For m16n8k32: groupID=lane/4 handles rows groupID and groupID+8)");

    println!("\n  Group | Batch pair (b, b+8) | F32 values        | FP8 values        | Pattern");
    println!("  ------|---------------------|-------------------|-------------------|--------");

    for group in 0..8 {
        let b0 = group;
        let b1 = group + 8;
        let b0_16 = 16 + group;
        let b1_16 = 16 + group + 8;

        let f32_b0 = v3_f32[b0 * nrows];
        let f32_b1 = v3_f32[b1 * nrows];
        let fp8_b0 = v3_fp8[b0 * nrows];
        let fp8_b1 = v3_fp8[b1 * nrows];

        let f32_b0_16 = v3_f32[b0_16 * nrows];
        let f32_b1_16 = v3_f32[b1_16 * nrows];
        let fp8_b0_16 = v3_fp8[b0_16 * nrows];
        let fp8_b1_16 = v3_fp8[b1_16 * nrows];

        let ok_lo = (fp8_b0 - f32_b0).abs() < 1.0 && (fp8_b1 - f32_b1).abs() < 1.0;
        let ok_hi = (fp8_b0_16 - f32_b0_16).abs() < 1.0 && (fp8_b1_16 - f32_b1_16).abs() < 1.0;

        println!("  {:5} | ({:2},{:2}) ({:2},{:2})     | {:.0},{:.0} / {:.0},{:.0}  | {:.0},{:.0} / {:.0},{:.0}  | {}",
                 group, b0, b1, b0_16, b1_16,
                 f32_b0, f32_b1, f32_b0_16, f32_b1_16,
                 fp8_b0, fp8_b1, fp8_b0_16, fp8_b1_16,
                 if ok_lo && ok_hi { "OK" } else { "WRONG" });
    }

    // TEST 5: RAW MAPPING TABLE
    // Create comprehensive mapping: for each output batch, which input batch?
    println!("\n  TEST 5: Complete Mapping Table (FP8 source batch for each output batch)");
    print!("  Output:  ");
    for b in 0..batch {
        print!("{:3}", b);
    }
    println!();
    print!("  FP8 src: ");
    for b in 0..batch {
        let fp8_val = v3_fp8[b * nrows];
        let src = fp8_val.round() as i32;
        print!("{:3}", src);
    }
    println!();
    print!("  Diff:    ");
    for b in 0..batch {
        let fp8_val = v3_fp8[b * nrows];
        let src = fp8_val.round() as i32;
        let diff = src - b as i32;
        if diff == 0 {
            print!("  .");
        } else {
            print!("{:+3}", diff);
        }
    }
    println!();

    // TEST 6: K-POSITION SENSITIVITY
    // Does the error depend on which K positions have non-zero activations?
    println!("\n  TEST 6: K-Position Sensitivity");
    println!("  Testing if errors depend on K position (first 32 vs last 32):");

    // Test with only first 32 K elements active
    let mut act_k_first = vec![0.0f32; batch * small_k];
    for b in 0..batch {
        for k in 0..32 {
            act_k_first[b * small_k + k] = (b + 1) as f32;
        }
    }
    let act_kf_f32 = Tensor::from_vec(act_k_first.clone(), (batch, 1, small_k), device)?;
    let act_kf_fp8 = act_kf_f32.to_dtype(DType::F8E4M3)?;

    let res_kf_f32 = qmatmul3.forward(&act_kf_f32)?;
    let res_kf_fp8 = qmatmul3.forward(&act_kf_fp8)?;
    let v_kf_f32: Vec<f32> = res_kf_f32.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let v_kf_fp8: Vec<f32> = res_kf_fp8.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    // Test with only last 32 K elements active
    let mut act_k_last = vec![0.0f32; batch * small_k];
    for b in 0..batch {
        for k in (small_k - 32)..small_k {
            act_k_last[b * small_k + k] = (b + 1) as f32;
        }
    }
    let act_kl_f32 = Tensor::from_vec(act_k_last.clone(), (batch, 1, small_k), device)?;
    let act_kl_fp8 = act_kl_f32.to_dtype(DType::F8E4M3)?;

    let res_kl_f32 = qmatmul3.forward(&act_kl_f32)?;
    let res_kl_fp8 = qmatmul3.forward(&act_kl_fp8)?;
    let v_kl_f32: Vec<f32> = res_kl_f32.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let v_kl_fp8: Vec<f32> = res_kl_fp8.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    println!("\n  batch | K=0-31 F32 | K=0-31 FP8 | K=96-127 F32 | K=96-127 FP8 | pattern");
    println!("  ------|------------|------------|--------------|--------------|--------");

    let mut k_first_errors = 0;
    let mut k_last_errors = 0;
    for b in 0..batch {
        let kf_f32 = v_kf_f32[b * nrows];
        let kf_fp8 = v_kf_fp8[b * nrows];
        let kl_f32 = v_kl_f32[b * nrows];
        let kl_fp8 = v_kl_fp8[b * nrows];

        let kf_ok = (kf_f32 - kf_fp8).abs() < 0.5;
        let kl_ok = (kl_f32 - kl_fp8).abs() < 0.5;

        if !kf_ok {
            k_first_errors += 1;
        }
        if !kl_ok {
            k_last_errors += 1;
        }

        let pattern = match (kf_ok, kl_ok) {
            (true, true) => "both OK",
            (true, false) => "K=96-127 wrong",
            (false, true) => "K=0-31 wrong",
            (false, false) => "both wrong",
        };

        println!(
            "  {:5} | {:10.2} | {:10.2} | {:12.2} | {:12.2} | {}",
            b, kf_f32, kf_fp8, kl_f32, kl_fp8, pattern
        );
    }

    println!(
        "\n  Summary: K=0-31 errors: {}/32, K=96-127 errors: {}/32",
        k_first_errors, k_last_errors
    );

    Ok(())
}
