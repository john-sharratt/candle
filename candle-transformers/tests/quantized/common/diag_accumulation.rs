//! Diagnostics for a batch-dependent accumulation fault in the quantised matmul
//! at batch sizes of 16 and up.
//!
//! Each diagnostic chooses inputs whose expected output is known. With unit
//! weights the output is the sum of the activated elements, so a doubled
//! accumulation shows as twice that sum. Others isolate one K-row or one K
//! element, or count which iterations contribute, to show which part of the
//! reduction misbehaves at the larger batch.

use super::*;

// NEW BATCH-SPECIFIC ACCUMULATION DIAGNOSTICS
// These are designed to catch the 2x accumulation bug seen at batch>=16

/// Diagnostic 15: Accumulation ratio test
/// Uses weights=1 so that output = sum of all activated Y elements.
/// If we get 2x the expected sum, something is being accumulated twice.
#[cfg(feature = "cuda")]
pub fn run_accumulation_ratio_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 15] Accumulation ratio test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Use non-degenerate weights to avoid zero scales for K-quants
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test several K-ranges to see accumulation behavior
    let test_cases: Vec<(&str, Vec<usize>)> = vec![
        ("K=0-31 (first 32)", (0..32).collect()),
        ("K=0-63 (first 64)", (0..64).collect()),
        ("K=64-127", (64..128).collect()),
        ("K=128-191", (128..192).collect()),
        ("K=192-255", (192..256).collect()),
        ("K=0-127 (first half)", (0..128).collect()),
        ("K=128-255 (second half)", (128..256).collect()),
        ("All K (0-255)", (0..256).collect()),
    ];

    println!("  Testing accumulation of specific K-ranges:");
    println!("  {:25} | Expected | Actual   | Ratio  | Status", "Range");
    println!("  {:-<25}-|----------|----------|--------|-------", "");

    let mut ratio_errors = Vec::new();

    for (name, k_positions) in &test_cases {
        // Create batched input with 1.0 at specified K positions
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for &k in k_positions {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        // Check the first output element (row 0, first batch)
        let expected = baseline_vec[0];
        let actual = gemx_vec[0];
        let ratio = if expected.abs() > 0.001 {
            actual / expected
        } else {
            0.0
        };

        let status = if (ratio - 1.0).abs() < 0.1 {
            "OK"
        } else {
            "FAIL"
        };
        println!(
            "  {:25} | {:8.4} | {:8.4} | {:6.3}x | {}",
            name, expected, actual, ratio, status
        );

        if (ratio - 1.0).abs() >= 0.1 {
            ratio_errors.push((name.to_string(), expected, actual, ratio));
        }
    }

    if ratio_errors.is_empty() {
        println!("  ✓ All accumulations have correct ratio (batch={})", batch);
    } else {
        println!("\n  ❌ Accumulation errors detected:");
        for (name, _exp, _act, ratio) in &ratio_errors {
            if (ratio - 2.0).abs() < 0.2 {
                println!("     {} appears to be accumulated 2x!", name);
            } else if (ratio - 0.5).abs() < 0.1 {
                println!(
                    "     {} appears to be accumulated 0.5x (half missing)!",
                    name
                );
            } else {
                println!("     {} has unexpected ratio {:.3}x", name, ratio);
            }
        }
    }

    Ok(())
}

/// Diagnostic 16: K-row isolation test for batch sizes
/// Tests which specific K-rows are contributing to the result.
/// With repacker layout [K/16, N, 8B], K-row R contains elements K=R*16..(R+1)*16
#[cfg(feature = "cuda")]
pub fn run_k_row_isolation_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 16] K-row isolation test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256; // 16 K-rows

    // Use non-degenerate weights to avoid zero scales for K-quants
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    println!("  Testing individual K-rows (each covers 16 K elements):");
    println!("  K-row | K-range     | Expected | Actual   | Ratio  | Status");
    println!("  ------|-------------|----------|----------|--------|-------");

    let mut row_results = Vec::new();

    for k_row in 0..16 {
        let k_start = k_row * 16;
        let k_end = k_start + 16;

        // Create batched input with 1.0 only in this K-row's range
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in k_start..k_end {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let expected = baseline_vec[0];
        let actual = gemx_vec[0];
        let ratio = if expected.abs() > 0.001 {
            actual / expected
        } else {
            0.0
        };

        let status = if (ratio - 1.0).abs() < 0.1 {
            "OK"
        } else if actual.abs() < 0.001 {
            "ZERO"
        } else {
            "FAIL"
        };

        println!(
            "  {:5} | {:4}-{:<4}   | {:8.4} | {:8.4} | {:6.3}x | {}",
            k_row,
            k_start,
            k_end - 1,
            expected,
            actual,
            ratio,
            status
        );

        row_results.push((k_row, expected, actual, ratio, status.to_string()));
    }

    // Analyze pattern
    let zeros: Vec<_> = row_results.iter().filter(|r| r.4 == "ZERO").collect();
    let doubles: Vec<_> = row_results
        .iter()
        .filter(|r| (r.3 - 2.0).abs() < 0.2)
        .collect();

    if !zeros.is_empty() {
        println!("\n  ⚠️  K-rows producing ZERO (not being read):");
        for (row, _, _, _, _) in &zeros {
            println!("      K-row {} (K={}-{})", row, row * 16, row * 16 + 15);
        }
    }

    if !doubles.is_empty() {
        println!("\n  ⚠️  K-rows producing 2x (read twice):");
        for (row, _, _, _, _) in &doubles {
            println!("      K-row {} (K={}-{})", row, row * 16, row * 16 + 15);
        }
    }

    Ok(())
}

/// Diagnostic 17: Compare batch=1 vs batch=16 element-by-element
/// For a single output row, compares what elements contribute to the sum
#[cfg(feature = "cuda")]
pub fn run_batch_comparison_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 17] Batch=1 vs Batch=16 comparison...");

    let nrows = 256;
    let ncols = 256;

    // Create identifiable weights: W[row, k] = (k+1) / 1000
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for row in 0..nrows {
        for k in 0..ncols {
            weights_data[row * ncols + k] = (k as f32 + 1.0) / 1000.0;
        }
    }
    let weights = Tensor::from_vec(weights_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Use Y = all 1s, so output[row] = sum(W[row, :]) = sum((k+1)/1000) for k=0..255
    // Expected sum ≈ (1 + 2 + ... + 256) / 1000 = 256*257/2/1000 = 32.896

    // Test at batch=1
    let input_1 = Tensor::full(1.0f32, (1, 1, ncols), device)?;
    let baseline_1 = compute_baseline(&qtensor, &input_1, 1, 1, nrows, ncols, device)?;
    let qmatmul_1 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx_1 = qmatmul_1.forward(&input_1)?;

    // Test at batch=16
    let input_16 = Tensor::full(1.0f32, (16, 1, ncols), device)?;
    let baseline_16 = compute_baseline(&qtensor, &input_16, 16, 1, nrows, ncols, device)?;
    let qmatmul_16 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx_16 = qmatmul_16.forward(&input_16)?;

    let base_1: Vec<f32> = baseline_1.flatten_all()?.to_vec1()?;
    let gemx_1_vec: Vec<f32> = gemx_1.flatten_all()?.to_vec1()?;
    let base_16: Vec<f32> = baseline_16.flatten_all()?.to_vec1()?;
    let gemx_16_vec: Vec<f32> = gemx_16.flatten_all()?.to_vec1()?;

    println!("  Row 0 output comparison:");
    println!("           | Baseline  | GEMX    | Ratio");
    println!("  ---------|-----------|-----------|-------");
    println!(
        "  Batch=1  | {:9.4} | {:9.4} | {:6.3}x",
        base_1[0],
        gemx_1_vec[0],
        gemx_1_vec[0] / base_1[0]
    );
    println!(
        "  Batch=16 | {:9.4} | {:9.4} | {:6.3}x",
        base_16[0],
        gemx_16_vec[0],
        gemx_16_vec[0] / base_16[0]
    );

    // Check consistency within batch=16
    let first_row_16 = gemx_16_vec[0];
    let mut batch_variance = 0.0f32;
    for b in 0..16 {
        let row_val = gemx_16_vec[b * nrows]; // First output row of each batch
        batch_variance = batch_variance.max((row_val - first_row_16).abs());
    }
    println!("\n  Batch=16 inter-batch variance: {:.6}", batch_variance);

    let ratio_1 = gemx_1_vec[0] / base_1[0];
    let ratio_16 = gemx_16_vec[0] / base_16[0];

    if (ratio_1 - 1.0).abs() < 0.05 && (ratio_16 - 1.0).abs() < 0.05 {
        println!("  ✓ Both batch sizes produce correct results");
    } else if (ratio_1 - 1.0).abs() < 0.05 && (ratio_16 - 2.0).abs() < 0.2 {
        println!("  ❌ Batch=1 is correct, but Batch=16 is 2x!");
        println!("     This indicates the kernel reads the same K-data twice for GEMM path");
    } else if (ratio_16 - 0.5).abs() < 0.1 {
        println!("  ❌ Batch=16 is 0.5x - half the K-data is missing");
    }

    // Check which K-regions contribute differently at batch=16
    println!("\n  Testing K-region contribution difference:");
    for k_start in (0..256).step_by(64) {
        let k_end = k_start + 64;

        let mut input_data_1 = vec![0.0f32; ncols];
        let mut input_data_16 = vec![0.0f32; 16 * ncols];
        for k in k_start..k_end {
            input_data_1[k] = 1.0;
            for b in 0..16 {
                input_data_16[b * ncols + k] = 1.0;
            }
        }

        let inp_1 = Tensor::from_vec(input_data_1, (1, 1, ncols), device)?;
        let inp_16 = Tensor::from_vec(input_data_16, (16, 1, ncols), device)?;

        let bl_1 = compute_baseline(&qtensor, &inp_1, 1, 1, nrows, ncols, device)?;
        let qm_1 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let ml_1 = qm_1.forward(&inp_1)?;

        let bl_16 = compute_baseline(&qtensor, &inp_16, 16, 1, nrows, ncols, device)?;
        let qm_16 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let ml_16 = qm_16.forward(&inp_16)?;

        let bl_1_v: Vec<f32> = bl_1.flatten_all()?.to_vec1()?;
        let ml_1_v: Vec<f32> = ml_1.flatten_all()?.to_vec1()?;
        let bl_16_v: Vec<f32> = bl_16.flatten_all()?.to_vec1()?;
        let ml_16_v: Vec<f32> = ml_16.flatten_all()?.to_vec1()?;

        let r1 = ml_1_v[0] / bl_1_v[0];
        let r16 = ml_16_v[0] / bl_16_v[0];

        let status = if (r1 - r16).abs() < 0.1 {
            "same"
        } else {
            "DIFF!"
        };
        println!(
            "    K={:3}-{:3}: ratio@b=1={:.3}x, ratio@b=16={:.3}x [{}]",
            k_start,
            k_end - 1,
            r1,
            r16,
            status
        );
    }

    Ok(())
}

/// Diagnostic 18: Single K-element probe test
/// Tests each individual K element (not K-row) to find exactly which ones are being duplicated.
/// Uses Y[k]=1.0 for exactly one k, and all weights=1.0
#[cfg(feature = "cuda")]
pub fn run_single_k_probe_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 18] Single K-element probe (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Use non-degenerate weights to avoid zero scales for K-quants
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test each K element in the first K-tile (0-15) and second K-tile (16-31)
    println!("  Testing individual K elements:");
    println!("  k   | K-row | pos-in-row | Expected | Actual   | Ratio  | Note");
    println!("  ----|-------|------------|----------|----------|--------|------");

    let mut observations = Vec::new();

    for k in 0..64 {
        // Test first 4 K-rows in detail
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            input_data[b * ncols + k] = 1.0;
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let expected = baseline_vec[0];
        let actual = gemx_vec[0];
        let ratio = if expected.abs() > 0.001 {
            actual / expected
        } else {
            0.0
        };

        let k_row = k / 16;
        let pos = k % 16;

        let note = if (ratio - 2.0).abs() < 0.2 {
            "2x!"
        } else if (ratio - 1.0).abs() < 0.1 {
            "OK"
        } else if ratio.abs() < 0.1 {
            "ZERO"
        } else {
            "??"
        };

        println!(
            "  {:3} | {:5} | {:10} | {:8.4} | {:8.4} | {:6.3}x | {}",
            k, k_row, pos, expected, actual, ratio, note
        );

        observations.push((k, k_row, pos, ratio, note.to_string()));
    }

    // Analyze pattern
    let doubled: Vec<_> = observations.iter().filter(|o| o.4 == "2x!").collect();
    let zeros: Vec<_> = observations.iter().filter(|o| o.4 == "ZERO").collect();

    println!("\n  Pattern Analysis:");
    if !doubled.is_empty() {
        println!(
            "  Doubled elements (2x): k = {:?}",
            doubled.iter().map(|o| o.0).collect::<Vec<_>>()
        );

        // Check if there's a pattern
        let k_rows_doubled: std::collections::HashSet<_> = doubled.iter().map(|o| o.1).collect();
        println!("  K-rows with doubled elements: {:?}", k_rows_doubled);

        let positions_doubled: std::collections::HashSet<_> = doubled.iter().map(|o| o.2).collect();
        println!(
            "  Positions within K-row that are doubled: {:?}",
            positions_doubled
        );
    }

    if !zeros.is_empty() {
        println!(
            "  Zero elements: k = {:?}",
            zeros.iter().map(|o| o.0).collect::<Vec<_>>()
        );
    }

    Ok(())
}

/// Diagnostic 19: K-iteration count test
/// Uses orthogonal inputs to count effective iterations
/// Input at different batch indices should sum independently
#[cfg(feature = "cuda")]
pub fn run_k_iteration_count_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 19] K-iteration count test...");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Create weights with distinct patterns per K-row
    // W[n, k] = k / 255.0 (so each K column has unique value)
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k in 0..ncols {
            weight_data[n * ncols + k] = k as f32 / 255.0;
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Create input: Y[b, k] = 1.0 if b == k % 16 (each batch index tests different K elements)
    let mut input_data = vec![0.0f32; batch * ncols];
    for k in 0..ncols {
        let b = k % 16;
        input_data[b * ncols + k] = 1.0;
    }
    let input = Tensor::from_vec(input_data.clone(), (batch, 1, ncols), device)?;

    // Compute baseline and GEMX
    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx_result = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

    println!("  Batch index test (each batch tests different K elements):");
    println!("  batch | K elems           | Expected | Actual   | Ratio");
    println!("  ------|-------------------|----------|----------|-------");

    for b in 0..16 {
        let expected = baseline_vec[b * nrows]; // First output element for batch b
        let actual = gemx_vec[b * nrows];
        let ratio = if expected.abs() > 0.001 {
            actual / expected
        } else {
            0.0
        };

        // K elements tested by this batch: b, b+16, b+32, ...
        let k_elems = format!("{},{},{},{},...", b, b + 16, b + 32, b + 48);
        println!(
            "  {:5} | {:17} | {:8.4} | {:8.4} | {:.3}x",
            b, k_elems, expected, actual, ratio
        );
    }

    Ok(())
}
