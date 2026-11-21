//! Diagnostics that trace which output columns receive data, to find where a
//! column mapping goes wrong.
//!
//! A matrix with one isolated value per column shows where each column lands.
//! Single-column probes place one column at a time, and the column-pair pattern
//! test checks whether the working columns follow the expected every-other-pair
//! layout.

use super::*;

/// Diagnostic 30: Column identity test - trace which output columns receive data
/// Uses a zero-weight matrix with isolated column values to see where each column maps
#[cfg(feature = "cuda")]
pub fn run_column_identity_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 30] Column identity test (batch=16)...");
    println!("  Purpose: Trace which output columns receive data from specific weight columns");

    let nrows = 256; // N = output dimension
    let ncols = 256; // K = input dimension
    let batch = 16;

    // Create weights where column n has value (n+1)*0.1 at K=0 only, rest is 0
    // W[n, k=0] = (n+1)*0.1, W[n, k>0] = 0
    // With Y[0]=1, Y[k>0]=0: expected out[n] = (n+1)*0.1
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        weight_data[n * ncols + 0] = (n as f32 + 1.0) * 0.1;
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;

    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = match QMatMulWrapper::from_qtensor(qtensor.clone()) {
        Ok(q) => q,
        Err(e) => {
            println!("  Warning: Could not create QMatMulWrapper: {}", e);
            return Ok(());
        }
    };

    // Input: Y[0]=1, rest=0
    let mut input_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        input_data[b * ncols + 0] = 1.0;
    }
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let gemx = qmatmul.forward(&input)?;

    let bl: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let ml: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    // Analyze the pattern for batch 0
    println!("\n  Output column pattern (batch 0, first 32 columns):");
    println!("  n  | Expected | Actual   | Ratio | Status");
    println!("  ---|----------|----------|-------|-------");

    let mut zero_cols = Vec::new();
    let mut nonzero_cols = Vec::new();

    for n in 0..32 {
        let exp = bl[n];
        let act = ml[n];
        let status = if act.abs() < 0.001 {
            zero_cols.push(n);
            "ZERO"
        } else if (exp - act).abs() < 0.1 {
            nonzero_cols.push((n, 1.0));
            "OK"
        } else {
            let ratio = if exp.abs() > 0.001 { act / exp } else { act };
            nonzero_cols.push((n, ratio));
            "WRONG"
        };
        if n < 20 || status != "WRONG" && status != "OK" {
            println!(
                "  {:3}| {:8.4} | {:8.4} | {:5.2} | {}",
                n,
                exp,
                act,
                if exp.abs() > 0.001 { act / exp } else { 0.0 },
                status
            );
        }
    }

    println!("\n  Summary:");
    println!("    Zero columns (0-31): {:?}", zero_cols);
    println!("    Pattern analysis:");

    // Check if pattern matches "every other pair" (0,1 work, 2,3 zero, 4,5 work, etc.)
    let mut pattern_0123 = true;
    for i in 0..8 {
        let c0 = i * 4;
        let c1 = c0 + 1;
        let c2 = c0 + 2;
        let c3 = c0 + 3;
        // Expected: c0,c1 have values, c2,c3 are zero
        let c0_nonzero = ml[c0].abs() > 0.001;
        let c1_nonzero = ml[c1].abs() > 0.001;
        let c2_zero = ml[c2].abs() < 0.001;
        let c3_zero = ml[c3].abs() < 0.001;
        if !(c0_nonzero && c1_nonzero && c2_zero && c3_zero) {
            pattern_0123 = false;
        }
    }

    if pattern_0123 {
        println!("    ✓ Pattern confirmed: columns 0,1,4,5,8,9... receive data; 2,3,6,7,10,11... are ZERO");
        println!("    This suggests a warp-level column assignment bug in fetch_to_registers or write_result");
    } else {
        // Show actual pattern
        print!("    Non-zero columns: ");
        for n in 0..32 {
            if ml[n].abs() > 0.001 {
                print!("{}, ", n);
            }
        }
        println!();
    }

    Ok(())
}

/// Diagnostic 31: Single column probe - set one column to identity, trace where it ends up
#[cfg(feature = "cuda")]
pub fn run_single_column_probe_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 31] Single column probe (batch=16)...");
    println!("  Purpose: For each input column c, set W[c,0]=1, see where it maps in output");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // For columns 0-15, create a mapping matrix
    println!("\n  Column mapping (where does weight column N end up in output?):");
    println!("  In Col | Output positions with data (first 32 outputs checked)");
    println!("  -------|--------------------------------------------------------");

    for probe_col in 0..16 {
        // Create weights with only W[probe_col, 0] = 1
        let mut weight_data = vec![0.0f32; nrows * ncols];
        weight_data[probe_col * ncols + 0] = 1.0;
        let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;

        let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
        let qmatmul = match QMatMulWrapper::from_qtensor(qtensor.clone()) {
            Ok(q) => q,
            Err(e) => {
                println!("  {:5} | Error: {}", probe_col, e);
                continue;
            }
        };

        // Input: Y[0]=1
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            input_data[b * ncols + 0] = 1.0;
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;
        let gemx = qmatmul.forward(&input)?;
        let ml: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

        // Find which output positions have data
        let mut out_positions = Vec::new();
        for n in 0..32 {
            if ml[n].abs() > 0.001 {
                out_positions.push((n, ml[n]));
            }
        }

        print!("  {:5} | ", probe_col);
        if out_positions.is_empty() {
            println!("NONE (all zero!)");
        } else {
            for (pos, val) in &out_positions {
                print!("out[{}]={:.2} ", pos, val);
            }
            println!();
        }
    }

    Ok(())
}

/// Diagnostic 32: Warp signature test - identify which warps contribute to which outputs
#[cfg(feature = "cuda")]
pub fn run_warp_signature_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 32] Warp signature analysis (batch=16)...");
    println!("  Purpose: Use unique per-K-row values to trace warp data flow");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Each K-row gets a unique "signature" value: row R has value 2^R (for first 10 rows)
    // This lets us trace which K-row data ends up where by looking at output bits
    println!("\n  Setting K-row signatures: K-row R has weight = R+1");

    // Create weights where all elements in K-row R have value (R+1)
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k_row in 0..16 {
            let k_start = k_row * 16;
            for k_off in 0..16 {
                weight_data[n * ncols + k_start + k_off] = (k_row + 1) as f32;
            }
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;

    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = match QMatMulWrapper::from_qtensor(qtensor.clone()) {
        Ok(q) => q,
        Err(e) => {
            println!("  Warning: Could not create QMatMulWrapper: {}", e);
            return Ok(());
        }
    };

    // Test 1: All Y=1 - should give sum(R+1 for R=0..15) * 16 = 136 * 16 = 2176 per output
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let gemx = qmatmul.forward(&input)?;

    let _bl: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let ml: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let expected_total = 136.0 * 16.0; // sum(1..16) * 16 elements per row

    println!("\n  With all Y[k]=1:");
    println!("    Expected per output: {:.1}", expected_total);
    println!("    Actual out[0]: {:.1}", ml[0]);
    println!("    Ratio: {:.3}x", ml[0] / expected_total);

    // Test 2: Individual K-row activation - isolate which K-rows map to output
    println!("\n  Per K-row isolation (Y[k]=1 only for that K-row's 16 elements):");
    println!("  K-row | Expected | out[0]   | out[1]   | out[2]   | out[3]   | Pattern");
    println!("  ------|----------|----------|----------|----------|----------|--------");

    for k_row in 0..8 {
        let k_start = k_row * 16;
        let k_end = k_start + 16;

        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in k_start..k_end {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;
        let gemx = qmatmul.forward(&input)?;
        let ml: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

        let expected = (k_row + 1) as f32 * 16.0;

        // Analyze pattern across first 4 outputs
        let vals = [ml[0], ml[1], ml[2], ml[3]];
        let pattern = if vals.iter().all(|&v| v.abs() < 0.001) {
            "ALL ZERO"
        } else if vals.iter().all(|&v| (v - expected).abs() < 1.0) {
            "CORRECT"
        } else {
            "MIXED"
        };

        println!(
            "  {:5} | {:8.1} | {:8.1} | {:8.1} | {:8.1} | {:8.1} | {}",
            k_row, expected, vals[0], vals[1], vals[2], vals[3], pattern
        );
    }

    Ok(())
}

/// Diagnostic 33: Column pair pattern analysis - detailed analysis of the 0,1,4,5... pattern
#[cfg(feature = "cuda")]
pub fn run_column_pair_pattern_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 33] Column pair pattern analysis (batch=16)...");
    println!("  Purpose: Analyze the 2-out-of-4 column pattern in detail");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Create uniform weights so all columns should have equal output
    let weight_data = vec![1.0f32; nrows * ncols];
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;

    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = match QMatMulWrapper::from_qtensor(qtensor.clone()) {
        Ok(q) => q,
        Err(e) => {
            println!("  Warning: Could not create QMatMulWrapper: {}", e);
            return Ok(());
        }
    };

    // Input: uniform 1s
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let gemx = qmatmul.forward(&input)?;

    let bl: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let ml: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let expected = bl[0]; // Should be 256 (sum of 256 1s)

    println!("\n  Expected output (uniform): {:.1}", expected);
    println!("\n  Column grouping analysis (groups of 4):");
    println!("  Group | Col+0    | Col+1    | Col+2    | Col+3    | Pattern");
    println!("  ------|----------|----------|----------|----------|----------");

    let mut pattern_counts = std::collections::HashMap::new();

    for group in 0..16 {
        let c0 = group * 4;
        let vals = [ml[c0], ml[c0 + 1], ml[c0 + 2], ml[c0 + 3]];

        // Categorize the pattern
        let pattern = format!(
            "{}{}{}{}",
            if vals[0].abs() > 1.0 { "X" } else { "0" },
            if vals[1].abs() > 1.0 { "X" } else { "0" },
            if vals[2].abs() > 1.0 { "X" } else { "0" },
            if vals[3].abs() > 1.0 { "X" } else { "0" }
        );

        *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;

        println!(
            "  {:5} | {:8.1} | {:8.1} | {:8.1} | {:8.1} | {}",
            group, vals[0], vals[1], vals[2], vals[3], pattern
        );
    }

    println!("\n  Pattern summary:");
    for (pattern, count) in &pattern_counts {
        println!("    {}: {} groups", pattern, count);
    }

    // Analyze: if pattern is XX00 (first two have data, last two zero)
    // This indicates b_sh_rd thread indexing issue
    let xx00_count = pattern_counts.get("XX00").unwrap_or(&0);
    let total_groups = 16;

    if *xx00_count == total_groups {
        println!("\n  DIAGNOSIS: All groups show XX00 pattern");
        println!("    This means b_sh_rd = threadIdx.x assigns threads 0,1 to columns 0,1");
        println!("    but threads 2,3 write to wrong locations (or their data is overwritten)");
        println!("    ");
        println!("    Likely cause: MMA fragment layout mismatch");
        println!("    - Thread's frag_c[i][j] doesn't map to expected output column");
        println!("    - write_result's c_sh_wr calculation may be incorrect");
    }

    // Additional test: Check if the "working" columns have correct values
    let mut working_cols_correct = 0;
    let mut working_cols_wrong = 0;
    for n in 0..64 {
        if ml[n].abs() > 1.0 {
            if (ml[n] - expected).abs() / expected < 0.05 {
                working_cols_correct += 1;
            } else {
                working_cols_wrong += 1;
            }
        }
    }

    println!("\n  Among non-zero columns (first 64):");
    println!("    Correct value: {} columns", working_cols_correct);
    println!("    Wrong value:   {} columns", working_cols_wrong);

    if working_cols_wrong > 0 {
        println!("\n    Even working columns have wrong values!");
        println!("    This suggests multiple issues:");
        println!("    1. K-iteration: reading wrong K-rows (too few or wrong ones)");
        println!("    2. Column mapping: data going to wrong columns");
    }

    Ok(())
}
