//! Diagnostics that find which K-rows a quantised matmul reads, and how many
//! times each is read.
//!
//! A wrap test covers the 256-wide K dimension in quarters, a per-position probe
//! sets one input at a time, and a read-pattern analysis weights each K-row so
//! the observed sum can be set against what a single read of each row would
//! give. The ratio shows how many times a row is read.

use super::*;

/// Diagnostic 26: Test K-row wrapping with larger K dimension
/// Uses the standard 256x256 dimensions that work with the kernel
/// Varies INPUT only (not weights) to avoid repack issues
#[cfg(feature = "cuda")]
pub fn run_krow_wrap_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 26] K-row wrapping pattern test...");

    // Create a single qtensor with uniform weights for all tests
    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Use Tensor::full which should definitely work
    let weights = Tensor::full(0.25f32, (nrows, ncols), device)?;
    let qtensor = match QTensor::quantize(&weights, ggml_dtype) {
        Ok(q) => q,
        Err(e) => {
            println!("  Warning: Could not quantize weights: {}", e);
            return Ok(());
        }
    };
    let qmatmul = match QMatMulWrapper::from_qtensor(qtensor.clone()) {
        Ok(q) => q,
        Err(e) => {
            println!("  Warning: Could not create QMatMulWrapper: {}", e);
            return Ok(());
        }
    };

    // Test 1: Only activate K=0-63 (K-rows 0-3)
    println!("  Test 1: Input only at K=0-63 (K-rows 0-3)");
    {
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in 0..64 {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let gemx = qmatmul.forward(&input)?;

        let exp: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let act: f32 = gemx.flatten_all()?.to_vec1::<f32>()?[0];

        let ratio = if exp.abs() > 0.001 { act / exp } else { 0.0 };
        println!("    Expected: {:.4}", exp);
        println!("    Actual:   {:.4}", act);
        println!("    Ratio:    {:.2}x", ratio);
        if (ratio - 4.0).abs() < 0.2 {
            println!("    ❌ K-rows 0-3 are being read 4x!");
        } else if (ratio - 1.0).abs() < 0.2 {
            println!("    ✓ K-rows 0-3 read correctly 1x");
        }
    }

    // Test 2: Only activate K=64-127 (K-rows 4-7)
    println!("\n  Test 2: Input only at K=64-127 (K-rows 4-7)");
    {
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in 64..128 {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let gemx = qmatmul.forward(&input)?;

        let exp: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let act: f32 = gemx.flatten_all()?.to_vec1::<f32>()?[0];

        println!("    Expected: {:.4}", exp);
        println!("    Actual:   {:.4}", act);
        if act.abs() < 0.01 {
            println!("    ❌ ZERO: K-rows 4-7 are not contributing!");
        } else {
            let ratio = act / exp;
            println!("    Ratio:    {:.2}x", ratio);
        }
    }

    // Test 3: Only activate K=128-191 (K-rows 8-11)
    println!("\n  Test 3: Input only at K=128-191 (K-rows 8-11)");
    {
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in 128..192 {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let gemx = qmatmul.forward(&input)?;

        let exp: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let act: f32 = gemx.flatten_all()?.to_vec1::<f32>()?[0];

        println!("    Expected: {:.4}", exp);
        println!("    Actual:   {:.4}", act);
        if act.abs() < 0.01 {
            println!("    ❌ ZERO: K-rows 8-11 are not contributing!");
        }
    }

    // Test 4: Only activate K=192-255 (K-rows 12-15)
    println!("\n  Test 4: Input only at K=192-255 (K-rows 12-15)");
    {
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in 192..256 {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let gemx = qmatmul.forward(&input)?;

        let exp: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let act: f32 = gemx.flatten_all()?.to_vec1::<f32>()?[0];

        println!("    Expected: {:.4}", exp);
        println!("    Actual:   {:.4}", act);
        if act.abs() < 0.01 {
            println!("    ❌ ZERO: K-rows 12-15 are not contributing!");
        }
    }

    Ok(())
}

/// Diagnostic 27: Y input isolation test
/// Sets Y[k]=1 for single k positions to see which input positions contribute
#[cfg(feature = "cuda")]
pub fn run_y_input_isolation_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 27] Y input isolation test (batch=16)...");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Use standard test weights (reuse single qtensor)
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;

    // Test each K position individually to find which ones contribute
    println!("  Weights: random test weights in [-0.5, 0.5]");
    println!("  Test: Y[k]=1, all others=0, check if output differs from zero");
    println!("\n  Scanning K positions (showing non-zero contributors)...");

    let mut contributing_k = Vec::new();
    let mut zero_k = Vec::new();

    // Test every 16th K position (one per K-row) for speed
    for k_row in 0..16 {
        let k = k_row * 16; // First element of each K-row

        let mut inp_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            inp_data[b * ncols + k] = 1.0;
        }
        let inp = Tensor::from_vec(inp_data, (batch, 1, ncols), device)?;

        let gemx = qmatmul.forward(&inp)?;
        let gemx_v: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

        // Check if output has non-zero values
        let max_abs = gemx_v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if max_abs > 0.001 {
            contributing_k.push((k_row, max_abs));
        } else {
            zero_k.push(k_row);
        }
    }

    println!("\n  K-rows that contribute (first element tested):");
    for (row, max_val) in &contributing_k {
        println!(
            "    K-row {:2} (K={}): max output={:.4}",
            row,
            row * 16,
            max_val
        );
    }

    if !zero_k.is_empty() {
        println!("\n  K-rows with ZERO output (not loaded!):");
        println!("    K-rows {:?}", zero_k);
    }

    // Summary
    println!("\n  Summary:");
    println!(
        "    Contributing K-rows: {} out of 16",
        contributing_k.len()
    );
    println!("    Zero K-rows: {} out of 16", zero_k.len());

    if contributing_k.len() == 4 && zero_k.len() == 12 {
        println!("    Pattern: Only first 4 K-rows (0-3) contribute!");
        println!("    This confirms the kernel only loads K=0-63");
    }

    Ok(())
}

/// Diagnostic 28: First 4 K-rows accuracy test
/// Tests if K-rows 0-3 compute correctly in isolation (ignoring that they're over-read)
#[cfg(feature = "cuda")]
pub fn run_first_4_krows_accuracy_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 28] First 4 K-rows accuracy test (batch=16)...");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Use standard test weights
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;

    println!("  Testing each K-row in isolation...");
    println!("  (Using only Y positions for that K-row, expect ratio=4x due to over-reading)\n");

    for k_row in 0..4 {
        let k_start = k_row * 16;
        let k_end = k_start + 16;

        let mut inp_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in k_start..k_end {
                inp_data[b * ncols + k] = 1.0;
            }
        }
        let inp = Tensor::from_vec(inp_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &inp, batch, 1, nrows, ncols, device)?;
        let gemx = qmatmul.forward(&inp)?;

        let bl_v: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let ml_v: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

        let ratio = if bl_v[0].abs() > 0.001 {
            ml_v[0] / bl_v[0]
        } else {
            0.0
        };

        // The ratio tells us how many times this K-row is being read
        println!(
            "    K-row {} (K={:3}-{:3}): exp={:.4}, act={:.4}, ratio={:.2}x",
            k_row,
            k_start,
            k_end - 1,
            bl_v[0],
            ml_v[0],
            ratio
        );
    }

    println!("\n  If all ratios ≈ 4.0x, this confirms K-rows 0-3 are each read 4 times.");
    println!("  If ratios vary (e.g., 4x, 2x, 2.67x, 2x), there's a more complex issue.");

    Ok(())
}

/// Diagnostic 29: K-row read pattern analysis
/// Uses weighted K-row values to determine exactly which K-rows are being read
#[cfg(feature = "cuda")]
pub fn run_krow_read_pattern_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 29] K-row read pattern analysis (batch=16)...");

    let nrows = 256;
    let ncols = 256;
    let batch = 16;

    // Use standard test weights
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;

    // Test with all 1s input
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let gemx = qmatmul.forward(&input)?;

    let bl_v: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let ml_v: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let exp = bl_v[0];
    let act = ml_v[0];

    println!("  Full input test (all Y[k]=1):");
    println!("    Expected: {:.4}", exp);
    println!("    Actual:   {:.4}", act);
    println!("    Ratio:    {:.4}x", act / exp);

    // Compute what the ratio would be if only K-rows 0-3 are read 4x
    // Expected = sum over all K, Actual = 4 * sum over K=0-63

    // Test: what's the contribution of K-rows 0-3 vs rest?
    println!("\n  Contribution breakdown:");

    let mut k_0_3_sum = 0.0f32;
    let mut k_4_15_sum = 0.0f32;

    for k_row in 0..16 {
        let k_start = k_row * 16;
        let k_end = k_start + 16;

        let mut inp_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in k_start..k_end {
                inp_data[b * ncols + k] = 1.0;
            }
        }
        let inp = Tensor::from_vec(inp_data, (batch, 1, ncols), device)?;
        let bl = compute_baseline(&qtensor, &inp, batch, 1, nrows, ncols, device)?;
        let contribution: f32 = bl.flatten_all()?.to_vec1::<f32>()?[0];

        if k_row < 4 {
            k_0_3_sum += contribution;
        } else {
            k_4_15_sum += contribution;
        }
    }

    println!("    K-rows 0-3 baseline contribution:  {:.4}", k_0_3_sum);
    println!("    K-rows 4-15 baseline contribution: {:.4}", k_4_15_sum);
    println!("    Total baseline: {:.4}", k_0_3_sum + k_4_15_sum);

    // If only K-rows 0-3 are read 4x:
    let predicted_act = k_0_3_sum * 4.0;
    println!("\n  If K-rows 0-3 read 4x (others ignored):");
    println!("    Predicted: {:.4}", predicted_act);
    println!("    Actual:    {:.4}", act);
    println!(
        "    Match:     {:.2}%",
        100.0 * (1.0 - (predicted_act - act).abs() / act.abs().max(0.001))
    );

    if ((predicted_act - act).abs() / act.abs().max(0.001)) < 0.1 {
        println!("\n  ✓ CONFIRMED: Kernel reads only K-rows 0-3, each 4 times");
    }

    Ok(())
}
