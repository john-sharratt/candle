//! Diagnostics that isolate the kernel pipeline: stage transitions, the
//! thread-to-column mapping, scale application, K-tile pairing, the
//! global-to-shared load and the MMA fragment.
//!
//! Each uses weights chosen so that the expected output is a known sum, such as
//! a distinct value for each 64-element stage or each K-row. A stage that does
//! not advance, a K-tile that does not load or a column that maps to the wrong
//! place shows up as a sum that counts only part of the expected terms.

use super::*;

// KERNEL COMPONENT ISOLATION DIAGNOSTICS
// These tests use clever weight/scale/input patterns to isolate specific
// kernel components and identify which part is malfunctioning.

/// Diagnostic 20: Stage boundary test
/// Tests whether the kernel correctly transitions between pipeline stages.
/// Uses weights where each 64-K-element block has a distinct value (1, 2, 3, 4).
/// If stages aren't advancing, we'll see only one value contributing.
#[cfg(feature = "cuda")]
pub fn run_stage_boundary_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!("\n[Diagnostic 20] Stage boundary test (batch={})...", batch);

    let nrows = 256;
    let ncols = 256; // 4 stages of 64 K-elements each

    // Create weights where stage S (K = S*64 to S*64+63) has value S+1
    // Stage 0: K=0-63 → weight=1.0
    // Stage 1: K=64-127 → weight=2.0
    // Stage 2: K=128-191 → weight=3.0
    // Stage 3: K=192-255 → weight=4.0
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k in 0..ncols {
            let stage = k / 64;
            weight_data[n * ncols + k] = (stage + 1) as f32;
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Input: all 1s
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    // Expected: sum of (1*64 + 2*64 + 3*64 + 4*64) = 64*(1+2+3+4) = 64*10 = 640
    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let expected = baseline_vec[0];
    let actual = gemx_vec[0];
    let ratio = actual / expected;

    println!("  Weights: stage 0=1.0, stage 1=2.0, stage 2=3.0, stage 3=4.0");
    println!("  Input: all 1.0");
    println!("  Expected sum: {:.2} (if all stages contribute)", expected);
    println!("  Actual sum:   {:.2}", actual);
    println!("  Ratio:        {:.3}x", ratio);

    // Diagnose which stages contributed
    // If only stage 0 (weight=1): actual ≈ 64
    // If only stages 0,1 (weight=1,2): actual ≈ 64*3 = 192
    // If all stages: actual ≈ 640
    if (actual - 64.0).abs() < 10.0 {
        println!("  ⚠️  Only stage 0 appears to contribute (K=0-63)");
    } else if (actual - 192.0).abs() < 30.0 {
        println!("  ⚠️  Only stages 0-1 appear to contribute (K=0-127)");
    } else if (ratio - 1.0).abs() < 0.1 {
        println!("  ✓ All stages contribute correctly");
    } else {
        println!("  ❌ Unexpected pattern - ratio={:.3}x", ratio);
    }

    // Also test each stage in isolation
    println!("\n  Per-stage isolation test:");
    for stage in 0..4 {
        let k_start = stage * 64;
        let k_end = k_start + 64;

        let mut inp_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in k_start..k_end {
                inp_data[b * ncols + k] = 1.0;
            }
        }
        let inp = Tensor::from_vec(inp_data, (batch, 1, ncols), device)?;

        let bl = compute_baseline(&qtensor, &inp, batch, 1, nrows, ncols, device)?;
        let qm = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let ml = qm.forward(&inp)?;

        let bl_v: Vec<f32> = bl.flatten_all()?.to_vec1()?;
        let ml_v: Vec<f32> = ml.flatten_all()?.to_vec1()?;

        let exp = bl_v[0];
        let act = ml_v[0];
        let r = if exp.abs() > 0.01 { act / exp } else { 0.0 };

        let status = if (r - 1.0).abs() < 0.1 {
            "OK"
        } else if act.abs() < 0.01 {
            "ZERO!"
        } else {
            "WRONG"
        };

        println!(
            "    Stage {} (K={:3}-{:3}, weight={}): exp={:8.2}, act={:8.2}, ratio={:.3}x [{}]",
            stage,
            k_start,
            k_end - 1,
            stage + 1,
            exp,
            act,
            r,
            status
        );
    }

    Ok(())
}

/// Diagnostic 21: Thread mapping test
/// Uses weights where column N has value N (0-255).
/// This tests whether threads correctly map to output columns.
#[cfg(feature = "cuda")]
pub fn run_thread_mapping_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 21] Thread-to-column mapping test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Weights: W[n, k] = n (output column identity)
    // This means output[n] = n * sum(input)
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k in 0..ncols {
            weight_data[n * ncols + k] = n as f32;
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Input: single 1.0 at K=0 (sum=1)
    let mut input_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        input_data[b * ncols] = 1.0; // Only K=0
    }
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    println!("  Weights: W[n,k] = n (column identity)");
    println!("  Input: Y[0]=1, others=0");
    println!("  Expected: out[n] ≈ n");

    // Check first batch's output columns
    let mut mismatches = 0;
    let mut first_mismatch = None;
    for n in 0..nrows {
        let exp = baseline_vec[n];
        let act = gemx_vec[n];
        if (exp - act).abs() > 1.0 {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((n, exp, act));
            }
        }
    }

    if mismatches == 0 {
        println!("  ✓ All 256 output columns map correctly");
    } else {
        println!("  ❌ {} column mismatches", mismatches);
        if let Some((n, exp, act)) = first_mismatch {
            println!(
                "     First mismatch at n={}: expected={:.2}, actual={:.2}",
                n, exp, act
            );
        }

        // Show pattern of first 16 outputs
        println!("  First 16 outputs (batch 0):");
        for n in 0..16 {
            let exp = baseline_vec[n];
            let act = gemx_vec[n];
            let marker = if (exp - act).abs() > 1.0 { " *" } else { "" };
            println!(
                "    out[{:2}]: exp={:6.2}, act={:6.2}{}",
                n, exp, act, marker
            );
        }
    }

    Ok(())
}

/// Diagnostic 22: Scale application test
/// Creates weights where the dequantized value depends heavily on the scale.
/// Tests whether scales are correctly loaded and applied per K-tile.
#[cfg(feature = "cuda")]
pub fn run_scale_application_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 22] Scale application test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Create weights with maximum quantized values
    // This makes the output very sensitive to scale values
    let weights = Tensor::full(7.0f32, (nrows, ncols), device)?; // Max for signed 4-bit
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Input: all 1s
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let exp = baseline_vec[0];
    let act = gemx_vec[0];
    let ratio = act / exp;

    println!("  Weights: all 7.0 (max signed 4-bit)");
    println!("  Input: all 1.0");
    println!("  Expected: {:.4}", exp);
    println!("  Actual:   {:.4}", act);
    println!("  Ratio:    {:.4}x", ratio);

    if (ratio - 1.0).abs() < 0.05 {
        println!("  ✓ Scales applied correctly");
    } else if (ratio - 4.0).abs() < 0.5 {
        println!("  ⚠️  4x error - scales may be applied 4 times");
    } else if ratio.abs() < 0.01 {
        println!("  ⚠️  Near zero - scales may not be loaded");
    } else {
        println!("  ❌ Unexpected scale behavior");
    }

    Ok(())
}

/// Diagnostic 23: K-tile pair loading test
/// Tests whether both K-tiles in a 32-element logical tile are loaded.
/// Uses weights where K-tile 0 (K=0-15) = 1.0 and K-tile 1 (K=16-31) = 2.0
#[cfg(feature = "cuda")]
pub fn run_ktile_pair_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 23] K-tile pair loading test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256;

    // Weights: alternating K-tiles within each 32-element block
    // K=0-15 (tile 0) → 1.0
    // K=16-31 (tile 1) → 2.0
    // K=32-47 (tile 0 of block 1) → 1.0
    // etc.
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k in 0..ncols {
            let within_block = k % 32;
            weight_data[n * ncols + k] = if within_block < 16 { 1.0 } else { 2.0 };
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test 1: Input only in first K-tile (K=0-15)
    let mut inp1_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        for k in 0..16 {
            inp1_data[b * ncols + k] = 1.0;
        }
    }
    let inp1 = Tensor::from_vec(inp1_data, (batch, 1, ncols), device)?;

    // Test 2: Input only in second K-tile (K=16-31)
    let mut inp2_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        for k in 16..32 {
            inp2_data[b * ncols + k] = 1.0;
        }
    }
    let inp2 = Tensor::from_vec(inp2_data, (batch, 1, ncols), device)?;

    // Test 3: Input in both K-tiles (K=0-31)
    let mut inp3_data = vec![0.0f32; batch * ncols];
    for b in 0..batch {
        for k in 0..32 {
            inp3_data[b * ncols + k] = 1.0;
        }
    }
    let inp3 = Tensor::from_vec(inp3_data, (batch, 1, ncols), device)?;

    let bl1 = compute_baseline(&qtensor, &inp1, batch, 1, nrows, ncols, device)?;
    let bl2 = compute_baseline(&qtensor, &inp2, batch, 1, nrows, ncols, device)?;
    let bl3 = compute_baseline(&qtensor, &inp3, batch, 1, nrows, ncols, device)?;

    let qm1 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let ml1 = qm1.forward(&inp1)?;
    let qm2 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let ml2 = qm2.forward(&inp2)?;
    let qm3 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let ml3 = qm3.forward(&inp3)?;

    let bl1_v: Vec<f32> = bl1.flatten_all()?.to_vec1()?;
    let bl2_v: Vec<f32> = bl2.flatten_all()?.to_vec1()?;
    let bl3_v: Vec<f32> = bl3.flatten_all()?.to_vec1()?;
    let ml1_v: Vec<f32> = ml1.flatten_all()?.to_vec1()?;
    let ml2_v: Vec<f32> = ml2.flatten_all()?.to_vec1()?;
    let ml3_v: Vec<f32> = ml3.flatten_all()?.to_vec1()?;

    let r1 = ml1_v[0] / bl1_v[0];
    let r2 = ml2_v[0] / bl2_v[0];
    let r3 = ml3_v[0] / bl3_v[0];

    println!(
        "  K-tile 0 (K=0-15, w=1): exp={:8.4}, act={:8.4}, ratio={:.3}x",
        bl1_v[0], ml1_v[0], r1
    );
    println!(
        "  K-tile 1 (K=16-31, w=2): exp={:8.4}, act={:8.4}, ratio={:.3}x",
        bl2_v[0], ml2_v[0], r2
    );
    println!(
        "  Both tiles (K=0-31):     exp={:8.4}, act={:8.4}, ratio={:.3}x",
        bl3_v[0], ml3_v[0], r3
    );

    let tile0_ok = (r1 - 1.0).abs() < 0.1;
    let tile1_ok = (r2 - 1.0).abs() < 0.1;
    let both_ok = (r3 - 1.0).abs() < 0.1;

    if tile0_ok && tile1_ok && both_ok {
        println!("  ✓ Both K-tiles in pair load correctly");
    } else {
        if !tile0_ok {
            println!("  ❌ K-tile 0 has ratio {:.3}x", r1);
        }
        if !tile1_ok {
            println!("  ❌ K-tile 1 has ratio {:.3}x", r2);
        }
        if !both_ok {
            println!("  ❌ Combined has ratio {:.3}x", r3);
        }

        if r1.abs() < 0.1 && r2.abs() < 0.1 {
            println!("  → Neither K-tile is read (completely broken)");
        } else if (r1 - 4.0).abs() < 0.5 && r2.abs() < 0.1 {
            println!("  → Only K-tile 0 is read, 4x (read 4 times, tile 1 ignored)");
        } else if r1.abs() < 0.1 && (r2 - 4.0).abs() < 0.5 {
            println!("  → Only K-tile 1 is read, 4x");
        }
    }

    Ok(())
}

/// Diagnostic 24: Global-to-shared B loading test
/// Uses a pattern where each K-row has a unique signature value.
/// Tests whether all K-rows are loaded into shared memory.
#[cfg(feature = "cuda")]
pub fn run_global_to_shared_diagnostic(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 24] Global→Shared B loading test (batch={})...",
        batch
    );

    let nrows = 256;
    let ncols = 256; // 16 K-rows (each is 16 elements)

    // Weights: K-row R has value (R+1) for all elements
    // K-row 0 (K=0-15) → 1.0
    // K-row 1 (K=16-31) → 2.0
    // ...
    // K-row 15 (K=240-255) → 16.0
    let mut weight_data = vec![0.0f32; nrows * ncols];
    for n in 0..nrows {
        for k in 0..ncols {
            let k_row = k / 16;
            weight_data[n * ncols + k] = (k_row + 1) as f32;
        }
    }
    let weights = Tensor::from_vec(weight_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Input: all 1s
    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    // Expected: sum(1*16 + 2*16 + ... + 16*16) = 16 * (1+2+...+16) = 16 * 136 = 2176
    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    let exp = baseline_vec[0];
    let act = gemx_vec[0];

    println!("  Weights: K-row R has value (R+1)");
    println!("  Expected (all K-rows): {:.2}", exp);
    println!("  Actual: {:.2}", act);
    println!("  Ratio: {:.3}x", act / exp);

    // Test each K-row individually
    println!("\n  Per K-row test:");
    let mut contributing_rows = Vec::new();
    let mut zero_rows = Vec::new();
    let mut wrong_rows = Vec::new();

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
        let qm = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let ml = qm.forward(&inp)?;

        let bl_v: Vec<f32> = bl.flatten_all()?.to_vec1()?;
        let ml_v: Vec<f32> = ml.flatten_all()?.to_vec1()?;

        let ratio = if bl_v[0].abs() > 0.01 {
            ml_v[0] / bl_v[0]
        } else {
            0.0
        };

        if (ratio - 1.0).abs() < 0.1 {
            contributing_rows.push(k_row);
        } else if ml_v[0].abs() < 0.1 {
            zero_rows.push(k_row);
        } else {
            wrong_rows.push((k_row, ratio));
        }
    }

    println!(
        "    Contributing correctly (ratio≈1): {:?}",
        contributing_rows
    );
    if !zero_rows.is_empty() {
        println!("    ❌ Zero output (not loaded): K-rows {:?}", zero_rows);
    }
    if !wrong_rows.is_empty() {
        println!("    ❌ Wrong ratio: {:?}", wrong_rows);
    }

    if zero_rows.is_empty() && wrong_rows.is_empty() {
        println!("  ✓ All 16 K-rows load correctly");
    }

    Ok(())
}

/// Diagnostic 25: MMA fragment test
/// Tests the tensor core MMA operation with known patterns.
/// Creates input/weight patterns that produce predictable MMA results.
#[cfg(feature = "cuda")]
pub fn run_mma_fragment_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 25] MMA fragment test...");

    let nrows = 16; // Minimal N for one MMA tile
    let ncols = 32; // One 32-element K-tile (2 physical 16-element tiles)
    let batch = 16; // One M-tile

    // Simple pattern: all 1s
    let weights = Tensor::full(1.0f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    let input_data = vec![1.0f32; batch * ncols];
    let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

    // Expected: each output element = 32 (sum of 32 ones * 1)
    let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let gemx = qmatmul.forward(&input)?;

    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx.flatten_all()?.to_vec1()?;

    println!(
        "  Config: N={}, K={}, M={} (minimal MMA tile)",
        nrows, ncols, batch
    );
    println!("  Weights: all 1.0");
    println!("  Input: all 1.0");

    // Check uniformity of output
    let mut max_diff = 0.0f32;
    let mut max_ratio = 1.0f32;
    for i in 0..batch * nrows {
        let exp = baseline_vec[i];
        let act = gemx_vec[i];
        let diff = (exp - act).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if exp.abs() > 0.01 {
            let r = act / exp;
            if (r - 1.0).abs() > (max_ratio - 1.0).abs() {
                max_ratio = r;
            }
        }
    }

    println!(
        "  Max diff across {} outputs: {:.4}",
        batch * nrows,
        max_diff
    );
    println!("  Worst ratio: {:.4}x", max_ratio);

    // Show first 4 outputs for each batch
    println!("\n  Output sample (first 4 outputs, batches 0,8,15):");
    for b in [0, 8, 15] {
        print!("    Batch {:2}: ", b);
        for n in 0..4 {
            let idx = b * nrows + n;
            let exp = baseline_vec[idx];
            let act = gemx_vec[idx];
            print!("({:.2},{:.2}) ", exp, act);
        }
        println!();
    }

    if max_diff < 1.0 {
        println!("  ✓ MMA produces correct uniform output");
    } else {
        println!("  ❌ MMA output has errors");
    }

    Ok(())
}
