//! Diagnostics that find which K positions a quantised matmul maps wrongly when
//! a regular test fails.
//!
//! Each runs a constructed matrix whose mismatches reveal the mapping: an
//! indicator matrix where each row picks out one K position, single hot inputs
//! for row isolation, a sweep over batch sizes, a check of the sub-block and
//! Y-stride layout, and a check that each scale covers its own region. They
//! sample a few positions to stay fast, and run when the tolerance is exceeded.

use super::*;

// DIAGNOSTIC TESTS FOR MATMUL MAPPING ERRORS
//
// When regular matmul tests fail tolerance, these diagnostic tests use specially
// constructed matrices to identify exactly WHERE the mapping is wrong:
//
// 1. Column Indicator Test: Each column has a unique non-zero element
//    - Input vector has 1.0 at each K position
//    - If output[i] == expected[i], element i is mapped correctly
//    - Any wrong element reveals which K positions are mis-mapped
//
// 2. Position Encoding Test: W[i,j] = encode(i,j)
//    - Output encodes which (row,col) pairs contributed
//    - Helps trace exactly which elements are being multiplied
//
// These tests only run on failure to avoid slowing down passing tests.

/// Run diagnostic tests to identify matmul element mapping errors.
/// Called automatically when tolerance is exceeded.
/// NOTE: Diagnostics are kept fast - only essential checks.
#[cfg(feature = "cuda")]
pub fn run_matmul_diagnostics(config: &QuantTestConfig, device: &Device) -> Result<()> {
    println!("  Running diagnostics for {}...", config.name);

    // Use minimum K-quant dimensions (256x256 = 1 quant block)
    let diag_nrows = 256;
    let diag_ncols = 256;

    run_indicator_diagnostic(config.dtype, diag_nrows, diag_ncols, device)?;
    run_row_isolation_diagnostic(config.dtype, diag_nrows, diag_ncols, device)?;
    run_batch_sensitivity_diagnostic(config.dtype, diag_nrows, diag_ncols, device)?;

    Ok(())
}

/// Diagnostic 1: Indicator matrix test
/// Creates weights where each row has a single non-zero element at a unique K position.
/// Input is all 1.0s, so output[row] should equal the value at that indicator position.
/// Any mismatch reveals which K positions are being accessed incorrectly.
#[cfg(feature = "cuda")]
pub(super) fn run_indicator_diagnostic(
    ggml_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<()> {
    // Create indicator weights: W[i, i % ncols] = 0.25 (scaled to fit quant range)
    // All other elements are 0.0
    // When multiplied by all-1.0 input, output[i] should be ~0.25
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for row in 0..nrows {
        let col = row % ncols;
        weights_data[row * ncols + col] = 0.25; // Use 0.25 to stay in good quant range
    }
    let weights = Tensor::from_vec(weights_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Input: all 1.0s
    let input = Tensor::ones((1, 1, ncols), DType::F32, device)?;

    // Baseline (GGML dequant + matmul)
    let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;

    // GEMX kernel
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;
    let gemx_result = qmatmul.forward(&input)?;

    // Compare element by element
    let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
    let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

    let mut mismatches = Vec::new();
    for (i, (b, m)) in baseline_vec.iter().zip(gemx_vec.iter()).enumerate() {
        let diff = (b - m).abs();
        if diff > 0.05 {
            // Generous tolerance for this diagnostic
            mismatches.push((i, *b, *m, diff));
        }
    }

    if mismatches.is_empty() {
        println!("  ✓ All {} indicator elements mapped correctly", nrows);
    } else {
        println!("  ❌ {} element mapping errors detected:", mismatches.len());
        // Show first 20 mismatches
        for (i, (idx, baseline, gemx, diff)) in mismatches.iter().take(20).enumerate() {
            let expected_k = idx % ncols;
            println!(
                "     [{:2}] Row {:3}: expected K={:3}, baseline={:.4}, gemx={:.4}, diff={:.4}",
                i, idx, expected_k, baseline, gemx, diff
            );
        }
        if mismatches.len() > 20 {
            println!("     ... and {} more", mismatches.len() - 20);
        }

        // Pattern analysis: check if mismatches follow a pattern
        analyze_mismatch_pattern(&mismatches, ncols);
    }

    Ok(())
}

/// Analyze mismatch pattern to identify systematic mapping errors
#[cfg(feature = "cuda")]
pub(super) fn analyze_mismatch_pattern(mismatches: &[(usize, f32, f32, f32)], ncols: usize) {
    if mismatches.len() < 4 {
        return;
    }

    println!("\n  Pattern Analysis:");

    // Check if mismatches are periodic
    let indices: Vec<usize> = mismatches.iter().map(|(i, _, _, _)| *i).collect();

    // Check for stride patterns
    if indices.len() >= 2 {
        let mut strides: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for window in indices.windows(2) {
            let stride = window[1].saturating_sub(window[0]);
            *strides.entry(stride).or_insert(0) += 1;
        }

        let mut stride_counts: Vec<_> = strides.into_iter().collect();
        stride_counts.sort_by(|a, b| b.1.cmp(&a.1));

        println!("  - Most common strides between mismatches:");
        for (stride, count) in stride_counts.iter().take(5) {
            let pct = 100.0 * *count as f32 / (indices.len() - 1) as f32;
            println!("    stride={}: {} occurrences ({:.1}%)", stride, count, pct);
        }
    }

    // Check which K positions (columns) are affected
    let k_positions: Vec<usize> = mismatches.iter().map(|(i, _, _, _)| i % ncols).collect();
    let unique_k: std::collections::HashSet<_> = k_positions.iter().collect();
    println!(
        "  - Unique K positions affected: {} out of {}",
        unique_k.len(),
        ncols
    );

    // Check for block-aligned issues (256-element K-quant blocks)
    let in_first_block = mismatches
        .iter()
        .filter(|(i, _, _, _)| (i % ncols) < 256)
        .count();
    let in_other_blocks = mismatches.len() - in_first_block;
    println!(
        "  - Mismatches in K[0..256]: {}, in K[256+]: {}",
        in_first_block, in_other_blocks
    );
}

/// Diagnostic 2: Row isolation test (fast version)
/// Tests specific K positions with single-element hot vectors.
/// Only tests a sampling of key positions to stay fast.
#[cfg(feature = "cuda")]
pub(super) fn run_row_isolation_diagnostic(
    ggml_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<()> {
    // Create uniform weights for simplicity
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test key K positions across the 256-element block structure
    // These hit important boundaries: start, sub-block edges, middle, end
    let test_k_positions = [0, 16, 32, 64, 128, 255];

    let mut k_errors: Vec<(usize, f32)> = Vec::new();

    for &k_pos in &test_k_positions {
        if k_pos >= ncols {
            continue;
        }

        // Create input with single 1.0 at position k_pos
        let mut input_data = vec![0.0f32; ncols];
        input_data[k_pos] = 1.0;
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        // Baseline
        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;

        // GEMX
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let max_diff: f32 = baseline_vec
            .iter()
            .zip(gemx_vec.iter())
            .map(|(b, m)| (b - m).abs())
            .fold(0.0, f32::max);

        if max_diff > 0.05 {
            k_errors.push((k_pos, max_diff));
        }
    }

    if k_errors.is_empty() {
        println!(
            "  ✓ All tested K positions ({:?}) map correctly",
            test_k_positions
        );
    } else {
        println!("  ❌ K position errors found:");
        for (k_pos, diff) in &k_errors {
            let sub_block = k_pos / 32;
            let elem_in_sub = k_pos % 32;
            println!(
                "     K={:3} (sub={}, elem={:2}): max_diff={:.4}",
                k_pos, sub_block, elem_in_sub, diff
            );
        }
    }

    Ok(())
}

/// Diagnostic 3: Batch-size sensitivity test (fast version)
/// Tests batch=1, 8, 64, 128 to find degradation patterns.
#[cfg(feature = "cuda")]
pub(super) fn run_batch_sensitivity_diagnostic(
    ggml_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<()> {
    let weights = create_test_weights(nrows, ncols, device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test key batch sizes - covers boundaries
    let batch_sizes = [1, 8, 64, 128];
    let mut results: Vec<(usize, f32)> = Vec::new();

    for &batch in &batch_sizes {
        let input = create_test_input(batch, 1, ncols, DType::F32, device)?;
        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let max_diff: f32 = baseline_vec
            .iter()
            .zip(gemx_vec.iter())
            .map(|(b, m)| (b - m).abs())
            .fold(0.0, f32::max);

        results.push((batch, max_diff));
        println!("    batch={:>3}: max_diff={:.6}", batch, max_diff);
    }

    // Analyze for discontinuities
    let base_max = results[0].1;
    let mut discontinuity: Option<usize> = None;

    for (batch, max_diff) in &results[1..] {
        if *max_diff > base_max * 1.5 && discontinuity.is_none() {
            discontinuity = Some(*batch);
        }
    }

    if let Some(batch) = discontinuity {
        println!(
            "  ⚠️  Discontinuity at batch={}: error jumps vs batch=1",
            batch
        );
        println!("     Likely causes: BATCH_TILE handling, shared memory, or thread sync issues");
    } else {
        println!("  ✓ No significant batch-size discontinuity");
    }

    Ok(())
}

// ADVANCED DIAGNOSTIC TESTS - Precision Flow Tracing
// These tests use specially crafted matrices to trace exactly where calculations
// flow, helping identify bugs in scale indexing, Y stride, element ordering, etc.

/// Diagnostic 4b: Subblock isolation test with configurable batch size
/// Same as Diagnostic 4 but runs at specified batch size to test GEMM path
#[cfg(feature = "cuda")]
pub fn run_subblock_isolation_diagnostic_batched(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    let (tile_size, num_tiles, tile_name) = match ggml_dtype {
        GgmlDType::Q5_K => (16, 16, "K-tile"),
        _ => (32, 8, "subblock"),
    };

    println!(
        "\n[Diagnostic 4b] {} isolation test (batch={}, {} {}s of {} elements)...",
        tile_name, batch, num_tiles, tile_name, tile_size
    );

    let nrows = 256;
    let ncols = 256;

    let weights = Tensor::full(0.125f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    let mut subblock_errors: Vec<(usize, f32, f32, f32)> = Vec::new();

    for subblock in 0..num_tiles {
        let start = subblock * tile_size;
        let end = start + tile_size;

        // Create batched input with 1.0 only in this subblock
        let mut input_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            for k in start..end {
                input_data[b * ncols + k] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (batch, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, batch, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        // Only check first batch element
        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let baseline_val = baseline_vec[0];
        let gemx_val = gemx_vec[0];
        let max_diff = baseline_vec[..nrows]
            .iter()
            .zip(gemx_vec[..nrows].iter())
            .map(|(b, m)| (b - m).abs())
            .fold(0.0f32, f32::max);

        if max_diff > 0.1 {
            subblock_errors.push((subblock, baseline_val, gemx_val, max_diff));
        }
    }

    if subblock_errors.is_empty() {
        println!(
            "  ✓ All {} {}s produce correct isolated results (batch={})",
            num_tiles, tile_name, batch
        );
    } else {
        println!(
            "  ❌ {} isolation errors found (batch={}):",
            tile_name, batch
        );
        for (sub, baseline, gemx, diff) in &subblock_errors {
            println!(
                "     {} {}: baseline={:.4}, gemx={:.4}, diff={:.4}",
                tile_name, sub, baseline, gemx, diff
            );
            let expected_scale_idx = sub / (tile_size / 16).max(1);
            println!(
                "       Expected scale index: {} (K range [{}-{}])",
                expected_scale_idx,
                sub * tile_size,
                sub * tile_size + tile_size - 1
            );
        }
    }

    Ok(())
}

/// Diagnostic 5b: Y stride verification test with configurable batch size
#[cfg(feature = "cuda")]
pub fn run_y_stride_diagnostic_batched(
    ggml_dtype: GgmlDType,
    device: &Device,
    batch: usize,
) -> Result<()> {
    println!(
        "\n[Diagnostic 5b] Y stride verification (batch={}, tile boundary test)...",
        batch
    );

    let nrows = 256;
    let ncols = 512;

    let mut weights_data = vec![0.0f32; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            weights_data[i * ncols + j] = (j as f32) / 1000.0;
        }
    }
    let weights = Tensor::from_vec(weights_data.clone(), (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Check tile boundaries (K/64 uses 16-element tiles)
    let tile_size = match ggml_dtype {
        GgmlDType::Q2_K | GgmlDType::Q3_K | GgmlDType::Q4_K | GgmlDType::Q5_K | GgmlDType::Q6_K => {
            16
        }
        _ => 32,
    };
    let mut tile_boundaries = Vec::new();
    for base in (0..ncols).step_by(256) {
        for t in (0..256).step_by(tile_size) {
            let start = base + t;
            let end = start + tile_size - 1;
            if start < ncols {
                tile_boundaries.push(start);
            }
            if end < ncols {
                tile_boundaries.push(end);
            }
        }
    }
    tile_boundaries.sort_unstable();
    tile_boundaries.dedup();
    println!("  Testing Y access at tile boundaries:");

    let mut boundary_errors = 0;
    for &boundary in &tile_boundaries {
        if boundary >= ncols {
            continue;
        }

        // Create batched hot vector at boundary position
        let mut hot_data = vec![0.0f32; batch * ncols];
        for b in 0..batch {
            hot_data[b * ncols + boundary] = 1.0;
        }
        let hot_input = Tensor::from_vec(hot_data, (batch, 1, ncols), device)?;

        let baseline_hot = compute_baseline(&qtensor, &hot_input, batch, 1, nrows, ncols, device)?;
        let qmatmul2 = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_hot = qmatmul2.forward(&hot_input)?;

        // Only check first batch element
        let b_vec: Vec<f32> = baseline_hot.flatten_all()?.to_vec1()?;
        let m_vec: Vec<f32> = gemx_hot.flatten_all()?.to_vec1()?;

        let max_diff = b_vec[..nrows]
            .iter()
            .zip(m_vec[..nrows].iter())
            .map(|(b, m)| (b - m).abs())
            .fold(0.0f32, f32::max);

        let status = if max_diff < 0.1 { "OK" } else { "FAIL" };
        if max_diff >= 0.1 {
            boundary_errors += 1;
            println!(
                "    K={:3}: max_diff={:.4} [{}]",
                boundary, max_diff, status
            );
        }
    }

    if boundary_errors == 0 {
        println!(
            "  ✓ All tile boundary positions read correctly (batch={})",
            batch
        );
    } else {
        println!(
            "  ❌ {} tile boundary errors (batch={}) - Y stride may be incorrect",
            boundary_errors, batch
        );
    }

    Ok(())
}

/// Diagnostic 6: Scale-element correspondence test
/// Creates weights where different subblocks have very different values,
/// then checks if the correct scale is applied to each element range.
#[cfg(feature = "cuda")]
pub fn run_scale_correspondence_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 6] Scale-element correspondence test...");

    let nrows = 256;
    let ncols = 256;

    // Both Q4_K and Q5_K have 8 scales per 256-element superblock (32 elements per scale)
    // Q5_K: 8 scales, each covers 2 K-tiles (32 elements)
    // Q4_K: 8 scales, each covers 1 subblock (32 elements)
    let num_scales = 8;
    let elements_per_scale = 32;

    // Create weights where each scale region has a distinct value:
    // Scale 0 (K=0-31): 0.1
    // Scale 1 (K=32-63): 0.2
    // ... etc up to scale 7: 0.8
    // This creates distinct scales per region during quantization
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            let scale_idx = j / elements_per_scale;
            weights_data[i * ncols + j] = (scale_idx as f32 + 1.0) * 0.1;
        }
    }
    let weights = Tensor::from_vec(weights_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Test: hot vector at start of each scale region
    println!("  Testing scale application at scale region starts:");
    let mut scale_errors: Vec<(usize, f32, f32)> = Vec::new();

    for scale_idx in 0..num_scales {
        let k_pos = scale_idx * elements_per_scale;

        let mut hot_data = vec![0.0f32; ncols];
        hot_data[k_pos] = 1.0;
        let hot_input = Tensor::from_vec(hot_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &hot_input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&hot_input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        // First output element should be close to (scale_idx+1)*0.1 (the weight value)
        let expected_approx = (scale_idx as f32 + 1.0) * 0.1;
        let baseline_val = baseline_vec[0];
        let gemx_val = gemx_vec[0];
        let diff = (baseline_val - gemx_val).abs();

        if diff > 0.05 {
            scale_errors.push((scale_idx, baseline_val, gemx_val));
            println!("    Scale {} (K={}): expected~{:.2}, baseline={:.4}, gemx={:.4}, diff={:.4} [MISMATCH]",
                     scale_idx, k_pos, expected_approx, baseline_val, gemx_val, diff);
        }
    }

    if scale_errors.is_empty() {
        println!("  ✓ All {} scales applied correctly", num_scales);
    } else {
        println!(
            "  ❌ {} scales have wrong values applied",
            scale_errors.len()
        );

        // Analyze pattern - are consecutive scales swapped?
        if scale_errors.len() >= 2 {
            println!("  Pattern analysis:");
            for (sub, _baseline, gemx) in &scale_errors {
                // Check if gemx value matches a different scale's expected value
                for check_sub in 0..num_scales {
                    let check_expected = (check_sub as f32 + 1.0) * 0.1;
                    if (gemx - check_expected).abs() < 0.05 {
                        println!("    Scale {} appears to use value from scale {} (gemx={:.3}, expected for {}={:.3})",
                                 sub, check_sub, gemx, check_sub, check_expected);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Diagnostic 7: Part-by-part accumulation test
/// Q4_K uses 4 parts (load_part_0..3). This test checks if each part's
/// contribution is correct by isolating specific K ranges.
#[cfg(feature = "cuda")]
pub fn run_part_accumulation_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    // Q5_K uses 1-part interface (16 K-tiles processed by 16 threads)
    // Q4_K uses 2-part interface (split into even/odd processing)
    let (_num_parts, part_name) = match ggml_dtype {
        GgmlDType::Q5_K => (1, "Q5_K 1-part (16 K-tiles)"),
        _ => (4, "Q4_K 4-part"),
    };

    println!("\n[Diagnostic 7] Part accumulation test ({})...", part_name);

    let nrows = 256;
    let ncols = 256;

    // Create uniform weights
    let weights = Tensor::full(0.125f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // For Q5_K: Each of 16 threads handles one K-tile (16 elements)
    // Test K-tiles 0, 4, 8, 12 (spread across the superblock)
    // For Q4_K: part structure as before
    let part_ranges: Vec<(usize, String, Vec<usize>)> = if ggml_dtype == GgmlDType::Q5_K {
        vec![
            (0, "K-tile 0 (elems 0-15)".to_string(), (0..16).collect()),
            (1, "K-tile 4 (elems 64-79)".to_string(), (64..80).collect()),
            (
                2,
                "K-tile 8 (elems 128-143)".to_string(),
                (128..144).collect(),
            ),
            (
                3,
                "K-tile 12 (elems 192-207)".to_string(),
                (192..208).collect(),
            ),
        ]
    } else {
        vec![
            (
                0,
                "Part 0 (even, first weights)".to_string(),
                vec![0, 16, 32, 48],
            ),
            (
                1,
                "Part 1 (odd, first weights)".to_string(),
                vec![128, 144, 160, 176],
            ),
            (
                2,
                "Part 2 (even, second weights)".to_string(),
                vec![64, 80, 96, 112],
            ),
            (
                3,
                "Part 3 (odd, second weights)".to_string(),
                vec![192, 208, 224, 240],
            ),
        ]
    };

    println!("  Testing accumulation for each loader part/tile:");
    let mut part_errors = 0;

    for (_part_idx, name, positions) in part_ranges.iter() {
        // Create Y with 1.0 only at positions for this part
        let mut input_data = vec![0.0f32; ncols];
        for &pos in positions {
            if pos < ncols {
                input_data[pos] = 1.0;
            }
        }
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let gemx_result = qmatmul.forward(&input)?;

        let baseline_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let gemx_vec: Vec<f32> = gemx_result.flatten_all()?.to_vec1()?;

        let max_diff = baseline_vec
            .iter()
            .zip(gemx_vec.iter())
            .map(|(b, m)| (b - m).abs())
            .fold(0.0f32, f32::max);

        let status = if max_diff < 0.1 { "OK" } else { "FAIL" };
        if max_diff >= 0.1 {
            part_errors += 1;
        }
        println!("    {}: max_diff={:.4} [{}]", name, max_diff, status);
    }

    if part_errors == 0 {
        println!("  ✓ All 4 parts accumulate correctly");
    } else {
        println!("  ❌ {} parts have accumulation errors", part_errors);
    }

    Ok(())
}
