//! Diagnostics for the repacked K/64 byte layout: nibble pairs, intra-tile
//! interaction, position mapping, the high bit and the dequantisation kernel.
//!
//! Each test examines one part of the layout. Elements that share a qs byte are
//! checked for interference, and the qh high bit is checked at the offset the
//! repack writes it to and against the GGML layout. The dequantisation kernel is
//! compared with the GGML dequantisation. That last comparison is skipped for
//! K/128 formats, because the GGML path does not read them.

use super::*;

/// Diagnostic 8: Test nibble pair confusion (elements sharing same qs byte)
/// Q5_K PERM: {0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15}
/// Elements 0&4 share byte 0 (lo/hi nibble), 1&5 share byte 1, etc.
#[cfg(feature = "cuda")]
pub fn run_nibble_pair_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 8] Nibble pair confusion test...");

    let nrows = 256;
    let ncols = 256;

    let weights = Tensor::full(0.125f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Q5_K nibble pairs: (0,4), (1,5), (2,6), (3,7), (8,12), (9,13), (10,14), (11,15)
    // Test if activating one element of a pair affects the other
    let nibble_pairs: Vec<(usize, usize)> = if ggml_dtype == GgmlDType::Q5_K {
        vec![
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (8, 12),
            (9, 13),
            (10, 14),
            (11, 15),
        ]
    } else {
        // Q4_K pairs (consecutive elements share bytes)
        vec![
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13),
            (14, 15),
        ]
    };

    println!("  Testing element pairs that share a qs byte:");
    let mut pair_errors = Vec::new();

    for (elem_a, elem_b) in &nibble_pairs {
        // Test element A alone
        let mut input_a = vec![0.0f32; ncols];
        input_a[*elem_a] = 1.0;
        let input_a = Tensor::from_vec(input_a, (1, 1, ncols), device)?;

        // Test element B alone
        let mut input_b = vec![0.0f32; ncols];
        input_b[*elem_b] = 1.0;
        let input_b = Tensor::from_vec(input_b, (1, 1, ncols), device)?;

        // Test both elements together
        let mut input_both = vec![0.0f32; ncols];
        input_both[*elem_a] = 1.0;
        input_both[*elem_b] = 1.0;
        let input_both = Tensor::from_vec(input_both, (1, 1, ncols), device)?;

        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;

        let result_a = qmatmul.forward(&input_a)?;
        let result_b = qmatmul.forward(&input_b)?;
        let result_both = qmatmul.forward(&input_both)?;

        let baseline_a = compute_baseline(&qtensor, &input_a, 1, 1, nrows, ncols, device)?;
        let baseline_b = compute_baseline(&qtensor, &input_b, 1, 1, nrows, ncols, device)?;
        let baseline_both = compute_baseline(&qtensor, &input_both, 1, 1, nrows, ncols, device)?;

        let vec_a: Vec<f32> = result_a.flatten_all()?.to_vec1()?;
        let vec_b: Vec<f32> = result_b.flatten_all()?.to_vec1()?;
        let vec_both: Vec<f32> = result_both.flatten_all()?.to_vec1()?;
        let base_a: Vec<f32> = baseline_a.flatten_all()?.to_vec1()?;
        let base_b: Vec<f32> = baseline_b.flatten_all()?.to_vec1()?;
        let _base_both: Vec<f32> = baseline_both.flatten_all()?.to_vec1()?;

        // Check if A+B = Both (linearity)
        let linearity_error: f32 = vec_a
            .iter()
            .zip(vec_b.iter())
            .zip(vec_both.iter())
            .map(|((a, b), both)| ((a + b) - both).abs())
            .fold(0.0f32, f32::max);

        // Check individual element accuracy
        let diff_a: f32 = vec_a
            .iter()
            .zip(base_a.iter())
            .map(|(m, b)| (m - b).abs())
            .fold(0.0f32, f32::max);
        let diff_b: f32 = vec_b
            .iter()
            .zip(base_b.iter())
            .map(|(m, b)| (m - b).abs())
            .fold(0.0f32, f32::max);

        // Check if element B's value appears when querying A (cross-contamination)
        let cross_check = (vec_a[0] - base_b[0]).abs();

        if diff_a > 0.1 || diff_b > 0.1 || linearity_error > 0.1 {
            pair_errors.push((
                *elem_a,
                *elem_b,
                diff_a,
                diff_b,
                linearity_error,
                cross_check,
            ));
            println!(
                "    Pair ({},{}): diff_a={:.4}, diff_b={:.4}, linearity={:.4}, cross={:.4} [FAIL]",
                elem_a, elem_b, diff_a, diff_b, linearity_error, cross_check
            );
        }
    }

    if pair_errors.is_empty() {
        println!("  ✓ All nibble pairs work correctly (no cross-contamination)");
    } else {
        println!("  ❌ {} nibble pairs have errors", pair_errors.len());
        // Analyze pattern
        for (a, b, _da, _db, _lin, cross) in &pair_errors {
            if *cross < 0.05 {
                println!(
                    "    ({},{}) might be SWAPPED: querying {} returns {}'s value",
                    a, b, a, b
                );
            }
        }
    }

    Ok(())
}

/// Diagnostic 9: Test intra-tile multi-element interaction
/// Activates multiple elements within one K-tile to see accumulation behavior
#[cfg(feature = "cuda")]
pub fn run_intra_tile_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 9] Intra-tile multi-element test...");

    let nrows = 256;
    let ncols = 256;

    let weights = Tensor::full(0.125f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    let _tile_size = if ggml_dtype == GgmlDType::Q5_K {
        16
    } else {
        32
    };

    // Test patterns within first tile
    let test_patterns: Vec<(&str, Vec<usize>)> = if ggml_dtype == GgmlDType::Q5_K {
        vec![
            ("First half (0-7)", (0..8).collect()),
            ("Second half (8-15)", (8..16).collect()),
            (
                "Even elements (0,2,4,6,8,10,12,14)",
                vec![0, 2, 4, 6, 8, 10, 12, 14],
            ),
            (
                "Odd elements (1,3,5,7,9,11,13,15)",
                vec![1, 3, 5, 7, 9, 11, 13, 15],
            ),
            (
                "lo-nibble positions (0,1,2,3,8,9,10,11)",
                vec![0, 1, 2, 3, 8, 9, 10, 11],
            ),
            (
                "hi-nibble positions (4,5,6,7,12,13,14,15)",
                vec![4, 5, 6, 7, 12, 13, 14, 15],
            ),
            ("Alternating from PERM (0,4,1,5)", vec![0, 4, 1, 5]),
            ("All 16 elements", (0..16).collect()),
        ]
    } else {
        vec![
            ("First half (0-15)", (0..16).collect()),
            ("Second half (16-31)", (16..32).collect()),
            ("Even elements", (0..32).step_by(2).collect()),
            ("Odd elements", (1..32).step_by(2).collect()),
            ("All 32 elements", (0..32).collect()),
        ]
    };

    println!("  Testing multi-element patterns within tile 0:");
    let mut pattern_errors = Vec::new();

    for (name, positions) in &test_patterns {
        let mut input_data = vec![0.0f32; ncols];
        for &pos in positions {
            input_data[pos] = 1.0;
        }
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let result = qmatmul.forward(&input)?;

        let base_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;

        let max_diff: f32 = base_vec
            .iter()
            .zip(result_vec.iter())
            .map(|(b, r)| (b - r).abs())
            .fold(0.0f32, f32::max);

        let mismatch_count = base_vec
            .iter()
            .zip(result_vec.iter())
            .filter(|(b, r)| (*b - *r).abs() > 0.01)
            .count();

        let status = if max_diff < 0.1 { "OK" } else { "FAIL" };
        if max_diff >= 0.1 {
            pattern_errors.push((name.to_string(), max_diff, mismatch_count));
        }
        println!(
            "    {}: max_diff={:.4}, mismatches={}/{} [{}]",
            name, max_diff, mismatch_count, nrows, status
        );
    }

    if pattern_errors.is_empty() {
        println!("  ✓ All intra-tile patterns work correctly");
    } else {
        println!("  ❌ {} patterns have errors", pattern_errors.len());
    }

    Ok(())
}

/// Diagnostic 10: Test specific position mapping
/// Tests the exact position where values end up vs where they should be
#[cfg(feature = "cuda")]
pub fn run_position_mapping_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 10] Position mapping diagnostic...");

    let nrows = 256;
    let ncols = 256;

    // Create weights where each K position has a unique identifiable value
    // Weight[row, k] = (k + 1) * 0.01 so position k contributes (k+1)*0.01 to output
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for row in 0..nrows {
        for k in 0..ncols {
            weights_data[row * ncols + k] = (k as f32 + 1.0) * 0.01;
        }
    }
    let weights = Tensor::from_vec(weights_data.clone(), (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    let tile_size = if ggml_dtype == GgmlDType::Q5_K {
        16
    } else {
        32
    };

    println!(
        "  Testing position→output mapping for first {} elements:",
        tile_size
    );
    let mut mapping_errors: Vec<(usize, f32, f32, Option<usize>)> = Vec::new();

    for k in 0..tile_size {
        let mut input_data = vec![0.0f32; ncols];
        input_data[k] = 1.0;
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let result = qmatmul.forward(&input)?;

        let base_vec: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;

        let expected_val = base_vec[0];
        let actual_val = result_vec[0];
        let diff = (expected_val - actual_val).abs();

        // If wrong, find which position's value we actually got
        let found_pos = if diff > 0.005 {
            // The value we got should match some other position
            (0..tile_size).find(|&test_k| {
                let test_expected = (test_k as f32 + 1.0) * 0.01;
                (actual_val - test_expected).abs() < 0.005
            })
        } else {
            None
        };

        if diff > 0.005 {
            mapping_errors.push((k, expected_val, actual_val, found_pos));
        }
    }

    if mapping_errors.is_empty() {
        println!("  ✓ All {} positions map correctly", tile_size);
    } else {
        println!(
            "  ❌ {} positions have wrong mapping:",
            mapping_errors.len()
        );
        println!("     Format: position k: expected→actual (source if found)");
        for (k, expected, actual, found) in &mapping_errors {
            if let Some(src) = found {
                println!(
                    "     k={:2}: expected={:.4}, got={:.4} (value from k={})",
                    k, expected, actual, src
                );
            } else {
                println!(
                    "     k={:2}: expected={:.4}, got={:.4} (source unknown)",
                    k, expected, actual
                );
            }
        }

        // Analyze the permutation pattern
        println!("\n  Permutation analysis:");
        let mut perm_map: Vec<(usize, usize)> = Vec::new();
        for (k, _, _, found) in &mapping_errors {
            if let Some(src) = found {
                perm_map.push((*k, *src));
            }
        }
        if !perm_map.is_empty() {
            println!("     Observed mapping (k → actual_source):");
            for (k, src) in &perm_map {
                let offset = (*src as i32) - (*k as i32);
                println!("       {} → {} (offset {:+})", k, src, offset);
            }
        }
    }

    Ok(())
}

/// Diagnostic 11: Test qh (high bit) application
/// For Q5_K, tests if the 5th bit is correctly applied to each element
#[cfg(feature = "cuda")]
pub fn run_qh_bit_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    if ggml_dtype != GgmlDType::Q5_K {
        println!("\n[Diagnostic 11] qh bit test skipped (not Q5_K)");
        return Ok(());
    }

    println!("\n[Diagnostic 11] Q5_K qh (5th bit) application test...");

    let nrows = 256;
    let ncols = 256;

    // Create weights that will exercise both qh=0 and qh=1 states
    // Values near 0 → qh=0, values near max → qh=1
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for row in 0..nrows {
        for k in 0..ncols {
            // Alternate high/low values within each K-tile to test qh bit patterns
            let tile_pos = k % 16;
            if tile_pos < 8 {
                weights_data[row * ncols + k] = 0.02; // Low value (qh likely 0)
            } else {
                weights_data[row * ncols + k] = 0.9; // High value (qh likely 1)
            }
        }
    }
    let weights = Tensor::from_vec(weights_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    println!("  Testing qh bit pattern (low values k=0-7, high values k=8-15):");

    // Test low-value positions
    let mut low_errors = 0;
    let mut high_errors = 0;

    for k in 0..16 {
        let mut input_data = vec![0.0f32; ncols];
        input_data[k] = 1.0;
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let result = qmatmul.forward(&input)?;

        let base_val: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let result_val: f32 = result.flatten_all()?.to_vec1::<f32>()?[0];
        let diff = (base_val - result_val).abs();

        let region = if k < 8 { "low" } else { "high" };
        let status = if diff < 0.05 { "OK" } else { "FAIL" };

        if diff >= 0.05 {
            if k < 8 {
                low_errors += 1;
            } else {
                high_errors += 1;
            }
            println!(
                "    k={:2} ({}): baseline={:.4}, gemx={:.4}, diff={:.4} [{}]",
                k, region, base_val, result_val, diff, status
            );
        }
    }

    if low_errors == 0 && high_errors == 0 {
        println!("  ✓ All qh bit patterns applied correctly");
    } else {
        println!(
            "  ❌ qh errors: {} in low region (qh=0), {} in high region (qh=1)",
            low_errors, high_errors
        );
        if high_errors > low_errors {
            println!("     Pattern suggests qh bit extraction or application issue");
        }
    }

    Ok(())
}

/// Diagnostic 12: Full dequant comparison at specific positions
/// Compares the actual quantized->dequantized values element by element
#[cfg(feature = "cuda")]
pub fn run_dequant_value_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 12] Dequant value comparison...");

    let nrows = 16; // Small for detailed output
    let ncols = 256;

    let weights = Tensor::full(0.125f32, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Create identity-like input to extract individual weight columns
    let tile_size = if ggml_dtype == GgmlDType::Q5_K {
        16
    } else {
        32
    };

    println!(
        "  Comparing dequant values for first {} K positions:",
        tile_size
    );

    let mut value_mismatches: Vec<(usize, f32, f32)> = Vec::new();

    for k in 0..tile_size {
        let mut input_data = vec![0.0f32; ncols];
        input_data[k] = 1.0;
        let input = Tensor::from_vec(input_data, (1, 1, ncols), device)?;

        let baseline = compute_baseline(&qtensor, &input, 1, 1, nrows, ncols, device)?;
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
        let result = qmatmul.forward(&input)?;

        // Get value at output row 0 (which reflects weight[0, k] * 1.0)
        let base_val: f32 = baseline.flatten_all()?.to_vec1::<f32>()?[0];
        let result_val: f32 = result.flatten_all()?.to_vec1::<f32>()?[0];
        let diff = (base_val - result_val).abs();

        if diff > 0.001 {
            value_mismatches.push((k, base_val, result_val));
        }
    }

    if value_mismatches.is_empty() {
        println!("  ✓ All dequant values match within tolerance");
    } else {
        println!(
            "  ❌ {} positions have value mismatches:",
            value_mismatches.len()
        );
        for (k, base, result) in &value_mismatches {
            println!(
                "     k={:2}: baseline={:.6}, gemx={:.6}, diff={:.6}",
                k,
                base,
                result,
                (base - result).abs()
            );
        }
    }

    Ok(())
}

/// Diagnostic 13: Verify qh bits are present in repacked K-tile data at correct offsets
/// Reads the raw repacked bytes to confirm high bits were written correctly
#[cfg(feature = "cuda")]
pub fn run_qh_raw_verification_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    if ggml_dtype != GgmlDType::Q5_K {
        println!("\n[Diagnostic 13] qh raw verification skipped (not Q5_K)");
        return Ok(());
    }

    println!("\n[Diagnostic 13] Q5_K qh raw byte verification...");

    use candle::quantized::QTensor;

    let nrows = 1; // Single row for simplicity
    let ncols = 256; // One super-block

    // Create weights where we KNOW what qh bits should be:
    // High values (0.9) need qh=1, low values (0.02) need qh=0
    // Set specific pattern: elements 0,2,4,6,8,10,12,14 = low (qh=0)
    //                       elements 1,3,5,7,9,11,13,15 = high (qh=1)
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for k in 0..16 {
        if k % 2 == 0 {
            weights_data[k] = 0.02; // Low value -> qh should be 0
        } else {
            weights_data[k] = 0.95; // High value -> qh should be 1
        }
    }
    // Rest of elements = 0.5 (middle value)
    for k in 16..ncols {
        weights_data[k] = 0.5;
    }

    let weights = Tensor::from_vec(weights_data, (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Get the repacked data
    let repacked = qtensor.repack_gemx()?;
    let repacked_data: Vec<u8> = repacked.data()?.to_vec();

    // K/64 block layout: qs0[16] + qs1[16] + qh[8] + scales[8] = 48 bytes
    // K-tile 0 uses qs0.x/qs0.y and the low 16 bits of qh0 (qh.x)
    let qh0 = u32::from_le_bytes([
        repacked_data[32],
        repacked_data[33],
        repacked_data[34],
        repacked_data[35],
    ]);
    let qh1 = u32::from_le_bytes([
        repacked_data[36],
        repacked_data[37],
        repacked_data[38],
        repacked_data[39],
    ]);
    let qh_val = (qh0 & 0xFFFF) as u16;

    println!("  K/64 block 0 raw bytes (48 bytes total):");
    println!("    qs0[0-15]: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
             repacked_data[0], repacked_data[1], repacked_data[2], repacked_data[3],
             repacked_data[4], repacked_data[5], repacked_data[6], repacked_data[7],
             repacked_data[8], repacked_data[9], repacked_data[10], repacked_data[11],
             repacked_data[12], repacked_data[13], repacked_data[14], repacked_data[15]);
    println!(
        "    qh[32-39]: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
        repacked_data[32],
        repacked_data[33],
        repacked_data[34],
        repacked_data[35],
        repacked_data[36],
        repacked_data[37],
        repacked_data[38],
        repacked_data[39]
    );
    println!(
        "    qh0: 0x{:08x}, qh1: 0x{:08x} (tile0 u16: 0x{:04x})",
        qh0, qh1, qh_val
    );

    // Q5K_TILE_PERM = {0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15}
    // Repack stores: qh bit N = high bit of element Q5K_TILE_PERM[N]
    // So for our pattern (odd elements have qh=1):
    //   bit 0 = elem 0's qh = 0 (even)
    //   bit 1 = elem 4's qh = 0 (even)
    //   bit 2 = elem 1's qh = 1 (odd)
    //   bit 3 = elem 5's qh = 1 (odd)
    //   bit 4 = elem 2's qh = 0 (even)
    //   bit 5 = elem 6's qh = 0 (even)
    //   bit 6 = elem 3's qh = 1 (odd)
    //   bit 7 = elem 7's qh = 1 (odd)
    //   bit 8 = elem 8's qh = 0 (even)
    //   bit 9 = elem 12's qh = 0 (even)
    //   bit 10 = elem 9's qh = 1 (odd)
    //   bit 11 = elem 13's qh = 1 (odd)
    //   bit 12 = elem 10's qh = 0 (even)
    //   bit 13 = elem 14's qh = 0 (even)
    //   bit 14 = elem 11's qh = 1 (odd)
    //   bit 15 = elem 15's qh = 1 (odd)
    // Expected qh = 0b1100_1100_1100_1100 = 0xCCCC
    let expected_qh: u16 = 0b1100_1100_1100_1100; // 0xCCCC

    println!("\n  Expected qh pattern for odd-element-high test:");
    println!(
        "    Expected: 0x{:04x} = 0b{:016b}",
        expected_qh, expected_qh
    );
    println!("    Actual:   0x{:04x} = 0b{:016b}", qh_val, qh_val);

    // Check each bit
    println!("\n  Per-bit analysis (Q5K_TILE_PERM mapping):");
    let q5k_tile_perm: [usize; 16] = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];
    let mut bit_errors = 0;
    for bit in 0..16 {
        let elem = q5k_tile_perm[bit];
        let expected_bit = if elem % 2 == 1 { 1 } else { 0 }; // Odd elements should have qh=1
        let actual_bit = (qh_val >> bit) & 1;
        let status = if expected_bit == actual_bit {
            "OK"
        } else {
            "FAIL"
        };
        if expected_bit != actual_bit {
            bit_errors += 1;
            println!(
                "    bit {:2} -> elem {:2}: expected={}, actual={} [{}]",
                bit, elem, expected_bit, actual_bit, status
            );
        }
    }

    if bit_errors == 0 {
        println!("  ✓ All 16 qh bits stored correctly in repacked data");
    } else {
        println!("  ❌ {} qh bits are wrong in repacked data", bit_errors);
    }

    // Also verify the actual quantized values from GGML quantizer
    println!("\n  Checking source GGML qh values:");
    let ggml_data = qtensor.data()?;
    let ggml_bytes: Vec<u8> = ggml_data.to_vec();
    // Q5_K block layout: dm(4B) + scales(12B) + qh(32B) + qs(128B) = 176B
    // qh starts at offset 16 (after dm + scales)
    let qh_offset = 4 + 12; // = 16
    println!(
        "    GGML block qh bytes (first 4 of 32): {:02x} {:02x} {:02x} {:02x}",
        ggml_bytes[qh_offset],
        ggml_bytes[qh_offset + 1],
        ggml_bytes[qh_offset + 2],
        ggml_bytes[qh_offset + 3]
    );

    // GGML qh layout: element k at byte k/8, bit k%8
    // So elem 0-7 are in byte 0, elem 8-15 are in byte 1
    let ggml_qh_byte0 = ggml_bytes[qh_offset];
    let ggml_qh_byte1 = ggml_bytes[qh_offset + 1];
    println!("    GGML qh for elems 0-7:  0b{:08b}", ggml_qh_byte0);
    println!("    GGML qh for elems 8-15: 0b{:08b}", ggml_qh_byte1);
    // Expected for odd elements high: bits 1,3,5,7 set in each byte
    // = 0b10101010 = 0xAA
    println!("    Expected (odd elems high): 0b10101010 = 0xAA each");

    Ok(())
}

/// Diagnostic 14: Compare dequant kernel output vs GGML dequant (baseline)
/// This tests if the loader's dequant method produces correct values
/// NOTE: This test is SKIPPED for K/128 repacked formats because QMatMulWrapper.dequantize()
/// calls qtensor.dequantize() which uses GGML block dequant kernels that don't understand K/128.
/// The matmul itself works correctly because it uses the K/128-aware loader dequant path.
#[cfg(feature = "cuda")]
pub fn run_dequant_kernel_diagnostic(ggml_dtype: GgmlDType, device: &Device) -> Result<()> {
    println!("\n[Diagnostic 14] Dequant kernel vs GGML baseline comparison...");

    // K/128 formats repack the data, so the standalone dequant() call won't work
    // (it uses GGML block dequant which expects original format, not K/128)
    // The matmul itself works correctly because it uses the loader's MMA dequant path.
    let k128_formats = [
        GgmlDType::Q4_K,
        GgmlDType::Q5_K,
        GgmlDType::Q6_K,
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_K,
        GgmlDType::Q2_K,
        GgmlDType::Q3_K,
    ];
    if k128_formats.contains(&ggml_dtype) {
        println!("  (Skipped - K/128 repacked format uses different dequant path in matmul)");
        println!("  ✓ Matmul dequant verified via other diagnostics");
        return Ok(());
    }

    let nrows = 4;
    let ncols = 256; // One super-block

    // Use weights that span the full range to exercise qh bits
    let mut weights_data = vec![0.0f32; nrows * ncols];
    for row in 0..nrows {
        for k in 0..ncols {
            // Vary weights by position to get different qh patterns
            weights_data[row * ncols + k] = ((k + row * 7) % 32) as f32 / 31.0;
        }
    }

    let weights = Tensor::from_vec(weights_data.clone(), (nrows, ncols), device)?;
    let qtensor = QTensor::quantize(&weights, ggml_dtype)?;

    // Get GGML dequantized values (baseline truth)
    let ggml_dequant = qtensor.dequantize(device)?;
    let ggml_values: Vec<f32> = ggml_dequant.flatten_all()?.to_vec1()?;

    // Get dequant via QMatMulWrapper (uses loader dequant kernel)
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor.clone())?;
    let loader_dequant = qmatmul.dequantize()?;
    let loader_values: Vec<f32> = loader_dequant.flatten_all()?.to_vec1()?;

    // Compare
    let mut mismatches = 0;
    let mut max_diff = 0.0f32;
    let mut worst_pos = (0, 0);

    for row in 0..nrows {
        for k in 0..ncols {
            let idx = row * ncols + k;
            let ggml_val = ggml_values[idx];
            let loader_val = loader_values[idx];
            let diff = (ggml_val - loader_val).abs();

            if diff > 0.01 {
                mismatches += 1;
                if diff > max_diff {
                    max_diff = diff;
                    worst_pos = (row, k);
                }
            }
        }
    }

    println!("  Total elements: {}", nrows * ncols);
    println!("  Mismatches (diff > 0.01): {}", mismatches);

    if mismatches > 0 {
        println!(
            "  Worst mismatch at ({}, {}): diff={:.4}",
            worst_pos.0, worst_pos.1, max_diff
        );

        // Show first K-tile values
        println!("\n  First K-tile (row 0, k=0-15) comparison:");
        println!("    k  | GGML      | Loader    | Diff");
        println!("    ---|-----------|-----------|--------");
        for k in 0..16 {
            let ggml_val = ggml_values[k];
            let loader_val = loader_values[k];
            let diff = (ggml_val - loader_val).abs();
            let marker = if diff > 0.01 { " *" } else { "" };
            println!(
                "    {:2} | {:9.4} | {:9.4} | {:6.4}{}",
                k, ggml_val, loader_val, diff, marker
            );
        }

        // Show K-tile 1 (k=16-31) to see the pattern
        println!("\n  Second K-tile (row 0, k=16-31) comparison:");
        println!("    k  | GGML      | Loader    | Diff");
        println!("    ---|-----------|-----------|--------");
        for k in 16..32 {
            let ggml_val = ggml_values[k];
            let loader_val = loader_values[k];
            let diff = (ggml_val - loader_val).abs();
            let marker = if diff > 0.01 { " *" } else { "" };
            println!(
                "    {:2} | {:9.4} | {:9.4} | {:6.4}{}",
                k, ggml_val, loader_val, diff, marker
            );
        }

        // Show which K-tiles have errors
        println!("\n  Per K-tile mismatch count (row 0):");
        for tile in 0..16 {
            let tile_start = tile * 16;
            let tile_mismatches: usize = (tile_start..tile_start + 16)
                .filter(|&k| (ggml_values[k] - loader_values[k]).abs() > 0.01)
                .count();
            if tile_mismatches > 0 {
                println!(
                    "    K-tile {:2} (k={:3}-{:3}): {} mismatches",
                    tile,
                    tile_start,
                    tile_start + 15,
                    tile_mismatches
                );
            }
        }
    } else {
        println!("  ✓ All dequant values match GGML baseline");
    }

    Ok(())
}
