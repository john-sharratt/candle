//! Runs the dtype sweep for a quantised configuration across batch sizes and
//! reports each row of the table.
//!
//! `run_all_dtype_tests` validates the repack first, then makes a quick F16
//! batch-1 check, then sweeps the batch sizes. Precision is compared only within
//! one kernel: the baseline resets at the boundary where the dispatcher changes
//! from the vector kernel to MMQ. Failures are collected and reported together,
//! and diagnostics run before the test fails. `SweepWeights` holds the fixture
//! that every row shares, so it is built once.

use super::*;

/// Run a single dtype test for a given quant config
#[cfg(feature = "cuda")]
pub fn run_dtype_test(
    config: &QuantTestConfig,
    input_dtype: DType,
    batch: usize,
    seq: usize,
    device: &Device,
) -> Result<ComparisonStats> {
    let dtype_name = format!("{:?}", input_dtype);
    let (rtol, atol) = get_tolerance_for(config.dtype, input_dtype);

    // Create and quantize weights
    let weights_f32 = create_test_weights(config.nrows, config.ncols, device)?;
    let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;

    // Create test input with distinct values per batch element
    let input = create_test_input(batch, seq, config.ncols, input_dtype, device)?;

    // Baseline: dequantize + matmul
    let baseline = compute_baseline(
        &qtensor,
        &input,
        batch,
        seq,
        config.nrows,
        config.ncols,
        device,
    )?;

    // GEMX path via wrapper
    let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;
    let gemx_result = qmatmul.forward(&input)?;

    // Compare using dtype-appropriate tolerance
    let stats = compare_tensors(&baseline, &gemx_result)?;

    // Print compact format: batch, dtype, max_diff
    println!(
        "  batch={:>3} seq={} {}: max_diff={:.6}",
        batch, seq, dtype_name, stats.max_diff
    );

    assert_approx_eq(&baseline, &gemx_result, rtol, atol)?;

    Ok(stats)
}

/// The weights a sweep reuses, built once.
///
/// Everything here is a pure function of [`QuantTestConfig`], so it is identical
/// for every dtype and batch size the sweep visits. Rebuilding it per
/// configuration is what made these tests the slowest in the workspace: at
/// 2048×2048 the fixture is a 4.2M-element host loop, an upload, a quantize and
/// a dequantize, repeated 44 times to compare 44 matmuls against it. Hoisting it
/// leaves the per-configuration work as the thing actually under test — build an
/// input, run both paths, compare.
#[cfg(feature = "cuda")]
pub struct SweepWeights {
    /// The quantized weights under test, wrapped for the GEMX path.
    pub(super) qmatmul: QMatMulWrapper,
    /// Dequantized and transposed, ready to be the reference matmul's operand.
    /// F32, so the comparison never pays a conversion error the kernel did not
    /// make.
    pub(super) weights_t: Tensor,
}

#[cfg(feature = "cuda")]
impl SweepWeights {
    pub fn build(config: &QuantTestConfig, device: &Device) -> Result<Self> {
        let weights_f32 = create_test_weights(config.nrows, config.ncols, device)?;
        let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;
        let weights_t = qtensor.dequantize(device)?.t()?;
        Ok(Self {
            qmatmul: QMatMulWrapper::from_qtensor(qtensor)?,
            weights_t,
        })
    }

    /// Does the kernel agree with the reference at all, at batch 1 in F16?
    ///
    /// The cheapest question worth asking before printing a 44-row table: a
    /// kernel that is broken outright makes every row below noise.
    pub(super) fn quick_check(&self, config: &QuantTestConfig, device: &Device) -> bool {
        let check = || -> Result<()> {
            let input = create_test_input(1, 1, config.ncols, DType::F16, device)?;
            let baseline = self.baseline(&input, 1, 1, config)?;
            let result = self.qmatmul.forward(&input)?;
            let (rtol, atol) = get_tolerance_for(config.dtype, DType::F16);
            assert_approx_eq(&baseline, &result, rtol, atol)
        };
        check().is_ok()
    }

    /// The reference result: dequantized weights through an ordinary matmul.
    pub(super) fn baseline(
        &self,
        input: &Tensor,
        batch: usize,
        seq: usize,
        config: &QuantTestConfig,
    ) -> Result<Tensor> {
        let input_2d = input
            .to_dtype(DType::F32)?
            .reshape((batch * seq, config.ncols))?;
        input_2d
            .matmul(&self.weights_t)?
            .reshape((batch, seq, config.nrows))
    }
}

/// Result of a single test run (for deferred failure reporting)
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct TestResult {
    pub dtype: DType,
    pub batch: usize,
    pub seq: usize,
    pub stats: Option<ComparisonStats>,
    pub error: Option<String>,
    pub tolerance_exceeded: bool,
    pub zero_distribution_exceeded: bool,
}

/// Run all dtype tests for a given quant config across multiple batch sizes.
/// Tracks precision across batch sizes to detect kernel bugs.
/// Collects all results and only fails at the end to show complete picture.
#[cfg(feature = "cuda")]
pub fn run_all_dtype_tests(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    // STEP 1: Validate repacking first - if this fails, matmul will definitely fail
    let repack_validation = test_repacking(config, device);
    if let Err(e) = repack_validation {
        println!("❌ {} repack FAILED: {}", config.name, e);
        return Err(e);
    }
    println!(
        "✓ {} repack validation passed (embedded K/128 layout)",
        config.name
    );

    // The fixture every row below shares. It depends only on `config`, so the
    // sweep used to rebuild this identical 2048×2048 matrix 44 times over —
    // quantizing and dequantizing 4.2M elements each time to compare one matmul
    // against it.
    let weights = SweepWeights::build(config, device)?;

    // STEP 2: Quick kernel functionality check (silent unless errors).
    //
    // Only the *small* config earns its own fixture: 256×256 is a different
    // shape — one quant block rather than eight — so it exercises tiling the
    // sweep never reaches. The large check is batch 1 at F16, which is literally
    // the sweep's first row, so it reuses the shared weights instead of building
    // a second copy of them to ask the same question.
    let small_config = QuantTestConfig::small(config.name, config.dtype);
    let small_works = test_single_config_quick(&small_config, device);
    let large_works = weights.quick_check(config, device);

    if !small_works || !large_works {
        println!(
            "⚠️  {} kernel issues: small={}, large={}",
            config.name,
            if small_works { "OK" } else { "FAIL" },
            if large_works { "OK" } else { "FAIL" }
        );
    }

    // STEP 3: Full batch size sweep with live table output
    let mut all_results: Vec<TestResult> = Vec::new();
    let mut any_degradation = false;
    let mut degraded_dtypes: Vec<String> = Vec::new();

    // Print table header
    println!("\n================================================================================");
    println!(
        "SUMMARY TABLE: {} ({}x{} matrix)",
        config.name, config.nrows, config.ncols
    );
    println!("================================================================================");
    println!(
        "{:<6} {:>5} {:>5}   {:>10}  {:>10}  {:>8} {:>6} {:>6} Status",
        "DType", "Batch", "Seq", "max_diff", "mean_diff", "max_rel", "bl_z%", "rs_z%"
    );
    println!("------------------------------------------------------------------------------------------");

    for &dtype in TEST_DTYPES {
        let dtype_name = format!("{:?}", dtype);
        let mut baseline_max_diff: Option<f32> = None;
        let mut degradation_detected = false;
        let mut crossed_mmq = false;
        let (rtol, atol) = get_tolerance_for(config.dtype, dtype);

        // Test across all batch sizes with seq=1 for simplicity
        for &batch_size in TEST_BATCH_SIZES {
            let result =
                run_dtype_test_no_fail(config, &weights, dtype, batch_size, 1, device, rtol, atol);

            // Print row immediately
            let dtype_str = format!("{:?}", result.dtype);
            if let Some(stats) = &result.stats {
                let bl_pct = if stats.num_elements > 0 {
                    100.0 * stats.baseline_near_zero as f64 / stats.num_elements as f64
                } else {
                    0.0
                };
                let rs_pct = if stats.num_elements > 0 {
                    100.0 * stats.result_near_zero as f64 / stats.num_elements as f64
                } else {
                    0.0
                };

                let status = if result.error.is_some() {
                    "ERR"
                } else if result.tolerance_exceeded {
                    "FAIL"
                } else {
                    "OK"
                };

                println!(
                    "{:<6} {:>5} {:>5}   {:>10.6}  {:>10.6}  {:>8.4} {:>5.2} {:>6.2} {}",
                    dtype_str,
                    result.batch,
                    result.seq,
                    stats.max_diff,
                    stats.mean_diff,
                    stats.max_rel_diff,
                    bl_pct,
                    rs_pct,
                    status
                );
            } else if let Some(err) = &result.error {
                println!(
                    "{:<6} {:>5} {:>5}   ERROR: {}",
                    dtype_str, result.batch, result.seq, err
                );
            }

            if let Some(stats) = &result.stats {
                // Precision must not degrade with batch size **within one
                // kernel**. Across the vec→MMQ boundary it legitimately does:
                // those are different kernels with different reduction orders,
                // and the dispatcher picks between them precisely by batch size
                // (`max_bm` in `QCudaStorage::fwd`). Measured on Q4_K, BF16:
                // 0.0049 at batch 1 rising to 0.0073 at batch 8, then 0.0300 at
                // 16, 0.0429 at 64, 0.0465 at 128 — a 4x step exactly at the
                // boundary and a smooth climb on either side of it.
                //
                // Comparing across it asserted an invariant the design does not
                // have ("batch size should NOT affect precision"), so the
                // baseline resets when the kernel changes and each arm is
                // judged against its own first batch.
                if batch_size > VEC_KERNEL_MAX_BATCH && baseline_max_diff.is_some() && !crossed_mmq
                {
                    crossed_mmq = true;
                    baseline_max_diff = None;
                }
                match baseline_max_diff {
                    Some(base_diff) => {
                        // For FP8, a fixed threshold: its own quantization error
                        // (2 ULP = 0.125) dwarfs any batch effect.
                        let threshold = if dtype == DType::F8E4M3 {
                            0.15
                        } else {
                            (base_diff * 3.0).max(base_diff + 0.01)
                        };
                        if stats.max_diff.abs() > threshold {
                            degradation_detected = true;
                        }
                    }
                    None => baseline_max_diff = Some(stats.max_diff),
                }
            }
            all_results.push(result);
        }

        if degradation_detected {
            any_degradation = true;
            degraded_dtypes.push(dtype_name);
        }
    }

    println!("------------------------------------------------------------------------------------------");

    // Collect failures
    let fail_results: Vec<_> = all_results
        .iter()
        .filter(|r| r.error.is_some() || r.tolerance_exceeded)
        .collect();

    if !fail_results.is_empty() {
        println!("  {} failures:", fail_results.len());
        for result in &fail_results {
            let dtype_str = format!("{:?}", result.dtype);
            if let Some(err) = &result.error {
                println!(
                    "    {:<6} batch={:>3}: ERROR - {}",
                    dtype_str, result.batch, err
                );
            } else if let Some(stats) = &result.stats {
                println!(
                    "    {:<6} batch={:>3}: max_diff={:.6} exceeded tolerance",
                    dtype_str, result.batch, stats.max_diff
                );
            }
        }
    }

    if any_degradation {
        println!(
            "  ❌ Precision degradation detected for: {}",
            degraded_dtypes.join(", ")
        );
    }

    // Run diagnostics on tolerance failure or degradation (before failing the test)
    let any_failures = !fail_results.is_empty();
    if any_failures {
        if let Err(e) = run_matmul_diagnostics(config, device) {
            println!("  Diagnostic error: {}", e);
        }
    }

    // Fail on tolerance exceeded or zero distribution shift
    if any_failures {
        let fail_count = all_results
            .iter()
            .filter(|r| r.error.is_some() || r.tolerance_exceeded || r.zero_distribution_exceeded)
            .count();
        let total_count = all_results.len();
        candle::bail!(
            "{} has {} tolerance/distribution failures out of {} tests",
            config.name,
            fail_count,
            total_count
        );
    }

    // Fail on batch-size degradation (separate from tolerance)
    if any_degradation {
        candle::bail!(
            "{} FAILED: Precision degrades with batch size for [{}]. \
            This is a KERNEL BUG - batch size should NOT affect precision. \
            Each batch element is computed independently.",
            config.name,
            degraded_dtypes.join(", ")
        );
    }

    println!(
        "\n✓ {} all dtype tests passed ({} batch sizes × {} dtypes)",
        config.name,
        TEST_BATCH_SIZES.len(),
        TEST_DTYPES.len()
    );
    Ok(())
}

/// Run a single dtype test without failing - returns result for deferred reporting
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dtype_test_no_fail(
    config: &QuantTestConfig,
    weights: &SweepWeights,
    input_dtype: DType,
    batch: usize,
    seq: usize,
    device: &Device,
    rtol: f32,
    atol: f32,
) -> TestResult {
    let _dtype_name = format!("{:?}", input_dtype);

    // Try to run the test
    let test_result = (|| -> Result<(ComparisonStats, bool, Tensor, Tensor)> {
        // Create test input with distinct values per batch element
        let input = create_test_input(batch, seq, config.ncols, input_dtype, device)?;

        // Baseline: the same dequantized weights every configuration compares
        // against, built once by the caller.
        let baseline = weights.baseline(&input, batch, seq, config)?;

        // GEMX path via wrapper
        let gemx_result = weights.qmatmul.forward(&input)?;

        // Compare
        let stats = compare_tensors(&baseline, &gemx_result)?;

        // Check tolerance
        let tolerance_exceeded = check_tolerance(&baseline, &gemx_result, rtol, atol);

        Ok((stats, tolerance_exceeded, baseline, gemx_result))
    })();

    match test_result {
        Ok((stats, tolerance_exceeded, _baseline, _gemx_result)) => {
            // Check for zero distribution shift (more than 1% difference)
            let zero_distribution_exceeded = if stats.num_elements > 0 {
                let bl_pct = 100.0 * stats.baseline_near_zero as f64 / stats.num_elements as f64;
                let rs_pct = 100.0 * stats.result_near_zero as f64 / stats.num_elements as f64;
                (bl_pct - rs_pct).abs() > 1.0
            } else {
                false
            };

            TestResult {
                dtype: input_dtype,
                batch,
                seq,
                stats: Some(stats),
                error: None,
                tolerance_exceeded,
                zero_distribution_exceeded,
            }
        }
        Err(e) => TestResult {
            dtype: input_dtype,
            batch,
            seq,
            stats: None,
            error: Some(format!("{}", e)),
            tolerance_exceeded: false,
            zero_distribution_exceeded: false,
        },
    }
}

/// Quick test to see if a config works at all (single batch, F16)
/// Returns true if kernel runs without error AND produces correct results
#[cfg(feature = "cuda")]
pub(super) fn test_single_config_quick(config: &QuantTestConfig, device: &Device) -> bool {
    let test = || -> Result<()> {
        let weights_f32 = create_test_weights(config.nrows, config.ncols, device)?;
        let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;
        let input = create_test_input(1, 1, config.ncols, DType::F16, device)?;

        // Compute baseline
        let baseline =
            compute_baseline(&qtensor, &input, 1, 1, config.nrows, config.ncols, device)?;

        // Compute GEMX result
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;
        let result = qmatmul.forward(&input)?;

        // Check correctness at this format's F16-activation tolerance
        let (rtol, atol) = get_tolerance_for(config.dtype, DType::F16);
        assert_approx_eq(&baseline, &result, rtol, atol)?;

        Ok(())
    };
    test().is_ok()
}

/// Verify result is not empty/garbage
#[cfg(feature = "cuda")]
pub fn verify_result_sanity(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    let weights_f32 = create_test_weights(config.nrows, config.ncols, device)?;
    let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;

    let input = create_test_input(1, 4, config.ncols, DType::BF16, device)?;

    let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;
    let result = qmatmul.forward(&input)?;

    // Check result is not all zeros
    let result_f32 = result.to_dtype(DType::F32)?.flatten_all()?;
    let result_vec: Vec<f32> = result_f32.to_vec1()?;

    // Check for NaN
    for (i, &v) in result_vec.iter().enumerate() {
        if v.is_nan() {
            candle::bail!("NaN in result at index {}", i);
        }
    }

    let all_zero = result_vec.iter().all(|&x| x.abs() < 1e-10);
    if all_zero {
        candle::bail!("Result is all zeros");
    }

    // Check result has reasonable magnitude (not garbage)
    let max_val = result_vec.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max_val < 0.01 {
        candle::bail!("Result max value too small: {}", max_val);
    }
    if max_val > 1000.0 {
        candle::bail!("Result max value suspiciously large: {}", max_val);
    }

    println!(
        "✓ {} result sanity check passed (max={})",
        config.name, max_val
    );
    Ok(())
}
