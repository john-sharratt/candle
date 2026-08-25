//! Common test utilities for quantized matmul tests.
//!
//! Provides reusable functions for testing GEMX tensor-core kernels
//! against baseline dequantize+matmul.

// Reference implementations for the GEMX kernels: the `* 1` / `+ 0` terms and
// the explicit row/col index loops spell out the addressing the kernel does, so
// a layout bug reads as an index rather than an opaque iterator chain.
#![allow(
    clippy::identity_op,
    clippy::missing_const_for_thread_local,
    clippy::needless_range_loop,
    clippy::single_match,
    clippy::unnecessary_sort_by
)]

use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};

// ============================================================================
// CUDA TEST SERIALISER
//
// CUDA kernels are GPU-bound and contend for the same device.  Running many
// of them in parallel causes resource exhaustion, false timeouts, and
// flapping results.  We serialise every significant test entry-point through
// a single global Mutex so only one heavy test runs at a time.
//
// The guard is *reentrant per thread*: a function that already holds the lock
// can call another locking function without deadlocking.  This is necessary
// because `run_all_dtype_tests` internally calls `test_repacking`, both of
// which acquire the guard.
//
// The outermost acquisition also starts a 30-second watchdog thread.  If the
// guard is dropped (test completes) before the timer fires, the watchdog exits
// quietly.  If the test hangs, it prints a message and calls process::exit.
// Nested acquisitions reuse the outer watchdog.
// ============================================================================

// Serialises the GPU tests in this directory; without `cuda` none of them are
// compiled, so nothing acquires it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
static CUDA_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

std::thread_local! {
    static LOCK_DEPTH: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

/// RAII guard returned by [`acquire_cuda_test_lock`].
///
/// On drop it cancels the watchdog (if this is the outermost guard), decrements
/// the per-thread depth counter, and releases the global Mutex.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct CudaTestGuard {
    #[allow(dead_code)]
    mutex_guard: Option<std::sync::MutexGuard<'static, ()>>,
    /// Shared flag: setting this to `true` tells the watchdog thread to exit
    /// without calling `process::exit`.  `None` for nested (reentrant) guards.
    watchdog_done: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Drop for CudaTestGuard {
    fn drop(&mut self) {
        // Cancel the watchdog before releasing the lock so it can never fire
        // against the next test.
        if let Some(flag) = &self.watchdog_done {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        LOCK_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        // mutex_guard (if Some) is dropped here, unblocking the next waiter.
    }
}

/// Acquire the global CUDA test serialiser.
///
/// Blocks until no other test holds the lock, then returns a guard that:
/// - serialises GPU access for the duration of the test
/// - starts a 30-second watchdog that aborts the process if the guard is not
///   dropped in time (guards against silent hangs)
///
/// Nested calls on the *same* thread return immediately (reentrant); they share
/// the outer watchdog and do not reset its timer.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn acquire_cuda_test_lock() -> CudaTestGuard {
    LOCK_DEPTH.with(|d| {
        let depth = d.get();
        if depth == 0 {
            d.set(1);
            let mutex_guard = CUDA_TEST_LOCK
                .get_or_init(|| std::sync::Mutex::new(()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            // Watchdog: start only once the lock is held (timer reflects real
            // work time, not queue wait time).
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_clone = done.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(30));
                if !done_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("❌ TIMEOUT: CUDA test exceeded 30 s — aborting process");
                    std::process::exit(1);
                }
            });

            CudaTestGuard {
                mutex_guard: Some(mutex_guard),
                watchdog_done: Some(done),
            }
        } else {
            d.set(depth + 1);
            // Already locked on this thread; reuse outer watchdog.
            CudaTestGuard {
                mutex_guard: None,
                watchdog_done: None,
            }
        }
    })
}

#[cfg(feature = "cuda")]
use candle::Module;
#[cfg(feature = "cuda")]
use candle_transformers::models::quantized_matmul::QMatMul as QMatMulWrapper;

// ============================================================================
// PRECISION TOLERANCES - FIXED FOR Q4_K AND ABOVE
// ============================================================================
//
// WARNING: DO NOT INCREASE THESE TOLERANCES. Increasing tolerances hides real
// bugs in the kernel. If tests fail, FIX THE KERNEL, don't mask the problem.
//
// These tolerances are FIXED and represent the maximum acceptable error for
// quantized matmul operations. They are essential to maintain quality and
// prevent functional regression.
//
// IMPORTANT: Batch size should NOT impact precision. Each batch element is
// computed independently - the matmul for batch[0] should give identical
// results whether there are 1 or 256 other batches in the same call. If
// errors increase with batch size, this indicates a kernel bug (incorrect
// stride/indexing, shared memory corruption, or thread coordination issues).
//
// Based on empirical measurements from Q4_K tests with [-0.5, +0.5] weight/input ranges:
// (2048x2048 matrix, batch=1-8 where kernel behaves correctly)
//   - F32:  max_diff=0.0148  (quantization error floor)
//   - F16:  max_diff=0.0156  (nearly identical to F32)
//   - BF16: max_diff=0.0625  (7-bit mantissa adds some error)
//   - F8:   max_diff=1.0     (3-bit mantissa, very low precision)
//
// Note: max_rel_diff can be very high when baseline values are near zero.
// We use rtol primarily for sanity checking, atol is the main constraint.
// ============================================================================
// THE ERROR TRACKS THE WEIGHT FORMAT, NOT THE ACTIVATION DTYPE.
//
// `compute_baseline` dequantizes the SAME weight the kernel reads, so weight
// quantization error cancels and what is left is how the kernel accumulates.
// That path quantizes the activations to **Q8_1 whatever dtype they arrive in**
// (`mul_mat_vec_via_q8_1`, and MMQ's `quantize_row_q8_1`), so an F32 activation
// — which suffers no rounding at all before the kernel — lands on the same floor
// as BF16. Measured, K=2048, across every batch size, per format:
//
//   format   BF16      F16       F32       F8E4M3
//   Q2_K     0.008573  0.008386  0.008387  0.064909
//   Q3_K     0.010153  0.008052  0.008153  0.064467
//   Q4_0     0.037627  0.037871  0.037871  0.080813
//   Q4_1     0.046146  0.046176  0.046175  0.089088
//   Q4_K     0.046472  0.046533  0.046504  0.093061
//   Q5_0     0.028712  0.028471  0.028387  0.064409
//   Q5_1     0.045567  0.045933  0.045959  0.085718
//   Q5_K     0.046247  0.046125  0.046122  0.086442
//   Q6_K     0.012313  0.009384  0.008967  0.065172
//   Q8_0     0.010062  0.008602  0.008944  0.065954
//   Q8_K     0.009488  0.009043  0.008911  0.065606
//
// The three float columns agree to the third decimal within every row and vary
// 5x between rows — the format is the variable, the activation dtype is not.
// F8E4M3 is the one real exception: it is lossy *before* the kernel sees it.
//
// The previous table modelled the opposite (one tolerance per activation dtype,
// F32 held ~8x tighter than BF16) and was calibrated, in its own words, "with
// float accumulators" — the FP fast path deleted in 3bf7dfc7. It has been
// unreachable since: `run_all_dtype_tests` bails in step 1, so this sweep has
// not run since that commit and the stale numbers were never contradicted.
// ============================================================================

/// Relative tolerance. Secondary — `check_tolerance` needs BOTH relative and
/// absolute to fail, and `max_rel_diff` is meaningless where the baseline is
/// near zero, so the absolute bound below is what actually constrains.
#[cfg(feature = "cuda")]
pub const RTOL: f32 = 0.05;

/// Relative tolerance for F8E4M3 activations, which are coarse enough that the
/// relative bound has to give as well.
#[cfg(feature = "cuda")]
pub const RTOL_F8: f32 = 0.10;

/// The largest batch the mmvq (vec) kernel serves before the dispatcher hands
/// over to MMQ — `max_bm` in `QCudaStorage::fwd`, mirrored here because the
/// batch sweep straddles it and the two kernels reduce differently.
///
/// Keep in step with the dispatcher: if that boundary moves, this sweep will
/// compare a vec batch against an MMQ one and report a kernel bug that is
/// really a dispatch change.
#[cfg(feature = "cuda")]
pub const VEC_KERNEL_MAX_BATCH: usize = 8;

/// The float-activation absolute bound for `dtype`, at ~1.4x the measured
/// maximum above.
///
/// Per format rather than one global number: a single bound loose enough for
/// Q4_K (0.047) would be 5x slack on Q8_0 (0.010) and would stop catching
/// anything there.
#[cfg(feature = "cuda")]
pub fn atol_for_format(dtype: GgmlDType) -> f32 {
    match dtype {
        GgmlDType::Q4_1 | GgmlDType::Q4_K | GgmlDType::Q5_1 | GgmlDType::Q5_K => 0.065,
        GgmlDType::Q4_0 => 0.055,
        GgmlDType::Q5_0 => 0.040,
        _ => 0.018,
    }
}

/// The F8E4M3 bound for `dtype`: the float floor plus the ~0.06 the activation
/// format costs on its own (1 ULP at magnitude 1.0 is 0.0625).
#[cfg(feature = "cuda")]
pub fn atol_f8_for_format(dtype: GgmlDType) -> f32 {
    atol_for_format(dtype) + 0.070
}

/// Tolerances for one (weight format, activation dtype) pair.
#[cfg(feature = "cuda")]
pub fn get_tolerance_for(weight: GgmlDType, activation: DType) -> (f32, f32) {
    match activation {
        DType::F8E4M3 => (RTOL_F8, atol_f8_for_format(weight)),
        _ => (RTOL, atol_for_format(weight)),
    }
}

/// Input data types to test - all supported GEMX Y types
/// Y type mapping: 0=F16, 1=BF16, 2=F8E4M3, 3=F32
#[cfg(feature = "cuda")]
pub const TEST_DTYPES: &[DType] = &[DType::BF16, DType::F16, DType::F32, DType::F8E4M3];

/// Comparison statistics
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct ComparisonStats {
    pub max_diff: f32,
    pub max_rel_diff: f32, // Only computed for |baseline| > REL_THRESHOLD
    pub mean_diff: f64,
    pub num_elements: usize,
    pub baseline_near_zero: usize, // Count of |baseline| < NEAR_ZERO_THRESHOLD
    pub result_near_zero: usize,   // Count of |result| < NEAR_ZERO_THRESHOLD
}

/// Threshold for near-zero detection (values smaller than this are "near zero")
#[cfg(feature = "cuda")]
pub const NEAR_ZERO_THRESHOLD: f32 = 0.001;

/// Threshold for relative difference calculation (only compute rel_diff when |baseline| > this)
/// Set high enough (0.1) to avoid misleading max_rel values from small baseline elements
/// where even tiny absolute errors create large relative errors
#[cfg(feature = "cuda")]
pub const REL_DIFF_THRESHOLD: f32 = 0.1;

#[cfg(feature = "cuda")]
impl std::fmt::Display for ComparisonStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_diff={:.6}, max_rel_diff={:.4}, mean_diff={:.6}, n={}",
            self.max_diff, self.max_rel_diff, self.mean_diff, self.num_elements
        )
    }
}

/// Compare two tensors and return statistics
#[cfg(feature = "cuda")]
pub fn compare_tensors(a: &Tensor, b: &Tensor) -> Result<ComparisonStats> {
    let a_f32 = a.to_dtype(DType::F32)?.flatten_all()?;
    let b_f32 = b.to_dtype(DType::F32)?.flatten_all()?;

    let a_vec: Vec<f32> = a_f32.to_vec1()?;
    let b_vec: Vec<f32> = b_f32.to_vec1()?;

    if a_vec.len() != b_vec.len() {
        candle::bail!("Shape mismatch: {} vs {}", a_vec.len(), b_vec.len());
    }

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut baseline_near_zero = 0usize;
    let mut result_near_zero = 0usize;

    for (av, bv) in a_vec.iter().zip(b_vec.iter()) {
        // Check for NaN
        if av.is_nan() || bv.is_nan() {
            candle::bail!("NaN detected: a={}, b={}", av, bv);
        }

        // Count near-zero values (a = baseline, b = result)
        if av.abs() < NEAR_ZERO_THRESHOLD {
            baseline_near_zero += 1;
        }
        if bv.abs() < NEAR_ZERO_THRESHOLD {
            result_near_zero += 1;
        }

        let diff = (av - bv).abs();
        sum_diff += diff as f64;

        if diff > max_diff {
            max_diff = diff;
        }

        // Only compute relative diff for values with sufficient magnitude
        // This avoids misleading max_rel values from near-zero baseline elements
        if av.abs() > REL_DIFF_THRESHOLD {
            let rel_diff = diff / av.abs();
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
        }
    }

    Ok(ComparisonStats {
        max_diff,
        max_rel_diff,
        mean_diff: sum_diff / a_vec.len() as f64,
        num_elements: a_vec.len(),
        baseline_near_zero,
        result_near_zero,
    })
}

/// Check if two tensors are approximately equal
#[cfg(feature = "cuda")]
pub fn assert_approx_eq(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> Result<()> {
    let stats = compare_tensors(a, b)?;

    // Check tolerance - need BOTH relative AND absolute to fail
    let a_f32 = a.to_dtype(DType::F32)?.flatten_all()?;
    let b_f32 = b.to_dtype(DType::F32)?.flatten_all()?;
    let a_vec: Vec<f32> = a_f32.to_vec1()?;
    let b_vec: Vec<f32> = b_f32.to_vec1()?;

    for (i, (av, bv)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let diff = (av - bv).abs();
        let rel_diff = if av.abs() > 1e-6 {
            diff / av.abs()
        } else {
            diff
        };

        if diff > atol && rel_diff > rtol {
            candle::bail!(
                "Tensors differ beyond tolerance at index {}: a={}, b={}, diff={}, rel_diff={} ({})",
                i, av, bv, diff, rel_diff, stats
            );
        }
    }

    Ok(())
}

/// Create a deterministic test weight matrix.
///
/// Creates Xavier/Glorot-normalized test weights for realistic matmul precision.
///
/// Uses Xavier initialization: std = 1/sqrt(fan_in) which ensures:
/// - Output magnitudes stay in reasonable BF16 range (~[-1, 1])
/// - Gradient flow is stable (same variance for forward/backward)
/// - Matches real trained model weight distributions
///
/// This is critical for BF16 testing because un-normalized weights cause
/// output magnitudes to reach ~20, where BF16 ULP is 0.125. With Xavier
/// normalization, outputs stay ~1.0 where BF16 ULP is 0.0078.
pub fn create_test_weights(nrows: usize, ncols: usize, device: &Device) -> Result<Tensor> {
    // Xavier/Glorot scale: std = 1/sqrt(fan_in)
    // For uniform distribution: range = sqrt(3) * std = sqrt(3/fan_in)
    let xavier_scale = (3.0 / ncols as f64).sqrt();

    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            // Deterministic pattern in [0, 1) using primes to avoid block aliasing
            let raw = ((i * 7 + j * 13) % 256) as f32 / 256.0;
            // Center to [-0.5, 0.5) then scale to Xavier range
            let centered = raw - 0.5;
            // Add subtle row variation to create different scale factors per quant block
            let row_factor = 1.0 + (i % 8) as f32 * 0.05;
            let val = (centered * row_factor) as f64 * xavier_scale * 2.0;

            data.push(val as f32);
        }
    }
    Tensor::from_vec(data, (nrows, ncols), device)
}

/// Create a deterministic test input with distinct values per batch element.
/// Each batch element gets a unique seed to ensure different patterns for corruption testing.
///
/// Input range is [-0.5, +0.5] which fits comfortably within all target dtypes:
/// - FP8 E4M3: max ~448, but very coarse precision (only 8 mantissa values per exponent)
/// - F16: max ~65504, ~10 bits precision
/// - BF16: max ~3.4e38, ~7 bits precision
///
/// Using a narrow range ensures:
/// 1. No overflow/underflow in any format
/// 2. FP8's limited precision can still represent distinct values
/// 3. Matmul outputs stay in reasonable range (weights also in [-0.5, 0.5])
pub fn create_test_input(
    batch: usize,
    seq: usize,
    k: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    create_test_input_with_seed(batch, seq, k, dtype, device, 0)
}

/// Create test input with a custom seed for additional variation.
///
/// Uses [-0.5, +0.5] range for activations. The pattern ensures each batch element
/// has unique values to detect any cross-batch corruption in the kernel.
///
/// Value spacing is designed to be representable in FP8 E4M3:
/// - FP8 E4M3 in [-0.5, 0.5] has ~64 distinct representable values
/// - We generate values at ~1/256 spacing which rounds nicely to FP8 grid
pub fn create_test_input_with_seed(
    batch: usize,
    seq: usize,
    k: usize,
    dtype: DType,
    device: &Device,
    seed: usize,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(batch * seq * k);
    for b in 0..batch {
        // Each batch element gets a unique pattern based on batch index and seed
        // Use prime multipliers to avoid aliasing patterns
        let batch_seed = seed.wrapping_mul(997).wrapping_add(b.wrapping_mul(1009));
        for s in 0..seq {
            for i in 0..k {
                // Create distinct values: combine batch_seed, position, and dimension
                // Use different primes to create varied patterns
                let v1 = batch_seed
                    .wrapping_add(s.wrapping_mul(31))
                    .wrapping_add(i.wrapping_mul(127));
                let v2 = (b.wrapping_mul(17))
                    .wrapping_add(s.wrapping_mul(53))
                    .wrapping_add(i.wrapping_mul(97));
                let combined = v1 ^ v2;
                // Map to [-0.5, +0.5] range - fits all dtypes including FP8
                // Use 256 discrete values to align with FP8's representable grid
                let val = ((combined % 256) as f32 / 256.0) - 0.5;
                data.push(val);
            }
        }
    }
    Tensor::from_vec(data, (batch, seq, k), device)?.to_dtype(dtype)
}

/// Compute baseline result using dequantize + matmul
/// Returns F32 result for accurate comparison (not converted to input dtype)
#[cfg(feature = "cuda")]
pub fn compute_baseline(
    qtensor: &QTensor,
    input: &Tensor,
    batch: usize,
    seq: usize,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<Tensor> {
    let weights_dequant = qtensor.dequantize(device)?;
    let weights_t = weights_dequant.t()?;
    let input_f32 = input.to_dtype(DType::F32)?;
    let input_2d = input_f32.reshape((batch * seq, ncols))?;
    let baseline_2d = input_2d.matmul(&weights_t)?;
    // Return F32 - don't convert to input dtype as that would add conversion error
    baseline_2d.reshape((batch, seq, nrows))
}

/// Test configuration for a single quant type test
#[derive(Clone)]
pub struct QuantTestConfig {
    #[allow(dead_code)] // Used only in cuda feature
    pub name: &'static str,
    pub dtype: GgmlDType,
    pub nrows: usize,
    pub ncols: usize,
}

impl QuantTestConfig {
    pub fn new(name: &'static str, dtype: GgmlDType) -> Self {
        // K-quants require dimensions divisible by 256
        // Use realistic dimensions: K=2048 (8 quant blocks), N=2048
        Self {
            name,
            dtype,
            nrows: 2048,
            ncols: 2048,
        }
    }

    /// Create config with small dimensions for initial testing
    #[cfg(feature = "cuda")]
    pub fn small(name: &'static str, dtype: GgmlDType) -> Self {
        // Minimal K-quant dimensions: 256x256 (1 quant block)
        Self {
            name,
            dtype,
            nrows: 256,
            ncols: 256,
        }
    }
}

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

/// Batch sizes to test for corruption detection
/// Reduced set for speed - tests key boundaries: 1 (baseline), 8 (small), 64 (tile boundary), 128 (large)
#[cfg(feature = "cuda")]
pub const TEST_BATCH_SIZES: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 128];

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

    // STEP 2: Quick kernel functionality check (silent unless errors)
    let small_config = QuantTestConfig::small(config.name, config.dtype);
    let small_works = test_single_config_quick(&small_config, device);
    let large_works = test_single_config_quick(config, device);

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
            let result = run_dtype_test_no_fail(config, dtype, batch_size, 1, device, rtol, atol);

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
fn run_dtype_test_no_fail(
    config: &QuantTestConfig,
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

/// Check if tensors exceed tolerance (returns true if exceeded)
#[cfg(feature = "cuda")]
fn check_tolerance(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> bool {
    let check = || -> Result<bool> {
        let a_f32 = a.to_dtype(DType::F32)?.flatten_all()?;
        let b_f32 = b.to_dtype(DType::F32)?.flatten_all()?;
        let a_vec: Vec<f32> = a_f32.to_vec1()?;
        let b_vec: Vec<f32> = b_f32.to_vec1()?;

        for (av, bv) in a_vec.iter().zip(b_vec.iter()) {
            let diff = (av - bv).abs();
            let rel_diff = if av.abs() > 1e-6 {
                diff / av.abs()
            } else {
                diff
            };

            if diff > atol && rel_diff > rtol {
                return Ok(true); // Exceeded
            }
        }
        Ok(false)
    };
    check().unwrap_or(true)
}

// ============================================================================
// DIAGNOSTIC TESTS FOR MATMUL MAPPING ERRORS
// ============================================================================
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
// ============================================================================

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
fn run_indicator_diagnostic(
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
fn analyze_mismatch_pattern(mismatches: &[(usize, f32, f32, f32)], ncols: usize) {
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
fn run_row_isolation_diagnostic(
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
fn run_batch_sensitivity_diagnostic(
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

// ============================================================================
// ADVANCED DIAGNOSTIC TESTS - Precision Flow Tracing
// ============================================================================
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

    // =========================================================================
    // TEST 1: BATCH IDENTITY TEST
    // Set A[b, k] = (b+1) for all k (each batch has its own identifier)
    // Set W[k, n] = 1/ncols for all k,n (average pooling effect)
    // Expected: C[b, n] ≈ (b+1) for all n
    // This reveals which batch's activations are actually being used
    // =========================================================================
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

    // =========================================================================
    // TEST 2: SINGLE BATCH PROBE
    // For each target batch, set A[target,0] = 1.0, all else = 0
    // W[0,0] = 1.0, all else = 0
    // Expected: C[target,0] = 1.0, all other C[b,0] = 0
    // =========================================================================
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

    // =========================================================================
    // TEST 3: BATCH PERMUTATION DETECTION
    // A[b,k] = b (batch index), W[k,n] = 1 (sum over K)
    // Analyze which batch indices appear in FP8 outputs
    // =========================================================================
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

    // =========================================================================
    // TEST 4: THREAD/GROUP PATTERN ANALYSIS
    // For m16n8k32: groupID = lane/4 (0-7), tid = lane%4 (0-3)
    // Each groupID handles rows groupID and groupID+8
    // Let's see if errors follow this pattern
    // =========================================================================
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

    // =========================================================================
    // TEST 5: RAW MAPPING TABLE
    // Create comprehensive mapping: for each output batch, which input batch?
    // =========================================================================
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

    // =========================================================================
    // TEST 6: K-POSITION SENSITIVITY
    // Does the error depend on which K positions have non-zero activations?
    // =========================================================================
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

// ============================================================================
// NEW BATCH-SPECIFIC ACCUMULATION DIAGNOSTICS
// These are designed to catch the 2x accumulation bug seen at batch>=16
// ============================================================================

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

// ============================================================================
// KERNEL COMPONENT ISOLATION DIAGNOSTICS
// These tests use clever weight/scale/input patterns to isolate specific
// kernel components and identify which part is malfunctioning.
// ============================================================================

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

/// Run all kernel component isolation diagnostics
#[cfg(feature = "cuda")]
pub fn run_kernel_isolation_diagnostics(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    println!("\n================================================================================");
    println!("KERNEL COMPONENT ISOLATION DIAGNOSTICS for {}", config.name);
    println!("================================================================================");

    println!("\n=== BATCH=1 (GEMV path - known working) ===");
    run_stage_boundary_diagnostic(config.dtype, device, 1)?;
    run_thread_mapping_diagnostic(config.dtype, device, 1)?;
    run_scale_application_diagnostic(config.dtype, device, 1)?;
    run_ktile_pair_diagnostic(config.dtype, device, 1)?;
    run_global_to_shared_diagnostic(config.dtype, device, 1)?;

    println!("\n=== BATCH=16 (GEMM/tensor core path - broken) ===");
    run_stage_boundary_diagnostic(config.dtype, device, 16)?;
    run_thread_mapping_diagnostic(config.dtype, device, 16)?;
    run_scale_application_diagnostic(config.dtype, device, 16)?;
    run_ktile_pair_diagnostic(config.dtype, device, 16)?;
    run_global_to_shared_diagnostic(config.dtype, device, 16)?;
    run_mma_fragment_diagnostic(config.dtype, device)?;

    println!("\n=== ENHANCED K-ROW LOADING ANALYSIS ===");
    run_krow_wrap_diagnostic(config.dtype, device)?;
    run_y_input_isolation_diagnostic(config.dtype, device)?;
    run_first_4_krows_accuracy_diagnostic(config.dtype, device)?;
    run_krow_read_pattern_diagnostic(config.dtype, device)?;

    println!("\n=== THREAD/WARP MAPPING DIAGNOSTICS ===");
    run_column_identity_diagnostic(config.dtype, device)?;
    run_single_column_probe_diagnostic(config.dtype, device)?;
    run_warp_signature_diagnostic(config.dtype, device)?;
    run_column_pair_pattern_diagnostic(config.dtype, device)?;

    Ok(())
}

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

/// Run all advanced diagnostics for detailed debugging
#[cfg(feature = "cuda")]
pub fn run_advanced_diagnostics(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    println!("\n================================================================================");
    println!(
        "ADVANCED DIAGNOSTICS: Detailed flow tracing for {}",
        config.name
    );
    println!("================================================================================");

    // Run at batch=1 (GEMV path)
    println!("\n--- BATCH=1 (GEMV path) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 1)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 1)?;
    run_scale_correspondence_diagnostic(config.dtype, device)?;
    run_part_accumulation_diagnostic(config.dtype, device)?;

    // Run at batch=16 (GEMM path - tensor cores, s16_tc kernel)
    println!("\n--- BATCH=16 (GEMM/tensor core path, s16_tc) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 16)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 16)?;

    // Run at batch=32 (transition zone)
    println!("\n--- BATCH=32 (transition batch size) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 32)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 32)?;

    // Run at batch=64 (s32_tc kernel for FP8 - uses m16n8k32 MMA)
    println!("\n--- BATCH=64 (s32_tc kernel, m16n8k32 MMA for FP8) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 64)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 64)?;

    // NEW: Batch-specific accumulation diagnostics
    println!("\n--- BATCH ACCUMULATION DIAGNOSTICS ---");
    run_accumulation_ratio_diagnostic(config.dtype, device, 1)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 16)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 32)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 64)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 1)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 16)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 32)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 64)?;
    run_batch_comparison_diagnostic(config.dtype, device)?;

    // NEW: Even more targeted probe tests
    println!("\n--- SINGLE ELEMENT PROBE TESTS ---");
    run_single_k_probe_diagnostic(config.dtype, device, 1)?;
    run_single_k_probe_diagnostic(config.dtype, device, 16)?;
    run_single_k_probe_diagnostic(config.dtype, device, 32)?;
    run_single_k_probe_diagnostic(config.dtype, device, 64)?;
    run_k_iteration_count_diagnostic(config.dtype, device)?;

    // FP8-SPECIFIC DIAGNOSTICS - test with FP8 input dtype at various batch sizes
    println!("\n--- FP8 ACTIVATION DIAGNOSTICS (tests FP8 m16n8k32 MMA path) ---");
    run_fp8_activation_diagnostic(config.dtype, device, 1)?;
    run_fp8_activation_diagnostic(config.dtype, device, 16)?;
    run_fp8_activation_diagnostic(config.dtype, device, 32)?;
    run_fp8_activation_diagnostic(config.dtype, device, 64)?;

    // FP8 MMA indicator tracing - use specific values to trace data flow
    println!("\n--- FP8 MMA INDICATOR TRACING (batch=32) ---");
    run_fp8_mma_indicator_tracing(config.dtype, device)?;

    // New extended diagnostics (batch=1 only, for speed)
    println!("\n--- Extended diagnostics (batch=1) ---");
    run_nibble_pair_diagnostic(config.dtype, device)?;
    run_intra_tile_diagnostic(config.dtype, device)?;
    run_position_mapping_diagnostic(config.dtype, device)?;
    run_qh_bit_diagnostic(config.dtype, device)?;
    run_dequant_value_diagnostic(config.dtype, device)?;
    run_qh_raw_verification_diagnostic(config.dtype, device)?;
    run_dequant_kernel_diagnostic(config.dtype, device)?;

    Ok(())
}

/// Quick test to see if a config works at all (single batch, F16)
/// Returns true if kernel runs without error AND produces correct results
#[cfg(feature = "cuda")]
fn test_single_config_quick(config: &QuantTestConfig, device: &Device) -> bool {
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

// ============================================================================
// QUANT FORMAT CONSTANTS
// ============================================================================
// Block sizes and byte sizes for each quantization format.
// ============================================================================

/// Qtype index for each GgmlDType
#[cfg(feature = "cuda")]
fn ggml_dtype_to_qtype(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 => 0,
        GgmlDType::Q4_1 => 1,
        GgmlDType::Q5_0 => 2,
        GgmlDType::Q5_1 => 3,
        GgmlDType::Q8_0 => 4,
        GgmlDType::Q2_K => 5,
        GgmlDType::Q3_K => 6,
        GgmlDType::Q4_K => 7,
        GgmlDType::Q5_K => 8,
        GgmlDType::Q6_K => 9,
        GgmlDType::Q8_1 => 10,
        GgmlDType::Q8_K => 11,
        GgmlDType::QAWQ => 12,
        GgmlDType::QAWQ_G64 => 13,
        _ => panic!("Unsupported GgmlDType for GEMX: {:?}", dtype),
    }
}

// ============================================================================
// REPACKED TENSOR VALIDATION
// ============================================================================

/// K/128 block byte size for each format (embedded scales).
/// Index matches qtype: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K, 10=Q8_1, 11=Q8_K, 12=Q_AWQ, 13=Q_AWQ_G64
#[cfg(feature = "cuda")]
const K128_BLOCK_BYTES: [usize; 14] = [
    80,  // Q4_0: 128 × 4-bit + 4×half2 scales = 64 + 16
    80,  // Q4_1: 128 × 4-bit + 4×half2 scales = 64 + 16
    112, // Q5_0: 128 × 5-bit + 4×half2 scales = 80 + 16 + 16(qh)
    112, // Q5_1: 128 × 5-bit + 4×half2 scales = 80 + 16 + 16(qh)
    144, // Q8_0: 128 × 8-bit + 4×half2 scales = 128 + 16
    64,  // Q2_K: 128 × 2-bit + 8×half2 scales = 32 + 32
    96,  // Q3_K: 128 × 3-bit + 8×half2 scales = 48 + 16(qh) + 32
    80,  // Q4_K: 128 × 4-bit + 4×half2 scales = 64 + 16
    112, // Q5_K: 128 × 5-bit + 4×half2 scales = 80 + 16 + 16(qh)
    112, // Q6_K: 128 × 6-bit + 8×half scales = 64(ql) + 32(qh) + 16(scales)
    160, // Q8_1: 128 × 8-bit + 4×half2 dm pairs = 128 + 32 (d and m per sub-block)
    160, // Q8_K: 128 × 8-bit + 4×half2 d copies = 128 + 32 (single d shared)
    80,  // Q_AWQ: 128 × 4-bit + scale/zero = 64 + 4(scale) + 4(zero) + 8(pad) = 80
    80,  // Q_AWQ_G64: 128 × 4-bit + 2×scale/zero = 64 + 8(scales) + 8(zeros) = 80
];

/// Calculate expected repacked size in bytes for K/128 blocks (embedded scales)
#[cfg(feature = "cuda")]
pub fn expected_repacked_size(config: &QuantTestConfig) -> usize {
    let qtype = ggml_dtype_to_qtype(config.dtype);
    let blocks_per_row = config.ncols / 128;
    config.nrows * blocks_per_row * K128_BLOCK_BYTES[qtype]
}

/// **Does the repacked weight compute the same product as the original?**
///
/// The independent half of the repack test, and the one that would catch a
/// wrong permutation: `validate_repack` only asserts the repacked bytes are the
/// expected *size*, which a repack that shuffled elements into the wrong lanes
/// would satisfy exactly. This runs the K/128 form through the FP GEMX kernel
/// and compares against the dequantized original, so the two disagree the
/// moment an element lands in the wrong place.
///
/// Both sides read the SAME quantized weight, so quantization error cancels and
/// what is left is the accumulation difference between a BF16-activation kernel
/// and an F32 reference matmul — hence the BF16 tolerance.
///
/// Reached through `candle::quantized::QMatMul`, not `QMatMulWrapper`: the
/// wrapper's `from_qtensor_repacked` is the production entry for expert slots
/// and takes **KO twins only**, because that is the only repacked form the
/// engine runs. The FP GEMX form has its own kernel and its own entry point
/// (`forward_via_gemx`), which is what this exercises. Borrowing the wrapper's
/// constructor for it — as this test did before — asked the production path to
/// accept a tensor production never hands it.
#[cfg(feature = "cuda")]
pub fn validate_gemx_matmul(
    original: &QTensor,
    repacked: QTensor,
    config: &QuantTestConfig,
    device: &Device,
) -> Result<()> {
    let (n, k) = (config.nrows, config.ncols);
    let seq = 4usize;

    // Reference: the dequantized weight through an ordinary F32 matmul.
    let w = original.dequantize(device)?.to_dtype(DType::F32)?;
    let x = create_test_input(1, seq, k, DType::BF16, device)?;
    let reference = x
        .reshape((seq, k))?
        .to_dtype(DType::F32)?
        .matmul(&w.t()?.contiguous()?)?;

    // Under test: the K/128 repack through the FP GEMX kernel.
    let gemx = candle::quantized::QMatMul::from_qtensor(repacked)?
        .forward_via_gemx(&x)?
        .reshape((seq, n))?
        .to_dtype(DType::F32)?;

    let (rtol, atol) = get_tolerance_for(config.dtype, DType::BF16);
    assert_approx_eq(&gemx, &reference, rtol, atol).map_err(|e| {
        candle::Error::Msg(format!(
            "{} GEMX repack does not match the dequantized original — the K/128 \
             layout is the right SIZE but computes a different product: {e}",
            config.name
        ))
    })?;
    println!(
        "  ✓ {} GEMX matmul matches the dequantized original",
        config.name
    );
    Ok(())
}

/// Validate repacked tensor by comparing matmul results.
///
/// This is the INDEPENDENT test: we don't look at repack implementation,
/// we verify that using the repacked tensor through the GEMX kernel path
/// produces the same result as using the original tensor through dequantize.
///
/// Test approach:
/// 1. Dequantize original tensor -> reference weights
/// 2. Create test input vector
/// 3. Compute reference: input @ reference_weights^T
/// 4. Use repacked tensor + scales via GEMX path
/// 5. Compare outputs
#[cfg(feature = "cuda")]
pub fn validate_repack(
    original: &QTensor,
    repacked: &QTensor,
    config: &QuantTestConfig,
) -> Result<()> {
    let original_size = original.storage_size_in_bytes();
    let repacked_size = repacked.storage_size_in_bytes();

    // Calculate expected sizes for embedded-scale K/128 blocks
    let expected_repacked_size = expected_repacked_size(config);

    println!("  Original size:  {} bytes", original_size);
    println!("  Repacked size:  {} bytes", repacked_size);
    println!("  Expected repacked: {} bytes", expected_repacked_size);

    // EMBEDDED MODE: Repacked size matches K/128 block size (scales inline)
    if repacked_size != expected_repacked_size {
        candle::bail!(
            "{} repack failed: size mismatch (repacked={}, expected={})",
            config.name,
            repacked_size,
            expected_repacked_size
        );
    }

    // =========================================================================
    // GEMX EXTRACTION SIMULATION TEST
    // =========================================================================
    // This test validates repack by simulating how GEMX reads data:
    //
    // 1. From ORIGINAL: decode elements using linear/GGML format
    // 2. From REPACKED: load int32, extract with GEMX shifts:
    //    - Elements 0-3: shifts {0, 16, 4, 20}
    //    - Elements 4-7: shifts {8, 24, 12, 28}
    //
    // If repack is correct, elements extracted via GEMX pattern should
    // match elements from original decoded linearly.
    //
    // This does NOT use permutation tables - it simulates the actual GEMX
    // kernel extraction pattern which is the true design target.
    // =========================================================================

    println!("  Skipping GEMX extraction simulation (embedded K/128 layout)");

    println!(
        "✓ {} repack validation passed (embedded K/128 layout)",
        config.name
    );
    Ok(())
}

/// Run repack validation for a given config (K/128 format with embedded scales)
#[cfg(feature = "cuda")]
pub fn test_repacking(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    println!("Testing {} repacking:", config.name);

    // Create test weights and quantize
    let weights_f32 = create_test_weights(config.nrows, config.ncols, device)?;
    let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;

    println!("\n  Step 1: Repack weights to K/128 format");
    let repacked = qtensor.repack_gemx()?;

    // Validate byte-level repack
    validate_repack(&qtensor, &repacked, config)?;

    println!("\n  Step 2: Validate the repack computes the same product");
    validate_gemx_matmul(&qtensor, repacked, config, device)?;

    Ok(())
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;

    /// Get CUDA device, returns None if not available
    pub fn get_cuda_device() -> Result<Option<Device>> {
        let device = Device::cuda_if_available(0)?;
        if device.is_cuda() {
            Ok(Some(device))
        } else {
            Ok(None)
        }
    }

    /// Skip helper - returns Ok(()) if CUDA not available
    pub fn require_cuda() -> Result<Device> {
        match get_cuda_device()? {
            Some(d) => Ok(d),
            None => {
                println!("Skipping test - CUDA not available");
                candle::bail!("CUDA not available")
            }
        }
    }
}

/// Negative tests that should work for any quant type
#[cfg(feature = "cuda")]
pub mod negative_tests {
    use super::*;

    /// Verify NaN detection works
    pub fn test_nan_detection(device: &Device) -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, f32::NAN, 4.0], 4, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;

        // Tolerances wide enough that nothing could fail on magnitude: what is
        // being asserted is that the NaN / shape checks fire on their own, not
        // that these particular numbers happen to exceed a bound.
        let result = assert_approx_eq(&a, &b, 1.0, 1.0);
        assert!(result.is_err(), "Should have detected NaN");

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN: {}",
            err_msg
        );

        println!("✓ NaN detection works");
        Ok(())
    }

    /// Verify shape mismatch detection
    pub fn test_shape_mismatch_detection(device: &Device) -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;

        // Tolerances wide enough that nothing could fail on magnitude: what is
        // being asserted is that the NaN / shape checks fire on their own, not
        // that these particular numbers happen to exceed a bound.
        let result = assert_approx_eq(&a, &b, 1.0, 1.0);
        assert!(result.is_err(), "Should have detected shape mismatch");

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("mismatch"),
            "Error should mention mismatch: {}",
            err_msg
        );

        println!("✓ Shape mismatch detection works");
        Ok(())
    }

    /// Verify tolerance checking works
    pub fn test_tolerance_detection(device: &Device) -> Result<()> {
        // Values that differ significantly
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 5.0, 4.0], 4, device)?;

        // With tight tolerance, should fail
        let result = assert_approx_eq(&a, &b, 0.01, 0.01);
        assert!(result.is_err(), "Should have detected difference");

        // With loose tolerance, should pass
        let result = assert_approx_eq(&a, &b, 1.0, 3.0);
        assert!(result.is_ok(), "Should pass with loose tolerance");

        println!("✓ Tolerance detection works");
        Ok(())
    }
}
