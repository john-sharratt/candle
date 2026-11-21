//! Holds the fixed tolerances that the quantised matmul tests compare against.
//!
//! The bounds are fixed, and the file records that raising them would hide
//! kernel bugs. The absolute bound is set per weight format at about 1.4 times
//! the largest error measured for that format, since one bound loose enough for
//! Q4_K would leave Q8_0 with several times its own error as slack. The error
//! follows the weight format rather than the activation dtype, because the
//! baseline quantises activations to Q8_1 whatever their dtype. FP8 activations
//! get an extra allowance for their own rounding, and the relative tolerance is
//! a secondary check.

use super::*;

// PRECISION TOLERANCES - FIXED FOR Q4_K AND ABOVE
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
