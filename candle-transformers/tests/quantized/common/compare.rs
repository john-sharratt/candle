//! Compares a result tensor with its baseline and reports the error statistics
//! and whether the tolerance was exceeded.
//!
//! `compare_tensors` gathers the largest absolute and relative error, counts the
//! near-zero baseline values, and checks for NaN. The relative error is taken
//! only where the baseline is large enough, because a relative error against a
//! tiny baseline value is meaningless. `assert_approx_eq` and `check_tolerance`
//! fail only when both the relative and the absolute bound are exceeded.

use super::*;

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

/// Check if tensors exceed tolerance (returns true if exceeded)
#[cfg(feature = "cuda")]
pub(super) fn check_tolerance(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> bool {
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
