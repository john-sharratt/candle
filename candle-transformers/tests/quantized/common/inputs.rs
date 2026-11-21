//! Builds the deterministic weights and activations the quantised matmul tests
//! feed through the kernels.
//!
//! Weights are Xavier-scaled so the outputs stay near 1, where BF16 precision is
//! adequate. Inputs lie in [-0.5, 0.5], which every target dtype represents,
//! including FP8, and each batch element gets its own pattern so that corruption
//! across batches shows. `compute_baseline` returns the dequantised weights
//! through an ordinary F32 matmul, so the reference adds no conversion error of
//! its own.

use super::*;

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
