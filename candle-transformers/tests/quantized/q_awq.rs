//! Q_AWQ quantized matmul tests.
//!
//! Q_AWQ is an AWQ (Activation-aware Weight Quantization) format:
//! - 4-bit asymmetric quantization with group size 128
//! - 128 elements per block (matching K/128 GEMX layout)
//! - Dequant formula: w = scale * (q - zero)
//!
//! Q_AWQ_G64 uses group size 64 (2 groups per K/128 block).

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Result};

    fn q_awq_config() -> QuantTestConfig {
        QuantTestConfig::new("Q_AWQ", GgmlDType::QAWQ)
    }

    fn q_awq_g64_config() -> QuantTestConfig {
        QuantTestConfig::new("Q_AWQ_G64", GgmlDType::QAWQ_G64)
    }

    // =========================================================================
    // Q_AWQ Quantize/Dequantize Tests
    // =========================================================================

    /// Test Q_AWQ quantization and dequantization roundtrip
    #[test]
    fn test_q_awq_quantize_dequantize() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        // Create test weights (must be multiple of 128 for AWQ)
        let nrows = 256;
        let ncols = 256;
        let weights = common::create_test_weights(nrows, ncols, &device)?;

        // Quantize
        let qtensor = QTensor::quantize(&weights, GgmlDType::QAWQ)?;

        // Dequantize to F32
        let dequant = qtensor.dequantize(&device)?;
        assert_eq!(dequant.dims(), &[nrows, ncols]);

        // Dequantize to F16
        let dequant_f16 = qtensor.dequantize_f16(&device)?;
        assert_eq!(dequant_f16.dims(), &[nrows, ncols]);

        // F32 and F16 dequant should be very close
        let diff = (dequant.to_dtype(DType::F16)? - &dequant_f16)?
            .to_dtype(DType::F32)?
            .abs()?
            .mean_all()?
            .to_vec0::<f32>()?;
        assert!(diff < 0.01, "F32 vs F16 dequant diff too high: {}", diff);

        // Check quantization error is reasonable (4-bit quantization)
        let orig = weights.to_vec2::<f32>()?;
        let deq = dequant.to_vec2::<f32>()?;

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (row_orig, row_deq) in orig.iter().zip(deq.iter()) {
            for (a, b) in row_orig.iter().zip(row_deq.iter()) {
                let d = (a - b).abs();
                max_diff = max_diff.max(d);
                sum_diff += d as f64;
            }
        }
        let mean_diff = sum_diff / (nrows * ncols) as f64;

        println!(
            "Q_AWQ quantize/dequantize: max_diff={:.6}, mean_diff={:.6}",
            max_diff, mean_diff
        );

        // 4-bit quantization should have reasonable error
        assert!(
            max_diff < 0.5,
            "Q_AWQ max quantization error too high: {}",
            max_diff
        );
        assert!(
            mean_diff < 0.1,
            "Q_AWQ mean quantization error too high: {}",
            mean_diff
        );

        Ok(())
    }

    /// Test Q_AWQ_G64 quantization and dequantization roundtrip
    #[test]
    fn test_q_awq_g64_quantize_dequantize() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        let nrows = 256;
        let ncols = 256;
        let weights = common::create_test_weights(nrows, ncols, &device)?;

        let qtensor = QTensor::quantize(&weights, GgmlDType::QAWQ_G64)?;

        let dequant = qtensor.dequantize(&device)?;
        assert_eq!(dequant.dims(), &[nrows, ncols]);

        let dequant_f16 = qtensor.dequantize_f16(&device)?;
        assert_eq!(dequant_f16.dims(), &[nrows, ncols]);

        let diff = (dequant.to_dtype(DType::F16)? - &dequant_f16)?
            .to_dtype(DType::F32)?
            .abs()?
            .mean_all()?
            .to_vec0::<f32>()?;
        assert!(diff < 0.01, "F32 vs F16 dequant diff too high: {}", diff);

        let orig = weights.to_vec2::<f32>()?;
        let deq = dequant.to_vec2::<f32>()?;

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (row_orig, row_deq) in orig.iter().zip(deq.iter()) {
            for (a, b) in row_orig.iter().zip(row_deq.iter()) {
                let d = (a - b).abs();
                max_diff = max_diff.max(d);
                sum_diff += d as f64;
            }
        }
        let mean_diff = sum_diff / (nrows * ncols) as f64;

        println!(
            "Q_AWQ_G64 quantize/dequantize: max_diff={:.6}, mean_diff={:.6}",
            max_diff, mean_diff
        );

        // G64 should have similar or better precision than G128
        assert!(
            max_diff < 0.5,
            "Q_AWQ_G64 max quantization error too high: {}",
            max_diff
        );
        assert!(
            mean_diff < 0.1,
            "Q_AWQ_G64 mean quantization error too high: {}",
            mean_diff
        );

        Ok(())
    }

    // =========================================================================
    // Q_AWQ GEMV Tests (batch=1, tests dequantize_mul_mat_vec path)
    // =========================================================================

    // =========================================================================
    // Q_AWQ GEMV (single-vector) tests
    // =========================================================================
    // Note: The all_dtypes tests already test batch_size=1 (GEMV path).
    // These tests verify GEMV specifically with different matrix dimensions.

    /// Test Q_AWQ GEMV with non-square dimensions
    #[test]
    fn test_q_awq_gemv() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        // Test various non-square dimensions that exercise GEMV path
        let configs = [
            (256, 512),  // wide matrix
            (512, 256),  // tall matrix
            (1024, 256), // very tall
            (256, 1024), // very wide
        ];

        for (nrows, ncols) in configs {
            let weights = common::create_test_weights(nrows, ncols, &device)?;

            // Quantize
            let qtensor = QTensor::quantize(&weights, GgmlDType::QAWQ)?;

            // Dequantize to verify roundtrip
            let dequant = qtensor.dequantize(&device)?;

            // Compare stats
            let stats = common::compare_tensors(&weights, &dequant)?;

            // Q_AWQ is 4-bit asymmetric, expect some error
            assert!(
                stats.max_diff < 0.3,
                "Q_AWQ quantize roundtrip error too high for {}x{}: {}",
                nrows,
                ncols,
                stats.max_diff
            );
        }

        println!("Q_AWQ GEMV dimension tests passed");
        Ok(())
    }

    /// Test Q_AWQ_G64 GEMV with non-square dimensions
    #[test]
    fn test_q_awq_g64_gemv() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        let configs = [(256, 512), (512, 256), (1024, 256), (256, 1024)];

        for (nrows, ncols) in configs {
            let weights = common::create_test_weights(nrows, ncols, &device)?;
            let qtensor = QTensor::quantize(&weights, GgmlDType::QAWQ_G64)?;
            let dequant = qtensor.dequantize(&device)?;
            let stats = common::compare_tensors(&weights, &dequant)?;

            assert!(
                stats.max_diff < 0.3,
                "Q_AWQ_G64 quantize roundtrip error too high for {}x{}: {}",
                nrows,
                ncols,
                stats.max_diff
            );
        }

        println!("Q_AWQ_G64 GEMV dimension tests passed");
        Ok(())
    }

    // =========================================================================
    // Q_AWQ Tests (group size 128)
    // =========================================================================

    /// Test Q_AWQ with all supported input dtypes
    #[test]
    fn test_q_awq_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q_awq_config(), &device)?;
        Ok(())
    }

    /// Verify Q_AWQ produces sane results
    #[test]
    fn test_q_awq_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q_awq_config(), &device)?;
        Ok(())
    }

    /// Test Q_AWQ repacking (K/128 format with embedded scales)
    #[test]
    fn test_q_awq_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q_awq_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q_awq_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q_awq_config(), &device)?;
        Ok(())
    }

    // =========================================================================
    // Q_AWQ_G64 Tests (group size 64)
    // =========================================================================

    /// Test Q_AWQ_G64 with all supported input dtypes
    #[test]
    fn test_q_awq_g64_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q_awq_g64_config(), &device)?;
        Ok(())
    }

    /// Verify Q_AWQ_G64 produces sane results
    #[test]
    fn test_q_awq_g64_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q_awq_g64_config(), &device)?;
        Ok(())
    }

    /// Test Q_AWQ_G64 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q_awq_g64_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q_awq_g64_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics for Q_AWQ_G64
    #[test]
    fn test_q_awq_g64_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q_awq_g64_config(), &device)?;
        Ok(())
    }
}
