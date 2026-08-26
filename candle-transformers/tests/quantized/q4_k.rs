//! Q4_K quantized matmul tests.
//!
//! Tests that the GEMX tensor-core kernel (via the wrapper) produces
//! results matching the baseline dequantize+matmul path.
//!
//! Q4_K is the primary validation target because it has F32 kernel support,
//! allowing us to compare kernel output against baseline with matching precision.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, get_tolerance_for, negative_tests, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::{DType, Result};

    fn q4k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q4_K", GgmlDType::Q4_K)
    }

    /// Test Q4_K with F16 and BF16 input dtypes
    #[test]
    fn test_q4_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q4k_config(), &device)?;
        Ok(())
    }

    /// HIGH-PRECISION VALIDATION: Test Q4_K with F32 input/output.
    /// This is the gold standard test - kernel output should exactly match
    /// baseline (dequantize + F32 matmul) since both use F32 arithmetic.
    #[test]
    fn test_q4_k_f32_validation() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        let config = q4k_config();
        let batch = 1;
        let seq = 4;
        let (rtol, atol) = get_tolerance_for(config.dtype, DType::F32);

        println!(
            "Testing {} with F32 (high-precision validation)",
            config.name
        );
        println!("  Using tight tolerance: rtol={}, atol={}", rtol, atol);

        common::run_dtype_test(&config, DType::F32, batch, seq, &device)?;
        Ok(())
    }

    /// Verify Q4_K produces sane results
    #[test]
    fn test_q4_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q4k_config(), &device)?;
        Ok(())
    }

    /// Test Q4_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q4_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q4k_config(), &device)?;
        Ok(())
    }

    /// Negative test: NaN detection
    #[test]
    fn test_nan_detection() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        negative_tests::test_nan_detection(&device)?;
        Ok(())
    }

    /// Negative test: shape mismatch detection
    #[test]
    fn test_shape_mismatch_detection() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        negative_tests::test_shape_mismatch_detection(&device)?;
        Ok(())
    }

    /// Negative test: tolerance detection
    #[test]
    fn test_tolerance_detection() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        negative_tests::test_tolerance_detection(&device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q4_k_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q4k_config(), &device)?;
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
mod cpu_tests {
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device, Module, Result};
    use candle_transformers::models::quantized_matmul::QMatMul as QMatMulWrapper;

    fn q4k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q4_K", GgmlDType::Q4_K)
    }

    /// On CPU, the wrapper should fall back to standard GGML kernels
    #[test]
    fn test_q4_k_cpu_fallback() -> Result<()> {
        let device = Device::Cpu;
        let config = q4k_config();

        let weights_f32 = common::create_test_weights(config.nrows, config.ncols, &device)?;
        let qtensor = QTensor::quantize(&weights_f32, config.dtype)?;

        // On CPU, supports_gemx_repacking() returns false
        assert!(!qtensor.supports_gemx_repacking());

        // Can still create wrapper (uses dummy scales)
        let qmatmul = QMatMulWrapper::from_qtensor(qtensor)?;

        let input = common::create_test_input(1, 4, config.ncols, DType::F32, &device)?;
        let result = qmatmul.forward(&input)?;

        // Should get a result with correct shape
        assert_eq!(result.dims(), &[1, 4, config.nrows]);

        println!("✓ Q4_K CPU fallback works");
        Ok(())
    }
}
