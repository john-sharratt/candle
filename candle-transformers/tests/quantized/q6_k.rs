//! Q6_K quantized matmul tests.
//!
//! Q6_K is a 6-bit K-quant format used primarily for output.weight (lm_head).
//! It has 16 scales per 256-element super-block (one scale per 16 elements).

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q6k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q6_K", GgmlDType::Q6_K)
    }

    /// Test Q6_K with all supported input dtypes (BF16, F16)
    #[test]
    fn test_q6_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q6k_config(), &device)?;
        Ok(())
    }

    /// Verify Q6_K produces sane results (non-zero, reasonable magnitude)
    #[test]
    fn test_q6_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q6k_config(), &device)?;
        Ok(())
    }

    /// Test Q6_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q6_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q6k_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q6_k_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q6k_config(), &device)?;
        Ok(())
    }
}
