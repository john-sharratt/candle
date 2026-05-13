//! Q5_K quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q5k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q5_K", GgmlDType::Q5_K)
    }

    /// Test Q5_K with all supported input dtypes
    #[test]
    fn test_q5_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q5k_config(), &device)?;
        Ok(())
    }

    /// Verify Q5_K produces sane results
    #[test]
    fn test_q5_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q5k_config(), &device)?;
        Ok(())
    }
    /// Test Q5_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q5_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q5k_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q5_k_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q5k_config(), &device)?;
        Ok(())
    }
}
