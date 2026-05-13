//! Q8_0 quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q8_0_config() -> QuantTestConfig {
        QuantTestConfig::new("Q8_0", GgmlDType::Q8_0)
    }

    /// Test Q8_0 with all supported input dtypes
    #[test]
    fn test_q8_0_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q8_0_config(), &device)?;
        Ok(())
    }

    /// Verify Q8_0 produces sane results
    #[test]
    fn test_q8_0_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q8_0_config(), &device)?;
        Ok(())
    }

    /// Test Q8_0 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q8_0_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q8_0_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q8_0_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q8_0_config(), &device)?;
        Ok(())
    }
}
