//! Q8_1 quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q8_1_config() -> QuantTestConfig {
        QuantTestConfig::new("Q8_1", GgmlDType::Q8_1)
    }

    /// Test Q8_1 with all supported input dtypes
    #[test]
    fn test_q8_1_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q8_1_config(), &device)?;
        Ok(())
    }

    /// Verify Q8_1 produces sane results
    #[test]
    fn test_q8_1_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q8_1_config(), &device)?;
        Ok(())
    }

    /// Test Q8_1 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q8_1_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q8_1_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q8_1_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q8_1_config(), &device)?;
        Ok(())
    }
}
