//! Q4_0 quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q4_0_config() -> QuantTestConfig {
        QuantTestConfig::new("Q4_0", GgmlDType::Q4_0)
    }

    /// Test Q4_0 with all supported input dtypes
    #[test]
    fn test_q4_0_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        common::run_all_dtype_tests(&q4_0_config(), &device)
    }

    /// Verify Q4_0 produces sane results
    #[test]
    fn test_q4_0_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q4_0_config(), &device)?;
        Ok(())
    }

    /// Test Q4_0 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q4_0_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q4_0_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q4_0_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q4_0_config(), &device)?;
        Ok(())
    }

    /// Kernel component isolation diagnostics
    #[test]
    fn test_q4_0_kernel_isolation() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_kernel_isolation_diagnostics(&q4_0_config(), &device)?;
        Ok(())
    }
}
