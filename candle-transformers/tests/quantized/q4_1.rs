//! Q4_1 quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q4_1_config() -> QuantTestConfig {
        QuantTestConfig::new("Q4_1", GgmlDType::Q4_1)
    }

    /// Test Q4_1 with all supported input dtypes
    #[test]
    fn test_q4_1_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q4_1_config(), &device)?;
        Ok(())
    }

    /// Verify Q4_1 produces sane results
    #[test]
    fn test_q4_1_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q4_1_config(), &device)?;
        Ok(())
    }

    /// Test Q4_1 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q4_1_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q4_1_config(), &device)?;
        Ok(())
    }
}
