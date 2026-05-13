//! Q3_K quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q3k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q3_K", GgmlDType::Q3_K)
    }

    /// Test Q3_K with all supported input dtypes
    #[test]
    fn test_q3_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q3k_config(), &device)?;
        Ok(())
    }

    /// Verify Q3_K produces sane results
    #[test]
    fn test_q3_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q3k_config(), &device)?;
        Ok(())
    }
    /// Test Q3_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q3_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q3k_config(), &device)?;
        Ok(())
    }
}
