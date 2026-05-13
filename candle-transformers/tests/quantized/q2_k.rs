//! Q2_K quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q2k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q2_K", GgmlDType::Q2_K)
    }

    /// Test Q2_K with all supported input dtypes (BF16, F16)
    #[test]
    fn test_q2_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q2k_config(), &device)?;
        Ok(())
    }

    /// Verify Q2_K produces sane results (non-zero, reasonable magnitude)
    #[test]
    fn test_q2_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q2k_config(), &device)?;
        Ok(())
    }
    /// Test Q2_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q2_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q2k_config(), &device)?;
        Ok(())
    }
}
