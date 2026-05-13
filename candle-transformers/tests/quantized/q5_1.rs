//! Q5_1 quantized matmul tests.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use candle::Result;
    use candle::quantized::GgmlDType;
    use crate::quantized::common::{self, QuantTestConfig};
    use crate::quantized::common::cuda::require_cuda;

    fn q5_1_config() -> QuantTestConfig {
        QuantTestConfig::new("Q5_1", GgmlDType::Q5_1)
    }
    
    /// Test Q5_1 with all supported input dtypes
    #[test]
    fn test_q5_1_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        
        common::run_all_dtype_tests(&q5_1_config(), &device)?;
        Ok(())
    }
    
    /// Verify Q5_1 produces sane results
    #[test]
    fn test_q5_1_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        
        common::verify_result_sanity(&q5_1_config(), &device)?;
        Ok(())
    }

    /// Test Q5_1 repacking (K/128 format with embedded scales)
    #[test]
    fn test_q5_1_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q5_1_config(), &device)?;
        Ok(())
    }
}
