//! Q8_K quantized matmul tests.
//!
//! Q8_K is an 8-bit K-quant format with 256-element super-blocks:
//! - float d (scale) shared across 256 elements
//! - int8_t qs[256] (quantized weights)
//! - int16_t bsums[16] (block sums for optimization)
//!
//! Dequant formula: value = d * qs[i]

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::quantized::common::cuda::require_cuda;
    use crate::quantized::common::{self, QuantTestConfig};
    use candle::quantized::GgmlDType;
    use candle::Result;

    fn q8_k_config() -> QuantTestConfig {
        QuantTestConfig::new("Q8_K", GgmlDType::Q8_K)
    }

    /// Test Q8_K with all supported input dtypes
    #[test]
    fn test_q8_k_all_dtypes() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_all_dtype_tests(&q8_k_config(), &device)?;
        Ok(())
    }

    /// Verify Q8_K produces sane results
    #[test]
    fn test_q8_k_result_sanity() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::verify_result_sanity(&q8_k_config(), &device)?;
        Ok(())
    }

    /// Test Q8_K repacking (K/128 format with embedded scales)
    #[test]
    fn test_q8_k_repacking() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::test_repacking(&q8_k_config(), &device)?;
        Ok(())
    }

    /// Advanced diagnostics: detailed flow tracing to identify matmul bugs
    #[test]
    fn test_q8_k_advanced_diagnostics() -> Result<()> {
        let device = match require_cuda() {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        common::run_advanced_diagnostics(&q8_k_config(), &device)?;
        Ok(())
    }
}
