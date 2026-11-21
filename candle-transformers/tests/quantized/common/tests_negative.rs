//! Tests that the NaN, shape and tolerance checks fire on their own.
//!
//! The NaN and shape tests use tolerances wide enough that no magnitude could
//! trip them, so a failure can only come from the check under test. The
//! tolerance test uses values that differ, with a tight bound that should fail
//! and a loose bound that should pass.

use super::*;

/// Negative tests that should work for any quant type
#[cfg(feature = "cuda")]
pub mod negative_tests {
    use super::*;

    /// Verify NaN detection works
    pub fn test_nan_detection(device: &Device) -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, f32::NAN, 4.0], 4, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;

        // Tolerances wide enough that nothing could fail on magnitude: what is
        // being asserted is that the NaN / shape checks fire on their own, not
        // that these particular numbers happen to exceed a bound.
        let result = assert_approx_eq(&a, &b, 1.0, 1.0);
        assert!(result.is_err(), "Should have detected NaN");

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN: {}",
            err_msg
        );

        println!("✓ NaN detection works");
        Ok(())
    }

    /// Verify shape mismatch detection
    pub fn test_shape_mismatch_detection(device: &Device) -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;

        // Tolerances wide enough that nothing could fail on magnitude: what is
        // being asserted is that the NaN / shape checks fire on their own, not
        // that these particular numbers happen to exceed a bound.
        let result = assert_approx_eq(&a, &b, 1.0, 1.0);
        assert!(result.is_err(), "Should have detected shape mismatch");

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("mismatch"),
            "Error should mention mismatch: {}",
            err_msg
        );

        println!("✓ Shape mismatch detection works");
        Ok(())
    }

    /// Verify tolerance checking works
    pub fn test_tolerance_detection(device: &Device) -> Result<()> {
        // Values that differ significantly
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, device)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 5.0, 4.0], 4, device)?;

        // With tight tolerance, should fail
        let result = assert_approx_eq(&a, &b, 0.01, 0.01);
        assert!(result.is_err(), "Should have detected difference");

        // With loose tolerance, should pass
        let result = assert_approx_eq(&a, &b, 1.0, 3.0);
        assert!(result.is_ok(), "Should pass with loose tolerance");

        println!("✓ Tolerance detection works");
        Ok(())
    }
}
