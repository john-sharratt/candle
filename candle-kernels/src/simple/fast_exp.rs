//! FFI bindings for fast_exp batch operations
//!
//! Provides a unified interface to dispatch fast exponential and activation
//! operations based on mode, precision, and data type enums.
//!
//! # Example
//! ```rust,ignore
//! use candle_kernels::simple::fast_exp::{run_fast_exp_batch, FastExpMode, FastExpPrecision, FastExpDType};
//!
//! unsafe {
//!     run_fast_exp_batch(
//!         FastExpMode::Softmax as i32,
//!         FastExpPrecision::High as i32,
//!         FastExpDType::F32 as i32,
//!         input_ptr,
//!         output_ptr,
//!         num_elements,
//!     );
//! }
//! ```

use core::ffi::c_void;

/// Mode for fast_exp computation
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastExpMode {
    /// Generic mode: safe for any input, full clamping [-88, 88]
    Generic = 0,
    /// Softmax mode: optimized for attention (assumes x <= 0, saves 1 op)
    Softmax = 1,
}

/// Precision level for fast_exp computation
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastExpPrecision {
    /// Cubic polynomial, ~0.009% max error (best for f32)
    High = 0,
    /// Quadratic polynomial, ~0.08% max error (good for f16)
    Medium = 1,
    /// Linear polynomial, ~1.5% max error (sufficient for bf16)
    Low = 2,
}

/// Data type for fast_exp operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastExpDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

/// Activation function type
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastActivation {
    /// sigmoid(x) = 1 / (1 + exp(-x))
    Sigmoid = 0,
    /// SiLU(x) = x * sigmoid(x)
    SiLU = 1,
    /// GELU(x) ≈ x * sigmoid(1.702 * x)
    GELU = 2,
}

impl FastExpPrecision {
    /// Returns the default precision for a given dtype
    pub fn default_for_dtype(dtype: FastExpDType) -> Self {
        match dtype {
            FastExpDType::F32 => FastExpPrecision::High,
            FastExpDType::F16 => FastExpPrecision::Medium,
            FastExpDType::BF16 => FastExpPrecision::Low,
        }
    }

    /// Returns the approximate max relative error for this precision level
    pub fn max_error_percent(&self) -> f32 {
        match self {
            FastExpPrecision::High => 0.009,
            FastExpPrecision::Medium => 0.08,
            FastExpPrecision::Low => 1.5,
        }
    }
}

extern "C" {
    /// Dispatches to the appropriate fast_exp batch kernel.
    ///
    /// # Parameters
    /// - `mode`: 0=Generic (safe for any input), 1=Softmax (assumes x <= 0)
    /// - `precision`: 0=High (cubic), 1=Medium (quadratic), 2=Low (linear)
    /// - `dtype`: 0=f32, 1=f16, 2=bf16
    /// - `inp`: Input tensor pointer
    /// - `out`: Output tensor pointer
    /// - `numel`: Number of elements
    ///
    /// # Supported Combinations
    /// - f32: All modes (Generic, Softmax) and all precisions (High, Medium, Low)
    /// - f16: Softmax mode with High or Medium precision
    /// - bf16: Softmax mode with Low precision only
    pub fn run_fast_exp_batch(
        mode: i32,
        precision: i32,
        dtype: i32,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
    );

    /// Dispatches to the appropriate fast activation kernel.
    ///
    /// # Parameters
    /// - `op`: 0=sigmoid, 1=silu, 2=gelu
    /// - `dtype`: 0=f32, 1=f16, 2=bf16
    /// - `inp`: Input tensor pointer
    /// - `out`: Output tensor pointer
    /// - `numel`: Number of elements
    ///
    /// # Supported Combinations
    /// - f32: All activations (sigmoid, silu, gelu)
    /// - f16: sigmoid only
    /// - bf16: sigmoid only
    pub fn run_fast_activation_batch(
        op: i32,
        dtype: i32,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
    );

    /// Reference exponential using hardware __expf for testing comparison.
    /// Only supports f32.
    pub fn run_reference_exp_batch(inp: *const f32, out: *mut f32, numel: usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_enum_values() {
        assert_eq!(FastExpMode::Generic as i32, 0);
        assert_eq!(FastExpMode::Softmax as i32, 1);
    }

    #[test]
    fn test_precision_enum_values() {
        assert_eq!(FastExpPrecision::High as i32, 0);
        assert_eq!(FastExpPrecision::Medium as i32, 1);
        assert_eq!(FastExpPrecision::Low as i32, 2);
    }

    #[test]
    fn test_dtype_enum_values() {
        assert_eq!(FastExpDType::F32 as i32, 0);
        assert_eq!(FastExpDType::F16 as i32, 1);
        assert_eq!(FastExpDType::BF16 as i32, 2);
    }

    #[test]
    fn test_activation_enum_values() {
        assert_eq!(FastActivation::Sigmoid as i32, 0);
        assert_eq!(FastActivation::SiLU as i32, 1);
        assert_eq!(FastActivation::GELU as i32, 2);
    }

    #[test]
    fn test_default_precision() {
        assert_eq!(
            FastExpPrecision::default_for_dtype(FastExpDType::F32),
            FastExpPrecision::High
        );
        assert_eq!(
            FastExpPrecision::default_for_dtype(FastExpDType::F16),
            FastExpPrecision::Medium
        );
        assert_eq!(
            FastExpPrecision::default_for_dtype(FastExpDType::BF16),
            FastExpPrecision::Low
        );
    }

    #[test]
    fn test_max_error() {
        assert!(FastExpPrecision::High.max_error_percent() < 0.01);
        assert!(FastExpPrecision::Medium.max_error_percent() < 0.1);
        assert!(FastExpPrecision::Low.max_error_percent() < 2.0);
    }
}
