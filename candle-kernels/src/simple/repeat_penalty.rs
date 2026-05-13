//! FFI bindings for repeat penalty operation dispatcher
//!
//! Provides a unified interface to dispatch repeat penalty operations
//! based on data type enum.

use core::ffi::c_void;

// =============================================================================
// Data type enum for repeat penalty operations
// =============================================================================

/// Data type enum for repeat penalty operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatPenaltyDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
}

impl RepeatPenaltyDType {
    /// Returns the size in bytes of this dtype
    pub fn size_in_bytes(&self) -> usize {
        match self {
            RepeatPenaltyDType::F32 => 4,
            RepeatPenaltyDType::F64 => 8,
            RepeatPenaltyDType::F16 => 2,
            RepeatPenaltyDType::BF16 => 2,
        }
    }

    /// Returns true if this dtype requires CUDA architecture >= 8.0
    pub fn requires_ampere(&self) -> bool {
        matches!(self, RepeatPenaltyDType::BF16)
    }
}

// =============================================================================
// Dispatcher function bindings
// =============================================================================

extern "C" {
    /// Dispatches to the appropriate repeat penalty kernel based on dtype.
    ///
    /// Applies repeat penalty to logits at specified token indices.
    /// For positive logits: divide by penalty (reduces probability)
    /// For negative/zero logits: multiply by penalty (reduces probability)
    ///
    /// This is used to discourage the model from repeating tokens that have
    /// already appeared in the generated sequence.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see RepeatPenaltyDType enum values)
    /// - `data`: Mutable logits array
    /// - `indices`: Token indices to penalize (previously generated tokens)
    /// - `num_indices`: Number of indices to penalize
    /// - `penalty`: Penalty value (typically > 1.0, e.g., 1.1 to 1.5)
    /// - `stream`: CUDA stream (use 0 for default stream)
    pub fn run_repeat_penalty(
        dtype: i32,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        penalty: f64,
        stream: *mut c_void,
    );

    /// Applies repeat penalty to multiple batches of logits.
    ///
    /// Each batch has its own set of indices to penalize.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see RepeatPenaltyDType enum values)
    /// - `data`: Mutable logits array, shape [batch_size, vocab_size]
    /// - `indices`: Token indices to penalize for each batch, shape [batch_size, max_indices]
    /// - `num_indices_per_batch`: Actual number of indices for each batch, shape [batch_size]
    /// - `batch_size`: Number of batches
    /// - `vocab_size`: Vocabulary size (stride between batches in data)
    /// - `max_indices`: Maximum number of indices per batch (stride between batches in indices)
    /// - `penalty`: Penalty value (typically > 1.0)
    /// - `stream`: CUDA stream (use 0 for default stream)
    pub fn run_repeat_penalty_batch(
        dtype: i32,
        data: *mut c_void,
        indices: *const u32,
        num_indices_per_batch: *const usize,
        batch_size: usize,
        vocab_size: usize,
        max_indices: usize,
        penalty: f64,
        stream: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_penalty_dtype_enum_values() {
        // Verify enum values match what the CUDA dispatcher expects
        assert_eq!(RepeatPenaltyDType::F32 as i32, 0);
        assert_eq!(RepeatPenaltyDType::F64 as i32, 1);
        assert_eq!(RepeatPenaltyDType::F16 as i32, 2);
        assert_eq!(RepeatPenaltyDType::BF16 as i32, 3);
    }

    #[test]
    fn test_repeat_penalty_dtype_sizes() {
        assert_eq!(RepeatPenaltyDType::F32.size_in_bytes(), 4);
        assert_eq!(RepeatPenaltyDType::F64.size_in_bytes(), 8);
        assert_eq!(RepeatPenaltyDType::F16.size_in_bytes(), 2);
        assert_eq!(RepeatPenaltyDType::BF16.size_in_bytes(), 2);
    }
}
