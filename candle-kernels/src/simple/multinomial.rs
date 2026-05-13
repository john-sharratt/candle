//! FFI bindings for multinomial operation dispatcher
//!
//! Provides a unified interface to dispatch multinomial sampling operations
//! based on data type enum. Supports full top-k and top-p filtering.

use core::ffi::c_void;

// =============================================================================
// Data type enum for multinomial operations
// =============================================================================

/// Data type enum for multinomial operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultinomialDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
}

impl MultinomialDType {
    /// Returns the size in bytes of this dtype
    pub fn size_in_bytes(&self) -> usize {
        match self {
            MultinomialDType::F32 => 4,
            MultinomialDType::F64 => 8,
            MultinomialDType::F16 => 2,
            MultinomialDType::BF16 => 2,
        }
    }

    /// Returns true if this dtype requires CUDA architecture >= 8.0
    pub fn requires_ampere(&self) -> bool {
        matches!(self, MultinomialDType::BF16)
    }
}

// =============================================================================
// Default parameters
// =============================================================================

/// Default number of threads for parallel reductions
pub const DEFAULT_NUM_THREADS: i32 = 256;

/// Calculate shared memory size for given number of threads
#[inline]
pub fn calculate_shared_mem_size(num_threads: i32) -> usize {
    // Shared memory is used for parallel max and sum reductions
    // Each thread needs sizeof(float) = 4 bytes
    (num_threads as usize) * 4
}

/// Calculate workspace size for multinomial sampling
#[inline]
pub fn calculate_workspace_size(vocab_size: usize) -> usize {
    // ProbIndex struct is 8 bytes (4 bytes float prob + 4 bytes uint32 index)
    vocab_size * 8
}

// =============================================================================
// Dispatcher function bindings
// =============================================================================

extern "C" {
    /// Dispatches to the appropriate multinomial sampling kernel based on dtype.
    ///
    /// GPU-accelerated multinomial sampling with full top-k and top-p support.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see MultinomialDType enum values)
    /// - `logits`: Input logits array
    /// - `output`: Output sampled index (single uint32_t)
    /// - `workspace`: Workspace for intermediate computations (size: vocab_size * 8 bytes)
    /// - `vocab_size`: Size of the vocabulary
    /// - `temperature`: Temperature for sampling (higher = more random)
    /// - `top_k`: Top-k filtering (0 = disabled)
    /// - `top_p`: Top-p (nucleus) filtering (1.0 = disabled)
    /// - `seed`: Random seed for sampling
    /// - `num_threads`: Number of threads for parallel reductions
    /// - `shared_mem_size`: Shared memory size in bytes
    /// - `stream`: CUDA stream (use 0 for default stream)
    pub fn run_multinomial(
        dtype: i32,
        logits: *const c_void,
        output: *mut u32,
        workspace: *mut f32,
        vocab_size: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        num_threads: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    /// Simple multinomial sampling without top-k/top-p filtering.
    ///
    /// Convenience function that calls run_multinomial with top_k=0 and top_p=1.0.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see MultinomialDType enum values)
    /// - `logits`: Input logits array
    /// - `output`: Output sampled index (single uint32_t)
    /// - `workspace`: Workspace for intermediate computations
    /// - `vocab_size`: Size of the vocabulary
    /// - `temperature`: Temperature for sampling
    /// - `seed`: Random seed for sampling
    /// - `num_threads`: Number of threads for parallel reductions
    /// - `shared_mem_size`: Shared memory size in bytes
    /// - `stream`: CUDA stream (use 0 for default stream)
    pub fn run_multinomial_simple(
        dtype: i32,
        logits: *const c_void,
        output: *mut u32,
        workspace: *mut f32,
        vocab_size: usize,
        temperature: f32,
        seed: u64,
        num_threads: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    /// Returns the required workspace size in bytes for multinomial sampling.
    pub fn get_multinomial_workspace_size(vocab_size: usize) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multinomial_dtype_enum_values() {
        // Verify enum values match what the CUDA dispatcher expects
        assert_eq!(MultinomialDType::F32 as i32, 0);
        assert_eq!(MultinomialDType::F64 as i32, 1);
        assert_eq!(MultinomialDType::F16 as i32, 2);
        assert_eq!(MultinomialDType::BF16 as i32, 3);
    }

    #[test]
    fn test_multinomial_dtype_sizes() {
        assert_eq!(MultinomialDType::F32.size_in_bytes(), 4);
        assert_eq!(MultinomialDType::F64.size_in_bytes(), 8);
        assert_eq!(MultinomialDType::F16.size_in_bytes(), 2);
        assert_eq!(MultinomialDType::BF16.size_in_bytes(), 2);
    }

    #[test]
    fn test_workspace_size_calculation() {
        assert_eq!(calculate_workspace_size(1000), 8000);
        assert_eq!(calculate_workspace_size(50000), 400000);
    }

    #[test]
    fn test_shared_mem_size_calculation() {
        assert_eq!(calculate_shared_mem_size(256), 1024);
        assert_eq!(calculate_shared_mem_size(512), 2048);
    }
}
