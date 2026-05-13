//! FFI bindings for cast operation dispatcher (regular + in-place)
//!
//! Provides unified interfaces to dispatch cast operations based on
//! source and destination data type enums using 2D lookup tables.
//!
//! ## Regular Cast (run_cast)
//! Standard cast with separate input/output buffers.
//!
//! ## In-Place Cast (run_cast_mut)
//! Memory-efficient in-place conversion within a single buffer.
//! Supports two execution modes:
//! 1. Cooperative grid mode: Uses cudaLaunchCooperativeKernel for grid-wide
//!    synchronization, enabling full GPU parallelism with memory safety.
//! 2. Single-block fallback: Uses standard kernel launch for compatibility.

use core::ffi::c_void;

/// Data type enum for cast operations
///
/// The order matches the lookup table in cast_dispatcher.cu
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastDType {
    F32 = 0,
    F64 = 1,
    U8 = 2,
    U32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F8E4M3 = 7,
}

impl CastDType {
    /// Returns the number of supported dtypes
    pub const fn count() -> usize {
        8
    }

    /// Try to convert from an integer
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F64),
            2 => Some(Self::U8),
            3 => Some(Self::U32),
            4 => Some(Self::I64),
            5 => Some(Self::F16),
            6 => Some(Self::BF16),
            7 => Some(Self::F8E4M3),
            _ => None,
        }
    }

    /// Returns a human-readable name for the dtype
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F8E4M3 => "f8_e4m3",
        }
    }

    /// Returns the size in bytes for this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F8E4M3 => 1,
        }
    }
}

/// Execution mode for in-place cast operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastMutMode {
    /// Automatically choose: prefer cooperative with fallback to single-block
    Auto = 0,
    /// Force single-block execution (slower but always works)
    SingleBlock = 1,
    /// Force cooperative launch (faster but may fail on some GPUs)
    Cooperative = 2,
}

extern "C" {
    // =========================================================================
    // Regular Cast Functions
    // =========================================================================

    /// Dispatches to the appropriate cast kernel based on source and destination dtypes.
    ///
    /// Uses a 2D lookup table [src_dtype][dst_dtype] internally.
    ///
    /// # Parameters
    /// - `src_dtype`: Source data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `dst_dtype`: Destination data type (same encoding as src_dtype)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array (dims followed by strides)
    /// - `inp`: Input tensor data pointer
    /// - `out`: Output tensor data pointer
    pub fn run_cast(
        src_dtype: i32,
        dst_dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        inp: *const c_void,
        out: *mut c_void,
    );

    // =========================================================================
    // In-Place Cast Functions
    // =========================================================================

    /// Dispatches to the appropriate in-place cast kernel.
    ///
    /// Uses a 2D lookup table [src_dtype][dst_dtype] internally.
    /// Automatically uses cooperative launch if supported, falling back to single-block.
    ///
    /// # Parameters
    /// - `src_dtype`: Source data type
    /// - `dst_dtype`: Destination data type
    /// - `numel`: Number of elements
    /// - `buf`: Buffer pointer (must be large enough for both source and destination)
    ///
    /// # Safety
    /// The buffer must be large enough to hold `numel * max(src_size, dst_size)` bytes.
    pub fn run_cast_mut(
        src_dtype: i32,
        dst_dtype: i32,
        numel: usize,
        buf: *mut c_void,
    );

    /// Dispatches to the appropriate in-place cast kernel with explicit mode selection.
    ///
    /// # Parameters
    /// - `mode`: Execution mode (0=auto, 1=single-block, 2=cooperative)
    pub fn run_cast_mut_with_mode(
        src_dtype: i32,
        dst_dtype: i32,
        numel: usize,
        buf: *mut c_void,
        mode: i32,
    );

    /// Returns 1 if cooperative launch is supported on the current device, 0 otherwise.
    pub fn cast_mut_supports_cooperative() -> i32;

    /// Returns the optimal number of blocks for cooperative launch with the given element count.
    pub fn cast_mut_get_optimal_blocks(numel: usize) -> i32;
}

// =============================================================================
// Safe Wrappers for Regular Cast
// =============================================================================

/// Safe wrapper for the regular cast dispatcher
///
/// # Safety
/// The caller must ensure:
/// - `info` points to valid memory containing `num_dims` dimensions followed by `num_dims` strides
/// - `inp` points to valid memory of the source dtype with at least `numel` elements
/// - `out` points to valid memory of the destination dtype with at least `numel` elements
/// - The memory regions don't overlap (or aliasing is acceptable for the operation)
#[inline]
pub unsafe fn dispatch_cast(
    src_dtype: CastDType,
    dst_dtype: CastDType,
    numel: usize,
    num_dims: usize,
    info: *const usize,
    inp: *const c_void,
    out: *mut c_void,
) {
    run_cast(
        src_dtype as i32,
        dst_dtype as i32,
        numel,
        num_dims,
        info,
        inp,
        out,
    );
}

// =============================================================================
// Safe Wrappers for In-Place Cast
// =============================================================================

/// Safe wrapper for the in-place cast dispatcher (auto mode)
///
/// # Safety
/// The caller must ensure:
/// - `buf` points to valid memory with at least `numel * max(src_dtype.size_bytes(), dst_dtype.size_bytes())` bytes
/// - The buffer is properly aligned for both source and destination types
/// - No other operations are accessing the buffer during the cast
#[inline]
pub unsafe fn dispatch_cast_mut(
    src_dtype: CastDType,
    dst_dtype: CastDType,
    numel: usize,
    buf: *mut c_void,
) {
    run_cast_mut(src_dtype as i32, dst_dtype as i32, numel, buf);
}

/// Safe wrapper for the in-place cast dispatcher with explicit mode selection
///
/// # Safety
/// Same requirements as `dispatch_cast_mut`.
#[inline]
pub unsafe fn dispatch_cast_mut_with_mode(
    src_dtype: CastDType,
    dst_dtype: CastDType,
    numel: usize,
    buf: *mut c_void,
    mode: CastMutMode,
) {
    run_cast_mut_with_mode(
        src_dtype as i32,
        dst_dtype as i32,
        numel,
        buf,
        mode as i32,
    );
}

// =============================================================================
// Query Functions
// =============================================================================

/// Check if cooperative launch is supported on the current device
#[inline]
pub fn supports_cooperative() -> bool {
    unsafe { cast_mut_supports_cooperative() != 0 }
}

/// Get the optimal number of blocks for cooperative launch
#[inline]
pub fn optimal_blocks(numel: usize) -> i32 {
    unsafe { cast_mut_get_optimal_blocks(numel) }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Returns whether the given dtype conversion is expanding (dst > src)
#[inline]
pub fn is_expanding(src: CastDType, dst: CastDType) -> bool {
    dst.size_bytes() > src.size_bytes()
}

/// Returns whether the given dtype conversion is shrinking (dst < src)
#[inline]
pub fn is_shrinking(src: CastDType, dst: CastDType) -> bool {
    dst.size_bytes() < src.size_bytes()
}

/// Returns whether the given dtype conversion is same-size
#[inline]
pub fn is_same_size(src: CastDType, dst: CastDType) -> bool {
    dst.size_bytes() == src.size_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_values() {
        assert_eq!(CastDType::F32 as i32, 0);
        assert_eq!(CastDType::F64 as i32, 1);
        assert_eq!(CastDType::U8 as i32, 2);
        assert_eq!(CastDType::U32 as i32, 3);
        assert_eq!(CastDType::I64 as i32, 4);
        assert_eq!(CastDType::F16 as i32, 5);
        assert_eq!(CastDType::BF16 as i32, 6);
        assert_eq!(CastDType::F8E4M3 as i32, 7);
    }

    #[test]
    fn test_dtype_from_i32() {
        assert_eq!(CastDType::from_i32(0), Some(CastDType::F32));
        assert_eq!(CastDType::from_i32(7), Some(CastDType::F8E4M3));
        assert_eq!(CastDType::from_i32(8), None);
        assert_eq!(CastDType::from_i32(-1), None);
    }

    #[test]
    fn test_dtype_names() {
        assert_eq!(CastDType::F32.name(), "f32");
        assert_eq!(CastDType::BF16.name(), "bf16");
        assert_eq!(CastDType::F8E4M3.name(), "f8_e4m3");
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(CastDType::U8.size_bytes(), 1);
        assert_eq!(CastDType::F16.size_bytes(), 2);
        assert_eq!(CastDType::F32.size_bytes(), 4);
        assert_eq!(CastDType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_dtype_count() {
        assert_eq!(CastDType::count(), 8);
    }

    #[test]
    fn test_mode_values() {
        assert_eq!(CastMutMode::Auto as i32, 0);
        assert_eq!(CastMutMode::SingleBlock as i32, 1);
        assert_eq!(CastMutMode::Cooperative as i32, 2);
    }

    #[test]
    fn test_conversion_direction() {
        // Expanding: u8 -> f32
        assert!(is_expanding(CastDType::U8, CastDType::F32));
        assert!(!is_shrinking(CastDType::U8, CastDType::F32));

        // Shrinking: f64 -> f32
        assert!(is_shrinking(CastDType::F64, CastDType::F32));
        assert!(!is_expanding(CastDType::F64, CastDType::F32));

        // Same size: f32 -> u32
        assert!(is_same_size(CastDType::F32, CastDType::U32));
        assert!(!is_expanding(CastDType::F32, CastDType::U32));
        assert!(!is_shrinking(CastDType::F32, CastDType::U32));
    }
}
