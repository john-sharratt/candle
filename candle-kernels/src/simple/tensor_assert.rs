// FFI binding for the in-place tensor finiteness/range assert kernel.
//
// See `simple/tensor_assert.cu` for the authoritative contract: the `AssertSlot`
// layout, the monotonic float-key encoding used for min/max, and the ordering
// ticket that lets a drain sort bad slots by when they actually went bad.
//
// `src`, `slot` and `seq_counter` are device pointers on `stream`; `dims` and
// `strides` are HOST pointers, read during the launch and copied by value into
// the kernel's parameter block, so a strided assert allocates no device memory.
// Passing `num_dims == 0` selects the contiguous fast path and ignores both.

use std::ffi::c_void;

/// Maximum rank the strided path indexes, mirrored from `ASSERT_MAX_DIMS` in
/// the `.cu`. The Rust wrapper validates against THIS constant so its check can
/// never drift from the launcher's silent-return guard — a drifted wrapper
/// would skip the launch and leave the slot reading clean for a tensor that was
/// never examined.
pub const MAX_DIMS: usize = 8;

/// dtype selector, mirrored from the `ASSERT_DT_*` codes in the `.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AssertDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F64 = 3,
    U8 = 4,
    U32 = 5,
    I64 = 6,
    F8E4M3 = 7,
    /// Not a candle `DType`. Kernel workspaces — tile tables, permutations,
    /// expert ids — are raw `i32` device slices, and an out-of-range index in
    /// one is exactly the corruption that surfaces downstream as an implausible
    /// magnitude rather than as a NaN.
    I32 = 8,
}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_tensor_assert(
        src: *const c_void,
        dtype: i32,
        elem_count: i64,
        num_dims: i32,
        dims: *const i64,
        strides: *const i64,
        slot: *mut c_void,
        seq_counter: *mut c_void,
        stream: *mut c_void,
    );

    pub fn run_tensor_assert_reset(slots: *mut c_void, n_slots: i32, stream: *mut c_void);
}
