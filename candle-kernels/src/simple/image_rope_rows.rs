// FFI binding for the in-place image-row turn; see `simple/image_rope_rows.cu`.
//
// For each listed row of a dense `[n_rows, heads, dim]` Q or K, every head's
// rotary pair `f < pairs` — channels `(f, f + dim/2)` — is rotated by that
// row's `(cos, sin)[i, f]`, in place.

use std::ffi::c_void;

/// `dtype` codes.
pub const IMAGE_ROPE_F32: i32 = 0;
pub const IMAGE_ROPE_F16: i32 = 1;
pub const IMAGE_ROPE_BF16: i32 = 2;
/// Returned for a type with no instantiation.
pub const IMAGE_ROPE_UNSUPPORTED_DTYPE: i32 = -1;
/// Returned for an odd `dim`, or more pairs than half the head.
pub const IMAGE_ROPE_BAD_SHAPE: i32 = -2;

extern "C" {
    // dtype:    one of the codes above, `x`'s element type.
    // x:        device `[n_rows, heads, dim]`, dense, turned in place.
    // rows:     device u32[n_listed], each a row of `x`, none twice.
    // cos, sin: device f32[n_listed * pairs], row `i` for `rows[i]`.
    // stream:   cudaStream_t.
    //
    // Returns 0 on launch, or one of the negative codes above.
    pub fn run_image_rope_rows(
        dtype: i32,
        x: *mut c_void,
        rows: *const u32,
        cos: *const f32,
        sin: *const f32,
        n_listed: i32,
        heads: i32,
        dim: i32,
        pairs: i32,
        stream: *mut c_void,
    ) -> i32;
}
