// FFI binding for the few-row F32 matmul. See `simple/f32_rows_matmul.cu`.

use std::ffi::c_void;

/// The most input rows one launch takes, mirroring `F32_ROWS_MATMUL_MAX_ROWS` in
/// `simple/f32_rows_matmul.cu`: one accumulator register per row per thread.
pub const F32_ROWS_MATMUL_MAX_ROWS: usize = 16;

extern "C" {
    /// `out[m, n] = Σ_k x[m, k] · w[n, k]` for `rows ≤ F32_ROWS_MATMUL_MAX_ROWS`.
    ///
    ///   x:   device f32 `[rows, k]`, rows `k` apart, 16-byte aligned
    ///   w:   device f32 `[n, k]`, rows `k` apart, 16-byte aligned
    ///   out: device f32 `[rows, n]`, every element written
    ///
    /// `k` must be a multiple of 4. Returns 0 on a launch, -1 for a row count
    /// outside `1..=F32_ROWS_MATMUL_MAX_ROWS` or an empty shape.
    pub fn run_f32_rows_matmul(
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
}
