// FFI binding for the quantized token-embedding lookup; see
// `simple/embed_gather_dequant.cu`.
//
// Gathers rows of a block-quantized table resident in VRAM by a device index
// array, dequantizes them through the shared `dequantize_block` functions, and
// writes F32 to up to two destinations: `wide`, each row repeated `replicas`
// times back to back, and `narrow`, each row once. Either may be null.

use std::ffi::c_void;

/// Returned when the table's format has no unit in the kernel.
pub const EMBED_GATHER_UNSUPPORTED_FORMAT: i32 = -1;
/// Returned when `ncols` is not a whole number of the format's units.
pub const EMBED_GATHER_RAGGED_ROW: i32 = -2;

extern "C" {
    // table:      device base of the row-major quantized table.
    // qtype:      `GgmlDType as u32` (the `QTYPE_*` numbering).
    // ids:        device u32[n_rows]. Ids at or beyond `n_src_rows` write zeros.
    // wide:       device f32[n_rows * replicas * ncols], or null.
    // replicas:   copies of each row in `wide`; ignored when `wide` is null.
    // narrow:     device f32[n_rows * ncols], or null.
    // ncols:      elements per row.
    // n_src_rows: rows in the table, for the bounds check.
    // n_rows:     rows to gather.
    // stream:     cudaStream_t.
    //
    // Returns 0 on launch, or one of the negative codes above.
    pub fn run_embed_gather_dequant_f32(
        table: *const c_void,
        qtype: i32,
        ids: *const u32,
        wide: *mut f32,
        replicas: i32,
        narrow: *mut f32,
        ncols: i64,
        n_src_rows: i64,
        n_rows: i32,
        stream: *mut c_void,
    ) -> i32;
}
