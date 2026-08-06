// FFI binding for the format-agnostic row gather.
//
// Copies indexed fixed-size rows from a row-major source into a contiguous
// destination. The source may be device memory or CUDA-registered host memory
// (`cuMemHostRegister` with `DEVICEMAP`), so a tensor left in the GGUF mmap can
// be read by the GPU over PCIe without occupying VRAM.
//
// Indices are read on the device, which is the point: the token ids are already
// a device tensor when the embedding is needed, and reading them back to gather
// on the CPU would synchronise the wave at the start of the forward.
//
// The kernel is format-blind. Every GGML type stores a row as a whole number of
// blocks, so `row_bytes = (ncols / block_size) * type_size` and the gather is
// the same contiguous byte copy for all of them; dequantization runs afterwards
// over the gathered buffer using the existing per-format kernels.

use std::ffi::c_void;

extern "C" {
    // src:        base address of the row-major source. Device memory, or host
    //             memory registered with `CU_MEMHOSTREGISTER_DEVICEMAP`.
    // indices:    device u32[n_rows] — source row index per output row. Ids at
    //             or beyond `n_src_rows` produce a zero row.
    // dst:        device buffer, `n_rows * row_bytes` bytes, written contiguously.
    // row_bytes:  bytes per row, identical for source and destination.
    // n_src_rows: rows in the source table, for the bounds check.
    // n_rows:     number of rows to gather.
    // stream:     cudaStream_t.
    pub fn run_gather_rows_bytes(
        src: *const c_void,
        indices: *const u32,
        dst: *mut c_void,
        row_bytes: i64,
        n_src_rows: i64,
        n_rows: i32,
        stream: *mut c_void,
    );
}
