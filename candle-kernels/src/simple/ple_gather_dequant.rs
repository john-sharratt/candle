// FFI binding for the PLE row dequant kernel.
//
// `run_ple_dequant_q8` widens gathered Q8_0 table records (170 bytes per
// 160-wide row) to F32 on the device — the receive half of the quantized
// PLE transfer; see `simple/ple_gather_dequant.cu`. Parity with the CPU
// Q8_0 dequant is pinned by `qwen4exp::ple_cache` tests.
//
//   records: device u8[n_rows * 170]
//   out:     device f32[n_rows * 160]
//   stream:  cudaStream_t

use std::ffi::c_void;

extern "C" {
    pub fn run_ple_dequant_q8(records: *const u8, out: *mut f32, n_rows: i32, stream: *mut c_void);
}
