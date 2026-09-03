// FFI binding for the fused W4A16 → Q4_KO repack kernel.
//
// `run_w4a16_repack_q4ko` converts every expert of a compressed-tensors
// int4-g128 tensor into the lane-major Q4_KO chunk layout in one launch —
// pure byte permutation plus the (f16 scale, f16 min) stores; see
// `simple/w4a16_repack.cu`. Byte-identity with the CPU reference
// (`ko_quant::pack_q4_ko`) is pinned by the `w4a16_convert_bench` harness.
//
//   words:      device u32[n_experts * nrows * ncols / 8] (packed nibbles)
//   scales:     device u16[n_experts * nrows * ncols / 128] (bf16 bits)
//   out:        device u8[n_experts * (nrows/8) * (ncols/128) * 544]
//   violations: device i32[1], zeroed by the caller; incremented per scale
//               that is not f16-exact (the caller refuses on nonzero)
//   stream:     cudaStream_t

use std::ffi::c_void;

extern "C" {
    pub fn run_w4a16_repack_q4ko(
        words: *const u32,
        scales: *const u16,
        out: *mut u8,
        violations: *mut i32,
        n_experts: i32,
        nrows: i32,
        ncols: i32,
        stream: *mut c_void,
    );
}
