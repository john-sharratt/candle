// FFI binding for the R16 KV gather kernel.
//
// Gathers K, Q-capture, and V data from paged R16/F16 arenas into a single
// combined F16 output tensor with one kernel launch + one DtoH copy.
//
// Grid:  `n_warps` blocks — one per (chunk_block_idx, kv_head_idx, palette_idx).
// Block: CHUNK_SIZE (32) threads, one per token.  One block = one hardware warp.
//
// Output layout (combined buffer, section_stride = n_warps × CHUNK_SIZE × sub_head_dim):
//   out_kqv[0               .. section_stride)   = K values
//   out_kqv[section_stride  .. 2×section_stride) = Q values (live-captured queries)
//   out_kqv[2×section_stride .. 3×section_stride) = V values
//
// Within each section, storage is D-MAJOR per warp (warp_id, d, token):
//   index = warp_id × CHUNK_SIZE × sub_head_dim + d × CHUNK_SIZE + token
//
// D-major was chosen so all 32 threads write consecutive 2-byte addresses for
// each fixed d → single 64-byte coalesced store per inner-loop iteration.
//
// The caller (backing.rs gather_r16_kv_probe) transposes d-major → token-major
// during the F16→F32 conversion pass to match the consumer contract expected
// by r16_block_to_turn_signatures.

use std::ffi::c_void;

extern "C" {
    // k_ptrs:       device i64[n_warps] — resolved K-chunk base addresses (R16 format)
    // v_ptrs:       device i64[n_warps] — resolved V-chunk base addresses (float F16)
    // out_kqv:      device half*        — combined [3 × n_warps × CHUNK_SIZE × sub_head_dim]
    //               layout: d-major per warp; see file header for section offsets
    // n_warps:      n_r16_blocks × n_kv_head × N_PALETTE
    // sub_head_dim: head_dim / N_PALETTE
    // stream:       cudaStream_t
    pub fn run_gather_r16_kv_f16(
        k_ptrs: *const i64,
        v_ptrs: *const i64,
        out_kqv: *mut c_void,
        n_warps: i32,
        sub_head_dim: i32,
        stream: *mut c_void,
    );
}
