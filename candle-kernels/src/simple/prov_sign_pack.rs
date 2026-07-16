// FFI binding for the provenance sign(Q) bit-pack kernel.
//
// Reads R16 Q (co-located in the K arena chunk at +64/dim-group), signs each
// dim, and packs sub_head_dim bits into a u32 per (warp, token) — all layers'
// sub-bands in ONE launch. Replaces the old per-layer f16 K/Q/V D2H + CPU sign
// pass; the host only D2H's the packed bits and XOR-folds them.
//
// Grid:  `n_warps` = n_layers × n_r16_blocks × n_kv_head × N_PALETTE
// Block: CHUNK_SIZE (32) threads, one per token.
//
// out[warp * CHUNK_SIZE + token]: u32, bit d set iff Q[token][d] >= 0
//   (d in 0..sub_head_dim; palette p → global head-dims [p*sub_head_dim ..]).

use std::ffi::c_void;

extern "C" {
    // q_ptrs:       device i64[n_warps] — resolved R16 K-chunk base addresses
    //               (Q is co-located at +64 within each 128-byte dim group)
    // out:          device u32*[n_warps * CHUNK_SIZE] — packed sign bits
    // n_warps:      n_layers × n_r16_blocks × n_kv_head × N_PALETTE
    // sub_head_dim: head_dim / N_PALETTE (must be <= 32; else no-op)
    // stream:       cudaStream_t
    pub fn run_prov_sign_pack(
        q_ptrs: *const i64,
        out: *mut c_void,
        n_warps: i32,
        sub_head_dim: i32,
        stream: *mut c_void,
    );
}
