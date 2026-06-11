//! FFI bindings for paged decode attention kernels

use core::ffi::c_void;

extern "C" {
    // ========================================================================
    // Per-dtype dispatchers — persistent slot buffer API
    // Takes a single `headers_ptr` pointing to SlotHeader[num_active_slots] on GPU.
    // ========================================================================

    pub fn run_paged_decode_fp16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        num_active_slots: i32,
        n_q_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        softmax_scale: f32,
        k_new: *const c_void,
        v_new: *const c_void,
        rope_cs: *const f32,
        rope_interleaved: i32,
        stream: *mut c_void,
    );

    pub fn run_paged_decode_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        num_active_slots: i32,
        n_q_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        softmax_scale: f32,
        k_new: *const c_void,
        v_new: *const c_void,
        rope_cs: *const f32,
        rope_interleaved: i32,
        stream: *mut c_void,
    );

    /// Regression-test entry for the int8 m16n8k32 MMA fragment loaders.
    /// Computes `C[16][8] = A[16][32] · B[8][32]ᵀ` (row-major int8 `A`/`B`, int32
    /// `C`) via `load_a_frag_m16k32` / `load_b_frag_n8k32` / `mma_int8_m16n8k32`
    /// — the exact path the int8 decode QK dot uses. A fragment-layout
    /// regression makes `C` disagree with a CPU reference. Single warp; test-only.
    pub fn mma_int8_m16n8k32_test(
        a: *const i8,
        b: *const i8,
        c: *mut i32,
        stream: *mut c_void,
    );
}
