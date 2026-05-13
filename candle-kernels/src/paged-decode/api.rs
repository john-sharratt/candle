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
}
