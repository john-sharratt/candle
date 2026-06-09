//! FFI bindings for the legacy V2 paged-decode attention kernel.
//!
//! Superseded as the default by the INT8 decode kernel (`paged_decode`).
//! Retained as the A/B regression reference and for head_dim=256.

use core::ffi::c_void;

extern "C" {
    pub fn run_paged_decode_legacy_fp16(
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

    pub fn run_paged_decode_legacy_bf16(
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
