//! FFI bindings for the fused QKV + paged-decode-attention v1 kernel.
//!
//! The single dispatch entry point switches over (shape × arch × dtype) to
//! the matching compiled launcher. Returns `cudaErrorNotSupported` (= 801)
//! for unmatched shapes; callers are expected to fall back to v2 in that
//! case.

use core::ffi::c_void;

extern "C" {
    /// v2-API-compatible dispatch for incremental INT8 MMA validation.
    ///
    /// Phase 1 passthrough: forwards to v2's `launch_paged_decode_attn`.
    /// Plugs into `PagedDecode` behind `CANDLE_FUSED_ATTN_V1=1`.
    pub fn fused_attn_v1_v2_compat_dispatch(
        q_dtype_tag: i32,        // 0 = F16, 1 = BF16
        head_dim: i32,
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        num_active_slots: i32,
        n_q_head: i32,
        n_kv_head: i32,
        softmax_scale: f32,
        k_new: *const c_void,
        v_new: *const c_void,
        rope_cs: *const f32,
        rope_interleaved: i32,
        stream_ptr: *mut c_void,
    ) -> i32;

    /// Single C-callable dispatch for the fused QKV + decode-attention kernel.
    ///
    /// Returns 0 on success, or a `cudaError_t` value on failure (notably
    /// `cudaErrorNotSupported` (801) when no compiled launcher matches the
    /// requested shape).
    pub fn fused_attn_v1_dispatch(
        head_dim: i32,
        n_q_head: i32,
        n_kv_head: i32,
        d_model: i32,
        rope_interleaved: i32,
        rope_style: i32,         // 0 = Full, 1 = Partial
        use_qk_norm: i32,
        use_sliding_window: i32,
        sm_version: i32,
        q_dtype_tag: i32,        // 0 = F16, 1 = BF16
        o_dtype_tag: i32,        // 0 = F16, 1 = BF16

        activations: *const c_void,
        w_qkv_q4: *const u8,
        w_qkv_scales: *const c_void,
        headers_ptr: *const u8,
        out: *mut c_void,
        num_active_slots: i32,
        softmax_scale: f32,
        rope_cs: *const f32,
        sliding_window_size: i32,
        stream_ptr: *mut c_void,
    ) -> i32;
}
