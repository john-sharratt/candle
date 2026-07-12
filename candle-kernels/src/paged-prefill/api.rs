//! FFI bindings for paged prefill attention kernels

use core::ffi::c_void;

extern "C" {
    // ========================================================================
    // INT8 prefix-attention prefill (docs/archived/prefill_optimization.md): GQA-packed
    // M, slice-aligned tiles, int8 m16n8k32 QK/PV directly over the quantized
    // arena. q_dtype: 1=F16, 2=BF16 (hard error otherwise).
    // ========================================================================
    pub fn run_paged_prefill_int8(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        headers_ptr: *const u8,
        cu_seqlens_q: *const u32,
        q_lens: *const u32,
        kv_lens: *const u32,
        o_ptr: *mut c_void,
        total_q: i32,
        batch_size: i32,
        n_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        max_q_len: i32,
        softmax_scale: f32,
        q_dtype: i32,
        rope_offsets: *const u32,
        rope_cs: *const f32,
        rope_interleaved: i32,
        stream: *mut c_void,
    );

}
