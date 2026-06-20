//! FFI bindings for paged prefill attention kernels

use core::ffi::c_void;

extern "C" {
    // ========================================================================
    // Main dispatcher - q_dtype determines Q/K/V/O type for prefill
    // Delegates to per-dtype dispatchers (fp16, bf16, f32) which each
    // contain a switch(head_dim) over supported head dimensions.
    // ========================================================================
    pub fn run_paged_prefill_chunks(
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
        max_blocks: i32,
        softmax_scale: f32,
        q_dtype: i32, // Q/K/V/O dtype: 0=F32, 1=F16, 2=BF16
        has_prefix: i32,
        rope_offsets: *const u32,
        rope_cs: *const f32,
        rope_interleaved: i32,
        write_offset_shifts: *const u32,
        stream: *mut c_void,
    );
}
