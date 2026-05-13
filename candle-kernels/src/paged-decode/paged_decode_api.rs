//! FFI bindings for paged decode attention kernels

use core::ffi::{c_int, c_void};

extern "C" {
    /// Paged decode attention - q_dtype determines Q/O type, arena formats are per-head
    pub fn run_paged_decode(
        q_ptr: *const c_void,
        per_head_table: *const i64, // Per-head table: (num_arenas * n_kv_head, 7) per row
        chunk_meta: *const u32,
        head_gids: *const i64,      // Per-head GIDs: [batch*max_blocks*n_kv_head*2] interleaved K/V
        kv_lens: *const u32,
        o_ptr: *mut c_void,
        batch_size: i32,
        n_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        arena_chunks: i32,
        chunk_size: i32,
        max_blocks: i32,
        softmax_scale: f32,
        q_dtype: i32, // Q/O dtype: 0=F32, 1=F16, 2=BF16
        // Fused KV scatter (all null or all non-null)
        k_new: *const c_void,
        v_new: *const c_void,
        write_offsets: *const u32,
        rope_offsets: *const u32,
        inv_freq: *const f32,
        rope_interleaved: i32,
    );
}
