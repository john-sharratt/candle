use core::ffi::{c_int, c_void};
use half::{bf16, f16};

extern "C" {
    pub(crate) fn run_paged_decode(
        q_ptr: *const c_void,
        arena_table: *const i64,  // Arena table: (num_arenas, 3) with [k_ptr, v_ptr, metadata] per row
        block_table: *const i64,
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
        q_dtype: i32,  // Q/O dtype: 0=F32, 1=F16, 2=BF16
        // Optional fused KV scatter (all null or all non-null)
        k_new: *const c_void,
        v_new: *const c_void,
        write_offsets: *const u32,
        // RoPE position offsets per batch element, shape [batch_size], dtype U32.
        rope_offsets: *const u32,
        // Per-dim inverse frequencies on device (rope_dim/2 floats). Must NOT be null.
        inv_freq: *const f32,
        // RoPE pairing: 0=non-interleaved, 1=interleaved.
        rope_interleaved: i32,
        // Per-block valid position counts [batch_size * max_blocks], dtype U16.
        // Kernel masks positions at or beyond this count to -inf.
        block_usage: *const u32,
        // Per-block canonical RoPE start positions [batch * max_blocks], dtype I32.
        // K is un-rotated; kernel applies RoPE at position + within + rope_offsets.
        chunk_rope_positions: *const i32,
    );
}
