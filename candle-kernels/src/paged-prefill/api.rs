//! FFI bindings for paged prefill attention kernels

use core::ffi::c_void;

use crate::rope::RopeRungsFfi;

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
        rungs: RopeRungsFfi,
        rope_interleaved: i32,
        stream: *mut c_void,
        // QSA selection, one row per PACKED QUERY (`cu_seqlens_q[b] + token`);
        // null for a full causal read. See `candle-kernels/src/qsa_select.cuh`.
        sel_entries: *const u32,
        sel_cnt: *const u32,
        sel_pages: *const u32,
        sel_page_win: *const u32,
        sel_stride: i32,
        sel_ratio: i32,
        // The pre-staged K/V (`src/paged-prefill/kv_stage.cuh`): null stages
        // nothing; otherwise every sequence with `q_len >= stage_min_q_len`
        // is staged into `stage_buf` (`stage_bytes` long; `stage_positions`
        // the planes' rows per head — the widest key window any launch
        // stages — and `stage_max_kv` the deepest sequence) one key window at
        // a time ahead of the attention kernel, which reads its columns from
        // there.
        stage_buf: *mut u8,
        stage_bytes: i64,
        stage_positions: i64,
        stage_min_q_len: i32,
        stage_max_kv: i32,
        // The cut (`PrefillCut` in `paged_prefill_int8_kernel.cuh`):
        // `n_groups` row groups of `group_blocks` grid-x blocks, group g
        // running `group_chunks[g]` key windows of `chunk_positions` — a HOST
        // array — and the online-softmax carry and resume table the windows
        // hand on, null when no group runs more than one window.
        group_chunks: *const u32,
        n_groups: i32,
        group_blocks: i32,
        chunk_positions: i32,
        carry: *mut f32,
        resume: *mut u32,
    );

    /// The pre-staging pass of [`run_paged_prefill_int8`] alone, into
    /// `stage_buf`. kv_dtype: 1=F16, 2=BF16 — the packed K/V's type.
    pub fn run_paged_prefill_kv_stage(
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        headers_ptr: *const u8,
        cu_seqlens_q: *const u32,
        q_lens: *const u32,
        kv_lens: *const u32,
        batch_size: i32,
        n_kv_head: i32,
        head_dim: i32,
        kv_dtype: i32,
        rungs: RopeRungsFfi,
        rope_interleaved: i32,
        stream: *mut c_void,
        stage_buf: *mut u8,
        stage_bytes: i64,
        stage_positions: i64,
        stage_min_q_len: i32,
        stage_max_kv: i32,
    );

}
