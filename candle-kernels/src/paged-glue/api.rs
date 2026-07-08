//! FFI bindings for the paged-glue attention kernel.
//!
//! The glue kernel is a derivative of the paged-decode INT8 kernel for the
//! reprojection "glue" forward: a handful of query tokens (G per slot) attend a
//! long *quantized* prefix, write their own K/V, and attend earlier glue. It is
//! the decode kernel widened from one query token to `G` — it streams each
//! prefix chunk once (dequant-once) and reuses it across all `G x hpg` query
//! rows via the tensor-core M-dimension. See `docs/paged_glue_kernel.md`.

use core::ffi::c_void;

extern "C" {
    /// FP16 glue attention. Inputs are FLAT-packed in `cu_seqlens_q` order:
    ///   `q_ptr`   : glue Q  `[total_q, n_q_head, head_dim]`
    ///   `k_new`   : glue K  `[total_q, n_kv_head, head_dim]` (scattered to the
    ///               writer slices, un-rotated; re-RoPE'd at read)
    ///   `v_new`   : glue V  `[total_q, n_kv_head, head_dim]`
    ///   `o_ptr`   : output  `[total_q, n_q_head, head_dim]`
    /// where `total_q = Σ q_lens`. Per slot `b`:
    ///   `cu_seqlens_q[b..b+1]` bound its glue rows, `q_lens[b]` is its glue
    ///   count, `kv_lens[b]` its total context (prefix + glue). The sealed prefix
    ///   and the glue writer chunks live in `headers_ptr[b]` (`SlotHeader`).
    ///   `col_actual_pos[cu_kvlens[b] + k]` is column `k`'s TRUE sequence
    ///   position, driving the actual-position causal mask + glue RoPE.
    ///   `glue_write_slice[t]` / `glue_write_in_blk[t]` are glue row `t`'s
    ///   writer-chunk slice index + in-block offset.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged_glue_fp16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        batch: i32,
        max_glue: i32,
        n_q_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        softmax_scale: f32,
        k_new: *const c_void,
        v_new: *const c_void,
        rope_cs: *const f32,
        rope_interleaved: i32,
        cu_seqlens_q: *const u32,
        q_lens: *const u32,
        // `kv_lens[b]`: the slot's total column count (the reserved gaps ARE the
        // glue queries, already counted). Each column's sequence position is read
        // from its chunk `rope_base` (`slice_rope`) — no `col_actual_pos`.
        kv_lens: *const u32,
        // Per glue row (`[Σ q_lens]`): the gap chunk slice index + in-block offset
        // its K/V scatters into; its position is `slice_rope(gap) + in_blk`.
        glue_write_slice: *const u32,
        glue_write_in_blk: *const u32,
        // Per glue row (`[Σ q_lens]`): forward bridge window in tokens. Row `t`
        // attends column `c` iff `cpos <= row_pos + fwd_ahead[t]`. `0` == causal.
        fwd_ahead: *const u32,
        stream: *mut c_void,
    );

    /// BF16 glue attention. See [`run_paged_glue_fp16`] for the argument layout.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged_glue_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        batch: i32,
        max_glue: i32,
        n_q_head: i32,
        n_kv_head: i32,
        head_dim: i32,
        softmax_scale: f32,
        k_new: *const c_void,
        v_new: *const c_void,
        rope_cs: *const f32,
        rope_interleaved: i32,
        cu_seqlens_q: *const u32,
        q_lens: *const u32,
        kv_lens: *const u32,
        glue_write_slice: *const u32,
        glue_write_in_blk: *const u32,
        fwd_ahead: *const u32,
        stream: *mut c_void,
    );
}
