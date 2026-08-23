//! FFI bindings for paged decode attention kernels

use core::ffi::c_void;

extern "C" {
    // ========================================================================
    // Per-dtype dispatchers — persistent slot buffer API
    // Takes a single `headers_ptr` pointing to SlotHeader[num_active_slots] on GPU.
    // ========================================================================

    /// Returns 0 on success, nonzero when the launch needed the split-KV
    /// partial pool and its allocation failed (VRAM exhausted) — nothing was
    /// launched and the output holds no result.
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
    ) -> i32;

    /// See [`run_paged_decode_fp16`] for the return-code contract.
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
    ) -> i32;

    /// B2: decode with fused q8a128 context output (head_dim 128 or 256). Writes the per-head
    /// attention context directly as q8a1024 blocks into `q8_out` (no FP store + standalone
    /// quantize), feeding `o_proj` via the int8 path. `gate` (nullable, fp16) folds the output
    /// gate `sigmoid(g) ⊙ ctx` into the combine — the gated lineages' post-attention gate with
    /// no extra launches. A slot's `n_q_head × head_dim` gate values are contiguous;
    /// consecutive slots are `gate_slot_stride` elements apart, so the gate may be a strided
    /// view of the fused `[q | gate]` projection with no copy (pass 0 when fully contiguous).
    /// Returns 0 on success, nonzero when the partial pool (required by every q8 emit) could
    /// not be allocated — nothing was launched and `q8_out` holds no result.
    pub fn run_paged_decode_fp16_q8(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        q8_out: *mut c_void,
        gate: *const c_void,
        gate_slot_stride: i64,
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
    ) -> i32;

    /// B2 bf16 sibling of [`run_paged_decode_fp16_q8`] (`gate` is bf16).
    pub fn run_paged_decode_bf16_q8(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        q8_out: *mut c_void,
        gate: *const c_void,
        gate_slot_stride: i64,
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
    ) -> i32;

    /// Regression-test entry for the int8 m16n8k32 MMA fragment loaders.
    /// Computes `C[16][8] = A[16][32] · B[8][32]ᵀ` (row-major int8 `A`/`B`, int32
    /// `C`) via `load_a_frag_m16k32` / `load_b_frag_n8k32` / `mma_int8_m16n8k32`
    /// — the exact path the int8 decode QK dot uses. A fragment-layout
    /// regression makes `C` disagree with a CPU reference. Single warp; test-only.
    pub fn mma_int8_m16n8k32_test(a: *const i8, b: *const i8, c: *mut i32, stream: *mut c_void);
}
