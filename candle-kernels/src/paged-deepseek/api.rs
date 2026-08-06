//! FFI bindings for the DeepSeek hybrid paged decode kernel.
//!
//! Single-latent K≡V attention (HEAD_DIM=512, MQA): the kernel walks the FP8
//! window slot (arena bands via `SlotHeader`) plus an index-driven selection of
//! f32 compressed entries, runs one online softmax over both sources, and the
//! combine folds splits + the per-head sink, normalizes, and de-rotates the
//! output at the query position. RoPE is computed in-kernel from the 32
//! YaRN-adjusted inverse frequencies — there is no position-sized table.

use core::ffi::c_void;

extern "C" {
    /// BF16 DeepSeek decode. Layouts:
    ///   `q_ptr`    : `[slots, n_q_head, 512]` bf16, **pre-RoPE** (roped in-kernel
    ///                at the query position derived from the writer slice).
    ///   `o_ptr`    : `[slots, n_q_head, 512]` bf16, final (normalized,
    ///                sink-folded, de-rotated) attention output.
    ///   `kv_new`   : `[slots, 512]` bf16 pre-RoPE latent, scattered into the
    ///                writer chunk's FP8 band arenas (K≡V single write).
    ///   `comp_ptr` : `[g_total, 512]` f32 pre-RoPE compressed entries.
    ///   `comp_pos` : `[g_total]` u32 group-start positions (RoPE at read).
    ///   `comp_idx` : `[slots, max_sel]` u32 ascending selected GIDs,
    ///                `0xFFFF_FFFF` padding.
    ///   `comp_cnt` : `[slots]` u32 selection counts.
    ///   `sinks`    : `[n_q_head]` f32 per-head sink logits.
    ///   `rope_freqs`: `[32]` f32 YaRN-adjusted inverse frequencies.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged_deepseek_decode_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        kv_new: *const c_void,
        comp_ptr: *const f32,
        comp_pos: *const u32,
        comp_idx: *const u32,
        comp_cnt: *const u32,
        sinks: *const f32,
        rope_freqs: *const f32,
        num_slots: i32,
        n_q_head: i32,
        softmax_scale: f32,
        window_size: i32,
        max_sel: i32,
        // > 0 pins the split-KV factor (deterministic accumulation order for
        // the mirror oracle); 0 sizes it from the SM count.
        num_splits_override: i32,
        // 0 skips the on-device write-len advance (the wave patches the length
        // host-side into a private snapshot); nonzero keeps it (live buffer).
        commit_write_len: i32,
        // Nullable stage-dump buffer (16608 f32; see kernel doc) — the mirror
        // oracle's stage-by-stage diagnostics.
        dbg: *mut f32,
        stream: *mut c_void,
    );

    /// BF16 DeepSeek prefill: many queries over a SETTLED arena slot (the
    /// host writes + commits every fresh latent before the launch — no fused
    /// scatter). Per-query positions and compressed selections; numerics are
    /// identical to running the decode entry once per token.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged_deepseek_prefill_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        q_pos: *const u32,
        // This layer's just-computed latents [fresh_rows, 512] bf16, keyed at
        // positions fresh_base + j (FP8-round-tripped in-kernel so key bits
        // match what future waves read from the arena). Null when fresh_rows=0.
        kv_fresh: *const c_void,
        comp_ptr: *const f32,
        comp_pos: *const u32,
        comp_idx: *const u32,
        comp_cnt: *const u32,
        sinks: *const f32,
        rope_freqs: *const f32,
        total_q: i32,
        n_q_head: i32,
        softmax_scale: f32,
        window_size: i32,
        max_sel: i32,
        fresh_rows: i32,
        fresh_base: i32,
        num_splits_override: i32,
        stream: *mut c_void,
    );

    /// Glue latent scatter: write `rows` bf16 latents into their RESERVED gap
    /// chunks (per-row block index + in-block offset from the reprojection's
    /// glue descriptors). Launch BEFORE the attention pass on the same stream.
    pub fn run_paged_deepseek_glue_scatter_bf16(
        kv: *const c_void,
        headers_ptr: *const u8,
        slices: *const u32,
        in_blk: *const u32,
        rows: i32,
        stream: *mut c_void,
    );

    /// Regression probe: kernel-side `ds_exp` over a device f32 array —
    /// asserted bit-identical to the CPU mirror's replica.
    pub fn run_deepseek_exp_probe(input: *const f32, out: *mut f32, n: i32, stream: *mut c_void);

    /// Regression probe: kernel-side RoPE trig for (pos, freq) pairs →
    /// interleaved (sin, cos) — asserted bit-identical to the CPU mirror.
    pub fn run_deepseek_sincos_probe(
        pos: *const i32,
        freq: *const f32,
        out: *mut f32,
        n: i32,
        stream: *mut c_void,
    );
}
