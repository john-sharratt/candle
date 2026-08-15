//! FFI bindings for the paged latent-attention decode/prefill kernels.
//!
//! Single-latent K≡V attention (HEAD_DIM=512, MQA): the kernel walks the FP8
//! window slot (arena bands via `SlotHeader`) plus an index-driven selection of
//! f32 compressed entries, runs one online softmax over both sources, and the
//! combine folds splits + the per-head sink, normalizes, and de-rotates the
//! output at the query position. RoPE reads a FACTORED cos/sin table (hi/lo
//! position split + angle-addition, ~768 KB per frequency set, L2-resident)
//! built once per set by `run_latent_rope_table_build` with the exact
//! bit-mirrorable trig.

use core::ffi::c_void;

extern "C" {
    /// BF16 latent decode. Layouts:
    ///   `q_ptr`    : `[slots, n_q_head, 512]` bf16, **pre-RoPE** (roped in-kernel
    ///                at the explicit `q_pos`).
    ///   `o_ptr`    : `[slots, n_q_head, 512]` bf16, final (normalized,
    ///                sink-folded, de-rotated) attention output.
    ///   `kv_new`   : `[slots, 512]` bf16 pre-RoPE latent, scattered into the
    ///                writer chunk's FP8 band arenas (K≡V single write).
    ///   `comp_i8`  : `[g_total, 512]` POSITION-FREE int8 cache (nope bands raw,
    ///                rope bands pre-rotation); `comp_scale` `[g_total, 16]`.
    ///   `comp_pos` : `[g_total]` u32 assembled position per entry — the decode
    ///                rotates the rope bands at read time from this.
    ///   `comp_idx` : `[slots, max_sel]` u32 ascending selected GIDs,
    ///                `0xFFFF_FFFF` padding.
    ///   `comp_cnt` : `[slots]` u32 selection counts.
    ///   `q_pos`    : `[slots]` u32 query position (explicit — the windowless
    ///                slot needs no writer slice, and the compressed causal
    ///                guard drops entries with `comp_pos > q_pos`).
    ///   `sinks`    : `[n_q_head]` f32 per-head sink logits.
    ///   `rope_tab` : factored cos/sin table (`run_latent_rope_table_build`).
    ///   `partial_acc`/`partial_ml`: CALLER-OWNED split-KV partial workspace
    ///     with capacity for `[slots*H, num_splits, 512]` /
    ///     `[slots*H, num_splits, 2]` f32 (the Rust `LatentWorkspace`, built
    ///     once and reused). Caller ownership keeps launches lock-free: no
    ///     device-side pool, no shared mutable state in the launcher.
    #[allow(clippy::too_many_arguments)]
    /// Two-region position-free corpus cache builder (review items B + D): nope
    /// bands `[0,448)` → int8 `nope_i8` + per-band amax `nope_scale` `[G,14]`;
    /// rope bands `[448,512)` → BF16 `rope_bf` `[G,64]` (pre-rotation). The HOT
    /// retrieval artifact — readers rotate the rope bands at read time, and both
    /// decode and prefill derive their int8 QK operand from this one
    /// representation, so their compressed paths agree.
    pub fn run_latent_build_corpus_cache(
        comp: *const f32,
        nope_i8: *mut u8,
        nope_scale: *mut f32,
        rope_bf: *mut c_void,
        g_lo: i32,
        g_hi: i32,
        stream: *mut c_void,
    );

    pub fn run_paged_latent_decode_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        kv_new: *const c_void,
        // Two-region cache: nope int8 `[G,448]` + per-nope-band scale `[G,14]`;
        // rope pre-rotation bf16 `[G,64]`.
        nope_i8: *const u8,
        nope_scale: *const f32,
        comp_rope: *const c_void,
        comp_idx: *const u32,
        comp_cnt: *const u32,
        // `[G_total]` u32 assembled position per entry — the decode rotates the
        // rope bands at read time from this (the cache stores them pre-rotation).
        comp_pos: *const u32,
        // `[num_slots]` u32 query position, explicit so the windowless slot
        // needs no writer slice to derive it.
        q_pos: *const u32,
        sinks: *const f32,
        rope_tab: *const f32,
        partial_acc: *mut f32,
        partial_ml: *mut f32,
        num_slots: i32,
        n_q_head: i32,
        softmax_scale: f32,
        window_size: i32,
        max_sel: i32,
        // Resolved split-KV factor (the caller's policy sized the workspace).
        num_splits: i32,
        // 0 skips the on-device write-len advance (the wave patches the length
        // host-side into a private snapshot); nonzero keeps it (live buffer).
        commit_write_len: i32,
        // 1 = every slot's token latent is already in the arena (host writeback,
        // write-len committed) — skip the fused scatter. The speculative-verify
        // path runs a block's positions as virtual slots over ONE shared writer
        // slice; per-slot scatters would clobber a single position.
        pre_scattered: i32,
        // Nullable stage-dump buffer (16608 f32; see kernel doc) — the mirror
        // oracle's stage-by-stage diagnostics.
        dbg: *mut f32,
        stream: *mut c_void,
    );

    /// BF16 latent prefill: many queries over a SETTLED arena slot (the
    /// host writes + commits every fresh latent before the launch — no fused
    /// scatter). Per-query positions and compressed selections; numerics are
    /// identical to running the decode entry once per token.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged_latent_prefill_bf16(
        q_ptr: *const c_void,
        headers_ptr: *const u8,
        o_ptr: *mut c_void,
        q_pos: *const u32,
        // [total_q] which prefill seq each query belongs to (selects its arena
        // slot header + fresh-diagonal slice — the whole prefill fleet in one launch).
        seq_of: *const u32,
        // All prefill seqs' just-computed latents packed [total_new, 512] bf16,
        // keyed at positions new_meta[s].base + j (FP8-round-tripped in-kernel so
        // key bits match what future waves read from the arena).
        kv_new: *const c_void,
        // Two-region corpus cache (same the decode reads): nope int8 `[G,448]` +
        // per-nope-band scale `[G,14]`, rope pre-rotation bf16 `[G,64]`.
        nope_i8: *const u8,
        nope_scale: *const f32,
        rope_bf: *const c_void,
        comp_pos: *const u32,
        comp_idx: *const u32,
        comp_cnt: *const u32,
        sinks: *const f32,
        rope_tab: *const f32,
        partial_acc: *mut f32,
        partial_ml: *mut f32,
        // Per-prefill roped+quantized corpus scratch: the launcher ropes +
        // per-band int8-quantizes `comp_ptr` into these once (g_total on the
        // first chunk), then the attention kernel reads them and skips the
        // per-query RoPE. `comp_i8` [G,512] int8, `comp_scale` [G,4] f32.
        comp_i8: *mut u8,
        comp_scale: *mut f32,
        // Pre-quantized PV operand: `comp_v8` [G,512] int8 scaled by the
        // corpus-global per-dim max|v| in `comp_vmax` [512] f32 (which the
        // caller must pass ZEROED — the pre-pass atomicMax-accumulates into
        // it). Comp tiles gather V bytes from `comp_v8` instead of
        // requantizing per tile; the PV epilogue scale is `comp_vmax`.
        comp_v8: *mut u8,
        comp_vmax: *mut f32,
        g_total: i32,
        total_q: i32,
        n_q_head: i32,
        softmax_scale: f32,
        window_size: i32,
        max_sel: i32,
        // [n_seq*4] per-seq new-token diagonal metadata (one uint4 each):
        // {rows, base, start, -}.
        new_meta: *const u32,
        num_splits: i32,
        // Writer-chunk float format tag (`ArenaFormat`): the fresh diagonal is
        // fake-quantized to it so its bits match what future waves read from the
        // arena (FP8 rounds; BF16/F16/F32 are lossless for the bf16 source).
        store_fmt: i32,
        stream: *mut c_void,
    );

    /// Glue latent scatter: write `rows` bf16 latents into their RESERVED gap
    /// chunks (per-row block index + in-block offset from the reprojection's
    /// glue descriptors). Launch BEFORE the attention pass on the same stream.
    pub fn run_paged_latent_glue_scatter_bf16(
        kv: *const c_void,
        headers_ptr: *const u8,
        slices: *const u32,
        in_blk: *const u32,
        rows: i32,
        stream: *mut c_void,
    );

    /// Regression probe: kernel-side `ds_exp` over a device f32 array —
    /// asserted bit-identical to the CPU mirror's replica.
    pub fn run_latent_exp_probe(input: *const f32, out: *mut f32, n: i32, stream: *mut c_void);

    /// Regression probe: kernel-side RoPE trig for (pos, freq) pairs →
    /// interleaved (sin, cos) — asserted bit-identical to the CPU mirror.
    pub fn run_latent_sincos_probe(
        pos: *const i32,
        freq: *const f32,
        out: *mut f32,
        n: i32,
        stream: *mut c_void,
    );

    /// Build the factored RoPE cos/sin table for one frequency set (once per
    /// set at model load). `tab` holds `(ROPE_HI_DIM + ROPE_LO_DIM) * n_freqs`
    /// interleaved (sin, cos) f32 pairs: the hi block (positions `row << 10`)
    /// followed by the lo block (positions `row`).
    pub fn run_latent_rope_table_build(
        freqs: *const f32,
        tab: *mut f32,
        n_freqs: i32,
        stream: *mut c_void,
    );

    /// SM count of the current device — input to the caller's split-factor
    /// policy (which sizes the partial workspace it allocates per launch).
    pub fn run_latent_sm_count() -> i32;
}
