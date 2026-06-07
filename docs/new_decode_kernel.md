# Fused QKV + Attention Kernel — Implementation v1

This is the implementation of the kernel designed in `FUSED_QKV_ATTN_DESIGN.md`. It is structured as a set of header files, each with a clear single responsibility. The file structure mirrors how this would be laid out in a real codebase.

**This revision integrates with the existing v2 codebase** (`attn_v2.cuh`, `convert_all.cuh`, `slot_types.cuh`, `pal_iter.cuh`, `arena_table.cuh`, `fast_exp.cuh`, `warp_reduce.cuh`). Reused primitives are explicitly identified inline rather than re-implemented.

Design principles followed throughout:

- **Reuse v2 primitives.** Anything from the existing codebase that fits the new architecture is used directly. Where v2's primitive needs adaptation (e.g., dequant to INT8 instead of FP16), we add a thin parallel function in the existing file rather than fork the codebase.
- **Zero-cost abstractions.** Templates and `constexpr` everywhere; no runtime polymorphism in the hot path.
- **Compile-time validation.** `static_assert` invariants at every boundary.
- **Single-responsibility files.** Each header does one thing.
- **Explicit phases.** Phase boundaries encoded in the type system (`Phase12View` vs `Phase4View`).
- **Honest comments.** Hacks and known performance compromises are labeled.

The build target is sm_89 first, with arch-specialized paths gated on template parameters that compile cleanly on sm_86 and sm_120.

---

## API alignment with v2

The new kernel's launch and call signatures are **deliberately as close to v2's as possible**. Anything that wasn't strictly required to change for fusion was kept identical, including parameter names, ordering, types, grid/block shape, and the post-kernel commit launch.

### Top-level launch signature comparison

```cpp
// v2 (existing):
template <typename Q_T, typename T, typename O, int HEAD_DIM>
void launch_paged_decode_attn(
    const Q_T*     q,                  // pre-projected, pre-RoPE Q
    const uint8_t* headers_ptr,
    O*             out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const T*       k_new,              // pre-projected K
    const T*       v_new,              // pre-projected V
    const float*   rope_cs,
    int            rope_interleaved,
    cudaStream_t   stream = nullptr);

// New (this kernel):
template <typename Q_T, typename O, typename Cfg, int SM_VERSION>
cudaError_t launch_fused_qkv_attn(
    const Q_T*     activations,        // CHANGED: replaces `q`
    const uint8_t* w_qkv_q4,           // ADDED:   replaces k_new/v_new
    const void*    w_qkv_scales,       // ADDED:   replaces k_new/v_new
    const uint8_t* headers_ptr,        // SAME
    O*             out,                // SAME
    int            num_active_slots,   // SAME
    int            n_q_head,           // SAME
    int            n_kv_head,          // SAME
    float          softmax_scale,      // SAME
    const float*   rope_cs,            // SAME
    int            sliding_window_size, // ADDED: replaces rope_interleaved
    cudaStream_t   stream = nullptr);  // SAME
```

### What changed and why

**Removed** (no longer applicable after fusion):
- `q` — Q is now produced internally from `activations` × `w_qkv`.
- `k_new`, `v_new` — likewise produced internally.
- `T` template parameter — KV cache dtype was the dtype of `k_new`/`v_new` for v2's scatter; the fused kernel's scatter targets the arena directly using whatever per-palette format the arena holds (Q4_0/Q8_0/R16/F16). The KV cache dtype is now per-palette runtime.
- `HEAD_DIM` template parameter — replaced by `Cfg`, which carries HEAD_DIM plus N_Q_HEADS, N_KV_HEADS, D_MODEL, ROPE_STYLE, ROPE_INTERLEAVED, USE_QK_NORM, USE_SLIDING_WINDOW. v2 takes n_q_head/n_kv_head at runtime; the fused kernel takes them as Cfg fields plus runtime sanity-check args (the runtime values are asserted to match Cfg).
- `rope_interleaved` runtime arg — moved to `Cfg::ROPE_INTERLEAVED` (compile-time, since one model = one binary).

**Added** (required for fusion or new feature):
- `activations`, `w_qkv_q4`, `w_qkv_scales` — the QKV projection inputs.
- `sliding_window_size` — for Mistral-style local attention; ignored unless `Cfg::USE_SLIDING_WINDOW`.

**Preserved verbatim**:
- `Q_T` and `O` template parameters — same names, same meaning, same ordering.
- `headers_ptr`, `out`, `num_active_slots`, `n_q_head`, `n_kv_head`, `softmax_scale`, `rope_cs`, `stream` — same names, same types, same positions.
- Grid layout `(num_active_slots, n_kv_head)`.
- Post-attention launch of `commit_decode_write_len_kernel<HEAD_DIM>` on the same stream, identical participant pattern.
- Internal naming: `slot_idx` (v2 uses `slot_idx` as the blockIdx.x name; we do too), `q_rope_pos` for the new token's RoPE position (v2 calls this the same thing), `n_kv_tiles` derived from `kv_len = ws_rope + ws_len + 1`.

### Internal kernel-level naming alignment

All inner functions use v2's variable names where the concept is shared:

| Concept | v2 name | New kernel name |
|---|---|---|
| current slot from blockIdx.x | `slot_idx` | `slot_idx` |
| current KV head from blockIdx.y | `kv_head_idx` | `kv_head_idx` |
| Q's rotary position | `ws_rope + ws_len` (computed) | `q_rope_pos` (named, computed identically) |
| KV cache length | `kv_len` | `kv_len` |
| Output base address | `out_base` | `out_base` (in writeback TODO) |
| Smem K buffer | `shared_k` | `smem_int8_K` (renamed because dtype + layout differ) |
| Smem V buffer | `shared_v` | `smem_int8_V` (same reason) |
| Per-warp softmax state | `m_i`, `l_i` (locals) | `softmax_state.m_i`, `.l_i` (struct) |

The kernel-local naming is intentionally similar so that anyone reading the v2 code can navigate the new kernel by analogy.

---

## File 0: `convert_all.cuh` — additions only

We need one new method on `ArenaAccessor` that mirrors `load_head_scaled` but writes INT8 *without applying the scale*. The deferred-scaling path needs the raw integer side; the scale flows through the parallel FP32 track and is applied at MMA output.

```cuda
// =============================================================================
// ADDITION TO convert_all.cuh — to be inserted alongside load_head_scaled.
//
// load_head_int8_unscaled<HEAD_DIM, USE_TC>
//
// Loads a head (or sub-head for palette routing) from arena format into INT8
// in shared memory WITHOUT applying the per-block scale. The scale is exposed
// separately via `out_scale` so consumer warps can apply it at MMA output.
//
// This is the deferred-scaling counterpart to load_head_scaled. The two
// functions share the underlying format-dispatch table; only the conversion
// step at the end differs:
//
//   load_head_scaled<T>:        nibble - 8 → multiply by scale → cast to T → store
//   load_head_int8_unscaled:    nibble - 8 → store directly as int8
//                               + write `scale` to *out_scale (one per warp/block)
//
// For Q4_0 / Q8_0 source: the integer "centered nibble" is in [-8, 7] for Q4_0
// and [-128, 127] for Q8_0, so it fits cleanly in INT8 without any range
// reduction. F16/BF16/FP8 sources get max-abs-scanned and re-quantized to INT8
// with a fresh scale (this is the lossy step for non-quantized sources, but
// matches the precision budget in design doc §3.3).
//
// Caller is responsible for:
//   - Allocating sufficient `out_scale` slots (one FP32 per palette per token).
//   - Applying the scale during MMA-output FP32 multiply.
// =============================================================================

// (This goes in convert_all.cuh, inside the ArenaAccessor class.)

template <int HEAD_DIM, bool USE_TC>
__device__ __forceinline__ void load_head_int8_unscaled(
    int8_t*       dst,            // smem destination, INT8
    float*        out_scale,      // single FP32 scale slot for this load
    int           chunk_idx,      // 0 when k_ptr is already chunk-start
    int           dim_idx_unused, // unused
    int           within,         // token index within the chunk
    int           lane,
    float         in_scale_hint   // for F16/BF16/FP8 sources: a max-abs estimate; ignored for Q4/Q8
) const {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;

    // Format dispatch — mirrors load_head_scaled's structure.
    if (fmt_ == ArenaFormat::Q4_0) {
        // Q4_0: 32 elements per block, 16 packed bytes + FP16 scale.
        // The "centered nibble" path produces values in [-8, 7] which fit in INT8 trivially.
        // Scale is read from the block header.
        const uint8_t* block = reinterpret_cast<const uint8_t*>(arena_) + ...;
        // [Existing block address computation; identical to load_head_scaled.]
        float blk_scale = ...;  // load FP16 scale, convert to FP32

        // Write the scale once (lane 0).
        if (lane == 0) *out_scale = blk_scale;

        // Extract nibbles for this lane's VEC dims.
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim_in_block = lane * VEC + j;
            int byte_idx = dim_in_block >> 1;
            int nibble_hi = dim_in_block & 1;
            uint8_t byte = block[byte_idx];
            int nibble = nibble_hi ? (byte >> 4) : (byte & 0xF);
            int8_t centered = static_cast<int8_t>(nibble) - 8;  // [-8, 7]
            dst[lane * VEC + j] = centered;
        }

    } else if (fmt_ == ArenaFormat::Q8_0) {
        // Q8_0: already INT8 in storage; just copy through.
        // Scale is read from block header same way.
        const int8_t* block_data = ...;
        float blk_scale = ...;
        if (lane == 0) *out_scale = blk_scale;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            dst[lane * VEC + j] = block_data[lane * VEC + j];
        }

    } else if (fmt_ == ArenaFormat::F16 || fmt_ == ArenaFormat::R16
            || fmt_ == ArenaFormat::BF16 || fmt_ == ArenaFormat::F8E4M3) {
        // FP source: load to FP32, find max-abs across the warp, derive a fresh
        // INT8 scale, re-quantize.
        float fp_vals[VEC];
        float local_max_abs = 0.f;
        // [Existing FP load logic from load_head_scaled, but landing in fp_vals registers.]
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float v = ...;  // load and cast to FP32
            fp_vals[j] = v;
            float av = fabsf(v);
            if (av > local_max_abs) local_max_abs = av;
        }
        float warp_max = warp_reduce_max(local_max_abs);
        float new_scale = (warp_max > 0.f) ? (warp_max / 127.f) : 1.f;
        float inv = 1.f / new_scale;
        if (lane == 0) *out_scale = new_scale;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float scaled = fp_vals[j] * inv;
            float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
            dst[lane * VEC + j] = static_cast<int8_t>(__float2int_rn(clamped));
        }

    } else {
        // F32: same as F16 path with different load width.
        // [Implementation analogous to F16 case.]
    }
}
```

> **Note on file 0.** The full body needs to mirror `load_head_scaled`'s exact format dispatch and block address computation, which I haven't reproduced inline. The structure above is the additive surface — the actual edit to `convert_all.cuh` is roughly 80 lines pasted next to the existing `load_head_scaled` method. The integer-only path for Q4_0/Q8_0 is straightforward; the FP-source paths reuse the same address arithmetic but route output to INT8 instead of T.

---

## File 1: `arch_traits.cuh`

Per-architecture compile-time constants for MMA shapes. **No reuse from v2** — this is new infrastructure since v2 doesn't use tensor core MMAs.

```cuda
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fused_attn {

// =============================================================================
// ArchTraits<SM_VERSION> — compile-time per-architecture parameters.
//
// The kernel uses INT8 MMA throughout (see design doc §3.1). The MMA tile shape
// differs between Ampere and Ada/Blackwell:
//
//   sm_86 (Ampere):       m16n8k16   — K depth halved
//   sm_89 (Ada):          m16n8k32
//   sm_120 (Blackwell):   m16n8k32   (forward-compat path)
//
// The "K-depth" difference matters because per-palette scale invariance
// requires each MMA to span exactly one palette along K. With HEAD_DIM=128,
// N_PALETTE=4:
//
//   - palette covers 32 dims
//   - Ada/Blackwell:  1 MMA per palette  (perfect alignment)
//   - Ampere:         2 MMAs per palette (sum partials inside palette)
// =============================================================================

template<int SM_VERSION>
struct ArchTraits;

template<>
struct ArchTraits<89> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    // m16n8k32 INT8 fragments (per-thread):
    //   A: 16x32 INT8, k-major  → 4 bytes per thread → 1 reg (uint32)
    //   B:  8x32 INT8, k-major  → 4 bytes per thread → 1 reg (uint32)
    //   C: 16x8  INT32          → 4 ints  per thread → 4 regs
    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 16;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<>
struct ArchTraits<86> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;  // halved vs Ada

    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 18;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<>
struct ArchTraits<120> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 11;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<int SM_VERSION>
constexpr bool is_supported_arch() {
    return SM_VERSION == 86 || SM_VERSION == 89 || SM_VERSION == 120;
}

} // namespace fused_attn
```

---

## File 2: `model_descriptor.cuh`

Compile-time model shape parameterization. New file, no v2 reuse.

```cuda
#pragma once
#include "arch_traits.cuh"

namespace fused_attn {

enum class RopeStyle : int {
    Full    = 0,
    Partial = 1,
};

// =============================================================================
// ModelDescriptor — compile-time shape constants for one model.
//
// Every parameter is a template argument so the compiler can specialize loops,
// propagate constants, and pick optimal register allocation per shape. Multiple
// binaries (~30-50 KB PTX each) is acceptable for the codegen win.
// =============================================================================

template<
    int        HEAD_DIM_,
    int        N_Q_HEADS_,
    int        N_KV_HEADS_,
    int        D_MODEL_,
    int        N_PALETTE_,
    RopeStyle  ROPE_STYLE_,
    bool       USE_QK_NORM_,
    bool       USE_SLIDING_WINDOW_,
    bool       ROPE_INTERLEAVED_>
struct ModelDescriptor {
    static constexpr int       HEAD_DIM           = HEAD_DIM_;
    static constexpr int       N_Q_HEADS          = N_Q_HEADS_;
    static constexpr int       N_KV_HEADS         = N_KV_HEADS_;
    static constexpr int       D_MODEL            = D_MODEL_;
    static constexpr int       N_PALETTE          = N_PALETTE_;
    static constexpr RopeStyle ROPE_STYLE         = ROPE_STYLE_;
    static constexpr bool      USE_QK_NORM        = USE_QK_NORM_;
    static constexpr bool      USE_SLIDING_WINDOW = USE_SLIDING_WINDOW_;
    static constexpr bool      ROPE_INTERLEAVED   = ROPE_INTERLEAVED_;

    // Derived
    static constexpr int GQA_GROUP        = N_Q_HEADS / N_KV_HEADS;
    static constexpr int Q_OUTPUT_DIM     = N_Q_HEADS  * HEAD_DIM;
    static constexpr int K_OUTPUT_DIM     = N_KV_HEADS * HEAD_DIM;
    static constexpr int V_OUTPUT_DIM     = N_KV_HEADS * HEAD_DIM;
    static constexpr int TOTAL_OUTPUT_DIM = Q_OUTPUT_DIM + K_OUTPUT_DIM + V_OUTPUT_DIM;
    static constexpr int DIMS_PER_PALETTE = HEAD_DIM / N_PALETTE;
    static constexpr int MIN_BATCH_FOR_KERNEL = (16 + GQA_GROUP - 1) / GQA_GROUP;

    static_assert(HEAD_DIM == 128,
        "v1 only supports HEAD_DIM=128.");
    static_assert(N_PALETTE == 4,
        "v1 only supports N_PALETTE=4 due to deferred-scaling alignment.");
    static_assert(N_Q_HEADS % N_KV_HEADS == 0,
        "GQA requires n_q_heads divisible by n_kv_heads.");
    static_assert(D_MODEL % 32 == 0,
        "D_MODEL must be divisible by MMA K-dim (32).");
    static_assert(DIMS_PER_PALETTE == 32,
        "Deferred scaling requires DIMS_PER_PALETTE == 32 = MMA_K on Ada.");
};

// =============================================================================
// NOTE: NO MODEL-NAMED TYPEDEFS.
//
// `ModelDescriptor` is a *shape*, not a *model*. Two models with identical
// shape parameters compile to identical kernel binaries, and the kernel's
// correctness depends only on the shape — not on which model the shape
// happened to come from.
//
// Concretely: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B all have
// `(HEAD_DIM=128, N_Q_HEADS=32, N_KV_HEADS=8, D_MODEL=4096, RoPE=Full,
// no QK-norm, no sliding-window)`. Naming a typedef `Llama3_1_8B` would
// falsely suggest the kernel cares about Llama specifically; it does not.
//
// Instead, `ModelDescriptor` is instantiated *inline at the launcher
// definition site* via the DEFINE_SHAPE_LAUNCHER macro in launch.cu. The
// launcher's name encodes the shape it specializes for. A launcher like
// `launch_fused_attn_h128_q32_kv8_d4096_sm89` works for any model that
// matches that shape — the Rust dispatch layer maps from concrete model to
// shape, not from concrete model to launcher.
//
// If you need to discuss a specific model in a comment or test, do it in
// prose ("the Qwen3-30B-A3B shape: h128-q32-kv4-d2048"), not by introducing
// a model-named alias. The shape tuple is the only thing that matters.
// =============================================================================

} // namespace fused_attn
```

---

## File 3: `smem_arena.cuh`

Phase-based smem layout. New file. The Q4-source buffers in Phase4View directly mirror the layout v2's `shared_k`/`shared_v` use after dequant, but ours are pre-dequant Q4 packed (we move the dequant onto W2/W3 instead of inline).

```cuda
#pragma once
#include "model_descriptor.cuh"
#include "arch_traits.cuh"
#include "../arena_table.cuh"  // for CHUNK_SIZE constant from v2 stack

namespace fused_attn {

namespace tile {
    static constexpr int TILE_N             = 32;  // KV tokens per attention tile
    static constexpr int N_PIPELINE_STAGES  = 3;
    static constexpr int N_W_STAGING_STAGES = 2;
}

// =============================================================================
// Phase12View — what the arena looks like during QKV projection.
// =============================================================================
template<typename Cfg>
struct Phase12View {
    // Activation vector — INT8 + per-32-block scale.
    int8_t activations_int8 [Cfg::D_MODEL];
    float  activations_scales[Cfg::D_MODEL / 32];

    // W_qkv weight Q4_0 source buffer (loader → here via cp.async).
    static constexpr int W_Q4_BYTES_PER_STAGE =
        (Cfg::D_MODEL / 2)  // 32 K-elems per chunk × n-out, BUT we stream by K-chunk
        * 32                // dim chunks per cp.async
        * Cfg::TOTAL_OUTPUT_DIM / 32;
    // [Note: actual sizing comes from the W_qkv tile shape; this is a placeholder.
    // The real size is sized at template instantiation by build_load_queue. Listed
    // here as a constant so the union sizing check works.]
    uint8_t w_q4_src[tile::N_W_STAGING_STAGES][32 * Cfg::TOTAL_OUTPUT_DIM / 2];

    // W_qkv INT8 staging — produced by W2/W3 dequant from w_q4_src.
    static constexpr int W_INT8_BYTES_PER_STAGE = 32 * Cfg::TOTAL_OUTPUT_DIM;
    int8_t w_staging_int8 [tile::N_W_STAGING_STAGES][W_INT8_BYTES_PER_STAGE];
    float  w_staging_scales[tile::N_W_STAGING_STAGES][Cfg::TOTAL_OUTPUT_DIM / 32];

    // K_new / V_new FP32 intermediate.
    float k_new_fp32[Cfg::N_KV_HEADS * Cfg::HEAD_DIM];
    float v_new_fp32[Cfg::N_KV_HEADS * Cfg::HEAD_DIM];
};

// =============================================================================
// Phase4View — what the arena looks like during attention.
//
// Note the alignment with v2's shared_k / shared_v: same per-stage layout, just
// in INT8 instead of T (FP16/BF16/FP8). The Q4 packed source buffers
// `smem_q_K` / `smem_q_V` are new — v2 dequants in-place via load_head_scaled
// at load time, while we separate load and dequant into different warp roles.
// =============================================================================
template<typename Cfg>
struct Phase4View {
    static constexpr int Q4_BYTES_PER_TOKEN = Cfg::HEAD_DIM / 2;

    // Q4 packed source — loader writes here, dequant warps consume.
    uint8_t smem_q_K[tile::N_PIPELINE_STAGES][tile::TILE_N][Q4_BYTES_PER_TOKEN];
    uint8_t smem_q_V[tile::N_PIPELINE_STAGES][tile::TILE_N][Q4_BYTES_PER_TOKEN];

    // INT8 buffers for MMA consumption.
    //   K is k-major: smem_int8_K[stage][dim][token]
    //   V is mn-major: smem_int8_V[stage][token][dim]
    int8_t smem_int8_K[tile::N_PIPELINE_STAGES][Cfg::HEAD_DIM][tile::TILE_N];
    int8_t smem_int8_V[tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::HEAD_DIM];

    // Scale tables.
    float smem_scale_K_pre [tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::N_PALETTE];
    float smem_scale_K_post[tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::N_PALETTE];
    float smem_scale_V     [tile::N_PIPELINE_STAGES][tile::TILE_N];

    // Per-token RoPE positions for K (W2 reads these during dequant + RoPE).
    uint32_t k_rope_positions[tile::N_PIPELINE_STAGES][tile::TILE_N];
};

template<typename Cfg>
union SmemArena {
    Phase12View<Cfg> phase12;
    Phase4View<Cfg>  phase4;

    __device__ Phase12View<Cfg>& as_phase12() { return phase12; }
    __device__ Phase4View<Cfg>&  as_phase4()  { return phase4;  }
};

template<typename Cfg>
constexpr bool smem_arena_fits_default() {
    return sizeof(SmemArena<Cfg>) <= 48 * 1024;
}

} // namespace fused_attn
```

> **Note on arena sizing.** I've laid out the structure but the W_qkv staging size constants are placeholders — getting the right K-chunk × output-dim size requires nailing down the stream chunk granularity from build_load_queue. This is a Phase A bring-up detail.

---

## File 4: `cp_async.cuh`

cp.async primitives and LoadDescriptor. **Reuses v2's `cp_async_wait` and `cp_async_commit`** verbatim.

```cuda
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fused_attn {

// =============================================================================
// cp.async primitives.
//
// REUSED FROM v2 (attn_v2.cuh): cp_async_wait<N, USE_TC>() and
// cp_async_commit<USE_TC>(). They're already defined in v2 and used throughout
// the codebase. We include them here by reference — the actual definitions
// live in attn_v2.cuh and are visible via that header.
//
// NEW: cp_async_cg_16(dst_smem, src_global)
//      The single-instruction VRAM→smem 16-byte transfer. v2 doesn't expose
//      this as a named function (it's inline in load_head_scaled), but we
//      need it as a primitive for loader_role.
// =============================================================================

__device__ __forceinline__ void cp_async_cg_16(
    void*       dst_smem,
    const void* src_global
) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_int), "l"(src_global)
        : "memory"
    );
}

// =============================================================================
// LoadDescriptor — describes one cp.async transfer.
//
// Loader warps process a queue of these. Encodes everything the loader needs:
//   - source/destination addresses
//   - byte count (must be multiple of 16)
//   - barrier IDs for the free-slot wait and ready signal
// =============================================================================

struct alignas(16) LoadDescriptor {
    const void* src_vram;       // 8 B
    void*       dst_smem;       // 8 B
    uint32_t    bytes;          // 4 B
    uint8_t     free_barrier;   // 1 B
    uint8_t     ready_barrier;  // 1 B
    uint8_t     sync_count;     // 1 B
    uint8_t     _pad;           // 1 B
};

static constexpr uint8_t BARRIER_NONE = 0xFF;

// =============================================================================
// Named barriers.
//
// CUDA's bar.sync / bar.arrive (16 named barriers per CTA). v2 doesn't use
// named barriers; it uses __syncthreads() and __syncwarp() throughout. The
// new kernel's warp specialization needs finer-grained barriers.
// =============================================================================

__device__ __forceinline__ void bar_sync(int barrier_id, int participants) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(participants) : "memory");
}

__device__ __forceinline__ void bar_arrive(int barrier_id, int participants) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(participants) : "memory");
}

namespace bar_id {
    static constexpr int W_OR_KV_LOADED   = 0;
    static constexpr int W_OR_KV_CONSUMED = 1;
    static constexpr int INT8_READY       = 2;
    static constexpr int INT8_CONSUMED    = 3;
    static constexpr int PHASE_2_TO_3     = 4;
    static constexpr int PHASE_3_TO_4     = 5;
}

} // namespace fused_attn
```

---

## File 5: `mma_wrappers.cuh`

INT8 MMA inline asm wrappers. New file, no v2 reuse (v2 doesn't use tensor cores).

```cuda
#pragma once
#include "arch_traits.cuh"
#include <cstdint>

namespace fused_attn {

// =============================================================================
// INT8 MMA wrappers.
//
// Per-arch instruction:
//   sm_86:  mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
//   sm_89:  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//   sm_120: same as sm_89 (forward-compat)
// =============================================================================

template<int SM_VERSION>
__device__ __forceinline__ void mma_int8(
    int32_t (&d)[4],
    const uint32_t (&a)[1],
    const uint32_t (&b)[1],
    const int32_t  (&c)[4]
);

template<>
__device__ __forceinline__ void mma_int8<89>(
    int32_t (&d)[4],
    const uint32_t (&a)[1],
    const uint32_t (&b)[1],
    const int32_t  (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4}, "
        "{%5}, "
        "{%6, %7, %8, %9};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]),
          "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

template<>
__device__ __forceinline__ void mma_int8<86>(
    int32_t (&d)[4],
    const uint32_t (&a)[1],
    const uint32_t (&b)[1],
    const int32_t  (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4}, "
        "{%5}, "
        "{%6, %7, %8, %9};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]),
          "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

template<>
__device__ __forceinline__ void mma_int8<120>(
    int32_t (&d)[4],
    const uint32_t (&a)[1],
    const uint32_t (&b)[1],
    const int32_t  (&c)[4]
) {
    mma_int8<89>(d, a, b, c);  // forward-compat
}

// =============================================================================
// MMA fragment loaders.
//
// For sm_89 m16n8k32, the cleanest approach is `ldmatrix.sync.aligned` which
// has the right lane-to-element mapping baked in. For Phase A, we use that
// for A operands and a strided load for B (since K is k-major and B is k-major,
// the strided load is direct).
//
// TODO(phase A): validate against PTX ISA m16n8k32 fragment layout. The exact
// indices below need to be confirmed against running output vs reference.
// =============================================================================

// Load a 16x32 INT8 A fragment from k-major smem using ldmatrix.
// `smem_a` points to the start of the 16-row matrix in smem.
__device__ __forceinline__ uint32_t load_a_frag_m16k32_ldmatrix(
    const int8_t* smem_a,
    int           lda_bytes,   // bytes between rows of A
    int           lane
) {
    // For ldmatrix.x4, each lane provides the smem address of a row.
    // Lane 0..15 provide row addresses; lanes 16..31 provide the same rows
    // for the second half of K. The hardware re-shuffles into the correct
    // fragment layout.
    int row = lane & 15;
    int col_chunk = (lane >> 4) * 16;  // 0 or 16, selects K half
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(
        smem_a + row * lda_bytes + col_chunk));

    uint32_t frag;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %0, %0, %0}, [%1];\n"
        : "=r"(frag) : "r"(addr)
    );
    // [Note: x4 returns 4 fragments for 4 separate matrices; we use 1 for
    // simplicity here. For full m16n8k32 the multi-fragment ldmatrix is
    // more efficient — Phase A optimization.]
    return frag;
}

// Load an 8x32 INT8 B fragment from k-major smem.
__device__ __forceinline__ uint32_t load_b_frag_n8k32_strided(
    const int8_t* smem_b,
    int           ldb_bytes,
    int           lane
) {
    // B is 8 rows × 32 cols (k-major). Each lane reads 4 INT8 from the matrix.
    // Lane mapping (m16n8k32 B-fragment per PTX ISA):
    //   thread t holds 4 INT8 from (row = t % 8, col = (t/8)*4 .. (t/8)*4+3) maybe?
    //   (Actual mapping is documented in PTX; this is a placeholder.)
    int row = lane & 7;
    int col = (lane >> 3) * 4;   // 0, 4, 8, ... up to 28
    return *reinterpret_cast<const uint32_t*>(smem_b + row * ldb_bytes + col);
}

} // namespace fused_attn
```

---

## File 6: `rope.cuh`

RoPE primitives. **Fully reuses v2's existing helpers** — `rope_cos_sin`, `apply_rope_rotary_f32`, `apply_rope_interleaved_f32` are already in `attn_v2.cuh` and they do exactly what we need.

```cuda
#pragma once
#include "../attn_v2.cuh"  // for rope_cos_sin, apply_rope_rotary_f32, apply_rope_interleaved_f32
#include "model_descriptor.cuh"

namespace fused_attn {

// =============================================================================
// RoPE primitives — REUSED FROM v2.
//
// v2 already has:
//   rope_cos_sin<HEAD_DIM>(pos, d_idx, rope_cs, cos_v, sin_v)
//   apply_rope_rotary_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs)
//   apply_rope_interleaved_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs)
//
// We just dispatch on the model descriptor's ROPE_STYLE / ROPE_INTERLEAVED
// flags rather than reimplementing.
//
// NEW: a thin dispatch wrapper that picks the right v2 function based on
// the model descriptor's compile-time flags.
// =============================================================================

template<int HEAD_DIM, int VEC, RopeStyle STYLE, bool INTERLEAVED>
__device__ __forceinline__ void apply_rope_dispatch(
    float*       regs,
    int          lane,
    int          pos,
    const float* rope_cs
) {
    if constexpr (STYLE == RopeStyle::Partial) {
        // Partial RoPE: rotate only first HEAD_DIM/2 dims.
        // Implementation: call v2's full-rotary on a halved virtual head_dim,
        // and zero out the contribution for upper-half dims.
        //
        // For HEAD_DIM=128, VEC=4 (lane covers 4 dims): lanes 0..15 cover
        // the rotated half, lanes 16..31 pass through unchanged.
        if (lane < 16) {
            apply_rope_rotary_f32<VEC, HEAD_DIM / 2>(regs, lane, pos, rope_cs);
        }
        // else: regs unchanged

    } else if constexpr (INTERLEAVED) {
        apply_rope_interleaved_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs);
    } else {
        apply_rope_rotary_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs);
    }
}

} // namespace fused_attn
```

> **Note on Partial RoPE:** The "halve and skip upper" approach above is a sketch — the actual partial-RoPE semantics for a given model may differ (some flavors rotate a contiguous prefix, others rotate alternating dims). For the Llama and Qwen models in the supported set, all use full RoPE, so this branch is dead code in v1. Worth implementing properly when we add a model that needs it.

---

## File 7: `dequant_store.cuh`

The dequant primitive. **Reuses `ArenaAccessor::load_head_int8_unscaled`** (from File 0's addition to convert_all.cuh) and v2's RoPE primitives.

```cuda
#pragma once
#include "../convert/convert_all.cuh"     // for ArenaAccessor::load_head_int8_unscaled
#include "../arena_table.cuh"              // for ArenaFormat, CHUNK_SIZE
#include "../simple/warp_reduce.cuh"       // for warp_reduce_max
#include "../pal_iter.cuh"                 // for PalIter (palette routing)
#include "model_descriptor.cuh"
#include "rope.cuh"

namespace fused_attn {

// =============================================================================
// dequant_kv_tile_K — phase 4 K-path dequant + RoPE + re-quantization.
//
// Reuses:
//   - ArenaAccessor::load_head_int8_unscaled (new addition to convert_all.cuh)
//     for the format-agnostic load and palette dispatch.
//   - PalIter for palette-routed dim ordering.
//   - apply_rope_dispatch for the RoPE step.
//   - warp_reduce_max for the new-scale max-abs reduction.
//
// Per-tile, this function:
//   1. For each token in tile and each palette: call load_head_int8_unscaled
//      to populate a per-lane FP32 register set (after dequant, before scale)
//      — NOTE: load_head_int8_unscaled writes INT8 to smem, but we need FP32
//      in registers for RoPE. So instead of using the unscaled variant here,
//      we use load_head_scaled<float> and apply RoPE/re-quant in registers.
//   2. Apply RoPE per-element using the pair-shuffle pattern (v2 helper).
//   3. Compute per-palette max-abs across the warp, derive new scale.
//   4. Re-quantize to INT8 with new scale.
//   5. Write INT8 to k-major smem destination + new scale to scale table.
// =============================================================================

template<typename Cfg>
__device__ void dequant_kv_tile_K(
    // ─── Source: arena pointers and palette routing ────────────────────
    const uint8_t*  head_ptr,        // KvHead pointer for this slot/kvhead
    int             tile_idx,        // tile index in attention loop
    int             tile_within_chunk_base,  // first within-chunk position in tile
    // ─── Destination: smem int8 + scales ───────────────────────────────
    int8_t*         dst_int8_kmajor, // [HEAD_DIM][TILE_N]
    int             dst_dim_stride,  // = TILE_N (bytes between dim rows)
    float*          out_scales_post, // [TILE_N][N_PALETTE]
    // ─── RoPE ──────────────────────────────────────────────────────────
    const float*    rope_cs_table,
    const uint32_t* rope_positions_per_token,  // [TILE_N]
    // ─── Geometry ──────────────────────────────────────────────────────
    int             lane,
    int             warp_in_pool     // unused; placeholder for future split
) {
    constexpr int HEAD_DIM    = Cfg::HEAD_DIM;
    constexpr int N_PALETTE   = Cfg::N_PALETTE;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int VEC         = HEAD_DIM / WARP_SIZE;  // 4 for HEAD_DIM=128
    constexpr int TILE_N      = tile::TILE_N;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    // Set up palette iterator from the head's k_pal map (REUSES v2's PalIter).
    PalIter<VEC, HEAD_DIM> ki;
    ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);

    // Per-tile loop over tokens.
    #pragma unroll 1  // don't fully unroll; each iter is heavy
    for (int t = 0; t < TILE_N; ++t) {
        int within = tile_within_chunk_base + t;

        // ─── Step 1: load + dequant FP32 into registers ─────────────────
        //
        // Use ArenaAccessor::load_head_scaled<float> with scale=1.0 to get
        // the raw post-quant values into FP32 registers, then we'll apply
        // scale inline and RoPE-rotate.
        //
        // [Architecturally cleaner alternative: have load_head_scaled
        // optionally skip the scale multiply when called with scale_hint=0.
        // For now, use the existing API and account for scale separately.]
        float k_regs[VEC];
        float k_scale_per_pal[N_PALETTE];

        constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;

        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            int      k_fmt   = kvhead_k_fmt<HEAD_DIM>(head_ptr, p);
            float    k_scale_p = kvhead_k_scale<HEAD_DIM>(head_ptr, p);
            k_scale_per_pal[p] = k_scale_p;

            // Load this palette's contribution to k_regs[].
            // PalIter tells us which lane's VEC-elements map to which palette;
            // for lanes that don't map to palette p, the load is a no-op.
            //
            // [Note: load_head_scaled<float> writes to smem. For register
            // residency we need a different load primitive. v2's load path
            // is smem-targeted; for this code path we use an explicit
            // per-element loop similar to load_head_scaled's body.]
            //
            // TODO(phase A): Either (a) extend load_head_scaled to support
            // a register destination via a callback functor, or (b) inline
            // the format-dispatch logic here. (a) is more reusable; (b) is
            // simpler to write first.
        }

        // [Placeholder body — actual load + dequant happens here using
        // approach (b) above for v1.]

        // ─── Step 2: gather into logical-dim order via PalIter ──────────
        //
        // After load above, k_regs[j] holds element from the lane's PalIter[j]
        // dim. Re-shuffling to logical order happens at write time below; for
        // RoPE we need the logical-dim element so use ki[j] indices.

        // ─── Step 3: apply RoPE in FP32 ──────────────────────────────────
        apply_rope_dispatch<HEAD_DIM, VEC, Cfg::ROPE_STYLE, Cfg::ROPE_INTERLEAVED>(
            k_regs, lane, rope_positions_per_token[t], rope_cs_table);

        // ─── Step 4: compute per-palette max-abs, derive new scale ──────
        float max_abs_per_pal[N_PALETTE] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim = lane * VEC + j;
            int pal = dim / Cfg::DIMS_PER_PALETTE;
            float av = fabsf(k_regs[j]);
            if (av > max_abs_per_pal[pal]) max_abs_per_pal[pal] = av;
        }
        float inv_scale_new[N_PALETTE];
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            float rmax = warp_reduce_max(max_abs_per_pal[p]);
            float new_scale = (rmax > 0.f) ? (rmax / 127.f) : 1.f;
            inv_scale_new[p] = 1.f / new_scale;
            if (lane == 0) {
                out_scales_post[t * N_PALETTE + p] = new_scale;
            }
        }

        // ─── Step 5: re-quantize to INT8, write k-major ─────────────────
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim = lane * VEC + j;
            int pal = dim / Cfg::DIMS_PER_PALETTE;
            float scaled = k_regs[j] * inv_scale_new[pal];
            float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
            int8_t q8 = static_cast<int8_t>(__float2int_rn(clamped));
            // k-major: dst[dim][token] = dst_int8 + dim*dst_dim_stride + t
            dst_int8_kmajor[dim * dst_dim_stride + t] = q8;
        }
    }
}

// =============================================================================
// dequant_kv_tile_V — phase 4 V-path dequant.
//
// Simpler than K because: no RoPE, scales already known per-token (loaded
// from arena via cp.async into smem_scale_V), output is mn-major (token-major).
//
// Reuses ArenaAccessor's format dispatch via load_head_int8_unscaled (new
// addition). The scale propagates through the parallel FP32 track in the
// consumer's P-fold-in step (see consumer_role).
// =============================================================================

template<typename Cfg>
__device__ void dequant_kv_tile_V(
    const uint8_t*  head_ptr,
    int             tile_idx,
    int             tile_within_chunk_base,
    int8_t*         dst_int8_mnmajor,  // [TILE_N][HEAD_DIM]
    int             dst_token_stride,   // = HEAD_DIM
    float*          out_scales_per_token,  // [TILE_N], filled by load_head_int8_unscaled
    int             lane,
    int             warp_in_pool
) {
    constexpr int HEAD_DIM    = Cfg::HEAD_DIM;
    constexpr int N_PALETTE   = Cfg::N_PALETTE;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int VEC         = HEAD_DIM / WARP_SIZE;
    constexpr int TILE_N      = tile::TILE_N;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    PalIter<VEC, HEAD_DIM> vi;
    vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);

    for (int t = 0; t < TILE_N; ++t) {
        int within = tile_within_chunk_base + t;
        constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;

        // For each palette, use ArenaAccessor::load_head_int8_unscaled to
        // populate the dst_int8_mnmajor[t][...] slice for this palette's dims.
        // This reuses v2's format dispatch and writes INT8 directly.
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            int      v_fmt   = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);

            ArenaAccessor v_acc((const char*)(uintptr_t)v_ptr_p, v_fmt,
                                sub_head_stride, sub_head_stride,
                                BLOCKS_PER_DIM, 0);

            // Output: dst_int8_mnmajor + t*HEAD_DIM + p*SUB_HEAD_DIM.
            // load_head_int8_unscaled writes the lane-mapped VEC elements.
            int8_t* dst_pal = dst_int8_mnmajor + t * dst_token_stride + p * SUB_HEAD_DIM;

            // For per-token V scale: the scale_V we want is loaded from arena
            // separately (see Phase4View::smem_scale_V populated by loader_role
            // via a separate cp.async). We pass nullptr here for the scale
            // output since the per-token scale isn't synthesized inside the
            // dequant — it's pre-known.
            //
            // For the v2 ArenaAccessor::load_head_int8_unscaled, the scale-out
            // slot writes one FP32 per palette per call; we only care about
            // the integer side here.
            float dummy_scale_out;
            v_acc.template load_head_int8_unscaled<SUB_HEAD_DIM, /*USE_TC=*/true>(
                dst_pal,
                &dummy_scale_out,
                /*chunk_idx=*/0,
                /*dim_idx_unused=*/0,
                within,
                lane,
                /*in_scale_hint=*/0.f
            );
        }
    }

    // Note: out_scales_per_token is populated by loader_role from the arena's
    // per-token V-scale tensor via cp.async, in parallel with the V data.
    // This function doesn't write to it.
}

} // namespace fused_attn
```

> **Note on dequant_store.cuh.** This file has a Phase A TODO that I want to be explicit about: getting the FP32-register output from the existing `ArenaAccessor` requires either extending the accessor with a register-targeted callback or inlining the format dispatch. The cleaner approach is the former, and it'd be a small (~30 line) addition to `convert_all.cuh`. Until then, the K path's body is sketched but not complete — it relies on inlining work that's straightforward but not literally written.

---

## File 8: `softmax_state.cuh`

Online softmax. **Borrows the structure from v2's `process_tile`** (the `m_i, l_i, alpha, beta` pattern with `fast_exp::exp2` reduction) but encapsulated as a per-warp state struct.

```cuda
#pragma once
#include "../fast_exp.cuh"            // REUSED: fast_exp::exp2 (vectorized exp)
#include "../simple/warp_reduce.cuh"   // REUSED: warp_reduce_sum, warp_reduce_max
#include <cuda_runtime.h>

namespace fused_attn {

// =============================================================================
// OnlineSoftmaxState.
//
// Borrows the m_i / l_i / alpha / beta pattern directly from v2's process_tile
// in attn_v2.cuh. Encapsulated as a struct so each consumer warp's state is
// register-resident with clear lifetime.
//
// Compared to v2's process_tile inline code:
//   - v2 interleaves softmax, K-dot, and V-accumulation per token in a single
//     loop with TILE_UNROLL = 4 or 2.
//   - The new kernel separates these into: QK^T MMA → logits FP32 → softmax
//     → P fold-in → PV MMA. Softmax operates on a logits vector after MMA.
//
// The math is identical; the iteration order differs.
// =============================================================================

struct OnlineSoftmaxState {
    float m_i;   // running max
    float l_i;   // running sum of exps

    __device__ __forceinline__ void init() {
        m_i = -1e38f;  // matches v2's sentinel
        l_i = 0.f;
    }

    // Update with a tile's logits.
    //
    // After this call:
    //   - logits[] is overwritten with exp(logit - new_m), ready for PV use
    //   - m_i and l_i are advanced
    //   - returns alpha (the rescale factor for the running output accumulator)
    //
    // Mirrors v2's update pattern but vectorized over a logits vector
    // instead of one logit at a time.
    template<int N_LOGITS_PER_THREAD>
    __device__ __forceinline__ float update(float (&logits)[N_LOGITS_PER_THREAD]) {
        // Step 1: tile-local max across this thread's logits.
        float tile_max = -1e38f;
        #pragma unroll
        for (int e = 0; e < N_LOGITS_PER_THREAD; ++e) {
            if (logits[e] > tile_max) tile_max = logits[e];
        }
        tile_max = warp_reduce_max(tile_max);  // REUSED from v2 codebase

        // Step 2: combine with running max.
        float new_m = fmaxf(m_i, tile_max);
        // First-tile guard: m_i = -1e38 → m_i - new_m underflows.
        // v2 uses fast_exp::exp2 which handles this; we trust it here too.
        float alpha;
        if (m_i < -1e30f) {
            alpha = 0.f;
        } else {
            // Use fast_exp::exp2 (REUSED from fast_exp.cuh) for the scalar.
            alpha = fast_exp::exp2<float, fast_exp::Softmax>(m_i - new_m);
        }

        // Step 3: replace logits with exp(logit - new_m), accumulate sum.
        float tile_sum = 0.f;
        #pragma unroll
        for (int e = 0; e < N_LOGITS_PER_THREAD; ++e) {
            float p = fast_exp::exp2<float, fast_exp::Softmax>(logits[e] - new_m);
            logits[e] = p;
            tile_sum += p;
        }
        tile_sum = warp_reduce_sum(tile_sum);  // REUSED from v2

        // Step 4: update running state.
        l_i = l_i * alpha + tile_sum;
        m_i = new_m;

        return alpha;
    }

    __device__ __forceinline__ float normalizer() const {
        return __fdividef(1.f, fmaxf(l_i, 1e-10f));
    }
};

} // namespace fused_attn
```

---

## File 9: `consumer_role.cuh`

The consumer warp main function. Structural equivalent of v2's main loop, but reorganized for tensor cores.

```cuda
#pragma once
#include "model_descriptor.cuh"
#include "arch_traits.cuh"
#include "smem_arena.cuh"
#include "mma_wrappers.cuh"
#include "rope.cuh"
#include "softmax_state.cuh"
#include "cp_async.cuh"
#include "../simple/warp_reduce.cuh"

namespace fused_attn {

// =============================================================================
// consumer_role<Cfg, Arch>
//
// 4 consumer warps (W4-W7) execute three phases:
//
//   Phase 2: QKV projection
//     - For each K-chunk along D_MODEL: INT8 MMA × N_TILES_PER_WARP, with a
//       parallel FP32 scale-product track that's applied at MMA-output.
//     - Output Q/K/V in FP32 register form, partitioned by N-dim.
//
//   Phase 3: Route by descriptor
//     - For each owned dim: route to Q (RoPE+quant), K_new (smem for W2),
//       or V_new (smem for W3) based on dim position.
//
//   Phase 4: Attention loop
//     - Per tile: 4 INT8 QK^T MMAs (one per palette), apply scales, sum FP32 logits.
//     - Online softmax (REUSES v2's OnlineSoftmaxState pattern).
//     - P fold-in (NEW: multiply by scale_V, re-quant, gives pure-INT PV).
//     - PV INT8 MMA × HEAD_DIM/8.
//     - Accumulator update: out_accum * alpha + scale_P * INT32.
//
// Reuses from v2:
//   - apply_rope_dispatch (which wraps v2's apply_rope_*_f32)
//   - OnlineSoftmaxState (which wraps v2's m_i/l_i/alpha/beta pattern)
//   - vec2_traits / load_vec2 for the output writeback (FP32 → FP16/BF16)
//   - warp_reduce_max / warp_reduce_sum throughout
// =============================================================================

template<typename Cfg, typename Arch, typename O>
__device__ void consumer_role(
    int                     warp_in_pool,    // 0..3
    int                     lane,
    SmemArena<Cfg>&         arena,
    const float*            rope_cs_table,
    uint32_t                q_rope_pos,         // = ws_rope + ws_len, read from slot
    float                   softmax_scale,      // matches v2's runtime softmax_scale arg
    int                     sliding_window_size,
    int                     n_kv_tiles,
    int                     slot_idx,            // matches v2's blockIdx.x → slot_idx naming
    int                     n_q_head,            // matches v2's runtime n_q_head arg
    O*                      out                  // [num_active_slots, n_q_head, HEAD_DIM]
) {
    using namespace fused_attn::tile;
    constexpr int N_DIMS_PER_WARP   = Cfg::TOTAL_OUTPUT_DIM / 4;
    constexpr int N_TILES_PER_WARP  = N_DIMS_PER_WARP / Arch::MMA_N;
    constexpr int N_K_CHUNKS        = Cfg::D_MODEL / Arch::MMA_K;

    static_assert(Cfg::TOTAL_OUTPUT_DIM % 4 == 0,
        "TOTAL_OUTPUT_DIM must split evenly across 4 consumer warps.");
    static_assert(N_DIMS_PER_WARP % Arch::MMA_N == 0,
        "N_DIMS_PER_WARP must be a multiple of MMA_N for clean tiling.");

    // ───────────────────────────────────────────────────────────────────
    // PHASE 2: QKV projection
    // ───────────────────────────────────────────────────────────────────
    int32_t q_partial[N_TILES_PER_WARP][Arch::C_REGS_PER_THREAD] = {};
    float   scale_product[N_TILES_PER_WARP] = {};
    int     my_n_dim_start = warp_in_pool * N_DIMS_PER_WARP;

    for (int k_chunk = 0; k_chunk < N_K_CHUNKS; ++k_chunk) {
        // Wait for loader+dequant to deliver this k_chunk's INT8 data.
        bar_sync(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);

        // Load A frag (activations for this k_chunk).
        const int8_t* a_smem = &arena.as_phase12()
            .activations_int8[k_chunk * Arch::MMA_K];
        uint32_t a_frag[1] = {
            load_a_frag_m16k32_ldmatrix(a_smem,
                                         /*lda_bytes=*/Cfg::D_MODEL,
                                         lane)
        };
        float scale_A = arena.as_phase12().activations_scales[k_chunk];

        #pragma unroll
        for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
            int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;
            int stage = k_chunk % N_W_STAGING_STAGES;

            const int8_t* b_smem = &arena.as_phase12()
                .w_staging_int8[stage][n_offset * Arch::MMA_K];
            uint32_t b_frag[1] = {
                load_b_frag_n8k32_strided(b_smem,
                                           /*ldb_bytes=*/Arch::MMA_K,
                                           lane)
            };
            float scale_B = arena.as_phase12()
                .w_staging_scales[stage][n_offset / 32];

            mma_int8<89>(  // TODO: arch-template
                q_partial[n_tile], a_frag, b_frag, q_partial[n_tile]);

            scale_product[n_tile] += scale_A * scale_B;
        }

        bar_arrive(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    // INT32 → FP32 with combined scale.
    float fp32_output[N_TILES_PER_WARP][Arch::C_REGS_PER_THREAD];
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
        #pragma unroll
        for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
            fp32_output[n_tile][r] =
                scale_product[n_tile] * static_cast<float>(q_partial[n_tile][r]);
        }
    }

    bar_sync(bar_id::PHASE_2_TO_3, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 3: route owned dims by descriptor
    // ───────────────────────────────────────────────────────────────────
    constexpr int Q_OUTPUT_DIM = Cfg::Q_OUTPUT_DIM;
    constexpr int K_END        = Q_OUTPUT_DIM + Cfg::K_OUTPUT_DIM;

    int8_t q_int8[N_TILES_PER_WARP][Arch::C_REGS_PER_THREAD];
    float  scale_Q[Cfg::N_PALETTE] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
        int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;

        if (n_offset < Q_OUTPUT_DIM) {
            // Q dim: RoPE in-register, then re-quantize.
            //
            // RoPE position for Q is the new token's absolute position:
            //   q_rope_pos = ws_rope + ws_len = kv_len - 1
            // (matches v2's `ws_rope + ws_len` formula exactly).
            //
            // The RoPE helpers from v2 expect a per-lane register vector
            // of VEC=HEAD_DIM/32 elements with a specific lane mapping.
            // Our consumer warp's MMA C-fragment has 4 INT32 regs per thread,
            // which we converted to FP32 above. The mapping from C-fragment
            // regs to logical dim positions follows the m16n8 layout.
            //
            // For HEAD_DIM=128 with 4 consumer warps splitting N: each warp
            // owns 32 dims of N-axis. Within a warp, lanes hold adjacent
            // groups of 4 dims.
            //
            // TODO(phase B): nail down the exact dim assignment per (lane, r).
            // For now, treat fp32_output[n_tile][r] as the FP32 value at
            // dim = n_offset + (lane%4)*2 + (r >> 1).

            // Apply RoPE using the v2 helper. Need to assemble fp32_output
            // into a VEC-sized register vector that matches v2's expected
            // layout (lane-contiguous dims).
            //
            // [Sketch only — phase B fills the precise indexing.]
            float v2_layout_regs[4];  // VEC=4 for HEAD_DIM=128
            // [Pack fp32_output into v2_layout_regs based on lane mapping.]

            apply_rope_dispatch<Cfg::HEAD_DIM, 4, Cfg::ROPE_STYLE,
                                Cfg::ROPE_INTERLEAVED>(
                v2_layout_regs, lane, static_cast<int>(q_rope_pos),
                rope_cs_table);

            // [Unpack v2_layout_regs back into fp32_output.]

            // Track per-palette max-abs for scale_Q.
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int n_within_tile = (lane % 4) * 2 + (r >> 1);
                int dim = n_offset + n_within_tile;
                int dim_in_head = dim % Cfg::HEAD_DIM;
                int pal = dim_in_head / Cfg::DIMS_PER_PALETTE;
                float av = fabsf(fp32_output[n_tile][r]);
                if (av > scale_Q[pal]) scale_Q[pal] = av;
            }

        } else if (n_offset < K_END) {
            // K_new dim: write FP32 to k_new smem for W2.
            int kv_offset = n_offset - Q_OUTPUT_DIM;
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int n_within_tile = (lane % 4) * 2 + (r >> 1);
                int row_in_m = (lane / 4);
                if (row_in_m == 0) {
                    arena.as_phase12().k_new_fp32[kv_offset + n_within_tile]
                        = fp32_output[n_tile][r];
                }
            }

        } else {
            // V_new dim: write FP32 to v_new smem for W3.
            int kv_offset = n_offset - K_END;
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int n_within_tile = (lane % 4) * 2 + (r >> 1);
                int row_in_m = (lane / 4);
                if (row_in_m == 0) {
                    arena.as_phase12().v_new_fp32[kv_offset + n_within_tile]
                        = fp32_output[n_tile][r];
                }
            }
        }
    }

    // Finalize per-palette scale_Q via warp-reduce.
    #pragma unroll
    for (int p = 0; p < Cfg::N_PALETTE; ++p) {
        scale_Q[p] = warp_reduce_max(scale_Q[p]) / 127.f;
        if (scale_Q[p] == 0.f) scale_Q[p] = 1.f;
    }

    // Quantize Q to INT8 using scale_Q.
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
        int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;
        if (n_offset < Q_OUTPUT_DIM) {
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int n_within_tile = (lane % 4) * 2 + (r >> 1);
                int dim = n_offset + n_within_tile;
                int dim_in_head = dim % Cfg::HEAD_DIM;
                int pal = dim_in_head / Cfg::DIMS_PER_PALETTE;
                float scaled = fp32_output[n_tile][r] / scale_Q[pal];
                float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
                q_int8[n_tile][r] = static_cast<int8_t>(__float2int_rn(clamped));
            }
        }
    }

    bar_sync(bar_id::PHASE_3_TO_4, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 4: attention loop
    // ───────────────────────────────────────────────────────────────────
    OnlineSoftmaxState softmax_state;
    softmax_state.init();

    constexpr int OUT_DIMS_PER_WARP   = Cfg::HEAD_DIM / 4;
    constexpr int OUT_REGS_PER_THREAD = OUT_DIMS_PER_WARP / 4;
    float out_accum[OUT_REGS_PER_THREAD] = {};

    for (int tile = 0; tile < n_kv_tiles; ++tile) {
        bar_sync(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);

        int stage = tile % N_PIPELINE_STAGES;

        // ─── QK^T: N_PALETTE INT8 MMAs ──────────────────────────────────
        int32_t logits_int[Cfg::N_PALETTE][Arch::C_REGS_PER_THREAD] = {};

        #pragma unroll
        for (int p = 0; p < Cfg::N_PALETTE; ++p) {
            // Assemble Q fragment from q_int8 for this palette's dim range.
            // [TODO phase B: lane-mapping for Q frag from per-palette slice.]
            uint32_t q_frag[1] = {0};

            const int8_t* k_smem = &arena.as_phase4()
                .smem_int8_K[stage][p * Cfg::DIMS_PER_PALETTE][0];
            uint32_t k_frag[1] = {
                load_b_frag_n8k32_strided(k_smem, /*ldb_bytes=*/TILE_N, lane)
            };

            mma_int8<89>(  // TODO: arch-template
                logits_int[p], q_frag, k_frag, logits_int[p]);
        }

        // ─── Apply per-palette scales, sum to FP32 logits ──────────────
        float logits_fp32[Arch::C_REGS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
            float sum = 0.f;
            int t_in_tile = (lane % 4) * 2 + (r >> 1);
            #pragma unroll
            for (int p = 0; p < Cfg::N_PALETTE; ++p) {
                float s_q = scale_Q[p];
                float s_k = arena.as_phase4()
                    .smem_scale_K_post[stage][t_in_tile][p];
                sum += s_q * s_k * static_cast<float>(logits_int[p][r]);
            }
            // Apply runtime softmax_scale (matches v2's `score = dot * softmax_scale`).
            logits_fp32[r] = sum * softmax_scale;
        }

        // ─── Sliding window mask (template flag) ────────────────────────
        if constexpr (Cfg::USE_SLIDING_WINDOW) {
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int t_in_tile = (lane % 4) * 2 + (r >> 1);
                int abs_t = tile * TILE_N + t_in_tile;
                // q_rope_pos is the position of the current token; window is
                // [q_rope_pos - sliding_window_size, q_rope_pos].
                if (abs_t < static_cast<int>(q_rope_pos) - sliding_window_size) {
                    logits_fp32[r] = -1e38f;
                }
            }
        }

        // ─── Online softmax (REUSED v2 pattern via OnlineSoftmaxState) ─
        float alpha = softmax_state.update(logits_fp32);
        // logits_fp32 now holds P_fp32 = exp(logit - new_m)

        // ─── P fold-in: multiply by scale_V[t], compute new scale_P ────
        float P_folded[Arch::C_REGS_PER_THREAD];
        float max_abs_P = 0.f;
        #pragma unroll
        for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
            int t_in_tile = (lane % 4) * 2 + (r >> 1);
            float s_v = arena.as_phase4().smem_scale_V[stage][t_in_tile];
            P_folded[r] = s_v * logits_fp32[r];
            float av = fabsf(P_folded[r]);
            if (av > max_abs_P) max_abs_P = av;
        }
        max_abs_P = warp_reduce_max(max_abs_P);
        float scale_P_new = (max_abs_P > 0.f) ? (max_abs_P / 127.f) : 1.f;
        float inv_scale_P = 1.f / scale_P_new;

        int8_t P_int8[Arch::C_REGS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
            float scaled = P_folded[r] * inv_scale_P;
            P_int8[r] = static_cast<int8_t>(__float2int_rn(
                fminf(fmaxf(scaled, -127.f), 127.f)));
        }

        // ─── PV: HEAD_DIM/8 INT8 MMAs ──────────────────────────────────
        int32_t pv_partial[OUT_REGS_PER_THREAD] = {};
        // [TODO phase B: PV inner tile loop with V-frag assembly.]

        // ─── Accumulator update ─────────────────────────────────────────
        #pragma unroll
        for (int r = 0; r < OUT_REGS_PER_THREAD; ++r) {
            out_accum[r] = out_accum[r] * alpha
                         + scale_P_new * static_cast<float>(pv_partial[r]);
        }

        bar_arrive(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    // ───────────────────────────────────────────────────────────────────
    // Output writeback
    // ───────────────────────────────────────────────────────────────────
    float inv_l = softmax_state.normalizer();

    // Each warp writes its OUT_DIMS_PER_WARP slice of one Q head.
    // The partitioning determines which (q_head_idx, dim_offset) range this
    // warp owns. For HEAD_DIM=128 with 4 warps each owning 32 dims:
    //   W4: dims  0..31
    //   W5: dims 32..63
    //   W6: dims 64..95
    //   W7: dims 96..127
    // (Each warp covers one quarter of every Q head's output.)
    //
    // Output tensor: [num_active_slots, n_q_head, HEAD_DIM] in O (FP16/BF16).
    // Address: out + (slot_idx * n_q_head + head_idx) * HEAD_DIM + dim_offset.
    // (Same indexing as v2's writeback in paged_decode_attn_v2_impl.)
    //
    // Reuse: the from_f32<O> conversion from convert_all.cuh.
    //
    // [TODO phase B: precise lane→(q_head, dim) indexing. The M-axis of the
    // MMA is the query axis (q_head × batch dim packed); lane mapping for
    // the M-direction follows the C-fragment layout. Use:
    //   int64_t out_base = ((int64_t)slot_idx * n_q_head + head_idx) * HEAD_DIM;
    //   out[out_base + dim] = from_f32<O>(out_accum[r] * inv_l);
    // ]

    (void)inv_l;
    (void)out;
    (void)n_q_head;
    (void)slot_idx;
}

} // namespace fused_attn
```

> **Note on consumer_role TODOs.** The five `phase B` placeholders are: (1) Q-fragment assembly per palette, (2) PV inner loop with V-fragment assembly, (3) output writeback addressing, (4) the lane→dim index mapping for the C-fragment, (5) RoPE register packing in/out of v2's expected lane layout. All are mechanical work that gets nailed down by validation against numerical reference. The structural skeleton including the v2-reused softmax pattern and the deferred-scaling math is complete.

---

## File 10: `dequant_role.cuh`

The dequant warp main function. **Reuses `write_regs_to_arena` and `write_regs_to_r16` from v2** for phase 3 scatter — this is a major reuse win since v2's scatter logic is exactly what we need.

```cuda
#pragma once
#include "model_descriptor.cuh"
#include "smem_arena.cuh"
#include "dequant_store.cuh"
#include "cp_async.cuh"
#include "../attn_v2.cuh"               // for write_regs_to_arena, write_regs_to_r16
#include "../slot_types.cuh"             // for slot accessors
#include "../arena_table.cuh"            // for ArenaFormat constants

namespace fused_attn {

// =============================================================================
// dequant_role<Cfg, Arch>
//
// W2 (is_k_warp=true):   K path with RoPE + per-(t,p) re-quant scale
// W3 (is_k_warp=false):  V path, no RoPE
//
// REUSES from v2:
//   - write_regs_to_arena<VEC>: scatter FP32 register vector to arena.
//   - write_regs_to_r16<VEC>:   R16 (FP16 K + Q-capture) scatter for write slice.
//   - PalIter:                  palette routing (already imported via dequant_store).
//   - rope helpers:             via apply_rope_dispatch.
// =============================================================================

template<typename Cfg, typename Arch, typename Q_T>
__device__ void dequant_role(
    bool                    is_k_warp,
    int                     lane,
    SmemArena<Cfg>&         arena,
    const float*            rope_cs_table,
    uint32_t                q_rope_pos,         // = ws_rope + ws_len, RoPE position for new K_new
    int                     n_kv_tiles,
    // ─── Slot machinery (from kernel args) ─────────────────────────────
    uint8_t*                write_slice_ptr,
    uint64_t                slices_ptr,
    int                     write_slice_idx,
    int                     n_slices,
    int                     kv_head_idx,
    int                     n_kv_head,
    int                     slot_idx,
    // ─── Activation passthrough for R16 Q-capture ──────────────────────
    const Q_T*              q_for_r16_capture,  // Q tensor for R16 write slice (typed)
    int                     n_q_head
) {
    using namespace fused_attn::tile;
    constexpr int HEAD_DIM = Cfg::HEAD_DIM;
    constexpr int VEC      = HEAD_DIM / WARP_SIZE;  // 4 for HD=128

    // ───────────────────────────────────────────────────────────────────
    // PHASE 2: dequant W_qkv → INT8 staging
    // ───────────────────────────────────────────────────────────────────
    constexpr int N_K_CHUNKS  = Cfg::D_MODEL / Arch::MMA_K;

    for (int k_chunk = 0; k_chunk < N_K_CHUNKS; ++k_chunk) {
        int stage = k_chunk % N_W_STAGING_STAGES;

        bar_sync(bar_id::W_OR_KV_LOADED, /*participants=*/2 * 32 + 2 * 32);

        // [Phase A: dequant W_qkv Q4 → INT8 in w_staging_int8.]
        // For W_qkv, the input scale flows through w_staging_scales (parallel
        // FP32 track) and the INT8 output is the raw nibble-centered value.
        // Each warp handles half the output dims: W2 lower, W3 upper.

        bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
        bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    bar_sync(bar_id::PHASE_2_TO_3, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 3: K_new RoPE + scatter (W2) / V_new scatter (W3)
    //
    // K_new and V_new were written by consumer warps to k_new_fp32 / v_new_fp32
    // in smem during phase 2. Now we apply RoPE (K only) and scatter to the
    // arena, and also re-quantize to INT8 for the tile-0 fast path.
    //
    // The scatter logic mirrors v2's "warp 0" block in paged_decode_attn_v2_impl
    // exactly — same identity-palette routing, same R16/float arena dispatch,
    // same Q-capture for R16 write slice.
    // ───────────────────────────────────────────────────────────────────
    if (is_k_warp) {
        // ── K_new path ─────────────────────────────────────────────────
        const uint16_t ws_offset = slice_offset(write_slice_ptr);
        const uint16_t ws_len    = slice_len(write_slice_ptr);
        const int      within    = static_cast<int>(ws_offset)
                                 + static_cast<int>(ws_len);

        if (within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr,
                                                          kv_head_idx);

            constexpr int LANES_PER_PAL  = WARP_SIZE / Cfg::N_PALETTE;
            constexpr int SUB_HEAD_DIM   = HEAD_DIM / Cfg::N_PALETTE;
            int pal        = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;

            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
            int      k_fmt   = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);

            if (k_ptr_p != 0) {
                char* k_arena = (char*)(uintptr_t)k_ptr_p;

                // Load K_new from smem and apply RoPE.
                float k_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    int dim = lane * VEC + j;
                    k_regs[j] = arena.as_phase12().k_new_fp32[dim];
                }

                apply_rope_dispatch<HEAD_DIM, VEC, Cfg::ROPE_STYLE,
                                    Cfg::ROPE_INTERLEAVED>(
                    k_regs, lane, static_cast<int>(q_rope_pos), rope_cs_table);

                // Scatter to arena — REUSES v2's write_regs_to_arena / write_regs_to_r16.
                if (k_fmt == ArenaFormat::R16) {
                    // R16: dual-write K + Q (Q-capture for the write slice).
                    // Q is in the input tensor at slot_idx, kv_head_idx*GQA.
                    int heads_per_group = n_q_head / n_kv_head;
                    if (heads_per_group < 1) heads_per_group = 1;
                    int q_head = kv_head_idx * heads_per_group;
                    // [Q capture: load Q for this warp's lane positions.]
                    float q_regs[VEC];
                    // [TODO: load q_regs from q_for_r16_capture at the right offset.]

                    // REUSED v2 helper.
                    write_regs_to_r16<VEC>(k_arena, /*chunk_byte_offset=*/0,
                                            within, local_lane, k_regs, q_regs);
                } else {
                    int k_esz = ArenaFormat::float_elem_size(k_fmt);
                    if (k_esz > 0) {
                        int64_t eo = static_cast<int64_t>(within) * SUB_HEAD_DIM;
                        // REUSED v2 helper.
                        write_regs_to_arena<VEC>(k_arena, eo, local_lane,
                                                  k_esz, k_fmt, k_regs);
                    }
                }

                // Also write K_new INT8 form to SMEM_int8_K stage 0 for the
                // tile-0 fast path.
                //
                // For tile 0, we want the consumer to read this directly without
                // going through cp.async + dequant. Compute per-palette scale
                // for K_new (max-abs across the warp), re-quant to INT8.
                //
                // [Phase A: re-quant K_new to INT8 with new per-palette scale,
                // write to smem_int8_K stage 0 + smem_scale_K_post stage 0.]
            }
        }

    } else {
        // ── V_new path ─────────────────────────────────────────────────
        const uint16_t ws_offset = slice_offset(write_slice_ptr);
        const uint16_t ws_len    = slice_len(write_slice_ptr);
        const int      within    = static_cast<int>(ws_offset)
                                 + static_cast<int>(ws_len);

        if (within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr,
                                                          kv_head_idx);

            constexpr int LANES_PER_PAL = WARP_SIZE / Cfg::N_PALETTE;
            constexpr int SUB_HEAD_DIM  = HEAD_DIM / Cfg::N_PALETTE;
            int pal        = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;

            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
            int      v_fmt   = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);

            if (v_ptr_p != 0) {
                char* v_arena = (char*)(uintptr_t)v_ptr_p;

                // Load V_new from smem (no RoPE).
                float v_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    int dim = lane * VEC + j;
                    v_regs[j] = arena.as_phase12().v_new_fp32[dim];
                }

                int v_esz = ArenaFormat::float_elem_size(v_fmt);
                if (v_esz > 0) {
                    int64_t eo = static_cast<int64_t>(within) * SUB_HEAD_DIM;
                    // REUSED v2 helper.
                    write_regs_to_arena<VEC>(v_arena, eo, local_lane,
                                              v_esz, v_fmt, v_regs);
                }

                // Also re-quant V_new to INT8 for tile-0 fast path.
                // [Phase A: same as K_new but no RoPE.]
            }
        }
    }

    bar_sync(bar_id::PHASE_3_TO_4, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 4: per-tile dequant
    //
    // For tile 0, the fast path: K_new/V_new are already in SMEM_int8 from
    // phase 3 above, so signal ready and skip.
    //
    // For tiles ≥ 1: standard dequant + RoPE (K only) + transpose.
    // ───────────────────────────────────────────────────────────────────
    for (int tile = 0; tile < n_kv_tiles; ++tile) {
        if (tile == 0) {
            bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
            bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
            continue;
        }

        int stage = tile % N_PIPELINE_STAGES;

        // The per-tile slice pointer/format setup. This walks slot machinery
        // analogously to v2's load_tile lambda.
        int k_base = tile * TILE_N;  // first global K position in this tile
        int my_slice_idx = chunk_div(k_base);
        int tile_within_base = chunk_mod(k_base);

        if (my_slice_idx >= n_slices) {
            // Past end — signal done, skip.
            bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
            bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
            continue;
        }

        const uint8_t* sl       = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx,
                                                       n_kv_head);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);

        bar_sync(bar_id::W_OR_KV_LOADED, /*participants=*/2 * 32 + 2 * 32);

        if (is_k_warp) {
            // K dequant + RoPE + re-quant + transpose to k-major.
            int8_t* dst_int8 = &arena.as_phase4().smem_int8_K[stage][0][0];
            float*  scales_out = &arena.as_phase4()
                .smem_scale_K_post[stage][0][0];
            const uint32_t* rope_pos = &arena.as_phase4()
                .k_rope_positions[stage][0];

            dequant_kv_tile_K<Cfg>(
                head_ptr,
                tile,
                tile_within_base,
                dst_int8,
                /*dst_dim_stride=*/TILE_N,
                scales_out,
                rope_cs_table,
                rope_pos,
                lane,
                /*warp_in_pool=*/0);

        } else {
            // V dequant + transpose to mn-major (no RoPE).
            int8_t* dst_int8 = &arena.as_phase4().smem_int8_V[stage][0][0];
            float*  scales_per_token = &arena.as_phase4().smem_scale_V[stage][0];

            dequant_kv_tile_V<Cfg>(
                head_ptr,
                tile,
                tile_within_base,
                dst_int8,
                /*dst_token_stride=*/HEAD_DIM,
                scales_per_token,
                lane,
                /*warp_in_pool=*/0);
        }

        bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
        bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }
}

} // namespace fused_attn
```

---

## File 11: `loader_role.cuh`

The loader. Pure cp.async dispatch driven by descriptor queue. **Reuses v2's `cp_async_commit<USE_TC>` and `cp_async_wait<N, USE_TC>`.**

```cuda
#pragma once
#include "model_descriptor.cuh"
#include "smem_arena.cuh"
#include "cp_async.cuh"
#include "../attn_v2.cuh"  // for cp_async_commit<true>, cp_async_wait<N, true>

namespace fused_attn {

template<typename Cfg, typename Arch>
__device__ void loader_role(
    int                     warp_in_pool,
    int                     lane,
    const LoadDescriptor*   queue,
    int                     n_queue_entries
) {
    // Round-robin partition: W0 takes even entries, W1 takes odd.
    for (int i = warp_in_pool; i < n_queue_entries; i += 2) {
        const LoadDescriptor& desc = queue[i];

        if (desc.free_barrier != BARRIER_NONE) {
            bar_sync(desc.free_barrier, desc.sync_count);
        }

        int n_chunks = desc.bytes / 16;
        for (int c = lane; c < n_chunks; c += 32) {
            void*       dst_chunk = static_cast<char*>(desc.dst_smem) + c * 16;
            const void* src_chunk = static_cast<const char*>(desc.src_vram) + c * 16;
            cp_async_cg_16(dst_chunk, src_chunk);
        }

        // REUSED v2 helpers.
        cp_async_commit</*USE_TC=*/true>();
        cp_async_wait<tile::N_PIPELINE_STAGES - 1, /*USE_TC=*/true>();

        bar_arrive(desc.ready_barrier, desc.sync_count);
    }

    cp_async_wait<0, /*USE_TC=*/true>();
}

} // namespace fused_attn
```

---

## File 12: `attn_fused_v1.cuh`

Top-level kernel. **API mirrors v2's `paged_decode_attn_v2_impl` and `paged_decode_v2_kernel`** as closely as possible. The differences are exactly what's required by fusion: Q is no longer pre-projected (we take `activations` instead), and `k_new`/`v_new` are no longer inputs (we compute them from `w_qkv` instead). Everything else — slot machinery, output layout, RoPE table, softmax_scale, runtime args — is identical to v2.

```cuda
#pragma once
#include "model_descriptor.cuh"
#include "arch_traits.cuh"
#include "smem_arena.cuh"
#include "cp_async.cuh"
#include "loader_role.cuh"
#include "dequant_role.cuh"
#include "consumer_role.cuh"
#include "../attn_v2.cuh"          // for commit_decode_write_len_kernel
#include "../slot_types.cuh"        // for SlotHeader, get_slot_header

namespace fused_attn {

// =============================================================================
// fused_qkv_attn_kernel<Q_T, O, Cfg, Arch>
//
// API ALIGNMENT WITH v2:
//
//   v2's paged_decode_v2_kernel signature:
//
//     void paged_decode_v2_kernel(
//         const Q_T* q,                         // pre-projected, pre-RoPE
//         const uint8_t* headers_ptr,
//         O* out,
//         int num_active_slots,
//         int n_q_head,
//         int n_kv_head,
//         float softmax_scale,
//         const T* k_new,                       // pre-projected
//         const T* v_new,                       // pre-projected
//         const float* rope_cs);
//
//   This kernel's signature differs ONLY in the I/O surface that fusion
//   eliminates and the new feature flag:
//     - REMOVED: q (replaced by activations)
//     - REMOVED: k_new, v_new (computed from w_qkv internally)
//     - ADDED:   activations, w_qkv_q4, w_qkv_scales (the QKV projection inputs)
//     - ADDED:   sliding_window_size (only used when Cfg::USE_SLIDING_WINDOW)
//
// All shared runtime args (headers_ptr, out, num_active_slots, n_q_head,
// n_kv_head, softmax_scale, rope_cs) keep v2's exact name and meaning.
//
// Templates:
//     Q_T = activation/Q dtype (FP16 or BF16). v2 calls this Q_T as well.
//     O   = output dtype (FP16, BF16, FP32). Matches v2.
//     Cfg = ModelDescriptor (compile-time shape). Replaces v2's HEAD_DIM
//           plus runtime n_q_head/n_kv_head/d_model with one compile-time bundle.
//     Arch = ArchTraits<SM_VERSION>.
//
// Grid: (num_active_slots, n_kv_head) — same as v2.
// Block: 256 threads (8 warps) — fixed. v2 dispatches 8 or 16 warps; we
//        always use 8 because the warp specialization model is fixed.
// =============================================================================

template<typename Q_T, typename O, typename Cfg, typename Arch>
__global__ __launch_bounds__(256, 2)
void fused_qkv_attn_kernel(
    // ─── QKV projection inputs (replaces v2's `q`, `k_new`, `v_new`) ───
    const Q_T*     activations,         // [num_active_slots, d_model]
    const uint8_t* w_qkv_q4,             // Q4_0 packed weights for QKV proj
    const void*    w_qkv_scales,         // FP16 per-block scales for w_qkv

    // ─── Slot machinery (SAME as v2) ───────────────────────────────────
    const uint8_t* headers_ptr,          // SlotHeader array
    O*             out,                  // [num_active_slots, n_q_head, HD]
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,

    // ─── Attention parameters (SAME as v2) ─────────────────────────────
    float          softmax_scale,
    const float*   rope_cs,              // cos/sin table

    // ─── New feature args (only used when corresponding Cfg flag set) ──
    int            sliding_window_size   // only used if Cfg::USE_SLIDING_WINDOW
) {
    int tid  = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    // Slot/kv-head indexing — naming and grid layout match v2 exactly.
    int slot_idx    = blockIdx.x;
    int kv_head_idx = blockIdx.y;

    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    // Read slot header (REUSES v2's get_slot_header).
    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);

    // Mirror v2's empty-slot handling: write zeros and exit.
    if (slot.n_slices == 0) {
        // [Phase B: same zero-write path as v2's early return.
        //  For each consumer warp's owned (q_head, dim) slice:
        //    out[slot_idx * n_q_head + head_idx][dim] = from_f32<O>(0.f);
        // ]
        return;
    }

    uint8_t* write_slice_ptr = get_slice_mut<Cfg::HEAD_DIM>(
        slot.slices_ptr, slot.write_slice, n_kv_head);

    // The Q RoPE position is derived inside the kernel exactly like v2 does:
    //   uint32_t q_rope_pos = ws_rope + ws_len.
    // (v2 calls this `q_rope_pos` in the apply-rope step.)
    const uint32_t ws_rope    = slice_rope(write_slice_ptr);
    const uint16_t ws_len     = slice_len(write_slice_ptr);
    const uint32_t q_rope_pos = ws_rope + ws_len;

    // Smem arena.
    static_assert(smem_arena_fits_default<Cfg>(),
        "Smem arena exceeds 48 KB default. Trim or use opt-in 64 KB mode.");

    __shared__ SmemArena<Cfg> arena;

    // Build load descriptor queue.
    constexpr int MAX_QUEUE_ENTRIES =
          1                                  // activations
        + (Cfg::D_MODEL / Arch::MMA_K) * 2   // W_qkv chunks × 2 stages
        + 256 * 4;                            // KV tiles up to 256 × 4 streams

    __shared__ LoadDescriptor queue[MAX_QUEUE_ENTRIES];
    __shared__ int            queue_count;

    if (tid == 0) {
        queue_count = build_load_queue<Cfg, Arch, Q_T>(
            activations, w_qkv_q4, w_qkv_scales,
            slot, slot_idx, kv_head_idx, n_kv_head,
            queue);
    }
    __syncthreads();

    // Tile count: kv_len = ws_rope + ws_len + 1 (mirrors v2), then ceil-div.
    int kv_len      = static_cast<int>(q_rope_pos) + 1;
    if (kv_len > static_cast<int>(slot.n_slices) * CHUNK_SIZE)
        kv_len = static_cast<int>(slot.n_slices) * CHUNK_SIZE;
    int n_kv_tiles  = (kv_len + tile::TILE_N - 1) / tile::TILE_N;

    // Dispatch threads to roles.
    if (warp < 2) {
        loader_role<Cfg, Arch>(
            /*warp_in_pool=*/warp,
            lane,
            queue,
            queue_count);

    } else if (warp < 4) {
        dequant_role<Cfg, Arch, Q_T>(
            /*is_k_warp=*/(warp == 2),
            lane,
            arena,
            rope_cs,
            q_rope_pos,
            n_kv_tiles,
            write_slice_ptr,
            slot.slices_ptr,
            static_cast<int>(slot.write_slice),
            static_cast<int>(slot.n_slices),
            kv_head_idx,
            n_kv_head,
            slot_idx,
            activations,        // for R16 Q-capture (typed Q_T)
            n_q_head);

    } else {
        consumer_role<Cfg, Arch, O>(
            /*warp_in_pool=*/(warp - 4),
            lane,
            arena,
            rope_cs,
            q_rope_pos,
            softmax_scale,
            sliding_window_size,
            n_kv_tiles,
            slot_idx,
            n_q_head,
            out);
    }
}

// =============================================================================
// build_load_queue — populate the descriptor queue.
//
// Walks slot machinery for KV tile sources + builds phase-2 entries from
// the QKV projection inputs. Same overall shape as v2's load_tile lambda's
// per-tile address computation, but pre-resolved at kernel entry.
// =============================================================================

template<typename Cfg, typename Arch, typename Q_T>
__device__ int build_load_queue(
    const Q_T*         activations,
    const uint8_t*     w_qkv_q4,
    const void*        w_qkv_scales,
    const SlotHeader&  slot,
    int                slot_idx,
    int                kv_head_idx,
    int                n_kv_head,
    LoadDescriptor*    queue
) {
    int count = 0;

    // ─── Phase 2 entries ───────────────────────────────────────────────
    // 1. Activation vector (one transfer).
    // 2. W_qkv chunks (one per K-chunk, double-buffered).
    //
    // [Phase A: populate from activations and w_qkv_q4 with the right barrier
    //  IDs (W_OR_KV_LOADED, W_OR_KV_CONSUMED).]

    // ─── Phase 4 entries ───────────────────────────────────────────────
    // For each KV tile (skip tile 0 — fast-path from K_new/V_new in smem):
    //   - K data (Q4 packed) → smem_q_K[stage]
    //   - V data (Q4 packed) → smem_q_V[stage]
    //   - K pre-RoPE scales  → smem_scale_K_pre[stage]
    //   - V scales           → smem_scale_V[stage]
    //   - K rope positions   → k_rope_positions[stage]
    //
    // The slice walk is identical in spirit to v2's load_tile lambda which
    // iterates over (slice, within-chunk position). Reuse the same chunk_div
    // / chunk_mod helpers and slice accessors.

    return count;
}

} // namespace fused_attn
```

---

## File 13: `launch.cu`

Launch shim. **API mirrors v2's `launch_paged_decode_attn`** exactly, with the same template parameter ordering (`Q_T`, `O`, then shape) and the same runtime argument list (with k_new/v_new replaced by activations/w_qkv).

```cuda
#pragma once
#include "attn_fused_v1.cuh"
#include "model_descriptor.cuh"
#include "../attn_v2.cuh"  // for commit_decode_write_len_kernel

namespace fused_attn {

// =============================================================================
// launch_fused_qkv_attn<Q_T, O, Cfg, SM_VERSION>
//
// Drop-in counterpart to v2's launch_paged_decode_attn. Side-by-side comparison:
//
//   v2:
//     launch_paged_decode_attn<Q_T, T, O, HEAD_DIM>(
//         q, headers_ptr, out,
//         num_active_slots, n_q_head, n_kv_head,
//         softmax_scale,
//         k_new, v_new, rope_cs,
//         rope_interleaved, stream);
//
//   This:
//     launch_fused_qkv_attn<Q_T, O, Cfg, SM_VERSION>(
//         activations, w_qkv_q4, w_qkv_scales,
//         headers_ptr, out,
//         num_active_slots, n_q_head, n_kv_head,
//         softmax_scale, rope_cs,
//         sliding_window_size, stream);
//
// Differences (intentional, fusion-driven):
//   - q → activations (Q is no longer pre-projected)
//   - k_new/v_new → w_qkv_q4/w_qkv_scales (KV is no longer pre-projected)
//   - T template param dropped (KV cache dtype is per-palette runtime in arena)
//   - HEAD_DIM → Cfg (carries N_Q_HEADS, N_KV_HEADS, D_MODEL, etc. as well)
//   - rope_interleaved → Cfg::ROPE_INTERLEAVED (compile-time per model)
//   - sliding_window_size new (ignored if !Cfg::USE_SLIDING_WINDOW)
//
// All other args are byte-identical to v2.
//
// Launch sequence is identical to v2:
//   1. Main kernel.
//   2. commit_decode_write_len_kernel<HEAD_DIM> on the same stream.
// =============================================================================

template<typename Q_T, typename O, typename Cfg, int SM_VERSION>
cudaError_t launch_fused_qkv_attn(
    // QKV projection inputs (replaces v2's q/k_new/v_new)
    const Q_T*     activations,
    const uint8_t* w_qkv_q4,
    const void*    w_qkv_scales,

    // Slot machinery (SAME as v2)
    const uint8_t* headers_ptr,
    O*             out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,

    // Attention parameters (SAME as v2)
    float          softmax_scale,
    const float*   rope_cs,

    // New feature arg (replaces v2's `rope_interleaved`, which moved to Cfg)
    int            sliding_window_size,

    cudaStream_t   stream = nullptr
) {
    using Arch = ArchTraits<SM_VERSION>;

    // Sanity check: runtime n_q_head/n_kv_head must match the compile-time
    // descriptor. (We accept them at runtime for v2-compatible API but the
    // template binary is specialized for one shape.)
    assert(n_q_head  == Cfg::N_Q_HEADS);
    assert(n_kv_head == Cfg::N_KV_HEADS);

    dim3 grid(num_active_slots, n_kv_head);
    dim3 block(256);  // 8 warps × 32 lanes

    fused_qkv_attn_kernel<Q_T, O, Cfg, Arch><<<grid, block, 0, stream>>>(
        activations, w_qkv_q4, w_qkv_scales,
        headers_ptr, out,
        num_active_slots, n_q_head, n_kv_head,
        softmax_scale, rope_cs,
        sliding_window_size);

    // Post-attention ws.len commit (REUSED from v2 verbatim, same launch
    // pattern as v2's launch_paged_decode_attn).
    constexpr int COMMIT_THREADS = 128;
    dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
    commit_decode_write_len_kernel<Cfg::HEAD_DIM>
        <<<commit_grid, COMMIT_THREADS, 0, stream>>>(
            headers_ptr, num_active_slots, n_kv_head);

    return cudaGetLastError();
}

// =============================================================================
// Concrete launchers — keyed by SHAPE, not by model.
//
// Each unique (shape × feature flags × architecture) combination produces one
// kernel binary with a deterministic name encoding its specialization. Multiple
// models that share a shape map to the same launcher.
//
// The macro takes the shape parameters directly and constructs the
// ModelDescriptor inline. There are no model-named typedefs.
//
// Naming convention:
//
//   launch_fused_attn_h{HD}_q{NQ}_kv{NKV}_d{DM}{_FLAGS}_sm{SM}
//
// where {_FLAGS} is empty for the no-flag base case, and a series of
// underscored tags for any non-default flags:
//
//   _ilv  = ROPE_INTERLEAVED
//   _prl  = ROPE_STYLE = Partial
//   _qkn  = USE_QK_NORM
//   _swin = USE_SLIDING_WINDOW
//
// Examples:
//
//   launch_fused_attn_h128_q32_kv4_d2048_sm89        — Qwen3-30B-A3B shape
//   launch_fused_attn_h128_q32_kv8_d4096_sm89        — Llama-3.1-8B / Mistral-7B-base / Qwen2.5-7B
//   launch_fused_attn_h128_q32_kv8_d4096_swin_sm89   — Mistral-7B with sliding window
//   launch_fused_attn_h128_q24_kv8_d3072_sm89        — Llama-3.2-3B
// =============================================================================

#define DEFINE_SHAPE_LAUNCHER(                                                       \
    HEAD_DIM, NQH, NKVH, DMODEL,                                                     \
    RSTYLE, RILVD, QKN, SW,                                                          \
    SM, NAME)                                                                        \
    template<typename Q_T, typename O>                                               \
    cudaError_t NAME(                                                                \
        const Q_T*     activations,                                                  \
        const uint8_t* w_qkv_q4,                                                     \
        const void*    w_qkv_scales,                                                 \
        const uint8_t* headers_ptr,                                                  \
        O*             out,                                                          \
        int            num_active_slots,                                             \
        int            n_q_head,                                                     \
        int            n_kv_head,                                                    \
        float          softmax_scale,                                                \
        const float*   rope_cs,                                                      \
        int            sliding_window_size,                                          \
        cudaStream_t   stream = nullptr                                              \
    ) {                                                                              \
        using Cfg = ModelDescriptor<                                                 \
            HEAD_DIM, NQH, NKVH, DMODEL, /*N_PALETTE=*/4,                            \
            RopeStyle::RSTYLE, /*USE_QK_NORM=*/QKN,                                  \
            /*USE_SLIDING_WINDOW=*/SW, /*ROPE_INTERLEAVED=*/RILVD>;                  \
        return launch_fused_qkv_attn<Q_T, O, Cfg, SM>(                               \
            activations, w_qkv_q4, w_qkv_scales,                                     \
            headers_ptr, out,                                                        \
            num_active_slots, n_q_head, n_kv_head,                                   \
            softmax_scale, rope_cs,                                                  \
            sliding_window_size, stream);                                            \
    }

// ─── sm_89 (Ada — primary dev target) ────────────────────────────────────────
//
// Each unique shape gets exactly one launcher. Multiple models can share each.

// h128, q32, kv4, d2048: Qwen3-30B-A3B (and any future model with this shape).
DEFINE_SHAPE_LAUNCHER(128, 32, 4, 2048, Full, false, false, false, 89,
                       launch_fused_attn_h128_q32_kv4_d2048_sm89)

// h128, q32, kv8, d4096: Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3-base, etc.
DEFINE_SHAPE_LAUNCHER(128, 32, 8, 4096, Full, false, false, false, 89,
                       launch_fused_attn_h128_q32_kv8_d4096_sm89)

// h128, q32, kv8, d4096 + sliding window: Mistral-7B-v0.1.
DEFINE_SHAPE_LAUNCHER(128, 32, 8, 4096, Full, false, false, true,  89,
                       launch_fused_attn_h128_q32_kv8_d4096_swin_sm89)

// h128, q24, kv8, d3072: Llama-3.2-3B.
DEFINE_SHAPE_LAUNCHER(128, 24, 8, 3072, Full, false, false, false, 89,
                       launch_fused_attn_h128_q24_kv8_d3072_sm89)

// h128, q16, kv8, d2048: Llama-3.2-1B.
DEFINE_SHAPE_LAUNCHER(128, 16, 8, 2048, Full, false, false, false, 89,
                       launch_fused_attn_h128_q16_kv8_d2048_sm89)

// h128, q64, kv8, d8192: Llama-3.1-70B.
DEFINE_SHAPE_LAUNCHER(128, 64, 8, 8192, Full, false, false, false, 89,
                       launch_fused_attn_h128_q64_kv8_d8192_sm89)

// h128, q32, kv32, d3072: Phi-3-mini (no GQA, n_q == n_kv).
DEFINE_SHAPE_LAUNCHER(128, 32, 32, 3072, Full, false, false, false, 89,
                       launch_fused_attn_h128_q32_kv32_d3072_sm89)

// ─── sm_86 (Ampere — RTX 3090) ───────────────────────────────────────────────

DEFINE_SHAPE_LAUNCHER(128, 32, 4, 2048, Full, false, false, false, 86,
                       launch_fused_attn_h128_q32_kv4_d2048_sm86)
DEFINE_SHAPE_LAUNCHER(128, 32, 8, 4096, Full, false, false, false, 86,
                       launch_fused_attn_h128_q32_kv8_d4096_sm86)
// (extend as needed)

// ─── sm_120 (Blackwell — RTX 5080) ───────────────────────────────────────────

DEFINE_SHAPE_LAUNCHER(128, 32, 4, 2048, Full, false, false, false, 120,
                       launch_fused_attn_h128_q32_kv4_d2048_sm120)
DEFINE_SHAPE_LAUNCHER(128, 32, 8, 4096, Full, false, false, false, 120,
                       launch_fused_attn_h128_q32_kv8_d4096_sm120)
// (extend as needed)

#undef DEFINE_SHAPE_LAUNCHER

// =============================================================================
// Single C-callable dispatch entry point.
//
// This is what Rust calls. It takes the kernel shape as runtime args and
// switches over the compiled-in shapes to find a matching launcher. If no
// match exists, returns cudaErrorNotSupported and the caller should fall back
// to v2.
//
// Why a dispatch function:
//   - Rust doesn't need to know the launcher's name. It passes its model's
//     shape and gets a kernel back.
//   - Adding a new shape is one line in the DEFINE_SHAPE_LAUNCHER block above
//     plus one line in the dispatch switch below. No Rust changes needed.
//   - The dispatch is runtime, but the cost is negligible (one switch per
//     kernel launch, dwarfed by the actual launch overhead).
//
// Dtype encoding: Q_T and O dtypes are passed as enum tags. The dispatch
// function instantiates the right (Q_T, O) combination of the matching
// launcher.
// =============================================================================

enum class DType : int {
    F16  = 0,
    BF16 = 1,
    F32  = 2,
};

struct ShapeKey {
    int  head_dim;
    int  n_q_head;
    int  n_kv_head;
    int  d_model;
    bool rope_interleaved;
    int  rope_style;          // 0=Full, 1=Partial
    bool use_qk_norm;
    bool use_sliding_window;
};

extern "C" cudaError_t launch_fused_attn_dispatch(
    // Shape and arch (selects which compiled launcher to call)
    ShapeKey       shape,
    int            sm_version,
    DType          q_dtype,
    DType          o_dtype,

    // Runtime args (passed through to the selected launcher)
    const void*    activations,
    const uint8_t* w_qkv_q4,
    const void*    w_qkv_scales,
    const uint8_t* headers_ptr,
    void*          out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const float*   rope_cs,
    int            sliding_window_size,
    cudaStream_t   stream
) {
    // Helper to encode the shape as a comparable tuple (we just inline the
    // checks; a real implementation might use a hash or sorted array).
    auto shape_matches = [&](int hd, int nq, int nkv, int dm,
                              bool ilv, int rs, bool qkn, bool sw) {
        return shape.head_dim == hd
            && shape.n_q_head == nq
            && shape.n_kv_head == nkv
            && shape.d_model == dm
            && shape.rope_interleaved == ilv
            && shape.rope_style == rs
            && shape.use_qk_norm == qkn
            && shape.use_sliding_window == sw;
    };

    // Type-erase to the right (Q_T, O) instantiation. We support FP16 and
    // BF16 for both Q and O in v1; FP32 output can be added later.
    //
    // The pattern is verbose but unavoidable in C++: we have to enumerate
    // dtype combinations because templates can't be selected from runtime
    // values. Two dtypes × two dtypes = four cases per shape.
    //
    // For brevity below, we show the pattern for one shape; extend uniformly.

    #define DISPATCH_DTYPES(LAUNCHER)                                              \
        do {                                                                       \
            if (q_dtype == DType::F16 && o_dtype == DType::F16) {                  \
                return LAUNCHER<__half, __half>(                                   \
                    static_cast<const __half*>(activations),                       \
                    w_qkv_q4, w_qkv_scales, headers_ptr,                           \
                    static_cast<__half*>(out),                                     \
                    num_active_slots, n_q_head, n_kv_head,                         \
                    softmax_scale, rope_cs, sliding_window_size, stream);          \
            }                                                                      \
            if (q_dtype == DType::BF16 && o_dtype == DType::BF16) {                \
                return LAUNCHER<__nv_bfloat16, __nv_bfloat16>(                     \
                    static_cast<const __nv_bfloat16*>(activations),                \
                    w_qkv_q4, w_qkv_scales, headers_ptr,                           \
                    static_cast<__nv_bfloat16*>(out),                              \
                    num_active_slots, n_q_head, n_kv_head,                         \
                    softmax_scale, rope_cs, sliding_window_size, stream);          \
            }                                                                      \
            return cudaErrorNotSupported;                                          \
        } while (0)

    if (sm_version == 89) {
        if (shape_matches(128, 32, 4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm89);
        if (shape_matches(128, 32, 8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm89);
        if (shape_matches(128, 32, 8, 4096, false, 0, false, true))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_swin_sm89);
        if (shape_matches(128, 24, 8, 3072, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q24_kv8_d3072_sm89);
        if (shape_matches(128, 16, 8, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q16_kv8_d2048_sm89);
        if (shape_matches(128, 64, 8, 8192, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q64_kv8_d8192_sm89);
        if (shape_matches(128, 32, 32, 3072, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv32_d3072_sm89);
    }

    if (sm_version == 86) {
        if (shape_matches(128, 32, 4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm86);
        if (shape_matches(128, 32, 8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm86);
    }

    if (sm_version == 120) {
        if (shape_matches(128, 32, 4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm120);
        if (shape_matches(128, 32, 8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm120);
    }

    #undef DISPATCH_DTYPES

    // No matching launcher compiled in. Caller falls back to v2.
    return cudaErrorNotSupported;
}

} // namespace fused_attn
```

---

## Reuse summary

What we now reuse from the existing codebase:

| Component | Source | Used in |
|---|---|---|
| `cp_async_commit<USE_TC>` | `attn_v2.cuh` | `loader_role` |
| `cp_async_wait<N, USE_TC>` | `attn_v2.cuh` | `loader_role` |
| `rope_cos_sin<HEAD_DIM>` | `attn_v2.cuh` | `apply_rope_dispatch` |
| `apply_rope_rotary_f32<VEC, HD>` | `attn_v2.cuh` | `apply_rope_dispatch` |
| `apply_rope_interleaved_f32<VEC, HD>` | `attn_v2.cuh` | `apply_rope_dispatch` |
| `write_regs_to_arena<VEC>` | `attn_v2.cuh` | `dequant_role` phase 3 K_new/V_new scatter |
| `write_regs_to_r16<VEC>` | `attn_v2.cuh` | `dequant_role` phase 3 R16 write slice scatter |
| `commit_decode_write_len_kernel<HD>` | `attn_v2.cuh` | `launch.cu` post-attention commit |
| `fast_exp::exp2<float, Softmax>` | `fast_exp.cuh` | `softmax_state.cuh` |
| `warp_reduce_sum`, `warp_reduce_max` | `warp_reduce.cuh` | softmax, scale reductions, P fold-in |
| `to_f32<T>`, `from_f32<T>` | `convert_all.cuh` | output writeback (FP32 → FP16/BF16) |
| `vec2_traits<T>`, `load_vec2<T>` | `attn_v2.cuh` | output writeback packed pair store |
| `ArenaAccessor` | `convert_all.cuh` | dequant primitives (with new `load_head_int8_unscaled` method) |
| `PalIter<VEC, HD>` | `pal_iter.cuh` | dequant K/V palette routing |
| `get_slot_header`, `get_slice`, `get_head` | `slot_types.cuh` | top-level kernel + dequant_role |
| `kvhead_k_ptr`, `kvhead_v_ptr`, `kvhead_*_fmt`, `kvhead_*_scale` | `slot_types.cuh` | dequant_role |
| `kvhead_k_pal_map`, `kvhead_v_pal_map` | `slot_types.cuh` | PalIter setup |
| `slice_offset`, `slice_len`, `slice_rope`, `slice_increment_len` | `slot_types.cuh` | scatter, masking, validation |
| `chunk_div`, `chunk_mod` | `slot_types.cuh` | tile→slice resolution |
| `ArenaFormat` constants | `arena_table.cuh` | format dispatch |
| `CHUNK_SIZE` | `arena_table.cuh` | tile bounds |

What's net-new infrastructure:

- `arch_traits.cuh` — MMA shape constants per arch (v2 has no tensor cores)
- `model_descriptor.cuh` — compile-time model parameterization (v2 hardcodes shapes via runtime args)
- `smem_arena.cuh` — phase-based union view (v2 uses simple `__shared__` arrays)
- `mma_wrappers.cuh` — INT8 MMA inline asm (new)
- `cp_async.cuh` — `cp_async_cg_16` named primitive + LoadDescriptor + named barriers (v2 inlines these)
- `dequant_store.cuh` — deferred-scaling dequant primitive (new; depends on the convert_all.cuh addition)
- `softmax_state.cuh` — encapsulates v2's m_i/l_i/alpha/beta pattern as a struct
- `consumer_role.cuh`, `dequant_role.cuh`, `loader_role.cuh` — warp-specialized control flow (new)
- `attn_fused_v1.cuh` — top-level kernel (new)
- `launch.cu` — per-(model, arch) launcher (new)

The one **non-additive change** to the existing codebase is the new `load_head_int8_unscaled` method on `ArenaAccessor` in `convert_all.cuh`. This is purely additive — existing v2 callers continue using `load_head_scaled` unchanged.

## What remains as Phase A/B work

Phase A (foundation, before any kernel test):

1. **`load_head_int8_unscaled` in `convert_all.cuh`** — the new method, ~80 lines, mirrors `load_head_scaled`'s format dispatch with INT8 output instead of T.
2. **MMA fragment lane mapping validation** — confirm `load_a_frag_m16k32_ldmatrix` and `load_b_frag_n8k32_strided` produce the exact byte layout the m16n8k32 instruction expects. Validate via a tiny standalone MMA test.
3. **`build_load_queue` body** — populate the descriptor queue from slot machinery.
4. **`compute_n_kv_tiles`** — derive from `kv_len = ws_rope + ws_len + 1`.
5. **Smem arena W_qkv staging size constants** — work out the exact K-chunk × output-dim sizing.

Phase B (kernel completion):

6. **Phase 2 W_qkv dequant** — the `dequant_role` body for phase 2 (currently a placeholder).
7. **Q-fragment assembly per palette** in `consumer_role` phase 4.
8. **PV inner tile loop** with V-fragment assembly.
9. **Lane→dim mapping for C-fragment** — needed for phase 3 RoPE on Q and output writeback.
10. **K_new / V_new INT8 re-quant for tile-0 fast path** in `dequant_role` phase 3.
11. **Output global writeback** with proper (batch, q_head, dim) addressing.

Phase B-late (correctness + numerical validation):

12. **End-to-end PPL test** vs v2 on Qwen3-30B-A3B at small ctx.
13. **Long-context PPL test** at 8K, 32K, 128K to validate the RoPE re-quant accuracy envelope.
14. **Cross-architecture identical output** on 3090/4090/5080.

The reuse work has materially reduced the implementation surface: rope, scatter, the post-attention commit kernel, all softmax math infrastructure, format dispatch, and slot machinery are all already done. The genuinely-new code is the warp-specialized control flow + INT8 MMA pipeline + the deferred-scaling math, which is the design's actual contribution.