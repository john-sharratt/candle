// SPDX-License-Identifier: MIT
//
// palette4_convert_kernel — KV-cache initial quantization and format conversion
// ==============================================================================
//
// PURPOSE
// -------
// The primary use of this kernel is the initial quantization of a KV-head:
// after the CompressionPolicy has analysed a head's activation statistics and
// selected the optimal per-palette format (fmt), outer scale, and palette map
// (pal_map), this kernel writes the canonical quantized arenas for the first
// time.  The source is typically F16 or F32; the destination is one of the
// ~20 quantized formats (Q4_KS, Q8_0, Q2_0, …).
//
// The kernel also handles format migration — re-quantizing an already-quantized
// head when the compression policy later decides to change levels (e.g. Q8_0 →
// Q4_KS as a block ages out of the hot tier).  This path is less frequent but
// uses the same code; any src/dst format combination is supported.
//
// Both key-side (IS_K=true) and value-side (IS_K=false) heads are handled.
//
// PALETTE MODEL
// -------------
// Each KV-head's 128 dimensions are partitioned into 4 palettes of 32 dims
// each.  The partitioning is stored as a 2-bit-per-dim packed byte array
// (pal_map) chosen by the policy to group dimensions with similar statistics.
// Src and dst heads can have *different* pal_maps — when the format or palette
// assignment changes the boundaries may be redrawn.
//
//   head_dim = 128 dims total, pal_map[byte] = four 2-bit palette ids
//
//   pal_map example (packed):
//     byte 0  = dims 0-3:   bits [1:0]=pal(dim0), [3:2]=pal(dim1), ...
//     dim d → palette  p = (pal_map[d/4] >> (2*(d%4))) & 3
//     dim d → local rank n = number of dims < d in the same palette
//
//   Each palette stores 32 dimensions independently as its own arena:
//     F16/F32/BF16 arena: channel-oriented (dim-minor), num_chunks*32 rows
//     Quant/R16 arena:    token-oriented blocks, 32 tokens per block
//
// DATA FLOW
// ---------
//
//  src arenas (GMEM)                smem_f16_buf             r_buf (regs)
//  ┌──────────┐                  ┌──────────────────┐      ┌──────────┐
//  │ pal 0    │  issue_load(c)   │ [32 tok][128 dim]│  →   │ 16 half2 │
//  │ pal 1    │ ──────────────→  │  all 4 palettes  │      │ 32 toks  │
//  │ pal 2    │  (warp fills     │  deq/cvt to f16  │      │ 1 col    │
//  │ pal 3    │   its 32 cols)   └──────────────────┘      └──────────┘
//  └──────────┘                          │  copy_to_regs()        │
//                                        │  (smem_xlat applied)   │
//                                        └───────────────────────→│
//                                                                  │ encode
//                                                                  ↓
//                                                         dst arenas (GMEM)
//                                                         ┌──────────┐
//                                                         │ pal 0    │
//                                                         │ pal 1    │
//                                                         │ pal 2    │
//                                                         │ pal 3    │
//                                                         └──────────┘
//
// THREAD/WARP ASSIGNMENT
// ----------------------
// Block = 128 threads = 4 warps.  Warp i owns palette i exclusively:
//
//   warp 0 (threads   0-31):  palette 0  — dims   0-31 of the head
//   warp 1 (threads  32-63):  palette 1  — dims  32-63
//   warp 2 (threads  64-95):  palette 2  — dims  64-95
//   warp 3 (threads  96-127): palette 3  — dims  96-127
//
// Within each warp, lane L owns dimension L of that palette (local rank L).
// Thread d = warp_id * 32 + lane.
//
// SMEM LAYOUT (8960 B total)
// --------------------------
//
//   smem_f16_buf [32][128]  = 8192 B   ← F16 staging for one chunk
//   smem_xlat    [128]      =  128 B   ← src smem column for each dst dim
//   smem_qscratch[4][32]    =  512 B   ← fp32 warp scratch for quant encode
//   smem_meta               =  128 B   ← per-palette src/dst format metadata
//                                        (fmt, arena ptr, outer scale)
//   Total: 8960 B
//
//   With cudaSharedmemCarveoutMaxShared (~100 KB on Ada Lovelace):
//     floor(102400 / 8960) = 11 blocks/SM  (smem ceiling)
//   __launch_bounds__(128, 8) hints register budget of 64 regs/thread.
//   The binding occupancy constraint in practice is the register budget.
//
// PIPELINE (register-buffer double-buffering)
// -------------------------------------------
// smem_f16_buf is a single 8 KB buffer.  To overlap DMA of chunk c+1 with
// encode of chunk c, the current chunk's data lives in per-thread registers
// (r_buf) while smem is being overwritten by the next chunk's DMA.
//
//   Timeline (F16 src, any dst):
//
//   ├─ PROLOGUE ──────────────────────────────────────────────────────
//   │  issue_load(0)       DMA or scalar fill of chunk 0 into smem
//   │  cp.async.wait_group 0   wait for chunk 0 to settle
//   │  __syncthreads()         all warps see the complete smem fill
//   │  copy_to_regs()          smem → r_buf (with xlat pre-translation)
//   │  issue_load(1)       kick off DMA for chunk 1 (overlaps encode)
//   │  cp.async.commit_group
//   │
//   ├─ LOOP c=0 ──────────────────────────────────────────────────────
//   │  ENCODE chunk 0 from r_buf          ←── DMA(1) running in smem
//   │  cp.async.wait_group 0              wait for chunk 1 to settle
//   │  __syncthreads()
//   │  copy_to_regs()                     smem(chunk 1) → r_buf
//   │  issue_load(2); cp.async.commit_group
//   │
//   ├─ LOOP c=1 ──────────────────────────────────────────────────────
//   │  ENCODE chunk 1 from r_buf          ←── DMA(2) running in smem
//   │  ...
//
//   r_buf is NEVER read while smem is being overwritten — DMA safety is
//   preserved for both float and quant dst paths.
//
// REGISTER BUDGET (64 regs at __launch_bounds__(128, 8))
// -------------------------------------------------------
//   half2 r_buf[16]:    16 regs (32 tokens packed as 16 half2 pairs)
//   r_dst_outer:         1 reg  (only persistent cold-metadata register)
//   loop vars + temps: ~35 regs
//   __noinline__ encode: fences encode's ~20-reg body from caller frame
//   Prologue scope {}: ~10 regs freed (head/layer indices, pal_map ptrs)
//   Result: REG=64, STACK=128 B confirmed via cuobjdump.
//
// Design: docs/quantize-kernel-rewrite-design.md

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "../blocks.cuh"
#include "../arena_table.cuh"
#include "../paged-decode/slot_types.cuh"
#include "quantize.cuh"
#include "../convert/convert_all.cuh"

// Chunk = 32 tokens per quantization block (shared with Rust CHUNK_SIZE).
// Num palettes = 4 always; the head dim is a template parameter (128 or 256),
// so dims-per-palette is HD/4 — 32 at 128, 64 at 256. At 256 one palette
// spans TWO warps: palette identity is `d / PAL_DIM`, never the warp id.
#define P4C_CHUNK_SIZE 32
#define P4C_NUM_PAL    4

// =============================================================================
// FORMAT CLAMP
// =============================================================================
//
// Before quantizing, input floats are clamped to keep the block scale d in a
// representable range.  The bounds differ by scale type:
//
//   FP8-scale formats (Q0, Q1_S, …): the scale is an E4M3 value whose max is
//     448.  Values above that saturate the scale itself and cause catastrophic
//     error, so we hard-clamp at 448 (or its format-specific multiplier).
//
//   Block-scale formats (Q4_0, Q8_KS, …): scale d is stored as FP16 and can
//     absorb any magnitude; we clamp at 512 (slightly above the typical
//     inv_scale*head_amax ≈ 256) so d stays in FP16 normal range.
//
__device__ __forceinline__ float p4c_format_clamp(int fmt) {
    switch (fmt) {
        case ArenaFormat::Q0:
        case ArenaFormat::Q0_V:
        case ArenaFormat::Q1_A:
        case ArenaFormat::Q0_X:
        case ArenaFormat::Q0_M2:
        case ArenaFormat::Q0_M4:
        case ArenaFormat::Q1_S:
            return 448.0f;
        case ArenaFormat::Q2_S: return 672.0f;
        case ArenaFormat::Q2_A: return 1792.0f;
        default:                return 512.0f;
    }
}

// =============================================================================
// PALETTE MAP HELPERS
// =============================================================================
//
// pal_map is a packed array of 2-bit palette ids, two bits per head dimension:
//
//   byte index:  0          1          ...   31
//   bits:       [76543210] [76543210]
//                ││││││││
//                ││││││└┘── dim 0 palette (bits 1:0)
//                ││││└┘──── dim 1 palette (bits 3:2)
//                ││└┘────── dim 2 palette (bits 5:4)
//                └┘──────── dim 3 palette (bits 7:6)
//
// 128 dims × 2 bits = 256 bits = 32 bytes.

// Return the palette id (0-3) for global head dimension g.
__device__ __forceinline__ int pal_map_get(const uint8_t* pal_map, int g) {
    return (pal_map[g / 4] >> (2 * (g % 4))) & 0x3;
}

// Return the global head dimension that is the n-th member of palette p.
// Used during xlat construction: for dst palette p, slot n → its global dim.
__device__ __forceinline__ int find_nth_dim_in_pal(
    const uint8_t* pal_map, int p, int n, int head_dim
) {
    int count = 0;
    for (int g = 0; g < head_dim; g++) {
        if (pal_map_get(pal_map, g) == p) {
            if (count == n) return g;
            count++;
        }
    }
    return -1;
}

// Return how many dims before global_d belong to palette p — i.e., the local
// rank of global_d within palette p.  Used to find which slot of a src arena
// corresponds to the global dim we need.
__device__ __forceinline__ int rank_in_pal(const uint8_t* pal_map, int p, int global_d) {
    int rank = 0;
    for (int g = 0; g < global_d; g++) {
        if (pal_map_get(pal_map, g) == p) rank++;
    }
    return rank;
}

// =============================================================================
// QUANT HELPERS
// =============================================================================

// Block size in bytes for each quantized format.  Returns 0 for non-quant
// formats (float_elem_size > 0), which the caller uses to branch float vs quant.
__device__ __forceinline__ int p4c_quant_block_bytes(int fmt) {
    switch (fmt) {
        case ArenaFormat::R16:     return sizeof(block_r16);
        case ArenaFormat::Q4_0:    return sizeof(block_q4_0);
        case ArenaFormat::Q4_1:    return sizeof(block_q4_1);
        case ArenaFormat::Q5_0:    return sizeof(block_q5_0);
        case ArenaFormat::Q5_1:    return sizeof(block_q5_1);
        case ArenaFormat::Q8_0:    return sizeof(block_q8_0);
        case ArenaFormat::Q8_1:    return sizeof(block_q8_1);
        case ArenaFormat::Q4_KS:   return sizeof(block_q4_ks);
        case ArenaFormat::Q8_KS:   return sizeof(block_q8_ks);
        case ArenaFormat::Q2_0:    return sizeof(block_q2_0);
        case ArenaFormat::Q3_0:    return sizeof(block_q3_0);
        case ArenaFormat::Q0:      return sizeof(block_q0);
        case ArenaFormat::Q1_S:    return sizeof(block_q1_s);
        case ArenaFormat::Q2_S:    return sizeof(block_q2_s);
        case ArenaFormat::Q2_A:    return sizeof(block_q2_a);
        case ArenaFormat::Q2_1:    return sizeof(block_q2_1);
        case ArenaFormat::Q3_1:    return sizeof(block_q3_1);
        case ArenaFormat::Q0_V:    return sizeof(block_q0_v);
        case ArenaFormat::Q1_A:    return sizeof(block_q1_a);
        case ArenaFormat::Q0_X:    return sizeof(block_q0_x);
        case ArenaFormat::Q0_M2:   return sizeof(block_q0_m2);
        case ArenaFormat::Q0_M4:   return sizeof(block_q0_m4);
        default: return 0;
    }
}

// Dispatch a warp-cooperative 32-element quantization block encode.
//
// __noinline__ is intentional: the 22-format switch body inlines ~20 registers
// of per-format temporaries.  As an inline function those registers would
// merge into the caller's live set and push the total well above the 64-reg
// budget set by __launch_bounds__(128,8).  The __noinline__ boundary lets the
// register allocator treat this as a true call frame — the callee's registers
// are freed on return and never overlap the caller's r_buf[16] + loop vars.
__device__ __noinline__ void p4c_encode_quant_block(
    const float* src32, void* dst_block, int fmt
) {
    switch (fmt) {
        case ArenaFormat::R16:     quantize_block_r16(src32, (block_r16*)dst_block); break;
        case ArenaFormat::Q4_0:    quantize_block_q4_0_vec(src32, (block_q4_0*)dst_block); break;
        case ArenaFormat::Q4_1:    quantize_block_q4_1(src32, (block_q4_1*)dst_block); break;
        case ArenaFormat::Q5_0:    quantize_block_q5_0(src32, (block_q5_0*)dst_block); break;
        case ArenaFormat::Q5_1:    quantize_block_q5_1(src32, (block_q5_1*)dst_block); break;
        case ArenaFormat::Q8_0:    quantize_block_q8_0_vec(src32, (block_q8_0*)dst_block); break;
        case ArenaFormat::Q8_1:    quantize_block_q8_1(src32, (block_q8_1*)dst_block); break;
        case ArenaFormat::Q4_KS:   quantize_block_q4_ks_vec(src32, (block_q4_ks*)dst_block); break;
        case ArenaFormat::Q8_KS:   quantize_block_q8_ks_vec(src32, (block_q8_ks*)dst_block); break;
        case ArenaFormat::Q2_0:    quantize_block_q2_0_vec(src32, (block_q2_0*)dst_block); break;
        case ArenaFormat::Q3_0:    quantize_block_q3_0(src32, (block_q3_0*)dst_block); break;
        case ArenaFormat::Q0:      quantize_block_q0(src32, (block_q0*)dst_block); break;
        case ArenaFormat::Q1_S:    quantize_block_q1_s_vec(src32, (block_q1_s*)dst_block); break;
        case ArenaFormat::Q2_S:    quantize_block_q2_s_vec(src32, (block_q2_s*)dst_block); break;
        case ArenaFormat::Q2_A:    quantize_block_q2_a_vec(src32, (block_q2_a*)dst_block); break;
        case ArenaFormat::Q2_1:    quantize_block_q2_1(src32, (block_q2_1*)dst_block); break;
        case ArenaFormat::Q3_1:    quantize_block_q3_1(src32, (block_q3_1*)dst_block); break;
        case ArenaFormat::Q0_V:    quantize_block_q0_v(src32, (block_q0_v*)dst_block); break;
        case ArenaFormat::Q1_A:    quantize_block_q1_a(src32, (block_q1_a*)dst_block); break;
        case ArenaFormat::Q0_X:    quantize_block_q0_x(src32, (block_q0_x*)dst_block); break;
        case ArenaFormat::Q0_M2:   quantize_block_q0_m2(src32, (block_q0_m2*)dst_block); break;
        case ArenaFormat::Q0_M4:   quantize_block_q0_m4(src32, (block_q0_m4*)dst_block); break;
        default: break;
    }
}

// =============================================================================
// MAIN KERNEL
// =============================================================================
//
// Grid:  dim3(num_kv_heads, num_layers)  — one block per (head, layer) pair.
// Block: 128 threads = 4 warps.
//
// The kernel converts all num_chunks chunks of the head from src to dst.
// A "chunk" is 32 consecutive tokens.  All data is in "arena" memory: a flat
// byte array per (palette, format) combination, whose internal layout depends
// on the format type (float vs quantized — see DATA LAYOUTS below).
//
// DATA LAYOUTS
// ------------
// Float arenas (F16, F32, BF16) — channel-oriented, chunk-major:
//
//   arena_base[chunk c, token t, local dim ld]
//     = arena_base + (c * CHUNK_SIZE * PAL_DIM + t * PAL_DIM + ld) * elem_size
//
//   smem load: thread d handles its own column (ld = lane), reading all
//   32 rows of the chunk → smem_f16_buf[t][d] = value at (c, t, ld).
//
// Quant/R16 arenas — token-oriented, dim-major:
//
//   block(local dim ld, chunk c)
//     = arena_base + (ld * num_chunks + c) * block_bytes
//
//   Within a block, element[t] = token t's value for that dim, quantized
//   together with the other 31 tokens of the same chunk.
//
// XLAT TABLE (smem_xlat)
// ----------------------
// Because src and dst pal_maps may differ, the 32 dims of dst palette p may
// not correspond to the same 32 dims in the src.  smem_xlat[d] records, for
// each dst thread d (= warp_id*32 + lane, owning dst dim `lane` of palette
// warp_id), the smem_f16_buf *column* that holds that dim's source values.
//
//   smem_xlat[d] = sp * PAL_DIM + s_ld
//     where sp   = src palette containing the global dim that dst dim d maps to
//           s_ld = local rank of that global dim within src palette sp
//
// After xlat is built, the mapping src→dst is purely a gather: thread d reads
// smem_f16_buf[*][smem_xlat[d]] regardless of which src palette spilled those
// values.  Warps on different sides of a palette-boundary crossing in the
// pal_map need not know about each other.
//
//   Example (simplified, 8-dim head, 2 dims per palette):
//
//   src pal_map:  dim0→pal0 dim1→pal1 dim2→pal0 dim3→pal1 ...
//   dst pal_map:  dim0→pal0 dim1→pal0 dim2→pal1 dim3→pal1 ...
//
//   smem_f16_buf after issue_load:
//     col 0 = src pal0, rank 0 = src dim 0
//     col 1 = src pal0, rank 1 = src dim 2
//     col 2 = src pal1, rank 0 = src dim 1
//     col 3 = src pal1, rank 1 = src dim 3
//
//   dst thread 0 (warp0, lane0) owns dst dim 0 → global dim 0 → src pal0 rank0 → smem col 0
//   dst thread 1 (warp0, lane1) owns dst dim 1 → global dim 1 → src pal1 rank0 → smem col 2
//   xlat = [0, 2, 1, 3, ...]
//
// REGISTER-BUFFER DOUBLE-BUFFERING
// ---------------------------------
// smem_f16_buf is a single 8 KB buffer (one chunk at a time).  The encode
// stage reads from the per-thread register buffer r_buf, not from smem, so
// the DMA for the next chunk can run in smem concurrently with encode.
//
//   r_buf[k] (half2) = half2(smem[tok_2k][xlat[d]], smem[tok_2k+1][xlat[d]])
//
//   After copy_to_regs(), thread d holds all 32 tokens for its dst column,
//   packed as 16 half2s.  The xlat gather is applied at this copy step, so
//   encode reads r_buf[k] as if it were already transposed and remapped.
//
// QUANT DST WARP TRANSPOSE
// ------------------------
// Quant encode requires 32 *tokens* for one *column* (to compute the block
// scale across the 32 tokens).  After copy_to_regs, the data layout is the
// opposite: each thread d owns all 32 tokens for column d.  A warp-level
// transpose extracts one column at a time via __shfl_sync:
//
//   Before (in r_buf, per thread):
//     thread 0:  r_buf[0..15] = tok0..tok31 for col 0
//     thread 1:  r_buf[0..15] = tok0..tok31 for col 1
//     ...
//     thread 31: r_buf[0..15] = tok0..tok31 for col 31
//
//   To encode col ld (for warp's palette), need scratch[lane] = tok_lane for col ld.
//   Thread `ld` holds all tokens for col ld.  16 shuffles broadcast r_buf[k]
//   from thread ld to all threads; each thread predicate-selects k = lane>>1
//   and extracts the low (even lane) or high (odd lane) half:
//
//     for k in 0..16:
//       all_get half2(tok_2k, tok_2k+1) from thread ld   ← __shfl_sync(_, r_buf[k], ld)
//     thread `lane` picks k = lane>>1:
//       scratch[lane] = tok_lane for col ld  ✓
//
//   After 32 such columns × 16 shuffles = 512 shuffle instructions, plus 32
//   warp-cooperative encode calls, the chunk's quant arena is fully written.
//   DMA overlap is preserved: encode reads r_buf throughout, never smem.

// Blocks/SM target keeping the register budget at 64 regs/thread:
// 65536 / (HD * blocks) = 64  →  8 blocks at HD 128, 4 at HD 256.
__host__ __device__ constexpr int p4c_min_blocks(int hd) { return 1024 / hd; }

template <int HD, bool IS_K>
// __launch_bounds__(HD, p4c_min_blocks(HD)): 1024 resident threads/SM either
// way, 64 regs/thread. smem is not the binding limit at either width
// (~9 KB at 128, ~17.5 KB at 256 under the ~100 KB carveout).
//   Achieved at 128: REG=64, STACK=128 B (verified with cuobjdump).
__global__ void __launch_bounds__(HD, p4c_min_blocks(HD))
palette4_convert_kernel(
    const uint8_t* __restrict__ heads_base,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t num_chunks
) {
    // Dims per palette: 32 at HD 128 (palette == warp, the historical shape),
    // 64 at HD 256 (palette == two warps). PAL_DIM % 32 == 0 always, so a
    // warp never straddles a palette boundary — which is what keeps the
    // warp-cooperative encode below palette-pure.
    constexpr int PAL_DIM = HD / P4C_NUM_PAL;
    static_assert(PAL_DIM % 32 == 0, "a warp must not straddle palettes");
    static_assert(3 * PAL_DIM + (PAL_DIM - 1) <= 255,
                  "smem_xlat packs (palette, rank) into one byte");
    // -------------------------------------------------------------------------
    // SHARED MEMORY
    //
    //  smem_f16_buf: staging area for one loaded chunk.
    //    Layout: [token 0..31][head_dim 0..127]
    //    Warp i fills columns [i*32 .. i*32+31] during issue_load.
    //    All warps read any column during copy_to_regs (via xlat).
    //
    //    smem_f16_buf[token][dim]:
    //      col   0.. 31  ← pal 0 (warp 0 loads)
    //      col  32.. 63  ← pal 1 (warp 1 loads)
    //      col  64.. 95  ← pal 2 (warp 2 loads)
    //      col  96..127  ← pal 3 (warp 3 loads)
    //
    //  smem_xlat: per-dst-dim src smem column (built once in prologue).
    //    smem_xlat[d] = sp * 32 + s_ld  (which column of smem_f16_buf to read)
    //
    //  smem_qscratch: 32-float workspace for one warp's quant encode.
    //    smem_qscratch[warp_id][lane] = token `lane`'s float value for the
    //    current column being encoded.  Filled via warp shuffle, then consumed
    //    by p4c_encode_quant_block.
    //
    //  smem_meta: cold per-palette metadata, written once in prologue and
    //    reloaded at use sites so it never needs a long-lived register.
    // -------------------------------------------------------------------------
    __shared__ __half smem_f16_buf[P4C_CHUNK_SIZE][HD];   // 8 KB @128, 16 KB @256
    __shared__ uint8_t smem_xlat[HD];
    __shared__ float smem_qscratch[HD / 32][32];          // one row per WARP
    __shared__ struct {
        int      src_fmt  [P4C_NUM_PAL];   //  16 B  — source arena format id
        uint64_t src_arena[P4C_NUM_PAL];   //  32 B  — source arena base pointer
        float    src_outer[P4C_NUM_PAL];   //  16 B  — source outer (head) scale
        int      dst_fmt  [P4C_NUM_PAL];   //  16 B  — destination arena format id
        uint64_t dst_arena[P4C_NUM_PAL];   //  32 B  — destination arena base pointer
        float    dst_outer[P4C_NUM_PAL];   //  16 B  — destination outer (head) scale
    } smem_meta;                            // 128 B

    // Thread identity. `pal` (not the warp id) indexes every per-palette
    // structure; `pld` is the dim within the palette; `lane` keeps its two
    // warp-local roles (token row in the float loads, shuffle lane in the
    // quant encode). At HD 128 pal == warp and pld == lane — the historical
    // identities — so the 128 instantiation is bit-identical to the old kernel.
    const int d       = threadIdx.x;         // global thread index, 0..HD-1
    const int warp_id = d >> 5;              // warp, 0..HD/32-1
    const int lane    = d & 31;              // lane within the warp
    const int pal     = d / PAL_DIM;         // palette index, 0..3
    const int pld     = d % PAL_DIM;         // dimension within palette

    // =========================================================================
    // PROLOGUE — populate smem_meta and build smem_xlat.
    //
    // All pointer and index variables (head_idx, src_head, pal_map pointers,
    // global_d, etc.) are declared inside this brace-delimited scope so the
    // compiler can prove they are dead before the chunk loop.  The register
    // allocator reuses those ~10 slots for r_buf[16] and loop temporaries,
    // keeping the live-register count within the 64-reg budget.
    // =========================================================================
    {
        const int head_idx  = blockIdx.x;
        const int layer_idx = blockIdx.y;
        // Each block processes one (head, layer) pair.  Src heads occupy the
        // first num_heads slots; dst heads occupy the next num_heads slots
        // (num_kv_heads is the number of distinct K/V heads, which may be
        // smaller than num_heads due to GQA).
        const int job = layer_idx * num_kv_heads + head_idx;
        constexpr size_t KVHEAD_SIZE = kv_head_byte_size<HD>();
        const uint8_t* src_head = heads_base + (size_t)job * KVHEAD_SIZE;
        const uint8_t* dst_head = heads_base + (size_t)(num_heads + job) * KVHEAD_SIZE;

        const uint8_t* src_pal_map = IS_K
            ? kvhead_k_pal_map<HD>(src_head)
            : kvhead_v_pal_map<HD>(src_head);
        const uint8_t* dst_pal_map = IS_K
            ? kvhead_k_pal_map<HD>(dst_head)
            : kvhead_v_pal_map<HD>(dst_head);

        // Threads 0-3 each populate one palette's metadata row in smem_meta.
        // The remaining 124 threads idle here; the write cost is negligible
        // vs. the xlat build below that uses all 128 threads.
        if (d < P4C_NUM_PAL) {
            const int p = d;
            if constexpr (IS_K) {
                smem_meta.src_fmt  [p] = kvhead_k_fmt  <HD>(src_head, p);
                smem_meta.dst_fmt  [p] = kvhead_k_fmt  <HD>(dst_head, p);
                smem_meta.src_arena[p] = kvhead_k_ptr  <HD>(src_head, p);
                smem_meta.dst_arena[p] = kvhead_k_ptr  <HD>(dst_head, p);
                smem_meta.src_outer[p] = kvhead_k_scale<HD>(src_head, p);
                smem_meta.dst_outer[p] = kvhead_k_scale<HD>(dst_head, p);
            } else {
                smem_meta.src_fmt  [p] = kvhead_v_fmt  <HD>(src_head, p);
                smem_meta.dst_fmt  [p] = kvhead_v_fmt  <HD>(dst_head, p);
                smem_meta.src_arena[p] = kvhead_v_ptr  <HD>(src_head, p);
                smem_meta.dst_arena[p] = kvhead_v_ptr  <HD>(dst_head, p);
                smem_meta.src_outer[p] = kvhead_v_scale<HD>(src_head, p);
                smem_meta.dst_outer[p] = kvhead_v_scale<HD>(dst_head, p);
            }
        }
        __syncthreads();  // smem_meta visible to all before xlat build reads it

        // Build smem_xlat[HD] — one entry per dst thread d.
        //
        // Thread d owns dst palette `pal`, local dim `pld`.
        //   global_d = the head dimension that is the pld-th member of dst
        //              palette `pal` (find_nth_dim_in_pal scans the dst pal_map)
        //   sp       = which src palette global_d belongs to (src pal_map lookup)
        //   s_ld     = local rank of global_d within src palette sp
        //   xlat[d]  = sp * PAL_DIM + s_ld = the smem column holding that dim
        //
        // After this, smem_xlat[d] is the smem_f16_buf column that holds the
        // src values for thread d's dst dimension — a fully resolved gather index.
        {
            int global_d = find_nth_dim_in_pal(dst_pal_map, pal, pld, HD);
            int sp       = pal_map_get(src_pal_map, global_d);
            int s_ld     = rank_in_pal(src_pal_map, sp, global_d);
            smem_xlat[d] = (uint8_t)(sp * PAL_DIM + s_ld);
        }
        __syncthreads();  // xlat visible to all warps before first copy_to_regs
    }
    // All prologue variables are now dead.  ~10 registers reclaimed.

    // r_dst_outer is the only persistent cold-metadata register.  It is read
    // on every token in the hot encode path, so keeping it in a register saves
    // a smem load per token.  All other metadata is reloaded from smem_meta at
    // each use site (issue_load and encode entry) to avoid holding ~6 more regs.
    const float r_dst_outer = smem_meta.dst_outer[pal];

    // =========================================================================
    // STAGE 1 ISSUER  (lambda, called from prologue and loop tail)
    //
    // Fills smem_f16_buf[0..31][warp_id*32 .. warp_id*32+31] with chunk c's
    // values for this warp's palette, converted to F16.
    //
    // Two source modes:
    //
    //   Float src (F16/F32/BF16, esz > 0):
    //     Source layout: arena_base[c * 32 * 32 + t * 32 + ld] × elem_size
    //     Thread d reads column `lane` across all 32 token rows.
    //     For aligned F16 src on Ampere+: 4 × cp.async.ca 16-byte transfers
    //     per thread (one token row × 32 F16 elements = 64 bytes), issued as
    //     asynchronous DMA → overlaps with the previous chunk's encode.
    //     For F32/BF16 or unaligned: scalar loads with __float2half conversion.
    //
    //   Quant/R16 src (esz == 0):
    //     Source layout: block(ld, c) = arena_base + (ld * num_chunks + c) × bb
    //     Thread d dequantizes block `lane` of chunk c, writing 32 F16 values
    //     (one per token) into smem_f16_buf[0..31][d].
    //     The dequant_element_inline call incorporates src_outer so smem holds
    //     actual float values (not normalized).
    // =========================================================================
    auto issue_load = [&](int c) {
        // Cold metadata: reload from smem_meta rather than keeping registers.
        const int   fmt       = smem_meta.src_fmt  [pal];
        const char* abase     = reinterpret_cast<const char*>(smem_meta.src_arena[pal]);
        const float src_outer = smem_meta.src_outer[pal];

        int esz = ArenaFormat::float_elem_size(fmt);
        if (esz > 0) {
            // Float src: channel-oriented, chunk c starts at byte offset
            //   c × CHUNK_SIZE × PAL_DIM × elem_size
            const char* chunk_base = abase
                + (int64_t)c * P4C_CHUNK_SIZE * PAL_DIM * esz;

            if (fmt == ArenaFormat::F16) {
                // row_src: the token-`lane` row of this warp's 32-column slice
                // of its palette (`warp_pd0` = the slice's first local dim; 0
                // always at HD 128, 0 or 32 at HD 256).
                // row_dst: smem row `lane` (token lane), columns d-lane..+31.
                // Each thread transfers its own 64-byte row (32 F16 values).
                // Issued as 4 × 16-byte cp.async on Ampere+ — async DMA that
                // returns immediately and completes before cp.async.wait_group.
                const int warp_pd0 = (warp_id * 32) % PAL_DIM;
                const char* row_src = chunk_base
                    + ((int64_t)lane * PAL_DIM + warp_pd0) * sizeof(__half);
                char* row_dst = reinterpret_cast<char*>(
                    &smem_f16_buf[lane][warp_id * 32]);
#if __CUDA_ARCH__ >= 800
                bool aligned = ((reinterpret_cast<uintptr_t>(row_src) & 0xF) == 0);
                if (aligned) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        uint32_t sa = static_cast<uint32_t>(
                            __cvta_generic_to_shared(row_dst + i * 16));
                        asm volatile(
                            "cp.async.ca.shared.global [%0], [%1], 16;"
                            :: "r"(sa), "l"(row_src + i * 16));
                    }
                } else {
                    const __half* p16 = reinterpret_cast<const __half*>(chunk_base);
                    for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                        smem_f16_buf[t][d] = p16[t * PAL_DIM + pld];
                }
#else
                const __half* p16 = reinterpret_cast<const __half*>(chunk_base);
                for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                    smem_f16_buf[t][d] = p16[t * PAL_DIM + pld];
#endif
            } else if (fmt == ArenaFormat::F32) {
                const float* p32 = reinterpret_cast<const float*>(chunk_base);
                for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                    smem_f16_buf[t][d] = __float2half(p32[t * PAL_DIM + pld]);
            } else if (fmt == ArenaFormat::BF16) {
                const __nv_bfloat16* pbf =
                    reinterpret_cast<const __nv_bfloat16*>(chunk_base);
                for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                    smem_f16_buf[t][d] = __float2half(
                        __bfloat162float(pbf[t * PAL_DIM + pld]));
            }
        } else {
            // Quant src: token-oriented block layout.
            // Thread d owns local dim `pld`; block(pld, c) is at:
            //   abase + (pld * num_chunks + c) * block_bytes
            // dequant_element_inline reads element t from that block and
            // applies src_outer to recover the actual float value.
            int bb = p4c_quant_block_bytes(fmt);
            const char* blk_base = abase
                + (int64_t)(pld * num_chunks + c) * bb;

            if (fmt == ArenaFormat::R16) {
                // R16 is raw half-precision — no dequant needed.
                const __half* p16 = reinterpret_cast<const __half*>(blk_base);
                for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                    smem_f16_buf[t][d] = p16[t];
            } else {
                for (int t = 0; t < P4C_CHUNK_SIZE; t++)
                    smem_f16_buf[t][d] = __float2half(
                        dequant_element_inline<float>(blk_base, t, fmt, src_outer));
            }
        }
    };

    // =========================================================================
    // REGISTER BUFFER + COPY LAMBDA
    //
    // r_buf[k] = half2(smem_f16_buf[2k][xlat[d]], smem_f16_buf[2k+1][xlat[d]])
    //
    // After copy_to_regs(), thread d holds all 32 tokens for its dst column,
    // packed as 16 half2 pairs and pre-translated via smem_xlat.  The gather
    // and format remapping are both folded into this one pass, so the encode
    // stage operates on r_buf without touching smem.
    //
    // copy_to_regs() MUST only be called immediately after a
    // cp.async.wait_group 0 + __syncthreads() pair, when smem_f16_buf is
    // fully settled and not being written by any in-flight DMA.
    // =========================================================================
    half2 r_buf[16];

    auto copy_to_regs = [&]() {
        const uint8_t src_col = smem_xlat[d];  // pre-resolved gather index
        #pragma unroll
        for (int k = 0; k < 16; k++)
            r_buf[k] = __halves2half2(smem_f16_buf[2*k    ][src_col],
                                      smem_f16_buf[2*k + 1][src_col]);
    };

    // =========================================================================
    // PIPELINE PROLOGUE
    //
    //   1. Load chunk 0 into smem synchronously (issue + commit + wait + sync).
    //   2. Copy smem → r_buf (chunk 0 is now in registers).
    //   3. Kick off DMA for chunk 1 (if it exists) — this runs concurrently
    //      with the encode of chunk 0 in the loop below.
    //
    // After the prologue:
    //   r_buf     = chunk 0 (safe to encode)
    //   smem      = chunk 0 (but about to be overwritten by DMA for chunk 1)
    //   cp.async  = chunk 1 group committed (if num_chunks > 1)
    // =========================================================================
    issue_load(0);
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::);
    asm volatile("cp.async.wait_group 0;" ::);
#endif
    __syncthreads();
    copy_to_regs();  // smem settled → r_buf

    if (1 < num_chunks) {
        issue_load(1);           // start DMA for chunk 1; smem now in flux
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;" ::);
#endif
    }

    // =========================================================================
    // MAIN CHUNK LOOP
    //
    // Each iteration encodes chunk c from r_buf (Stage 2), then advances the
    // pipeline: wait for chunk c+1 DMA, copy smem → r_buf, kick DMA for c+2.
    //
    // Loop invariant at top of iteration c:
    //   r_buf holds chunk c's data (gathered and translated, ready to encode).
    //   smem  is being overwritten by DMA for chunk c+1 (or idle if c=last).
    //   cp.async group for chunk c+1 has been committed (if it exists).
    // =========================================================================
    for (int c = 0; c < num_chunks; c++) {

        // =====================================================================
        // STAGE 2 — ENCODE
        //
        // Reads r_buf exclusively.  smem may have concurrent DMA in flight for
        // chunk c+1 but is never accessed here — this is the core DMA-safety
        // invariant of the register-buffer scheme.
        //
        // Cold dst metadata (fmt, arena ptr) reloaded from smem_meta here
        // rather than held in persistent registers.
        // =====================================================================
        {
            const int   dst_fmt  = smem_meta.dst_fmt  [pal];
            char* const dst_base = reinterpret_cast<char*>(smem_meta.dst_arena[pal]);

            int esz = ArenaFormat::float_elem_size(dst_fmt);
            if (esz > 0) {
                // ─── Float dst (F16 / F32 / BF16) ───────────────────────────
                //
                // r_buf[k] already holds the gathered, pre-translated F16 values.
                // Thread d writes its column `lane` across all 32 token rows of
                // chunk c.  Each token pair is packed into one half2, so the
                // unrolled loop writes 2 rows per iteration — 16 iterations
                // total cover all 32 tokens.
                //
                // Dst layout mirror of the src float layout:
                //   dst_base + (c * 32 * 32 + t * 32 + lane) * esz
                //
                // Outer scale: dst values = r_buf values × r_dst_outer.
                // r_buf values entered smem as actual floats (or F16 at scale 1)
                // and r_dst_outer re-applies the head scale for the dst format.
                char* chunk_base = dst_base
                    + (int64_t)c * P4C_CHUNK_SIZE * PAL_DIM * esz;

                if (dst_fmt == ArenaFormat::F16) {
                    __half* p16 = reinterpret_cast<__half*>(chunk_base);
                    const half2 scale2 = __float2half2_rn(r_dst_outer);
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        half2 sc = __hmul2(r_buf[k], scale2);
                        p16[(2*k    ) * PAL_DIM + pld] = __low2half(sc);
                        p16[(2*k + 1) * PAL_DIM + pld] = __high2half(sc);
                    }
                } else if (dst_fmt == ArenaFormat::F32) {
                    float* p32 = reinterpret_cast<float*>(chunk_base);
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        p32[(2*k    ) * PAL_DIM + pld] =
                            __half2float(__low2half (r_buf[k])) * r_dst_outer;
                        p32[(2*k + 1) * PAL_DIM + pld] =
                            __half2float(__high2half(r_buf[k])) * r_dst_outer;
                    }
                } else if (dst_fmt == ArenaFormat::BF16) {
                    __nv_bfloat16* pbf = reinterpret_cast<__nv_bfloat16*>(chunk_base);
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        pbf[(2*k    ) * PAL_DIM + pld] =
                            __float2bfloat16(__half2float(__low2half (r_buf[k])) * r_dst_outer);
                        pbf[(2*k + 1) * PAL_DIM + pld] =
                            __float2bfloat16(__half2float(__high2half(r_buf[k])) * r_dst_outer);
                    }
                }
            } else {
                // ─── Quant/R16 dst ───────────────────────────────────────────
                //
                // Quant encode (p4c_encode_quant_block) is warp-cooperative:
                // all 32 threads in the warp must provide scratch[lane] = the
                // fp32 value for token `lane` before the call, and they all
                // collaborate to compute the block scale and pack the bits.
                //
                // The data layout mismatch:
                //   r_buf layout:   thread t owns all 32 tokens for column t
                //   encode needs:   scratch[lane] = token `lane` for column ld
                //
                // Solution — warp transpose via 16 __shfl_sync calls per column:
                //
                //   Thread ld (srcLane=ld in the shuffle) holds:
                //     r_buf[k] = half2(tok_{2k}, tok_{2k+1}) for col ld, k=0..15
                //
                //   16 shuffles broadcast r_buf[0], r_buf[1], ..., r_buf[15]
                //   from thread ld.  Thread `lane` predicate-selects k = lane>>1
                //   (the half2 pair containing its token) and extracts:
                //     low  half if lane is even  → tok_{lane & ~1} = tok_lane ✓
                //     high half if lane is odd   → tok_{lane |  1} = tok_lane ✓
                //
                //   After 16 shuffles: scratch[lane] = tok_lane for col ld.
                //   __syncwarp() before encode ensures all scratch writes are
                //   visible across the warp before any thread reads them in the
                //   encode function.  A second __syncwarp() after encode guards
                //   re-use of scratch on the next iteration.
                //
                // Total per chunk: 32 cols × 16 shuffles = 512 shuffle insns.
                // This is acceptable — the encode itself is far more expensive
                // (reduction ops for block scale, bit packing), and DMA overlap
                // is preserved since encode never touches smem.
                const int    bb      = p4c_quant_block_bytes(dst_fmt);
                float* const scratch = smem_qscratch[warp_id];
                // The warp's 32-column slice sits at local dims
                // [warp_pd0, warp_pd0+32) of its palette — 0 always at HD 128,
                // 0 or 32 at HD 256. Every column encodes independently (a
                // quant block is one dim × 32 tokens), so each warp encoding
                // only its own slice covers the palette exactly.
                const int warp_pd0 = (warp_id * 32) % PAL_DIM;

                for (int wl = 0; wl < 32; wl++) {
                    // Gather tok_lane for the warp's wl-th column from the
                    // in-warp thread that owns it.
                    unsigned h2u = 0;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        unsigned tmp = __shfl_sync(0xffffffff,
                            *reinterpret_cast<const unsigned*>(&r_buf[k]), wl);
                        if ((lane >> 1) == k) h2u = tmp;
                    }
                    const half2 pair = *reinterpret_cast<const half2*>(&h2u);
                    scratch[lane] = __half2float((lane & 1) ? __high2half(pair) : __low2half(pair))
                                    * r_dst_outer;
                    __syncwarp();  // scratch[0..31] all written before encode reads

                    // Dst quant layout: block(ld, c) = dst_base + (ld * num_chunks + c) * bb
                    const int ld = warp_pd0 + wl;
                    char* blk_addr = dst_base + (int64_t)(ld * num_chunks + c) * bb;
                    p4c_encode_quant_block(scratch, blk_addr, dst_fmt);
                    __syncwarp();  // encode complete before scratch is reused for wl+1
                }
            }
        }

        // =====================================================================
        // PIPELINE ADVANCE (loop tail)
        //
        // Wait for chunk c+1's DMA to finish, snapshot smem into r_buf, and
        // kick off DMA for chunk c+2.  Skipped entirely on the last iteration
        // (no further chunks to load or decode).
        //
        // All four warps execute this block identically — __syncthreads() is
        // a block barrier and must be reached by every warp on every iteration.
        // =====================================================================
        if (c + 1 < num_chunks) {
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.wait_group 0;" ::);
#endif
            __syncthreads();  // smem has chunk c+1 fully settled; DMA for c+1 done
            copy_to_regs();   // r_buf ← chunk c+1 (gathered + translated)

            // Kick off DMA for chunk c+2 now, so it overlaps with encode of c+1.
            if (c + 2 < num_chunks) {
                issue_load(c + 2);
#if __CUDA_ARCH__ >= 800
                asm volatile("cp.async.commit_group;" ::);
#endif
            }
        }
    }
}

// =============================================================================
// C DISPATCHER
// =============================================================================

extern "C" void run_quantize_palette4_convert(
    const uint8_t* heads_base,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t num_layers,
    int32_t num_chunks,
    int32_t is_k,
    int32_t head_dim,
    cudaStream_t stream
) {
    if (num_kv_heads == 0 || num_layers == 0 || num_chunks == 0) return;

    // cudaSharedmemCarveoutMaxShared partitions the SM's unified memory bank
    // to maximise shared memory (~100 KB on Ada Lovelace) at the cost of L1.
    // With 8960 B/block this enables floor(102400/8960) = 11 blocks/SM smem-
    // ceiling, vs. 5 blocks/SM with the default 48 KB partition.
    //
    // L1 loss is acceptable: F16 src loads use cp.async (L1-bypassing), store
    // paths go direct to L2, and quant-src scalar loads are too streaming for
    // L1 to help.  The attribute is idempotent at the driver level so calling
    // it per-invocation is safe.
    // One block per (head, layer) pair.  block.x = head_dim threads.
    dim3 grid(num_kv_heads, num_layers);
    dim3 block(head_dim);

    #define P4C_LAUNCH(HD)                                                       \
        do {                                                                     \
            if (is_k) {                                                          \
                cudaFuncSetAttribute(                                            \
                    palette4_convert_kernel<HD, true>,                           \
                    cudaFuncAttributePreferredSharedMemoryCarveout,              \
                    cudaSharedmemCarveoutMaxShared);                             \
                palette4_convert_kernel<HD, true><<<grid, block, 0, stream>>>(   \
                    heads_base, num_heads, num_kv_heads, num_chunks);            \
            } else {                                                             \
                cudaFuncSetAttribute(                                            \
                    palette4_convert_kernel<HD, false>,                          \
                    cudaFuncAttributePreferredSharedMemoryCarveout,              \
                    cudaSharedmemCarveoutMaxShared);                             \
                palette4_convert_kernel<HD, false><<<grid, block, 0, stream>>>(  \
                    heads_base, num_heads, num_kv_heads, num_chunks);            \
            }                                                                    \
        } while (0)

    switch (head_dim) {
        case 128: P4C_LAUNCH(128); break;
        case 256: P4C_LAUNCH(256); break;
        default: break; // the Rust dispatch bails on unsupported widths first
    }
    #undef P4C_LAUNCH
}
