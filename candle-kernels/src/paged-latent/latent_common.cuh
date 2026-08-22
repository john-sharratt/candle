#pragma once
// =============================================================================
// latent_common.cuh — shared pieces of the paged latent-attention kernels
// (single-latent K≡V geometry, MQA n_kv_head=1): constants, bit-mirrorable
// trig + the factored RoPE table, glue scatter, partial pool.
//
// Fork of the paged-decode INT8 kernel restructured for HEAD_DIM=512:
//   • K and V are the SAME 512-d latent: one arena read serves both the int8
//     QK^T and the FP PV accumulate (halves KV traffic; one smem region).
//   • M-tile = 16 query heads staged int8 in smem (all heads share the one
//     latent), N-tile = 8 keys, K = 512 = NPAL(16) bands × 1 m16n8k32 k-step
//     (SUB = 512/16 = 32 = one MMA tile per band).
//   • Hybrid two-source key stream feeding ONE online softmax: the sliding
//     window (arena slot walk, position-clamped to the last `window_size`
//     tokens) followed by the selected compressed entries (index-driven walk
//     over a device GID list).
//   • RoPE reads the FACTORED cos/sin table (rope_lookup below: θ split at
//     bit 10 + the angle-addition identity, ~768 KB per frequency set,
//     L2-resident, built once at load with the exact bit-mirrorable trig —
//     covers every position under the 1M context cap with margin). Only the
//     trailing ROPE_DIM dims rotate (nope‖rope split), interleaved pairs.
//   • The window arena is uniform FP8 E4M3 in ascending-dim band order
//     (16 bands of 32), so keys load with direct pointer math — no palette
//     map, no format dispatch. Band pointers/outer-scales still come from the
//     KvHead record (k_* fields; v_* ignored — K≡V).
//   • The kernel ALWAYS emits split-KV partials (un-normalized ΣwV, m, l in
//     the natural-e domain). The companion combine kernel merges splits,
//     folds the per-head learned sink, normalizes, DE-ROTATES the output's
//     rope dims at the query position (inverse rotation — linear, so it
//     commutes with the merge), and writes the final output.
//
// All exponentials use fast_exp's cubic-polynomial e^x (plain f32 arithmetic,
// reproducible bit-for-bit by the CPU mirror oracle).
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

#include "../fast_exp.cuh"
#include "../blocks.cuh"
#include "../mma/mma_wrappers.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../paged-decode/decode_helpers.cuh"
#include "../arena_table.cuh"
#include "../convert/convert_all.cuh"

namespace latent_attn {

constexpr int HEADS_TILE = 16;  // MMA M dimension: query heads per block
constexpr int KEYS_TILE = 8;    // MMA N dimension: keys per tile
constexpr int WARPS = 8;        // block = 256 threads
constexpr int NPAL = 16;        // 32-dim bands (SUB = HEAD_DIM / NPAL); the last
                                // two bands ([448,480),[480,512)) are the
                                // RoPE-only bands. The shared KvHead accessors
                                // take <HEAD_DIM, NPAL> so the 16-band record
                                // offsets match the host. SUB=32 = one m16n8k32
                                // tile per band; the pal_map is 4-bit (names 16).

__device__ __forceinline__ float ds_exp(float x) {
    // fast_exp cubic e^x (Softmax mode: lower clamp only). Bit-reproducible.
    return fast_exp::exp<float, fast_exp::Softmax, fast_exp::High>(x);
}

__device__ __forceinline__ float fp8_to_f32(uint8_t b) {
    __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}

// Per-band window read: one element of a chunk's band arena,
// dispatched on the KvHead's per-band format tag (`kvhead_k_fmt`). Float
// bands — F8E4M3 is the writer chunk's format — are token-major:
// `within*SUB + d`. Quant bands (sealed chunks compressed by the policy) are
// token-oriented GGML blocks: with CHUNK_SIZE = 32, block `d` holds the 32
// tokens of dim `d`, so the block address is `d*BLOCK_BYTES` and the element
// index is `within & 31`. `outer` is the per-band decoder-divide scale
// (KvHead k_scale); the block converters apply it, the float paths divide.
template <int SUB>
__device__ __forceinline__ float load_band_elem(
    uint64_t band_ptr, int fmt, float outer, int within, int d_in_band
) {
    const uint8_t* base = (const uint8_t*)(uintptr_t)band_ptr;
    if (fmt == ArenaFormat::F8E4M3)
        return fp8_to_f32(base[(int64_t)within * SUB + d_in_band]) / outer;
    const int esz = ArenaFormat::float_elem_size(fmt);
    if (esz > 0) {
        // Other float dtypes (F16/BF16/F32), token-major like FP8.
        const uint8_t* p = base + ((int64_t)within * SUB + d_in_band) * esz;
        float v = fmt == ArenaFormat::F32 ? *reinterpret_cast<const float*>(p)
            : fmt == ArenaFormat::F16 ? __half2float(*reinterpret_cast<const __half*>(p))
                                      : __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
        return v / outer;
    }
    const int bytes = ArenaAccessor::get_quant_block_bytes(fmt);
    const void* blk = base + (int64_t)d_in_band * bytes;
    return dequant_element_inline<float, true>(blk, within & 31, fmt, outer);
}

// Store one pre-RoPE latent element into a chunk's band arena, dispatched on
// the band's float format tag — the exact write mirror of load_band_elem. Only
// float formats occur on a writer chunk (the k_format ceiling); quantized bands
// are produced by the seal path, never by these fused scatters. `v` is the
// element in f32; the store pre-scales by the per-band `outer` (the record's
// k_scale) and rounds to the band dtype (bf16/f16 round-to-nearest, fp8 via the
// E4M3 converter — the writer default). Because load returns
// `decode(stored) / outer`, the symmetric encode stores `dtype(v * outer)`, so
// any per-band decoder-divide scale round-trips exactly (`outer == 1` for the
// scale-free BF16/F16/F32 bands, leaving those byte-identical). Token-major
// layout, matching the float read in load_band_elem.
template <int SUB>
__device__ __forceinline__ void store_band_elem(
    uint64_t band_ptr, int fmt, float outer, int within, int d_in_band, float v
) {
    const int64_t idx = (int64_t)within * SUB + d_in_band;
    uint8_t* base = (uint8_t*)(uintptr_t)band_ptr;
    const float s = __fmul_rn(v, outer);
    if (fmt == ArenaFormat::BF16)
        ((__nv_bfloat16*)base)[idx] = __float2bfloat16(s);
    else if (fmt == ArenaFormat::F16)
        ((__half*)base)[idx] = __float2half(s);
    else if (fmt == ArenaFormat::F32)
        ((float*)base)[idx] = s;
    else
        ((__nv_fp8_e4m3*)base)[idx] = __nv_fp8_e4m3(s);
}

// ─── Bit-mirrorable RoPE trig primitives ─────────────────────────────────────
// The angle pos·freq is reduced in DOUBLE precision (an f32 product is
// unusable at depth — ulp(10⁶ rad) ≈ 0.06 rad) down to a quadrant residual
// r ∈ [-π/4, π/4] plus quadrant k. sin/cos then come from short minimax
// polynomials in PLAIN f32 arithmetic (the archive compiles `-fmad=false`), so
// every operation — reduction included — is exact-rounded and reproduced
// bit-for-bit by the CPU mirror oracle.
// Every operation below uses the explicit round-to-nearest intrinsics
// (`__fmul_rn`/`__fadd_rn`/`__dmul_rn`/…): the compiler is NOT permitted to
// contract an explicit intrinsic into an fma, so the arithmetic here is
// exact-rounded IEEE regardless of `-fmad` — the property the CPU mirror's
// plain Rust ops reproduce bit-for-bit.
//
// The attention kernels do NOT run these per key: FP64 runs at 1/64 rate on
// consumer Blackwell and every key of every tile needs ROPE_DIM/2 angles. They
// read a FACTORED cos/sin table instead (rope_lookup below), built ONCE per
// frequency set by latent_rope_table_kernel using these exact primitives —
// same bits, none of the cost. These functions stay for the table builder and
// the mirror-parity probes.
__device__ __forceinline__ void rope_angle(int pos, float freq, float& r, int& k) {
    double a = __dmul_rn((double)pos, (double)freq);
    double t = floor(__dmul_rn(a, 0.15915494309189535)); // ·1/2π
    a = __dsub_rn(a, __dmul_rn(t, 6.283185307179586));   // [0, 2π)
    double q = floor(__dadd_rn(__dmul_rn(a, 0.6366197723675814), 0.5)); // ·2/π
    r = (float)__dsub_rn(a, __dmul_rn(q, 1.5707963267948966));
    k = ((int)q) & 3;
}

__device__ __forceinline__ void ds_sincos(float r, int k, float& s, float& c) {
    float x2 = __fmul_rn(r, r);
    // sin on [-π/4, π/4] (cephes sinf coefficients).
    float sp_in = -1.9515295891e-4f;
    sp_in = __fadd_rn(__fmul_rn(sp_in, x2), 8.3321608736e-3f);
    sp_in = __fadd_rn(__fmul_rn(sp_in, x2), -1.6666654611e-1f);
    float rt = __fmul_rn(r, x2);
    float sp = __fadd_rn(r, __fmul_rn(rt, sp_in));
    // cos on [-π/4, π/4] (cephes cosf coefficients).
    float cp_in = 2.443315711809948e-5f;
    cp_in = __fadd_rn(__fmul_rn(cp_in, x2), -1.388731625493765e-3f);
    cp_in = __fadd_rn(__fmul_rn(cp_in, x2), 4.166664568298827e-2f);
    float x4 = __fmul_rn(x2, x2);
    float cp = __fsub_rn(1.0f, __fmul_rn(0.5f, x2));
    cp = __fadd_rn(cp, __fmul_rn(x4, cp_in));
    switch (k) {
        case 0: s = sp;  c = cp;  break;
        case 1: s = cp;  c = -sp; break;
        case 2: s = -sp; c = -cp; break;
        default: s = -cp; c = sp; break;
    }
}

// ─── Factored position→(sin, cos) table ──────────────────────────────────────
// θ = pos·f with pos = hi·2¹⁰ + lo splits the trig by the angle-addition
// identity:  sin θ = s_hi·c_lo + c_hi·s_lo,  cos θ = c_hi·c_lo − s_hi·s_lo.
// Two tiny tables — (sin, cos) of (hi·2¹⁰)·f and of lo·f, each entry produced
// by the exact rope_angle/ds_sincos pair above — cover every position below
// ROPE_HI_DIM·2¹⁰ = 2M in (2048+1024)·NF·8 B ≈ 768 KB per frequency set:
// permanently L2-resident, so a per-key angle costs 2 float2 loads + 6
// exact-rounded f32 ops instead of an FP64 reduction + two polynomials. The
// attention kernel is a bounded WINDOW into the substrate (the engine's context
// hard-cap is 1M), so every position it sees — query and key alike — is < 2M and
// lands in the table; the growing corpus is retrieved into that window, not
// attended at raw substrate positions. The combination is the same plain
// round-to-nearest arithmetic on both device and mirror, so bit-exactness is
// preserved end to end.
//
// Layout (float2 = (sin, cos)): [ROPE_HI_DIM][NF] hi block, then
// [ROPE_LO_DIM][NF] lo block. NF = ROPE_DIM/2 frequencies.
constexpr int ROPE_LO_BITS = 10;
constexpr int ROPE_LO_DIM = 1 << ROPE_LO_BITS;  // 1024
constexpr int ROPE_HI_DIM = 2048;               // positions < 2^21 (2M, ≥ 1M cap)

template <int NF>
__device__ __forceinline__ void rope_lookup(
    const float* __restrict__ tab, int pos, int j, float& s, float& c
) {
    // min-clamp: a corrupt position past the 2M table span must not read OOB
    // (the score is garbage either way; the clamp keeps it a *deterministic*
    // garbage the mirror reproduces).
    int hi = min(pos >> ROPE_LO_BITS, ROPE_HI_DIM - 1);
    int lo = pos & (ROPE_LO_DIM - 1);
    const float2* th = (const float2*)tab;
    const float2* tl = th + ROPE_HI_DIM * NF;
    float2 h = th[hi * NF + j];
    float2 l = tl[lo * NF + j];
    s = __fadd_rn(__fmul_rn(h.x, l.y), __fmul_rn(h.y, l.x));
    c = __fsub_rn(__fmul_rn(h.y, l.y), __fmul_rn(h.x, l.x));
}

// Interleaved-pair RoPE on a [pair0, pair1] register pair at position `pos`,
// frequency index `j`. Forward: (x0 c − x1 s, x0 s + x1 c) — explicit-rounded
// so the rotation cannot be contracted either.
template <int NF>
__device__ __forceinline__ void rope_pair(
    float& x0, float& x1, const float* __restrict__ tab, int pos, int j
) {
    float c, s;
    rope_lookup<NF>(tab, pos, j, s, c);
    float r0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    float r1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
    x0 = r0;
    x1 = r1;
}

// Table builder: one thread per (row, freq) entry. Row < ROPE_HI_DIM is the hi
// block at position row·2¹⁰; the remainder is the lo block at position
// row − ROPE_HI_DIM. Runs once per frequency set at model load.
__global__ void latent_rope_table_kernel(
    const float* __restrict__ freqs, float* __restrict__ tab, int n_freqs
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = (ROPE_HI_DIM + ROPE_LO_DIM) * n_freqs;
    if (i >= total) return;
    int j = i % n_freqs;
    int row = i / n_freqs;
    int pos = row < ROPE_HI_DIM ? (row << ROPE_LO_BITS) : (row - ROPE_HI_DIM);
    float r, s, c;
    int k;
    rope_angle(pos, freqs[j], r, k);
    ds_sincos(r, k, s, c);
    tab[2 * i] = s;
    tab[2 * i + 1] = c;
}

// =============================================================================
// Glue latent scatter: write latents into their RESERVED gap chunks (block index
// + in-block offset per row, from the reprojection's PendingGlue descriptors).
// Launched stream-ordered BEFORE the attention pass, so glue keys read from the
// arena like any window key — no double-source, no intra-launch race. One warp
// per row.
// =============================================================================
// BATCHED over runs (hot-path invariant 2b). Three separate wave phases scatter
// this way — the glue islands, the prefill prompt writeback, and the speculative
// verify writeback — and each ran one launch per SEQUENCE, plus two pageable
// uploads per sequence for its slice/offset arrays. Every run names its own
// destination slot by address, so there is nothing to concatenate: the whole
// fleet goes in one launch off a descriptor table.
//
// Table layout, array-of-structs, GLUE_SCATTER_WORDS i64 per run:
//     [0] kv base pointer, [rows, HEAD_DIM] pre-RoPE latents
//     [1] headers pointer — the run's SlotHeader[1]
//     [2] slices pointer, [rows] u32 gap block index
//     [3] in-block offset pointer, [rows] u32
//     [4] rows
//
// grid.y indexes the run, grid.x covers the widest run's warps; shorter runs
// exit on the row bound.
#define GLUE_SCATTER_WORDS 5

template <typename T, int HEAD_DIM>
__global__ void latent_glue_scatter_kernel(
    const long long* __restrict__ desc,
    int n_runs
) {
    const int e = blockIdx.y;
    if (e >= n_runs) return;
    const long long* __restrict__ dsc = desc + (long long)e * GLUE_SCATTER_WORDS;
    const T* __restrict__ kv = (const T*)dsc[0];
    const uint8_t* __restrict__ headers = (const uint8_t*)dsc[1];
    const uint32_t* __restrict__ slices = (const uint32_t*)dsc[2];
    const uint32_t* __restrict__ in_blk = (const uint32_t*)dsc[3];
    const int rows = (int)dsc[4];

    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int DPT = HEAD_DIM / 32;
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (int)threadIdx.x % 32;
    if (row >= rows) return;
    const SlotHeader& slot = get_slot_header(headers, 0);
    const uint8_t* slp =
        get_slice<HEAD_DIM>(slot.slices_ptr, (int)slices[row], 1);
    const uint8_t* head_ptr = get_head<HEAD_DIM>(slp, 0);
    const T* src = kv + (int64_t)row * HEAD_DIM;
    int within = (int)in_blk[row];
    #pragma unroll
    for (int j = 0; j < DPT; ++j) {
        int d = lane * DPT + j;
        int band = d / SUB;
        uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM, NPAL>(head_ptr, band);
        if (band_ptr != 0) {
            const int fmt = kvhead_k_fmt<HEAD_DIM, NPAL>(head_ptr, band);
            const float outer = kvhead_k_scale<HEAD_DIM, NPAL>(head_ptr, band);
            store_band_elem<SUB>(band_ptr, fmt, outer, within, d % SUB,
                                 to_f32<T>(src[d]));
        }
    }
}

// There is NO device-side partial pool: the split-KV partial workspace is
// OWNED BY THE CALLER (the Rust side's fixed `LatentWorkspace`, built once at
// model load; prefill chunks its queries to the fixed capacity) and passed
// in. The launcher has no allocator, no static, no lock — callers that must
// not share a buffer (concurrent host threads on one stream) simply hold
// their own workspace.

// Two-region position-free corpus cache builder (review items B + D). Quantizes
// entries [g_lo, g_hi) into the HOT retrieval artifact (the f32 gallery is
// archival):
//   • NOPE bands [0, NOPE_DIM)  → int8 `nope_i8`, per-band component-amax
//     `nope_scale` (never rotate, so a component bound is exact; 7-bit).
//   • ROPE bands [NOPE_DIM, HEAD_DIM) → BF16 `rope_bf` PRE-rotation (float, 8-bit
//     mantissa, matches the window ring's BF16 rope tail — no √2 pair-magnitude
//     margin, no clip). Readers rotate these at read time from `comp_pos`.
// Same NPAL*32 launch contract + grid-stride as `latent_quant_corpus_range_kernel`.
template <int HEAD_DIM, int ROPE_DIM>
__global__ void latent_build_corpus_cache_kernel(
    const float* __restrict__ comp,        // [G, HEAD_DIM] f32 pre-RoPE (canonical)
    int8_t* __restrict__ nope_i8,          // [G, NOPE_DIM] out (nope int8)
    float* __restrict__ nope_scale,        // [G, NOPE_BANDS] out (per-nope-band amax)
    __nv_bfloat16* __restrict__ rope_bf,   // [G, ROPE_DIM] out (rope pre-rotation bf16)
    int g_lo,
    int g_hi
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    constexpr int NOPE_BANDS = NOPE_DIM / SUB;
    constexpr int DPL = SUB / 32;
    static_assert(NOPE_DIM % SUB == 0, "rope/nope split must fall on a band boundary");
    const int band = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x % 32;
    const bool rope_band = band >= NOPE_BANDS;
    for (int gid = g_lo + (int)blockIdx.x; gid < g_hi; gid += (int)gridDim.x) {
        const float* src = comp + (int64_t)gid * HEAD_DIM;
        float v[DPL];
        #pragma unroll
        for (int j = 0; j < DPL; ++j) v[j] = src[band * SUB + lane * DPL + j];
        if (rope_band) {
            // ROPE band → BF16 pre-rotation (no scale, no quant); rotated at read.
            __nv_bfloat16* dst = rope_bf
                + (int64_t)gid * ROPE_DIM + (band - NOPE_BANDS) * SUB + lane * DPL;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) dst[j] = __float2bfloat16(v[j]);
        } else {
            // NOPE band → int8, per-band component amax (warp-uniform reduction).
            float mx = 0.f;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) mx = fmaxf(mx, fabsf(v[j]));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            // Explicit IEEE division/reciprocal (NOT `/ 127.f`) so the mirror
            // (`build_corpus_cache`) reproduces the op bit-for-bit under fast-math.
            float scale = __fdiv_rn(mx, 127.f);
            if (scale == 0.f) scale = 1.f;
            if (lane == 0) nope_scale[(int64_t)gid * NOPE_BANDS + band] = scale;
            float inv = __frcp_rn(scale);
            int8_t* dst = nope_i8 + (int64_t)gid * NOPE_DIM + band * SUB + lane * DPL;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) {
                float qv = fminf(fmaxf(v[j] * inv, -127.f), 127.f);
                dst[j] = (int8_t)__float2int_rn(qv);
            }
        }
    }
}

// SM count for the caller's split-factor policy (thread-safe magic-static
// init; the value is a device constant).
inline int latent_sm_count() {
    static const int sm = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    return sm;
}
}  // namespace latent_attn
