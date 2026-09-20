#pragma once
// ============================================================================
// INT8 tile element helpers — shared by the INT8 prefill kernel and the INT8
// tile decode kernel.
//
// Both kernels stage one 32-token slice per tile: each palette's raw quant
// block span is bulk-copied to shared memory, then every element is decoded
// from that copy (or from global for a non-hop palette) in natural dim order.
// This header holds the pieces that stage and decode a single element:
//
//   - the frequency-indexed RoPE table lookup and the in-register RoPE window
//   - the cp.async fences the raw-span fill relies on
//   - int8 requantisation against a precomputed window scale
//   - runtime-format single-element FP decode, int8 read-through, and the
//     unified "one arena element as FP32" accessor over quant and dtype palettes
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "convert_all.cuh"
#include "../rope/rope_table.cuh"

namespace int8_elem {

/// Apply RoPE in place over a register window `x[N_WIN]` where lane `l` holds
/// dims {l + 32w : w in 0..N_WIN} of one head row.
///
/// `rope` is the sequence's view of its rung (`rope_table.cuh`), queried by
/// FREQUENCY, as the decode kernels query it: the half-split pairing
/// (d, d + HD/2) reads frequency d, the interleaved pairing (2i, 2i + 1)
/// reads frequency d >> 1. A Q rotation passes `rope.for_q()`.
///
/// Half-split (`rope_interleaved == 0`, Qwen/GPT-NeoX): pair (d, d + HD/2)
/// lives IN-THREAD as windows (w, w + N_WIN/2) — pure register math.
/// Interleaved (`== 1`, LLaMA/GPT-J): pair (2i, 2i + 1) spans lanes
/// (even, odd) of the SAME window (32 | 32w keeps dim parity = lane parity),
/// so one `lane ^ 1` shuffle per window fetches the partner — the same
/// exchange the decode kernels' `apply_rope_interleaved_f32` uses.
///
/// Callers must be warp-uniform (every lane executes the shuffle): every
/// call site guards on warp-uniform row/token conditions.
template <int HEAD_DIM, int N_WIN>
__device__ __forceinline__ void i8_apply_rope(
    float (&x)[N_WIN], int pos, int lane, int rope_interleaved,
    const RopeView& rope)
{
    if (rope_interleaved) {
        const float sign = (lane & 1) ? 1.f : -1.f;
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) {
            int d = lane + 32 * w;
            float c, s;
            rope_cs_at(rope, pos, d >> 1, c, s);
            float partner = __shfl_sync(0xffffffffu, x[w], lane ^ 1);
            x[w] = x[w] * c + sign * partner * s;
        }
    } else {
        #pragma unroll
        for (int w = 0; w < N_WIN / 2; ++w) {
            float c, s;
            rope_cs_at(rope, pos, lane + 32 * w, c, s);
            float lo = x[w], hi = x[w + N_WIN / 2];
            x[w] = lo * c - hi * s;
            x[w + N_WIN / 2] = lo * s + hi * c;
        }
    }
}

template <typename QT>
__device__ __forceinline__ float qt_to_f32(QT v);
template <>
__device__ __forceinline__ float qt_to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float qt_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename QT>
__device__ __forceinline__ QT qt_from_f32(float v);
template <>
__device__ __forceinline__ __half qt_from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 qt_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

/// cp.async fences for the raw-block staging fill. Groups are per-thread:
/// every thread commits and drains its own copies before the block-wide
/// staging barrier makes them visible (a bare __syncthreads does NOT
/// fence cp.async).
__device__ __forceinline__ void i8_cp_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

__device__ __forceinline__ void i8_cp_wait0() {
    asm volatile("cp.async.wait_group 0;" ::);
}

/// 16-byte global→shared bulk copy (both pointers 16-byte aligned).
__device__ __forceinline__ void i8_cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                    "l"(src));
}

/// The int8 code of `v · inv_scale` in the low byte of one FFMA. 1.5·2²³ has
/// a unit ulp, so `fma(v, inv, MAGIC)` rounds the exact product to the
/// nearest integer (ties to even, the IEEE default) in the one rounding the
/// FFMA performs, and the sum's low mantissa byte is that integer in two's
/// complement for |v · inv| < 2²². One instruction against the FMUL, F2I and
/// pack of a conversion. Every caller quantises against an absmax scale
/// (`inv = 127 / absmax`) or a probability (`p · 127`), so |v · inv| ≤ 127
/// and no saturation is needed. Every int8 code the tile kernel writes — K,
/// V (both paths), P and Q — comes through here, so the paths agree bit for
/// bit.
constexpr float I8_CODE_MAGIC = 12582912.f;   // 1.5 · 2^23
__device__ __forceinline__ uint32_t i8_code_bits(float v, float inv_scale) {
    return __float_as_uint(fmaf(v, inv_scale, I8_CODE_MAGIC));
}

/// Quantize a value against a precomputed window scale (0 ⇒ all-zero window).
__device__ __forceinline__ int8_t i8_quant(float v, float inv_scale) {
    return (int8_t)(i8_code_bits(v, inv_scale) & 0xffu);
}

/// Runtime-format single-element FP decode from a token-oriented quant
/// block (`blk` points at ONE dim's block; `e` is the token within it).
/// Same numerics as load_head_quant_token_oriented: value / scale.
__device__ __forceinline__ float i8_dequant_elem(
    int fmt, const char* blk, int e, float scale)
{
    switch (fmt) {
#define I8_DQ(F, B) \
    case ArenaFormat::F: \
        return BlockConverter<B, float>::load_element((const B*)blk, e, scale)
        I8_DQ(R16, block_r16);
        I8_DQ(Q4_0, block_q4_0);
        I8_DQ(Q4_1, block_q4_1);
        I8_DQ(Q5_0, block_q5_0);
        I8_DQ(Q5_1, block_q5_1);
        I8_DQ(Q8_0, block_q8_0);
        I8_DQ(Q8_1, block_q8_1);
        I8_DQ(Q4_KS, block_q4_ks);
        I8_DQ(Q8_KS, block_q8_ks);
        I8_DQ(Q3_0, block_q3_0);
        I8_DQ(Q3_1, block_q3_1);
        I8_DQ(Q2_0, block_q2_0);
        I8_DQ(Q2_1, block_q2_1);
        I8_DQ(Q2_A, block_q2_a);
        I8_DQ(Q2_S, block_q2_s);
        I8_DQ(Q1_S, block_q1_s);
        I8_DQ(Q0, block_q0);
        I8_DQ(Q0_V, block_q0_v);
        I8_DQ(Q1_A, block_q1_a);
        I8_DQ(Q0_X, block_q0_x);
        I8_DQ(Q0_M2, block_q0_m2);
        I8_DQ(Q0_M4, block_q0_m4);
#undef I8_DQ
        // A non-arena format reached block extraction — fail loud (the
        // same __trap idiom as the accessor's block addressing).
        default: __trap(); return 0.f;
    }
}

/// Runtime-format single-element int8 read-through (V). Same families and
/// numerics as load_head_int8_readthrough's dispatcher.
__device__ __forceinline__ Int8Sample i8_rt_elem(int fmt, const char* blk, int e)
{
    switch (fmt) {
#define I8_RT(F, B) \
    case ArenaFormat::F: return BlockInt8<B>::load((const B*)blk, e)
        I8_RT(Q8_0, block_q8_0);
        I8_RT(Q4_0, block_q4_0);
        I8_RT(Q5_0, block_q5_0);
        I8_RT(Q2_0, block_q2_0);
        I8_RT(Q3_0, block_q3_0);
        I8_RT(Q4_KS, block_q4_ks);
        I8_RT(Q8_KS, block_q8_ks);
        I8_RT(Q8_1, block_q8_1);
        I8_RT(Q2_S, block_q2_s);
        I8_RT(Q1_S, block_q1_s);
        I8_RT(Q1_A, block_q1_a);
        I8_RT(Q0, block_q0);
        I8_RT(Q0_M2, block_q0_m2);
        I8_RT(Q0_M4, block_q0_m4);
        I8_RT(Q0_X, block_q0_x);
#undef I8_RT
        default: __trap(); return Int8Sample{0, 0.f};
    }
}

/// True for the int8 read-through formats whose block scale is the SAME for
/// every token of the block. The tile kernels carry one scale per (dim,
/// block) into the PV MMA, so a format whose scale depends on the token
/// index (the sink-protected Q4_KS/Q8_KS: tokens 0–3 carry their own fine
/// scale) cannot pass through as raw int8 and takes the FP requant path.
__device__ __forceinline__ bool i8_tile_readthrough_format(int fmt) {
    return ArenaAccessor::is_int8_readthrough_format(fmt)
        && fmt != ArenaFormat::Q4_KS && fmt != ArenaFormat::Q8_KS;
}

// ----------------------------------------------------------------------------
// Hoisted-dispatch decoders. A thread that decodes a whole token run of ONE
// palette resolves the palette's format once — `i8_with_format` switches on
// the format and hands the body a compile-time tag — and the body's per-
// element loads are then straight-line template code: no jump table per
// element, the block header and the scale reciprocal hoisted out of the
// token loop, and the address space of a shared-memory span visible to the
// compiler (the caller passes the span pointer, not a stored generic one).
// ----------------------------------------------------------------------------

/// A token-oriented quant palette: block `rank` holds the dim's 32 tokens.
template <typename B> struct I8QuantTag { using Block = B; };
/// A channel-oriented dtype palette: element (within, rank) at within·sub + rank.
template <typename E> struct I8DtypeTag { using Elem = E; };

// The block structs are addressed by sizeof(Block) here and by
// ArenaAccessor::get_quant_block_bytes on the accessor path; the two tables
// must agree or a template decoder mis-strides the palette span.
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q4_1) == 20 &&
              sizeof(block_q5_0) == 22 && sizeof(block_q5_1) == 24 &&
              sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36 &&
              sizeof(block_q4_ks) == 20 && sizeof(block_q8_ks) == 36 &&
              sizeof(block_q2_0) == 10 && sizeof(block_q3_0) == 14 &&
              sizeof(block_r16) == 128 && sizeof(block_q0) == 1 &&
              sizeof(block_q1_s) == 5 && sizeof(block_q2_s) == 9 &&
              sizeof(block_q2_a) == 10 && sizeof(block_q2_1) == 12 &&
              sizeof(block_q3_1) == 16 && sizeof(block_q0_v) == 2 &&
              sizeof(block_q1_a) == 6 && sizeof(block_q0_x) == 2 &&
              sizeof(block_q0_m2) == 3 && sizeof(block_q0_m4) == 8,
              "arena block struct sizes must match get_quant_block_bytes");

/// Run `f(tag)` for the arena format `fmt` — every quant family and every
/// dtype. A non-arena format fails loud, as the accessor's block addressing
/// does.
template <typename F>
__device__ __forceinline__ void i8_with_format(int fmt, F&& f)
{
    switch (fmt) {
#define I8_FMT_Q(FMT, B) case ArenaFormat::FMT: f(I8QuantTag<B>{}); break
#define I8_FMT_D(FMT, E) case ArenaFormat::FMT: f(I8DtypeTag<E>{}); break
        I8_FMT_Q(R16, block_r16);
        I8_FMT_Q(Q4_0, block_q4_0);
        I8_FMT_Q(Q4_1, block_q4_1);
        I8_FMT_Q(Q5_0, block_q5_0);
        I8_FMT_Q(Q5_1, block_q5_1);
        I8_FMT_Q(Q8_0, block_q8_0);
        I8_FMT_Q(Q8_1, block_q8_1);
        I8_FMT_Q(Q4_KS, block_q4_ks);
        I8_FMT_Q(Q8_KS, block_q8_ks);
        I8_FMT_Q(Q3_0, block_q3_0);
        I8_FMT_Q(Q3_1, block_q3_1);
        I8_FMT_Q(Q2_0, block_q2_0);
        I8_FMT_Q(Q2_1, block_q2_1);
        I8_FMT_Q(Q2_A, block_q2_a);
        I8_FMT_Q(Q2_S, block_q2_s);
        I8_FMT_Q(Q1_S, block_q1_s);
        I8_FMT_Q(Q0, block_q0);
        I8_FMT_Q(Q0_V, block_q0_v);
        I8_FMT_Q(Q1_A, block_q1_a);
        I8_FMT_Q(Q0_X, block_q0_x);
        I8_FMT_Q(Q0_M2, block_q0_m2);
        I8_FMT_Q(Q0_M4, block_q0_m4);
        I8_FMT_D(F16, __half);
        I8_FMT_D(BF16, __nv_bfloat16);
        I8_FMT_D(F32, float);
        I8_FMT_D(F8E4M3, __nv_fp8_e4m3);
#undef I8_FMT_Q
#undef I8_FMT_D
        default: __trap();
    }
}

/// Run `f(tag)` for a dtype format only. Callers gate on
/// `i8_is_dtype_format`; the switch's default arm is the FP8 case rather
/// than a trap so the dispatch is three compares on a warp-uniform word.
template <typename F>
__device__ __forceinline__ void i8_with_dtype_format(int fmt, F&& f)
{
    switch (fmt) {
        case ArenaFormat::F16:  f(I8DtypeTag<__half>{}); break;
        case ArenaFormat::BF16: f(I8DtypeTag<__nv_bfloat16>{}); break;
        case ArenaFormat::F32:  f(I8DtypeTag<float>{}); break;
        default:                f(I8DtypeTag<__nv_fp8_e4m3>{}); break;
    }
}

__device__ __forceinline__ bool i8_is_dtype_format(uint32_t fmt) {
    return fmt == ArenaFormat::BF16 || fmt == ArenaFormat::F16 ||
           fmt == ArenaFormat::F32 || fmt == ArenaFormat::F8E4M3;
}

/// `i8_with_dtype_format` less F32: the dtype formats whose four-element
/// vector is at most two words, for a caller that holds the raw words of
/// a quad across other work and has budgeted two registers per token for
/// them. Callers gate on `i8_is_narrow_dtype_format`; the default arm is
/// FP8, so the dispatch is two compares on a warp-uniform word.
template <typename F>
__device__ __forceinline__ void i8_with_narrow_dtype_format(int fmt, F&& f)
{
    switch (fmt) {
        case ArenaFormat::F16:  f(I8DtypeTag<__half>{}); break;
        case ArenaFormat::BF16: f(I8DtypeTag<__nv_bfloat16>{}); break;
        default:                f(I8DtypeTag<__nv_fp8_e4m3>{}); break;
    }
}

__device__ __forceinline__ bool i8_is_narrow_dtype_format(uint32_t fmt) {
    return fmt == ArenaFormat::BF16 || fmt == ArenaFormat::F16 || fmt == ArenaFormat::F8E4M3;
}

/// Run `f(tag)` for an int8 read-through format (the BlockInt8 families).
/// Callers gate on i8_tile_readthrough_format; anything else fails loud.
template <typename F>
__device__ __forceinline__ void i8_with_rt_format(int fmt, F&& f)
{
    switch (fmt) {
#define I8_RTF(FMT, B) case ArenaFormat::FMT: f(I8QuantTag<B>{}); break
        I8_RTF(Q8_0, block_q8_0);
        I8_RTF(Q4_0, block_q4_0);
        I8_RTF(Q5_0, block_q5_0);
        I8_RTF(Q2_0, block_q2_0);
        I8_RTF(Q3_0, block_q3_0);
        I8_RTF(Q8_1, block_q8_1);
        I8_RTF(Q2_S, block_q2_s);
        I8_RTF(Q1_S, block_q1_s);
        I8_RTF(Q1_A, block_q1_a);
        I8_RTF(Q0, block_q0);
        I8_RTF(Q0_M2, block_q0_m2);
        I8_RTF(Q0_M4, block_q0_m4);
        I8_RTF(Q0_X, block_q0_x);
#undef I8_RTF
        default: __trap();
    }
}

/// One dim of a palette span, resolved once per (thread, dim) and then read
/// per token with `at(within)`: FP32 divided by the palette scale
/// (load_head_scaled's semantics). `base` is the palette's span — its raw
/// shared-memory copy or its global bytes. The quant form holds the dim's
/// block pointer; the dtype form holds the dim's first element and the
/// scale reciprocal (`v * rcp(scale)` is what the fast-math divide computes,
/// hoisted out of the token loop).
// ----------------------------------------------------------------------------
// Aligned token quads of a quant block. `I8BlockQuad<B>::load4(blk, t, r, o)`
// decodes tokens t..t+3 of one dim's block, t a multiple of 4, as FP32 × r
// (r the palette scale reciprocal): the block header once, then the quad's
// codes with the fewest naturally aligned loads the layout allows. A block
// sits at rank × sizeof(B) from a 16-byte palette span, so a 34-byte block
// is 2-aligned and a 36-byte one 4-aligned; the offsets below respect
// that. The primary template is the per-element decoder run four times,
// which the layouts without a specialisation fall back to.
//
// The block is arena memory, which nothing writes while a decode kernel
// runs, so every load is `__ldg`: a global load the compiler knows cannot
// alias the shared stores and atomics a caller interleaves with the
// decode, and so can issue ahead of them. A plain load through a generic
// pointer might be shared memory as far as the compiler can tell, and each
// one then waits behind the previous slot's store.
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t i8_ld_u8(const void* p) { return __ldg((const unsigned char*)p); }
__device__ __forceinline__ uint32_t i8_ld_u16(const void* p) { return __ldg((const unsigned short*)p); }
__device__ __forceinline__ uint32_t i8_ld_u32(const void* p) { return __ldg((const unsigned int*)p); }
__device__ __forceinline__ float i8_h2f(uint32_t bits16) {
    return __half2float(__ushort_as_half((unsigned short)(bits16 & 0xffffu)));
}
/// Signed byte j of a little-endian word as FP32.
__device__ __forceinline__ float i8_s8_of(uint32_t w, int j) {
    return (float)(int)((int8_t)(w >> (8 * j)));
}

template <typename B> struct I8BlockQuad {
    static __device__ __forceinline__ void load4(const B* blk, int t, float r, float (&o)[4]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            o[j] = BlockConverter<B, float>::load_element(blk, t + j, 1.f) * r;
    }
};
template <> struct I8BlockQuad<block_q8_0> {          // half d; int8 qs[32]   (2-aligned)
    static __device__ __forceinline__ void load4(const block_q8_0* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const float d = i8_h2f(i8_ld_u16(p)) * r;
        const uint32_t w = i8_ld_u16(p + 2 + t) | (i8_ld_u16(p + 4 + t) << 16);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * i8_s8_of(w, j);
    }
};
template <> struct I8BlockQuad<block_q8_1> {          // half2 ds; int8 qs[32] (4-aligned)
    static __device__ __forceinline__ void load4(const block_q8_1* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const float d = i8_h2f(i8_ld_u32(p)) * r;
        const uint32_t w = i8_ld_u32(p + 4 + t);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * i8_s8_of(w, j);
    }
};
template <> struct I8BlockQuad<block_q8_ks> {         // half d; u8 sa, sb; int8 qs[32] (4-aligned)
    static __device__ __forceinline__ void load4(const block_q8_ks* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const uint32_t h = i8_ld_u32(p);
        const uint32_t fine = (t == 0) ? ((h >> 16) & 0xffu) : (h >> 24);   // tokens 0–3 are sub-block A
        const float d = i8_h2f(h) * ((float)fine * INV_255) * r;
        const uint32_t w = i8_ld_u32(p + 4 + t);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * i8_s8_of(w, j);
    }
};
template <> struct I8BlockQuad<block_q4_0> {          // half d; u8 qs[16]      (2-aligned)
    static __device__ __forceinline__ void load4(const block_q4_0* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const float d = i8_h2f(i8_ld_u16(p)) * r;
        const int b = 2 + (t & 15);
        const uint32_t w = (i8_ld_u16(p + b) | (i8_ld_u16(p + b + 2) << 16)) >> ((t >> 4) * 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * ((float)((w >> (8 * j)) & 15u) - 8.f);
    }
};
template <> struct I8BlockQuad<block_q4_1> {          // half2 dm; u8 qs[16]    (4-aligned)
    static __device__ __forceinline__ void load4(const block_q4_1* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const uint32_t dm = i8_ld_u32(p);
        const float d = i8_h2f(dm) * r, m = i8_h2f(dm >> 16) * r;
        const uint32_t w = i8_ld_u32(p + 4 + (t & 15)) >> ((t >> 4) * 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * (float)((w >> (8 * j)) & 15u) + m;
    }
};
template <> struct I8BlockQuad<block_q4_ks> {         // half d; u8 sa, sb; u8 qs[16] (4-aligned)
    static __device__ __forceinline__ void load4(const block_q4_ks* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const uint32_t h = i8_ld_u32(p);
        const uint32_t fine = (t == 0) ? ((h >> 16) & 0xffu) : (h >> 24);
        const float d = i8_h2f(h) * ((float)fine * INV_255) * r;
        const uint32_t w = i8_ld_u32(p + 4 + (t & 15)) >> ((t >> 4) * 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * ((float)((w >> (8 * j)) & 15u) - 8.f);
    }
};
template <> struct I8BlockQuad<block_q3_0> {          // half d; u8 qh[4]; u8 qs[8] (2-aligned)
    static __device__ __forceinline__ void load4(const block_q3_0* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const float d = i8_h2f(i8_ld_u16(p)) * r;
        const uint32_t qh = i8_ld_u8(p + 2 + (t >> 3)) >> (t & 7);
        const uint32_t qs = i8_ld_u8(p + 6 + (t >> 2));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = ((qs >> (2 * j)) & 3u) | (((qh >> j) & 1u) << 2);
            o[j] = d * ((float)q - 3.5f);
        }
    }
};
template <> struct I8BlockQuad<block_q3_1> {          // half2 dm; u8 qh[4]; u8 qs[8] (4-aligned)
    static __device__ __forceinline__ void load4(const block_q3_1* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const uint32_t dm = i8_ld_u32(p);
        const float d = i8_h2f(dm) * r, m = i8_h2f(dm >> 16) * r;
        const uint32_t qh = i8_ld_u8(p + 4 + (t >> 3)) >> (t & 7);
        const uint32_t qs = i8_ld_u8(p + 8 + (t >> 2));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = ((qs >> (2 * j)) & 3u) | (((qh >> j) & 1u) << 2);
            o[j] = d * (float)q + m;
        }
    }
};
template <> struct I8BlockQuad<block_q2_0> {          // half d; u8 qs[8]       (2-aligned)
    static __device__ __forceinline__ void load4(const block_q2_0* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const float d = i8_h2f(i8_ld_u16(p)) * r;
        const uint32_t qs = i8_ld_u8(p + 2 + (t >> 2));
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * ((float)((qs >> (2 * j)) & 3u) - 1.5f);
    }
};
template <> struct I8BlockQuad<block_q2_1> {          // half2 dm; u8 qs[8]     (4-aligned)
    static __device__ __forceinline__ void load4(const block_q2_1* blk, int t, float r, float (&o)[4]) {
        const uint8_t* p = (const uint8_t*)blk;
        const uint32_t dm = i8_ld_u32(p);
        const float d = i8_h2f(dm) * r, m = i8_h2f(dm >> 16) * r;
        const uint32_t qs = i8_ld_u8(p + 4 + (t >> 2));
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = d * (float)((qs >> (2 * j)) & 3u) + m;
    }
};
template <> struct I8BlockQuad<block_r16> {           // half d[32]; u16 q[32]  (16-aligned)
    static __device__ __forceinline__ void load4(const block_r16* blk, int t, float r, float (&o)[4]) {
        const uint2 w = __ldg((const uint2*)((const uint8_t*)blk + 2 * t));
        o[0] = i8_h2f(w.x) * r;
        o[1] = i8_h2f(w.x >> 16) * r;
        o[2] = i8_h2f(w.y) * r;
        o[3] = i8_h2f(w.y >> 16) * r;
    }
};

template <typename B> struct I8QuantDim {
    const B* blk;
    float scale;
    __device__ __forceinline__ float at(int within) const {
        return BlockConverter<B, float>::load_element(blk, within, scale);
    }
};
template <typename E> struct I8DtypeDim {
    const E* first;
    int sub;
    float inv_scale;
    __device__ __forceinline__ float at(int within) const {
        return to_float<E>(first[(int64_t)within * sub]) * inv_scale;
    }
};
template <typename B>
__device__ __forceinline__ I8QuantDim<B> i8_tag_dim(
    I8QuantTag<B>, const char* base, int rank, int sub, float scale)
{
    return I8QuantDim<B>{ (const B*)(base + (int64_t)rank * sizeof(B)), scale };
}
template <typename E>
__device__ __forceinline__ I8DtypeDim<E> i8_tag_dim(
    I8DtypeTag<E>, const char* base, int rank, int sub, float scale)
{
    return I8DtypeDim<E>{ (const E*)base + rank, sub, 1.f / scale };
}

/// True for the dtype tags — the palettes a quad can read as one vector.
template <typename Tag> struct I8IsDtypeTag { static constexpr bool value = false; };
template <typename E> struct I8IsDtypeTag<I8DtypeTag<E>> { static constexpr bool value = true; };

/// The words of one token's four-element vector of E: four for F32, two
/// for the 16-bit types, one for FP8.
template <typename E> constexpr int i8_dtype_words = (int)sizeof(E);

/// The four elements of one token's quad from its raw words (`raw4`'s
/// layout, the first `i8_dtype_words<E>` of `w`), unscaled.
template <typename E, int W>
__device__ __forceinline__ void i8_dtype_cvt4(const uint32_t (&w)[W], float (&o)[4])
{
    static_assert(W >= i8_dtype_words<E>, "the words of a four-element vector of E");
    if constexpr (sizeof(E) == 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) o[k] = __uint_as_float(w[k]);
    } else if constexpr (std::is_same_v<E, __nv_bfloat16>) {
        o[0] = __uint_as_float(w[0] << 16);
        o[1] = __uint_as_float(w[0] & 0xffff0000u);
        o[2] = __uint_as_float(w[1] << 16);
        o[3] = __uint_as_float(w[1] & 0xffff0000u);
    } else if constexpr (sizeof(E) == 2) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[0]));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&w[1]));
        o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint8_t b = (uint8_t)(w[0] >> (8 * k));
            o[k] = to_float<E>(*reinterpret_cast<const E*>(&b));
        }
    }
}

/// Four consecutive ranks of a dtype palette, read per token as ONE aligned
/// vector of 4·sizeof(E) bytes at first + within·sub. The vector never
/// straddles a token when the first rank and `sub` are multiples of 4 (the
/// palette spans are 16-byte aligned), which is what `i8_tag_quad`'s
/// callers check. `raw4` is the load alone — the vector's words, so a
/// caller can issue several tokens' loads before any converts — and
/// `cvt4` the elements × the scale reciprocal; a palette at the default
/// outer scale of 1.0 (every dtype palette the seal writes without an
/// override) skips the four multiplies, a quad-uniform branch. `at4` is
/// the two together. The word array is the caller's, sized for the
/// widest format it holds; only the first `WORDS` are touched.
template <typename E> struct I8DtypeQuad {
    static constexpr int WORDS = i8_dtype_words<E>;
    const E* first;
    int sub;
    float inv_scale;
    // Coherent loads, not `__ldg`: the kernel's own scatter writes the
    // step's token into the write slice this same launch reads back.
    template <int W>
    __device__ __forceinline__ void raw4(int within, uint32_t (&w)[W]) const {
        static_assert(W >= WORDS, "the words of a four-element vector of E");
        const E* p = first + (int64_t)within * sub;
        if constexpr (sizeof(E) == 4) {
            const uint4 v = *(const uint4*)p;
            w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
        } else if constexpr (sizeof(E) == 2) {
            const uint2 v = *(const uint2*)p;
            w[0] = v.x; w[1] = v.y;
        } else {
            w[0] = *(const uint32_t*)p;
        }
    }
    template <int W>
    __device__ __forceinline__ void cvt4(const uint32_t (&w)[W], float (&o)[4]) const {
        i8_dtype_cvt4<E>(w, o);
        if (inv_scale != 1.f) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) o[k] *= inv_scale;
        }
    }
    __device__ __forceinline__ void at4(int within, float (&o)[4]) const {
        uint32_t w[WORDS];
        raw4(within, w);
        cvt4(w, o);
    }
};
template <typename E>
__device__ __forceinline__ I8DtypeQuad<E> i8_tag_quad(
    I8DtypeTag<E>, const char* base, int rank, int sub, float scale)
{
    return I8DtypeQuad<E>{ (const E*)base + rank, sub, 1.f / scale };
}

/// Two values quantised against one scale, packed little-endian in the low
/// half-word (a in the low byte): two FFMAs (`i8_code_bits`) and one byte
/// permute gathering their low bytes. The high half-word is unspecified.
__device__ __forceinline__ uint32_t i8_pack2(float a, float b, float inv_scale) {
    return __byte_perm(i8_code_bits(a, inv_scale), i8_code_bits(b, inv_scale), 0x0040u);
}

/// Four values quantised against one window scale (0 ⇒ zeros) and packed
/// little-endian, v[0] in the low byte: four FFMAs and three byte permutes
/// for the four bytes, against the FMUL + F2I per value and two saturating
/// packs of a conversion.
__device__ __forceinline__ uint32_t i8_pack4(const float (&v)[4], float inv_scale) {
    const uint32_t lo = i8_pack2(v[0], v[1], inv_scale);
    const uint32_t hi = i8_pack2(v[2], v[3], inv_scale);
    return __byte_perm(lo, hi, 0x5410u);
}

/// Int8 read-through sample of element `within` of dim `rank` under a
/// read-through tag: the centred int8 and its per-(dim, block) scale.
template <typename B>
__device__ __forceinline__ Int8Sample i8_tag_rt(
    I8QuantTag<B>, const char* base, int rank, int within)
{
    return BlockInt8<B>::load((const B*)(base + (int64_t)rank * sizeof(B)), within);
}

/// One arena element as FP32, from either a quant-block span (bb > 0:
/// `base` is the palette's raw smem copy or its global span) or a
/// channel-oriented dtype palette (bb == 0: element addressing).
/// Matches load_head_scaled's semantics: decoded value / scale (the
/// dtype identity fast path skips the divide only when scale == 1.0f,
/// where /1.0f is exact anyway).
__device__ __forceinline__ float i8_arena_elem(
    int fmt, int bb, const char* base, int rank, int within, float scale, int sub)
{
    if (bb > 0)
        return i8_dequant_elem(fmt, base + (int64_t)rank * bb, within, scale);
    const int es = ArenaFormat::float_elem_size(fmt);
    const char* pe = base + ((int64_t)within * sub + rank) * es;
    float v;
    if (fmt == ArenaFormat::F16) {
        v = __half2float(*(const __half*)pe);
    } else if (fmt == ArenaFormat::BF16) {
        v = __bfloat162float(*(const __nv_bfloat16*)pe);
    } else if (fmt == ArenaFormat::F32) {
        v = *(const float*)pe;
    } else { // F8E4M3
        v = to_float<__nv_fp8_e4m3>(*(const __nv_fp8_e4m3*)pe);
    }
    return v / scale;
}

} // namespace int8_elem
