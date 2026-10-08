#pragma once
// ============================================================================
// K/V PRE-STAGING — every position of a bulk launch's sequences, staged once
// ============================================================================
//
// One warp per four consecutive positions of one (sequence, KV head): the
// same `i8_stage_quad` the attention kernel's own staging runs, for the quad
// that starts at the position's four-aligned group, written to the stage
// planes (`kv_stage.cuh`) instead of a tile. The attention kernel takes a
// warp's columns from here exactly when its own staging would have decoded
// that same four-aligned quad, so a staged column carries the bits a decoded
// one would.
//
// Grid: x over 32-position groups (8 warps × 4) of the deepest staged
// sequence, y = KV head, z = sequence. A sequence too short to stage, and a
// warp past its sequence's kv length, exit at once.
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "column_stage.cuh"
#include "kv_stage.cuh"

namespace prefill_int8 {

constexpr int KV_PRESTAGE_WARPS = 8;
constexpr int KV_PRESTAGE_THREADS = KV_PRESTAGE_WARPS * 32;
constexpr int KV_PRESTAGE_POSITIONS = KV_PRESTAGE_WARPS * 4; // per block

template <typename QT, int HEAD_DIM>
__global__ void __launch_bounds__(KV_PRESTAGE_THREADS)
paged_prefill_kv_prestage_kernel(
    const QT* __restrict__ k_packed,   // [total_q, n_kv_head, HD] packed, unrotated
    const QT* __restrict__ v_packed,   // [total_q, n_kv_head, HD]
    const uint8_t* __restrict__ headers_ptr,
    const uint32_t* __restrict__ cu_seqlens_q,
    const uint32_t* __restrict__ q_lens,
    const uint32_t* __restrict__ kv_lens,
    int n_kv_head,
    const RopeRungs rungs,
    int rope_interleaved,
    const PrefillKvStage stage
) {
    constexpr int N_WIN = HEAD_DIM / 32;
    __shared__ WarpPalette<HEAD_DIM> s_wpal[KV_PRESTAGE_WARPS];

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int b = (int)blockIdx.z;
    const int kv_head_idx = (int)blockIdx.y;

    const int q_len = (int)q_lens[b];
    if (q_len < stage.min_q_len) return; // block-uniform: not a staged sequence
    const int kv_len = (int)kv_lens[b];
    const int pos0 = ((int)blockIdx.x * KV_PRESTAGE_WARPS + warp) * 4;
    if (pos0 >= kv_len) return; // warp-uniform: nothing below `__syncwarp` scope
    int prefix_len = kv_len - q_len;
    if (prefix_len < 0) prefix_len = 0;

    const int64_t seq_base = kv_stage_seq_base(q_lens, kv_lens, b, stage.min_q_len);
    // The host sized the planes from its own lengths; a device length that
    // reaches past them would write another sequence's rows, or past the
    // buffer. Refuse rather than stage over them.
    if (seq_base + kv_len > stage.positions) __trap();

    const SlotHeader& slot_hdr = get_slot_header(headers_ptr, b);
    const RopeView rope = rope_view(rungs, slot_hdr.rope_rung);
    const int q_start = (int)cu_seqlens_q[b];

    const int64_t row0 = (int64_t)kv_head_idx * stage.positions + seq_base + pos0;
    int8_t* const kc = kv_stage_k<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;
    __half* const ks = kv_stage_k_scale<HEAD_DIM>(stage, n_kv_head) + row0 * N_WIN;
    __half* const vv = kv_stage_v<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;

    int bound_slice = -1;
    i8_stage_quad<QT, HEAD_DIM>(
        s_wpal[warp], bound_slice, slot_hdr, n_kv_head, kv_head_idx,
        k_packed, v_packed, q_start, prefix_len, kv_len, pos0, lane,
        rope_interleaved, rope,
        [&](int tt, int w, int8_t code, float scale) {
            if (pos0 + tt >= kv_len) return;
            kc[tt * HEAD_DIM + lane + 32 * w] = code;
            if (lane == 0) ks[tt * N_WIN + w] = __float2half(scale);
        },
        [&](int tt, int w, float v) {
            if (pos0 + tt >= kv_len) return;
            vv[tt * HEAD_DIM + lane + 32 * w] = __float2half(v);
        });
}

/// Stage every position of the launch's sequences with `q_len >=
/// stage.min_q_len` into `stage`. `max_kv_len` is the deepest of them, which
/// sizes the grid; `stage_bytes` is the buffer's length, refused when the
/// planes for `stage.positions` do not fit it.
template <typename QT, int HEAD_DIM>
inline void launch_paged_prefill_kv_prestage(
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    int32_t batch_size,
    int32_t n_kv_head,
    int32_t max_kv_len,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    const PrefillKvStage stage,
    int64_t stage_bytes,
    cudaStream_t stream
) {
    const int64_t need = kv_stage_bytes<HEAD_DIM>(stage.positions, n_kv_head);
    if (stage.buf == nullptr || need > stage_bytes || max_kv_len <= 0) {
        fprintf(stderr,
                "PAGED PREFILL KV STAGE: %lld B of planes for %lld positions × %d KV heads "
                "(hd %d, deepest %d) against a %lld B buffer at %p\n",
                (long long)need, (long long)stage.positions, n_kv_head, HEAD_DIM, max_kv_len,
                (long long)stage_bytes, (void*)stage.buf);
        abort();
    }
    dim3 grid((uint32_t)((max_kv_len + KV_PRESTAGE_POSITIONS - 1) / KV_PRESTAGE_POSITIONS),
              (uint32_t)n_kv_head, (uint32_t)batch_size);
    paged_prefill_kv_prestage_kernel<QT, HEAD_DIM><<<grid, KV_PRESTAGE_THREADS, 0, stream>>>(
        (const QT*)k_ptr, (const QT*)v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
        (int)n_kv_head, rungs, (int)rope_interleaved, stage);
}

} // namespace prefill_int8
