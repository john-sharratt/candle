// =============================================================================
// MoE BUCKETIZE — shared definitions
// =============================================================================
// The bounds, encodings and block-wide scan both bucketize kernels use: the
// general one (`moe_bucketize.cu`) and the narrow one
// (`moe_bucketize_narrow.cuh`). Each kernel is ONE block of BUCKETIZE_THREADS
// threads — one thread per expert up to MAX_EXPERTS, which every per-expert
// scan relies on.
// =============================================================================
#pragma once

#include <stdint.h>
#include <stdio.h>

#define BUCKETIZE_THREADS 512
#define BUCKETIZE_WARPS (BUCKETIZE_THREADS / 32)
// MAX_EXPERTS 512 is Qwen3.8-Flash-Next's width (Qwen3-MoE has 128, Qwen3.5 has
// 256).
#define MAX_EXPERTS 512
static_assert(MAX_EXPERTS <= BUCKETIZE_THREADS, "the per-expert scans hold one expert per thread");
#define MAX_TOPK 32
#define INVALID_ROW 0xFFFFFFFFu
// The token ranges `[lo, hi)` scored as decode, passed by value in the launch
// parameters: a wave's decode rows, its verify segments, and the last token of
// each prompt — the token the sequence's decode continues from.
#define MAX_DECODE_RANGES 32
struct DecodeRanges {
    uint32_t n;
    uint32_t lo[MAX_DECODE_RANGES];
    uint32_t hi[MAX_DECODE_RANGES];
};
// Where an expert's gate entry points.
#define CLS_VRAM 0
#define CLS_PINNED 1
#define CLS_COLD 2
// A promotion offer with no resident expert behind it.
#define PROMO_EMPTY 0xFFFFFFFFFFFFFFFFull
// The expert a skipped victim's log entry names.
#define PROMO_SKIP 0xFFFFu

// The parameter list both kernels take, so the launcher hands either the same
// arguments. Documented at `moe_bucketize_kernel` in `moe_bucketize.cu`.
#define MOE_BUCKETIZE_KERNEL_PARAMS                                                           \
    const uint32_t* __restrict__ topk_ids, const int n_tokens, const int k,                   \
    const int n_experts, const int tile_w, uint32_t* __restrict__ tok_ids,                    \
    uint32_t* __restrict__ weight_ids, int32_t* __restrict__ tile_expert,                     \
    int32_t* __restrict__ tile_b_start, int32_t* __restrict__ tile_b_cnt,                     \
    uint32_t* __restrict__ perm, uint32_t* __restrict__ rw_ids,                               \
    int32_t* __restrict__ token_starts, int32_t* __restrict__ header,                         \
    uint32_t* __restrict__ inv, int32_t* __restrict__ scan, const uint64_t* gate_row,         \
    const long long table_plane, uint64_t* __restrict__ snap, const uint64_t pinned0_lo,      \
    const uint64_t pinned0_hi, const uint64_t pinned1_lo, const uint64_t pinned1_hi,          \
    const DecodeRanges decode, uint32_t* __restrict__ summary, const uint32_t summary_seq,    \
    int32_t* __restrict__ remote, int32_t* __restrict__ counters,                             \
    const uint64_t* promo_slots, uint64_t* promo_log, uint32_t* promo_head,                   \
    const uint32_t* promo_tail, const uint32_t promo_cap, uint32_t* promo_marks,              \
    const uint32_t* promo_reserve, const uint32_t* promo_sweep,                               \
    const uint64_t* promo_victims, const uint64_t* promo_retarget,                            \
    const uint32_t* slot_owner, const uint64_t zone_end, const uint64_t zone_slot_bytes,      \
    const uint32_t zone_slots, const int32_t row, uint64_t* __restrict__ remote_dst,          \
    uint64_t* started_rows, const uint64_t ticket, const uint32_t* ahead_window,              \
    const uint32_t* ahead_depth, const uint32_t* ahead_n, const uint32_t* ahead_list,         \
    const uint64_t* ahead_src, const uint32_t ahead_cap, const int32_t rows,                  \
    const uint64_t* row_layout, uint64_t* ahead_items, uint32_t* ahead_done

// The tile order's rank of each class: pinned, then cold, then VRAM.
__device__ __forceinline__ int class_rank(uint8_t cls) {
    return cls == CLS_PINNED ? 0 : (cls == CLS_COLD ? 1 : 2);
}

// Whether token `t` lies in one of the decode ranges.
__device__ __forceinline__ bool token_is_decode(uint32_t t, const DecodeRanges& decode) {
    for (uint32_t r = 0; r < decode.n; r++) {
        if (t >= decode.lo[r] && t < decode.hi[r]) {
            return true;
        }
    }
    return false;
}

// Inclusive scan across a warp.
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
    const int lane = (int)(threadIdx.x & 31);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const T n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) {
            v += n;
        }
    }
    return v;
}

// Exclusive scan of `v` across the block, in thread order; the block total
// lands in `*total`. `sh` is this scan's own `[BUCKETIZE_WARPS + 1]` buffer.
// Every thread of the block must call it.
template <typename T>
__device__ __forceinline__ T block_exclusive_scan(T v, T* sh, T* total) {
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    const T inclusive = warp_inclusive_scan(v);
    if (lane == 31) {
        sh[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        const T w = lane < BUCKETIZE_WARPS ? sh[lane] : (T)0;
        const T wi = warp_inclusive_scan(w);
        if (lane < BUCKETIZE_WARPS) {
            sh[lane] = wi - w;
        }
        if (lane == BUCKETIZE_WARPS - 1) {
            sh[BUCKETIZE_WARPS] = wi;
        }
    }
    __syncthreads();
    *total = sh[BUCKETIZE_WARPS];
    return sh[warp] + inclusive - v;
}
