#pragma once
// ============================================================================
// PRE-STAGED K/V — the layout a bulk prefill launch reads its tiles from
// ============================================================================
//
// A bulk prefill launch serves a few query tokens per block, and every block
// that selects a key position decodes it again: format dispatch, palette
// rank, RoPE, the per-window absmax and int8 quantisation of K, the FP16
// conversion of V. The pre-staging pass (`kv_prestage_kernel.cuh`) does that
// once per position, for every sequence of the launch long enough to amortise
// it, and the attention kernel's staging copies a tile's columns out of the
// result with 16-byte `cp.async` — the same bytes its own decode would have
// written into the tile slabs, so everything after the staging barrier is
// unchanged.
//
// Three planes, each [n_kv_head][positions][…], position-major within a head
// so a warp's four consecutive columns are one contiguous run:
//
//   K codes   int8  [HD]        post-RoPE, quantised per 32-dim window
//   K scales  fp16  [HD / 32]   the windows' absmax / 127
//   V         fp16  [HD]        natural dim order
//
// A launch stages one KEY WINDOW `[chunk_lo, chunk_hi)` of each staged
// sequence that has a query token in its row group (a token at or past
// `tok_lo`): a sequence's rows are its positions in the window, and they start
// at the sum of the window rows of the staged sequences before it in the
// batch. `positions` is the planes' row count per head — the widest window
// any launch over the batch stages — so the plane offsets are the same for
// every launch. An uncut launch's window is `[0, KV_STAGE_UNBOUNDED)`, which
// stages every sequence whole. The first two planes are rounded up to 256
// bytes so every plane starts on the bump alignment. The byte count is stated
// once on the host as well (`candle_nn::kv_cache::prefill_kv_stage_bytes`),
// and the launcher refuses a buffer shorter than the layout here.
// ============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

namespace prefill_int8 {

/// A key window with no upper end.
constexpr int KV_STAGE_UNBOUNDED = 0x7fffffff;

/// The pre-staged K/V a launch reads, or `buf == nullptr` for a launch that
/// stages nothing. A sequence is staged iff its `q_len >= min_q_len`; this
/// launch stages its key positions in `[chunk_lo, chunk_hi)` iff it also has a
/// query token at or past `tok_lo`.
struct PrefillKvStage {
    uint8_t* buf;
    int64_t positions;
    int min_q_len;
    int chunk_lo;
    int chunk_hi;
    int tok_lo;
};

constexpr int64_t KV_STAGE_ALIGN = 256;

__host__ __device__ constexpr int64_t kv_stage_align(int64_t bytes) {
    return (bytes + KV_STAGE_ALIGN - 1) / KV_STAGE_ALIGN * KV_STAGE_ALIGN;
}

template <int HEAD_DIM>
__host__ __device__ constexpr int64_t kv_stage_k_bytes(int64_t positions, int n_kv_head) {
    return kv_stage_align((int64_t)n_kv_head * positions * HEAD_DIM);
}

template <int HEAD_DIM>
__host__ __device__ constexpr int64_t kv_stage_scale_bytes(int64_t positions, int n_kv_head) {
    return kv_stage_align((int64_t)n_kv_head * positions * (HEAD_DIM / 32) * 2);
}

template <int HEAD_DIM>
__host__ __device__ constexpr int64_t kv_stage_v_bytes(int64_t positions, int n_kv_head) {
    return (int64_t)n_kv_head * positions * HEAD_DIM * 2;
}

/// Bytes the three planes take for `positions` staged positions.
template <int HEAD_DIM>
__host__ __device__ constexpr int64_t kv_stage_bytes(int64_t positions, int n_kv_head) {
    return kv_stage_k_bytes<HEAD_DIM>(positions, n_kv_head)
         + kv_stage_scale_bytes<HEAD_DIM>(positions, n_kv_head)
         + kv_stage_v_bytes<HEAD_DIM>(positions, n_kv_head);
}

template <int HEAD_DIM>
__device__ __forceinline__ int8_t* kv_stage_k(const PrefillKvStage& s, int n_kv_head) {
    return reinterpret_cast<int8_t*>(s.buf);
}

template <int HEAD_DIM>
__device__ __forceinline__ __half* kv_stage_k_scale(const PrefillKvStage& s, int n_kv_head) {
    return reinterpret_cast<__half*>(s.buf + kv_stage_k_bytes<HEAD_DIM>(s.positions, n_kv_head));
}

template <int HEAD_DIM>
__device__ __forceinline__ __half* kv_stage_v(const PrefillKvStage& s, int n_kv_head) {
    return reinterpret_cast<__half*>(s.buf + kv_stage_k_bytes<HEAD_DIM>(s.positions, n_kv_head)
                                           + kv_stage_scale_bytes<HEAD_DIM>(s.positions, n_kv_head));
}

/// Rows a sequence of `kv_len` positions stages in `s`'s window.
__host__ __device__ __forceinline__ int64_t kv_stage_seq_rows(int kv_len, const PrefillKvStage& s) {
    const int64_t n = (int64_t)kv_len - s.chunk_lo;
    const int64_t w = (int64_t)s.chunk_hi - s.chunk_lo;
    return n < 0 ? 0 : (n > w ? w : n);
}

/// Where sequence `b`'s window rows start within one head's run: the window
/// rows of the staged sequences before it that have a token in the row group.
/// A batch is at most a wave's sequences, so the walk is a handful of cached
/// loads per block.
__device__ __forceinline__ int64_t kv_stage_seq_base(
    const uint32_t* __restrict__ q_lens, const uint32_t* __restrict__ kv_lens,
    int b, const PrefillKvStage& s)
{
    int64_t base = 0;
    for (int i = 0; i < b; ++i) {
        const int qi = (int)q_lens[i];
        if (qi >= s.min_q_len && qi > s.tok_lo) base += kv_stage_seq_rows((int)kv_lens[i], s);
    }
    return base;
}

/// `N`-byte global→shared async copy (4, 8 or 16; both addresses `N`-aligned).
/// 16 bytes bypasses L1 (`.cg`) — a staged row is read once per tile; the
/// narrower forms only exist as `.ca`.
template <int N>
__device__ __forceinline__ void kv_stage_cp_async(void* dst, const void* src) {
    static_assert(N == 4 || N == 8 || N == 16, "cp.async copies 4, 8 or 16 bytes");
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    if constexpr (N == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(d), "l"(src), "n"(N));
    }
}

} // namespace prefill_int8
