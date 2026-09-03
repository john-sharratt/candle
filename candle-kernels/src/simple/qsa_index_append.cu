// =============================================================================
// QSA index-cache append: pool → RMS-norm → RoPE → store, for a whole wave
// =============================================================================
// `IndexCache::append` prepares one indexer key per `ratio` tokens. A completed
// block's key is a function of the block alone — the mean of its `ratio` raw
// projected rows, the indexer's `k_norm`, and a rotation at the block's FIRST
// absolute position — so it is computed once and stored ready to score.
//
// Eagerly that is roughly THIRTY launches per sequence per layer per wave, and
// `nsys` over the selection microbenchmark says what they are: a `cat` of the
// open block's held rows, `ratio − 1` strided `badd_f32`, an `affine_f32` for
// the mean, an RMS-norm expanded into `usqr`/`fast_sum`/`bdiv`/`usqrt`/`bmul`,
// a RoPE expanded into an `is_u32_f32` gather of the cos/sin rows plus four
// `bmul`/`badd`, and a `slice_assign` expanded into `const_set_u8`,
// `where_u8_f32` and `copy2d_u8` — because candle's `slice_assign` writes a
// whole new tensor rather than the rows that changed.
//
// Measured on the width sweep of `tests/qsa_index_bench.rs`: ~21,500 launches
// for 20.5 ms of GPU time against ~208 ms of wall clock. The GPU was busy a
// tenth of the time and the actual GEMMs were 7% of even that. The arithmetic
// was never the cost; issuing it was.
//
// So the whole append runs as ONE launch over every completing block of every
// sequence in the wave (hot-path invariant 5), writing its keys directly into
// each sequence's cache (which is also what removes the `slice_assign`).
//
// ---- The descriptor table ---------------------------------------------------
//
// A block's `ratio` raw rows do NOT live in one dense run, and requiring that
// they did would put the copy back. The block that completes first in a wave
// straddles two sources: the rows the PREVIOUS wave left in the open block, and
// the rows this wave contributed. Every later block of the span is contiguous
// in the wave's projection.
//
// So a job carries at most two SEGMENTS and reads both in place through their
// own base pointers (hot-path invariant 2b). Rows are contiguous within a
// segment at `d` elements apart, which every producer guarantees: the wave
// projection is row-major and the carry buffer is written by
// `qsa_index_carry` below.
//
// Table layout: array-of-structs, QSA_APPEND_JOB_WORDS i64 per job, read once
// per block and broadcast to its threads.
//
//     [0] dst    float*        this block key's row in the sequence's cache
//     [1] src0   const float*  first segment's base row (0 when the block does
//                              not straddle)
//     [2] n0     rows taken from segment 0; segment 1 supplies `ratio − n0`
//     [3] src1   const float*  second segment's base row
//     [4] pos    the block's first absolute position — its rotation angle
//
// ---- Shape ------------------------------------------------------------------
//
// One block per job, `d` threads, each owning one channel. At the released
// geometry `d` is 128, so a block is four FULL warps and the RMS-norm's
// reduction is two shuffle rounds plus a four-entry shared fold.
//
// Fusing the norm is the one place this kernel can go further than
// `compressor_pool`, whose note explains why it left its own RMS-norm out: it
// tiles `d` across blocks to have any blocks at all, so folding the norm in
// would need a cross-block reduction. Here `d` is a single block's width, so
// the reduction is internal and the norm is free.
//
// `float4` over the channel axis is NOT used, and `compressor_pool`'s note is
// the reason — it was implemented there, measured at every shape, and was
// slower at all of them. With `d` fixed, four channels per thread means a
// quarter of the lanes, and this kernel is short of parallel work at decode
// width (a wave completes about one block per sequence), not short of
// bandwidth. The vector win here is that eight passes over memory collapse into
// registers, not that any one pass moves wider words.
//
// ---- Numerics ---------------------------------------------------------------
//
// The expression is the eager chain's, in the eager chain's order: the mean
// before the norm, `x / sqrt(mean(x²) + eps) · w` with a true divide rather
// than `rsqrtf` (the archive compiles `--use_fast_math`, and `rsqrtf` is a
// different, and differently-rounded, function), then the NeoX half-split
// rotation. It is not bit-exact to the eager chain and should not be expected
// to be: fusing keeps the intermediates in registers where the eager path
// rounded each one out to memory. `qsa_index_append_matches_host` gates it the
// way `compressor_pool` gates its own fusion — against exact host arithmetic,
// asserting the kernel is at least as faithful as the chain it replaces.

#include <cuda_runtime.h>
#include <stdint.h>

#define QSA_APPEND_JOB_WORDS 5
#define QSA_APPEND_CARRY_WORDS 3

namespace qsa_index_append {

// The widest `d` a single block can own, which is also the reduction's bound.
constexpr int MAX_D = 1024;

// Sum across the block through the buffer the rotation needs anyway, so the
// norm costs no shared memory of its own.
//
// A tree rather than warp shuffles: `d` is not required to be a multiple of 32
// (the indexer's own oracle runs a 16-wide head), and a shuffle whose mask
// names lanes the launch never created is undefined rather than merely wrong.
// The tree is correct for any `blockDim.x`, and on a kernel whose whole purpose
// is to stop paying launch overhead the extra barriers are not the cost.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int c = threadIdx.x;
    const int n = blockDim.x;
    scratch[c] = v;
    __syncthreads();
    for (int s = 1; s < n; s <<= 1) {
        const int partner = c + s;
        float add = (partner < n) ? scratch[partner] : 0.0f;
        __syncthreads();
        if ((c & ((s << 1) - 1)) == 0) scratch[c] += add;
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// One completing block key per CUDA block.
//
// `cos`/`sin` are the rope tables, `[max_pos, rope_dim/2]` row-major, so a
// block reads its own row at `pos`.
__global__ __launch_bounds__(MAX_D) void append_kernel(
    const long long* __restrict__ jobs,
    const float* __restrict__ k_norm,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int d,
    int rope_dim,
    int ratio,
    float eps,
    int n_jobs
) {
    const int j = blockIdx.x;
    if (j >= n_jobs) return;
    const int c = threadIdx.x;

    const long long* job = jobs + (long long)j * QSA_APPEND_JOB_WORDS;
    float* dst = (float*)(uintptr_t)job[0];
    const float* src0 = (const float*)(uintptr_t)job[1];
    const int n0 = (int)job[2];
    const float* src1 = (const float*)(uintptr_t)job[3];
    const int pos = (int)job[4];

    // ---- pool: the mean of the block's `ratio` raw rows -----------------
    float acc = 0.0f;
    for (int r = 0; r < n0; ++r) {
        acc += src0[(long long)r * d + c];
    }
    for (int r = n0; r < ratio; ++r) {
        acc += src1[(long long)(r - n0) * d + c];
    }
    float x = acc / (float)ratio;

    // ---- RMS-norm over the channel axis ---------------------------------
    // Dynamic, so a block reserves `d` floats rather than the 1024-wide worst
    // case: at the released `d` of 128 that is 512 bytes against 4,096, which
    // is the difference between shared memory bounding occupancy and not
    // appearing in it at all.
    extern __shared__ float sv[];
    const float ss = block_sum(x * x, sv);
    // `mean_keepdim` then `+ eps` then `sqrt`, then a divide — the eager form.
    x = x / sqrtf(ss / (float)d + (float)eps);
    x = x * k_norm[c];

    // ---- RoPE, NeoX half-split ------------------------------------------
    // Channel `c < half` pairs with `c + half`; `[rope_dim, d)` passes through.
    sv[c] = x;
    __syncthreads();
    const int half = rope_dim >> 1;
    float out;
    if (c < half) {
        const float co = cos_tab[(long long)pos * half + c];
        const float si = sin_tab[(long long)pos * half + c];
        out = sv[c] * co - sv[c + half] * si;
    } else if (c < rope_dim) {
        const int k = c - half;
        const float co = cos_tab[(long long)pos * half + k];
        const float si = sin_tab[(long long)pos * half + k];
        out = sv[c] * co + sv[k] * si;
    } else {
        out = sv[c];
    }
    dst[c] = out;
}

// Carry the wave's trailing rows — the ones that do not complete a block — into
// each sequence's own open-block buffer, so the next wave can read them in
// place. One block per span; `rows · d` is at most `(ratio − 1) · d`.
//
//     [0] dst  float*        the sequence's open-block buffer
//     [1] src  const float*  first trailing row in the wave projection
//     [2] rows
__global__ void carry_kernel(
    const long long* __restrict__ carries,
    int d,
    int n_carry
) {
    const int i = blockIdx.x;
    if (i >= n_carry) return;
    const long long* c = carries + (long long)i * QSA_APPEND_CARRY_WORDS;
    float* dst = (float*)(uintptr_t)c[0];
    const float* src = (const float*)(uintptr_t)c[1];
    const int rows = (int)c[2];
    for (long long e = threadIdx.x; e < (long long)rows * d; e += blockDim.x) {
        dst[e] = src[e];
    }
}

} // namespace qsa_index_append

extern "C" void run_qsa_index_append(
    const long long* jobs,
    const float* k_norm,
    const float* cos_tab,
    const float* sin_tab,
    int32_t d,
    int32_t rope_dim,
    int32_t ratio,
    float eps,
    int32_t n_jobs,
    void* stream
) {
    if (n_jobs <= 0) return;
    const unsigned shmem = (unsigned)d * (unsigned)sizeof(float);
    qsa_index_append::append_kernel<<<(unsigned)n_jobs, (unsigned)d, shmem,
                                     (cudaStream_t)stream>>>(
        jobs, k_norm, cos_tab, sin_tab, d, rope_dim, ratio, eps, n_jobs);
}

extern "C" void run_qsa_index_carry(
    const long long* carries,
    int32_t d,
    int32_t n_carry,
    void* stream
) {
    if (n_carry <= 0) return;
    qsa_index_append::carry_kernel<<<(unsigned)n_carry, 256, 0,
                                    (cudaStream_t)stream>>>(carries, d, n_carry);
}
