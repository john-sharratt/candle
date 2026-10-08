// Greedy pick over many blocks per row.
//
// The batched sampler picks a greedy token with ONE block per row: 1,024
// threads stream a whole 248,320-wide vocabulary row on one SM, ~40 µs, and
// a speculative step makes five such picks one after another (four draft
// positions, then the verify's rows). This spreads each row over
// `ARGMAX_ROW_BLOCKS` blocks that each reduce a slice to one 64-bit key and
// fold it into the row's slot with one `atomicMax`; the last block of a row
// to finish decodes the winner, stores it, and returns the slot and its
// counter to zero for the next launch.
//
// THE KEY IS THE SAMPLER'S OWN ORDER, BIT FOR BIT
// ------------------------------------------------
// The sampler's greedy pick (`branchless_argmax_typed`) has thread `i mod 1024`
// keep the FIRST index of its stride class holding its largest value (a strict
// `>`, so NaN never wins and +0 ties −0), then a tree that keeps the lower
// thread on ties. Its winner is therefore the largest value, ties broken by the
// smallest `i mod 1024`, then by the smallest `i`; and a row whose largest value
// is −∞ (or has no number at all) returns 0. One unsigned key reproduces that:
//
//     key(i) = order(value) << 32 | (KEY_LOW_MAX − ((i mod 1024) << 18 | i >> 10))
//
// where `order` maps a float to an unsigned that sorts as the float does (−0
// folded onto +0, NaN onto 0, below every number). The largest key is the
// sampler's pick; a −∞ winner is reported as 0.

#include <cuda_runtime.h>
#include <stdint.h>

namespace argmax_rows {

constexpr int THREADS = 256;
// Slices per row. 248,320 floats over 32 blocks is ~7,760 a block — a few
// float4 loads a thread, every block resident at once for a five-row verify.
constexpr int ROW_BLOCKS = 32;
// The low word's ceiling: the sampler's stride class takes the top 10 bits
// (`i mod 1024 < 2^10`, shifted by 18), and `i >> 10` the rest (a vocabulary
// below 2^28 keeps `i >> 10 < 2^18`).
constexpr uint32_t KEY_LOW_MAX = 0x0FFFFFFFu;

// The sortable form of a float: +0 and −0 share one value, NaN sorts below −∞.
__device__ __forceinline__ uint32_t order_bits(float v) {
    uint32_t b = __float_as_uint(v);
    if (b == 0x80000000u) b = 0u;                     // −0 ties +0
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) {
        return 0u;                                    // NaN never wins
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// −∞'s sortable value: a winner at or below it means the sampler returns 0.
constexpr uint32_t NEG_INF_ORDER = ~0xFF800000u;

__device__ __forceinline__ unsigned long long key_of(float v, uint32_t i) {
    const uint32_t low = KEY_LOW_MAX - (((i & 1023u) << 18) | (i >> 10));
    return ((unsigned long long)order_bits(v) << 32) | low;
}

__device__ __forceinline__ uint32_t index_of(unsigned long long key) {
    const uint32_t enc = KEY_LOW_MAX - (uint32_t)(key & 0xFFFFFFFFull);
    return ((enc & 0x3FFFFu) << 10) | (enc >> 18);
}

__device__ __forceinline__ unsigned long long warp_max(unsigned long long k) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, k, off);
        k = o > k ? o : k;
    }
    return k;
}

// Grid (ROW_BLOCKS, rows). Block `x` of row `r` reduces `[x·span, (x+1)·span)`
// of the row's first `live` entries.
extern "C" __global__ void __launch_bounds__(THREADS) argmax_rows_f32_kernel(
    const float* __restrict__ logits,  // [rows, row_stride]
    int row_stride,
    int live,
    uint32_t* __restrict__ out,        // [rows]
    unsigned long long* __restrict__ slots,   // [rows], zero between launches
    unsigned int* __restrict__ arrived        // [rows], zero between launches
) {
    const int row = (int)blockIdx.y;
    const float* __restrict__ x = logits + (size_t)row * row_stride;
    const int span = ((live + ROW_BLOCKS - 1) / ROW_BLOCKS + 3) & ~3;
    const int lo = (int)blockIdx.x * span;
    const int hi = min(live, lo + span);

    unsigned long long best = 0ull;
    // `lo` is a multiple of four; the vector walk needs the row base 16-byte
    // aligned too, which a row stride of a multiple of four keeps.
    const bool vec = (row_stride & 3) == 0 && ((uintptr_t)x & 15u) == 0;
    int i = lo + 4 * (int)threadIdx.x;
    if (vec) {
        for (; i + 3 < hi; i += 4 * THREADS) {
            const float4 v = *reinterpret_cast<const float4*>(x + i);
            unsigned long long k = key_of(v.x, (uint32_t)i);
            unsigned long long k1 = key_of(v.y, (uint32_t)i + 1);
            unsigned long long k2 = key_of(v.z, (uint32_t)i + 2);
            unsigned long long k3 = key_of(v.w, (uint32_t)i + 3);
            k = k1 > k ? k1 : k;
            k = k2 > k ? k2 : k;
            k = k3 > k ? k3 : k;
            best = k > best ? k : best;
        }
        // The slice's last partial quad.
        for (int j = i; j < hi && j < i + 4; ++j) {
            const unsigned long long k = key_of(x[j], (uint32_t)j);
            best = k > best ? k : best;
        }
    } else {
        for (int j = lo + (int)threadIdx.x; j < hi; j += THREADS) {
            const unsigned long long k = key_of(x[j], (uint32_t)j);
            best = k > best ? k : best;
        }
    }

    best = warp_max(best);
    __shared__ unsigned long long warp_best[THREADS / 32];
    __shared__ bool last;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) warp_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = 0ull;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; ++w) b = warp_best[w] > b ? warp_best[w] : b;
        atomicMax(&slots[row], b);
        // The slot's update is visible before this block counts in, so the last
        // block to arrive reads every block's key.
        __threadfence();
        last = atomicAdd(&arrived[row], 1u) == (unsigned int)(ROW_BLOCKS - 1);
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        const unsigned long long k = atomicExch(&slots[row], 0ull);
        const uint32_t ord = (uint32_t)(k >> 32);
        out[row] = ord <= NEG_INF_ORDER ? 0u : index_of(k);
        arrived[row] = 0u;
    }
}

} // namespace argmax_rows

// The rows' slots and counters, zero between launches (the launch leaves them
// so). Returns 0 when launched, 1 for a shape it does not run (rows past the
// grid's 65,535 y-extent among them), 2 when the launch itself failed.
extern "C" int run_argmax_rows_f32(
    const float* logits, int rows, int row_stride, int live,
    unsigned int* out, unsigned long long* slots, unsigned int* arrived, void* stream
) {
    if (rows <= 0 || rows > 65535 || live <= 0 || live > row_stride || live >= (1 << 28)) {
        return 1;
    }
    const dim3 grid(argmax_rows::ROW_BLOCKS, (unsigned)rows, 1);
    // A launch error left by an earlier, unrelated launch is not this one's.
    (void)cudaGetLastError();
    argmax_rows::argmax_rows_f32_kernel<<<grid, argmax_rows::THREADS, 0, (cudaStream_t)stream>>>(
        logits, row_stride, live, out, slots, arrived);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
