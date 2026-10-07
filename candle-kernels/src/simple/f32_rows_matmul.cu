// =============================================================================
// F32 rows × weightᵀ for a few rows: `out[m, n] = Σ_k x[m, k] · w[n, k]`
// =============================================================================
// The QSA indexer's two projections, at decode and verify width — a handful of
// F32 rows against a `[n, 2560]` F32 weight of 128 or 512 rows. The library
// GEMM runs these on a small-N kernel of 16 or 64 blocks, each walking a long
// stretch of K: 23 µs a call for 1.3–5.2 MB of weights, about a tenth of the
// card's bandwidth, twice per attention layer on every forward.
//
// Here every output column is ONE block of 128 threads, and the block's
// threads cut K between them: thread t takes the float4 at `4·t + 512·j` for
// every j, so a block issues its whole weight row at once and the launch is as
// many blocks as columns — 512 for the query projection, enough to keep the
// row reads in flight across the card. Each thread folds its K slice for every
// input row in the order of j; the warp folds its 32 lanes with an xor tree,
// and the block adds its four warps in warp order. The order is fixed by the
// launch shape alone, so the result is deterministic run to run.
//
// Rows are bounded by F32_ROWS_MATMUL_MAX_ROWS (one accumulator register
// each); `k` must be a multiple of 4 and both operands 16-byte aligned with
// rows `k` floats apart. The host checks all of it.

#include <cuda_runtime.h>

#define F32_ROWS_MATMUL_MAX_ROWS 16
#define F32_ROWS_MATMUL_THREADS 128

template <int ROWS>
__device__ __forceinline__ void f32_rows_matmul_body(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
    int n, int k)
{
    __shared__ float s_part[F32_ROWS_MATMUL_THREADS / 32][ROWS];
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const float4* __restrict__ wrow = reinterpret_cast<const float4*>(w + (long long)col * k);
    const int k4 = k >> 2;

    float acc[ROWS];
    #pragma unroll
    for (int m = 0; m < ROWS; ++m) acc[m] = 0.f;

    #pragma unroll 4
    for (int i = tid; i < k4; i += F32_ROWS_MATMUL_THREADS) {
        const float4 wv = wrow[i];
        #pragma unroll
        for (int m = 0; m < ROWS; ++m) {
            const float4 xv = reinterpret_cast<const float4*>(x + (long long)m * k)[i];
            acc[m] = fmaf(xv.x, wv.x, acc[m]);
            acc[m] = fmaf(xv.y, wv.y, acc[m]);
            acc[m] = fmaf(xv.z, wv.z, acc[m]);
            acc[m] = fmaf(xv.w, wv.w, acc[m]);
        }
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int m = 0; m < ROWS; ++m) {
        float v = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) s_part[warp][m] = v;
    }
    __syncthreads();
    if (tid < ROWS) {
        float v = 0.f;
        #pragma unroll
        for (int wi = 0; wi < F32_ROWS_MATMUL_THREADS / 32; ++wi) v += s_part[wi][tid];
        out[(long long)tid * n + col] = v;
    }
}

// One instantiation per row count, so every accumulator is a register and no
// lane does work for rows the launch does not have.
#define F32_ROWS_MATMUL_KERNEL(R)                                                      \
    extern "C" __global__ void __launch_bounds__(F32_ROWS_MATMUL_THREADS)              \
    f32_rows_matmul_##R(const float* __restrict__ x, const float* __restrict__ w,      \
                        float* __restrict__ out, int n, int k)                         \
    {                                                                                  \
        f32_rows_matmul_body<R>(x, w, out, n, k);                                      \
    }

F32_ROWS_MATMUL_KERNEL(1)
F32_ROWS_MATMUL_KERNEL(2)
F32_ROWS_MATMUL_KERNEL(3)
F32_ROWS_MATMUL_KERNEL(4)
F32_ROWS_MATMUL_KERNEL(5)
F32_ROWS_MATMUL_KERNEL(6)
F32_ROWS_MATMUL_KERNEL(7)
F32_ROWS_MATMUL_KERNEL(8)
F32_ROWS_MATMUL_KERNEL(9)
F32_ROWS_MATMUL_KERNEL(10)
F32_ROWS_MATMUL_KERNEL(11)
F32_ROWS_MATMUL_KERNEL(12)
F32_ROWS_MATMUL_KERNEL(13)
F32_ROWS_MATMUL_KERNEL(14)
F32_ROWS_MATMUL_KERNEL(15)
F32_ROWS_MATMUL_KERNEL(16)

// Returns 0 on a launch, -1 for a row count outside 1..=MAX_ROWS (the host
// routes those to the library GEMM and never asks).
extern "C" int run_f32_rows_matmul(
    const float* x, const float* w, float* out, int rows, int n, int k, void* stream)
{
    if (rows < 1 || rows > F32_ROWS_MATMUL_MAX_ROWS || n <= 0 || k <= 0) return -1;
    const dim3 grid((unsigned)n);
    const dim3 block(F32_ROWS_MATMUL_THREADS);
    cudaStream_t s = (cudaStream_t)stream;
    switch (rows) {
#define F32_ROWS_MATMUL_CASE(R)                                                         \
    case R: f32_rows_matmul_##R<<<grid, block, 0, s>>>(x, w, out, n, k); break;
        F32_ROWS_MATMUL_CASE(1)
        F32_ROWS_MATMUL_CASE(2)
        F32_ROWS_MATMUL_CASE(3)
        F32_ROWS_MATMUL_CASE(4)
        F32_ROWS_MATMUL_CASE(5)
        F32_ROWS_MATMUL_CASE(6)
        F32_ROWS_MATMUL_CASE(7)
        F32_ROWS_MATMUL_CASE(8)
        F32_ROWS_MATMUL_CASE(9)
        F32_ROWS_MATMUL_CASE(10)
        F32_ROWS_MATMUL_CASE(11)
        F32_ROWS_MATMUL_CASE(12)
        F32_ROWS_MATMUL_CASE(13)
        F32_ROWS_MATMUL_CASE(14)
        F32_ROWS_MATMUL_CASE(15)
        F32_ROWS_MATMUL_CASE(16)
#undef F32_ROWS_MATMUL_CASE
    }
    return 0;
}
