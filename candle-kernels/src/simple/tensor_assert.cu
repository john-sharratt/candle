// =============================================================================
// tensor_assert — in-place finiteness/range statistics over a tensor
// =============================================================================
//
// One launch reads a tensor where it already lives and folds four statistics
// into a preallocated device slot: NaN count, Inf count, and the min/max of the
// FINITE values. Nothing is allocated, nothing is copied, nothing is read back.
// The slot is drained later, at a synchronisation the caller already performs.
//
// That is the whole point of the design. An instrument that allocates or
// synchronises inside the region it observes is not an instrument, it is a
// change to the program: a probe built from `sum_all().to_scalar()` costs one
// device sync per checkpoint and a full-tensor transient per call, and both
// perturb exactly the launch ordering and arena layout a concurrency fault
// depends on. Here the only footprint is the launch itself.
//
// Min/max cover the finite values only. NaN and Inf are counted separately, and
// admitting them into the range would make it useless — one Inf and every
// subsequent report reads `max = inf`, which says nothing about the data.
//
// ## Ordering across sites
//
// Because nothing is read back, "which checkpoint went bad first" cannot be
// answered by observation order on the host. Instead the first thread to move a
// slot from clean to bad claims a ticket from a single global counter, so the
// drain can sort the bad slots by the order in which they ACTUALLY went bad —
// which is the diagnostic question. A slot that never went bad has seq 0.
//
// ## Monotonic float keys
//
// `atomicMin`/`atomicMax` exist for u32 but not for f32 on every architecture,
// and a CAS loop over floats would serialise. IEEE-754 floats admit an
// order-preserving map into u32:
//
//   key(f)  = bits ^ (sign ? 0xFFFFFFFF : 0x80000000)
//   bits(k) = (k & 0x80000000) ? (k ^ 0x80000000) : ~k
//
// For non-negative f the top bit is set and magnitude order is preserved; for
// negative f the complement both inverts the magnitude order and clears the top
// bit, placing every negative below every positive. So plain integer
// atomicMin/atomicMax on the key are exactly float min/max. The inverse runs on
// the host at drain time.
//
//   Grid:  grid-stride, capped
//   Block: ASSERT_BLOCK threads
//
// Empty tensors are a no-op (the launcher returns before launching).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define ASSERT_BLOCK 256
#define ASSERT_MAX_BLOCKS 4096
#define ASSERT_MAX_DIMS 8

// Mirrored by `AssertSlot` in candle-core `tensor_assert::slots`. The layouts
// must agree exactly — the host reads this struct straight out of the drained
// device buffer.
struct AssertSlot {
    unsigned int nan;      // NaN elements seen
    unsigned int inf;      // +/-Inf elements seen
    unsigned int min_key;  // monotonic key of the smallest finite value
    unsigned int max_key;  // monotonic key of the largest finite value
    unsigned int seq;      // 1-based order stamp of the first bad observation; 0 = never bad
    unsigned int elems;    // elements examined
    unsigned int pad0;
    unsigned int pad1;
};

// Strides and dims travel by value in the launch parameters rather than through
// a device array, so a strided assert allocates nothing either.
struct AssertLayout {
    int64_t dims[ASSERT_MAX_DIMS];
    int64_t strides[ASSERT_MAX_DIMS];
    int num_dims;  // 0 => contiguous, index directly
};

__device__ __forceinline__ unsigned int assert_f32_key(float v) {
    unsigned int b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ---------------------------------------------------------------------------
// Per-element load: every supported dtype widened to float for the statistics.
//
// Integer types cannot be NaN or Inf, so their `finite` is unconditionally
// true; they still carry min/max, which is what makes an assert useful on an
// index or id tensor. F64 checks finiteness at full width and only then
// narrows for the key, so a finite 1e300 is not miscounted as Inf by the
// narrowing itself; its recorded max saturates, which the host reports.
// ---------------------------------------------------------------------------

template <typename T>
struct AssertLoad;

template <>
struct AssertLoad<float> {
    __device__ __forceinline__ static float value(const float* p, int64_t i) { return p[i]; }
};
template <>
struct AssertLoad<__half> {
    __device__ __forceinline__ static float value(const __half* p, int64_t i) {
        return __half2float(p[i]);
    }
};
template <>
struct AssertLoad<__nv_bfloat16> {
    __device__ __forceinline__ static float value(const __nv_bfloat16* p, int64_t i) {
        return __bfloat162float(p[i]);
    }
};
template <>
struct AssertLoad<double> {
    __device__ __forceinline__ static float value(const double* p, int64_t i) {
        const double v = p[i];
        if (isnan(v)) return __int_as_float(0x7fc00000);  // quiet NaN
        if (isinf(v)) return v > 0 ? INFINITY : -INFINITY;
        // Finite but out of float range saturates to +/-FLT_MAX rather than to
        // an infinity, so a finite f64 is never counted as Inf.
        if (v > 3.402823466e+38) return 3.402823466e+38f;
        if (v < -3.402823466e+38) return -3.402823466e+38f;
        return (float)v;
    }
};
template <>
struct AssertLoad<uint8_t> {
    __device__ __forceinline__ static float value(const uint8_t* p, int64_t i) { return (float)p[i]; }
};
template <>
struct AssertLoad<uint32_t> {
    __device__ __forceinline__ static float value(const uint32_t* p, int64_t i) { return (float)p[i]; }
};
template <>
struct AssertLoad<int64_t> {
    __device__ __forceinline__ static float value(const int64_t* p, int64_t i) { return (float)p[i]; }
};
template <>
struct AssertLoad<int32_t> {
    __device__ __forceinline__ static float value(const int32_t* p, int64_t i) { return (float)p[i]; }
};
template <>
struct AssertLoad<__nv_fp8_e4m3> {
    __device__ __forceinline__ static float value(const __nv_fp8_e4m3* p, int64_t i) {
        // Same decode as `Convert<__nv_fp8_e4m3, T>` in simple/cast.cu.
        return __half2float(__nv_cvt_fp8_to_halfraw(p[i].__x, __NV_E4M3));
    }
};

// Offset of logical element `i` under a (possibly strided) layout.
__device__ __forceinline__ int64_t assert_offset(const AssertLayout& lay, int64_t i) {
    if (lay.num_dims == 0) return i;
    int64_t off = 0;
    int64_t rem = i;
#pragma unroll
    for (int d = ASSERT_MAX_DIMS - 1; d >= 0; --d) {
        if (d >= lay.num_dims) continue;
        const int64_t dim = lay.dims[d];
        const int64_t idx = rem % dim;
        rem /= dim;
        off += idx * lay.strides[d];
    }
    return off;
}

template <typename T>
__global__ void tensor_assert_kernel(
    const T* __restrict__ src,
    int64_t n,
    AssertLayout lay,
    AssertSlot* __restrict__ slot,
    unsigned int* __restrict__ seq_counter
) {
    unsigned int l_nan = 0;
    unsigned int l_inf = 0;
    unsigned int l_min = 0xFFFFFFFFu;
    unsigned int l_max = 0u;
    unsigned int l_cnt = 0;

    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = AssertLoad<T>::value(src, assert_offset(lay, i));
        l_cnt++;
        if (isnan(v)) {
            l_nan++;
        } else if (isinf(v)) {
            l_inf++;
        } else {
            const unsigned int k = assert_f32_key(v);
            l_min = min(l_min, k);
            l_max = max(l_max, k);
        }
    }

    // Warp-reduce, then ONE atomic per warp rather than one per element.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        l_nan += __shfl_down_sync(0xffffffffu, l_nan, off);
        l_inf += __shfl_down_sync(0xffffffffu, l_inf, off);
        l_cnt += __shfl_down_sync(0xffffffffu, l_cnt, off);
        l_min = min(l_min, __shfl_down_sync(0xffffffffu, l_min, off));
        l_max = max(l_max, __shfl_down_sync(0xffffffffu, l_max, off));
    }

    if ((threadIdx.x & 31) != 0) return;

    if (l_cnt) atomicAdd(&slot->elems, l_cnt);
    if (l_min != 0xFFFFFFFFu) atomicMin(&slot->min_key, l_min);
    if (l_max != 0u) atomicMax(&slot->max_key, l_max);
    if (l_nan) atomicAdd(&slot->nan, l_nan);
    if (l_inf) atomicAdd(&slot->inf, l_inf);

    // Claim the order ticket exactly once per slot, on the first bad warp.
    // The CAS installs a sentinel so no second warp can also claim; the winner
    // then overwrites it with the real ticket. A reader between the two sees
    // the sentinel, which is why the drain runs after a synchronisation.
    if (l_nan || l_inf) {
        if (atomicCAS(&slot->seq, 0u, 0xFFFFFFFFu) == 0u) {
            atomicExch(&slot->seq, atomicAdd(seq_counter, 1u) + 1u);
        }
    }
}

// Reset every slot to the identity for the reductions above: counters zero,
// min_key at +infinity of the key order and max_key at -infinity, so the first
// real value replaces both.
__global__ void tensor_assert_reset_kernel(AssertSlot* __restrict__ slots, int n_slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slots) return;
    AssertSlot s;
    s.nan = 0;
    s.inf = 0;
    s.min_key = 0xFFFFFFFFu;
    s.max_key = 0u;
    s.seq = 0;
    s.elems = 0;
    s.pad0 = 0;
    s.pad1 = 0;
    slots[i] = s;
}

// dtype codes — mirrored by `AssertDType` in candle-core `tensor_assert`.
#define ASSERT_DT_F32 0
#define ASSERT_DT_F16 1
#define ASSERT_DT_BF16 2
#define ASSERT_DT_F64 3
#define ASSERT_DT_U8 4
#define ASSERT_DT_U32 5
#define ASSERT_DT_I64 6
#define ASSERT_DT_F8E4M3 7
// Not a candle `DType`: kernel workspaces (tile tables, permutations, expert
// ids) are raw i32 device slices, and an out-of-range index there is exactly
// the kind of corruption that shows up downstream as an implausible magnitude
// rather than as a NaN. Reading their min/max is how that gets caught.
#define ASSERT_DT_I32 8

extern "C" void run_tensor_assert(
    const void* src,
    int32_t dtype,
    int64_t elem_count,
    int32_t num_dims,
    const int64_t* dims,
    const int64_t* strides,
    void* slot,
    void* seq_counter,
    void* stream
) {
    if (elem_count <= 0 || src == nullptr || slot == nullptr) return;
    if (num_dims < 0 || num_dims > ASSERT_MAX_DIMS) return;

    AssertLayout lay;
    lay.num_dims = num_dims;
    for (int d = 0; d < ASSERT_MAX_DIMS; ++d) {
        lay.dims[d] = (d < num_dims && dims) ? dims[d] : 1;
        lay.strides[d] = (d < num_dims && strides) ? strides[d] : 0;
    }

    int64_t want = (elem_count + ASSERT_BLOCK - 1) / ASSERT_BLOCK;
    if (want > ASSERT_MAX_BLOCKS) want = ASSERT_MAX_BLOCKS;
    if (want < 1) want = 1;
    const dim3 grid((unsigned int)want, 1, 1), block(ASSERT_BLOCK, 1, 1);
    cudaStream_t s = (cudaStream_t)stream;
    AssertSlot* sl = (AssertSlot*)slot;
    unsigned int* sq = (unsigned int*)seq_counter;

    switch (dtype) {
        case ASSERT_DT_F32:
            tensor_assert_kernel<float><<<grid, block, 0, s>>>((const float*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_F16:
            tensor_assert_kernel<__half><<<grid, block, 0, s>>>((const __half*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_BF16:
            tensor_assert_kernel<__nv_bfloat16><<<grid, block, 0, s>>>((const __nv_bfloat16*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_F64:
            tensor_assert_kernel<double><<<grid, block, 0, s>>>((const double*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_U8:
            tensor_assert_kernel<uint8_t><<<grid, block, 0, s>>>((const uint8_t*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_U32:
            tensor_assert_kernel<uint32_t><<<grid, block, 0, s>>>((const uint32_t*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_I64:
            tensor_assert_kernel<int64_t><<<grid, block, 0, s>>>((const int64_t*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_F8E4M3:
            tensor_assert_kernel<__nv_fp8_e4m3><<<grid, block, 0, s>>>((const __nv_fp8_e4m3*)src, elem_count, lay, sl, sq);
            break;
        case ASSERT_DT_I32:
            tensor_assert_kernel<int32_t><<<grid, block, 0, s>>>((const int32_t*)src, elem_count, lay, sl, sq);
            break;
        default:
            break;
    }
}

extern "C" void run_tensor_assert_reset(void* slots, int32_t n_slots, void* stream) {
    if (slots == nullptr || n_slots <= 0) return;
    const int threads = 128;
    const dim3 grid((unsigned int)((n_slots + threads - 1) / threads), 1, 1), block(threads, 1, 1);
    tensor_assert_reset_kernel<<<grid, block, 0, (cudaStream_t)stream>>>((AssertSlot*)slots, n_slots);
}
