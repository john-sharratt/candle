#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

template <typename T>
__device__ __forceinline__ float load_value(const T* ptr);
template <typename T>
__device__ __forceinline__ float load_cached(const T* ptr);
template <typename T>
__device__ __forceinline__ T store_value(float value);

template <>
__device__ __forceinline__ float load_value<float>(const float* ptr) { return *ptr; }
template <>
__device__ __forceinline__ float load_cached<float>(const float* ptr) { return __ldg(ptr); }
template <>
__device__ __forceinline__ float load_value<__half>(const __half* ptr) { return __half2float(*ptr); }
template <>
__device__ __forceinline__ float load_cached<__half>(const __half* ptr) { return __half2float(__ldg(ptr)); }
template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(const __nv_bfloat16* ptr) { return __bfloat162float(*ptr); }
template <>
__device__ __forceinline__ float load_cached<__nv_bfloat16>(const __nv_bfloat16* ptr) { return __bfloat162float(__ldg(ptr)); }

template <>
__device__ __forceinline__ float store_value<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half store_value<__half>(float value) { return __float2half(value); }
template <>
__device__ __forceinline__ __nv_bfloat16 store_value<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <typename T>
__global__ __launch_bounds__(256, 2) void score_weighted_rows_kernel(
    const T* __restrict__ vectors,
    const float* __restrict__ scales,
    const T* __restrict__ activation,
    int vector_count,
    int hidden,
    int64_t vector_stride,
    int64_t scale_stride,
    int64_t activation_stride,
    float threshold,
    uint8_t* __restrict__ active) {
    __shared__ float warp_sums[8];
    for (int row = blockIdx.x; row < vector_count; row += gridDim.x) {
        float dot = 0.0f;
        for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
            dot += load_value(activation + column * activation_stride) *
                load_cached(vectors + row * vector_stride + column);
        }
        for (int width = 16; width > 0; width >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, width);
        }
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) warp_sums[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            dot = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
            for (int width = 16; width > 0; width >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, width);
            }
            if (lane == 0) warp_sums[0] = dot;
        }
        __syncthreads();
        if (threadIdx.x == 0) active[row] = warp_sums[0] >= threshold;
        __syncthreads();
    }
}

template <typename T>
__global__ __launch_bounds__(256, 2) void apply_weighted_rows_kernel(
    const T* __restrict__ vectors,
    const float* __restrict__ scales,
    T* __restrict__ activation,
    const uint8_t* __restrict__ active,
    int vector_count,
    int hidden,
    int64_t vector_stride,
    int64_t scale_stride,
    int64_t activation_stride) {
    for (int column = blockIdx.x * blockDim.x + threadIdx.x;
         column < hidden;
         column += blockDim.x * gridDim.x) {
        float delta = 0.0f;
        for (int row = 0; row < vector_count; ++row) {
            if (active[row]) {
                delta += scales[row * scale_stride] *
                    load_cached(vectors + row * vector_stride + column);
            }
        }
        float value = load_value(activation + column * activation_stride);
        activation[column * activation_stride] = store_value<T>(value + delta);
    }
}

} // namespace

extern "C" int run_weighted_row_accum(
    int dtype,
    const void* vectors,
    const float* scales,
    void* activation,
    int vector_count,
    int hidden,
    int64_t vector_stride,
    int64_t scale_stride,
    int64_t activation_stride,
    float threshold,
    void* stream) {
    if (vector_count < 0 || hidden < 0 || vector_stride <= 0 ||
        scale_stride <= 0 || activation_stride <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (vector_count == 0 || hidden == 0) return static_cast<int>(cudaSuccess);

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    uint8_t* active = nullptr;
    cudaError_t allocation = cudaMallocAsync(
        reinterpret_cast<void**>(&active),
        static_cast<size_t>(vector_count) * sizeof(uint8_t),
        cuda_stream);
    if (allocation != cudaSuccess) return static_cast<int>(allocation);

    const int score_blocks = min(vector_count, 65535);
    const int apply_blocks = min((hidden + 255) / 256, 65535);
    int threads = 32;
    if (hidden > 32) threads = 64;
    if (hidden > 64) threads = 128;
    if (hidden > 128) threads = 256;

    switch (dtype) {
        case 0:
            score_weighted_rows_kernel<float><<<score_blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(vectors), scales,
                static_cast<const float*>(activation), vector_count, hidden,
                vector_stride, scale_stride, activation_stride, threshold, active);
            apply_weighted_rows_kernel<float><<<apply_blocks, 256, 0, cuda_stream>>>(
                static_cast<const float*>(vectors), scales,
                static_cast<float*>(activation), active, vector_count, hidden,
                vector_stride, scale_stride, activation_stride);
            break;
        case 1:
            score_weighted_rows_kernel<__half><<<score_blocks, threads, 0, cuda_stream>>>(
                static_cast<const __half*>(vectors), scales,
                static_cast<const __half*>(activation), vector_count, hidden,
                vector_stride, scale_stride, activation_stride, threshold, active);
            apply_weighted_rows_kernel<__half><<<apply_blocks, 256, 0, cuda_stream>>>(
                static_cast<const __half*>(vectors), scales,
                static_cast<__half*>(activation), active, vector_count, hidden,
                vector_stride, scale_stride, activation_stride);
            break;
        case 2:
            score_weighted_rows_kernel<__nv_bfloat16><<<score_blocks, threads, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(vectors), scales,
                static_cast<const __nv_bfloat16*>(activation), vector_count, hidden,
                vector_stride, scale_stride, activation_stride, threshold, active);
            apply_weighted_rows_kernel<__nv_bfloat16><<<apply_blocks, 256, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(vectors), scales,
                static_cast<__nv_bfloat16*>(activation), active, vector_count, hidden,
                vector_stride, scale_stride, activation_stride);
            break;
        default:
            cudaFreeAsync(active, cuda_stream);
            return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t launch = cudaGetLastError();
    cudaError_t release = cudaFreeAsync(active, cuda_stream);
    return static_cast<int>(launch != cudaSuccess ? launch : release);
}
