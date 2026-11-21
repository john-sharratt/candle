// Launch-error bookkeeping for the quantized matmul, the L2 flush utility used
// by the benchmarks, and the launch-configuration constants.
//
// `qmm_last_launch_error` and `qmm_last_launch_site` report the CUDA call
// behind this thread's last QMM_LAUNCH_FAILED. `flush_l2_cache` reads through a
// buffer to evict L2 contents, and the constants size the kernel launches.

// The CUDA call behind this thread's last QMM_LAUNCH_FAILED, and what it
// returned (`matmul_status.cuh`).
thread_local cudaError_t qmm_launch_error = cudaSuccess;
thread_local const char* qmm_launch_site = "";

// The error behind this thread's last QMM_LAUNCH_FAILED, as CUDA names it.
extern "C" const char* qmm_last_launch_error() { return cudaGetErrorName(qmm_launch_error); }

// The call that returned it.
extern "C" const char* qmm_last_launch_site() { return qmm_launch_site; }

// =============================================================================
// L2 CACHE FLUSH UTILITY
// =============================================================================
// Provides a fast way to evict L2 cache contents for realistic benchmarking.
// This simulates real LLaMA inference where different matrices alternate.

// Static buffer for L2 flush (allocated once, reused)
static __device__ char* g_flush_buffer = nullptr;
static size_t g_flush_buffer_size = 0;

// Simple kernel to read through buffer and evict L2 contents
__global__ void flush_l2_kernel(const char* __restrict__ buffer, size_t size, volatile int* dummy) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    int sum = 0;
    for (size_t i = idx; i < size; i += stride) {
        sum += buffer[i];
    }
    // Prevent optimization - write to global memory
    if (idx == 0) {
        *dummy = sum;
    }
}

// Host function to flush L2 cache
// buffer: Pre-allocated device buffer larger than L2 cache
// size: Size of buffer in bytes (should be >= 2x L2 cache size)
// stream: the stream the flush is ordered on, and waited for
//
// Waits for `stream`, never the whole device: a device-wide synchronise is
// refused by a graph capture on any thread, and the refusal invalidates it.
extern "C" void flush_l2_cache(const void* buffer, size_t size, cudaStream_t stream) {
    // Use enough threads to saturate memory bandwidth
    const int threads = 256;
    const int blocks = (size + threads * 16 - 1) / (threads * 16);  // 16 bytes per thread min

    // Allocate a tiny dummy output to prevent optimization
    static int* d_dummy = nullptr;
    if (d_dummy == nullptr) {
        cudaMalloc(&d_dummy, sizeof(int));
    }

    flush_l2_kernel<<<blocks, threads, 0, stream>>>(static_cast<const char*>(buffer), size, d_dummy);
    cudaStreamSynchronize(stream);
}

// =============================================================================
// KERNEL LAUNCH CONFIGURATION
// =============================================================================
// Grid dimensions: (row_blocks, batch_tiles)
//   - row_blocks = ceil(nrows_x / rows_per_cuda_block)
//   - batch_tiles = ceil(batch_size / BATCH_TILE)
// Block dimensions: num_threads = 4 warps × 32 = 128 threads
// 
// rows_per_cuda_block = 16 (constant, matches kernel.cuh)
// BATCH_TILE varies by kernel variant: 1, 8, or 64

constexpr int ROWS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int WARP_SIZE = 32;  // threads per warp
