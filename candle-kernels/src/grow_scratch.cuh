// Grow-only device scratch a launcher keeps across launches.
//
// A launcher that sizes its scratch from the launch shape grows it the first
// time a larger shape arrives. The memory comes from the allocator the host
// installs (`candle_set_kernel_scratch`), which serves it from the device
// reservation's held ground — a block taken here with `cudaMalloc` was memory
// outside the reservation that no accounting on the host could see. Only a
// process that never installs one (a standalone kernel test) allocates with
// `cudaMalloc`.
//
// Growth can happen while the calling thread is recording a graph, where
// anything that waits on the stream is illegal. So growth:
//
// - runs in the thread's relaxed capture mode, which lets an allocation through
//   while a capture is open — neither allocator is stream-ordered on the
//   capturing stream, so nothing is recorded;
// - never gives back the block it replaces. A launch already queued, or
//   recorded into a graph that will be replayed, may still read it, and nothing
//   here can wait for those. Growth at least doubles, so what is kept is
//   smaller than what is live.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

// `bytes` of device memory on device `device`, kept for the life of the
// process, or null when none can be had.
typedef void* (*candle_kernel_scratch_fn)(int device, size_t bytes);

// The installed allocator, or null. Defined in `paged-decode/kernel_scratch.cu`.
extern "C" candle_kernel_scratch_fn g_candle_kernel_scratch;

// Make `*block` hold at least `need` bytes. Returns `cudaSuccess`, or the
// allocation's error with `*block` and `*cap` unchanged.
inline cudaError_t grow_scratch(void** block, size_t* cap, size_t need) {
    if (need <= *cap) return cudaSuccess;
    size_t want = need > 2 * *cap ? need : 2 * *cap;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    void* fresh = nullptr;
    cudaError_t err = cudaSuccess;
    candle_kernel_scratch_fn take = g_candle_kernel_scratch;
    if (take != nullptr) {
        int device = 0;
        err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            fresh = take(device, want);
            if (fresh == nullptr) err = cudaErrorMemoryAllocation;
        }
    } else {
        err = cudaMalloc(&fresh, want);
    }
    cudaThreadExchangeStreamCaptureMode(&mode);
    if (err != cudaSuccess) {
        // Not this launch's error to leave behind for the next status check.
        (void)cudaGetLastError();
        return err;
    }
    *block = fresh;
    *cap = want;
    return cudaSuccess;
}
