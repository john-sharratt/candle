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
// - hands the block it replaces back through the installed `release`, which
//   frees it only once every launch that may still read it — queued, or
//   recorded into the segment being captured — has been issued. Keeping every
//   replaced block instead stranded each step of the growth for the life of
//   the process: seven blocks, 362 MiB, filled the held ground on Flash-Next
//   and refused everything after them into the CUDA pool.
// - asks for what the shape needs plus an eighth, not double. Doubling a
//   353 MiB block asked for 706 MiB, more than the whole held ground.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

// `bytes` of device memory on device `device`, held until given back through
// the matching release, or null when none can be had.
typedef void* (*candle_kernel_scratch_fn)(int device, size_t bytes);

// Give back a block `candle_kernel_scratch_fn` returned on device `device`.
// Launches already queued or recorded may still read it; the host frees it
// behind them.
typedef void (*candle_kernel_scratch_release_fn)(int device, void* block);

// The installed allocator and its release, or null. Defined in
// `paged-decode/kernel_scratch.cu`.
extern "C" candle_kernel_scratch_fn g_candle_kernel_scratch;
extern "C" candle_kernel_scratch_release_fn g_candle_kernel_scratch_release;

// Make `*block` hold at least `need` bytes. Returns `cudaSuccess`, or the
// allocation's error with `*block` and `*cap` unchanged.
inline cudaError_t grow_scratch(void** block, size_t* cap, size_t need) {
    if (need <= *cap) return cudaSuccess;
    size_t want = need + need / 8;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    void* fresh = nullptr;
    int device = 0;
    cudaError_t err = cudaSuccess;
    candle_kernel_scratch_fn take = g_candle_kernel_scratch;
    if (take != nullptr) {
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
    void* replaced = *block;
    *block = fresh;
    *cap = want;
    // A `cudaMalloc` block (no allocator installed) is kept: nothing here can
    // order its free behind the launches that may still read it.
    candle_kernel_scratch_release_fn release = g_candle_kernel_scratch_release;
    if (replaced != nullptr && take != nullptr && release != nullptr) {
        release(device, replaced);
    }
    return cudaSuccess;
}
