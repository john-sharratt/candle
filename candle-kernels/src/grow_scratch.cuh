// Grow-only device scratch a launcher keeps across launches.
//
// A launcher that sizes its scratch from the launch shape grows it the first
// time a larger shape arrives. That can happen while the calling thread is
// recording a graph, where an ordinary `cudaMalloc` is refused and anything
// that waits on the stream is illegal. So growth:
//
// - allocates in the thread's relaxed capture mode, which lets the allocation
//   through while a capture is open — `cudaMalloc` is not stream-ordered and
//   records nothing, so it is safe to make there;
// - never frees the block it replaces. A launch already queued, or recorded
//   into a graph that will be replayed, may still read it, and nothing here can
//   wait for those. Growth at least doubles, so what is kept is smaller than
//   what is live.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

// Make `*block` hold at least `need` bytes. Returns `cudaSuccess`, or the
// allocation's error with `*block` and `*cap` unchanged.
inline cudaError_t grow_scratch(void** block, size_t* cap, size_t need) {
    if (need <= *cap) return cudaSuccess;
    size_t want = need > 2 * *cap ? need : 2 * *cap;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    void* fresh = nullptr;
    cudaError_t err = cudaMalloc(&fresh, want);
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
