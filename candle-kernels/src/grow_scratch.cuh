// Device scratch a launcher keeps between launches, for as long as its epoch.
//
// A launcher that sizes its scratch from the launch shape takes a block the
// first time a larger shape arrives, and keeps it while the host's scratch
// epoch stands. The memory comes from the allocator the host installs
// (`candle_set_kernel_scratch`): inside a forward, a carve from that forward's
// own tier, which goes with the forward; outside one, the reservation's held
// ground. The host moves the epoch when a forward opens and when it ends
// (`candle_set_kernel_scratch_epoch`), so a block carved for one forward is never
// read by a launch after it — the launcher sees the epoch moved, hands its block
// back and takes a fresh one. Only a process that never installs an allocator (a
// standalone kernel test) allocates with `cudaMalloc`.
//
// Taking can happen while the calling thread is recording a graph, where
// anything that waits on the stream is illegal. So taking:
//
// - runs in the thread's relaxed capture mode, which lets an allocation through
//   while a capture is open — neither allocator is stream-ordered on the
//   capturing stream, so nothing is recorded;
// - hands the block it replaces back through the installed `release`, which
//   frees a held block only once every launch that may still read it — queued,
//   or recorded into the segment being captured — has been issued, and leaves a
//   carved one to its forward;
// - asks for what the shape needs plus an eighth, not double. Doubling a
//   353 MiB block asked for 706 MiB.
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

// The installed allocator, its release, and the current epoch. Defined in
// `paged-decode/kernel_scratch.cu`.
extern "C" candle_kernel_scratch_fn g_candle_kernel_scratch;
extern "C" candle_kernel_scratch_release_fn g_candle_kernel_scratch_release;
extern "C" unsigned long long g_candle_kernel_scratch_epoch;

// Make `*block` hold at least `need` bytes in the current epoch. `*epoch` is the
// epoch `*block` was taken in. Returns `cudaSuccess`, or the allocation's error
// with `*block` and `*cap` as they were after any epoch change.
inline cudaError_t grow_scratch(void** block, size_t* cap, unsigned long long* epoch,
                                size_t need) {
    candle_kernel_scratch_fn take = g_candle_kernel_scratch;
    candle_kernel_scratch_release_fn release = g_candle_kernel_scratch_release;
    int device = 0;
    if (take != nullptr && *epoch != g_candle_kernel_scratch_epoch) {
        // A block from another epoch may be a carve of a forward that has
        // ended: it goes back, and this epoch's is taken below.
        if (*block != nullptr && release != nullptr && cudaGetDevice(&device) == cudaSuccess) {
            release(device, *block);
        }
        *block = nullptr;
        *cap = 0;
        *epoch = g_candle_kernel_scratch_epoch;
    }
    if (need <= *cap) return cudaSuccess;
    size_t want = need + need / 8;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    void* fresh = nullptr;
    cudaError_t err = cudaSuccess;
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
        // Whatever it held is not enough for this launch; the caller runs
        // without it (or fails), and the next launch asks again.
        return err;
    }
    void* replaced = *block;
    *block = fresh;
    *cap = want;
    // A `cudaMalloc` block (no allocator installed) is kept: nothing here can
    // order its free behind the launches that may still read it.
    if (replaced != nullptr && take != nullptr && release != nullptr) {
        release(device, replaced);
    }
    return cudaSuccess;
}
