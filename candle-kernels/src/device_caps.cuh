#pragma once

#include <cuda_runtime.h>

// =============================================================================
// SHARED DEVICE CAPABILITIES CACHE
// =============================================================================
//
// Provides a single query-once cache for GPU device properties used by kernels
// to make runtime decisions (tensor core dispatch, shared memory staging, L2
// cache sizing, etc.).
//
// Replaces three independent caching patterns that existed previously:
//   - dispatcher.cu:         file-scope statics (g_cached_*)
//   - paged_decode_kernel:   function-scope statics (cached_*)
//   - paged_prefill_kernel:  no caching (re-queried every call)
//
// Usage:
//   const auto& caps = get_device_caps();
//   bool use_tc = (caps.sm_version >= 800) && head_dim_ok;
//   size_t l2_budget = caps.l2_cache_size * 70 / 100;
//
// Thread safety: The static locals in get_device_caps() are safe for concurrent
// access from the host (C++11 guarantees thread-safe initialization of function-
// scope statics). Multi-device programs that switch devices between launches
// will trigger re-caching when the device_id changes.
//
// Note: This is HOST-SIDE code only. Each translation unit (.cu file) gets its
// own cache instance, which is populated on first use.
// =============================================================================

struct DeviceCaps {
    int sm_version;         // major*100 + minor*10, e.g. 800=Ampere, 860=GA10x, 890=Ada
    int sm_count;           // Number of streaming multiprocessors
    size_t smem_default;    // Default shared memory per block (bytes)
    size_t smem_optin;      // Maximum opt-in shared memory per block (bytes)
    size_t l2_cache_size;   // Total L2 cache size (bytes)
    size_t l2_persist_max;  // Maximum L2 persistence region (bytes, 0 if unsupported)
    bool has_native_fp8;    // SM >= 890: native FP8 MMA instructions
};

/// Query-once device capabilities. Caches on first call per device per TU.
/// Re-caches if the CUDA device changes between calls.
inline const DeviceCaps& get_device_caps() {
    static int cached_device_id = -1;
    static DeviceCaps caps = {};

    int device_id = 0;
    cudaGetDevice(&device_id);

    if (device_id != cached_device_id) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        cached_device_id = device_id;
        caps.sm_version     = props.major * 100 + props.minor * 10;
        caps.sm_count       = props.multiProcessorCount;
        caps.smem_default   = static_cast<size_t>(props.sharedMemPerBlock);
        caps.smem_optin     = static_cast<size_t>(props.sharedMemPerBlockOptin);
        caps.l2_cache_size  = static_cast<size_t>(props.l2CacheSize);
        caps.l2_persist_max = static_cast<size_t>(props.persistingL2CacheMaxSize);
        caps.has_native_fp8 = (caps.sm_version >= 890);
    }

    return caps;
}
