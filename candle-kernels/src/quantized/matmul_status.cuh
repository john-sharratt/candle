#pragma once

// Status codes returned by the quantized-matmul host launchers
// (`run_quantized_matmul`, `run_qkv_segmented_matmul`,
// `run_grouped_quantized_matmul`).
//
// The launchers pick a kernel out of a static table indexed by (quantization
// format, output dtype, tiling mode). A miss used to `return` silently, which
// leaves the destination buffer holding whatever the arena last put there — a
// wrong-numbers bug with no launch failure, no CUDA error, and nothing in the
// profile to look at. Returning a code makes the miss a caller-visible error.
//
// Mirrored by `MatmulStatus` in candle-kernels/src/quantized/api.rs; the two
// must agree.
#define QMM_OK             0  // kernel launched
#define QMM_BAD_QTYPE      1  // quantization format has no matmul kernel
#define QMM_NO_SEGMENTS    2  // caller passed zero weight segments
#define QMM_BAD_YTYPE      3  // activation type outside the dispatch table
#define QMM_NO_KERNEL      4  // no kernel for this (format, output dtype) pair
#define QMM_BAD_OUT_DTYPE  5  // output dtype outside the dispatch table
#define QMM_BAD_SPLIT      6  // split-K depth outside [2, K-tiles], a narrow launch outside its
                              // geometry, or a format that never splits
#define QMM_BAD_TILE_MODE  7  // int8 token-tile width (`n_sub`) with no kernel
#define QMM_LAUNCH_FAILED  8  // the kernel was chosen but its launch returned an error
//
// A launcher that skipped its launch is the silent-loss case a recorded graph
// cannot survive: the missing node would be missing from every replay. So the
// status covers the launch itself, not only the table lookup.
//
// A failed launch also records WHICH CUDA call failed and with what error, on
// this thread, read back by `qmm_last_launch_error` / `_site`
// (`dispatcher.cu`). A bare QMM_LAUNCH_FAILED says nothing about the cause: a
// launch inside a recorded wave can fail on its own refusal or report an
// earlier one, and only the error and the call tell those apart.

#include <cuda_runtime.h>

extern thread_local cudaError_t qmm_launch_error;
extern thread_local const char* qmm_launch_site;

// Record `err` from the call at `site` and return QMM_LAUNCH_FAILED.
static inline int qmm_launch_failed(cudaError_t err, const char* site) {
    qmm_launch_error = err;
    qmm_launch_site = site;
    return QMM_LAUNCH_FAILED;
}
