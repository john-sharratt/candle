#pragma once

// Status codes returned by the quantized-matmul host launchers
// (`run_quantized_matmul`, `run_qkv_segmented_matmul`).
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
