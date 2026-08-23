// NVTX3 range shim.
//
// The pipeline spans (`candle-transformers/src/models/profile.rs`) annotate the
// trace so `nsys stats --report nvtx_kern_sum` attributes every kernel to the
// span that launched it, instead of leaving it to neighbour-histogram guesswork
// over a per-launch CSV.
//
// Why a shim rather than a Rust NVTX binding: NVTX3 is HEADER-ONLY. The old
// `nvToolsExt.lib` import library was dropped in the CUDA 12 series (12.4, 12.9
// and 13.3 on this machine all ship `include/nvtx3/` and no lib), so a binding
// that emits `link-lib=nvToolsExt` — cudarc's `nvtx` feature does exactly that —
// fails at link with LNK1181 on any modern toolkit. The header carries the whole
// implementation, so compiling it here into the kernels archive is both the
// supported mechanism and the one that needs nothing to link against.
//
// Cost when not profiling: the NVTX3 header resolves these to an injection
// pointer that stays null unless a tool (nsys, Nsight Compute) attached at
// process start, so each call is a null check and a return.

#include <nvtx3/nvToolsExt.h>

extern "C" {

// Open a range and return its id. Pairs with `candle_nvtx_range_end`.
//
// This is the START/END form, NOT push/pop: ranges may overlap and may close out
// of order, which is what lets a span end where the work ends rather than at the
// end of its enclosing scope.
unsigned long long candle_nvtx_range_start(const char *message) {
    return (unsigned long long)nvtxRangeStartA(message);
}

// Close the range opened by `candle_nvtx_range_start`.
void candle_nvtx_range_end(unsigned long long id) {
    nvtxRangeEnd((nvtxRangeId_t)id);
}

} // extern "C"
