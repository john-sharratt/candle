#pragma once
// The weights a split-K or narrow dense launch multiplies its one activation by,
// as SEGMENTS of the output row: segment `s` is a KO weight of `n[s]` rows whose
// row tiles — 32 rows for split-K, 8 for narrow — are the launch's tiles
// `[tile_start[s], tile_start[s+1])` and whose outputs are columns
// `[col_off[s], col_off[s] + n[s])` of the `dst_stride`-wide result. Projections
// that read the same operand — a MoE layer's router, its shared expert's gate_up
// and its gate — run as one launch this way, each output column computed exactly
// as it is alone. One weight is a one-segment table. Passed by value in the
// kernel parameters, so a launch uploads nothing. Shared by the kernels
// (`kernel.cuh`, `dense_narrow.cuh`) and their launchers (`dispatcher.cu`), which
// build and read it.

#define SPLITK_MAX_SEGS 4

struct SplitKSegs {
    const void* w[SPLITK_MAX_SEGS];
    int tile_start[SPLITK_MAX_SEGS + 1];
    int n[SPLITK_MAX_SEGS];
    int col_off[SPLITK_MAX_SEGS];
    int num;
};
