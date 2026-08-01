// Shared finalize math for the BDP scan backends.
//
// Both the scalar scan (`bdp_scan.cu`) and the tensor-core finalize
// (`bdp_bmma.cu`) turn a segment's per-case agreement statistics into the
// `z * margin` vote through THIS function, compiled in the same archive group
// with the same flags — so the two backends' float math is one code path and
// their votes bit-match whenever their (exact, integer) inputs match.
#pragma once

// z*margin vote for a segment leader: `top1`/`top2` are the leading and
// runner-up per-case best agreements, `sum`/`sumsq` the agreement sum and
// sum-of-squares over the segment's `n_gal` scanned tokens.
__device__ __forceinline__ float bdp_vote(unsigned int top1, unsigned int top2,
                                          unsigned long long sum,
                                          unsigned long long sumsq, int n_gal) {
    const float n = (float)n_gal;
    const float mean = (float)sum / n;
    float var = (float)sumsq / n - mean * mean;
    if (var < 1e-6f) {
        var = 1e-6f;
    }
    float z = ((float)top1 - mean) / sqrtf(var);
    if (z < 0.0f) {
        z = 0.0f;
    }
    const float margin = (float)(top1 - top2);
    return z * margin;
}
