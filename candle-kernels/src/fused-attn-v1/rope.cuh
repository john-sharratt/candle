#pragma once
// =============================================================================
// rope.cuh — RoPE dispatch wrapper around v2's existing helpers.
//
// Reuses apply_rope_rotary_f32 and apply_rope_interleaved_f32 from
// paged_decode_kernel.cuh, dispatching on the model descriptor's compile-time
// ROPE_STYLE / ROPE_INTERLEAVED flags.
// =============================================================================

#include "../paged-decode/paged_decode_kernel.cuh"
#include "model_descriptor.cuh"

namespace fused_attn {

template<int HEAD_DIM, int VEC, RopeStyle STYLE, bool INTERLEAVED>
__device__ __forceinline__ void apply_rope_dispatch(
    float*       regs,
    int          lane,
    int          pos,
    const float* rope_cs
) {
    if constexpr (STYLE == RopeStyle::Partial) {
        // Partial RoPE: rotate only first HEAD_DIM/2 dims. v1's supported model
        // set uses Full RoPE only; this branch is dead code but kept for future
        // models that need partial rotary.
        if (lane < 16) {
            apply_rope_rotary_f32<VEC, HEAD_DIM / 2>(regs, lane, pos, rope_cs);
        }
    } else if constexpr (INTERLEAVED) {
        apply_rope_interleaved_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs);
    } else {
        apply_rope_rotary_f32<VEC, HEAD_DIM>(regs, lane, pos, rope_cs);
    }
}

} // namespace fused_attn
