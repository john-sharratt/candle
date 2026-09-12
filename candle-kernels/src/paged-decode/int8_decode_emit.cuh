// ============================================================================
// Split-KV decode: the merged row's emit, shared by the standalone combine
// kernel and the tile kernel's in-kernel merge tree.
//
// One row is one (slot, query head): HEAD_DIM threads, thread d owning dim d,
// each holding the row's normalised attention value `val`. The emit is either
// the plain O store, or — B2, `q8_out` non-null — the fused q8a128 context:
// the row is HEAD_DIM/128 whole 128-tiles, thread d's tile is
// row·(HEAD_DIM/128) + d/128; amax and Σx per tile reduce by a warp butterfly
// then a pairwise combine of the tile's four warp results, and thread d writes
// its quant byte, the tile's first thread the per-128 {scale, sum}. The value
// is rounded through O first to mirror the unfused FP store + re-quant; the
// optional output gate multiplies in as sigmoid(g) in F32 with a second
// O-rounding, so the quantised value matches gating an O-stored context.
//
// Both callers run this with HEAD_DIM threads and the same thread→dim map, so
// the reduction trees — and therefore the bytes — are identical whichever
// kernel merged the splits.
//
// These bytes are MODE-AGNOSTIC: the q8a1024 flat-grouped layout is
// byte-identical for the matmul's V (mode-1, Bm=16) and X (mode-2, Bm=32)
// variants — the mode only changes how the matmul tiles the SAME bytes. The
// choice rides in the `Q8a128Operand.ytype` the Rust side derives from the
// token count when it wraps `q8_out`; nothing here decides it.
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "../blocks.cuh"
#include "../convert/convert_all.cuh"

namespace fused_attn {

/// Emit one merged row. `sh_amax` / `sh_sum` are HEAD_DIM/32 floats of shared
/// memory the caller provides for the per-tile warp combine; the q8 path
/// synchronises the block, so every one of the row's HEAD_DIM threads must
/// call this together.
template <typename O, int HEAD_DIM>
__device__ __forceinline__ void int8_decode_emit_row(
    float val,
    int64_t row,
    int d,
    O* __restrict__ out,
    uint8_t* __restrict__ q8_out,
    const O* __restrict__ gate,
    int64_t gate_slot_stride,
    int row_heads,
    float* sh_amax,
    float* sh_sum
) {
    if constexpr (HEAD_DIM % 128 == 0) {
        if (q8_out != nullptr) {
            float vr = to_f32<O>(from_f32<O>(val));
            if (gate != nullptr) {
                // Sigmoid rounded through O before the multiply — the exact
                // arithmetic of the unfused chain (sigmoid(gate) stored in O,
                // then an O-precision elementwise multiply on the O context).
                // Computed in DOUBLE (this archive's `--use_fast_math` maps
                // float expf to the approximate intrinsic, whose few-ulp wobble
                // flips the O rounding exactly when the sigmoid lands on an
                // O-dtype boundary; double exp's ≤1-ulp error survives the
                // narrow to F32 only within ~2⁻²⁹ of an F32 boundary, which the
                // O rounding then absorbs — the CPU byte oracle mirrors the
                // same f64 chain).
                const int64_t g_slot = (row / row_heads) * gate_slot_stride;
                const int64_t g_off = (row % row_heads) * HEAD_DIM + d;
                const float g = to_f32<O>(gate[g_slot + g_off]);
                const float sig =
                    to_f32<O>(from_f32<O>((float)(1.0 / (1.0 + exp((double)(-g))))));
                vr = to_f32<O>(from_f32<O>(vr * sig));
            }
            float amax = fabsf(vr);
            float s = vr;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
                s += __shfl_xor_sync(0xffffffff, s, off, 32);
            }
            const int warp = d >> 5;
            const int lane = d & 31;
            if (lane == 0) { sh_amax[warp] = amax; sh_sum[warp] = s; }
            __syncthreads();
            // Combine the tile's four warp results pairwise — (w0,w2)+(w1,w3) is
            // the butterfly's summation order, keeping the bytes identical to the
            // former warp-shuffle combine at HEAD_DIM 128.
            const int tb = (d >> 7) << 2;
            const float tile_amax = fmaxf(fmaxf(sh_amax[tb], sh_amax[tb + 2]),
                                          fmaxf(sh_amax[tb + 1], sh_amax[tb + 3]));
            const float tile_sum = (sh_sum[tb] + sh_sum[tb + 2]) + (sh_sum[tb + 1] + sh_sum[tb + 3]);
            // IEEE divisions (`__fdiv_rn`) despite the archive's fast math: the
            // quant boundary sits at half a code, and the approximate division's
            // ±2-ulp wobble is exactly what flips a byte there — the CPU byte
            // oracle mirrors these two ops bit-for-bit.
            const float id = (tile_amax != 0.f) ? __fdiv_rn(127.f, tile_amax) : 0.f;
            uint8_t* obytes = q8_out;
            const int64_t flat = row * (HEAD_DIM / 128) + (d >> 7);
            obytes[q8a1024_qs_off(flat) + (d & 127)] = (int8_t)__float2int_rn(vr * id);
            if ((d & 127) == 0) {
                half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(flat));
                // Σx normalised by amax — see blocks.cuh. `id` carries the
                // amax==0 guard, and the same IEEE-division reasoning applies:
                // the normalisation reuses `id` rather than dividing again.
                ds[0] = make_half2(__float2half_rn(__fdiv_rn(tile_amax, 127.f)),
                                   __float2half_rn(tile_sum * id * (1.f / 127.f)));
            }
            // The shared pair is free for the caller's next row once every
            // thread has read it.
            __syncthreads();
            return;
        }
    }
    out[row * HEAD_DIM + d] = from_f32<O>(val);
}

} // namespace fused_attn
