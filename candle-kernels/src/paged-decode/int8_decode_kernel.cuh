#pragma once
// =============================================================================
// int8_decode_kernel.cuh — v2-API-compatible decode-attention kernel (Track A).
//
// Drop-in for v2's paged_decode_attn, computing the QK^T as an INT8 m16n8k32 MMA
// per palette (lane-collective INT8 dot fallback for head dims whose palette
// isn't 32-wide) and the PV in INT8, with a per-32-token tile-batched softmax
// and the §1A V read-through. Same slot-header / paged-arena interface as v2.
//
// # When this kernel faults and the address does not say where
//
// The kernels are built *without* `--generate-line-info`, so a fault here
// surfaces as a bare device address rather than a file and line. Rebuild with
// the `kernel-lineinfo` feature for the debugging session:
//
//     cargo test -p candle-transformers --features cuda,kernel-lineinfo ...
//
// then drop it again. It is off by default because it is two thirds of the
// compiled archive — 175 KB of `.text` against 592 KB of debug sections in a
// measured cubin — and every CUDA test binary links a copy. See
// `candle-kernels/build_utils.rs`.
// =============================================================================

#include <assert.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "../fast_exp.cuh"
#include "../arena_table.cuh"
#include "../grow_scratch.cuh"
#include "../simple/warp_reduce.cuh"
#include "../convert/convert_all.cuh"
#include "../blocks.cuh"
#include "slot_types.cuh"
#include "pal_iter.cuh"
// Shared decode helpers (vec2_traits, load_vec2, cp_async_*, RoPE, scatter,
// write-len commit) — formerly inline in the V2 paged_decode_kernel.cuh.
#include "decode_helpers.cuh"
#include "int8_decode_emit.cuh"
#include "../mma/mma_wrappers.cuh"
// QSA block-sparse selection: `sel.entries == nullptr` for every model whose
// attention layers read the whole causal prefix, which is all of them but
// Qwen3.8-Flash-Next's 12 full-attention layers.
#include "../qsa_select.cuh"
// Wide-head (HEAD_DIM 256) decode on the INT8 tensor cores: one slice per
// tile, the group's query heads as the MMA's M rows.
#include "int8_decode_tile_kernel.cuh"

namespace fused_attn {

#include "int8_decode_kernel/decode_attn.cuh"

#include "int8_decode_kernel/warp_stripe.cuh"

#include "int8_decode_kernel/batched_m_bmma.cuh"

#include "int8_decode_kernel/split_kv_combine.cuh"

#include "int8_decode_kernel/launch.cuh"

} // namespace fused_attn
