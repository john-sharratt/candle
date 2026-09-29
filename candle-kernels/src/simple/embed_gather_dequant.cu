// =============================================================================
// embed_gather_dequant — token-embedding lookup from a QUANTIZED device table
// =============================================================================
//
// PURPOSE
// -------
// Gather the rows named by a device index array out of a block-quantized table
// resident in VRAM, dequantize them, and write the result as F32 — in one
// launch, with no staging buffer and no second pass.
//
// The table stays in its on-disk form. A 248,320 x 2,560 Q8_0 embedding is
// 644 MiB resident this way against 1,212 MiB as BF16 or 2,425 MiB as F32,
// and a forward reads one row per token, so the widened form buys nothing but
// VRAM taken from the expert cache.
//
// TWO DESTINATIONS, EITHER OPTIONAL
// ---------------------------------
//   wide   — each row written `replicas` times, back to back: `[n, replicas,
//            ncols]`. A hyper-connection model's residual is `hc` copies of
//            the embedding, and writing them here is what removes the
//            broadcast-then-materialise copy the consumer would otherwise make.
//   narrow — each row once: `[n, ncols]`. The consumer that wants the bare
//            embedding (a draft head's `enorm(embed(t))`) reads this.
//
// Both come from the same dequantized values, so a caller asking for both gets
// two views of one lookup rather than two lookups.
//
// NUMERICS, AND WHY ONLY Q8_0 IS DISPATCHED
// ------------------------------------------
// Values come from `dequantize_block` in `dequant/dequant.cuh`, so this kernel
// defines no arithmetic of its own. That header is NOT the code
// `QTensor::dequantize` runs, though — that is the llama.cpp-derived set in
// `simple/quantized.cu` — and the two are only interchangeable where they
// agree bit for bit. The unit tests check exactly that, per format:
//
//   Q8_0 — `scale * q`, one f32 product per element in both. Identical.
//   Q4_0 — `dequant.cuh` computes `(q - 8) * d`, `quantized.cu` computes
//          `d * q - 8d`; they round differently in the last place.
//   The K-quants and Q5_x — at least one of them faults with a misaligned
//          address when handed blocks packed back to back in a table row
//          (a 22-, 110- or 210-byte block is only 2-byte aligned there).
//
// So a format is dispatched only once the parity test covers it. Q8_0 is the
// one every checkpoint that uses this lookup stores its embedding in.
//
// GRID AND BLOCK CONFIGURATION
// ----------------------------
//   Grid:  (n_rows, 1, 1) — one CUDA block per gathered row
//   Block: (128, 1, 1)    — four warps sharing the row's units
//   Smem:  none
//
// The `dequantize_block` overloads are warp-cooperative: one warp turns one
// UNIT into its elements — two Q8_0 blocks, 64 elements. Each warp walks units
// `warp, warp + 4, ...`, so every lane of a warp takes the same iterations and
// the warp stays converged, which those functions require.
//
// OUT-OF-RANGE INDICES
// --------------------
// An id at or beyond `n_src_rows` writes zero rows to every destination
// rather than reading past the table. The destinations are allocated
// uninitialised by the caller, so this is also what keeps "every byte is
// written" true for a malformed id.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

#include "../blocks.cuh"
#include "../quantized/block_compact.cuh"
#include "../quantized/math.cuh"
#include "../dequant/dequant.cuh"

#define EMBED_GATHER_THREADS 128
#define EMBED_GATHER_WARPS (EMBED_GATHER_THREADS / WARP_SIZE)

template <typename block_t, int BLOCKS_PER_UNIT, int ELEMS_PER_UNIT>
__global__ void __launch_bounds__(EMBED_GATHER_THREADS) embed_gather_dequant_f32_kernel(
    const block_t *__restrict__ table,
    const uint32_t *__restrict__ ids,
    float *__restrict__ wide,
    int32_t replicas,
    float *__restrict__ narrow,
    int64_t ncols,
    int64_t n_src_rows,
    int32_t n_rows) {
  const int32_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int64_t id = (int64_t)ids[row];
  float *wide_row = wide ? wide + (int64_t)row * replicas * ncols : nullptr;
  float *narrow_row = narrow ? narrow + (int64_t)row * ncols : nullptr;

  if (id >= n_src_rows) {
    for (int64_t i = threadIdx.x; i < ncols; i += blockDim.x) {
      for (int32_t r = 0; r < replicas && wide_row; ++r) {
        wide_row[r * ncols + i] = 0.0f;
      }
      if (narrow_row) {
        narrow_row[i] = 0.0f;
      }
    }
    return;
  }

  const int64_t units = ncols / ELEMS_PER_UNIT;
  const int64_t blocks_per_row = units * BLOCKS_PER_UNIT;
  const block_t *src_row = table + id * blocks_per_row;
  const int warp = threadIdx.x / WARP_SIZE;

  for (int64_t u = warp; u < units; u += EMBED_GATHER_WARPS) {
    const block_t *src = src_row + u * BLOCKS_PER_UNIT;
    const int64_t col = u * ELEMS_PER_UNIT;
    for (int32_t r = 0; r < replicas && wide_row; ++r) {
      dequantize_block(src, wide_row + r * ncols + col);
    }
    if (narrow_row) {
      dequantize_block(src, narrow_row + col);
    }
  }
}

template <typename block_t, int BLOCKS_PER_UNIT, int ELEMS_PER_UNIT>
static int32_t launch(
    const void *table,
    const uint32_t *ids,
    float *wide,
    int32_t replicas,
    float *narrow,
    int64_t ncols,
    int64_t n_src_rows,
    int32_t n_rows,
    cudaStream_t stream) {
  if (ncols % ELEMS_PER_UNIT != 0) {
    return -2;
  }
  embed_gather_dequant_f32_kernel<block_t, BLOCKS_PER_UNIT, ELEMS_PER_UNIT>
      <<<n_rows, EMBED_GATHER_THREADS, 0, stream>>>(
          (const block_t *)table, ids, wide, replicas, narrow, ncols, n_src_rows, n_rows);
  return 0;
}

// Returns 0 on launch, -1 for a format this kernel has no unit for, -2 for a
// row width that is not a whole number of units. The launcher reports rather
// than skipping silently because the destinations are uninitialised: an
// unlaunched gather would hand back garbage as the residual stream.
extern "C" int32_t run_embed_gather_dequant_f32(
    const void *table,
    int32_t qtype,
    const uint32_t *ids,
    float *wide,
    int32_t replicas,
    float *narrow,
    int64_t ncols,
    int64_t n_src_rows,
    int32_t n_rows,
    void *stream) {
  if (n_rows <= 0) {
    return 0;
  }
  if (wide == nullptr) {
    replicas = 0;
  }
  cudaStream_t s = (cudaStream_t)stream;
  switch (qtype) {
  case QTYPE_Q8_0: return launch<block_q8_0, 2, 2 * QK8_0>(table, ids, wide, replicas, narrow, ncols, n_src_rows, n_rows, s);
  default: return -1;
  }
}
