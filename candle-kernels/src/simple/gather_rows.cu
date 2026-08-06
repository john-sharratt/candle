// =============================================================================
// gather_rows_bytes — format-agnostic gather of fixed-size rows
// =============================================================================
//
// PURPOSE
// -------
// Copy `n_rows` rows, selected by a device-side index array, from a row-major
// source into a contiguous destination. The source may be device memory OR
// CUDA-registered host memory (`cuMemHostRegister` with `DEVICEMAP`), in which
// case the reads cross PCIe and the hardware handles the transfer.
//
// The motivating use is the token-embedding lookup. Holding `token_embd.weight`
// dequantized in VRAM costs 594 MiB on Qwen3-30B-A3B (151936 x 2048 f16) while a
// forward reads exactly one row per token — the worst VRAM-per-access ratio in
// the model. Leaving the table in the already-registered GGUF mmap and gathering
// only the rows a wave needs trades that VRAM for ~2.25 MiB of PCIe traffic per
// 2048-token forward.
//
// WHY THE GATHER RUNS HERE AND NOT ON THE CPU
// -------------------------------------------
// The token ids are already a device tensor when the embedding is needed. A CPU
// gather would have to read them back first, and a device-to-host read is a
// *synchronisation*, not just a copy: it drains the wave's pipeline at the exact
// point the embedding is required, which is the start of the forward.
//
// WHY BYTES, NOT ELEMENTS
// -----------------------
// This kernel is deliberately ignorant of quantization format. Every GGML type
// stores a row as a whole number of fixed-size blocks:
//
//     row_bytes = (ncols / block_size) * type_size
//
// so "gather row i" is a contiguous byte copy of `row_bytes` from
// `src + idx[i] * row_bytes`, identical for Q4_K, Q6_K, Q8_0, F16 and every
// other format. Dequantization is left to the existing per-format kernels, which
// run afterwards over the gathered (now contiguous) staging buffer.
//
// Fusing dequantization in here instead would mean re-implementing the ~29
// formats `QCudaStorage::dequantize` already dispatches, with no way to keep the
// two numerically identical. Format-blind means a new quantization type needs no
// change here at all.
//
// GRID AND BLOCK CONFIGURATION
// ----------------------------
//   Grid:  (n_rows, 1, 1) — one block per gathered row
//   Block: (256, 1, 1)    — cooperative copy of one row
//   Smem:  none
//
// COALESCING
// ----------
// Threads copy 16 bytes each via `uint4`, so a warp reads 32 x 16 = 512
// contiguous bytes per iteration — fully coalesced *within* a row. Rows
// themselves are scattered, which is the point of a gather; each block therefore
// starts a fresh burst, and for a 1152-byte Q4_K row that burst is smaller than
// the granularity host reads are served at. That is the cost of the trade, and
// it is why the caller overlaps this with compute rather than blocking on it.
//
// The 16-byte path needs both sides 16-byte aligned. `row_bytes` is a multiple
// of the type size (18, 32, 144, 210 bytes for Q4_0/Q8_0/Q4_K/Q6_K), so
// alignment is NOT guaranteed; the kernel checks and falls back to a byte-wise
// copy for the whole row rather than reading across an alignment boundary.
//
// OUT-OF-RANGE INDICES
// --------------------
// An id at or beyond `n_src_rows` writes a zero row instead of reading out of
// bounds. A malformed id is a data problem in one sequence; faulting the kernel
// would take down every other sequence sharing the forward.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void gather_rows_bytes_kernel(
    const uint8_t *__restrict__ src,
    const uint32_t *__restrict__ indices,
    uint8_t *__restrict__ dst,
    int64_t row_bytes,
    int64_t n_src_rows,
    int32_t n_rows) {
  const int32_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }

  uint8_t *d = dst + (int64_t)row * row_bytes;
  const int64_t id = (int64_t)indices[row];

  if (id >= n_src_rows) {
    for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
      d[i] = 0;
    }
    return;
  }

  const uint8_t *s = src + id * row_bytes;
  const bool aligned16 = ((((uintptr_t)s) | ((uintptr_t)d)) & 0xF) == 0;

  if (aligned16) {
    const int64_t n_vec = row_bytes >> 4;
    const uint4 *sv = reinterpret_cast<const uint4 *>(s);
    uint4 *dv = reinterpret_cast<uint4 *>(d);
    for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
      dv[i] = sv[i];
    }
    for (int64_t i = (n_vec << 4) + threadIdx.x; i < row_bytes; i += blockDim.x) {
      d[i] = s[i];
    }
  } else {
    for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
      d[i] = s[i];
    }
  }
}

extern "C" void run_gather_rows_bytes(
    const void *src,
    const uint32_t *indices,
    void *dst,
    int64_t row_bytes,
    int64_t n_src_rows,
    int32_t n_rows,
    void *stream) {
  if (n_rows <= 0 || row_bytes <= 0) {
    return;
  }
  // 256 threads move 4 KiB per pass at the 16-byte width — enough that a typical
  // row (1152 B for Q4_K at 2048 columns) completes in one pass without
  // launching threads that would idle immediately.
  const int threads = 256;
  gather_rows_bytes_kernel<<<n_rows, threads, 0, (cudaStream_t)stream>>>(
      (const uint8_t *)src, indices, (uint8_t *)dst, row_bytes, n_src_rows,
      n_rows);
}
