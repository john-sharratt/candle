// =============================================================================
// gather_r16_kv_f16 — batch gather of R16 K/Q and float-F16 V chunks
// =============================================================================
//
// PURPOSE
// -------
// The slow path (dump_sequence_r16_kv_chunks) extracts KV data for a probe
// window by issuing one synchronous memcpy_dtov per (head, palette) sub-band
// per block.  For a 2-block window with 8 KV heads and N_PALETTE=4 that is
// 2 × 8 × 4 = 64 sub-bands × 3 tensors (K, Q, V) = 192 synchronous stalls,
// each requiring a CPU/GPU round-trip.  At ~10 µs per stall this serialises
// ~2 ms of otherwise-idle time on every newline token.
//
// This kernel collapses the entire gather into:
//   1. One tiny HtoD upload of resolved chunk-pointer arrays (8 bytes × n_warps)
//   2. One async kernel launch (no CPU wait)
//   3. One synchronous DtoH copy of the combined K/Q/V output tensor
//
// GRID AND BLOCK CONFIGURATION
// ----------------------------
// Each CUDA block handles exactly one (chunk_block_idx, kv_head_idx, palette_idx)
// triple — called a "warp" in the application-level sense throughout this file.
// With CHUNK_SIZE = 32, exactly 32 CUDA threads are launched per block, one per
// token.  That means every CUDA block is also exactly one hardware warp, so
// there is zero intra-block synchronisation overhead and no __syncthreads needed.
//
//   Grid:   (n_warps, 1, 1)  where n_warps = n_r16_blocks × n_kv_head × N_PALETTE
//   Block:  (CHUNK_SIZE, 1, 1) = 32 threads
//   Smem:   none
//
// INPUT DATA FORMATS
// ------------------
//
// k_ptrs[warp_id]: byte address of the R16 K chunk for this triple.
//   R16 chunk layout (block_r16 from blocks.cuh):
//     sub_head_dim groups of 128 bytes, one group per dim-slice d:
//       group[d]:
//         bytes [  0 ..  63]: K  F16[CHUNK_SIZE]   — K[token=0..31] at dim d
//         bytes [ 64 .. 127]: Q  F16[CHUNK_SIZE]   — Q[token=0..31] at dim d (captured live)
//     Read address for K[token][d] = k_ptr + d*128 + token*2
//     Read address for Q[token][d] = k_ptr + d*128 + 64 + token*2
//   For fixed d, consecutive tokens (= consecutive threads) are 2 bytes apart
//   → COALESCED read (32 × 2 = 64 bytes = one L1 cache line per iteration).
//
// v_ptrs[warp_id]: byte address of the float-F16 V chunk for this triple.
//   V chunk layout: token-major F16 [CHUNK_SIZE, sub_head_dim]:
//     Read address for V[token][d] = v_ptr + (token * sub_head_dim + d) * 2
//   For fixed d, consecutive threads are sub_head_dim × 2 bytes apart
//   → NON-COALESCED (stride = 64 bytes for sub_head_dim=32).
//   Acceptable because each chunk is ≤ 2 KB (32 tokens × 32 dims × 2 bytes),
//   which fits in L1 cache after the first pass (d=0).  Subsequent iterations
//   are L1 hits with ~4 cycle latency — no shared memory needed.
//
// OUTPUT FORMAT
// -------------
// Single combined buffer out_kqv sized 3 × n_warps × CHUNK_SIZE × sub_head_dim.
//
//   stride = n_warps × CHUNK_SIZE × sub_head_dim
//
//   out_kqv[0        .. stride)   = K values
//   out_kqv[stride   .. 2×stride) = Q values (live-captured queries)
//   out_kqv[2×stride .. 3×stride) = V values
//
// Within each section, the layout is D-MAJOR per warp:
//
//   index(warp_id, d, token) = warp_id × CHUNK_SIZE × sub_head_dim
//                            + d × CHUNK_SIZE
//                            + token
//
// D-major is chosen for COALESCED WRITES: for fixed d in the inner loop,
// all 32 threads write to consecutive 2-byte addresses (stride = 2 bytes),
// which the GPU merges into a single 64-byte L2 store transaction.
//
// The consumer (backing.rs, gather_r16_kv_probe) transposes d-major →
// token-major during the existing F16→F32 conversion pass, so no extra
// allocation is needed on the CPU side.
//
// WHY NOT TOKEN-MAJOR OUTPUT?
// ---------------------------
// Token-major output index = warp_id × CS × SHD + token × SHD + d.
// For fixed d and 32 concurrent threads, the write stride is SHD × 2 bytes.
// At SHD = 32 (head_dim=128, N_PALETTE=4): stride = 64 bytes per thread.
// The GPU issues 32 separate 64-byte write requests instead of one 64-byte
// coalesced request: 32× more write transactions, wasting 97% of bus bandwidth.
//
// =============================================================================

#include "blocks.cuh"

__global__ void gather_r16_kv_f16_kernel(
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    __half* __restrict__ out_kqv,
    int sub_head_dim
) {
    const int warp_id = blockIdx.x;
    const int token   = threadIdx.x;  // 0 .. CHUNK_SIZE-1

    const int64_t k_base = k_ptrs[warp_id];
    const int64_t v_base = v_ptrs[warp_id];

    // Section offsets within out_kqv.
    const int section   = (int)gridDim.x * CHUNK_SIZE * sub_head_dim;
    const int out_base  = warp_id * CHUNK_SIZE * sub_head_dim;

    for (int d = 0; d < sub_head_dim; d++) {
        // K + Q: read from dim-major R16 block (block_r16 layout).
        // Group d spans bytes [d*128, d*128+128).  K occupies [0,64), Q [64,128).
        // token*2: stride 2 bytes between consecutive CUDA threads → coalesced.
        const __half k_val = *(const __half*)(k_base + (int64_t)d * 128 + token * 2);
        const __half q_val = *(const __half*)(k_base + (int64_t)d * 128 + 64 + token * 2);

        // V: token-major F16.  Non-coalesced (stride = sub_head_dim*2 between threads)
        // but chunk fits in L1 after d=0 pass so later iterations are cheap L1 hits.
        const __half v_val = *(const __half*)(v_base + ((int64_t)(token * sub_head_dim + d)) * 2);

        // D-major write: all 32 threads target addresses [out_base + d*CHUNK_SIZE + 0..31].
        // Stride = 2 bytes → fully coalesced single 64-byte store per iteration.
        const int idx = out_base + d * CHUNK_SIZE + token;
        out_kqv[idx]               = k_val;
        out_kqv[section     + idx] = q_val;
        out_kqv[2 * section + idx] = v_val;
    }
}

extern "C" void run_gather_r16_kv_f16(
    const int64_t* k_ptrs,
    const int64_t* v_ptrs,
    void*          out_kqv,
    int            n_warps,
    int            sub_head_dim,
    cudaStream_t   stream
) {
    if (n_warps <= 0 || sub_head_dim <= 0) return;
    gather_r16_kv_f16_kernel<<<n_warps, CHUNK_SIZE, 0, stream>>>(
        k_ptrs, v_ptrs, (__half*)out_kqv, sub_head_dim
    );
}
