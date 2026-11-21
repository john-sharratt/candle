// The batch 5 to 15 tensor-core kernel and the TC16 dispatcher.
//
// `sN_tc` is the tensor-core kernel for batch sizes 5 to 15 with runtime-padded
// MMA. The `tc16` namespace is the unified TC kernel dispatcher, and
// `quantized_matmul_tc16_entry` is its TC16 kernel entry point.

// =============================================================================
// SN_TC: TENSOR CORE KERNEL FOR BATCH 5-15 (RUNTIME PADDED MMA)
// =============================================================================
//
// OPERATION: dst[batch, row] = Σ_k Y[batch, k] × W[row, k]
//
// This kernel handles batch sizes 5-15 (never 1-4, never >= 16) by:
//   1. Loading actual batch data into rows 0..(batch_size-1)
//   2. Zero-padding rows batch_size..15 in the MMA activation fragment
//   3. Running the same MMA m16n8k16 as s16_tc
//   4. Only writing valid outputs (batch < batch_size)
//
// OPTIMIZATIONS FOR BATCH 5-15:
//   - Vectorized activation loading with simple `b < batch_size` predicate
//   - Batches 0-4 are ALWAYS valid (no check needed in hot paths)
//   - b0 (0-7): Only groupID 5-7 need batch_size check
//   - b1 (8-15): Always need check since batch_size < 16
//   - Single batch block (no gridDim.y loop)
//
// MMA m16n8k16: C[m,n] += A[m,k] × B[k,n]
//   A = activations [16 batch, 16 K]  - rows >= batch_size are zeroed
//   B = weights [16 K, 8 rows]        - same as s16_tc
//   C = output [16 batch, 8 rows]     - only write valid batches
//
// =============================================================================

namespace sN_tc {

// Import common constants and functions
using namespace tc_common;

// -----------------------------------------------------------------------------
// Load activations with VECTORIZED loads and zero-padding for batch 1-15
//
// Uses int4 (16-byte) vectorized loads with a simple b < batch_n predicate;
// rows batch_n..15 of the tile are written as zeros so the MMA sees padding.
// batch_n is uniform across the block, so the predicate never diverges a warp
// beyond the row boundary it encodes.
// batch_offset is applied to get global indices for memory reads
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations_padded(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int k_offset, int y_stride, int tid, int batch_n, int batch_offset)
{
    // === VECTORIZED HALF → HALF PATH (most common) ===
    // 128 threads × 2 iterations × 8 elements = 2048 elements
    // Each thread loads 16 bytes (8 halfs) per iteration
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;   // local batch 0-15
            const int k = elem_idx % K_TILE;   // K offset
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            int4 data;
            if (b < batch_n) {
                data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            } else {
                data = ZERO4;  // Zero-padding for unused batches
            }
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            int4 data;
            if (b < batch_n) {
                data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            } else {
                data = ZERO4;
            }
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED FLOAT → HALF PATH ===
    // 128 threads × 4 iterations × 4 elements = 2048 elements
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        const half2 ZERO2 = {__float2half(0.0f), __float2half(0.0f)};
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            half2 h0, h1;
            if (b < batch_n) {
                float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                h0 = __floats2half2_rn(f4.x, f4.y);
                h1 = __floats2half2_rn(f4.z, f4.w);
            } else {
                h0 = ZERO2;
                h1 = ZERO2;
            }
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED FLOAT → BF16 PATH ===
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 ZERO2 = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            __nv_bfloat162 h0, h1;
            if (b < batch_n) {
                float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                h0 = __floats2bfloat162_rn(f4.x, f4.y);
                h1 = __floats2bfloat162_rn(f4.z, f4.w);
            } else {
                h0 = ZERO2;
                h1 = ZERO2;
            }
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === FP8 PATH - vectorized uint32 loads ===
    else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) {
        // FP8: 128 threads × 4 iterations × 4 elements = 2048 elements
        // Each thread loads 4 bytes (4 FP8) per iteration
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            uint32_t data;
            if (b < batch_n) {
                data = *reinterpret_cast<const uint32_t*>(&vy[gb * y_stride + gk]);
            } else {
                data = 0;
            }
            *reinterpret_cast<uint32_t*>(&smem_A[b][k]) = data;
        }
    }
    // === FALLBACK: Scalar path for other type combinations ===
    else {
        constexpr int TOTAL_ELEMENTS = BATCH_TILE * K_TILE;
        constexpr int ELEMS_PER_THREAD = (TOTAL_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int idx = tid + i * NUM_THREADS;
            if (idx < TOTAL_ELEMENTS) {
                const int b = idx / K_TILE;    // local batch
                const int k = idx % K_TILE;
                const int gk = k_offset + k;
                const int gb = batch_offset + b;  // global batch index
                
                compute_t val;
                if (b < batch_n) {
                    float f;
                    if constexpr (std::is_same_v<act_t, half>) {
                        f = __half2float(vy[gb * y_stride + gk]);
                    } else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) {
                        f = __bfloat162float(vy[gb * y_stride + gk]);
                    } else {
                        f = static_cast<float>(vy[gb * y_stride + gk]);
                    }
                    if constexpr (std::is_same_v<compute_t, half>) {
                        val = __float2half(f);
                    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                        val = __float2bfloat16(f);
                    }
                } else {
                    if constexpr (std::is_same_v<compute_t, half>) {
                        val = __float2half(0.0f);
                    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                        val = __float2bfloat16(0.0f);
                    }
                }
                smem_A[b][k] = val;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Remainder TC tile for batch 1-15 (the rows left over after the 16-row tiles)
//   1. One batch block covers all batch_n rows; the MMA runs on a zero-padded
//      16-row tile
//   2. Vectorized activation loading with a simple batch predicate
//   3. Output writes predicated on b < batch_n
// batch_n is read at runtime: one body serves every remainder, so the kernel
// set carries one remainder kernel rather than fifteen. The predicates cost a
// compare per activation load and per output store — nothing beside the
// weight stream the tile is bound by.
//
// batch_offset: this segment's batches start at batch_offset (global row index)
// row_tile_idx: row tile from the hierarchical grid decode
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tcN_kernel_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    compute_t smem_A[][K_STRIDE],
    uint8_t* smem_W_flat,
    int batch_n,
    int batch_offset,
    int row_tile_idx)
{
    const int n_block = row_tile_idx;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    
    const int row0 = n_block * N_TILE;
    
    if (row0 >= nrows) return;
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;
    
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int k_blocks = ncols / K_TILE;

    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_padded<compute_t, act_t>(smem_A, activations,
                                                  k_blk * K_TILE, y_stride, tid, batch_n, batch_offset);

        // cp.async: SM80+ always has async copy
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0,
                                            k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Process K/128: first half (k16 slices 0-3)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, i * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }

        }
        // Second half (k16 slices 4-7)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, (4 + i) * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }

        __syncthreads();
    }

    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    const int b0_local = groupID;
    const int b1_local = groupID + 8;
    const int b0 = batch_offset + b0_local;
    const int b1 = batch_offset + b1_local;

    if (b0_local < batch_n) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[0]);
            dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[1]);
        }
    }

    if (b1_local < batch_n) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[2]);
            dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[3]);
        }
    }
}

} // namespace sN_tc


// =============================================================================
// UNIFIED TC KERNEL DISPATCHER
// =============================================================================
// Single kernel entry point that dispatches to existing tc_kernel implementations
// based on blockIdx.y and the segment plan passed as kernel arguments.
//
// This enables a single kernel launch for multi-segment TC work (e.g., tc12+tc5
// for batch 17), keeping weights hot in L2 cache across all batch tiles.
//
// The dispatcher forks as early as possible based on blockIdx.y:
// - Segment 0: blockIdx.y in [0, seg1_tile_start)
// - Segment 1: blockIdx.y in [seg1_tile_start, total_tiles)
//
// Each segment calls the appropriate existing optimized tc_kernel with
// the batch_offset parameter to handle global batch indexing.
// =============================================================================

namespace tc16 {

using namespace tc_common;

// =============================================================================
// UNIFIED TC KERNEL - Handles all TC for batch 1-31 (and grid.y tiling for 32+)
// =============================================================================
// HAS_REMAINDER: whether R = batch_size % 16 is nonzero
//   - false: just tc16 with grid.y tiling
//   - true: tc16 tiles + one tcR remainder tile (R read from batch_size)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
//
// Decode: row_tile = row_group * gridDim.y + blockIdx.y
//         batch_tile = batch_group * gridDim.x + blockIdx.x
//
// SMEM OPTIMIZATION: Declares unified SMEM once and passes to _impl functions
// to avoid NVCC summing SMEM from multiple __shared__ declarations.
// =============================================================================
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, bool HAS_REMAINDER>
__device__ void dispatch_tc16_tcN(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int batch_size, int row_groups)
{
    // Unified SMEM declaration - tc16 size covers both tc16 and tcN paths
    constexpr int K128_BYTES = sizeof(block_c_t);
    __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];  // 16 × 136
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];  // 32 × block_bytes
    
    // =========================================================================
    // HIERARCHICAL GRID DECODE
    // =========================================================================
    // z = row_group + batch_group × num_row_groups (rows inner, batches outer)
    const int row_group = blockIdx.z % row_groups;
    const int batch_group = blockIdx.z / row_groups;
    
    // Decode actual tile indices from hierarchical grid
    const int row_tile_idx = row_group * gridDim.y + blockIdx.y;
    const int batch_tile_idx = batch_group * gridDim.x + blockIdx.x;
    
    // Total batch tiles for bounds checking
    const int tc16_tiles = batch_size / 16;
    const int total_batch_tiles = tc16_tiles + (HAS_REMAINDER ? 1 : 0);
    const int row_tiles = (nrows + N_TILE - 1) / N_TILE;

    // Early exit if out of bounds
    if (row_tile_idx >= row_tiles || batch_tile_idx >= total_batch_tiles) return;

    // Compute tiling from batch_size
    const int remainder_start = tc16_tiles * 16;

    if constexpr (!HAS_REMAINDER) {
        // R=0: All tiles are tc16 (grid.y = tc16_tiles)
        s16_tc::tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
            weights, activations, dst, ncols, nrows, y_stride, dst_stride, 
            batch_size, smem_A, smem_W_flat, 0, batch_tile_idx, tc16_tiles,
            row_tile_idx);
    } else {
        // R=1-15: tc16 tiles + one tcR remainder tile
        if (batch_tile_idx >= tc16_tiles) {
            // Remainder tile: tcN kernel
            sN_tc::tcN_kernel_impl<block_c_t, compute_t, act_t, output_t>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride,
                smem_A, smem_W_flat, batch_size - remainder_start, remainder_start,
                row_tile_idx);
        } else {
            // TC16 tiles
            s16_tc::tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride, 
                batch_size, smem_A, smem_W_flat, 0, batch_tile_idx, tc16_tiles,
                row_tile_idx);
        }
    }
}

} // namespace tc16


// =============================================================================
// quantized_matmul_tc16_entry - TC16 kernel entry point
// =============================================================================
// Handles all TC for batch 1+ with automatic tiling:
//   - batch 1-15: single tcR tile (R = batch_size)
//   - batch 16: single tc16 tile
//   - batch 17-31: tc16 + tcR (R = batch_size % 16)
//   - batch 32+: tc16 with grid.y tiling + optional tcR remainder
//
// HAS_REMAINDER: whether R = batch_size % 16 is nonzero
//   - false: Only tc16 tiles (no remainder)
//   - true: tc16 tiles + one tcR remainder tile
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups
// =============================================================================
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t, bool HAS_REMAINDER>
static __device__ void quantized_matmul_tc16_entry(
    const void * __restrict__ vx,
    const act_t * __restrict__ vy,
    output_t * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_y,
    const int nrows_dst, 
    const int batch_size,
    const int row_groups)
{
    using compute_t = std::conditional_t<
        std::is_same_v<act_t, float>, half,
        std::conditional_t<
            std::is_same_v<act_t, half>, half,
            std::conditional_t<
                std::is_same_v<act_t, __nv_bfloat16>, __nv_bfloat16,
                std::conditional_t<
                    std::is_same_v<act_t, __nv_fp8_e4m3>, __nv_fp8_e4m3,
                    half
                >
            >
        >
    >;

    using block_c_t = block_compact_t<block_q_t>;

    const auto* weights = reinterpret_cast<const block_c_t*>(vx);

    // Hierarchical grid decode computed internally
    tc16::dispatch_tc16_tcN<block_c_t, compute_t, act_t, output_t, HAS_REMAINDER>(
        weights, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups);
}
