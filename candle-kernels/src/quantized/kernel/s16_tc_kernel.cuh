// The tensor-core kernel for batches of 16 and above.
//
// The `s16_tc` namespace, the tensor-core kernel for batch sizes 16 and above,
// built on the `tc_common` loaders and MMA wrappers.

namespace s16_tc {

// Import common constants
using namespace tc_common;

// -----------------------------------------------------------------------------
// Load activations: global → shared memory (cooperative across all threads)
// Vectorized paths for half and float inputs for better memory bandwidth
// Note: Assumes batch_size % BATCH_TILE == 0 and ncols % K_TILE == 0
//       (true for all common LLM dimensions) - allows branchless loading
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int batch_start, int k_offset, int y_stride, int tid)
{
    // === VECTORIZED HALF → HALF PATH ===
    // 128 threads × 2 iterations × 8 elements = 2048 elements
    // Each thread loads 16 consecutive bytes (8 halfs) per iteration
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED FLOAT → HALF PATH ===
    // 128 threads × 4 iterations × 4 elements = 2048 elements
    // Each thread loads float4 (16 bytes) and converts to 4 halfs
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
            
            // Convert and store as 2 half2
            half2 h0 = __floats2half2_rn(f4.x, f4.y);
            half2 h1 = __floats2half2_rn(f4.z, f4.w);
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    // Same as half path - 16 bytes = 8 bf16 elements
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === SCALAR FALLBACK PATH (FP8, mixed types) ===
    else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * NUM_THREADS;
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float val;
            if constexpr (std::is_same_v<act_t, float>) {
                val = vy[gb * y_stride + gk];
            } else if constexpr (std::is_same_v<act_t, half>) {
                val = __half2float(vy[gb * y_stride + gk]);
            } else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) {
                val = __bfloat162float(vy[gb * y_stride + gk]);
            } else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) {
                val = to_f32(vy[gb * y_stride + gk]);
            }
            
            if constexpr (std::is_same_v<compute_t, half>) {
                smem_A[b][k] = __float2half(val);
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                smem_A[b][k] = __float2bfloat16(val);
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                smem_A[b][k] = __nv_fp8_e4m3(__float2half(val));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// TC kernel implementation: takes SMEM as parameters for unified dispatch
// This avoids SMEM summing when called from dispatch functions
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc16_kernel_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    compute_t smem_A[][K_STRIDE],
    uint8_t* smem_W_flat,
    int batch_offset = 0,
    int batch_tile_idx_in = 0,   // Actual batch tile index (not blockIdx.x)
    int total_batch_tiles = 0,   // Total batch tiles (for bounds check)
    int row_tile_idx = -1)       // -1 = use blockIdx.y (legacy), >= 0 = hierarchical decode
{
    // Hierarchical grid decode: use passed row_tile_idx if >= 0, else blockIdx.y
    const int n_block = (row_tile_idx >= 0) ? row_tile_idx : (int)blockIdx.y;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    
    const int row0 = n_block * N_TILE;
    // For hierarchical grid: batch_offset already contains the base batch
    const int batch0 = batch_offset;
    
    // Bounds check: row must be valid and batch must be in range
    if (row0 >= nrows || batch0 >= batch_size) return;
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;
    
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    const int k_blocks = ncols / K_TILE;
    
    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations<compute_t, act_t>(smem_A, activations, batch0, k_blk * K_TILE, 
                                            y_stride, tid);
        
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
    const int b0 = batch0 + groupID;
    const int b1 = batch0 + groupID + 8;
    
    if (b0 < batch_size) {
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
    if (b1 < batch_size) {
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

// -----------------------------------------------------------------------------
// Main TC kernel: standalone wrapper that declares SMEM and calls impl
// Uses cp.async for overlapped memory transfers with cooperative loading
//
// batch_offset: Global batch index where this segment starts (for output indexing)
// tile_offset:  blockIdx.y value where this segment starts (for local tile calculation)
// local_batch_tiles: Number of batch tiles this segment should process (for bounds)
//                    Default 0 means use gridDim.y (standalone mode)
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc16_kernel(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    int batch_offset = 0,
    int tile_offset = 0,
    int local_batch_tiles = 0)
{
    assert(nrows % N_TILE == 0 && "TC kernel requires nrows to be a multiple of 32");
    assert(ncols % K_TILE == 0 && "TC kernel requires ncols to be a multiple of 128");
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];
    
    tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
        weights, activations, dst, ncols, nrows, y_stride, dst_stride, batch_size,
        smem_A, smem_W_flat, batch_offset, tile_offset, local_batch_tiles);
}

} // namespace s16_tc
