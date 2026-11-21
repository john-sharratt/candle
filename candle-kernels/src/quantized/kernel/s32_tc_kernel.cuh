// The batch-32 tensor-core kernel and its shared-memory loaders.
//
// The `s32_tc` namespace: its tile constants, `load_activations_32`, the
// `load_frag_a_*` fragment loaders for the m16n8k16, m16n8k32 and m16n8k64
// operand layouts, and `tc32_kernel_impl`, which takes its shared memory as
// parameters, with the standalone `tc32_kernel` wrapper.

// Configuration constants for s32_tc
constexpr int BATCH_TILE_32 = 32;       // 32 batches per tile (2 × 16 MMA)
constexpr int N_TILE = 32;              // 32 rows per block (same as s16_tc)
constexpr int K_TILE = 128;             // K/128 block size
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MMA_ITERS = K_TILE / MMA_K;  // 8 iterations per K/128
constexpr int HALF_MMA_ITERS = MMA_ITERS / 2;  // 4 iterations (for 2x k16 batching)
constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE_TC = 32;
constexpr int NUM_WARPS = 4;
constexpr int K_PAD = 8;
constexpr int K_STRIDE = K_TILE + K_PAD;

// Import shared helpers from tc_common
using tc_common::load_frag_a;
using tc_common::load_weights_async_coop;
using tc_common::load_weights_warp;
using tc_common::mma_m16n8k16;
using tc_common::mma_m16n8k32;
using tc_common::dequant_weights_4x_k16;
using tc_common::cp_async_commit;
using tc_common::cp_async_wait_all;
using tc_common::write_output;

// -----------------------------------------------------------------------------
// Load activations for 32 batches: global → shared memory (cooperative)
// Vectorized paths for half and float inputs for better memory bandwidth
// Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
// Note: Assumes batch_size % 32 == 0 and ncols % K_TILE == 0
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations_32(
    compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    const act_t* __restrict__ vy,
    int batch_start, int k_offset, int y_stride, int tid)
{
    // === VECTORIZED HALF → HALF PATH ===
    // 128 threads × 4 iterations × 8 elements = 4096 elements (32 batches × 128 K)
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
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
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
            half2 h0 = __floats2half2_rn(f4.x, f4.y);
            half2 h1 = __floats2half2_rn(f4.z, f4.w);
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === SCALAR FALLBACK PATH ===
    else {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int idx = tid + i * NUM_THREADS;
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            if (b < BATCH_TILE_32) {
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
}

// Load FragA for s32_tc: load from 32-batch shared memory using ldmatrix
// batch_half: 0 for batches 0-15, 1 for batches 16-31
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_16(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    int k_start, int lane, int batch_half)
{
    const int batch_offset = batch_half * 16;  // 0 or 16
    
    // Use ldmatrix for 16-bit types (half and bfloat16) — always available on SM80+
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        // ldmatrix addressing for m8n8.x4 with batch offset
        const int tile_idx = lane / 8;
        const int row_in_tile = lane % 8;
        const int m_offset = (tile_idx & 1) * 8;
        const int k_offset = (tile_idx >> 1) * 8;
        
        const uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_A[batch_offset + m_offset + row_in_tile][k_start + k_offset]));
        
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
            : "r"(addr)
        );
        return;
    }
    
    // Fallback for non-ldmatrix path or non-16-bit types
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID][k_col]), float(smem_A[batch_offset + groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID + 8][k_col]), float(smem_A[batch_offset + groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID][k_col + 8]), float(smem_A[batch_offset + groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID + 8][k_col + 8]), float(smem_A[batch_offset + groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load FragA for m16n8k32: mirrors mma_m16n8k32 operand layout
// -----------------------------------------------------------------------------
// For FP8 on SM89+: Loads 16 fp8 values packed into a_lo[0..1] and a_hi[0..1]
//   - a_lo[0], a_lo[1], a_hi[0], a_hi[1] form the 4-register A operand
//   - Each register holds 4 packed fp8 values with INTERLEAVED ROW layout
//   - a_lo[2..3] and a_hi[2..3] are unused
//
// CRITICAL: m16n8k32 A operand has INTERLEAVED row layout per PTX ISA:
//   Each 32-bit register packs 4 FP8 bytes as:
//     byte[0] = A[row0, col_base]      (groupID)
//     byte[1] = A[row0, col_base+1]    (groupID)
//     byte[2] = A[row1, col_base]      (groupID+8, interleaved!)
//     byte[3] = A[row1, col_base+1]    (groupID+8, interleaved!)
//
//   Register-to-column mapping (tid = threadID_in_group = lane % 4):
//     a_lo[0]: cols tid*4+0,  tid*4+1   (K offset 0..15 region)
//     a_lo[1]: cols tid*4+2,  tid*4+3   (K offset 0..15 region)
//     a_hi[0]: cols tid*4+16, tid*4+17  (K offset 16..31 region)
//     a_hi[1]: cols tid*4+18, tid*4+19  (K offset 16..31 region)
//
// For F16/BF16 (and FP8 pre-SM89): Loads two k16 slices
//   - a_lo[4]: First k16 slice (k_start to k_start+15)
//   - a_hi[4]: Second k16 slice (k_start+16 to k_start+31)
//
// batch_half: 0 for batches 0-15, 1 for batches 16-31
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_32(
    uint32_t frag_a_lo[4],
    uint32_t frag_a_hi[4],
    const compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    int k_start, int lane, int batch_half)
{
    const int batch_offset = batch_half * 16;
    
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // Native FP8 m16n8k32: pack 16 fp8 into 4 registers with INTERLEAVED layout
        const int groupID = lane / 4;           // 0-7, selects row pair
        const int tid = lane % 4;               // 0-3, selects K-column group
        
        const int row0 = batch_offset + groupID;      // Rows 0-7 (or 16-23)
        const int row1 = batch_offset + groupID + 8;  // Rows 8-15 (or 24-31)
        
        // Column bases for the 4 registers:
        //   a_lo[0]: cols tid*4+0, tid*4+1     (within first K16)
        //   a_lo[1]: cols tid*4+2, tid*4+3     (within first K16)
        //   a_hi[0]: cols tid*4+16, tid*4+17   (within second K16)
        //   a_hi[1]: cols tid*4+18, tid*4+19   (within second K16)
        const int c0  = k_start + tid * 4;        // cols 0,1 for a_lo[0]
        const int c2  = k_start + tid * 4 + 2;    // cols 2,3 for a_lo[1]
        const int c16 = k_start + tid * 4 + 16;   // cols 16,17 for a_hi[0]
        const int c18 = k_start + tid * 4 + 18;   // cols 18,19 for a_hi[1]
        
        // Pack with INTERLEAVED row layout: {row0_col0, row0_col1, row1_col0, row1_col1}
        // This is required by PTX ISA for m16n8k32.e4m3 A operand
        // Optimized: load 2 bytes at a time (uint16) and combine, halving memory ops
        auto pack_interleaved = [&](int col_base) -> uint32_t {
            // Load 2 consecutive FP8 from row0 as uint16 (cols col_base, col_base+1)
            uint32_t lo = *reinterpret_cast<const uint16_t*>(&smem_A[row0][col_base]);
            // Load 2 consecutive FP8 from row1 as uint16 (cols col_base, col_base+1)
            uint32_t hi = *reinterpret_cast<const uint16_t*>(&smem_A[row1][col_base]);
            // Combine: lo in bits[0:15], hi in bits[16:31]
            return lo | (hi << 16);
        };
        
        frag_a_lo[0] = pack_interleaved(c0);   // rows {0,8} × cols {tid*4+0, tid*4+1}
        frag_a_lo[1] = pack_interleaved(c2);   // rows {0,8} × cols {tid*4+2, tid*4+3}
        frag_a_hi[0] = pack_interleaved(c16);  // rows {0,8} × cols {tid*4+16, tid*4+17}
        frag_a_hi[1] = pack_interleaved(c18);  // rows {0,8} × cols {tid*4+18, tid*4+19}
        // frag_a_lo[2..3] and frag_a_hi[2..3] unused in native FP8 path
        return;
    }
#endif
    
    // F16/BF16/pre-SM89 FP8: Load two separate k16 slices
    load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, batch_half);
    load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start + 16, lane, batch_half);
}

// -----------------------------------------------------------------------------
// Load FragA for m16n8k64 (INT4): mirrors mma_m16n8k64_s4/u4 operand layout
// -----------------------------------------------------------------------------
// Loads 64 int4 values (32 per thread distributed across warp) packed into 4 registers.
// Each register holds 8 packed int4 values (4 bits each × 8 = 32 bits).
//
// For INT4 activation data stored as packed nibbles:
//   - smem_A contains int4 values packed 2 per byte (low nibble first)
//   - K_STRIDE should be 64 (or multiple) to align with k64 operations
//
// MMA A operand layout for m16n8k64.s4:
//   - 4 registers per thread
//   - Each register holds 8 int4 values
//   - Thread (lane) distribution follows standard m16n8 pattern
//
// Parameters:
//   frag_a[4]:  Output A fragment (4 × uint32, each with 8 packed int4)
//   smem_A:     Shared memory with packed int4 activations (2 per byte)
//   k_start:    Starting K column (should be multiple of 64)
//   lane:       Thread lane ID (0-31)
//   batch_half: 0 for batches 0-15, 1 for batches 16-31
// -----------------------------------------------------------------------------
template <typename int4_storage_t>  // uint8_t for packed storage, or custom type
__device__ __forceinline__ void load_frag_a_64(
    uint32_t frag_a[4],
    const int4_storage_t smem_A[BATCH_TILE_32][K_STRIDE / 2],  // K/2 because 2 int4 per byte
    int k_start, int lane, int batch_half)
{
    // INT4 MMA fragment load — always available on SM80+
    const int batch_offset = batch_half * 16;
    
    // For m16n8k64 with int4:
    // Each thread loads 32 int4 values = 16 bytes = 4 × uint32
    // Thread mapping follows m16n8 pattern:
    //   groupID (0-7) selects which pair of rows
    //   threadID_in_group (0-3) selects K position within 64-element span
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    
    // Each thread handles elements at k_start + threadID_in_group * 16 (stride of 16 int4 = 8 bytes)
    // Load 8 int4 per register from two rows (groupID and groupID+8)
    // k position: threadID_in_group * 16 gives offset in int4 units
    //             threadID_in_group * 8 gives offset in byte units (2 int4 per byte)
    const int k_byte_offset = (k_start / 2) + threadID_in_group * 8;
    
    // Load 4 bytes (8 int4) per register, from rows based on groupID
    // Register 0: row=groupID,     k=[0..7]   relative to thread's k position
    // Register 1: row=groupID+8,   k=[0..7]
    // Register 2: row=groupID,     k=[32..39] (next 32 k positions)
    // Register 3: row=groupID+8,   k=[32..39]
    frag_a[0] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID][k_byte_offset]);
    frag_a[1] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID + 8][k_byte_offset]);
    frag_a[2] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID][k_byte_offset + 16]);  // +16 bytes = +32 int4
    frag_a[3] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID + 8][k_byte_offset + 16]);
}

// -----------------------------------------------------------------------------
// TC kernel implementation for batch tile = 32 - takes SMEM as parameters
// batch_tile_idx: Decoded batch tile index (hierarchical or simple grid)
// row_tile_idx: Decoded row tile index (-1 = use blockIdx.y for legacy)
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc32_kernel_impl(
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
    
    float frag_c_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float frag_c_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    const int k_blocks = ncols / K_TILE;
    
    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_32<compute_t, act_t>(smem_A, activations, batch0, k_blk * K_TILE, 
                                               y_stride, tid);
        
        // cp.async: SM80+ always has async copy
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0, 
                                            k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        
        // Process K/128 tile - use k32 path for FP8 SM89+, k16 path otherwise
        uint32_t frag_a_lo[4], frag_a_hi[4];
        
        // Non-FP8 or pre-SM89: Use standard k16 path
        // First half: k16 slices 0-3
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_start = i * MMA_K;
                
                load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, 0);
                mma_m16n8k16<compute_t>(frag_a_lo, frag_b + i * 2, frag_c_lo);
                
                load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start, lane, 1);
                mma_m16n8k16<compute_t>(frag_a_hi, frag_b + i * 2, frag_c_hi);
            }
        }
        
        // Second half: k16 slices 4-7
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_start = (4 + i) * MMA_K;
                
                load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, 0);
                mma_m16n8k16<compute_t>(frag_a_lo, frag_b + i * 2, frag_c_lo);
                
                load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start, lane, 1);
                mma_m16n8k16<compute_t>(frag_a_hi, frag_b + i * 2, frag_c_hi);
            }
        }
        
        __syncthreads();
    }
    
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    
    // Batches 0-15
    {
        const int b0 = batch0 + groupID;
        const int b1 = batch0 + groupID + 8;
        
        if (b0 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_lo[0]);
                dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_lo[1]);
            }
        }
        if (b1 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_lo[2]);
                dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_lo[3]);
            }
        }
    }
    
    // Batches 16-31
    {
        const int b0 = batch0 + 16 + groupID;
        const int b1 = batch0 + 16 + groupID + 8;
        
        if (b0 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_hi[0]);
                dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_hi[1]);
            }
        }
        if (b1 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_hi[2]);
                dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_hi[3]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Main TC kernel for batch tile = 32 - standalone wrapper
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc32_kernel(
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
    
    __shared__ compute_t smem_A[BATCH_TILE_32][K_STRIDE];
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];
    
    tc32_kernel_impl<block_c_t, compute_t, act_t, output_t>(
        weights, activations, dst, ncols, nrows, y_stride, dst_stride, batch_size,
        smem_A, smem_W_flat, batch_offset, tile_offset, local_batch_tiles);
}
