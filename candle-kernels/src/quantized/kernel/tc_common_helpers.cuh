// The tile constants, fragment loaders and MMA wrappers shared by the
// tensor-core kernels.
//
// The `tc_common` namespace: the tile constants `BATCH_TILE_I8`, `K_TILE` and
// `KI8_STRIDE`, the `ldmatrix` and `cp.async` loaders, the `mma_m16n8k*`
// instruction wrappers, and `write_output`, which the batch-16 and batch-32
// kernels share.

// Configuration constants shared by s16_tc and sN_tc
constexpr int BATCH_TILE = 16;      // 16 batches per MMA (M dimension) — FP16 path
// INT8 path token tile. N_SUBTILES_I8 = BATCH_TILE_I8/16 m16 token sub-tiles run per
// weight load, the weight k-block staged ONCE in smem and reused across them. At 16
// (single sub-tile) this is the plain per-tile schedule; at 32 it trades doubled
// activation smem for halved weight-L2 re-reads — a net win only when the weight bytes
// dominate (measured: Q8 only; light formats regress on the occupancy hit). Only the int8
// impl/entries use this constant directly; the dense grid (dispatcher.cu) and the MoE
// tiler (cuda.rs grouped_matmul_gemx_q8a128) carry the matching token count as literals —
// keep all three in sync. FP16 stays at BATCH_TILE = 16.
constexpr int BATCH_TILE_I8 = 16;
constexpr int N_SUBTILES_I8 = BATCH_TILE_I8 / 16;  // m16 token sub-tiles per block
constexpr int N_TILE = 32;          // 32 rows per block (4 warps × 8 rows)
constexpr int K_TILE = 128;         // K/128 block size
// INT8 activation smem row stride: K_TILE + 16B pad. The pad shifts each row's bank
// by 4 (stride 144B → (144/4)%32 = 4) so the 8 rows ldmatrix reads per tile land in
// distinct banks — conflict-free, the int8 analogue of the FP path's K_STRIDE pad.
constexpr int KI8_STRIDE = K_TILE + 16;
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

// =============================================================================
// LDMATRIX CONFIGURATION
// =============================================================================
// ldmatrix.sync.aligned.m8n8.x4 is available on sm_75+ (Turing and later)
// It loads 4 × 8×8 matrices (128 bytes total) in one instruction, matching
// the fragment layout expected by mma.m16n8k16.
//
// Benefits:
//   - Single instruction loads all 4 fragments (vs 4 separate loads)
//   - Hardware-optimized shared memory access pattern
//   - Better latency hiding
//
// Requirements:
//   - Shared memory must be 16-byte aligned
//   - Each lane provides address for its portion of the 8×8 tile
//   - Data must be contiguous in 16-byte chunks
// =============================================================================

// Feature detection for ldmatrix — SM80 floor guarantees SM75+ (Turing+)
#define TC_USE_LDMATRIX 1

// -----------------------------------------------------------------------------
// Load FragA using ldmatrix (sm_75+)
// ldmatrix.sync.aligned.m8n8.x4 loads four 8×8 matrices from shared memory
// directly into the register layout expected by mma.m16n8k16.
//
// For mma.m16n8k16 with row-major A[16,16]:
//   - ldmatrix loads 4 × 8×8 tiles arranged as:
//     [0][2]    where each tile is 8×8
//     [1][3]
//   - Each lane in the warp provides one address
//   - The hardware collects 8 addresses per 8×8 tile
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_ldmatrix(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE][K_STRIDE],
    int k_start, int lane)
{
    // Use ldmatrix for 16-bit types (half and bfloat16) — SM80+ always has ldmatrix
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        // ldmatrix addressing for m8n8.x4:
        // - Lanes 0-7: provide addresses for tile 0 (rows 0-7, K 0-7)
        // - Lanes 8-15: provide addresses for tile 1 (rows 8-15, K 0-7)
        // - Lanes 16-23: provide addresses for tile 2 (rows 0-7, K 8-15)
        // - Lanes 24-31: provide addresses for tile 3 (rows 8-15, K 8-15)
        //
        // Each lane provides address to one row of its 8×8 tile
        const int tile_idx = lane / 8;      // Which of the 4 tiles (0-3)
        const int row_in_tile = lane % 8;   // Row within the 8×8 tile (0-7)
        
        // Tile layout for m16n8k16:
        //   Tile 0: rows 0-7,  K k_start..k_start+7
        //   Tile 1: rows 8-15, K k_start..k_start+7
        //   Tile 2: rows 0-7,  K k_start+8..k_start+15
        //   Tile 3: rows 8-15, K k_start+8..k_start+15
        const int m_offset = (tile_idx & 1) * 8;  // 0 for tiles 0,2; 8 for tiles 1,3
        const int k_offset = (tile_idx >> 1) * 8; // 0 for tiles 0,1; 8 for tiles 2,3
        
        // Address of this lane's row in shared memory
        // Each row has 8 half values (16 bytes) contiguous in K dimension
        const uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_A[m_offset + row_in_tile][k_start + k_offset]));
        
        // ldmatrix.sync.aligned.m8n8.x4 loads 4 matrices, output to frag_a[0-3]
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
            : "r"(addr)
        );
        return;
    }
    // Fallback for non-ldmatrix path or non-half types
    // (compile-time eliminated when ldmatrix path is taken)
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col]), float(smem_A[groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col]), float(smem_A[groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col + 8]), float(smem_A[groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col + 8]), float(smem_A[groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load FragA: shared memory → registers for MMA (original non-ldmatrix version)
// For mma.m16n8k16 row-major A with f16/bf16, PTX specifies:
//   groupID           = lane / 4  (range 0-7)
//   threadID_in_group = lane % 4  (range 0-3)
//
//   frag[0] (a0,a1): M=groupID,   K=threadID_in_group*2+{0,1}
//   frag[1] (a2,a3): M=groupID+8, K=threadID_in_group*2+{0,1}
//   frag[2] (a4,a5): M=groupID,   K=threadID_in_group*2+8+{0,1}
//   frag[3] (a6,a7): M=groupID+8, K=threadID_in_group*2+8+{0,1}
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE][K_STRIDE],
    int k_start, int lane)
{
    // Use ldmatrix for 16-bit types (half and bfloat16) — always available on SM80+
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        load_frag_a_ldmatrix(frag_a, smem_A, k_start, lane);
        return;
    }
    
    const int groupID = lane / 4;           // 0-7
    const int threadID_in_group = lane % 4; // 0-3
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col]), float(smem_A[groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col]), float(smem_A[groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col + 8]), float(smem_A[groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col + 8]), float(smem_A[groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load weights: global → shared memory using cp.async (all 32 rows per block)
// Weight layout is K-major: weights[k_blk * nrows + row]
// Uses 16-byte async copies for better memory bandwidth utilization
// All 128 threads cooperatively load all 32 rows for better coalescing
// -----------------------------------------------------------------------------
template <typename block_c_t>
__device__ __forceinline__ void load_weights_async_coop(
    uint8_t* smem_W_flat,  // Flat buffer: 32 rows × block_bytes bytes
    const block_c_t* __restrict__ weights,
    int row0, int k_blk, int nrows, int k_blocks, int tid)
{
    constexpr int block_bytes = sizeof(block_c_t);
    constexpr int int4s_per_block = block_bytes / 16;  // 16-byte chunks per block
    constexpr int total_int4s = N_TILE * int4s_per_block;  // 32 rows × int4s_per_block
    constexpr int loads_per_thread = (total_int4s + NUM_THREADS - 1) / NUM_THREADS;
    
    const block_c_t* row_base = &weights[k_blk * nrows + row0];
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; ++i) {
        const int int4_idx = tid + i * NUM_THREADS;
        
        if (int4_idx < total_int4s) {
            const int row_local = int4_idx / int4s_per_block;
            const int chunk_in_row = int4_idx % int4s_per_block;
            
            uint8_t* dst = smem_W_flat + row_local * block_bytes + chunk_in_row * 16;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(&row_base[row_local]) + chunk_in_row * 16;
            
            // ...existing code...
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                   "l"(src)
            );
        }
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// 16-byte cg cp.async (global → shared). The q8a128 qs is 16-aligned within its
// block, so each thread's chunk is a single wide async copy.
__device__ __forceinline__ void cp_async_cg16(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// 16-byte ca cp.async (global → shared) — caches the line in L1 as well as L2.
// Used for activations: each expert-token-tile's activation is re-read by every
// row-tile block (grid.y), so keeping it L1-resident turns those re-reads into L1
// hits instead of L2 traffic. (The .cg variant bypasses L1 and made the int8 path
// L2-bandwidth-bound while the LDG-based FP path stayed DRAM-bound.) Weights keep
// .cg: they are streamed with little intra-SM reuse and would only thrash L1.
__device__ __forceinline__ void cp_async_ca16(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// 4-byte ca cp.async (global → shared) — for the per-token activation (scale,sum) half2.
__device__ __forceinline__ void cp_async_ca4(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// Legacy sync load for fallback (pre-Ampere)
template <typename block_c_t>
__device__ __forceinline__ void load_weights_warp(
    uint8_t* smem_W,  // 8 × block_bytes bytes per warp
    const block_c_t* __restrict__ weights,
    int row_base, int k_blk, int nrows, int k_blocks, int lane)
{
    constexpr int block_bytes = sizeof(block_c_t);
    constexpr int ints_per_block = block_bytes / sizeof(int);
    constexpr int total_ints = 8 * ints_per_block;
    constexpr int loads_per_thread = (total_ints + WARP_SIZE_TC - 1) / WARP_SIZE_TC;
    
    int* dst = reinterpret_cast<int*>(smem_W);
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; ++i) {
        const int int_idx = lane + i * WARP_SIZE_TC;
        
        if (int_idx < total_ints) {
            const int row_local = int_idx / ints_per_block;
            const int word_in_row = int_idx % ints_per_block;
            const int global_row = row_base + row_local;
            
            int val = 0;
            if (global_row < nrows && row_local < 8) {
                const block_c_t* blk = &weights[k_blk * nrows + global_row];
                val = blk->data[word_in_row];
            }
            dst[int_idx] = val;
        }
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k16
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k16(
    const uint32_t* a, const uint32_t* b, float* c)
{
    if constexpr (std::is_same_v<compute_t, half>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // FP8 uses f16 MMA for k16 - native FP8 MMA is k32 only
        // Activations and weights should be in f16 format for this path
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k32
// -----------------------------------------------------------------------------
// Processes 32 K elements at once.
//
// For FP8 on SM89+: Uses native mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   - a[4]: 4 × uint32, each holding 4 packed fp8 = 16 fp8 total
//   - b[2]: 2 × uint32, each holding 4 packed fp8 = 8 fp8 total
//   Note: a_lo[0..1] and a_hi[0..1] are used as the 4-register A operand
//         b[0..1] are used as the 2-register B operand
//
// For F16/BF16 (and FP8 pre-SM89 fallback): Uses two consecutive m16n8k16 MMAs
//   - a_lo[4]: First k16 slice (4 × half2 = 8 f16)
//   - a_hi[4]: Second k16 slice (4 × half2 = 8 f16)
//   - b[4]: b[0..1] for first k16, b[2..3] for second k16
//
// Parameters:
//   a_lo[4], a_hi[4]: A operands (interpretation depends on compute_t and arch)
//   b[4]:             B operand
//   c[4]:             Accumulator (4 × float, accumulated in place)
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k32(
    const uint32_t* a_lo, const uint32_t* a_hi,
    const uint32_t* b, float* c)
{
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        // Native FP8 m16n8k32 on SM89+ (Ada/Hopper)
        // A operand: 4 registers with 4 fp8 each = 16 fp8
        // B operand: 2 registers with 4 fp8 each = 8 fp8
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_hi[0]), "r"(a_hi[1]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
#else
        // Pre-SM89 fallback: use 2× f16 k16 MMAs (assumes f16 format input)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
#endif
    } else if constexpr (std::is_same_v<compute_t, half>) {
        // First k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        // Second k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        // First k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        // Second k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k64 (INT4 / sub-byte types)
// -----------------------------------------------------------------------------
// Processes 64 K elements at once.
//
// For INT4 on SM80+: Uses native mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
//   - a[4]: 4 × uint32, each holding 8 packed int4 = 32 int4 per thread
//   - b[2]: 2 × uint32, each holding 8 packed int4 = 16 int4 per thread
//   - c[4]: 4 × int32 accumulator
//
// For other types: Falls back to 4× m16n8k16 (f16) or 2× m16n8k32 (fp8)
//   - Requires input data converted to appropriate format
//
// Parameters:
//   a[4]: A operand (4 registers)
//   b[4]: B operand (4 registers for fallback, only b[0..1] used for native int4)
//   c[4]: Accumulator (4 × int32 for int4, 4 × float for fallback)
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k64(
    const uint32_t* a,
    const uint32_t* b,
    int32_t* c)
{
    // Native INT4 m16n8k64 MMA — always available on SM80+
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// Overload with float accumulator for when we need to fall back to f16/fp8 paths
// This handles cases where int4 weights are dequantized to floating point
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k64(
    const uint32_t* a,
    const uint32_t* b,
    float* c)
{
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // 2× m16n8k32 for FP8 (64 elements = 2 × 32)
        mma_m16n8k32<compute_t>(a, a + 2, b, c);        // First k32: a[0..1], a[2..3]
        mma_m16n8k32<compute_t>(a + 4, a + 6, b + 2, c); // Second k32: would need more regs
        // Note: This simplified version assumes caller provides 8 A regs for 2× k32
    } else if constexpr (std::is_same_v<compute_t, half>) {
        // 4× m16n8k16 for F16 (64 elements = 4 × 16)
        // Requires a[0..3] for k0-15, a[4..7] for k16-31, etc.
        // b[0..1] for each k16 slice
        uint32_t a_slice[4];
        
        // k=0..15
        a_slice[0] = a[0]; a_slice[1] = a[1]; a_slice[2] = a[2]; a_slice[3] = a[3];
        mma_m16n8k16<compute_t>(a_slice, b, c);
        
        // k=16..31
        a_slice[0] = a[4]; a_slice[1] = a[5]; a_slice[2] = a[6]; a_slice[3] = a[7];
        mma_m16n8k16<compute_t>(a_slice, b + 2, c);
        
        // k=32..47
        a_slice[0] = a[8]; a_slice[1] = a[9]; a_slice[2] = a[10]; a_slice[3] = a[11];
        mma_m16n8k16<compute_t>(a_slice, b + 4, c);
        
        // k=48..63
        a_slice[0] = a[12]; a_slice[1] = a[13]; a_slice[2] = a[14]; a_slice[3] = a[15];
        mma_m16n8k16<compute_t>(a_slice, b + 6, c);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        // 4× m16n8k16 for BF16
        uint32_t a_slice[4];
        
        a_slice[0] = a[0]; a_slice[1] = a[1]; a_slice[2] = a[2]; a_slice[3] = a[3];
        mma_m16n8k16<compute_t>(a_slice, b, c);
        
        a_slice[0] = a[4]; a_slice[1] = a[5]; a_slice[2] = a[6]; a_slice[3] = a[7];
        mma_m16n8k16<compute_t>(a_slice, b + 2, c);
        
        a_slice[0] = a[8]; a_slice[1] = a[9]; a_slice[2] = a[10]; a_slice[3] = a[11];
        mma_m16n8k16<compute_t>(a_slice, b + 4, c);
        
        a_slice[0] = a[12]; a_slice[1] = a[13]; a_slice[2] = a[14]; a_slice[3] = a[15];
        mma_m16n8k16<compute_t>(a_slice, b + 6, c);
    }
}

// -----------------------------------------------------------------------------
// Dequant wrapper for 4× m16n8k16 MMA: prepares frag_b[8] for half K/128 tile
// -----------------------------------------------------------------------------
// Processes 4 k16 slices (half of K/128). Better occupancy than 8x.
// half_idx: 0 = k16 slices 0-3, 1 = k16 slices 4-7 (template parameter)
// Explicit if constexpr dispatch ensures direct calls without indirection.
template <typename block_c_t, typename compute_t, int half_idx>
__device__ __forceinline__ void dequant_weights_4x_k16(
    const uint8_t* smem_rows, int lane,
    uint32_t frag_b[8])
{
    if constexpr (std::is_same_v<block_c_t, block_c_q8_0>) {
        dequant_q8_0_for_4x_mma_k16_runtime<compute_t, half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_0>) {
        gemx_dequant_traits<block_c_q4_0, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_1>) {
        gemx_dequant_traits<block_c_q4_1, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_0>) {
        gemx_dequant_traits<block_c_q5_0, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_1>) {
        gemx_dequant_traits<block_c_q5_1, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q2_K>) {
        gemx_dequant_traits<block_c_q2_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q3_K>) {
        gemx_dequant_traits<block_c_q3_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_K>) {
        gemx_dequant_traits<block_c_q4_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_K>) {
        gemx_dequant_traits<block_c_q5_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q6_K>) {
        gemx_dequant_traits<block_c_q6_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else {
        // Fallback for any unlisted types
        using Traits = gemx_dequant_traits<block_c_t, compute_t, half>;
        Traits::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    }
}

// -----------------------------------------------------------------------------
// Output write helper with type dispatch
// -----------------------------------------------------------------------------
// Output writes use .cs (streaming) cache hint to avoid polluting L1/L2
// Output data is written once and never re-read, so streaming eviction is optimal
template <typename output_t>
__device__ __forceinline__ void write_output(
    output_t* dst, int dst_stride, int b, int row, float val)
{
    // Use .cs cache hint (streaming, bypass L2) for all output types
    if constexpr (std::is_same_v<output_t, float>) {
        asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "f"(val));
    } else if constexpr (std::is_same_v<output_t, half>) {
        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "h"(__half_as_ushort(__float2half(val))));
    } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "h"(__bfloat16_as_ushort(__float2bfloat16(val))));
    } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
        unsigned int raw_byte = static_cast<unsigned int>(*reinterpret_cast<const unsigned char*>(&from_f32<__nv_fp8_e4m3>(val)));
        asm volatile("st.global.cs.u8 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "r"(raw_byte));
    }
}
