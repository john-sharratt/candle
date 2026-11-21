// FP8 E4M3 conversion helpers, the in-place cast kernels, and the strided,
// small-tensor and dispatching kernels.
//
// `FP8x4` packs and unpacks four FP8 E4M3 values, and the vectorised FP8
// kernels convert them to and from F32 and BF16. The `cast_mut_*` kernels cast
// a buffer in place, using a cooperative grid sync where it is available and a
// single-block phased fallback otherwise. `cast_strided`, `cast_small` and
// `cast_kernel` handle strided input, tiny tensors and the main dispatch.

// =============================================================================
// FP8 E4M3 pack/unpack helpers - reduces code duplication
// =============================================================================

struct FP8x4 {
    // Unpack 4 FP8 values from uint32 to float4
    __device__ __forceinline__ static float4 unpack_to_f32(uint32_t packed) {
        float4 result;
        result.x = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 0) & 0xFF, __NV_E4M3));
        result.y = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
        result.z = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
        result.w = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
        return result;
    }
    
    // Pack 4 floats into uint32 as FP8
    __device__ __forceinline__ static uint32_t pack_from_f32(float4 vals) {
        __nv_fp8_e4m3 v0(vals.x), v1(vals.y), v2(vals.z), v3(vals.w);
        return (uint32_t(v0.__x) << 0) | (uint32_t(v1.__x) << 8) | 
               (uint32_t(v2.__x) << 16) | (uint32_t(v3.__x) << 24);
    }
    
    // Unpack to bfloat162 pair (for bf16 output)
    __device__ __forceinline__ static void unpack_to_bf16x2(uint32_t packed, __nv_bfloat162& out0, __nv_bfloat162& out1) {
        float f0 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 0) & 0xFF, __NV_E4M3));
        float f1 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
        float f2 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
        float f3 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
        out0 = __float22bfloat162_rn(make_float2(f0, f1));
        out1 = __float22bfloat162_rn(make_float2(f2, f3));
    }
    
    // Pack from bfloat162 pair
    __device__ __forceinline__ static uint32_t pack_from_bf16x2(__nv_bfloat162 bf0, __nv_bfloat162 bf1) {
        float2 f0 = __bfloat1622float2(bf0);
        float2 f1 = __bfloat1622float2(bf1);
        __nv_fp8_e4m3 v0(f0.x), v1(f0.y), v2(f1.x), v3(f1.y);
        return (uint32_t(v0.__x) << 0) | (uint32_t(v1.__x) << 8) | 
               (uint32_t(v2.__x) << 16) | (uint32_t(v3.__x) << 24);
    }
};

// =============================================================================
// FP8 E4M3 <-> F32 optimized conversions (4 elements at once via uint32)
// =============================================================================

// FP8 is 1 byte, so we can process 4 at a time using uint32_t as transport
template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f8_f32(
    const size_t numel,
    const __nv_fp8_e4m3 * __restrict__ inp,
    float * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // uint32 needs 4-byte alignment, float4 needs 16-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<16>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[i]).__x, __NV_E4M3));
        }
        return;
    }
    
    // Process 4 FP8 values at a time
    const size_t numel4 = numel / 4;
    const uint32_t* __restrict__ inp4 = reinterpret_cast<const uint32_t*>(inp);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::unpack_to_f32(inp4[idx]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::unpack_to_f32(inp4[i]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[remainder_idx]).__x, __NV_E4M3));
    }
}

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f32_f8(
    const size_t numel,
    const float * __restrict__ inp,
    __nv_fp8_e4m3 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // float4 needs 16-byte alignment, uint32 needs 4-byte alignment
    const bool inp_aligned = is_aligned<16>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __nv_fp8_e4m3(load_readonly(&inp[i]));
        }
        return;
    }
    
    // Process 4 floats at a time, output 4 FP8 packed in uint32
    const size_t numel4 = numel / 4;
    const float4* __restrict__ inp4 = reinterpret_cast<const float4*>(inp);
    uint32_t* __restrict__ out4 = reinterpret_cast<uint32_t*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::pack_from_f32(inp4[idx]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::pack_from_f32(inp4[i]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __nv_fp8_e4m3(load_readonly(&inp[remainder_idx]));
    }
}

// =============================================================================
// BF16 <-> E4M3 optimized conversions (4 elements at once)
// =============================================================================

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_bf16_f8(
    const size_t numel,
    const __nv_bfloat16 * __restrict__ inp,
    __nv_fp8_e4m3 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // bfloat162 needs 4-byte alignment, uint32 needs 4-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __nv_fp8_e4m3(__bfloat162float(load_readonly(&inp[i])));
        }
        return;
    }
    
    // Process 4 bf16 at a time (2x bfloat162), output 4 FP8 packed in uint32
    const size_t numel4 = numel / 4;
    const __nv_bfloat162* __restrict__ inp2 = reinterpret_cast<const __nv_bfloat162*>(inp);
    uint32_t* __restrict__ out4 = reinterpret_cast<uint32_t*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::pack_from_bf16x2(inp2[idx * 2], inp2[idx * 2 + 1]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::pack_from_bf16x2(inp2[i * 2], inp2[i * 2 + 1]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __nv_fp8_e4m3(__bfloat162float(load_readonly(&inp[remainder_idx])));
    }
}

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f8_bf16(
    const size_t numel,
    const __nv_fp8_e4m3 * __restrict__ inp,
    __nv_bfloat16 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // uint32 needs 4-byte alignment, bfloat162 needs 4-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            float f = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[i]).__x, __NV_E4M3));
            out[i] = __float2bfloat16_rn(f);
        }
        return;
    }
    
    // Process 4 FP8 at a time (uint32), output 4 bf16 as 2x bfloat162
    const size_t numel4 = numel / 4;
    const uint32_t* __restrict__ inp4 = reinterpret_cast<const uint32_t*>(inp);
    __nv_bfloat162* __restrict__ out2 = reinterpret_cast<__nv_bfloat162*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            FP8x4::unpack_to_bf16x2(inp4[idx], out2[idx * 2], out2[idx * 2 + 1]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        FP8x4::unpack_to_bf16x2(inp4[i], out2[i * 2], out2[i * 2 + 1]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        float f = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[remainder_idx]).__x, __NV_E4M3));
        out[remainder_idx] = __float2bfloat16_rn(f);
    }
}

// =============================================================================
// IN-PLACE CAST OPERATIONS (cast_mut_*)
// =============================================================================
// These kernels convert dtype in-place within the same buffer, avoiding
// allocation of a new tensor. This is memory-efficient for type conversions.
//
// MEMORY SAFETY CHALLENGE:
// In-place conversion has overlapping source/destination regions. Without
// careful ordering, writes can corrupt unread source data.
//
// SOLUTION: Two execution modes for maximum parallelism with safety:
//
// 1. COOPERATIVE GRID MODE (high parallelism):
//    - Uses cooperative_groups for grid-wide synchronization
//    - All blocks read simultaneously → grid.sync() → all blocks write
//    - Requires cudaLaunchCooperativeKernel on host side
//    - Full GPU utilization with memory safety
//
// 2. SINGLE-BLOCK FALLBACK MODE:
//    - For GPUs/drivers without cooperative launch support
//    - Single block with block-level __syncthreads()
//    - Lower parallelism but universally compatible
//
// Three traversal patterns based on size relationship:
// - Shrinking (sizeof(T) < sizeof(S)): Forward traversal
// - Expanding (sizeof(T) > sizeof(S)): Backward traversal  
// - Same-size (sizeof(T) == sizeof(S)): Any direction with staging
//
// Only contiguous tensors supported (no strided access for in-place).
// =============================================================================

// -----------------------------------------------------------------------------
// COOPERATIVE GRID MODE: Maximum parallelism with grid-wide synchronization
// These kernels use cooperative_groups::this_grid().sync() for safety
// MUST be launched with cudaLaunchCooperativeKernel
// -----------------------------------------------------------------------------

// Shrinking cast with cooperative grid sync - forward traversal
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) <= sizeof(S), "Use shrinking for sizeof(T) <= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Process in grid-stride loop with cooperative sync
    for (size_t base = 0; base < numel; base += grid_stride) {
        const size_t idx = base + global_id;
        
        // Phase 1: All threads across entire grid read their element
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        // Grid-wide barrier: ensure ALL blocks finish reading before ANY write
        grid.sync();
        
        // Phase 2: All threads write their converted value
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        // Grid-wide barrier before next iteration
        grid.sync();
    }
}

// Expanding cast with cooperative grid sync - backward traversal
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) >= sizeof(S), "Use expanding for sizeof(T) >= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Calculate total chunks for backward iteration
    const size_t total_chunks = (numel + grid_stride - 1) / grid_stride;
    
    // Process from end to start
    for (size_t chunk = total_chunks; chunk > 0; chunk--) {
        const size_t base = (chunk - 1) * grid_stride;
        const size_t idx = base + global_id;
        
        // Phase 1: All threads read
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        // Grid-wide barrier
        grid.sync();
        
        // Phase 2: All threads write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        // Sync before next chunk
        grid.sync();
    }
}

// Same-size cast with cooperative grid sync - uses registers (no shared memory needed)
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_same_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) == sizeof(S), "Use same for sizeof(T) == sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel; base += grid_stride) {
        const size_t idx = base + global_id;
        
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        grid.sync();
    }
}

// -----------------------------------------------------------------------------
// COOPERATIVE GRID MODE: Vectorized variants (vec2)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Alignment check - fall back to scalar if not aligned
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        cast_mut_shrinking_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel2; base += grid_stride) {
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
    
    // Handle odd element
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
}

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    if (!is_aligned<Traits::DST_ALIGN>(buf)) {
        cast_mut_expanding_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Handle odd element FIRST (highest index, processed first in backward traversal)
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t total_chunks = (numel2 + grid_stride - 1) / grid_stride;
    
    for (size_t chunk = total_chunks; chunk > 0; chunk--) {
        const size_t base = (chunk - 1) * grid_stride;
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
}

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_same_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        cast_mut_same_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel2; base += grid_stride) {
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
    
    // Handle odd element
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
}

// -----------------------------------------------------------------------------
// SINGLE-BLOCK FALLBACK MODE: For GPUs without cooperative launch support
// These use __syncthreads() within a single block
// MUST be launched with gridDim.x == 1
// -----------------------------------------------------------------------------

// Safe forward traversal for shrinking casts (sizeof(T) < sizeof(S))
// Uses register staging with __syncthreads between read and write phases
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_phased(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) <= sizeof(S), "Use shrinking for sizeof(T) <= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks with explicit synchronization
    // This ensures all reads complete before any writes within each chunk
    for (size_t base = 0; base < numel; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Read into register
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        __syncthreads();  // Ensure ALL reads complete before ANY writes
        
        // Phase 2: Convert and write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Safe backward traversal for expanding casts (sizeof(T) > sizeof(S))
// Processes from end to start with register staging
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_phased(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) >= sizeof(S), "Use expanding for sizeof(T) >= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks from the END, moving backward
    // This ensures we don't overwrite unread source data
    size_t chunks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (size_t chunk = chunks; chunk > 0; chunk--) {
        size_t base = (chunk - 1) * BLOCK_SIZE;
        size_t idx = base + threadIdx.x;
        
        // Phase 1: Read into register
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        __syncthreads();  // Ensure ALL reads complete before ANY writes
        
        // Phase 2: Convert and write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Staged conversion for same-size types (sizeof(T) == sizeof(S))
// Uses shared memory to avoid read-after-write hazards
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_staged(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) == sizeof(S), "Use staged for sizeof(T) == sizeof(S)");
    
    __shared__ S shared_buf[BLOCK_SIZE];
    
    S* __restrict__ data_in = reinterpret_cast<S*>(buf);
    T* __restrict__ data_out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks
    for (size_t base = 0; base < numel; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Load source values into shared memory
        if (idx < numel) {
            shared_buf[threadIdx.x] = data_in[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write back
        if (idx < numel) {
            data_out[idx] = Convert<S, T>::apply(shared_buf[threadIdx.x]);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Vectorized shrinking - vec2 variant with phased execution
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Check alignment for vectorized path
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        // Scalar fallback
        cast_mut_shrinking_phased<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 elements with phased execution
    for (size_t base = 0; base < numel2; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Read vec2 into register
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        __syncthreads();
    }
    
    // Handle odd element (scalar)
    if (numel % 2 && threadIdx.x == 0) {
        S* inp = reinterpret_cast<S*>(buf);
        T* out = reinterpret_cast<T*>(buf);
        out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
    }
}

// -----------------------------------------------------------------------------
// Vectorized expanding - vec2 variant with backward phased execution
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Check alignment for vectorized path
    if (!is_aligned<Traits::DST_ALIGN>(buf)) {
        // Scalar fallback
        cast_mut_expanding_phased<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    // Handle odd element FIRST (it's at the highest address for expanding)
    if (numel % 2 && threadIdx.x == 0) {
        S* inp = reinterpret_cast<S*>(buf);
        T* out = reinterpret_cast<T*>(buf);
        out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
    }
    __syncthreads();
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 elements backward with phased execution
    size_t chunks = (numel2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (size_t chunk = chunks; chunk > 0; chunk--) {
        size_t base = (chunk - 1) * BLOCK_SIZE;
        size_t idx = base + threadIdx.x;
        
        // Phase 1: Read vec2 into register
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Vectorized staged (same-size) - vec2 variant with shared memory
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_staged_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    __shared__ SrcVec shared_buf[BLOCK_SIZE];
    
    // Check alignment
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        // Fall back to scalar staged
        cast_mut_staged<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ data_in = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ data_out = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 chunks with shared memory staging
    for (size_t base = 0; base < numel2; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        if (idx < numel2) {
            shared_buf[threadIdx.x] = data_in[idx];
        }
        __syncthreads();
        
        if (idx < numel2) {
            data_out[idx] = Convert<SrcVec, DstVec>::apply(shared_buf[threadIdx.x]);
        }
        __syncthreads();
    }
    
    // Handle odd element (scalar)
    if (numel % 2 && threadIdx.x == 0) {
        S* scalar_in = reinterpret_cast<S*>(buf);
        T* scalar_out = reinterpret_cast<T*>(buf);
        scalar_out[numel - 1] = Convert<S, T>::apply(scalar_in[numel - 1]);
    }
}

// =============================================================================
// Strided kernel - with shared memory coalescing (Optimization 2)
// =============================================================================

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_strided(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    __shared__ S shared_buf[BLOCK_SIZE];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    for (unsigned int base = blockIdx.x * BLOCK_SIZE; base < numel; base += grid_stride) {
        const unsigned int idx = base + tid;
        
        // Coalesced read from strided input -> shared memory
        if (idx < numel) {
            unsigned int strided_idx = get_strided_index(idx, num_dims, dims, strides);
            shared_buf[tid] = load_readonly(&inp[strided_idx]);
        }
        __syncthreads();
        
        // Convert and coalesced write to output
        if (idx < numel) {
            out[idx] = Convert<S, T>::apply(shared_buf[tid]);
        }
        __syncthreads();
    }
}

// =============================================================================
// Small tensor kernel - single warp, minimal overhead (Optimization 5)
// Uses warp-uniform execution for best efficiency
// =============================================================================

template<typename S, typename T>
__device__ void cast_small(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    const unsigned int tid = threadIdx.x;
    
    // Warp-uniform branch: all threads in warp take same path
    if (info == nullptr) {
        // Contiguous path - simple indexed access
        if (tid < numel) {
            out[tid] = Convert<S, T>::apply(load_readonly(&inp[tid]));
        }
    } else {
        const size_t *dims = info;
        const size_t *strides = info + num_dims;
        
        if (is_contiguous(num_dims, dims, strides)) {
            if (tid < numel) {
                out[tid] = Convert<S, T>::apply(load_readonly(&inp[tid]));
            }
        } else {
            if (tid < numel) {
                unsigned int strided_idx = get_strided_index(tid, num_dims, dims, strides);
                out[tid] = Convert<S, T>::apply(load_readonly(&inp[strided_idx]));
            }
        }
    }
}

// =============================================================================
// Main dispatch kernel (Optimization 4, 8 - branch elimination via dispatch)
// =============================================================================

template<typename S, typename T, int BLOCK_SIZE = 256>
__device__ void cast_kernel(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    // Small tensor path: single block handles everything
    if (numel <= 32 && gridDim.x == 1) {
        cast_small<S, T>(numel, num_dims, info, inp, out);
        return;
    }
    
    // Check contiguity - info being nullptr means contiguous
    if (info == nullptr) {
        cast_contiguous<S, T, BLOCK_SIZE>(numel, inp, out);
    } else {
        const size_t *dims = info;
        const size_t *strides = info + num_dims;
        if (is_contiguous(num_dims, dims, strides)) {
            cast_contiguous<S, T, BLOCK_SIZE>(numel, inp, out);
        } else {
            cast_strided<S, T, BLOCK_SIZE>(numel, num_dims, dims, strides, inp, out);
        }
    }
}
