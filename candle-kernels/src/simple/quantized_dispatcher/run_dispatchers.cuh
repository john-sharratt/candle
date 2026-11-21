// The host dispatchers that choose and launch the quantize, dequantize and
// mul-mat kernels.
//
// The `run_*` host functions for the q8_1, q8a128 and KO quantize and
// dequantize kernels, the unified `run_quantize_block`, `run_dequantize_block`
// and `run_dequantize_mxfp4`, the mul-mat-vec dispatchers
// `run_dequantize_mul_mat_vec` and `run_mul_mat_vec_q8_1`, and `run_mul_mat`,
// which selects the tensor-core or MMQ path.

// =============================================================================
// DISPATCHER: quantize_q8_1 (legacy interface)
// =============================================================================

extern "C" void run_quantize_q8_1(
    const float* src,
    void* dst,
    int32_t elem_count,
    int32_t ky,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int kx = elem_count;
    int kx_padded = pad(kx, MATRIX_ROW_PADDING);
    int num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    
    dim3 grid(num_blocks, ky, 1);
    dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);

    quantize_q8_1<<<grid, block, 0, s>>>(src, dst, kx, kx_padded);
}


// Q8A128 activation block. The quantize_q8a128_kernel / dequantize_q8a128_kernel
// templates are defined in quantize/quantize_q8a128.cuh and
// dequant/dequant_q8a128.cuh (included at the top of this file); the extern "C"
// dispatch wrappers + the unified run_*_block cases live here.

// dtype: 0=F16, 1=BF16, 2=F32 (YType ordering).
extern "C" void run_quantize_q8a128(
    const void* act, void* out, int32_t rows, int32_t cols, int32_t dtype,
    int32_t sum_norm, void* stream)
{
    // One warp per 128-tile; 8 warps/block, grid-strided over all tiles.
    const int total_tiles = (int)(((int64_t)rows * cols) / 128);
    const int block_dim = 256;
    const int grid = (total_tiles + 7) / 8;
    dim3 grid_dim(grid > 0 ? grid : 1, 1, 1);
    dim3 block(block_dim, 1, 1);
    cudaStream_t s = (cudaStream_t)stream;
    switch (dtype) {
        case 0: quantize_q8a128_kernel<half><<<grid_dim, block, 0, s>>>(
            (const half*)act, (block_q8a128*)out, rows, cols, sum_norm); break;
        case 1: quantize_q8a128_kernel<__nv_bfloat16><<<grid_dim, block, 0, s>>>(
            (const __nv_bfloat16*)act, (block_q8a128*)out, rows, cols, sum_norm); break;
        case 2: quantize_q8a128_kernel<float><<<grid_dim, block, 0, s>>>(
            (const float*)act, (block_q8a128*)out, rows, cols, sum_norm); break;
    }
}

extern "C" void run_dequantize_q8a128(
    const void* in, void* out, int32_t rows, int32_t cols, int32_t dtype, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const int total_tiles = (int)(((int64_t)rows * cols) / 128);
    const int grid = (total_tiles + 7) / 8;
    dim3 grid_dim(grid > 0 ? grid : 1, 1, 1);
    dim3 block(256, 1, 1);  // 8 warps, grid-strided
    switch (dtype) {
        case 0: dequantize_q8a128_kernel<half><<<grid_dim, block, 0, s>>>(
            (const block_q8a128*)in, (half*)out, rows, cols); break;
        case 1: dequantize_q8a128_kernel<__nv_bfloat16><<<grid_dim, block, 0, s>>>(
            (const block_q8a128*)in, (__nv_bfloat16*)out, rows, cols); break;
        case 2: dequantize_q8a128_kernel<float><<<grid_dim, block, 0, s>>>(
            (const block_q8a128*)in, (float*)out, rows, cols); break;
    }
}

// KO weight quantize/dequant: F32 [nrows × ncols] (row-major) ↔ lane-major per-128 KO
// chunks. qtype = QTYPE_Q{4,5,6,8}_KO. One warp per 1024-weight chunk (8 rows × 128 K),
// 8 warps/block, grid-strided. nrows must be a multiple of 8, ncols a multiple of 128.
extern "C" void run_quantize_ko(
    const float* w, void* out, int32_t nrows, int32_t ncols, int32_t qtype, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const int total_chunks = (nrows / 8) * (ncols / 128);
    const int grid = (total_chunks + 7) / 8;
    dim3 grid_dim(grid > 0 ? grid : 1, 1, 1);
    dim3 block(256, 1, 1);
    uint8_t* ob = (uint8_t*)out;
    switch (qtype) {
        case QTYPE_Q4_KO: quantize_ko_affine_kernel<15, 0, 0><<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        case QTYPE_Q5_KO: quantize_ko_affine_kernel<31, 0, 128><<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        case QTYPE_Q6_KO: quantize_ko_affine_kernel<63, 256, 0><<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        case QTYPE_Q8_KO: quantize_q8_ko_kernel<<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        case QTYPE_Q2_KO: quantize_q2_ko_kernel<<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        case QTYPE_Q3_KO: quantize_q3_ko_kernel<<<grid_dim, block, 0, s>>>(w, ob, nrows, ncols); break;
        // NOTE: deliberately no QTYPE_MXFP4_KO arm — MXFP4 repacks by an exact host-side
        // byte permutation (ko_quant::mxfp4_native_to_ko_gpu_chunk), never a requant.
        default: return; // Invalid qtype — callers must not reach here
    }
}

extern "C" void run_dequantize_ko(
    const void* in, float* out, int32_t nrows, int32_t ncols, int32_t qtype, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const int total_chunks = (nrows / 8) * (ncols / 128);
    const int grid = (total_chunks + 7) / 8;
    dim3 grid_dim(grid > 0 ? grid : 1, 1, 1);
    dim3 block(256, 1, 1);
    const uint8_t* ib = (const uint8_t*)in;
    switch (qtype) {
        case QTYPE_Q4_KO: dequantize_ko_affine_kernel<0, 0><<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        case QTYPE_Q5_KO: dequantize_ko_affine_kernel<0, 128><<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        case QTYPE_Q6_KO: dequantize_ko_affine_kernel<256, 0><<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        case QTYPE_Q8_KO: dequantize_q8_ko_kernel<<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        case QTYPE_Q2_KO: dequantize_q2_ko_kernel<<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        case QTYPE_Q3_KO: dequantize_q3_ko_kernel<<<grid_dim, block, 0, s>>>(ib, out, nrows, ncols); break;
        default: return; // Invalid qtype — callers must not reach here
    }
}

// =============================================================================
// DISPATCHER: quantize_block (unified interface)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
//
// elem_count: total number of f32 elements to quantize
// The function calculates number of quantized blocks based on qtype's block size

extern "C" void run_quantize_block(
    const float* src,
    void* dst,
    int32_t elem_count,
    int32_t qtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // q8a128: q8 activation, packed into q8a1024 flat-grouped blocks (8 × 128-tiles
    // per 1152B super-block, qs de-interleaved from the per-32 ds; see blocks.cuh).
    // One warp per 128-tile, 8 warps/block grid-strided — different from the generic
    // per-warp-block geometry below, so dispatch it directly. The flat placement is
    // position-independent (rows=1 here matches any (rows,cols) split with the same
    // tile count). f32-in; the typed (bf16/f16) path is run_quantize_q8a128.
    if (qtype == QTYPE_Q8A128V || qtype == QTYPE_Q8A128X) {
        const int ntile = ceil_div(elem_count, QK8A128);
        const int grid = (ntile + 7) / 8;
        // Raw Σx. This unified entry is the GGML/weight-side quantize; the
        // amax-normalised convention belongs to the ACTIVATION path, whose
        // producer is `run_quantize_q8a128` and whose choice rides on
        // `Q8a128Operand::sum_scale`. Pinning it here rather than plumbing a
        // flag keeps the two entries byte-identical, which
        // `unified_block_quantize_matches_typed_q8a128` asserts.
        quantize_q8a128_kernel<float><<<dim3(grid > 0 ? grid : 1, 1, 1), dim3(256, 1, 1), 0, s>>>(
            src, (block_q8a128*)dst, 1, ntile * QK8A128, /*sum_norm=*/0);
        return;
    }

    // Calculate number of quantized blocks based on format
    int num_blocks;
    int block_size;  // elements per quantized block
    int threads_per_block = 32;  // warp size - our kernels are warp-cooperative
    bool is_k_quant = false;  // K-quants use __shared__ memory, need 1 warp per block

    switch (qtype) {
        // Standard 32-element formats
        case QTYPE_Q4_0:
        case QTYPE_Q4_1:
        case QTYPE_Q5_0:
        case QTYPE_Q5_1:
        case QTYPE_Q8_0:
        case QTYPE_Q8_1:
        case QTYPE_Q4_KS:
        case QTYPE_Q8_KS:
        case QTYPE_Q2_0:
        case QTYPE_Q3_0:
            block_size = 32;
            break;
        // K-quant 256-element formats
        case QTYPE_Q2_K:
        case QTYPE_Q3_K:
        case QTYPE_Q4_K:
        case QTYPE_Q5_K:
        case QTYPE_Q6_K:
        case QTYPE_Q8_K:
            block_size = 256;
            is_k_quant = true;
            break;
        // AWQ 128-element formats (both use 128 elements per block)
        case QTYPE_QAWQ:
        case QTYPE_QAWQ_G64:
            block_size = 128;
            break;
        default:
            return;  // Invalid qtype
    }

    num_blocks = ceil_div(elem_count, block_size);

    // K-quants use __shared__ memory that's shared across all warps in a thread block.
    // Each warp needs exclusive access to shared memory, so we must launch with 1 warp per block.
    // Standard quants don't use shared memory, so they can use multiple warps per block.
    dim3 grid, block_dim;
    if (is_k_quant) {
        // K-quants: 1 warp (32 threads) per thread block - each block gets exclusive shared memory
        grid = dim3(num_blocks, 1, 1);
        block_dim = dim3(threads_per_block, 1, 1);  // 32 threads = 1 warp
    } else {
        // Standard quants: 4 warps per thread block for better occupancy
        int blocks_per_grid = ceil_div(num_blocks, 4);
        grid = dim3(blocks_per_grid, 1, 1);
        block_dim = dim3(threads_per_block * 4, 1, 1);  // 128 threads = 4 warps
    }

    switch (qtype) {
        case QTYPE_Q4_0:
            quantize_tensor_q4_0<<<grid, block_dim, 0, s>>>(src, (block_q4_0*)dst, num_blocks);
            break;
        case QTYPE_Q4_1:
            quantize_tensor_q4_1<<<grid, block_dim, 0, s>>>(src, (block_q4_1*)dst, num_blocks);
            break;
        case QTYPE_Q5_0:
            quantize_tensor_q5_0<<<grid, block_dim, 0, s>>>(src, (block_q5_0*)dst, num_blocks);
            break;
        case QTYPE_Q5_1:
            quantize_tensor_q5_1<<<grid, block_dim, 0, s>>>(src, (block_q5_1*)dst, num_blocks);
            break;
        case QTYPE_Q8_0:
            quantize_tensor_q8_0<<<grid, block_dim, 0, s>>>(src, (block_q8_0*)dst, num_blocks);
            break;
        case QTYPE_Q2_K:
            quantize_tensor_q2_K<<<grid, block_dim, 0, s>>>(src, (block_q2_K*)dst, num_blocks);
            break;
        case QTYPE_Q3_K:
            quantize_tensor_q3_K<<<grid, block_dim, 0, s>>>(src, (block_q3_K*)dst, num_blocks);
            break;
        case QTYPE_Q4_K:
            quantize_tensor_q4_K<<<grid, block_dim, 0, s>>>(src, (block_q4_K*)dst, num_blocks);
            break;
        case QTYPE_Q5_K:
            quantize_tensor_q5_K<<<grid, block_dim, 0, s>>>(src, (block_q5_K*)dst, num_blocks);
            break;
        case QTYPE_Q6_K:
            quantize_tensor_q6_K<<<grid, block_dim, 0, s>>>(src, (block_q6_K*)dst, num_blocks);
            break;
        case QTYPE_Q8_1:
            quantize_tensor_q8_1<<<grid, block_dim, 0, s>>>(src, (block_q8_1*)dst, num_blocks);
            break;
        case QTYPE_Q4_KS:
            quantize_tensor_q4_ks<<<grid, block_dim, 0, s>>>(src, (block_q4_ks*)dst, num_blocks);
            break;
        case QTYPE_Q8_KS:
            quantize_tensor_q8_ks<<<grid, block_dim, 0, s>>>(src, (block_q8_ks*)dst, num_blocks);
            break;
        case QTYPE_Q2_0:
            quantize_tensor_q2_0<<<grid, block_dim, 0, s>>>(src, (block_q2_0*)dst, num_blocks);
            break;
        case QTYPE_Q3_0:
            quantize_tensor_q3_0<<<grid, block_dim, 0, s>>>(src, (block_q3_0*)dst, num_blocks);
            break;
        case QTYPE_Q8_K:
            quantize_tensor_q8_K<<<grid, block_dim, 0, s>>>(src, (block_q8_K*)dst, num_blocks);
            break;
        case QTYPE_QAWQ:
            quantize_tensor_q_awq<<<grid, block_dim, 0, s>>>(src, dst, num_blocks);
            break;
        case QTYPE_QAWQ_G64:
            quantize_tensor_q_awq_g64<<<grid, block_dim, 0, s>>>(src, dst, num_blocks);
            break;
    }
}

// =============================================================================
// DISPATCHER: dequantize_block
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
// out_dtype: 0=F32, 1=F16, 2=BF16

// Standalone MXFP4 dequant (out_dtype: 0=F32, 1=F16, 2=BF16). Kept separate from the
// QTYPE-indexed run_dequantize_block dispatch so MXFP4 does not perturb the locked
// QTYPE_COUNT size tables. Same 256-elems-per-block / 32-thread launch as q4_0.
extern "C" void run_dequantize_mxfp4(
    const void* src,
    void* dst,
    int32_t elem_count,
    int32_t out_dtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int num_blocks = (elem_count + 255) / 256;
    if (num_blocks < 1) num_blocks = 1;
    dim3 grid(num_blocks, 1, 1), block(32, 1, 1);
    int nb32 = elem_count / 32;
    if (out_dtype == 0)
        dequantize_block_mxfp4_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
    else if (out_dtype == 1)
        dequantize_block_mxfp4_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
    else
        dequantize_block_mxfp4_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
}

extern "C" void run_dequantize_block(
    const void* src,
    void* dst,
    int32_t elem_count,
    int32_t qtype,
    int32_t out_dtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // q8a128: q8 activation in q8a1024 flat-grouped blocks. out_dtype here is the
    // unified ordering 0=F32, 1=F16, 2=BF16 (NB: the typed run_dequantize_q8a128 uses
    // the YType ordering 0=F16,1=BF16,2=F32 instead). One warp per 128-tile.
    if (qtype == QTYPE_Q8A128V || qtype == QTYPE_Q8A128X) {
        const int ntile = ceil_div(elem_count, QK8A128);
        const int grid = (ntile + 7) / 8;
        const dim3 g(grid > 0 ? grid : 1, 1, 1), b(256, 1, 1);  // 8 warps, grid-strided
        const int cols = ntile * QK8A128;
        if (out_dtype == 0)
            dequantize_q8a128_kernel<float><<<g, b, 0, s>>>((const block_q8a128*)src, (float*)dst, 1, cols);
        else if (out_dtype == 1)
            dequantize_q8a128_kernel<half><<<g, b, 0, s>>>((const block_q8a128*)src, (half*)dst, 1, cols);
        else
            dequantize_q8a128_kernel<__nv_bfloat16><<<g, b, 0, s>>>((const block_q8a128*)src, (__nv_bfloat16*)dst, 1, cols);
        return;
    }

    // Determine grid/block dimensions based on quant type
    int block_dim;
    int num_blocks;
    int nb32 = 0;

    bool is_k128_type = (qtype == QTYPE_QAWQ || qtype == QTYPE_QAWQ_G64);  // K/128 AWQ native types
    bool is_k_quant =
        (qtype == QTYPE_Q2_K || qtype == QTYPE_Q3_K || qtype == QTYPE_Q4_K ||
         qtype == QTYPE_Q5_K || qtype == QTYPE_Q6_K || qtype == QTYPE_Q8_K);

    if (is_k128_type) {
        // K/128 types: 128 elements per block, 32 threads
        block_dim = 32;
        num_blocks = ceil_div(elem_count, 128);
    } else if (is_k_quant) {
        // GGML K-quants: 256 elements per block
        block_dim = (qtype == QTYPE_Q4_K) ? 32 : 64;
        num_blocks = ceil_div(elem_count, 256);
    } else {
        // Basic quants: 32 elements per block
        if (qtype == QTYPE_Q5_0 || qtype == QTYPE_Q5_1) {
            block_dim = CUDA_DEQUANTIZE_BLOCK_SIZE;
            num_blocks = ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
            nb32 = elem_count;
        } else {
            block_dim = 32;
            num_blocks = ceil_div(elem_count, 256);
            nb32 = elem_count / 32;
        }
    }

    dim3 grid(num_blocks, 1, 1);
    dim3 block(block_dim, 1, 1);

    // Dispatch based on qtype and out_dtype
    switch (qtype) {
        // Basic quants (with k parameter)
        case QTYPE_Q4_0:
            if (out_dtype == 0) dequantize_block_q4_0_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_0_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_0_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q4_1:
            if (out_dtype == 0) dequantize_block_q4_1_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_1_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_1_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q5_0:
            if (out_dtype == 0) dequantize_block_q5_0_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q5_0_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q5_0_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q5_1:
            if (out_dtype == 0) dequantize_block_q5_1_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q5_1_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q5_1_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q8_0:
            if (out_dtype == 0) dequantize_block_q8_0_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_0_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_0_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;

        // K-quants (no k parameter)
        case QTYPE_Q2_K:
            if (out_dtype == 0) dequantize_block_q2_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q2_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q2_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q3_K:
            if (out_dtype == 0) dequantize_block_q3_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q3_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q3_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q4_K:
            if (out_dtype == 0) dequantize_block_q4_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q4_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q4_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q5_K:
            if (out_dtype == 0) dequantize_block_q5_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q5_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q5_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q6_K:
            if (out_dtype == 0) dequantize_block_q6_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q6_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q6_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q8_1:
            if (out_dtype == 0) dequantize_block_q8_1_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_1_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_1_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q4_KS:
            if (out_dtype == 0) dequantize_block_q4_ks_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_ks_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_ks_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q8_KS:
            if (out_dtype == 0) dequantize_block_q8_ks_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_ks_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_ks_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q2_0:
            if (out_dtype == 0) dequantize_block_q2_0_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q2_0_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q2_0_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q3_0:
            if (out_dtype == 0) dequantize_block_q3_0_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q3_0_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_q3_0_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_R16: // R16 (extract K from block_r16::d[])
            if (out_dtype == 0) dequantize_block_r16_f32<<<grid, block, 0, s>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_r16_f16<<<grid, block, 0, s>>>(src, (__half*)dst, nb32);
            else dequantize_block_r16_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst, nb32);
            break;

        // Q8_K (K-quant 8-bit, 256 elements per block)
        case QTYPE_Q8_K:
            if (out_dtype == 0) dequantize_block_q8_K_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q8_K_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q8_K_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;

        // K/128 AWQ types (no k parameter, 128 elements per block)
        case QTYPE_QAWQ:
            if (out_dtype == 0) dequantize_block_q_awq_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q_awq_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q_awq_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_QAWQ_G64:
            if (out_dtype == 0) dequantize_block_q_awq_g64_f32<<<grid, block, 0, s>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q_awq_g64_f16<<<grid, block, 0, s>>>(src, (__half*)dst);
            else dequantize_block_q_awq_g64_bf16<<<grid, block, 0, s>>>(src, (__nv_bfloat16*)dst);
            break;
    }
}

// =============================================================================
// DISPATCHER: dequantize_mul_mat_vec
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).

extern "C" void run_dequantize_mul_mat_vec(
    const void* vx,
    const float* y,
    float* dst,
    int32_t ncols,
    int32_t nrows,
    int32_t qtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);

    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);

    switch (qtype) {
        case QTYPE_Q4_0: dequantize_mul_mat_vec_q4_0_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q4_1: dequantize_mul_mat_vec_q4_1_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_0: dequantize_mul_mat_vec_q5_0_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_1: dequantize_mul_mat_vec_q5_1_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_0: dequantize_mul_mat_vec_q8_0_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q2_K: dequantize_mul_mat_vec_q2_k<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q3_K: dequantize_mul_mat_vec_q3_k<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q4_K: dequantize_mul_mat_vec_q4_k<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_K: dequantize_mul_mat_vec_q5_k<<<grid, block, 0, s>>>(vx, y, dst, ncols); break;  // q5_k has different signature
        case QTYPE_Q6_K: dequantize_mul_mat_vec_q6_k<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_1: dequantize_mul_mat_vec_q8_1_cuda<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_K: dequantize_mul_mat_vec_q8_k<<<grid, block, 0, s>>>(vx, y, dst, ncols, nrows); break;
    }
}

// =============================================================================
// DISPATCHER: mul_mat_vec_q8_1 (batched)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
// b_size: 1-8

extern "C" void run_mul_mat_vec_q8_1(
    const void* vx,
    const void* vy,
    float* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t b_size,
    int32_t qtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Compute grid/block dimensions based on batch size
    int nblocks, nwarps;
    switch (b_size) {
        case 1:
            nblocks = nrows_x;
            nwarps = 4;
            break;
        case 2:
        case 3:
        case 4:
            nblocks = ceil_div(nrows_x, 2);
            nwarps = 4;
            break;
        default:  // 5-8
            nblocks = ceil_div(nrows_x, 2);
            nwarps = 2;
            break;
    }
    
    dim3 grid(nblocks, 1, 1);
    dim3 block(WARP_SIZE, nwarps, 1);
    
    // Macro to dispatch based on batch size
    #define DISPATCH_BSIZE(qname) \
        switch (b_size) { \
            case 1: mul_mat_vec_##qname##_q8_1_cuda1<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 2: mul_mat_vec_##qname##_q8_1_cuda2<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 3: mul_mat_vec_##qname##_q8_1_cuda3<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 4: mul_mat_vec_##qname##_q8_1_cuda4<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 5: mul_mat_vec_##qname##_q8_1_cuda5<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 6: mul_mat_vec_##qname##_q8_1_cuda6<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 7: mul_mat_vec_##qname##_q8_1_cuda7<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 8: mul_mat_vec_##qname##_q8_1_cuda8<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
        }
    
    switch (qtype) {
        case QTYPE_Q4_0: DISPATCH_BSIZE(q4_0); break;
        case QTYPE_Q4_1: DISPATCH_BSIZE(q4_1); break;
        case QTYPE_Q5_0: DISPATCH_BSIZE(q5_0); break;
        case QTYPE_Q5_1: DISPATCH_BSIZE(q5_1); break;
        case QTYPE_Q8_0: DISPATCH_BSIZE(q8_0); break;
        case QTYPE_Q2_K: DISPATCH_BSIZE(q2_K); break;
        case QTYPE_Q3_K: DISPATCH_BSIZE(q3_K); break;
        case QTYPE_Q4_K: DISPATCH_BSIZE(q4_K); break;
        case QTYPE_Q5_K: DISPATCH_BSIZE(q5_K); break;
        case QTYPE_Q6_K: DISPATCH_BSIZE(q6_K); break;
        case QTYPE_Q8_1: DISPATCH_BSIZE(q8_1); break;
        case QTYPE_Q8_K: DISPATCH_BSIZE(q8_K); break;
    }

    #undef DISPATCH_BSIZE
}

// =============================================================================
// DISPATCHER: mul_mat (tensor core / MMQ)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).

extern "C" void run_mul_mat(
    const void* vx,
    const void* vy,
    float* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t ncols_y,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t qtype,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Get MMQ tile sizes based on quant type
    int mmq_x, mmq_y;
    switch (qtype) {
        case QTYPE_Q4_0:
        case QTYPE_Q4_1:
        case QTYPE_Q2_K:
        case QTYPE_Q4_K:
        case QTYPE_Q5_K:
            mmq_x = 64; mmq_y = 128;
            break;
        case QTYPE_Q5_0:
        case QTYPE_Q5_1:
        case QTYPE_Q8_0:
            mmq_x = 128; mmq_y = 64;
            break;
        case QTYPE_Q3_K:
            mmq_x = 128; mmq_y = 128;
            break;
        case QTYPE_Q6_K:
        case QTYPE_Q8_K:
            mmq_x = 64; mmq_y = 64;
            break;
        default:
            mmq_x = 64; mmq_y = 128;
            break;
    }

    // A small batch takes the mmq_x = 16 instantiation: the dot work scales
    // with the tile's batch width while the weight-tile loads do not, so a
    // 12-row batch under a 64-wide tile spends 4/5 of its arithmetic on clamped
    // padding columns whose results the store discards. Measured on Q6_K
    // [12288, 4096] (rows → µs): one narrow column serves 9..16 at ~107 vs the
    // wide tile's ~251, and even two columns — a second full pass over the
    // weight tiles, but a concurrent one — serve 17..32 at ~200. From three
    // columns up the extra passes cancel the padding saved and the wide tile
    // takes over. The curve lives in candle-transformers
    // `quantized_matmul::tests::qmatmul_row_count_curve`; re-measure before
    // moving this boundary.
    const bool narrow = ncols_y <= 32;
    if (narrow) mmq_x = 16;

    dim3 grid(ceil_div(nrows_x, mmq_y), ceil_div(ncols_y, mmq_x), 1);
    dim3 block(WARP_SIZE, 4, 1);

    #define DISPATCH_MMQ(qname) \
        if (narrow) { \
            mul_mat_##qname##_x16<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); \
        } else { \
            mul_mat_##qname<<<grid, block, 0, s>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); \
        }

    switch (qtype) {
        case QTYPE_Q4_0: DISPATCH_MMQ(q4_0); break;
        case QTYPE_Q4_1: DISPATCH_MMQ(q4_1); break;
        case QTYPE_Q5_0: DISPATCH_MMQ(q5_0); break;
        case QTYPE_Q5_1: DISPATCH_MMQ(q5_1); break;
        case QTYPE_Q8_0: DISPATCH_MMQ(q8_0); break;
        case QTYPE_Q2_K: DISPATCH_MMQ(q2_K); break;
        case QTYPE_Q3_K: DISPATCH_MMQ(q3_K); break;
        case QTYPE_Q4_K: DISPATCH_MMQ(q4_K); break;
        case QTYPE_Q5_K: DISPATCH_MMQ(q5_K); break;
        case QTYPE_Q6_K: DISPATCH_MMQ(q6_K); break;
        case QTYPE_Q8_1: DISPATCH_MMQ(q8_1); break;
        case QTYPE_Q8_K: DISPATCH_MMQ(q8_K); break;
    }

    #undef DISPATCH_MMQ
}
