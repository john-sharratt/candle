// The host dispatcher for the fused kernel, the register round-trip parity
// entry, and the per-side error sampling kernels.
//
// `run_select_kv_format_palette4_paged` is the host entry point that launches
// the quantile and fused selection kernels. The parity entry checks the
// register round trips against the Rust side, and the sampling kernels measure
// quantization error per side (K and V) over the paged arena.

// =============================================================================
// HOST-SIDE DISPATCHER  —  fused paged palette4
// =============================================================================
//
// Two-pass pipeline. Both kernels are launched with grid.x = total_heads
// (= chunks · n_kv_head). Pass 1 writes the per-(chunk, head) statistics
// pass 2 needs (head amax, p95, q-relevance quantiles). Pass 2 reads
// those statistics, the per-head K and V activations, and the candidate
// format set, and writes the full set of per-head selection outputs.
//
// Both passes run sequentially on the caller's stream. There is no
// concurrency benefit from launching them on separate streams because
// pass 2 has a hard data dependency on every output of pass 1.
//
//   pass 1 (1 block × 128 threads per (chunk, head))
//     ├─ k_head_amax, v_head_amax
//     ├─ k_head_p95,  v_head_p95
//     └─ q_relevance_median, q_relevance_spread
//                 │
//                 ▼
//   pass 2 (1 block × 128 threads per (chunk, head))
//     ├─ palette_tags, palette_scale  (per slot)
//     ├─ palette_map, effective_block_tags  (per block)
//     ├─ head_tag  (per head)
//     └─ q_relevance_out  (per block, optional)
//
// Threshold conventions
// ---------------------
// V-side thresholds use the same lo/hi convention as K-side:
// `*_threshold_lo > *_threshold_hi` numerically. lo is the lenient
// threshold (applied at low q-relevance for K, at non-sink tokens for V);
// hi is the stricter threshold (applied at high relevance / peak sink).
//
// Sink detection on V is automatic and parameter-free (z-score + tanh
// on chunk-local statistics — see Phase 2.5 in the kernel). To disable
// sink protection on V, pass v_threshold_lo == v_threshold_hi. To
// disable q-relevance scaling on K, pass k_threshold_lo == k_threshold_hi.

extern "C" void run_select_kv_format_palette4_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    float* q_relevance_median,
    float* q_relevance_spread,
    float* k_head_amax,
    float* v_head_amax,
    float* k_head_p95,
    float* v_head_p95,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    float k_threshold_hi,
    float k_threshold_lo,
    float v_threshold_hi,
    float v_threshold_lo,
    int total_heads,
    int blocks_per_head,
    int n_kv_head,
    int arena_chunks,
    const int* valid_ranges,
    int*   k_palette_tags,
    int*   v_palette_tags,
    float* k_palette_scale,
    float* v_palette_scale,
    int*   k_palette_map,
    int*   v_palette_map,
    int*   k_effective_block_tags,
    int*   v_effective_block_tags,
    int*   k_head_tags,
    int*   v_head_tags,
    float* q_relevance_out,
    cudaStream_t stream
) {
    if (total_heads == 0) return;

    // Pass 1: compute per-(chunk,head) relevance quantiles AND head amaxes.
    // (Generic over blocks_per_head — no HB dispatch needed.)
    approximate_q_relevance_quantiles<<<total_heads, QREL_QUANTILE_THREADS, 0, stream>>>(
        per_head_table_raw,
        head_gids,
        q_relevance_median,
        q_relevance_spread,
        k_head_amax,
        v_head_amax,
        k_head_p95,
        v_head_p95,
        blocks_per_head,
        total_heads,
        n_kv_head,
        arena_chunks,
        valid_ranges
    );

    // Pass 2: fused selection + palette4 grouping, one block per (chunk, head).
    // Dispatch the HB template instantiation matching blocks_per_head
    // (= head_dim); the per-instantiation MaxShared carveout maximises the
    // shared-memory partition on Ada/Hopper so the HB=128 layout (~12.6 KB)
    // reaches 8 blocks/SM and the HB=256 layout (~23.8 KB) reaches 4.
    auto launch_pass2 = [&](auto hb_tag) {
        constexpr int HB = decltype(hb_tag)::value;
        cudaFuncSetAttribute(
            select_kv_format_palette4_paged<HB>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        select_kv_format_palette4_paged<HB><<<total_heads, FUSED_THREADS_PER_BLOCK, 0, stream>>>(
            per_head_table_raw,
            head_gids,
            q_relevance_median,
            q_relevance_spread,
            k_head_amax,
            v_head_amax,
            k_head_p95,
            v_head_p95,
            k_candidates,
            v_candidates,
            num_k_candidates,
            num_v_candidates,
            k_threshold_hi,
            k_threshold_lo,
            v_threshold_hi,
            v_threshold_lo,
            total_heads,
            blocks_per_head,
            n_kv_head,
            arena_chunks,
            valid_ranges,
            k_palette_tags,
            v_palette_tags,
            k_palette_scale,
            v_palette_scale,
            k_palette_map,
            v_palette_map,
            k_effective_block_tags,
            v_effective_block_tags,
            k_head_tags,
            v_head_tags,
            q_relevance_out);
    };

    switch (blocks_per_head) {
        case 128: launch_pass2(HbTag<128>{}); break;
        case 256: launch_pass2(HbTag<256>{}); break;
        default:  return;  // unsupported head_dim: no selection outputs written
    }
}

// =============================================================================
// REGISTER ROUND-TRIP PARITY — test entry point
// =============================================================================
// For each 32-element block, the search's round trip of `fmt` two ways: through
// the block's bytes (encode into shared memory, decode each element back) and
// through `lane_roundtrip_for_fmt`. The search uses the register path wherever
// `lane_roundtrip` has it, on the guarantee that the two agree bit for bit; the
// test that calls this asserts it. One warp per block, on the caller's stream.
// `recon_lane` gets the bytes path's value for a format without a register path.

__global__ void select_roundtrip_parity_kernel(
    const float* __restrict__ src,
    float* __restrict__ recon_bytes,
    float* __restrict__ recon_lane,
    int num_blocks,
    float outer,
    int fmt,
    int is_k)
{
    __shared__ float   f32_smem[FUSED_WARP_SIZE];
    __shared__ uint8_t quant_smem[MAX_QUANT_BLOCK_BYTES];
    const int blk  = blockIdx.x;
    const int lane = threadIdx.x;
    if (blk >= num_blocks) return;

    const float orig = src[blk * FUSED_WARP_SIZE + lane];
    with_select_fmt(fmt, [&](auto tag) {
        constexpr int FMT = decltype(tag)::value;
        f32_smem[lane] = orig * outer;
        __syncwarp();
        quantize_block_for_fmt<FMT>(f32_smem, quant_smem, is_k != 0);
        __syncwarp();
        const float bytes = dequant_element_for_fmt<FMT>(quant_smem, lane, outer, is_k != 0);
        recon_bytes[blk * FUSED_WARP_SIZE + lane] = bytes;
        if constexpr (lane_roundtrip<FMT>::value) {
            recon_lane[blk * FUSED_WARP_SIZE + lane] =
                lane_roundtrip_for_fmt<FMT>(orig * outer, outer, f32_smem, is_k != 0);
        } else {
            recon_lane[blk * FUSED_WARP_SIZE + lane] = bytes;
        }
    });
}

extern "C" void run_select_roundtrip_parity(
    const void* src,
    void* recon_bytes,
    void* recon_lane,
    int num_blocks,
    float outer,
    int fmt,
    int is_k,
    void* stream)
{
    if (num_blocks <= 0) return;
    select_roundtrip_parity_kernel<<<num_blocks, FUSED_WARP_SIZE, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)recon_bytes, (float*)recon_lane,
        num_blocks, outer, fmt, is_k);
}

// =============================================================================
// PER-SIDE ERROR SAMPLING KERNEL
// =============================================================================
//
// Samples per-block reconstruction error for one side (K or V) across a
// candidate format set. Used by the older non-fused selection path and
// by the offline threshold-sweep tooling — it does NOT feed into the
// palette4 kernel above (the fused kernel does its own threshold
// evaluation inline).
//
// For each (chunk, dim, candidate, head), the kernel evaluates
// `quantize → dequantize` on a single sampled lane (`sample_token`) of
// each block and records:
//
//   error_out      : max|x − roundtrip(x)| for the sampled element
//   q_relevance_out: per-block Σ(q²k²)/Σ(k²)   (K side only, written
//                    once when quant_index == 0 to avoid clobbering
//                    across candidates)
//
// Grid mapping is intentionally:
//   grid.x = head_idx
//   grid.y = quant_index    (which candidate format to test)
//   grid.z = dim_idx        (block-in-head)
// Chunk is handled inside the block by assigning one warp per chunk and
// looping over the active paged batch.

extern "C" __global__ void sample_quant_errors_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const int* __restrict__ candidates,
    int num_candidates,
    float* __restrict__ error_out,
    float* __restrict__ q_relevance_out,
    int sample_token,
    int side_is_k,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    __shared__ float   smem_f32  [SELECT_WARPS_PER_BLOCK][32];
    __shared__ uint8_t smem_quant[SELECT_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];

    const int head_idx = blockIdx.x;
    const int quant_index = blockIdx.y;
    const int dim_idx = blockIdx.z;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    if (head_idx >= n_kv_head || quant_index >= num_candidates || dim_idx >= head_dim) return;
    if (sample_token < 0 || sample_token >= 32) return;

    const int fmt = candidates[quant_index];
    float*   warp_f32   = smem_f32  [warp_in_block];
    uint8_t* warp_quant = smem_quant[warp_in_block];

    // One CUDA block samples one dim, which lives in exactly one palette band;
    // address that band through its own gid.
    const int sub_head_dim = head_dim / N_PALETTE;
    const int band     = dim_idx / sub_head_dim;
    const int dim_in_band = dim_idx - band * sub_head_dim;

    for (int chunk_idx = warp_in_block; chunk_idx < num_chunks; chunk_idx += SELECT_WARPS_PER_BLOCK) {
        const char* chunk_data;
        int src_fmt;
        float band_outer;
        resolve_band_source(per_head_table_raw, head_gids, chunk_idx, head_idx,
                            n_kv_head, arena_chunks, band, side_is_k == 0,
                            &chunk_data, &src_fmt, &band_outer);

        float x_val;
        float k_val = 0.0f;
        float q_val = 0.0f;
        if (ArenaFormat::is_quantized(src_fmt)) {
            const int blk_bytes = quant_block_bytes(src_fmt);
            const char* blk_ptr = chunk_data + (int64_t)dim_in_band * blk_bytes;
            x_val = side_is_k
                ? dequant_element_inline<float, true >(blk_ptr, lane, src_fmt, 1.0f)
                : dequant_element_inline<float, false>(blk_ptr, lane, src_fmt, 1.0f);
            if (side_is_k) {
                k_val = x_val;
                q_val = dequant_q_element(blk_ptr, lane, src_fmt);
            }
        } else {
            const int elem_in_chunk = float_band_elem(dim_in_band, lane, sub_head_dim);
            x_val = load_as_float(chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(src_fmt));
            if (side_is_k) {
                k_val = x_val;
            }
        }

        float x_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            x_rt = x_val;
        } else {
            warp_f32[lane] = x_val;
            __syncwarp();
            quantize_to_smem(warp_f32, warp_quant, fmt, side_is_k);
            __syncwarp();
            x_rt = side_is_k
                ? dequant_element_inline<float, true >((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f)
                : dequant_element_inline<float, false>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float sample_err = max_abs_error_warp(x_val, x_rt);

        float q_relevance = 1.0f;
        if (side_is_k) {
            const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
            if (has_q) {
                q_relevance = block_relevance(k_val, q_val);
                q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
            }
        }

        if (lane == sample_token) {
            const int out_idx = (((chunk_idx * head_dim) + dim_idx) * num_candidates + quant_index) * n_kv_head + head_idx;
            error_out[out_idx] = sample_err;
            if (quant_index == 0) {
                const int rel_idx = ((chunk_idx * head_dim) + dim_idx) * n_kv_head + head_idx;
                q_relevance_out[rel_idx] = q_relevance;
            }
        }
    }
}

extern "C" void run_sample_quant_errors_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    const int* candidates,
    int num_candidates,
    float* error_out,
    float* q_relevance_out,
    int sample_token,
    int side_is_k,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks,
    void* stream
) {
    if (num_chunks == 0 || n_kv_head == 0 || head_dim == 0 || num_candidates == 0) return;

    const int warps_per_block = SELECT_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    dim3 grid((unsigned int)n_kv_head, (unsigned int)num_candidates, (unsigned int)head_dim);

    sample_quant_errors_paged<<<grid, threads_per_block, 0, (cudaStream_t)stream>>>(
        per_head_table_raw,
        head_gids,
        candidates,
        num_candidates,
        error_out,
        q_relevance_out,
        sample_token,
        side_is_k,
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks
    );
}

// =============================================================================
// FUSED K + V ERROR SAMPLING KERNEL
// =============================================================================
//
// Same role as `sample_quant_errors_paged`, but processes K and V
// together in a single launch. The fusion is mainly for table-lookup
// amortisation: each warp reads the per-head table entry once and uses
// it for both K and V samples, halving the global-memory traffic for
// that lookup vs. running the per-side kernel twice.
//
// (Q-relevance is computed from the K side but not used here — it's
// reused in the threshold-sweep tooling, where downstream code weights
// V errors by the K-derived q-relevance to make V compression stricter
// where Q attends to K.)
//
// Grid mapping: same as `sample_quant_errors_paged`:
//   grid.x = head
//   grid.y = quant_index     (shared K/V — must use the same candidate list)
//   grid.z = head_dim
// Chunk is handled inside the block by assigning one warp per chunk.

extern "C" __global__ void sample_quant_errors_kv_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const int* __restrict__ candidates,
    int num_candidates,
    float* __restrict__ k_error_out,
    float* __restrict__ v_error_out,
    int sample_token,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks
) {
    __shared__ float   smem_f32  [SELECT_WARPS_PER_BLOCK][32];
    __shared__ uint8_t smem_quant[SELECT_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];

    const int head_idx = blockIdx.x;
    const int quant_index = blockIdx.y;
    const int dim_idx = blockIdx.z;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    if (head_idx >= n_kv_head || quant_index >= num_candidates || dim_idx >= head_dim) return;
    if (sample_token < 0 || sample_token >= 32) return;

    const int fmt = candidates[quant_index];
    float*   warp_f32   = smem_f32  [warp_in_block];
    uint8_t* warp_quant = smem_quant[warp_in_block];

    // One CUDA block samples one dim, which lives in exactly one palette band;
    // address that band through its own gid (K and V bands independently).
    const int sub_head_dim = head_dim / N_PALETTE;
    const int band     = dim_idx / sub_head_dim;
    const int dim_in_band = dim_idx - band * sub_head_dim;

    for (int chunk_idx = warp_in_block; chunk_idx < num_chunks; chunk_idx += SELECT_WARPS_PER_BLOCK) {
        // ── K pass ────────────────────────────────────────────────────
        const char* k_chunk_data;
        int k_src_fmt;
        float k_band_outer;
        resolve_band_source(per_head_table_raw, head_gids, chunk_idx, head_idx,
                            n_kv_head, arena_chunks, band, false,
                            &k_chunk_data, &k_src_fmt, &k_band_outer);

        float k_val = 0.0f;
        float q_val = 0.0f;
        if (ArenaFormat::is_quantized(k_src_fmt)) {
            const int blk_bytes = quant_block_bytes(k_src_fmt);
            const char* blk_ptr = k_chunk_data + (int64_t)dim_in_band * blk_bytes;
            k_val = dequant_element_inline<float, true>(blk_ptr, lane, k_src_fmt, 1.0f);
            q_val = dequant_q_element(blk_ptr, lane, k_src_fmt);
        } else {
            const int elem_in_chunk = float_band_elem(dim_in_band, lane, sub_head_dim);
            k_val = load_as_float(k_chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(k_src_fmt));
        }

        float k_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            k_rt = k_val;
        } else {
            warp_f32[lane] = k_val;
            __syncwarp();
            quantize_to_smem(warp_f32, warp_quant, fmt, /*is_k=*/true);
            __syncwarp();
            k_rt = dequant_element_inline<float, true>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float k_err = max_abs_error_warp(k_val, k_rt);

        float q_relevance = 1.0f;
        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            q_relevance = block_relevance(k_val, q_val);
            q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
        }

        // ── V pass ────────────────────────────────────────────────────
        const char* v_chunk_data;
        int v_src_fmt;
        float v_band_outer;
        resolve_band_source(per_head_table_raw, head_gids, chunk_idx, head_idx,
                            n_kv_head, arena_chunks, band, true,
                            &v_chunk_data, &v_src_fmt, &v_band_outer);

        float v_val = 0.0f;
        if (ArenaFormat::is_quantized(v_src_fmt)) {
            const int blk_bytes = quant_block_bytes(v_src_fmt);
            const char* blk_ptr = v_chunk_data + (int64_t)dim_in_band * blk_bytes;
            v_val = dequant_element_inline<float, false>(blk_ptr, lane, v_src_fmt, 1.0f);
        } else {
            const int elem_in_chunk = float_band_elem(dim_in_band, lane, sub_head_dim);
            v_val = load_as_float(v_chunk_data, elem_in_chunk, arena_fmt_to_dtype_code(v_src_fmt));
        }

        float v_rt;
        if (fmt == SELECT_FMT_BF16 || fmt == SELECT_FMT_F16) {
            v_rt = v_val;
        } else {
            warp_f32[lane] = v_val;
            __syncwarp();
            quantize_to_smem(warp_f32, warp_quant, fmt, /*is_k=*/false);
            __syncwarp();
            v_rt = dequant_element_inline<float, false>((const char*)warp_quant, lane, select_fmt_to_arena_fmt(fmt), 1.0f);
        }
        const float v_err = max_abs_error_warp(v_val, v_rt);

        // ── Write outputs ─────────────────────────────────────────────
        if (lane == sample_token) {
            const int out_idx = (((chunk_idx * head_dim) + dim_idx) * num_candidates + quant_index) * n_kv_head + head_idx;
            k_error_out[out_idx] = k_err;
            v_error_out[out_idx] = v_err;
        }
    }
}

extern "C" void run_sample_quant_errors_kv_paged(
    const int64_t* per_head_table_raw,
    const int64_t* head_gids,
    const int* candidates,
    int num_candidates,
    float* k_error_out,
    float* v_error_out,
    int sample_token,
    int num_chunks,
    int n_kv_head,
    int head_dim,
    int arena_chunks,
    void* stream
) {
    if (num_chunks == 0 || n_kv_head == 0 || head_dim == 0 || num_candidates == 0) return;

    const int warps_per_block = SELECT_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    dim3 grid((unsigned int)n_kv_head, (unsigned int)num_candidates, (unsigned int)head_dim);

    sample_quant_errors_kv_paged<<<grid, threads_per_block, 0, (cudaStream_t)stream>>>(
        per_head_table_raw,
        head_gids,
        candidates,
        num_candidates,
        k_error_out,
        v_error_out,
        sample_token,
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks
    );
}
