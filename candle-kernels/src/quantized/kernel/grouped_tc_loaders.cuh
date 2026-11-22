// The activation loaders, tile store and FP grouped matmul of the grouped
// tensor-core GEMM.
//
// The shared pieces of the `grouped_tc` namespace: `load_activations_runtime`,
// `store_tile_output`, the FP `grouped_matmul_impl`, the q8a128 and SiLU
// activation loaders, `RING_I8`, `load_warp_chunk_int8`, and the affine fold
// helpers `ko_affine_fold`, `kseg_lo` and `load_a_frag_kseg` that the INT8 path
// uses.

using namespace tc_common;

// Runtime-batch activation load: BATCH_TILE×K_TILE into smem, rows >= b_cnt
// zeroed (M-dim padding). Mirrors s16_tc::load_activations but bounds at runtime
// so a tile may carry 1..16 tokens without reading past its expert's slice.
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations_runtime(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int b_start, int b_cnt, int k_offset, int y_stride, int tid)
{
    if constexpr ((std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) ||
                  (std::is_same_v<act_t, __nv_bfloat16> &&
                   std::is_same_v<compute_t, __nv_bfloat16>)) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            int4 data = (b < b_cnt)
                ? *reinterpret_cast<const int4*>(&vy[(b_start + b) * y_stride + k_offset + k])
                : ZERO4;
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    } else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        const half2 ZERO2 = {__float2half(0.f), __float2half(0.f)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            half2 h0 = ZERO2, h1 = ZERO2;
            if (b < b_cnt) {
                float4 f4 = *reinterpret_cast<const float4*>(
                    &vy[(b_start + b) * y_stride + k_offset + k]);
                h0 = __floats2half2_rn(f4.x, f4.y);
                h1 = __floats2half2_rn(f4.z, f4.w);
            }
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    } else if constexpr (std::is_same_v<act_t, float> &&
                         std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 ZERO2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            __nv_bfloat162 h0 = ZERO2, h1 = ZERO2;
            if (b < b_cnt) {
                float4 f4 = *reinterpret_cast<const float4*>(
                    &vy[(b_start + b) * y_stride + k_offset + k]);
                h0 = __floats2bfloat162_rn(f4.x, f4.y);
                h1 = __floats2bfloat162_rn(f4.z, f4.w);
            }
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k + 2]) = h1;
        }
    } else {
        // Scalar fallback (fp8 / mixed); zero rows beyond b_cnt.
        constexpr int TOTAL = BATCH_TILE * K_TILE;
        for (int idx = tid; idx < TOTAL; idx += NUM_THREADS) {
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            float val = 0.f;
            if (b < b_cnt) {
                const int off = (b_start + b) * y_stride + k_offset + k;
                if constexpr (std::is_same_v<act_t, float>) val = vy[off];
                else if constexpr (std::is_same_v<act_t, half>) val = __half2float(vy[off]);
                else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) val = __bfloat162float(vy[off]);
                else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) val = to_f32(vy[off]);
            }
            if constexpr (std::is_same_v<compute_t, half>) smem_A[b][k] = __float2half(val);
            else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) smem_A[b][k] = __float2bfloat16(val);
            else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) smem_A[b][k] = __nv_fp8_e4m3(__float2half(val));
        }
    }
}

// Shared tile-output store: the accumulated frag_c[4] → dst for the two token
// rows (b0, b1) this lane owns, at columns out_row / out_row+1. The (batch,row)
// mapping is identical for the FP and INT8 grouped impls, so both call this.
template <typename output_t>
__device__ __forceinline__ void store_tile_output(
    output_t* __restrict__ dst, const float frag_c[4],
    int dst_stride, int warp_row_base, int b_start, int b_cnt, int lane)
{
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    const int bend = b_start + b_cnt;
    const int b0 = b_start + groupID;
    const int b1 = b_start + groupID + 8;

    if (b0 < bend) {
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
    if (b1 < bend) {
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

// One (expert-tile, row-tile): MMA over K for up to 16 tokens × 32 rows.
// Identical inner loop to s16_tc::tc16_kernel_impl, but the weight pointer is
// the per-expert pointer and the batch slice is [b_start, b_start+b_cnt).
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void grouped_matmul_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int b_start, int b_cnt, int row_tile_idx,
    compute_t smem_A[BATCH_TILE][K_STRIDE], uint8_t* smem_W_flat)
{
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    const int row0 = row_tile_idx * N_TILE;
    if (row0 >= nrows || b_cnt <= 0) return;

    constexpr int K128_BYTES = sizeof(block_c_t);
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;

    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k_blocks = ncols / K_TILE;

    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_runtime<compute_t, act_t>(
            smem_A, activations, b_start, b_cnt, k_blk * K_TILE, y_stride, tid);
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0, k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

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

    store_tile_output<output_t>(dst, frag_c, dst_stride, warp_row_base, b_start, b_cnt, lane);
}

// =============================================================================
// INT8 q8a128 PATH — same grid / entry / store machinery as the FP grouped impl
// above; only the activation load, weight unpack, and MMA+fold differ. The
// contraction runs on the INT8 m16n8k32 tensor core: q8a128 activations (raw int8
// qs) × quantized weights (raw integers), int32 accumulate, deferred-scale fold to
// F32. Affine KO formats fold with ONE (scale, min) per 128-K per output row: the
// four k32 sub-MMAs of a 128-K tile collapse into a single int32 BEFORE the scale
// is applied (finer-scale sources — Q4_K per-32, Q6_K per-16 — are re-quantized to
// that per-128 grid by `to_ko`; see the note in loader/gemx_dequant.cuh). MXFP4
// (`is_mxfp4_persub`) instead folds each sub-MMA's int32 with its own E8M0
// power-of-two scale immediately — the fold drains one reusable accumulator per
// sub, so it holds no extra registers, and the per-32 scales apply exactly (no
// re-quantization of an E8M0 format onto an affine grid).
// =============================================================================

// Activation tile load: global → shared via cp.async (.ca, L1-resident — the tile is re-read
// by every grid.y row-tile). Issues the async copies ONLY; the caller commits them in the same
// cp.async group as the weight chunk and waits later, so the activation load pipelines a tile
// ahead of compute (its latency hides under the prior tile's MMA). qs: 128 threads × 16 B
// (token = tid>>3, kchunk = (tid&7)*16); ds: one (scale,sum) half2 per token. Rows ≥ b_cnt are
// zero-padded with a plain smem store (no global read, not part of the cp.async group).
template <int N_SUB = 1>
__device__ __forceinline__ void load_q8a128_activations(
    const block_q8a128* __restrict__ act, int b_start, int b_cnt, int k_blk,
    int tiles_per_row, int tid,
    int8_t smem_A_i8[N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[N_SUB * 16])
{
    // q8a1024 flat-grouped activation: flat_tile = token*tiles_per_row + k_blk locates
    // the tile's de-interleaved qs (128-aligned line) and ds within its super-block.
    const uint8_t* abytes = reinterpret_cast<const uint8_t*>(act);
    const int kchunk = (tid & 7) * 16;
    #pragma unroll
    for (int thalf = 0; thalf < N_SUB; ++thalf) {
        const int token = (tid >> 3) + thalf * 16;
        if (token < b_cnt) {
            const int64_t flat = (int64_t)(b_start + token) * tiles_per_row + k_blk;
            cp_async_ca16(&smem_A_i8[token][kchunk], abytes + q8a1024_qs_off(flat) + kchunk);
        } else {
            *reinterpret_cast<int4*>(&smem_A_i8[token][kchunk]) = make_int4(0, 0, 0, 0);
        }
    }
    // ds: one (scale, sum) per token (per-128). One thread per token; threads past the tile idle.
    // 128 threads cover up to 128 tokens, so one pass handles N_SUB*16 (N_SUB ≤ 8).
    {
        const int t2 = tid;   // token
        if (t2 < N_SUB * 16) {
            if (t2 < b_cnt) {
                const int64_t flat = (int64_t)(b_start + t2) * tiles_per_row + k_blk;
                cp_async_ca4(&smem_A_ds[t2], abytes + q8a1024_ds_off(flat));   // (scale,sum) fp16
            } else {
                smem_A_ds[t2] = make_half2(__float2half(1.f), __float2half(0.f));
            }
        }
    }
}

// Activation tile computed in place from F32 rows: `silu(proj[token, k·128 + ·])`, quantized
// straight into the shared tile — the operand `gr_silu_q8` would have written and
// load_q8a128_activations copied, with neither the producer's launch nor its round trip.
// One warp per token row of the tile (4 warps stride over N_SUB·16 tokens), lane `l` owning
// elements [4l, 4l+4) — `gr_silu_q8`'s mapping, through the same `fast_exp::silu` and the same
// `quantize_q8a128_tile`, so the bytes are its. `proj` rows are `proj_stride` floats apart
// (the gate columns of a wider projection) and 16-byte aligned. Synchronous stores, not
// cp.async: the caller's next barrier publishes them exactly as it publishes the copies.
// Rows ≥ b_cnt are zero-padded as the copying loader pads them.
//
// Every one of a warp's token rows is loaded before the first is quantized: the loads are
// independent and the quantize is a serial butterfly, so issuing them one token at a time
// paid a round trip per token where this pays one per tile.
template <int N_SUB = 1>
__device__ __forceinline__ void load_silu_f32_activations(
    const float* __restrict__ proj, int proj_stride, int b_start, int b_cnt, int k_blk,
    int tid, int sum_norm,
    int8_t smem_A_i8[N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[N_SUB * 16])
{
    constexpr int ROWS_PER_WARP = N_SUB * 16 / 4;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float4 v[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int token = warp + r * 4;
        if (token < b_cnt) {
            v[r] = *reinterpret_cast<const float4*>(
                proj + (int64_t)(b_start + token) * proj_stride + k_blk * K_TILE + lane * 4);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int token = warp + r * 4;
        if (token < b_cnt) {   // warp-uniform: the quantize's butterfly is full-warp
            const Q8a128Lane t = quantize_q8a128_tile(
                fast_exp::silu<float>(v[r].x), fast_exp::silu<float>(v[r].y),
                fast_exp::silu<float>(v[r].z), fast_exp::silu<float>(v[r].w), sum_norm);
            *reinterpret_cast<char4*>(&smem_A_i8[token][lane * 4]) = t.q;
            if (lane == 0) smem_A_ds[token] = t.ds;
        } else {
            *reinterpret_cast<char4*>(&smem_A_i8[token][lane * 4]) = make_char4(0, 0, 0, 0);
            if (lane == 0) smem_A_ds[token] = make_half2(__float2half(1.f), __float2half(0.f));
        }
    }
}

// The impl's one activation-tile load, from whichever source the instantiation reads:
// the q8a128 operand (copied, cp.async) or, with ACT_SILU, F32 rows quantized in place.
template <int N_SUB, bool ACT_SILU>
__device__ __forceinline__ void load_act_tile(
    const block_q8a128* __restrict__ act, const float* __restrict__ act_f32, int act_f32_stride,
    int b_start, int b_cnt, int k_blk, int tiles_per_row, int tid, int sum_norm,
    int8_t smem_A_i8[N_SUB * 16][KI8_STRIDE], half2 smem_A_ds[N_SUB * 16])
{
    if constexpr (ACT_SILU) {
        load_silu_f32_activations<N_SUB>(act_f32, act_f32_stride, b_start, b_cnt, k_blk, tid,
                                         sum_norm, smem_A_i8, smem_A_ds);
    } else {
        load_q8a128_activations<N_SUB>(act, b_start, b_cnt, k_blk, tiles_per_row, tid,
                                       smem_A_i8, smem_A_ds);
    }
}

// One (expert-tile, row-tile) on the INT8 tensor core: load_q8a128_activations + a per-warp
// weight ring, in 8-row chunks. At the two-stage pipeline every unsplit launch runs, each warp
// streams its row-group's chunks through ONE slot — the dequant drains the loaded chunk to
// registers, freeing the slot for the next prefetch, and the double-buffered activations
// provide the load/compute overlap. RING_I8 is that one slot, and sizes the weight smem of the
// two-stage entries; the deeper split-K pipeline sizes its own ring of STAGES−1 slots.
constexpr int RING_I8 = 1;

// Load this warp's own k1024 weight chunks for `k_blk` into `slot_ptr` with its 32 lanes: the
// RN consecutive 8-row groups from `warp_row_base`. `weights` is an array of k1024 chunks laid
// out [k_blk][row-group of 8], so a warp's RN chunks of one K tile are contiguous; each carries
// its quants AND its inline scales, so this single contiguous coalesced cp.async brings them in
// (no separate scale stream). Caller commits the group.
template <typename block_c_t, int RN = 1>
__device__ __forceinline__ void load_warp_chunk_int8(
    uint8_t* slot_ptr, const block_c_t* __restrict__ weights, int k_blk,
    int warp_row_base, int nrows, int lane)
{
    constexpr int N16 = RN * int8_chunk_bytes<block_c_t>::value / 16;   // 16B units in the chunks
    const uint8_t* sbytes;
    if constexpr (is_scale_separate<block_c_t>::value) {
        // KO k1024: one 8-row chunk, indexed [k_blk][row-group of 8].
        sbytes = reinterpret_cast<const uint8_t*>(&weights[(int64_t)k_blk * (nrows / 8) + warp_row_base / 8]);
    } else {
        // Legacy non-KO: 8 contiguous per-row blocks at [k_blk][row].
        sbytes = reinterpret_cast<const uint8_t*>(&weights[(int64_t)k_blk * nrows + warp_row_base]);
    }
    for (int i = lane; i < N16; i += WARP_SIZE_TC) {
        cp_async_cg16(slot_ptr + i * 16, sbytes + i * 16);   // quants (+ inline scales for KO)
    }
}

// MXFP4 runs the PER-SUB fold: the four per-32 subs already MMA separately, so
// each sub's int32 accumulator folds with its own E8M0 scale `2^(e_sub-128)`
// in FP32 — the dequant is then a pure codebook expansion (loader/mxfp4.cuh),
// with no per-element exponent-alignment arithmetic. The affine KO formats
// keep the single per-128 (d, m) fold.
template <typename T>
struct is_mxfp4_persub {
    static constexpr bool value = false;
};
template <>
struct is_mxfp4_persub<block_c_mxfp4_k1024> {
    static constexpr bool value = true;
};

// Blocks per SM the prefill-width int8 entries (dense mode-4, grouped mode-4 / mode-8 — RN > 1)
// are compiled for: `__launch_bounds__(128, wide_min_blocks<block_c_t, N_SUB, DENSE>())`. Three
// — 168 registers a thread — is the budget wherever ptxas fits the tile in it without a spill,
// which `ptxas -v` shows for every affine KO tile on sm_120 and every one but the mode-8 tile on
// sm_86 / sm_89 (8 bytes spilled). MXFP4's per-sub fold keeps four scales a fold row per chunk:
// its tiles spill at three on sm_86 / sm_89, and its dense tile (which also holds the PDL
// prefetch state) on sm_120 too. Those take two. The third block is worth keeping where it fits:
// DeepSeek's MXFP4 experts at mode-8 measured 6.53 ms at two blocks against 5.90 at three.
template <typename block_c_t, int N_SUB, bool DENSE>
constexpr int wide_min_blocks()
{
    constexpr bool mxfp4 = is_mxfp4_persub<block_c_t>::value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
    if (mxfp4 || N_SUB >= 8) {
        return 2;
    }
#else
    if (mxfp4 && DENSE) {
        return 2;
    }
#endif
    return 3;
}

// The affine per-128 fold of one K tile into one m16 token sub-tile's accumulator `acc`
// (frag_c layout: [0] token g / row rl, [1] token g / row rl+1, [2] token g+8 / row rl,
// [3] token g+8 / row rl+1). `C` is the tile's exact int32 dot product — the four k32
// sub-MMAs collapsed — `d0`/`d1` the (scale, min) of rows rl/rl+1, `a0`/`a1` the
// activation's stored (scale, Σ-field) of tokens g/g+8, and `sum_a`/`sum_b` the hoisted
// coefficients that rebuild Σx from that field, whichever convention wrote it
// (blocks.cuh):
//   raw    (0, 1): fmaf(a.x, 0, 1) = 1     → a.y, the stored Σx, EXACTLY
//   by-amax(127,0): fmaf(a.x, 127, 0)      → a.y · amax = Σx
// The raw arm is bit-identical to reading `a.y` directly, because multiplying by an
// exact 1.0f is exact in IEEE.
//
// Every int8 dense entry folds through this one function — the unsplit and split-K
// impl and the narrow entry (`dense_narrow.cuh`) — so the expression, and with it the
// compiler's FMA contraction of it, is the same everywhere: a tile's fold joins the
// running sum in one add, and the entries that sum per-tile folds in tile order from
// zero reproduce the unsplit kernel's accumulator bit for bit.
__device__ __forceinline__ void ko_affine_fold(
    float* acc, const int32_t C[4], float2 d0, float2 d1, float2 a0, float2 a1,
    float sum_a, float sum_b)
{
    const float a0_sum = a0.y * fmaf(a0.x, sum_a, sum_b);
    const float a1_sum = a1.y * fmaf(a1.x, sum_a, sum_b);
    acc[0] += d0.x * a0.x * (float)C[0] + d0.y * a0_sum;
    acc[1] += d1.x * a0.x * (float)C[1] + d1.y * a0_sum;
    acc[2] += d0.x * a1.x * (float)C[2] + d0.y * a1_sum;
    acc[3] += d1.x * a1.x * (float)C[3] + d1.y * a1_sum;
}

// The affine fold's K order. A tile's exact int32 dot product does not depend on the order its
// 128 K products are summed in, so the impl runs the tile as four m16n8k32 MMAs over PAIRED K
// segments rather than one per 32-K sub: MMA q's B operand is two dequanted registers of the same
// half of two subs — {b[s][h], b[s+1][h]} — and its A fragment the matching 16-byte segments,
// `kseg_lo(q)` for the first 16 K of the MMA and `kseg_lo(q) + 32` for the second. A format that
// unpacks a chunk with one wide shared-memory load per half (Q8's two LDS.128, `q8_K.cuh`) then
// hands the MMA a register pair the load already made adjacent, where the per-sub order
// {b[s][0], b[s][1]} needed a MOV per MMA to build it.
//   q  B operand                 A segments (bytes of the 128-K tile)
//   0  {b[0][0], b[1][0]}        0..15,  32..47
//   1  {b[2][0], b[3][0]}        64..79, 96..111
//   2  {b[0][1], b[1][1]}        16..31, 48..63
//   3  {b[2][1], b[3][1]}        80..95, 112..127
__host__ __device__ constexpr int kseg_lo(int q) { return (q & 1) * 64 + (q >> 1) * 16; }

// The A fragment of MMA q (see kseg_lo): ldmatrix.x4 over the m16 sub-tile at `tile_row0`
// (rows KI8_STRIDE bytes apart), matrices 0/1 (rows 0-7 / 8-15) from byte `klo` and matrices
// 2/3 from byte `khi` — `load_a_frag_m16k32_ldmatrix` with the second half's K offset free.
__device__ __forceinline__ void load_a_frag_kseg(
    uint32_t (&a)[4], const int8_t* tile_row0, int klo, int khi, int lane)
{
    const int tile_idx = lane >> 3;
    const int row = (tile_idx & 1) * 8 + (lane & 7);
    const int k = (tile_idx >> 1) ? khi : klo;
    fused_attn::ldmatrix_x4_b16(a, static_cast<uint32_t>(__cvta_generic_to_shared(
        tile_row0 + row * KI8_STRIDE + k)));
}
