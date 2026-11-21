// The grouped quantized matmul, which runs every MoE expert in one launch so
// that their tiles together fill the machine.
//
// `run_grouped_quantized_matmul` runs all expert tiles in a single launch. The
// `DECLARE_GROUPED` macros declare the grouped kernels for each quantized type,
// and the wide-Bm twins cover the KO rows 14 to 19.

// =============================================================================
// GROUPED MATMUL — all MoE experts in a single launch
// =============================================================================
// One kernel over a device-side (tile → expert, batch-slice) table. Collapses
// the per-expert segment loop above into a single cudaLaunchKernel whose grid
// (total_tiles × row_tiles) spans all experts at once, fixing both the launch
// count and the per-expert occupancy starvation. The tile tables + weight
// pointer array are built and uploaded by the caller (candle-core).

#define DECLARE_GROUPED(name) \
    extern "C" __global__ void name##_grouped( \
        const void*, const void*, const void*, const void*, \
        const void*, void*, int, int, int, int, int);
#define DECLARE_GROUPED3(base) \
    DECLARE_GROUPED(base##_f16) DECLARE_GROUPED(base##_bf16) DECLARE_GROUPED(base##_f32)

DECLARE_GROUPED3(q4_0)  DECLARE_GROUPED3(q4_1)  DECLARE_GROUPED3(q5_0)
DECLARE_GROUPED3(q5_1)  DECLARE_GROUPED3(q8_0)  DECLARE_GROUPED3(q2_K)
DECLARE_GROUPED3(q3_k)  DECLARE_GROUPED3(q4_k)  DECLARE_GROUPED3(q5_k)
DECLARE_GROUPED3(q6_k)  DECLARE_GROUPED3(q8_1)  DECLARE_GROUPED3(q8_k)
DECLARE_GROUPED3(q_awq) DECLARE_GROUPED3(q_awq_g64)

#undef DECLARE_GROUPED3
#undef DECLARE_GROUPED

#define GROUPED_ROW(base) \
    { (void*)base##_f16_grouped, (void*)base##_bf16_grouped, (void*)base##_f32_grouped }

// [qtype_kernel_row][ytype] — same ordering as the kernels[14][3] table above.
static void* grouped_kernels[14][3] = {
    GROUPED_ROW(q4_0),  GROUPED_ROW(q4_1),  GROUPED_ROW(q5_0),  GROUPED_ROW(q5_1),
    GROUPED_ROW(q8_0),  GROUPED_ROW(q2_K),  GROUPED_ROW(q3_k),  GROUPED_ROW(q4_k),
    GROUPED_ROW(q5_k),  GROUPED_ROW(q6_k),  GROUPED_ROW(q8_1),  GROUPED_ROW(q8_k),
    GROUPED_ROW(q_awq), GROUPED_ROW(q_awq_g64),
};

#undef GROUPED_ROW

// q8a128 TC path — INT8-MMA: q8a128 activations (raw int8 qs) are multiplied
// against Q4_K weights (raw 4-bit nibbles) on the m16n8k32 int8 tensor core, with
// the deferred-scale fold (d_w·s_a·C + m_w·Σx) applied post-MMA and F32 output.
// See grouped_tc_int8 in kernel.cuh. Same launch ABI as the FP grouped kernels
// (one block_q8a128 activation pointer). Registered for ytype==3 →
// grouped_kernels_int8[row].
extern "C" __global__ void q4_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q8_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q4_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q4_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q5_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q5_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q5_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q6_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q3_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q2_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q8_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q8_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q_awq_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q_awq_g64_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
// KO byte-permuted twins (rows 14-17).
extern "C" __global__ void q4_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q5_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q6_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
extern "C" __global__ void q8_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
// MXFP4_KO per-sub twin (row 18).
extern "C" __global__ void mxfp4_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
// Q2_KO 2-bit affine twin (row 19).
extern "C" __global__ void q2_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);
// Q3_KO 3-bit affine twin (row 20).
extern "C" __global__ void q3_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int, int);

// Wide-Bm (mode-4 / mode-8) grouped twins — KO rows only (14-19): the int8
// impl is KO-exclusive, and the wide tiles exist for the routed-expert
// PREFILL regime where those are the only formats in play. The host requests
// a wide mode only for KO dtypes (cuda.rs), so the null rows are never asked
// for; a null lookup returns without launching rather than mis-launching.
#define DECLARE_GROUPED_WIDE(name) \
    extern "C" __global__ void name##_grouped_m4( \
        const void*, const void*, const void*, const void*, const void*, \
        void*, int, int, int, int, int); \
    extern "C" __global__ void name##_grouped_m8( \
        const void*, const void*, const void*, const void*, const void*, \
        void*, int, int, int, int, int);
DECLARE_GROUPED_WIDE(q4_ko_int8_f32)
DECLARE_GROUPED_WIDE(q5_ko_int8_f32)
DECLARE_GROUPED_WIDE(q6_ko_int8_f32)
DECLARE_GROUPED_WIDE(q8_ko_int8_f32)
DECLARE_GROUPED_WIDE(mxfp4_ko_int8_f32)
DECLARE_GROUPED_WIDE(q2_ko_int8_f32)
DECLARE_GROUPED_WIDE(q3_ko_int8_f32)
#undef DECLARE_GROUPED_WIDE

static void* grouped_kernels_int8_m4[21] = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    (void*)q4_ko_int8_f32_grouped_m4,    // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped_m4,    // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped_m4,    // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped_m4,    // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped_m4, // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped_m4,    // 19  q2_KO
    (void*)q3_ko_int8_f32_grouped_m4,    // 20  q3_KO
};
static void* grouped_kernels_int8_m8[21] = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    (void*)q4_ko_int8_f32_grouped_m8,    // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped_m8,    // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped_m8,    // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped_m8,    // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped_m8, // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped_m8,    // 19  q2_KO
    (void*)q3_ko_int8_f32_grouped_m8,    // 20  q3_KO
};

// [qtype_kernel_row] — same row ordering as grouped_kernels above.
static void* grouped_kernels_int8[21] = {
    (void*)q4_0_int8_f32_grouped,      // 0   q4_0
    (void*)q4_1_int8_f32_grouped,      // 1   q4_1
    (void*)q5_0_int8_f32_grouped,      // 2   q5_0
    (void*)q5_1_int8_f32_grouped,      // 3   q5_1
    (void*)q8_0_int8_f32_grouped,      // 4   q8_0
    (void*)q2_k_int8_f32_grouped,      // 5   q2_K
    (void*)q3_k_int8_f32_grouped,      // 6   q3_K
    (void*)q4_k_int8_f32_grouped,      // 7   q4_K
    (void*)q5_k_int8_f32_grouped,      // 8   q5_K
    (void*)q6_k_int8_f32_grouped,      // 9   q6_K
    (void*)q8_1_int8_f32_grouped,      // 10  q8_1
    (void*)q8_k_int8_f32_grouped,      // 11  q8_K
    (void*)q_awq_int8_f32_grouped,     // 12  q_awq
    (void*)q_awq_g64_int8_f32_grouped, // 13  q_awq_g64
    (void*)q4_ko_int8_f32_grouped,     // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped,     // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped,     // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped,     // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped,  // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped,     // 19  q2_KO
    (void*)q3_ko_int8_f32_grouped,     // 20  q3_KO
};

/// Single-launch grouped matmul over all expert tiles.
///
/// All pointer arguments are DEVICE pointers, prepared by the caller:
/// - weight_ptrs:  uint64_t[num_experts] — each expert's K/128 weight pointer
/// - tile_expert:  int[num_tiles] — owning expert id per tile
/// - tile_b_start: int[num_tiles] — stacked-batch start row per tile
/// - tile_b_cnt:   int[num_tiles] — tokens in the tile (1..16)
/// - vy:           stacked activations [total_batch, K]
/// - dst:          stacked output [total_batch, N]
///
/// ncols_x = K, nrows_x = N, y_stride = K, dst_stride = N.
///
/// Returns a QMM_* status. No tiles is QMM_OK: there is nothing to compute.
extern "C" int run_grouped_quantized_matmul(
    const void* weight_ptrs,
    const void* tile_expert,
    const void* tile_b_start,
    const void* tile_b_cnt,
    const void* vy,
    void* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t y_stride,
    int32_t dst_stride,
    int32_t num_tiles,
    int32_t qtype,
    int32_t ytype,
    int32_t n_sub,     // int8 token-tile width / 16: 2 (Bm 32), 4 (Bm 64), 8 (Bm 128)
    int32_t row_fast,  // grid axis order: 1 = row tiles fast (see kernel entry doc)
    int32_t sum_norm,  // q8a128 Σx convention (`SumScale::as_code()`); ytype==3 only
    // A launch over a live expert table (see `MoeLive` in kernel.cuh): the grid
    // gains its worker row. Null for every other launch. ytype==3 only.
    const MoeLive* live,
    // The caller's stream — the device handle's own, so the launch is ordered
    // with every other op the handle issues.
    void* stream)
{
    int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < 0) {
        return QMM_BAD_QTYPE;
    }
    if (ytype < 0 || ytype > 3) {
        return QMM_BAD_YTYPE;
    }
    if (num_tiles <= 0) {
        return QMM_OK;
    }
    // KO byte-permuted formats (rows >= 14) have INT8 grouped kernels only; the FP
    // `grouped_kernels` table is sized to the 14 base formats. Reject a misrouted
    // FP (ytype != 3) call on a KO row before indexing out of bounds.
    if (ytype != 3 && kernel_row >= 14) {
        return QMM_BAD_QTYPE;
    }
    // Activation input selected by `ytype`, same 10-arg launch ABI either way:
    //   0/1/2 (F16/BF16/F32) → FP activations → FP16-MMA grouped kernel.
    //   3 (q8a128)           → q8 activations (`vy` is a block_q8a128 buffer) →
    //                          INT8-MMA grouped kernel (raw int8 × Q4_K nibbles,
    //                          deferred-scale fold, F32 output). TC-only — callers
    //                          only pass q8a128 when tensor cores exist.
    // `n_sub` selects the int8 tile width — the caller sized the tile tables
    // to 16·n_sub, so a mode without a kernel MUST refuse rather than launch a
    // narrower kernel that would drop the tiles' upper rows.
    void* kfn;
    // Rows per block: 32 for the FP and mode-2 tiles, 32·RN for the wide int8 tiles
    // (`INSTANTIATE_KERNEL_GROUPED_INT8_M4/_M8`: RN = 4 / 2).
    int row_div = 32;
    if (ytype == 3) {
        switch (n_sub) {
            case 2: kfn = grouped_kernels_int8[kernel_row]; break;
            case 4: kfn = grouped_kernels_int8_m4[kernel_row]; row_div = 128; break;
            case 8: kfn = grouped_kernels_int8_m8[kernel_row]; row_div = 64; break;
            default: return QMM_BAD_TILE_MODE;
        }
    } else {
        kfn = grouped_kernels[kernel_row][ytype];
    }
    if (kfn == nullptr) {
        return QMM_NO_KERNEL;
    }

    const int row_tiles = (nrows_x + row_div - 1) / row_div;
    // Grid axis order is the caller's per-launch choice (see the kernel entry doc):
    // row_fast puts row tiles on x so consecutive blocks share one token tile's
    // activation slab in L2; token-fast is the transpose. Whichever axis carries
    // the token tiles is bounded (≤ a few thousand, far under the 65535 y limit).
    dim3 grid = row_fast ? dim3(row_tiles, num_tiles, 1) : dim3(num_tiles, row_tiles, 1);
    if (ytype == 3 && live != nullptr) {
        // Live: row tiles on x, the workers' rows ahead of the tiles (see "A live
        // expert table" in kernel.cuh).
        const int worker_rows = (live->workers + row_tiles - 1) / row_tiles;
        grid = dim3(row_tiles, num_tiles + worker_rows, 1);
    }
    dim3 block(WARP_SIZE, 4, 1);  // 128 threads (4 warps × 32)

    // The int8 grouped kernels take one argument the FP ones do not: the q8a128
    // Σx convention. The FP globals have no q8a128 header to interpret, so their
    // signature is unchanged and their launch must NOT name it — `args[]` is
    // positional and matched against the kernel's own parameter list.
    if (ytype == 3) {
        MoeLive live_v = live != nullptr ? *live : MoeLive{};
        int row_fast_v = live != nullptr ? 1 : row_fast;
        void* args[] = {
            (void*)&weight_ptrs, (void*)&tile_expert, (void*)&tile_b_start, (void*)&tile_b_cnt,
            (void*)&vy, (void*)&dst,
            (void*)&ncols_x, (void*)&nrows_x, (void*)&y_stride, (void*)&dst_stride,
            (void*)&row_fast_v, (void*)&sum_norm, (void*)&live_v,
        };
        if (const cudaError_t e = cudaLaunchKernel(kfn, grid, block, args, 0, (cudaStream_t)stream);
            e != cudaSuccess) {
            return qmm_launch_failed(e, "grouped int8: launch");
        }
    } else {
        void* args[] = {
            (void*)&weight_ptrs, (void*)&tile_expert, (void*)&tile_b_start, (void*)&tile_b_cnt,
            (void*)&vy, (void*)&dst,
            (void*)&ncols_x, (void*)&nrows_x, (void*)&y_stride, (void*)&dst_stride,
            (void*)&row_fast,
        };
        if (const cudaError_t e = cudaLaunchKernel(kfn, grid, block, args, 0, (cudaStream_t)stream);
            e != cudaSuccess) {
            return qmm_launch_failed(e, "grouped: launch");
        }
    }
    return QMM_OK;
}
