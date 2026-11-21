// The dense INT8 matmul family: the q8a128 kernel tables, the split-K and
// narrow launches, and the unified quantized matmul dispatcher.
//
// `DECL_DENSE_INT8` declares the q8a128 INT8 dense kernels that the regular
// QMatMul uses. `run_dense_int8_splitk` and `run_dense_int8_narrow` launch the
// split-K and narrow variants, and `run_quantized_matmul` is the extern entry
// point that dispatches on qtype and ytype with greedy batch decomposition.

// =============================================================================
// Unified dispatcher function with TABLE-DRIVEN batch decomposition
// =============================================================================
// qtype: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K
// ytype: 0=F16, 1=BF16, 2=F32
//
// Dispatch flow (SM80+):
// 1. TC path for ALL batch >= 1 (tc16 for 1-31, tc32 for 32+). TC beat the
//    CUDA-core GEMV kernels in every measured case, batch 1-2 included.
// 2. CUDA-core GEMV (s1..s8) only on pre-SM80 GPUs without tensor cores.
// (MoE callers should prefer run_grouped_quantized_matmul, which fuses all
//  experts into one launch instead of looping this per-expert.)

// Segment descriptor: one per expert (MoE) or one total (non-MoE)
typedef struct {
    const void* weights;     // Device pointer to quantized weight data
    int32_t     batch_count; // Batches in this segment (greedy boundary)
} vx_segment_t;

// q8a128 INT8 DENSE kernels — the regular (non-MoE) QMatMul. Same INT8 m16n8k32
// core as the grouped kernels, single weight, implicit tile schedule (blockIdx.x →
// batch slice). Launched from run_quantized_matmul on ytype==3. Row ordering
// matches qtype_to_matmul_kernel_index (same as grouped_kernels_int8).
//
// Three output dtypes per format: the accumulator is F32 in registers either way,
// so the tag only names the width of the final store. A dense projection is read
// back at the model's activation dtype, and storing narrow there is what removes
// the separate cast launch (and halves the bytes the result occupies on the wave
// span). The grouped/MoE table stays F32-only — see grouped_kernels_int8.
#define DECL_DENSE_INT8(name) \
    extern "C" __global__ void name(const void*, const void*, void*, int, int, int, int, int);

#define DECL_DENSE_INT8_ALL(base) \
    DECL_DENSE_INT8(base##_f16_dense) \
    DECL_DENSE_INT8(base##_bf16_dense) \
    DECL_DENSE_INT8(base##_f32_dense)

#define DECL_DENSE_INT8_M2_ALL(base) \
    DECL_DENSE_INT8(base##_f16_dense_m2) \
    DECL_DENSE_INT8(base##_bf16_dense_m2) \
    DECL_DENSE_INT8(base##_f32_dense_m2) \
    DECL_DENSE_INT8(base##_f16_dense_m4) \
    DECL_DENSE_INT8(base##_bf16_dense_m4) \
    DECL_DENSE_INT8(base##_f32_dense_m4)

DECL_DENSE_INT8_ALL(q4_0_int8)
DECL_DENSE_INT8_ALL(q4_1_int8)
DECL_DENSE_INT8_ALL(q5_0_int8)
DECL_DENSE_INT8_ALL(q5_1_int8)
DECL_DENSE_INT8_ALL(q8_0_int8)
DECL_DENSE_INT8_ALL(q2_k_int8)
DECL_DENSE_INT8_ALL(q3_k_int8)
DECL_DENSE_INT8_ALL(q4_k_int8)
DECL_DENSE_INT8_ALL(q5_k_int8)
DECL_DENSE_INT8_ALL(q6_k_int8)
DECL_DENSE_INT8_ALL(q8_1_int8)
DECL_DENSE_INT8_ALL(q8_k_int8)
DECL_DENSE_INT8_ALL(q_awq_int8)
DECL_DENSE_INT8_ALL(q_awq_g64_int8)
// KO byte-permuted twins (rows 14-17).
DECL_DENSE_INT8_ALL(q4_ko_int8)
DECL_DENSE_INT8_ALL(q5_ko_int8)
DECL_DENSE_INT8_ALL(q6_ko_int8)
DECL_DENSE_INT8_ALL(q8_ko_int8)
// MXFP4_KO per-sub twin (row 18).
DECL_DENSE_INT8_ALL(mxfp4_ko_int8)
// Q2_KO 2-bit affine twin (row 19).
DECL_DENSE_INT8_ALL(q2_ko_int8)
// Q3_KO 3-bit affine twin (row 20).
DECL_DENSE_INT8_ALL(q3_ko_int8)

// q8a128 mode-1 → mode-2 (Bm=32 weight-reuse) → mode-4 (Bm=64 × 128-row prefill tile). The
// DENSE tiling is decided in Rust (`int8_matmul_mode::q8a128_dense_tile`, a grid-fill rule)
// and passed in as `tile_mode`. Mode-2 and mode-4 exist for the KO formats only — they are the
// ones a prefill wave runs.
DECL_DENSE_INT8_M2_ALL(q4_ko_int8)
DECL_DENSE_INT8_M2_ALL(q5_ko_int8)
DECL_DENSE_INT8_M2_ALL(q6_ko_int8)
DECL_DENSE_INT8_M2_ALL(q8_ko_int8)
DECL_DENSE_INT8_M2_ALL(mxfp4_ko_int8)
DECL_DENSE_INT8_M2_ALL(q2_ko_int8)
DECL_DENSE_INT8_M2_ALL(q3_ko_int8)

// Split-K twins of the KO dense kernels — decode width, narrow N (see run_dense_int8_splitk).
#define DECL_DENSE_INT8_SK(name) \
    extern "C" __global__ void name(const SplitKSegs, const void*, void*, int, int, int, int, \
                                    float*, unsigned int*);
#define DECL_DENSE_INT8_SK_ALL(base) \
    DECL_DENSE_INT8_SK(base##_f16_dense_sk) \
    DECL_DENSE_INT8_SK(base##_bf16_dense_sk) \
    DECL_DENSE_INT8_SK(base##_f32_dense_sk)
DECL_DENSE_INT8_SK_ALL(q4_ko_int8)
DECL_DENSE_INT8_SK_ALL(q5_ko_int8)
DECL_DENSE_INT8_SK_ALL(q6_ko_int8)
DECL_DENSE_INT8_SK_ALL(q8_ko_int8)
DECL_DENSE_INT8_SK_ALL(q2_ko_int8)
DECL_DENSE_INT8_SK_ALL(q3_ko_int8)
#undef DECL_DENSE_INT8_SK_ALL
#undef DECL_DENSE_INT8_SK

// Narrow twins of the KO dense kernels — decode width, K across a block's warps (see
// run_dense_int8_narrow).
#define DECL_DENSE_INT8_NW(name) \
    extern "C" __global__ void name(const SplitKSegs, const void*, void*, int, int, int, int);
#define DECL_DENSE_INT8_NW_ALL(base) \
    DECL_DENSE_INT8_NW(base##_f16_dense_nw) \
    DECL_DENSE_INT8_NW(base##_bf16_dense_nw) \
    DECL_DENSE_INT8_NW(base##_f32_dense_nw)
DECL_DENSE_INT8_NW_ALL(q4_ko_int8)
DECL_DENSE_INT8_NW_ALL(q5_ko_int8)
DECL_DENSE_INT8_NW_ALL(q6_ko_int8)
DECL_DENSE_INT8_NW_ALL(q8_ko_int8)
DECL_DENSE_INT8_NW_ALL(q2_ko_int8)
DECL_DENSE_INT8_NW_ALL(q3_ko_int8)
#undef DECL_DENSE_INT8_NW_ALL
#undef DECL_DENSE_INT8_NW

#undef DECL_DENSE_INT8_M2_ALL
#undef DECL_DENSE_INT8_ALL
#undef DECL_DENSE_INT8

// One row per output dtype, in OutDType order (0=F16, 1=BF16, 2=F32).
#define DENSE_INT8_ROW(tag) { \
    (void*)q4_0_int8_##tag##_dense,      /* 0   q4_0 */ \
    (void*)q4_1_int8_##tag##_dense,      /* 1   q4_1 */ \
    (void*)q5_0_int8_##tag##_dense,      /* 2   q5_0 */ \
    (void*)q5_1_int8_##tag##_dense,      /* 3   q5_1 */ \
    (void*)q8_0_int8_##tag##_dense,      /* 4   q8_0 */ \
    (void*)q2_k_int8_##tag##_dense,      /* 5   q2_K */ \
    (void*)q3_k_int8_##tag##_dense,      /* 6   q3_K */ \
    (void*)q4_k_int8_##tag##_dense,      /* 7   q4_K */ \
    (void*)q5_k_int8_##tag##_dense,      /* 8   q5_K */ \
    (void*)q6_k_int8_##tag##_dense,      /* 9   q6_K */ \
    (void*)q8_1_int8_##tag##_dense,      /* 10  q8_1 */ \
    (void*)q8_k_int8_##tag##_dense,      /* 11  q8_K */ \
    (void*)q_awq_int8_##tag##_dense,     /* 12  q_awq */ \
    (void*)q_awq_g64_int8_##tag##_dense, /* 13  q_awq_g64 */ \
    (void*)q4_ko_int8_##tag##_dense,     /* 14  q4_KO */ \
    (void*)q5_ko_int8_##tag##_dense,     /* 15  q5_KO */ \
    (void*)q6_ko_int8_##tag##_dense,     /* 16  q6_KO */ \
    (void*)q8_ko_int8_##tag##_dense,     /* 17  q8_KO */ \
    (void*)mxfp4_ko_int8_##tag##_dense,  /* 18  mxfp4_KO */ \
    (void*)q2_ko_int8_##tag##_dense,     /* 19  q2_KO */ \
    (void*)q3_ko_int8_##tag##_dense,     /* 20  q3_KO */ \
}

// Indexed by [out_dtype][kernel_row].
static void* dense_kernels_int8[3][21] = {
    DENSE_INT8_ROW(f16),
    DENSE_INT8_ROW(bf16),
    DENSE_INT8_ROW(f32),
};
#undef DENSE_INT8_ROW

// Indexed by [out_dtype][kernel_row - 14]:
// Q4_KO=14, Q5_KO=15, Q6_KO=16, Q8_KO=17, MXFP4_KO=18, Q2_KO=19, Q3_KO=20.
#define DENSE_INT8_M2_ROW(tag) { \
    (void*)q4_ko_int8_##tag##_dense_m2, \
    (void*)q5_ko_int8_##tag##_dense_m2, \
    (void*)q6_ko_int8_##tag##_dense_m2, \
    (void*)q8_ko_int8_##tag##_dense_m2, \
    (void*)mxfp4_ko_int8_##tag##_dense_m2, \
    (void*)q2_ko_int8_##tag##_dense_m2, \
    (void*)q3_ko_int8_##tag##_dense_m2, \
}

static void* dense_kernels_int8_m2[3][7] = {
    DENSE_INT8_M2_ROW(f16),
    DENSE_INT8_M2_ROW(bf16),
    DENSE_INT8_M2_ROW(f32),
};
#undef DENSE_INT8_M2_ROW

// Indexed by [out_dtype][kernel_row - KO_ROW_FIRST], the mode-2 table's order: the mode-4
// prefill tile (Bm=64 × 128 rows, `INSTANTIATE_KERNEL_DENSE_INT8_M4`).
#define DENSE_INT8_M4_ROW(tag) { \
    (void*)q4_ko_int8_##tag##_dense_m4, \
    (void*)q5_ko_int8_##tag##_dense_m4, \
    (void*)q6_ko_int8_##tag##_dense_m4, \
    (void*)q8_ko_int8_##tag##_dense_m4, \
    (void*)mxfp4_ko_int8_##tag##_dense_m4, \
    (void*)q2_ko_int8_##tag##_dense_m4, \
    (void*)q3_ko_int8_##tag##_dense_m4, \
}
static void* dense_kernels_int8_m4[3][7] = {
    DENSE_INT8_M4_ROW(f16),
    DENSE_INT8_M4_ROW(bf16),
    DENSE_INT8_M4_ROW(f32),
};
#undef DENSE_INT8_M4_ROW
static_assert(
    sizeof(dense_kernels_int8_m4[0]) == sizeof(dense_kernels_int8_m2[0]),
    "every KO format with a mode-2 tile must have a mode-4 tile");

// First kernel row with a mode-2 (Bm=32 weight-reuse) twin, and how many there are.
//
// **Derived from the m2 table's own width, not written down.** The count used to be a literal
// `kernel_row <= 19` in the dense dispatch, and a KO format added past it stayed silently on
// mode-1: the answer is still correct, the format simply never gets the prefill path, and
// nothing anywhere reports it. The static_assert ties the two tables together so a row added
// to one without the other fails the build instead.
constexpr int KO_ROW_FIRST = 14;
constexpr int KO_ROW_COUNT =
    (int)(sizeof(dense_kernels_int8_m2[0]) / sizeof(dense_kernels_int8_m2[0][0]));
static_assert(
    KO_ROW_FIRST + KO_ROW_COUNT
        == (int)(sizeof(dense_kernels_int8[0]) / sizeof(dense_kernels_int8[0][0])),
    "every dense kernel row from KO_ROW_FIRST up must have a mode-2 twin");

// Indexed by [out_dtype][kernel_row - KO_ROW_FIRST], the mode-2 table's order. MXFP4 has no
// split twin: its per-sub fold accumulates straight into the sum, which per-tile partials
// cannot reproduce, so a split MXFP4 launch is refused as QMM_BAD_SPLIT.
#define DENSE_INT8_SK_ROW(tag) { \
    (void*)q4_ko_int8_##tag##_dense_sk, \
    (void*)q5_ko_int8_##tag##_dense_sk, \
    (void*)q6_ko_int8_##tag##_dense_sk, \
    (void*)q8_ko_int8_##tag##_dense_sk, \
    nullptr, /* mxfp4: never splits */ \
    (void*)q2_ko_int8_##tag##_dense_sk, \
    (void*)q3_ko_int8_##tag##_dense_sk, \
}
static void* dense_kernels_int8_sk[3][7] = {
    DENSE_INT8_SK_ROW(f16),
    DENSE_INT8_SK_ROW(bf16),
    DENSE_INT8_SK_ROW(f32),
};
#undef DENSE_INT8_SK_ROW
static_assert(
    (int)(sizeof(dense_kernels_int8_sk[0]) / sizeof(dense_kernels_int8_sk[0][0])) == KO_ROW_COUNT,
    "every KO format must have a split-K entry (null where it never splits)");

// Indexed by [out_dtype][kernel_row - KO_ROW_FIRST], the mode-2 table's order. MXFP4 has no
// narrow twin, for the reason it has no split one.
#define DENSE_INT8_NW_ROW(tag) { \
    (void*)q4_ko_int8_##tag##_dense_nw, \
    (void*)q5_ko_int8_##tag##_dense_nw, \
    (void*)q6_ko_int8_##tag##_dense_nw, \
    (void*)q8_ko_int8_##tag##_dense_nw, \
    nullptr, /* mxfp4: never runs narrow */ \
    (void*)q2_ko_int8_##tag##_dense_nw, \
    (void*)q3_ko_int8_##tag##_dense_nw, \
}
static void* dense_kernels_int8_nw[3][7] = {
    DENSE_INT8_NW_ROW(f16),
    DENSE_INT8_NW_ROW(bf16),
    DENSE_INT8_NW_ROW(f32),
};
#undef DENSE_INT8_NW_ROW
static_assert(
    (int)(sizeof(dense_kernels_int8_nw[0]) / sizeof(dense_kernels_int8_nw[0][0])) == KO_ROW_COUNT,
    "every KO format must have a narrow entry (null where it never runs narrow)");

// The 8-row weight chunk of each narrow entry's format, in the table's order — what its
// shared-memory ring is sized in (`narrow_layout.cuh`). Zero where there is no entry.
static const int dense_narrow_chunk_bytes[7] = {
    (int)sizeof(block_c_q4_KO_k1024),
    (int)sizeof(block_c_q5_KO_k1024),
    (int)sizeof(block_c_q6_KO_k1024),
    (int)sizeof(block_c_q8_KO_k1024),
    0,
    (int)sizeof(block_c_q2_KO_k1024),
    (int)sizeof(block_c_q3_KO_k1024),
};

// Whether each narrow kernel has been opted into NARROW_SMEM_CAP of dynamic shared memory.
// Set once per kernel, before its first launch; the attribute is idempotent, so two threads
// that both find it unset both set the same value.
static std::atomic<bool> dense_narrow_smem_opted[3][7] = {};

// Split-K int8 dense matmul: the mode-1 tile grid (ceil(M/16) × ΣN/32) times `splits`
// slices of K, over `num_segs` weight segments of one KO format sharing the activation
// (`splitk_segs.cuh`) — `weights[s]` of `nrows[s]` rows, each a multiple of 32, written
// to consecutive column ranges of the `ΣN`-wide output. Rust decides `splits`
// (q8a128_dense_k_splits) and owns the scratch: `ws` holds K-tiles × M × ΣN F32 partials
// (one per K tile), `counters` one zeroed u32 per (batch tile, row tile), returned to
// zero by the kernel itself. Both must be stream-ordered with every other split-K
// launch that shares them.
extern "C" int run_dense_int8_splitk(
    const void* const* weights,
    const int32_t* nrows,
    int32_t num_segs,
    const void* vy,
    void* dst,
    int32_t ncols_x,      // K
    int32_t total_batch,  // M
    int32_t qtype,
    int32_t out_dtype,
    int32_t sum_norm,
    int32_t splits,
    float* ws,
    unsigned int* counters,
    void* stream
) {
    if (out_dtype < 0 || out_dtype > 2) {
        return QMM_BAD_OUT_DTYPE;
    }
    const int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < KO_ROW_FIRST || kernel_row - KO_ROW_FIRST >= KO_ROW_COUNT) {
        return QMM_BAD_QTYPE;
    }
    void* kfn = dense_kernels_int8_sk[out_dtype][kernel_row - KO_ROW_FIRST];
    // A null entry is a format that never splits; a depth outside [2, K-tiles] would leave
    // K tiles unwalked or run blocks with nothing to sum.
    if (kfn == nullptr || splits < 2 || splits > ncols_x / 128) {
        return QMM_BAD_SPLIT;
    }
    if (num_segs < 1) {
        return QMM_NO_SEGMENTS;
    }
    if (num_segs > SPLITK_MAX_SEGS) {
        return QMM_NO_KERNEL;
    }
    SplitKSegs segs;
    int tile = 0, col = 0;
    for (int s = 0; s < num_segs; ++s) {
        if (nrows[s] <= 0 || nrows[s] % 32 != 0) {
            return QMM_NO_KERNEL;
        }
        segs.w[s] = weights[s];
        segs.tile_start[s] = tile;
        segs.n[s] = nrows[s];
        segs.col_off[s] = col;
        tile += nrows[s] / 32;
        col += nrows[s];
    }
    segs.tile_start[num_segs] = tile;
    segs.num = num_segs;
    const int batch_tiles = (total_batch + 15) / 16;
    dim3 grid(batch_tiles, tile, splits);
    dim3 block(WARP_SIZE, 4, 1);
    const int dst_stride = col;
    void* args[] = {
        (void*)&segs, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&total_batch,
        (void*)&dst_stride, (void*)&sum_norm, (void*)&ws, (void*)&counters,
    };
    // The split-K entry waits (`pdl_wait`) before its activation, partials and counters.
    if (const cudaError_t e = launch_pdl(kfn, grid, block, args, 0, (cudaStream_t)stream);
        e != cudaSuccess) {
        return qmm_launch_failed(e, "dense split-K: launch");
    }
    return QMM_OK;
}

// Narrow int8 dense matmul: one block per 8-row output tile of ΣN, `warps` warps walking K
// between them and summing in shared memory (grouped_tc::quantized_matmul_dense_narrow_entry_int8),
// over `num_segs` weight segments of one KO format sharing the activation (`splitk_segs.cuh`,
// tiles of 8 rows) — `weights[s]` of `nrows[s]` rows, each a multiple of 32, written to
// consecutive column ranges of the `ΣN`-wide output. At most NARROW_MAX_ROWS tokens. No
// scratch: the reduction never leaves the block. Rust decides when it runs and at how many
// warps (`int8_split_k::q8a128_dense_plan`).
extern "C" int run_dense_int8_narrow(
    const void* const* weights,
    const int32_t* nrows,
    int32_t num_segs,
    const void* vy,
    void* dst,
    int32_t ncols_x,      // K
    int32_t total_batch,  // M
    int32_t qtype,
    int32_t out_dtype,
    int32_t sum_norm,
    int32_t warps,
    void* stream
) {
    if (out_dtype < 0 || out_dtype > 2) {
        return QMM_BAD_OUT_DTYPE;
    }
    const int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < KO_ROW_FIRST || kernel_row - KO_ROW_FIRST >= KO_ROW_COUNT) {
        return QMM_BAD_QTYPE;
    }
    const int fmt = kernel_row - KO_ROW_FIRST;
    void* kfn = dense_kernels_int8_nw[out_dtype][fmt];
    const int k_tiles = ncols_x / 128;
    // A null entry is a format that never runs narrow. One token tile of at most
    // NARROW_MAX_ROWS rows, whole K tiles, a warp count the launch bounds admit.
    if (kfn == nullptr || total_batch < 1 || total_batch > NARROW_MAX_ROWS
        || ncols_x <= 0 || ncols_x % 128 != 0 || warps < 1 || warps > NARROW_MAX_WARPS) {
        return QMM_BAD_SPLIT;
    }
    const int smem = narrow_smem_bytes(dense_narrow_chunk_bytes[fmt], k_tiles, warps);
    if (smem > NARROW_SMEM_CAP) {
        return QMM_BAD_SPLIT;
    }
    if (num_segs < 1) {
        return QMM_NO_SEGMENTS;
    }
    if (num_segs > SPLITK_MAX_SEGS) {
        return QMM_NO_KERNEL;
    }
    SplitKSegs segs;
    int tile = 0, col = 0;
    for (int s = 0; s < num_segs; ++s) {
        if (nrows[s] <= 0 || nrows[s] % 32 != 0) {
            return QMM_NO_KERNEL;
        }
        segs.w[s] = weights[s];
        segs.tile_start[s] = tile;
        segs.n[s] = nrows[s];
        segs.col_off[s] = col;
        tile += nrows[s] / 8;
        col += nrows[s];
    }
    segs.tile_start[num_segs] = tile;
    segs.num = num_segs;
    if (!dense_narrow_smem_opted[out_dtype][fmt].load(std::memory_order_acquire)) {
        if (const cudaError_t e = cudaFuncSetAttribute(
                kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, NARROW_SMEM_CAP);
            e != cudaSuccess) {
            return qmm_launch_failed(e, "dense narrow: cudaFuncSetAttribute");
        }
        dense_narrow_smem_opted[out_dtype][fmt].store(true, std::memory_order_release);
    }
    dim3 grid(1, tile, 1);
    dim3 block(WARP_SIZE, warps, 1);
    const int dst_stride = col;
    void* args[] = {
        (void*)&segs, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&total_batch,
        (void*)&dst_stride, (void*)&sum_norm,
    };
    // The narrow entry waits (`pdl_wait`) before its activation.
    if (const cudaError_t e = launch_pdl(kfn, grid, block, args, (size_t)smem, (cudaStream_t)stream);
        e != cudaSuccess) {
        return qmm_launch_failed(e, "dense narrow: launch");
    }
    return QMM_OK;
}

// The fused-activation dense matmul: `dst [M, N] = silu(proj[:, 0..K]) · Wᵀ` for a Q8_KO
// weight and an F32 output, the activation quantized by the kernel's own loader
// (grouped_tc::quantized_matmul_dense_silu_entry_int8). `proj` rows are `proj_stride`
// floats apart; `mode2` picks the Bm=32 tile exactly as the dense launch's does.
extern "C" __global__ void q8_ko_int8_f32_dense_silu(
    const void*, const float*, int, float*, int, int, int, int, int);
extern "C" __global__ void q8_ko_int8_f32_dense_silu_m2(
    const void*, const float*, int, float*, int, int, int, int, int);

extern "C" int run_dense_int8_silu_q8ko_f32(
    const void* weights,
    const float* proj,
    int32_t proj_stride,
    float* dst,
    int32_t ncols_x,      // K: the activation's width, the weight's columns
    int32_t nrows_x,      // N
    int32_t total_batch,  // M
    int32_t sum_norm,
    int32_t mode2,
    void* stream
) {
    // K whole tiles inside each row; N whole 32-row tiles.
    if (ncols_x <= 0 || ncols_x % 128 != 0 || proj_stride < ncols_x || nrows_x % 32 != 0) {
        return QMM_NO_KERNEL;
    }
    // An empty product has nothing to write — and a zero-sized grid is a launch
    // error that would otherwise surface at whatever op checks next.
    if (total_batch <= 0 || nrows_x <= 0) {
        return QMM_OK;
    }
    const int bm = mode2 ? 32 : 16;
    dim3 grid((total_batch + bm - 1) / bm, (nrows_x + 31) / 32, 1);
    dim3 block(WARP_SIZE, 4, 1);
    const int dst_stride = nrows_x;
    void* args[] = {
        (void*)&weights, (void*)&proj, (void*)&proj_stride, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&total_batch,
        (void*)&dst_stride, (void*)&sum_norm,
    };
    // The fused-activation entry waits (`pdl_wait`) before it reads `proj`.
    if (const cudaError_t e = launch_pdl(
            mode2 ? (void*)q8_ko_int8_f32_dense_silu_m2 : (void*)q8_ko_int8_f32_dense_silu,
            grid, block, args, 0, (cudaStream_t)stream);
        e != cudaSuccess) {
        return qmm_launch_failed(e, "dense silu: launch");
    }
    return QMM_OK;
}

extern "C" int run_quantized_matmul(
    const vx_segment_t* segments,
    int32_t num_segments,
    const void* vy,
    void* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t qtype,
    int32_t ytype,
    size_t weight_bytes,  // Weight tensor size in bytes for L2 cache decision (FP path)
    int32_t tile_mode,    // int8 dense tiling: 0 = mode-1 (Bm=16), 1 = mode-2 (Bm=32), 2 = mode-4
                          // (Bm=64 × 128 rows; KO formats). Rust decides (`DenseTile`).
    int32_t out_dtype,    // int8 dense store width: 0 = F16, 1 = BF16, 2 = F32. FP path ignores it
                          // (there the output dtype is the activation dtype).
    int32_t sum_norm,     // q8a128 Σx convention (`SumScale::as_code()`); the int8 dense path
                          // only — the FP kernels below read no q8a128 header.
    void* stream_arg      // the stream every launch below is issued on
) {
    cudaStream_t stream = (cudaStream_t)stream_arg;
    // Lookup table for kernel sets: [qtype][ytype][use_tc]
    // ytype: 0=F16, 1=BF16, 2=F32
    // use_tc: 0 = CUDA cores, 1 = tensor cores (for s64)
    static kernel_set_t kernels[14][3][2] = {
        // Q4_0 (qtype=0)
        {{ KERNEL_SET(q4_0_f16), KERNEL_SET_TC(q4_0_f16) },
         { KERNEL_SET(q4_0_bf16), KERNEL_SET_TC(q4_0_bf16) },
         { KERNEL_SET(q4_0_f32), KERNEL_SET_TC(q4_0_f32) }},
        // Q4_1 (qtype=1)
        {{ KERNEL_SET(q4_1_f16), KERNEL_SET_TC(q4_1_f16) },
         { KERNEL_SET(q4_1_bf16), KERNEL_SET_TC(q4_1_bf16) },
         { KERNEL_SET(q4_1_f32), KERNEL_SET_TC(q4_1_f32) }},
        // Q5_0 (qtype=2)
        {{ KERNEL_SET(q5_0_f16), KERNEL_SET_TC(q5_0_f16) },
         { KERNEL_SET(q5_0_bf16), KERNEL_SET_TC(q5_0_bf16) },
         { KERNEL_SET(q5_0_f32), KERNEL_SET_TC(q5_0_f32) }},
        // Q5_1 (qtype=3)
        {{ KERNEL_SET(q5_1_f16), KERNEL_SET_TC(q5_1_f16) },
         { KERNEL_SET(q5_1_bf16), KERNEL_SET_TC(q5_1_bf16) },
         { KERNEL_SET(q5_1_f32), KERNEL_SET_TC(q5_1_f32) }},
        // Q8_0 (qtype=4)
        {{ KERNEL_SET(q8_0_f16), KERNEL_SET_TC(q8_0_f16) },
         { KERNEL_SET(q8_0_bf16), KERNEL_SET_TC(q8_0_bf16) },
         { KERNEL_SET(q8_0_f32), KERNEL_SET_TC(q8_0_f32) }},
        // Q2_K (qtype=5)
        {{ KERNEL_SET(q2_K_f16), KERNEL_SET_TC(q2_K_f16) },
         { KERNEL_SET(q2_K_bf16), KERNEL_SET_TC(q2_K_bf16) },
         { KERNEL_SET(q2_K_f32), KERNEL_SET_TC(q2_K_f32) }},
        // Q3_K (qtype=6)
        {{ KERNEL_SET(q3_k_f16), KERNEL_SET_TC(q3_k_f16) },
         { KERNEL_SET(q3_k_bf16), KERNEL_SET_TC(q3_k_bf16) },
         { KERNEL_SET(q3_k_f32), KERNEL_SET_TC(q3_k_f32) }},
        // Q4_K (qtype=7)
        {{ KERNEL_SET(q4_k_f16), KERNEL_SET_TC(q4_k_f16) },
         { KERNEL_SET(q4_k_bf16), KERNEL_SET_TC(q4_k_bf16) },
         { KERNEL_SET(q4_k_f32), KERNEL_SET_TC(q4_k_f32) }},
        // Q5_K (qtype=8)
        {{ KERNEL_SET(q5_k_f16), KERNEL_SET_TC(q5_k_f16) },
         { KERNEL_SET(q5_k_bf16), KERNEL_SET_TC(q5_k_bf16) },
         { KERNEL_SET(q5_k_f32), KERNEL_SET_TC(q5_k_f32) }},
        // Q6_K (qtype=9)
        {{ KERNEL_SET(q6_k_f16), KERNEL_SET_TC(q6_k_f16) },
         { KERNEL_SET(q6_k_bf16), KERNEL_SET_TC(q6_k_bf16) },
         { KERNEL_SET(q6_k_f32), KERNEL_SET_TC(q6_k_f32) }},
        // Q8_1 (qtype=10)
        {{ KERNEL_SET(q8_1_f16), KERNEL_SET_TC(q8_1_f16) },
         { KERNEL_SET(q8_1_bf16), KERNEL_SET_TC(q8_1_bf16) },
         { KERNEL_SET(q8_1_f32), KERNEL_SET_TC(q8_1_f32) }},
        // Q8_K (qtype=11)
        {{ KERNEL_SET(q8_k_f16), KERNEL_SET_TC(q8_k_f16) },
         { KERNEL_SET(q8_k_bf16), KERNEL_SET_TC(q8_k_bf16) },
         { KERNEL_SET(q8_k_f32), KERNEL_SET_TC(q8_k_f32) }},
        // Q_AWQ (qtype=12)
        {{ KERNEL_SET(q_awq_f16), KERNEL_SET_TC(q_awq_f16) },
         { KERNEL_SET(q_awq_bf16), KERNEL_SET_TC(q_awq_bf16) },
         { KERNEL_SET(q_awq_f32), KERNEL_SET_TC(q_awq_f32) }},
        // Q_AWQ_G64 (qtype=13)
        {{ KERNEL_SET(q_awq_g64_f16), KERNEL_SET_TC(q_awq_g64_f16) },
         { KERNEL_SET(q_awq_g64_bf16), KERNEL_SET_TC(q_awq_g64_bf16) },
         { KERNEL_SET(q_awq_g64_f32), KERNEL_SET_TC(q_awq_g64_f32) }},
    };

    // Map GgmlDType-aligned qtype values to the kernel-table row (0..13).
    // `qtype_to_matmul_kernel_index` lives in block_compact.cuh and returns
    // -1 for any format that has no matmul kernel.
    int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < 0) {
        return QMM_BAD_QTYPE;
    }

    // q8a128 INT8 path: `vy` is a block_q8a128 buffer, the single weight is
    // segments[0].weights (non-MoE → one segment), the output stores at `out_dtype`.
    // Same INT8 m16n8k32 core as the grouped path. TC-only. nrows_y is the batch M;
    // nrows_x is N. ytype 3 = Q8A128: the ONE int8 activation type. The mode (mode-1
    // Bm=16 vs mode-2 Bm=32 weight-reuse) is a kernel/tiling property the dispatcher
    // picks from the token count — the q8a1024 activation layout is mode-independent.
    if (ytype == 3) {
        if (num_segments < 1) {
            return QMM_NO_SEGMENTS;
        }
        if (out_dtype < 0 || out_dtype > 2) {
            return QMM_BAD_OUT_DTYPE;
        }
        const void* weights = segments[0].weights;
        const int total_batch = nrows_y;                  // M
        const int y_stride = ncols_x;                     // unused by the int8 kernel (ABI)
        const int dst_stride = nrows_x;                   // N
        const bool ko_row = kernel_row >= KO_ROW_FIRST && kernel_row - KO_ROW_FIRST < KO_ROW_COUNT;
        void* kfn;
        int batch_div;
        int row_div;
        if (tile_mode == 2 && ko_row) {
            kfn = dense_kernels_int8_m4[out_dtype][kernel_row - KO_ROW_FIRST];  // prefill tile
            batch_div = 64;                               // Bm = 64 (mode-4, N_SUB=4)
            row_div = 128;                                // 32 rows × RN=4
        } else if (tile_mode == 1 && ko_row) {
            kfn = dense_kernels_int8_m2[out_dtype][kernel_row - KO_ROW_FIRST];  // Bm=32 weight-reuse
            batch_div = 32;                               // Bm = 32 (mode-2, N_SUB=2)
            row_div = 32;
        } else if (tile_mode == 0) {
            // The wider tiles exist for the KO rows alone; a wider tile asked of any
            // other row is refused below, never quietly run as mode-1.
            kfn = dense_kernels_int8[out_dtype][kernel_row];
            batch_div = 16;                               // BATCH_TILE_I8 = 16 (mode-1)
            row_div = 32;
        } else {
            return QMM_BAD_TILE_MODE;
        }
        if (kfn == nullptr) {
            return QMM_NO_KERNEL;
        }
        const int batch_tiles = (total_batch + batch_div - 1) / batch_div;
        const int row_tiles = (nrows_x + row_div - 1) / row_div;
        dim3 grid(batch_tiles, row_tiles, 1);
        dim3 block(WARP_SIZE, 4, 1);                       // 128 threads (4 warps × 32)
        void* args[] = {
            (void*)&weights, (void*)&vy, (void*)&dst,
            (void*)&ncols_x, (void*)&nrows_x, (void*)&total_batch,
            (void*)&y_stride, (void*)&dst_stride, (void*)&sum_norm,
        };
        // The KO entries wait (`pdl_wait`) before reading the activation, so they launch
        // programmatically; the legacy rows are no-op entries that never reach here in
        // production and keep the ordinary launch.
        const bool ko = kernel_row >= KO_ROW_FIRST && kernel_row - KO_ROW_FIRST < KO_ROW_COUNT;
        const cudaError_t launched = ko ? launch_pdl(kfn, grid, block, args, 0, stream)
                                        : cudaLaunchKernel(kfn, grid, block, args, 0, stream);
        if (launched != cudaSuccess) {
            return qmm_launch_failed(launched, ko ? "matmul KO: launch" : "matmul: launch");
        }
        return QMM_OK;
    }

    if (ytype < 0 || ytype > 2) {
        return QMM_BAD_YTYPE;
    }
    // KO byte-permuted formats (rows >= 14) have INT8 kernels only; the FP kernel
    // table `kernels` is sized to the 14 base formats. KO reaches here only on a
    // misrouted FP call — reject rather than index out of bounds.
    if (kernel_row >= 14) {
        return QMM_BAD_QTYPE;
    }

    // Cache device properties (queried once per device) via shared header
    const auto& caps = get_device_caps();

    // Determine tensor core usage from SM version
    // TC path uses MMA instructions - all dtypes can use TC!
    // ytype: 0=F16, 1=BF16, 2=F32
    // SM80+ (Ampere/Ada): TC enabled for F16/BF16/F32 (ytype 0, 1, 2)
    //   F32 uses FP16 MMA but accumulates in FP32 for precision
    bool use_tc = false;
    if (ytype == 0 || ytype == 1 || ytype == 2) {  // F16, BF16, or F32
        use_tc = (caps.sm_version >= 800);
    }

    // Select kernel set from lookup table
    // tc_idx: 0 = CUDA cores, 1 = tensor cores (s16_tc variant)
    int tc_idx = use_tc ? 1 : 0;
    const kernel_set_t& kset = kernels[kernel_row][ytype][tc_idx];
    
    // Runtime validation: check that at least s1 kernel is linked (catches missing .cu files)
    // This is a one-time low-cost check that prevents silent NaN from unlinked kernels
    if (kset.s1 == nullptr) {
        // Kernel not compiled for this (format, activation dtype).
        return QMM_NO_KERNEL;
    }
    
    // Calculate Y element size for pointer arithmetic
    // ytype: 0=F16 (2 bytes), 1=BF16 (2 bytes), 2=F32 (4 bytes)
    static const int y_elem_sizes[3] = { 2, 2, 4 };
    const int y_elem_size = y_elem_sizes[ytype];
    
    // Calculate dst element size (same as Y type)
    // F16/BF16 = 2 bytes, F32 = 4 bytes
    const int dst_elem_size = y_elem_sizes[ytype];
    
    // =========================================================================
    // L2 CACHE DECISION
    // =========================================================================
    // L2 cache threshold: 80% of L2 (leave room for activations)
    size_t l2_threshold = (caps.l2_cache_size * 80) / 100;
    
    // Dispatch strategy selection:
    // - L2-cached weights: Use iter kernels (high occupancy hides L2 latency)
    // - DRAM-bound weights: Use greedy s8+sN (more weight reuse per launch)
    bool use_l2_path = (weight_bytes < l2_threshold);
    
    // =========================================================================
    // SEGMENT LOOP: dispatch each segment (expert) independently
    // =========================================================================
    // Each segment gets full greedy decomposition (TC → k1/k2/k3 → bulk → remainder).
    // Host-serial, GPU-async: each cudaLaunchKernel returns immediately.
    // Single-segment (non-MoE) is 1-iteration — same codepath, zero branching.
    
    int batch_offset = 0;
    
    for (int seg = 0; seg < num_segments; ++seg) {
        const void* vx = segments[seg].weights;
        int seg_batch = segments[seg].batch_count;
        if (seg_batch == 0) continue;
        
        const void* vy_slice = static_cast<const char*>(vy) + (size_t)batch_offset * nrows_y * y_elem_size;
        void* dst_slice = static_cast<char*>(dst) + (size_t)batch_offset * nrows_dst * dst_elem_size;
        
        int remaining = seg_batch;
        
        // =================================================================
        // TC PATH (plan-driven)
        // =================================================================
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        if (is_tc32_kernel(plan.tc_kernel) && plan.tc_batch > 0) {
            if (const cudaError_t e = launch_tc32(kset, plan.tc_kernel == K_TC32_R, vx, vy_slice,
                                                  dst_slice, ncols_x, nrows_x, nrows_y, nrows_dst,
                                                  plan.tc_batch, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment tc32: launch");
            }
            remaining = 0;
        } else if (is_tc16_kernel(plan.tc_kernel) && plan.tc_batch > 0) {
            if (const cudaError_t e = launch_tc16(kset, plan.tc_kernel == K_TC16_R, vx, vy_slice,
                                                  dst_slice, ncols_x, nrows_x, nrows_y, nrows_dst,
                                                  plan.tc_batch, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment tc16: launch");
            }
            remaining = 0;
        }
        
        // =================================================================
        // GEMV KERNELS from plan (remainder after TC, or full batch if no TC)
        // =================================================================
        // Note: vx is the same weight pointer for all launches in this segment.
        // Only vy_slice/dst_slice advance (different batch data, same weights).
        
        // k1
        if (plan.k1 != K_NONE && plan.b1 > 0 && remaining > 0) {
            int actual_b1 = (plan.b1 <= remaining) ? plan.b1 : remaining;
            if (const cudaError_t e = launch_kernel_by_type(kset, plan.k1, actual_b1, vx, vy_slice,
                                                            dst_slice, ncols_x, nrows_x, nrows_y,
                                                            nrows_dst, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment k1: launch");
            }
            remaining -= actual_b1;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b1 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b1 * nrows_dst * dst_elem_size;
        }
        
        // k2
        if (plan.k2 != K_NONE && plan.b2 > 0 && remaining > 0) {
            int actual_b2 = (plan.b2 <= remaining) ? plan.b2 : remaining;
            if (const cudaError_t e = launch_kernel_by_type(kset, plan.k2, actual_b2, vx, vy_slice,
                                                            dst_slice, ncols_x, nrows_x, nrows_y,
                                                            nrows_dst, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment k2: launch");
            }
            remaining -= actual_b2;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b2 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b2 * nrows_dst * dst_elem_size;
        }
        
        // k3 (L2 path 3-kernel cases)
        if (plan.k3 != K_NONE && plan.b3 > 0 && remaining > 0) {
            int actual_b3 = (plan.b3 <= remaining) ? plan.b3 : remaining;
            if (const cudaError_t e = launch_kernel_by_type(kset, plan.k3, actual_b3, vx, vy_slice,
                                                            dst_slice, ncols_x, nrows_x, nrows_y,
                                                            nrows_dst, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment k3: launch");
            }
            remaining -= actual_b3;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b3 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b3 * nrows_dst * dst_elem_size;
        }
        
        // =================================================================
        // BULK CATCH: Reduce any remaining >= 8 using s8 with grid.y
        // =================================================================
        while (remaining >= 8) {
            int bulk = (remaining / 8) * 8;
            if (const cudaError_t e = launch_kernel(kset.s8, 8, vx, vy_slice, dst_slice, ncols_x,
                                                    nrows_x, nrows_y, nrows_dst, bulk,
                                                    GRID_LAYOUT_ROW_FAST, 0, stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment s8: launch");
            }
            remaining -= bulk;
            vy_slice = static_cast<const char*>(vy_slice) + bulk * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + bulk * nrows_dst * dst_elem_size;
        }
        
        // =================================================================
        // FINAL REMAINDER: Handle remaining 1-7 batches
        // =================================================================
        if (remaining >= 1 && remaining <= 7) {
            void* remainder_kernels[] = {
                nullptr, kset.s1, kset.s2, kset.s3, kset.s4, kset.s5, kset.s6, kset.s7
            };
            if (const cudaError_t e = launch_kernel(remainder_kernels[remaining], remaining, vx,
                                                    vy_slice, dst_slice, ncols_x, nrows_x, nrows_y,
                                                    nrows_dst, remaining, GRID_LAYOUT_ROW_FAST, 0,
                                                    stream);
                e != cudaSuccess) {
                return qmm_launch_failed(e, "segment remainder: launch");
            }
        }
        
        batch_offset += seg_batch;
    }
    return QMM_OK;
}
