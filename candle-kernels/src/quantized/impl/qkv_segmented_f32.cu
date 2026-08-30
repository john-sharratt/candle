// =============================================================================
// SEGMENTED QKV INT8 DENSE MATMUL
// =============================================================================
// One launch over a SHARED q8a128 activation × up to 3 KO weights (q, k, v) of
// possibly-DIFFERENT formats, writing the concatenated [M, Nq+Nk+Nv] output.
//
// Per thread-block, blockIdx.y is the GLOBAL 32-row N-tile; it resolves to one
// segment (segment boundaries align to N_TILE=32, so the format is block-uniform
// — no warp divergence), and the matching grouped_matmul_impl_int8<format> runs
// with that segment's (weight, N, dst column offset, segment-local N-tile). The
// q8a128 activation tile is shared across segments (re-read from L2). The MMA
// core is the EXACT one the per-format dense kernels use, so the fused result is
// float-identical to running q/k/v as three separate dense matmuls.
//
// fmt codes: 0=Q4_KO, 1=Q5_KO, 2=Q6_KO, 3=Q8_KO. The accumulator is F32 in
// registers regardless; `dst_t` only picks the width of the final store, so q/k/v
// land directly at the attention path's activation dtype.
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../matmul_status.cuh"
#include "../loader/q4_K.cuh"
#include "../loader/q5_K.cuh"
#include "../loader/q6_K.cuh"
#include "../loader/q8_K.cuh"
// The narrow twins' unpack traits. A missing loader header here does NOT fall back to
// something slower — the primary `gemx_dequant_traits` template has no
// `dequant_all_subs_int8` at all, so the omission is a compile error rather than a wrong
// answer. That is the only reason every format this file dispatches can be trusted to be the
// format it names.
#include "../loader/q2_KO.cuh"
#include "../loader/q3_KO.cuh"

// Per-format K/128 params (mirror the per-format *_f32.cu files): q2/q3/q4/q5/q6 use
// vdr=2 (16 threads / K-128 block); q8 uses qi=16, vdr=1.
namespace {
constexpr int QKV_QK = 128;
constexpr int QKV_QI_KQ = 32;
constexpr int QKV_VDR_KQ = 2;
constexpr int QKV_QI_Q8 = 16;
constexpr int QKV_VDR_Q8 = 1;
} // namespace

// KO formats this kernel dispatches, and the width of the `fmt` index. Must equal the arm
// count of the switch in `qkv_segmented_impl` and the row count of `kuni` — and must match
// `ko_fmt_code` in `candle-core/src/quantized/cuda.rs`, which is where a dtype becomes an
// index. A dtype missing from either side is not a wrong answer, it is a **missing** one:
// the mixed-format switch's default arm leaves that segment's columns unwritten.
#define QKV_FMT_COUNT 6

// Host-filled segment descriptor (must match the Rust `QkvSeg`).
struct qkv_seg_t {
    const void* weights;  // device ptr to this segment's KO weight
    int fmt;              // 0=Q4_KO 1=Q5_KO 2=Q6_KO 3=Q8_KO
    int n_tile_start;     // cumulative 32-row N-tile index where this segment begins
    int n_size;           // N (output rows) of this segment
    int dst_col_off;      // column offset of this segment in the fused output row
};

// Inside grouped_tc so the tiling constants (N_TILE/KI8_STRIDE via `using tc_common`, RING_I8) and
// grouped_matmul_impl_int8 resolve exactly as they do for the per-format dense entry.
namespace grouped_tc {

template <typename dst_t, int N_SUB>
__device__ __forceinline__ void qkv_segmented_impl(
    ::qkv_seg_t s0, ::qkv_seg_t s1, ::qkv_seg_t s2, int num_segs,
    const block_q8a128* __restrict__ act, dst_t* __restrict__ dst,
    int ncols_x, int total_batch, int y_stride, int dst_stride) {
    // Segments arrive by value in the kernel param space (≤3 for qkv) — no device-array upload.
    const ::qkv_seg_t segs[3] = {s0, s1, s2};
    constexpr int BATCH = N_SUB * 16;
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    const int gy = blockIdx.y;

    // Resolve the segment owning this N-tile (segments ordered by n_tile_start).
    int seg = 0;
    for (int s = 1; s < num_segs; ++s) {
        if (gy >= segs[s].n_tile_start) seg = s;
    }
    const int local_tile = gy - segs[seg].n_tile_start;
    const int seg_n = segs[seg].n_size;
    dst_t* seg_dst = dst + segs[seg].dst_col_off;
    const void* w = segs[seg].weights;

    // Activation smem is format-independent; the weight slot is sized for the
    // widest KO format (Q8_KO) so every branch fits.
    __shared__ __align__(16) int8_t smem_A_i8[2][BATCH][KI8_STRIDE];
    __shared__ __align__(16) half2 smem_A_ds[2][BATCH];
    constexpr int MAXCB = int8_chunk_bytes<block_compact_t<block_c_q8_KO>>::value;
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RING_I8 * MAXCB];

    switch (segs[seg].fmt) {
    case 0:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_KQ, block_c_q4_KO, QKV_VDR_KQ, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q4_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    case 1:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_KQ, block_c_q5_KO, QKV_VDR_KQ, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q5_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    case 2:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_KQ, block_c_q6_KO, QKV_VDR_KQ, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q6_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    case 3:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_Q8, block_c_q8_KO, QKV_VDR_Q8, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q8_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    case 4:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_KQ, block_c_q2_KO, QKV_VDR_KQ, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q2_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    case 5:
        grouped_matmul_impl_int8<QKV_QK, QKV_QI_KQ, block_c_q3_KO, QKV_VDR_KQ, dst_t, N_SUB>(
            reinterpret_cast<const block_compact_t<block_c_q3_KO>*>(w), act, seg_dst,
            ncols_x, seg_n, y_stride, dst_stride, b_start, b_cnt, local_tile,
            smem_A_i8, smem_A_ds, smem_W_flat);
        break;
    default:
        // Unreachable: `run_qkv_segmented_matmul` rejects an out-of-range `fmt` before the
        // launch, and `ko_fmt_code` on the Rust side never produces one. Falling through
        // silently would leave this segment's output columns **untouched** — the caller cannot
        // distinguish that from a completed matmul, which is why the validation is at the host
        // and this arm is only a compiler obligation.
        break;
    }
}

// UNIFORM-format fast path: all segments share one KO format, so there is no per-block format
// switch and the weight smem is sized EXACTLY for that format — identical occupancy/codegen to the
// single-weight dense kernel (i.e. to a physical concat), plus a 3-compare segment scan. This is
// what makes same-dtype q/k/v as fast as concat while keeping one launch.
template <typename dst_t, int N_SUB, int qk, int qi, typename block_q_t, int vdr>
__device__ __forceinline__ void qkv_seg_uniform_impl(
    ::qkv_seg_t s0, ::qkv_seg_t s1, ::qkv_seg_t s2, int num_segs,
    const block_q8a128* __restrict__ act, dst_t* __restrict__ dst,
    int ncols_x, int total_batch, int y_stride, int dst_stride) {
    const ::qkv_seg_t segs[3] = {s0, s1, s2};
    constexpr int BATCH = N_SUB * 16;
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    const int gy = blockIdx.y;
    int seg = 0;
    for (int s = 1; s < num_segs; ++s) {
        if (gy >= segs[s].n_tile_start) seg = s;
    }
    const int local_tile = gy - segs[seg].n_tile_start;
    const int seg_n = segs[seg].n_size;
    dst_t* seg_dst = dst + segs[seg].dst_col_off;
    const void* w = segs[seg].weights;

    __shared__ __align__(16) int8_t smem_A_i8[2][BATCH][KI8_STRIDE];
    __shared__ __align__(16) half2 smem_A_ds[2][BATCH];
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RING_I8 *
                                   int8_chunk_bytes<block_compact_t<block_q_t>>::value];
    grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, dst_t, N_SUB>(
        reinterpret_cast<const block_compact_t<block_q_t>*>(w), act, seg_dst, ncols_x, seg_n,
        y_stride, dst_stride, b_start, b_cnt, local_tile, smem_A_i8, smem_A_ds, smem_W_flat);
}

} // namespace grouped_tc

// Mixed-format entries, one per output dtype × tiling mode.
#define QKV_MIXED_KERNEL(TAG, DST_T, NSUB)                                                          \
    extern "C" __global__ void LAUNCH_BOUNDS_TC16 qkv_segmented_int8_##TAG##_dense_m##NSUB(         \
        qkv_seg_t s0, qkv_seg_t s1, qkv_seg_t s2, int num_segs, const block_q8a128* act,            \
        DST_T* dst, int ncols_x, int total_batch, int y_stride, int dst_stride) {                   \
        grouped_tc::qkv_segmented_impl<DST_T, NSUB>(s0, s1, s2, num_segs, act, dst, ncols_x,        \
                                                    total_batch, y_stride, dst_stride);            \
    }

QKV_MIXED_KERNEL(f16, half, 1)
QKV_MIXED_KERNEL(f16, half, 2)
QKV_MIXED_KERNEL(bf16, __nv_bfloat16, 1)
QKV_MIXED_KERNEL(bf16, __nv_bfloat16, 2)
QKV_MIXED_KERNEL(f32, float, 1)
QKV_MIXED_KERNEL(f32, float, 2)
#undef QKV_MIXED_KERNEL

// Single-format fast-path kernels (uniform q/k/v dtype): exact smem, no format switch.
#define QKV_UNIFORM_KERNEL(FMT, TAG, DST_T, BLOCK, QI, VDR, NSUB)                                   \
    extern "C" __global__ void LAUNCH_BOUNDS_TC16 qkv_seg_uniform_##FMT##_##TAG##_m##NSUB(          \
        qkv_seg_t s0, qkv_seg_t s1, qkv_seg_t s2, int num_segs, const block_q8a128* act,            \
        DST_T* dst, int ncols_x, int total_batch, int y_stride, int dst_stride) {                   \
        grouped_tc::qkv_seg_uniform_impl<DST_T, NSUB, QKV_QK, QI, BLOCK, VDR>(                      \
            s0, s1, s2, num_segs, act, dst, ncols_x, total_batch, y_stride, dst_stride);           \
    }

// Both tiling modes for one (format, output dtype).
#define QKV_UNIFORM_FMT(FMT, TAG, DST_T, BLOCK, QI, VDR)                                            \
    QKV_UNIFORM_KERNEL(FMT, TAG, DST_T, BLOCK, QI, VDR, 1)                                          \
    QKV_UNIFORM_KERNEL(FMT, TAG, DST_T, BLOCK, QI, VDR, 2)

// All three output dtypes for one format.
#define QKV_UNIFORM_ALL(FMT, BLOCK, QI, VDR)                                                        \
    QKV_UNIFORM_FMT(FMT, f16, half, BLOCK, QI, VDR)                                                 \
    QKV_UNIFORM_FMT(FMT, bf16, __nv_bfloat16, BLOCK, QI, VDR)                                       \
    QKV_UNIFORM_FMT(FMT, f32, float, BLOCK, QI, VDR)

QKV_UNIFORM_ALL(q4ko, block_c_q4_KO, QKV_QI_KQ, QKV_VDR_KQ)
QKV_UNIFORM_ALL(q5ko, block_c_q5_KO, QKV_QI_KQ, QKV_VDR_KQ)
QKV_UNIFORM_ALL(q6ko, block_c_q6_KO, QKV_QI_KQ, QKV_VDR_KQ)
QKV_UNIFORM_ALL(q8ko, block_c_q8_KO, QKV_QI_Q8, QKV_VDR_Q8)
QKV_UNIFORM_ALL(q2ko, block_c_q2_KO, QKV_QI_KQ, QKV_VDR_KQ)
QKV_UNIFORM_ALL(q3ko, block_c_q3_KO, QKV_QI_KQ, QKV_VDR_KQ)
#undef QKV_UNIFORM_ALL
#undef QKV_UNIFORM_FMT
#undef QKV_UNIFORM_KERNEL

// Host launcher. `h_segs` is a HOST pointer to a `num_segs`-long (≤3) qkv_seg_t array — copied into
// by-value kernel params here, so there is NO per-call device upload. `total_n_tiles` =
// Σ ceil(seg_n/32) (grid.y); `dst_stride` = N_total. The q8a128 activation is shared across segments.
// `out_dtype` picks the store width (0=F16, 1=BF16, 2=F32) — `dst` must point at that type.
// Returns a QMM_* status; a caller that ignores it cannot tell a launch from a no-op.
extern "C" int run_qkv_segmented_matmul(
    const void* h_segs, int num_segs, const void* act, void* dst,
    int ncols_x, int total_n_tiles, int total_batch, int dst_stride, int mode2, int out_dtype) {
    if (out_dtype < 0 || out_dtype > 2) {
        return QMM_BAD_OUT_DTYPE;
    }
    const qkv_seg_t* segs = reinterpret_cast<const qkv_seg_t*>(h_segs);
    qkv_seg_t s0{};
    qkv_seg_t s1{};
    qkv_seg_t s2{};
    if (num_segs > 0) s0 = segs[0];
    if (num_segs > 1) s1 = segs[1];
    if (num_segs > 2) s2 = segs[2];
    const int BATCH = mode2 ? 32 : 16;
    const int batch_tiles = (total_batch + BATCH - 1) / BATCH;
    dim3 grid(batch_tiles, total_n_tiles, 1);
    dim3 block(32, 4, 1); // 128 threads (4 warps × 32), same as the per-format dense kernels
    int y_stride = ncols_x; // unused by the int8 ABI

    // Uniform-format fast path: when every segment shares one KO format, dispatch the single-format
    // kernel (exact smem, no switch) so same-dtype q/k/v matches the single-weight dense path.
    int uniform_fmt = num_segs > 0 ? segs[0].fmt : -1;
    for (int i = 1; i < num_segs; ++i) {
        if (segs[i].fmt != uniform_fmt) {
            uniform_fmt = -1;
            break;
        }
    }
    // [out_dtype][fmt][mode2] and [out_dtype][mode2] — same ordering as OutDType. The `fmt`
    // index is `ko_fmt_code`'s: 0=Q4_KO, 1=Q5_KO, 2=Q6_KO, 3=Q8_KO, 4=Q2_KO, 5=Q3_KO. The two
    // small widths were appended rather than sorted in so the existing codes stay stable.
    static void* const kuni[3][QKV_FMT_COUNT][2] = {
        {{(void*)qkv_seg_uniform_q4ko_f16_m1, (void*)qkv_seg_uniform_q4ko_f16_m2},
         {(void*)qkv_seg_uniform_q5ko_f16_m1, (void*)qkv_seg_uniform_q5ko_f16_m2},
         {(void*)qkv_seg_uniform_q6ko_f16_m1, (void*)qkv_seg_uniform_q6ko_f16_m2},
         {(void*)qkv_seg_uniform_q8ko_f16_m1, (void*)qkv_seg_uniform_q8ko_f16_m2},
         {(void*)qkv_seg_uniform_q2ko_f16_m1, (void*)qkv_seg_uniform_q2ko_f16_m2},
         {(void*)qkv_seg_uniform_q3ko_f16_m1, (void*)qkv_seg_uniform_q3ko_f16_m2}},
        {{(void*)qkv_seg_uniform_q4ko_bf16_m1, (void*)qkv_seg_uniform_q4ko_bf16_m2},
         {(void*)qkv_seg_uniform_q5ko_bf16_m1, (void*)qkv_seg_uniform_q5ko_bf16_m2},
         {(void*)qkv_seg_uniform_q6ko_bf16_m1, (void*)qkv_seg_uniform_q6ko_bf16_m2},
         {(void*)qkv_seg_uniform_q8ko_bf16_m1, (void*)qkv_seg_uniform_q8ko_bf16_m2},
         {(void*)qkv_seg_uniform_q2ko_bf16_m1, (void*)qkv_seg_uniform_q2ko_bf16_m2},
         {(void*)qkv_seg_uniform_q3ko_bf16_m1, (void*)qkv_seg_uniform_q3ko_bf16_m2}},
        {{(void*)qkv_seg_uniform_q4ko_f32_m1, (void*)qkv_seg_uniform_q4ko_f32_m2},
         {(void*)qkv_seg_uniform_q5ko_f32_m1, (void*)qkv_seg_uniform_q5ko_f32_m2},
         {(void*)qkv_seg_uniform_q6ko_f32_m1, (void*)qkv_seg_uniform_q6ko_f32_m2},
         {(void*)qkv_seg_uniform_q8ko_f32_m1, (void*)qkv_seg_uniform_q8ko_f32_m2},
         {(void*)qkv_seg_uniform_q2ko_f32_m1, (void*)qkv_seg_uniform_q2ko_f32_m2},
         {(void*)qkv_seg_uniform_q3ko_f32_m1, (void*)qkv_seg_uniform_q3ko_f32_m2}},
    };
    static void* const kmix[3][2] = {
        {(void*)qkv_segmented_int8_f16_dense_m1, (void*)qkv_segmented_int8_f16_dense_m2},
        {(void*)qkv_segmented_int8_bf16_dense_m1, (void*)qkv_segmented_int8_bf16_dense_m2},
        {(void*)qkv_segmented_int8_f32_dense_m1, (void*)qkv_segmented_int8_f32_dense_m2},
    };
    // **An out-of-range `fmt` is refused here, not in the kernel.** The mixed-format switch's
    // `default` arm can only fall through, which leaves that segment's output columns untouched
    // — indistinguishable from a completed matmul at every level above. So the range check has
    // to happen where a status can still be returned.
    for (int i = 0; i < num_segs; ++i) {
        if (segs[i].fmt < 0 || segs[i].fmt >= QKV_FMT_COUNT) {
            return QMM_NO_KERNEL;
        }
    }
    const int mi = mode2 ? 1 : 0;
    void* kfn;
    if (uniform_fmt >= 0 && uniform_fmt < QKV_FMT_COUNT) {
        kfn = kuni[out_dtype][uniform_fmt][mi];
    } else {
        kfn = kmix[out_dtype][mi];
    }
    if (kfn == nullptr) {
        return QMM_NO_KERNEL;
    }
    void* args[] = {(void*)&s0,      (void*)&s1,         (void*)&s2,       (void*)&num_segs,
                    (void*)&act,      (void*)&dst,        (void*)&ncols_x,  (void*)&total_batch,
                    (void*)&y_stride, (void*)&dst_stride};
    cudaLaunchKernel(kfn, grid, block, args, 0, nullptr);
    return QMM_OK;
}
