// The split-KV combine kernel and the device scratch that holds the per-split
// partial states.
//
// `int8_decode_combine_kernel` merges the per-split partial flash states for
// each head. `fused_attn_partial_pool` is the device scratch reused across the
// launches of one scratch generation, and `fused_attn_sm_count` reports the
// cached SM count used to size the split factor.

// -----------------------------------------------------------------------------
// Split-KV combine: merge the num_splits per-split partial flash-states for each
// (slot, query-head) into the final normalized output. One block per output row
// (slot*n_q_head + qh); HEAD_DIM threads, each owning one output dim. The merge
// is the standard log-sum-exp in NATURAL base — the flash kernels accumulate
// with `fast_exp` e^x (the `exp2` there is the float2-vectorized form, not
// base-2), so the per-split maxima in `partial_ml` are natural-log magnitudes:
//   gm = max_s m_s;  out = (Σ_s ΣwV_s · e^(m_s-gm)) / (Σ_s l_s · e^(m_s-gm)).
// Null partials (m=-inf, l=0) contribute zero.
//
// The split factor is sized to fill the device, not to the depth, so at short
// context nearly every partial is null (one live split of 110 at a 31-token
// row). The block therefore reads every split's (m, l) once, HEAD_DIM splits
// per round across its threads, and compacts the LIVE splits into shared
// arrays in split order; the merge then walks only those. The merge's
// arithmetic — ascending split order, `acc += pa·w; L += l·w` — is the same
// whichever splits are live, so the bytes match the CPU byte oracle exactly.
//
// The launch also commits the step's write-slice length when
// `commit_headers` is non-null (thread 0 of each slot's head-0 block): the
// commit has to follow every split's read of the write slice, and this
// kernel is already the launch that does, so a separate commit launch would
// be one more fixed cost per step.
// -----------------------------------------------------------------------------
template <typename O, int HEAD_DIM>
__global__ void int8_decode_combine_kernel(
    O* __restrict__ out,
    const float* __restrict__ partial_acc,   // [row][split][HEAD_DIM]
    const float* __restrict__ partial_ml,    // [row][split][2]
    int num_rows,
    int num_splits,
    uint8_t* __restrict__ q8_out,            // non-null → emit q8a128 (B2; HEAD_DIM % 128 == 0)
    const O* __restrict__ gate,              // non-null → val ⊙ sigmoid(gate) before the emit.
                                             // A slot's n_q_head·HEAD_DIM gate values are
                                             // contiguous; consecutive slots are
                                             // gate_slot_stride elements apart, so the gate can
                                             // be a strided view of the fused [q|gate]
                                             // projection with no copy.
    int64_t gate_slot_stride,
    int row_heads,                           // n_q_head — decomposes `row` into (slot, head)
                                             // for the gate and the commit
    const uint8_t* __restrict__ commit_headers,  // non-null → commit each slot's write-slice
                                                 // length (slot headers of the decode launch)
    int n_kv_head
) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "combine block is HEAD_DIM threads, whole warps");
    constexpr int WARPS = HEAD_DIM / WARP_SIZE;
    constexpr int ROUNDS = (MAX_SPLITS + HEAD_DIM - 1) / HEAD_DIM;

    int row = (int)blockIdx.x;
    if (row >= num_rows) return;
    const int d = (int)threadIdx.x;
    const int warp = d >> 5;
    const int lane = d & 31;

    if (commit_headers != nullptr && d == 0 && (row % row_heads) == 0) {
        commit_decode_write_len<HEAD_DIM>(commit_headers, row / row_heads, n_kv_head);
    }

    const float* ml = partial_ml + (int64_t)row * num_splits * 2;
    const float* pa = partial_acc + (int64_t)row * num_splits * HEAD_DIM;

    __shared__ float s_red[WARPS];
    __shared__ int   s_cnt[WARPS];
    __shared__ float s_w[MAX_SPLITS];
    __shared__ float s_l[MAX_SPLITS];
    __shared__ int   s_idx[MAX_SPLITS];

    // Thread d owns splits d, d + HEAD_DIM, ... — one (m, l) pair each.
    float m_r[ROUNDS];
    float l_r[ROUNDS];
    float gm = -1e38f;
    #pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        const int s = d + k * HEAD_DIM;
        m_r[k] = -1e38f;
        l_r[k] = 0.f;
        if (s < num_splits) {
            const float2 v = *reinterpret_cast<const float2*>(ml + (int64_t)s * 2);
            m_r[k] = v.x;
            l_r[k] = v.y;
            gm = fmaxf(gm, v.x);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, off, 32));
    }
    if (lane == 0) s_red[warp] = gm;
    __syncthreads();
    #pragma unroll
    for (int w = 0; w < WARPS; ++w) gm = fmaxf(gm, s_red[w]);

    // Compact the live splits in ascending split order: a warp ballot ranks a
    // thread within its warp, the warp counts rank the warps, and `base`
    // carries the count across rounds. A null partial (no tokens) writes only
    // its (m, l); its accumulator slot in the uninitialised pool is never
    // touched, so it is dropped on l rather than read. A split that saw
    // tokens always has l > 0.
    int base = 0;
    #pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        if (k * HEAD_DIM >= num_splits) break;   // block-uniform
        const int s = d + k * HEAD_DIM;
        const bool live = (s < num_splits) && (l_r[k] != 0.f);
        const uint32_t bal = __ballot_sync(0xffffffff, live);
        if (lane == 0) s_cnt[warp] = __popc(bal);
        __syncthreads();
        int pos = base;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const int c = s_cnt[w];
            if (w < warp) pos += c;
            base += c;
        }
        pos += __popc(bal & ((1u << lane) - 1u));
        if (live) {
            // Natural base to match the flash kernels' e^x accumulation — a 2^Δ
            // weight here would under-shrink low-max partials (2^Δ > e^Δ for
            // Δ<0) and skew the merged softmax wherever per-split maxima differ.
            s_w[pos] = expf(m_r[k] - gm);
            s_l[pos] = l_r[k];
            s_idx[pos] = s;
        }
        __syncthreads();
    }
    const int n_live = base;

    float acc = 0.f, L = 0.f;
    #pragma unroll 4
    for (int i = 0; i < n_live; ++i) {
        const float w = s_w[i];
        acc += pa[(int64_t)s_idx[i] * HEAD_DIM + d] * w;
        L   += s_l[i] * w;
    }
    float inv = __fdividef(1.f, fmaxf(L, 1e-10f));
    float val = acc * inv;

    // The emit — the plain O store, or B2's fused q8a128 context with the
    // optional output gate — is `int8_decode_emit_row`, shared with the tile
    // kernel's in-kernel merge so both paths produce the same bytes.
    __shared__ float sh_amax[HEAD_DIM / 32];
    __shared__ float sh_sum[HEAD_DIM / 32];
    // Raw Σx, pinned. These bytes are an ATTENTION CONTEXT — a convex
    // combination of V rows, so its per-128 sums are bounded by the same
    // magnitudes the residual stream already carries and never approach f16's
    // 65504. The Rust side wraps them with `Q8a128Operand`'s default
    // `SumScale::Raw`, and the two must agree: a producer here that normalised
    // while the operand reported raw would hand the matmul a header it
    // reinterprets, which is a wrong number rather than an error. A model that
    // ever needs otherwise threads the flag from the wrap site, as
    // `quantize_acts_q8a128` and the fused norms do.
    int8_decode_emit_row<O, HEAD_DIM>(val, (int64_t)row, d, out, q8_out, gate,
                                      gate_slot_stride, row_heads, sh_amax, sh_sum,
                                      /*sum_norm=*/0);
}

// SM count (cached) — used to size the split-KV factor to fill the device.
inline int fused_attn_sm_count() {
    static int sm = 0;
    if (sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        if (sm <= 0) sm = 1;
    }
    return sm;
}

// Device scratch for split-KV partials, reused across the launches of one scratch
// epoch — inside a forward, a carve of that forward's tier — and taken with
// `grow_scratch` (safe while the thread records a graph). Single-stream decode
// only (the pool is process-global, not per-stream).
inline void fused_attn_partial_pool(
    int64_t rows, int splits, int head_dim, float** acc_out, float** ml_out
) {
    static void* g_acc = nullptr;
    static void* g_ml  = nullptr;
    static size_t g_cap_acc = 0;  // capacity in bytes
    static size_t g_cap_ml  = 0;
    static unsigned long long g_epoch_acc = 0;
    static unsigned long long g_epoch_ml  = 0;
    size_t need_acc = (size_t)(rows * splits * head_dim) * sizeof(float);
    size_t need_ml  = (size_t)(rows * splits * 2) * sizeof(float);
    if (grow_scratch(&g_acc, &g_cap_acc, &g_epoch_acc, need_acc) != cudaSuccess ||
        grow_scratch(&g_ml, &g_cap_ml, &g_epoch_ml, need_ml) != cudaSuccess) {
        *acc_out = nullptr; *ml_out = nullptr; return;
    }
    *acc_out = (float*)g_acc;
    *ml_out  = (float*)g_ml;
}
