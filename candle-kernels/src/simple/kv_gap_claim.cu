// =============================================================================
// GAP CLAIM: destinations for the gap's quantize, chosen on the device
// =============================================================================
// `docs/kv_quantize_in_the_gap.md` §3.2–§3.4.
//
// The format selection leaves each (job, head)'s palette tags, outer scales and
// per-channel palette slots in device memory. The host has leased, per size
// class, a stack of free slot addresses. These kernels turn the one into
// destinations from the other without the host seeing either:
//
//   1. `gap_claim_kernel` (one block): every band's index in its class — the
//      count of bands of that class before it in table order, bands ordered
//      job, head, palette, K before V, continuing from the count earlier policy
//      groups of the same gap took (`taken`). A job is deferred when any of its
//      bands' indices falls past its class's lease. Deterministic: a prefix
//      count, not a race on an atomic cursor.
//   2. `gap_fill_kernel` (one block per head): the head's destination KvHead
//      record — packed palette maps, record format codes, outer scales and slot
//      addresses (the sink for a deferred job) — written into the convert's
//      buffer and into the mapped summary, with each band's lease index and
//      each job's deferred flag.
//   3. `gap_complete_kernel` (one thread): after the convert, the completion
//      word the host polls.
//
// The KvHead layout written here is the one `build_kvhead_bytes_raw`
// (candle-core `quantized/cuda.rs`) writes on the host:
//   [k pal map hd/4][v pal map hd/4][k ptr ×4][v ptr ×4][k fmt ×4][v fmt ×4]
//   [k scale ×4][v scale ×4]   — hd/2 + 104 bytes.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

namespace kv_gap_claim {

constexpr int N_PAL = 4;
constexpr int SIDES = 2;
constexpr int BANDS_PER_HEAD = N_PAL * SIDES;
constexpr int CLAIM_THREADS = 256;
// At least `SizeClass::COUNT`; the host refuses a ladder longer than this.
constexpr int MAX_CLASS = 16;
// Select codes are below this; the host's code tables are this long.
constexpr int N_CODES = 64;
constexpr int FILL_THREADS = 64;

__device__ __forceinline__ int rec_bytes(int hd) { return hd / 2 + 104; }

// A band's select code: the side's uniform override when set, else the
// selection's palette tag.
__device__ __forceinline__ int band_code(const int32_t* tags, int head, int p, int override_code) {
    return override_code >= 0 ? override_code : tags[head * N_PAL + p];
}

__device__ __forceinline__ int code_to_class(const int8_t* code_class, int code) {
    return (code >= 0 && code < N_CODES) ? (int)code_class[code] : -1;
}

__global__ void gap_claim_kernel(
    const int32_t* __restrict__ k_tags,
    const int32_t* __restrict__ v_tags,
    int32_t k_override,
    int32_t v_override,
    const int8_t* __restrict__ code_class,
    const uint32_t* __restrict__ lease_len,
    uint32_t* __restrict__ taken,
    uint32_t* __restrict__ band_index,
    uint32_t* __restrict__ job_deferred,
    uint32_t* __restrict__ error_word,
    int32_t n_jobs,
    int32_t n_kv_head,
    int32_t n_class)
{
    __shared__ uint32_t cnt[CLAIM_THREADS][MAX_CLASS + 1];
    const int t = threadIdx.x;
    // Contiguous job ranges per thread keep table order across threads.
    const int j0 = (int)(((int64_t)n_jobs * t) / CLAIM_THREADS);
    const int j1 = (int)(((int64_t)n_jobs * (t + 1)) / CLAIM_THREADS);

    uint32_t mine[MAX_CLASS];
    #pragma unroll
    for (int c = 0; c < MAX_CLASS; ++c) mine[c] = 0;
    for (int j = j0; j < j1; ++j) {
        for (int h = 0; h < n_kv_head; ++h) {
            const int head = j * n_kv_head + h;
            for (int p = 0; p < N_PAL; ++p) {
                const int kc = code_to_class(code_class, band_code(k_tags, head, p, k_override));
                const int vc = code_to_class(code_class, band_code(v_tags, head, p, v_override));
                if (kc >= 0 && kc < n_class) mine[kc]++;
                if (vc >= 0 && vc < n_class) mine[vc]++;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MAX_CLASS; ++c) cnt[t][c] = mine[c];
    __syncthreads();

    // Exclusive scan per class across threads, starting where earlier groups
    // of this gap left the class. One thread per class walks the threads.
    if (t < n_class) {
        uint32_t run = taken[t];
        for (int u = 0; u < CLAIM_THREADS; ++u) {
            const uint32_t x = cnt[u][t];
            cnt[u][t] = run;
            run += x;
        }
        taken[t] = run;
    }
    __syncthreads();

    uint32_t next[MAX_CLASS];
    #pragma unroll
    for (int c = 0; c < MAX_CLASS; ++c) next[c] = cnt[t][c];
    for (int j = j0; j < j1; ++j) {
        bool deferred = false;
        for (int h = 0; h < n_kv_head; ++h) {
            const int head = j * n_kv_head + h;
            for (int p = 0; p < N_PAL; ++p) {
                for (int s = 0; s < SIDES; ++s) {
                    const int code = s == 0 ? band_code(k_tags, head, p, k_override)
                                            : band_code(v_tags, head, p, v_override);
                    const int c = code_to_class(code_class, code);
                    const int b = (head * N_PAL + p) * SIDES + s;
                    if (c < 0 || c >= n_class) {
                        // A code the host's table does not cover: the host
                        // validated every candidate before launch, so this is
                        // a broken table. Defer the job and say so.
                        // A plain store: every writer stores the same value,
                        // and the word may be mapped host memory, where a
                        // system-scope atomic is not available everywhere.
                        *(volatile uint32_t*)error_word = 1u;
                        band_index[b] = 0xffffffffu;
                        deferred = true;
                        continue;
                    }
                    const uint32_t idx = next[c]++;
                    band_index[b] = idx;
                    if (idx >= lease_len[c]) deferred = true;
                }
            }
        }
        job_deferred[j] = deferred ? 1u : 0u;
    }
}

__global__ void gap_fill_kernel(
    const int32_t* __restrict__ k_tags,
    const int32_t* __restrict__ v_tags,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ k_map,
    const int32_t* __restrict__ v_map,
    int32_t k_override,
    int32_t v_override,
    const int8_t* __restrict__ code_class,
    const uint8_t* __restrict__ code_fmt,
    const uint64_t* __restrict__ stack,
    const uint32_t* __restrict__ stack_off,
    const uint32_t* __restrict__ band_index,
    const uint32_t* __restrict__ job_deferred,
    uint64_t sink,
    uint8_t* __restrict__ dst_records,
    uint8_t* __restrict__ summary_heads,
    uint32_t* __restrict__ summary_flags,
    int32_t head_dim,
    int32_t n_kv_head)
{
    const int head = blockIdx.x;
    const int job = head / n_kv_head;
    const int t = threadIdx.x;
    const int rec = rec_bytes(head_dim);
    const int pal_bytes = head_dim / 4;
    const int pal_dim = head_dim / N_PAL;
    const bool deferred = job_deferred[job] != 0;
    uint8_t* d = dst_records + (size_t)head * rec;
    uint8_t* s = summary_heads + (size_t)head * (rec + BANDS_PER_HEAD * 4);

    // Packed palette maps: one byte (four channels) per thread.
    if (t < pal_bytes) {
        uint8_t kb = 0, vb = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int dim = 4 * t + q;
            const int ident = dim / pal_dim;
            const int ks = k_override >= 0 ? ident : (k_map[(size_t)head * head_dim + dim] & 3);
            const int vs = v_override >= 0 ? ident : (v_map[(size_t)head * head_dim + dim] & 3);
            kb |= (uint8_t)(ks << (2 * q));
            vb |= (uint8_t)(vs << (2 * q));
        }
        d[t] = kb;
        d[pal_bytes + t] = vb;
        s[t] = kb;
        s[pal_bytes + t] = vb;
    }

    // Per band: slot address, record format code, outer scale, lease index.
    if (t < BANDS_PER_HEAD) {
        const int side = t / N_PAL;  // 0 = K, 1 = V
        const int p = t % N_PAL;
        const int code = side == 0 ? band_code(k_tags, head, p, k_override)
                                   : band_code(v_tags, head, p, v_override);
        const int c = code_to_class(code_class, code);
        const uint32_t idx = band_index[(head * N_PAL + p) * SIDES + side];
        const uint64_t addr = (deferred || c < 0) ? sink : stack[stack_off[c] + idx];
        const uint8_t fmt = (code >= 0 && code < N_CODES) ? code_fmt[code] : 0;
        const bool unit = side == 0 ? k_override >= 0 : v_override >= 0;
        const float scale = unit ? 1.0f
                                 : (side == 0 ? k_scale[head * N_PAL + p] : v_scale[head * N_PAL + p]);
        const int ptr_off = head_dim / 2 + side * 32 + p * 8;
        const int fmt_off = head_dim / 2 + 64 + side * 4 + p;
        const int scale_off = head_dim / 2 + 72 + side * 16 + p * 4;
        *(uint64_t*)(d + ptr_off) = addr;
        *(uint64_t*)(s + ptr_off) = addr;
        d[fmt_off] = fmt;
        s[fmt_off] = fmt;
        *(float*)(d + scale_off) = scale;
        *(float*)(s + scale_off) = scale;
        // Lease indices in band order: palette-major, K before V.
        *(uint32_t*)(s + rec + (p * SIDES + side) * 4) = idx;
    }

    if (t == 0 && head % n_kv_head == 0) {
        summary_flags[job] = deferred ? 1u : 0u;
    }
    __threadfence_system();
}

__global__ void gap_complete_kernel(uint32_t* word, uint32_t seq) {
    // Everything enqueued before this on the stream has completed; the fence
    // orders it ahead of the word for the host that polls it.
    __threadfence_system();
    *(volatile uint32_t*)word = seq;
}

}  // namespace kv_gap_claim

extern "C" void run_kv_gap_claim(
    const int32_t* k_tags,
    const int32_t* v_tags,
    int32_t k_override,
    int32_t v_override,
    const int8_t* code_class,
    const uint32_t* lease_len,
    uint32_t* taken,
    uint32_t* band_index,
    uint32_t* job_deferred,
    uint32_t* error_word,
    int32_t n_jobs,
    int32_t n_kv_head,
    int32_t n_class,
    void* stream)
{
    if (n_jobs <= 0 || n_class <= 0 || n_class > kv_gap_claim::MAX_CLASS) return;
    kv_gap_claim::gap_claim_kernel<<<1, kv_gap_claim::CLAIM_THREADS, 0, (cudaStream_t)stream>>>(
        k_tags, v_tags, k_override, v_override, code_class, lease_len, taken, band_index,
        job_deferred, error_word, n_jobs, n_kv_head, n_class);
}

extern "C" void run_kv_gap_fill(
    const int32_t* k_tags,
    const int32_t* v_tags,
    const float* k_scale,
    const float* v_scale,
    const int32_t* k_map,
    const int32_t* v_map,
    int32_t k_override,
    int32_t v_override,
    const int8_t* code_class,
    const uint8_t* code_fmt,
    const uint64_t* stack,
    const uint32_t* stack_off,
    const uint32_t* band_index,
    const uint32_t* job_deferred,
    uint64_t sink,
    uint8_t* dst_records,
    uint8_t* summary_heads,
    uint32_t* summary_flags,
    int32_t n_heads,
    int32_t head_dim,
    int32_t n_kv_head,
    void* stream)
{
    if (n_heads <= 0) return;
    kv_gap_claim::gap_fill_kernel<<<n_heads, kv_gap_claim::FILL_THREADS, 0, (cudaStream_t)stream>>>(
        k_tags, v_tags, k_scale, v_scale, k_map, v_map, k_override, v_override, code_class,
        code_fmt, stack, stack_off, band_index, job_deferred, sink, dst_records, summary_heads,
        summary_flags, head_dim, n_kv_head);
}

extern "C" void run_kv_gap_complete(uint32_t* word, uint32_t seq, void* stream) {
    kv_gap_claim::gap_complete_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(word, seq);
}
