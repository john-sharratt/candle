// =============================================================================
// MoE BUCKETIZE — the CPU class (docs/cpu_expert_compute.md §5.1–§5.3)
// =============================================================================
// Both bucketize kernels decide, per layer, which of the layer's misses the
// host CPU pool computes instead of the GPU copying them over the link, and
// publish that work to the pool. A CPU expert emits no GEMM tiles, is not
// on the remote list and claims no promotion slot; its rows stay in the
// expert-grouped row layout, and the down launch's CPU item drops the pool's
// result rows into exactly those rows of the down output — so the gather,
// `perm`, `token_starts` and the scatter are untouched, and a CPU row is
// weighted and summed exactly as a GPU row is.
//
// THE SPLIT (thread 0, over the eligible experts in ascending routed count —
// the cheapest for the CPU, while a copy costs the GPU the same whatever the
// count — cold before pinned within a count, `cpu_order_key` — and at most
// `max_experts` of them, a multiple of the pool's performance cores): an
// expert is eligible when it is a miss — pinned, its slot image in host memory
// the pool can read, or COLD (§5.4), which never crosses the link: the pool
// computes it from the warm tier's pageable copy at once if it has one, else
// from the pad slot the stager reads it into either way (the summary keeps its
// cold bit) once that is published — and routed at most `max_n` times — the
// host's cut, past which the CPU's compute alone exceeds the copy. A cold
// expert waits on its pack read on either side, so the split prices it as it
// prices a pinned one. For the first k of them on the CPU:
//
//   T_cpu(k) = handoff + max(k · read, Σ cpu_n[n_i]) + rows · row_ns
//   T_gpu(k) = Σ over the misses left on the GPU of (copy + gpu_n[n_i])
//
// with `read` the pool's per-expert DRAM read while the link streams, or
// alone once the GPU has no copies left. The k minimising max(T_cpu, T_gpu)
// wins if it beats k = 0 by `min_gain_ns`, within `cap_rows` assignments. The
// costs are measured on the host and arrive by value each launch
// (`SplitParams`): a graph segment rewrites them like any parameter.
//
// THE WORK SLOT (mapped host memory, one per invocation in flight): a header
// `{n_experts, n_rows, row, seq}`, then per CPU expert in ascending id
// `{expert, ring_start, count, _, gate, up, down}` — its rows' place in the
// row ring, and its three projections' host addresses (the snapshot's, which a
// pinned expert's are; 0 for a cold expert, whose addresses the pool finds
// itself — its pageable copy's, or the live table's once the stager publishes
// it). `dst_row[j]` (device) maps ring row j to its expert-grouped row, for the
// export kernel and the down launch's CPU item.
// The pool is woken by the export kernel's doorbell, not by this kernel: the
// rows it reads are not in the ring until the export has run.
// =============================================================================
#pragma once

#include "moe_bucketize_common.cuh"
#include "../moe_read_ahead.cuh"

// Routed counts the split's cost tables cover; a count past it is past any cut.
#define CPU_MAX_N 32

struct CpuSplit {
    uint32_t cap_rows;          // most CPU assignments a layer may take; 0 = the CPU class is off
    uint32_t max_n;             // the cut: counts above it stay on the GPU (≤ CPU_MAX_N)
    float copy_ns;              // one expert over the link
    float read_shared_ns;       // the pool reading one expert while the link streams
    float read_alone_ns;        // the same with the link idle
    float handoff_ns;           // the fixed handoff per layer with CPU work
    float row_ns;               // per CPU assignment: its input row out and its result row back
    float min_gain_ns;          // the CPU path must beat all-GPU by this
    float explore_ns;           // > 0 on an exploring invocation: take the cheapest single
                                // expert when it loses to all-GPU by at most this (§5.3.2)
    uint32_t promote;           // which CPU experts the next launch copies into VRAM
                                // behind the pool (CPU_PROMOTE_*, §5.11)
    uint32_t max_experts;       // most CPU experts a launch takes, a multiple of
                                // the pool's performance cores; its other misses
                                // stay on the GPU
    float cpu_n_ns[CPU_MAX_N + 1];  // the pool computing one expert for n tokens; [0] unused
    float gpu_n_ns[CPU_MAX_N + 1];  // the grouped GEMM for one expert at n tokens; [0] unused
};

// `CpuSplit::promote`: none; the cold experts the pool served; every expert it
// served.
#define CPU_PROMOTE_NONE 0u
#define CPU_PROMOTE_COLD 1u
#define CPU_PROMOTE_ALL 2u

// THE CARRY (device `u32[CPU_CARRY_WORDS]`): `{seq, n, then n (row, expert)
// pairs}` — the CPU experts this launch's pool serves that the NEXT launch
// promotes (`promotion_walk`). Only the next: its `summary_seq` is this one's
// plus one, and the pad slots it copies from are pinned only that long.
#define CPU_CARRY_WORDS (2 + 2 * AHEAD_MAX)

struct CpuOut {
    unsigned char* work;        // this invocation's mapped work slot, or null
    uint32_t* dst_row;          // device [a_ub]: ring row → expert-grouped row
    uint32_t* carry;            // the carry, or null: nothing is promoted behind the pool
    // THE INPUTS (`cpu_inputs`): each distinct token the CPU experts route
    // crosses the link once, however many of them it routes to.
    uint32_t* tok_first;        // device [n_tokens] scratch
    uint32_t* in_tok;           // device [cap]: input row → token
    uint32_t* inputs;           // mapped [1 + cap]: the input rows, then ring row → input row
};

// Work slot layout.
struct CpuWorkHeader {
    uint32_t n_experts;
    uint32_t n_rows;
    int32_t row;
    uint32_t seq;
};
struct CpuWorkExpert {
    uint32_t expert;
    uint32_t ring_start;
    uint32_t count;
    uint32_t _pad;
    uint64_t gate;
    uint64_t up;
    uint64_t down;
};
static_assert(sizeof(CpuSplit) == 11 * 4 + 2 * (CPU_MAX_N + 1) * 4,
              "CpuSplit is moe_bucketize.rs's CpuSplit byte for byte");
static_assert(sizeof(CpuWorkHeader) == 16, "work slot header layout");
static_assert(sizeof(CpuWorkExpert) == 40, "work slot expert layout");
#define CPU_WORK_SLOT_BYTES (sizeof(CpuWorkHeader) + MAX_EXPERTS * sizeof(CpuWorkExpert))

// The summary bit a CPU expert carries; the count field is the 28 bits below.
#define SUMMARY_CPU_BIT 28

// The split's order key: routed count, then cold before pinned. Both cost the
// GPU a copy, but a cold expert on the GPU stalls the gate launch on its pack
// read, where on the pool that read overlaps the GPU's own work.
#define CPU_ORDER_KEYS (2 * (CPU_MAX_N + 1))
__device__ __forceinline__ int32_t cpu_order_key(int32_t cnt, uint8_t cls) {
    return 2 * cnt + (cls == CLS_COLD ? 0 : 1);
}

// Shared scratch for the split.
struct CpuSplitShared {
    int32_t hist[CPU_ORDER_KEYS + 1];
    int32_t order[MAX_EXPERTS];
    float gpu_total;
    int32_t n_misses;
    int32_t eligible;
    int32_t take;
    unsigned long long scan[BUCKETIZE_WARPS + 1];
    int32_t n_cpu_experts;
    int32_t n_cpu_rows;
};

__device__ __forceinline__ float cpu_table(const float* t, int32_t n) {
    // Past the table, linear in n from its last bucket.
    return n <= CPU_MAX_N ? t[n] : t[CPU_MAX_N] * (float)n / (float)CPU_MAX_N;
}

// Whether a routed expert may go to the CPU: a miss routed at most `max_n`
// times, and — if cold — with no promotion of it in flight (`marked`). A
// promotion lands by publishing a VRAM address into the live entry the pool
// would be waiting on, and the pool reads only host memory; the GPU's own
// cold wait reads either.
__device__ __forceinline__ bool cpu_eligible(uint8_t cls, uint8_t marked, int32_t cnt, uint32_t max_n) {
    return cnt > 0 && (uint32_t)cnt <= max_n &&
           (cls == CLS_PINNED || (cls == CLS_COLD && !marked));
}

// Decides the layer's CPU experts into `sh_cpu` (1 = CPU), at most
// `cpu.max_experts` of them. Every thread of the block calls it, after the
// classification barrier; it ends with a barrier.
__device__ __forceinline__ void cpu_split_decide(
    const CpuSplit& cpu, int n_experts, const int32_t* sh_counts, const uint8_t* sh_cls,
    const uint8_t* sh_marked, uint8_t* sh_cpu, CpuSplitShared& s)
{
    const uint32_t max_experts = cpu.max_experts;
    const int tid = (int)threadIdx.x;
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        sh_cpu[e] = 0;
    }
    if (tid == 0) {
        for (int n = 0; n < CPU_ORDER_KEYS + 1; n++) {
            s.hist[n] = 0;
        }
        s.gpu_total = 0.f;
        s.n_misses = 0;
        s.take = 0;
    }
    __syncthreads();
    if (cpu.cap_rows == 0) {
        return;
    }
    const uint32_t max_n = cpu.max_n < CPU_MAX_N ? cpu.max_n : CPU_MAX_N;
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        const int32_t cnt = sh_counts[e];
        if (cnt <= 0 || sh_cls[e] == CLS_VRAM) {
            continue;
        }
        atomicAdd(&s.n_misses, 1);
        atomicAdd(&s.gpu_total, cpu.copy_ns + cpu_table(cpu.gpu_n_ns, cnt));
        if (cpu_eligible(sh_cls[e], sh_marked[e], cnt, max_n)) {
            atomicAdd(&s.hist[cpu_order_key(cnt, sh_cls[e]) + 1], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        // Key k's bin is hist[k + 1]; the exclusive prefix turns it into the
        // start of key k in ascending-key order.
        int32_t run = 0;
        for (int n = 0; n < CPU_ORDER_KEYS + 1; n++) {
            const int32_t c = s.hist[n];
            s.hist[n] = run;
            run += c;
        }
        s.eligible = run;
    }
    __syncthreads();
    // Ties broken by ascending id: the eligible experts of the same key below this one.
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        const int32_t cnt = sh_counts[e];
        if (!cpu_eligible(sh_cls[e], sh_marked[e], cnt, max_n)) {
            continue;
        }
        const int32_t key = cpu_order_key(cnt, sh_cls[e]);
        int32_t rank = 0;
        for (int x = 0; x < e; x++) {
            rank += cpu_order_key(sh_counts[x], sh_cls[x]) == key &&
                    cpu_eligible(sh_cls[x], sh_marked[x], sh_counts[x], max_n);
        }
        s.order[s.hist[key + 1] + rank] = e;
    }
    __syncthreads();
    if (tid == 0) {
        const int32_t eligible = s.eligible;
        const float base = s.gpu_total;
        float best = base;
        int32_t best_k = 0;
        float comp = 0.f, gpu_left = base;
        float first = base;
        uint32_t rows = 0;
        for (int32_t j = 0; j < eligible; j++) {
            const int32_t e = s.order[j];
            const int32_t cnt = sh_counts[e];
            if (rows + (uint32_t)cnt > cpu.cap_rows || (uint32_t)j >= max_experts) {
                break;
            }
            rows += (uint32_t)cnt;
            comp += cpu_table(cpu.cpu_n_ns, cnt);
            gpu_left -= cpu.copy_ns + cpu_table(cpu.gpu_n_ns, cnt);
            const int32_t kk = j + 1;
            const float read = kk < s.n_misses ? cpu.read_shared_ns : cpu.read_alone_ns;
            const float mem = (float)kk * read;
            const float t_cpu = cpu.handoff_ns + (mem > comp ? mem : comp) + (float)rows * cpu.row_ns;
            const float t_gpu = gpu_left > 0.f ? gpu_left : 0.f;
            const float tot = t_cpu > t_gpu ? t_cpu : t_gpu;
            if (kk == 1) {
                first = tot;
            }
            if (tot < best) {
                best = tot;
                best_k = kk;
            }
        }
        s.take = base - best >= cpu.min_gain_ns ? best_k : 0;
        // Exploring: a split that gave the CPU nothing still sends the
        // cheapest expert when the model says it loses by little, so the
        // measured costs stay fresh and a pessimistic model cannot lock the
        // CPU out for good.
        if (s.take == 0 && eligible > 0 && max_experts > 0 && cpu.explore_ns > 0.f &&
            first - base <= cpu.explore_ns && (uint32_t)sh_counts[s.order[0]] <= cpu.cap_rows) {
            s.take = 1;
        }
    }
    __syncthreads();
    for (int j = tid; j < s.take; j += BUCKETIZE_THREADS) {
        sh_cpu[s.order[j]] = 1;
    }
    __syncthreads();
}

// Publishes the CPU experts: thread `e` holds expert `e` (so n_experts ≤ the
// block), `offset` its first expert-grouped row. Writes the work slot's
// entries and header, `dst_row`, and `header[5..7]` = {CPU rows, CPU experts}.
// Every thread of the block calls it.
__device__ __forceinline__ void cpu_publish(
    const CpuOut& out, int n_experts, int e, int32_t cnt, int32_t offset, bool is_cpu,
    const uint64_t* snap, int32_t row, uint32_t seq, int32_t* header, CpuSplitShared& s)
{
    // `take` is block-uniform after the split's last barrier: a layer with no
    // CPU expert writes its zeros and skips the scan.
    if (s.take == 0) {
        if (threadIdx.x == 0) {
            s.n_cpu_experts = 0;
            s.n_cpu_rows = 0;
            header[5] = 0;
            header[6] = 0;
        }
        return;
    }
    const unsigned long long mine = is_cpu ? (((unsigned long long)cnt << 32) | 1ull) : 0ull;
    unsigned long long totals = 0ull;
    const unsigned long long before = block_exclusive_scan<unsigned long long>(mine, s.scan, &totals);
    const uint32_t ring_start = (uint32_t)(before >> 32);
    const uint32_t j = (uint32_t)(before & 0xffffffffull);
    const uint32_t n_rows = (uint32_t)(totals >> 32);
    const uint32_t n_cpu = (uint32_t)(totals & 0xffffffffull);
    if (is_cpu) {
        if (out.work != nullptr) {
            CpuWorkExpert* ex = reinterpret_cast<CpuWorkExpert*>(out.work + sizeof(CpuWorkHeader));
            ex[j].expert = (uint32_t)e;
            ex[j].ring_start = ring_start;
            ex[j].count = (uint32_t)cnt;
            ex[j]._pad = 0;
            ex[j].gate = snap[e];
            ex[j].up = snap[n_experts + e];
            ex[j].down = snap[2 * n_experts + e];
        }
        for (int32_t q = 0; q < cnt; q++) {
            out.dst_row[ring_start + q] = (uint32_t)(offset + q);
        }
    }
    if (threadIdx.x == 0) {
        s.n_cpu_experts = (int32_t)n_cpu;
        s.n_cpu_rows = (int32_t)n_rows;
        header[5] = (int32_t)n_rows;
        header[6] = (int32_t)n_cpu;
        if (out.work != nullptr) {
            CpuWorkHeader* h = reinterpret_cast<CpuWorkHeader*>(out.work);
            h->n_experts = n_cpu;
            h->n_rows = n_rows;
            h->row = row;
            h->seq = seq;
        }
    }
}

// The bit a token's `tok_first` word carries once it holds the token's input
// row rather than its first ring row.
#define CPU_INPUT_RANKED 0x80000000u

// THE INPUTS. After `cpu_publish`, with `tok_ids` (grouped row → token) final:
// the distinct tokens the layer's CPU ring rows route, in order of each
// token's first ring row, become the INPUT ROWS — `in_tok[r]` the token of
// input row `r`, which the export kernel copies from the MoE input, once — and
// `inputs` gets their count, then each ring row's input row, for the pool. A
// decode token routed to three CPU experts is one row over the link, not
// three. `header[7]` = the input rows. Every thread of the block calls it.
__device__ __forceinline__ void cpu_inputs(
    const CpuOut& out, int n_tokens, const uint32_t* tok_ids, int32_t* header, CpuSplitShared& s)
{
    const int tid = (int)threadIdx.x;
    // `n_cpu_rows` is thread 0's store in `cpu_publish`.
    __syncthreads();
    const int32_t n_rows = s.n_cpu_rows;
    if (n_rows == 0 || out.inputs == nullptr) {
        if (tid == 0) {
            header[7] = 0;
            if (out.inputs != nullptr) {
                out.inputs[0] = 0u;
            }
        }
        return;
    }
    for (int t = tid; t < n_tokens; t += BUCKETIZE_THREADS) {
        out.tok_first[t] = 0xffffffffu;
    }
    __syncthreads();
    for (int32_t j = tid; j < n_rows; j += BUCKETIZE_THREADS) {
        atomicMin(&out.tok_first[tok_ids[out.dst_row[j]]], (uint32_t)j);
    }
    __syncthreads();
    // Rank each token at its first ring row, a block-wide group of ring rows
    // at a time. A group's leaders all read before any writes its rank (the
    // scan's barriers), and a ranked word never equals a later ring row.
    uint32_t carried = 0;
    for (int32_t g0 = 0; g0 < n_rows; g0 += BUCKETIZE_THREADS) {
        const int32_t j = g0 + tid;
        uint32_t t = 0;
        bool lead = false;
        if (j < n_rows) {
            t = tok_ids[out.dst_row[j]];
            lead = out.tok_first[t] == (uint32_t)j;
        }
        unsigned long long total = 0ull;
        const unsigned long long before =
            block_exclusive_scan<unsigned long long>(lead ? 1ull : 0ull, s.scan, &total);
        if (lead) {
            const uint32_t r = carried + (uint32_t)before;
            out.in_tok[r] = t;
            out.tok_first[t] = r | CPU_INPUT_RANKED;
        }
        carried += (uint32_t)total;
        __syncthreads();
    }
    for (int32_t j = tid; j < n_rows; j += BUCKETIZE_THREADS) {
        out.inputs[1 + j] = out.tok_first[tok_ids[out.dst_row[j]]] & ~CPU_INPUT_RANKED;
    }
    if (tid == 0) {
        header[7] = (int32_t)carried;
        out.inputs[0] = carried;
    }
}
