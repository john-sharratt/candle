# Arena Compaction Kernel Design

Status: Draft  
Last updated: 2026-04-13

---

## 1. Problem

`compact_arenas()` currently tombstones fully-empty arenas and truncates tail arenas. It does **no data movement**. A sparse arena with 1 live slot consumes the same GPU memory as a full one. This design adds a GPU compaction pass that physically relocates live GID slots into dense arenas so the sparse ones can be freed.

---

## 2. Memory Model

```
arena_base[arena_idx] + (gid % ARENA_CHUNKS) * stride_bytes
```

- `ARENA_CHUNKS = 8192` slots per arena
- `stride_bytes = 32 × bytes_per_block(format)` for all formats (except P2: `= 256`)
- One GID = one raw byte region of `stride_bytes`; always a multiple of 16
- A compaction **move** is a raw `memcpy` of `stride_bytes` bytes — no dequantisation

| Format | stride_bytes | tier |
|--------|-------------|------|
| F32 / R16 | 4096 | large |
| F16 / BF16 | 2048 | large |
| Q8_0 / Q8_1 / Q8_KS | 1088–1152 | medium |
| Q4_0 / Q4_1 / Q4_KS | 576–640 | medium |
| Q3_1 / P2 | 256–512 | small |
| Q3_0 / Q2_x / Q1_S_E4 | 160–448 | small |
| Q0 | 32 | small |

---

## 3. Kernel 1 — `arena_compact_copy`

Single kernel, launched three times in parallel streams with different `blockDim`.

```c
struct CompactMove {
    void*       dst;           // arena_base[dst_arena] + dst_slot * stride
    const void* src;           // arena_base[src_arena] + src_slot * stride
    uint32_t    stride_bytes;  // uniform per stream (one format per bucket)
};

// blockDim is constant per launch (128, 32, or N<32)
// loop handles stride > blockDim*16 naturally
__global__ void arena_compact_copy(
    const CompactMove* moves, int stride_bytes)
{
    const CompactMove& m = moves[blockIdx.x];
    for (int off = threadIdx.x * 16; off < stride_bytes; off += blockDim.x * 16) {
        uint4 val;
        memcpy(&val, (const char*)m.src + off, 16);
        memcpy((char*)m.dst + off, &val, 16);
    }
}
```

**Greedy blockDim dispatch:**

```c
// stream[0]: stride >= 2048  (F32, R16, F16, BF16)
arena_compact_copy<<<n_large,  128, 0, stream[0]>>>(large_moves,  stride_large);

// stream[1]: 512 <= stride < 2048  (Q8_x, Q4_x)
arena_compact_copy<<<n_medium,  32, 0, stream[1]>>>(medium_moves, stride_medium);

// stream[2]: stride < 512  (Q3_x, Q2_x, Q0, P2)
arena_compact_copy<<<n_small,    N, 0, stream[2]>>>(small_moves,  stride_small);
// where N = stride_small / 16  (exact, sub-warp)
```

Each bucket is homogeneous in stride — the compiler can unroll the loop and `__launch_bounds__` gives the register allocator precise info per instantiation.

---

## 4. Kernel 2 — `arena_compact_patch`

After data is moved, GPU block-table entries pointing to `src_gid` must be rewritten to `dst_gid`. The CPU sorts the move table by `src_gid` before upload; each GPU thread does a binary search.

```c
__global__ void arena_compact_patch(
    int32_t*       block_table,
    int            num_entries,
    const int32_t* src_gids,   // sorted ascending, device ptr
    const int32_t* dst_gids,
    int            num_moves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;
    int32_t entry = block_table[idx];
    if (entry < 0) return;  // -1 = empty slot

    int lo = 0, hi = num_moves;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (src_gids[mid] < entry) lo = mid + 1; else hi = mid;
    }
    if (lo < num_moves && src_gids[lo] == entry)
        block_table[idx] = dst_gids[lo];
}
// Launch: <<<ceil(num_entries, 256), 256, 0, stream[0]>>>
// Optional: load src_gids/dst_gids into SMEM if num_moves <= 2048 (16 KB)
```

---

## 5. Pipeline

### 5.1 Stages

```
Stage 1  Consolidation list
         Scan arenas by occupancy. Select evacuation candidates (occupancy < threshold).
         Output: Vec<(src_arena, dst_arena)>
              |
              v
Stage 2  GID src->dst list
         For each src_arena: live_gids = slots \ free_set.
         Allocate matching dst GIDs in dense target arenas.
         Output: Vec<(src_gid, dst_gid)>
              |
              v
Stage 3  Memory move slice list
         Resolve each pair to raw pointers:
           src_ptr = arena_base[src_arena] + src_slot x stride
           dst_ptr = arena_base[dst_arena] + dst_slot x stride
         Output: Vec<CompactMove>
              |
              v
Stage 4  Partition into format buckets + sort GID pairs
         bucket[0]: stride >= 2048  -> blockDim=128
         bucket[1]: 512 <= stride < 2048  -> blockDim=32
         bucket[2]: stride < 512  -> blockDim=N
         Also sort (src_gid, dst_gid) pairs by src_gid for patch kernel.
         Output: 3 x Vec<CompactMove>  +  sorted Vec<(i32, i32)>
              |
              v
Stage 5  H->D async upload
         cudaMemcpyAsync per non-empty bucket into pinned+device buffer.
         Upload sorted src_gids[] and dst_gids[] for patch kernel.
         Output: 3 x CudaSlice<CompactMove>  +  2 x CudaSlice<i32>
              |
              v
Stage 6  GPU execution
         stream[0/1/2]: copy kernels run concurrently (one per bucket)
         stream[0] (after copy): patch kernel rewrites GPU block_table
         Output: updated arena memory + updated GPU block_table
              |
              v
Stage 7  CPU post-sync bookkeeping
         cudaStreamSynchronize (all streams)
         Patch CPU block tables for all layers
         Recycle src_gids -> pool free lists
         Tombstone + release empty src arenas
```

### 5.2 Diagram

```
  CPU side                                      GPU side
  ──────────────────────────────────────────────────────────────────────

  [ Arena pool ]
       |
       v
  +──────────────────────────+
  | Stage 1: Consolidation   |
  | rank by occupancy        |
  | pick src + dst arenas    |
  +─────────────+────────────+
                |  Vec<(src_arena, dst_arena)>
                v
  +──────────────────────────+
  | Stage 2: GID src->dst    |
  | invert free_set per arena|
  | alloc dst GIDs in dense  |
  +─────────────+────────────+
                |  Vec<(src_gid, dst_gid)>
                v
  +──────────────────────────+
  | Stage 3: Resolve ptrs    |
  | GID -> (dst_ptr, src_ptr)|
  +─────────────+────────────+
                |  Vec<CompactMove>
                v
  +──────────────────────────+
  | Stage 4: Bucket + sort   |
  | [0] stride >= 2048       |
  | [1] 512 <= stride < 2048 |
  | [2] stride < 512         |
  +──+──────────+──────────+─+
     |          |          |
     v          v          v
  +──────+  +──────+  +──────+
  |Buf[0]|  |Buf[1]|  |Buf[2]|   Stage 5: H->D cudaMemcpyAsync
  +──+───+  +──+───+  +──+───+
     |  DMA    |  DMA     |  DMA
     v          v          v
  +────────────────────────────────────────────────────────────────+
  |                           GPU                                  |
  |  stream[0]           stream[1]           stream[2]            |
  |  copy<<<n0,128>>>    copy<<<n1,32>>>     copy<<<n2,N>>>        |
  |  F32/R16/F16/BF16    Q8_x / Q4_x         Q3_x / Q2_x / Q0    |
  |       |                   |                   |               |
  |       +───────────────────+───────────────────+               |
  |                           | all copies done                   |
  |                           v                                   |
  |                  +─────────────────+                          |
  |                  |  patch kernel   |  stream[0]               |
  |                  | binary search   |                          |
  |                  | rewrite GIDs    |                          |
  |                  +────────+────────+                          |
  +───────────────────────────+────────────────────────────────── +
                              |
                    cudaStreamSynchronize
                              |
  +───────────────────────────v──────────────────+
  | Stage 7: CPU bookkeeping                      |
  | Patch CPU block tables (all layers)           |
  | Recycle src_gids -> pool free lists           |
  | Tombstone + release empty src arenas          |
  +───────────────────────────────────────────────+
```

---

## 6. C API

```c
// Kernel 1: raw byte copy — format-agnostic.
// Grid: <<<num_moves, blockDim>>>  (blockDim = 128, 32, or N per bucket)
extern "C" void run_arena_compact_copy(
    const struct CompactMove* moves,
    int32_t num_moves,
    cudaStream_t stream
);

// Kernel 2: GID patch in GPU block table.
// src_gids must be sorted ascending.
// Grid: <<<ceil(num_entries, 256), 256>>>
extern "C" void run_arena_compact_patch(
    int32_t*       block_table,
    int32_t        num_entries,
    const int32_t* src_gids,
    const int32_t* dst_gids,
    int32_t        num_moves,
    cudaStream_t   stream
);
```

---

## 7. Reasoning and Summary

**Why CPU-side move plan?**  
The arena free-list is already CPU-resident; pointer resolution and occupancy scoring are O(moves) with negligible cost (< 200 µs for 2000 moves). Doing this on the GPU would require an extra scan kernel and a device→host readback, adding latency with no benefit.

**Why three streams with fixed blockDim per bucket?**  
Stride ranges from 32 bytes (Q0) to 4096 bytes (F32) — a 128× spread. A single blockDim cannot be efficient across that range: blockDim=128 wastes 94% of threads on Q0; blockDim=2 starves the SM scheduler for F16. Three homogeneous buckets allow the compiler to unroll the inner copy loop and `__launch_bounds__` to communicate exact register pressure per instantiation. The three streams execute concurrently on modern GPUs.

**Why binary search in the patch kernel?**  
For 100–4000 moves, a sorted array and a ~12-iteration binary search is optimal. All thread blocks read the same small array — L2 caches it after the first block. The SMEM variant (load into shared memory when `num_moves <= 2048`) eliminates even L2 pressure.

**Why raw-pointer move descriptors rather than GID pairs on the GPU?**  
The GPU kernel becomes a pure memcpy — one `uint4` load and one store per 16 bytes. No indexing arithmetic inside the kernel; no need for a GPU-resident arena base pointer table.

**Correctness invariant:**  
CPU block tables are patched *only after* `cudaStreamSynchronize` confirms both the data copy and the GPU block table patch are complete. The decode engine never observes a half-moved GID state.

**Expected latency** (Llama-3-8B, batch=32, 1000-move event):

| Phase | Cost |
|-------|------|
| Stages 1–4 (CPU plan) | < 200 µs |
| Stage 5 (H→D DMA) | < 50 µs |
| Stage 6 copy kernels | < 10 µs |
| Stage 6 patch kernel | < 100 µs |
| Stage 7 CPU bookkeeping | < 300 µs |
| **Total** | **< 1 ms** |

Compaction runs between decode batches and is never on the critical path.

---

## 8. GPU Resource Budget

Reference GPU: RTX 4090 (Ada Lovelace, sm_89)  
- 128 SMs, 1024 threads/SM max, 65 536 registers/SM, 100 KB SMEM/SM  
- 16 KB L1 per SM, 72 MB L2

### 8.1 `arena_compact_copy` — per bucket

The copy kernel is register-minimal: two pointers, one int, one loop counter, one `uint4` = 4 × u32. Measured on similar `memcpy` kernels: **~16 registers/thread**.

| Bucket | blockDim | stride | loop iters (F16/worst) | regs/thread | blocks/SM | occupancy |
|--------|----------|--------|------------------------|-------------|-----------|-----------|
| large  | 128 | 2048–4096 | 1–2 | ~16 | 8 | **100%** |
| medium | 32  | 576–1152  | 1–3 | ~16 | 32 | **100%** |
| small  | N≤31 | 32–512  | 1   | ~16 | ≥32 | ~50–100% |

All three buckets hit full or near-full occupancy. No SMEM used → no SMEM pressure. Register count leaves room for the compiler to keep the loop counter and `uint4` in registers without spill.

**SMEM:** 0 bytes. No shared memory allocation in the copy kernel.

**`__launch_bounds__`:**
```c
__launch_bounds__(128, 8)   // bucket 0: max 128 threads, min 8 blocks/SM
__launch_bounds__( 32, 16)  // bucket 1: max 32 threads,  min 16 blocks/SM
__launch_bounds__( 32, 16)  // bucket 2: same as bucket 1 (N rounds up to warp)
```

These hints tell the compiler not to over-allocate registers for larger warps, which would otherwise reduce max blocks/SM.

### 8.2 `arena_compact_patch` — block table rewrite

The patch kernel does a binary search over `num_moves` entries per thread. Sources of register pressure:
- `lo`, `hi`, `mid` loop variables (3 regs)
- `src_gids` and `dst_gids` pointers (2 × 64-bit = 4 regs)
- `entry` value and `idx` (2 regs)
- return address, stack frame: ~4 regs

Estimated: **~18 registers/thread** with 256 threads/block.

| blockDim | regs/thread | blocks/SM (reg-limited) | SMEM | occupancy |
|----------|-------------|-------------------------|------|-----------|
| 256 | ~18 | 14 | 0 | ~87% |
| 256 | ~18 | 8 (w/ SMEM 16 KB) | 16 KB | ~50% |

**Without SMEM** (num_moves > 2048): binary search hits L2. With 2× 4096 × 4 = 32 KB of src/dst arrays, L2 hit rate is high given all 14 blocks/SM search the same array. Acceptable.

**With SMEM** (num_moves ≤ 2048): load src_gids + dst_gids into 16 KB shared memory. Reduces max blocks/SM from 14 → 8 (SMEM-limited), but eliminates all global reads for the search arrays. Net win when num_moves is small (L2 pressure from repeated small reads is otherwise measurable).

```c
// SMEM variant threshold: use when num_moves <= 2048
#define PATCH_SMEM_THRESHOLD 2048

__launch_bounds__(256, 6)  // conservative: allow 6 blocks/SM with 16 KB SMEM each
```

### 8.3 Grid sizing

**Copy kernel** (per bucket):

```
gridDim.x = num_moves_in_bucket
```

One block per move. No grid-stride loop needed — each move is independent and blockIdx.x directly indexes it. On 4090 with 128 SMs and 1000 moves: ~8 blocks/SM, which is also the target min blocks/SM from `__launch_bounds__`.

**Patch kernel:**

```
num_entries  = batch_size × max_blocks × (GIDS_PER_HEAD × n_kv_head)
gridDim.x    = (num_entries + 255) / 256
```

For Llama-3-8B, batch=32, max_blocks=128, 8 kv_heads, GIDS_PER_HEAD=8:  
`32 × 128 × 64 = 262 144` entries → `gridDim.x = 1024` blocks.  
On 4090: 1024 / 128 = 8 waves. Each wave is one SM pass → ~50 µs.

**Stream concurrency:**  
Three copy streams + one patch stream (sequenced after copies on stream[0]). The three copy launches are independent work items from the GPU scheduler's perspective. On Ada, concurrent kernel execution between streams is supported when SMs are otherwise underutilised — which they will be early in each stream's smaller grid.

### 8.4 Summary table

| Kernel | blockDim | gridDim | regs/thread | SMEM/block | occupancy | bottleneck |
|--------|----------|---------|-------------|------------|-----------|------------|
| copy (large) | 128 | num_large_moves | ~16 | 0 | 100% | memory BW |
| copy (medium) | 32 | num_medium_moves | ~16 | 0 | 100% | memory BW |
| copy (small) | N | num_small_moves | ~16 | 0 | ~75% | latency |
| patch (no SMEM) | 256 | ceil(entries/256) | ~18 | 0 | 87% | L2 BW |
| patch (SMEM) | 256 | ceil(entries/256) | ~18 | 16 KB | 50% | compute |

All kernels are memory-bound in practice. Total GPU wall clock for a 1000-move compaction: < 100 µs, dominated by the patch kernel's block table scan.
