# Grid-as-Cache-Hierarchy: CUDA Grid Dimension Design for L2 Cache Optimization

## Mapping Grid Dimensions to Cache Boundaries in Quantized GEMM

**Author:** John  
**Date:** January 2026  
**Target Hardware:** NVIDIA RTX 4090 (Ada Lovelace)  
**Application:** Quantized LLM inference (Q4_K), Battle Cities NPC engine

---

## 1. Problem Statement

Quantized GEMM kernels for LLM inference exhibit severe performance degradation at larger batch sizes due to L2 cache thrashing. Benchmarks on Qwen 32B (5120×5120, Q4_K) show:

| Batch Size | Measured Time | Expected Time | Degradation |
|------------|--------------|---------------|-------------|
| 256        | 2.07 ms      | ~2.0 ms       | None        |
| 512        | 7.17 ms      | ~4.1 ms       | 1.75×       |
| 1024       | 12.04 ms     | ~8.2 ms       | 1.47×       |
| 2048       | 23.54 ms     | ~16.4 ms      | 1.44×       |

The root cause is uncontrolled cache behavior: the default CUDA block scheduler makes no distinction between data with different reuse patterns, leading to high-reuse data (activations) being evicted by low-reuse data (weights) in L2 cache.

---

## 2. Core Insight

### 2.1 Memory Hierarchy Reality

At high occupancy, the GPU memory hierarchy has three effective tiers:

| Tier | Memory | Scope | Capacity (4090) | Management |
|------|--------|-------|-----------------|------------|
| 1 | Shared Memory | Per-block | ~16-20 KB/block | Explicit (programmer controlled) |
| 2 | L2 Cache | Global (all SMs) | 64 MB | Implicit (cache hints influence) |
| 3 | DRAM | Global | 24 GB | Source/sink |

**Why L1 is not a factor:** Each SM has 128 KB of unified on-chip memory split between shared memory and L1 data cache. At high occupancy (8 blocks/SM) with typical shared memory usage (~16 KB/block for double-buffered pipeline), the split is:

```
8 blocks × 16 KB = 128 KB shared memory → L1 data cache ≈ 0 KB
```

Any design relying on L1 data cache at high occupancy is fiction. This design operates entirely within shared memory and L2.

### 2.2 Data Access Patterns

The GEMM operation `Output[row, batch] = Σ_k Weight[row, k] × Activation[k, batch]` has two fundamentally different data access patterns:

- **Weights** are indexed by `(row, k)` — shared across all batches for a given row.
- **Activations** are indexed by `(k, batch)` — shared across all rows for a given batch.

Both data types are loaded from DRAM through L2 into shared memory via `cp.async`. The question is how to maximize L2 hit rate for each.

### 2.3 Two Types of L2 Reuse

**Temporal locality (weights):** Blocks on the same SM execute at similar times. If they need the same weight tile, their `cp.async` loads cluster temporally. The first load misses L2 and fetches from DRAM; subsequent loads from co-located blocks hit L2.

**Spatial locality (activations):** Blocks in the same wave execute concurrently across all SMs. If they need the same activation tile, one SM's load populates L2 and all other SMs hit.

CUDA's block scheduling assigns consecutive `linear_id` values to the same or nearby SMs:

```
linear_id = blockIdx.x + blockIdx.y × gridDim.x + blockIdx.z × gridDim.x × gridDim.y
```

By choosing which data dimension maps to which grid dimension, we control which data benefits from temporal locality (same SM) versus spatial locality (same wave). The grid structure itself becomes the L2 cache management strategy.

---

## 3. Grid Dimension Assignment

### 3.1 Three Dimensions, Three L2 Access Patterns

| Grid Dim | Data Dimension | L2 Access Pattern | Role |
|----------|---------------|-------------------|------|
| **x** | Batch tile | Temporal locality | Varies across blocks on same SM. Same-row blocks load same weight tile at similar times → L2 hit after first load. |
| **y** | Row tile (within wave) | Spatial locality | Varies across SMs within one wave. Same-batch blocks across all SMs share activation tile → L2 populated once, hit by all. |
| **z** | Wave index | Sequential persistence | Wave boundaries. Controls what persists in L2 between waves. |

### 3.2 Why This Assignment

**x = batch (fast dimension):** Consecutive `linear_id` values land on the same SM. Blocks with the same `y` and `z` but different `x` share an SM. Since `y` encodes the row tile, all these blocks need the same weight tile. They issue `cp.async` loads for this weight tile at nearly the same time. The first block's load misses L2 and fetches from DRAM; the other 7 blocks on the same SM hit L2. This is **temporal L2 locality** — same data, clustered in time.

**y = row within wave (medium dimension):** Blocks with different `y` but same `z` are all in the same wave — they execute concurrently across different SMs. Since `x` encodes the batch tile, blocks with the same `x` across different `y` values all need the same activation tile. One SM loads it into L2; all other SMs in the wave hit. This is **spatial L2 locality** — same data, spread across space but concurrent in time.

**z = wave boundary (slow dimension):** The `z` dimension creates explicit wave boundaries. All blocks in `z=0` (totalling `gridDim.x × gridDim.y`) execute as one wave before `z=1` begins. This is not guaranteed by the CUDA spec but follows from the scheduler's linear_id ordering and is reliably observed in practice.

### 3.3 Why x Must Be Batch, Not Row

An SM receives ~8 consecutive blocks. If `x = batch`:

```
SM gets: (batch=0,row=R), (batch=1,row=R), (batch=2,row=R), ...
         Row constant → all 8 blocks load same weight tile
         First block: L2 miss → DRAM
         Blocks 2-8: L2 hit (temporal locality) ✓
```

If `x = row` (wrong):

```
SM gets: (row=0,batch=B), (row=1,batch=B), (row=2,batch=B), ...
         Row changes every block → each block loads different weight tile
         Every block: L2 miss → DRAM ✗
```

The fast dimension must be the one we want to **vary** while holding the L2-cached data **constant**.

---

## 4. Dimension Sizing

### 4.1 x: Sized to Occupancy (Temporal L2 Unit)

`gridDim.x` determines how many batch-tile blocks share one SM and thus benefit from temporal L2 locality for weight tiles. The constraint is SM occupancy — how many blocks can run concurrently on one SM:

```
gridDim.x = min(batch_tiles, blocks_per_SM)
```

More blocks per SM means more L2 temporal reuse for weights (8 blocks sharing one weight tile = 7/8 = 87.5% L2 hit rate).

For batch=512: `gridDim.x = min(16, 8) = 8`

For batch=2048: `gridDim.x = min(64, 8) = 8`

### 4.2 y: Sized to L2 Capacity (Spatial L2 Unit)

`gridDim.y` determines how many SM-groups (x-groups) fit in L2 simultaneously — the spatial sharing domain. The constraint is L2 capacity available for activations:

```
L2_usable = L2_total × utilization_factor
          = 64 MB × 0.70 = 44.8 MB

The 0.70 factor accounts for:
  - Weight streaming pressure (.cs, but still transient L2 occupancy)
  - L2 associativity conflicts (16-way set-associative)
  - Other system overhead

Activation working set per SM-group (x blocks):
  act_per_x_group = gridDim.x × TILE_BATCH × K × sizeof(half)
                  = 8 × 32 × 5120 × 2 = 2.62 MB

y from L2 capacity:
  y_l2 = L2_usable / act_per_x_group
       = 44.8 MB / 2.62 MB = 17

y from occupancy (wave can't exceed max concurrent blocks):
  y_occ = max_concurrent_blocks / gridDim.x
        = 1024 / 8 = 128

gridDim.y = min(y_l2, y_occ, row_tiles)
          = min(17, 128, 160) = 17
```

**Note:** On 4090 with these tile sizes, L2 capacity is the binding constraint, not occupancy. This differs from the earlier analysis that assumed per-K-iteration working sets.

### 4.3 x × y: Wave Size (Spatial L2 Domain)

```
wave_size = gridDim.x × gridDim.y = 8 × 17 = 136 blocks per wave
```

All 136 blocks execute concurrently and share L2. This is smaller than max occupancy (1024) because L2 capacity for activations is the binding constraint.

### 4.4 z: Covers Overflow in Both Dimensions

```
row_groups = ceil(row_tiles / gridDim.y) = ceil(160 / 17) = 10
batch_groups = ceil(batch_tiles / gridDim.x) = ceil(16 / 8) = 2

gridDim.z = row_groups × batch_groups = 10 × 2 = 20
```

Z encodes both row overflow and batch overflow, with row as the inner loop.

### 4.5 Concrete Sizing Table (RTX 4090)

| Parameter | batch=512, M=5120 | batch=2048, M=5120 |
|-----------|-------------------|---------------------|
| batch_tiles | 16 | 64 |
| row_tiles | 160 | 160 |
| gridDim.x | 8 | 8 |
| gridDim.y | 17 | 17 |
| gridDim.z | 10 × 2 = 20 | 10 × 8 = 80 |
| Total blocks | 2,720 | 10,880 |
| row_groups | 10 | 10 |
| batch_groups | 2 | 8 |
| wave_size | 136 | 136 |

---

## 5. Z-Stride Ordering and DRAM Streaming

### 5.1 Memory Layouts

Understanding DRAM access patterns requires knowing how data is laid out:

```
Weights (Q4_K): W[out_features, in_features] row-major

  Address(row, k) = base + row × K × bytes_per_elem + k × bytes_per_elem
  
  For K=5120, Q4_K (0.5625 bytes/elem):
    Stride to next row: 5120 × 0.5625 = 2880 bytes
    
  Consecutive rows: NOT contiguous (2880 byte gaps)
  Consecutive k: contiguous (good for K-loop)


Activations: A[batch, hidden] row-major

  Address(batch, k) = base + batch × K × sizeof(half) + k × sizeof(half)
  
  For K=5120:
    Stride to next batch: 5120 × 2 = 10240 bytes (10 KB)
    
  Consecutive batches: NOT contiguous (10 KB gaps)
  Consecutive k: contiguous (good for K-loop)
```

### 5.2 DRAM Prefetch Behavior

GDDR6X characteristics relevant to prefetching:

- 256-bit memory bus (32 bytes per transaction)
- Burst length 16 → 512 bytes per burst
- Prefetcher detects sequential and strided patterns
- Prefetch state persists within a kernel execution

The prefetcher learns stride patterns and issues speculative fetches. Breaking the pattern (e.g., jumping to a distant address) wastes prefetch bandwidth and requires re-learning.

### 5.3 Z-Stride Options

Z must iterate over both excess row groups and batch groups. Two orderings:

**Option A: Rows inner (row_group cycles first)**
```
z = row_group + batch_group × num_row_groups

z=0: batch_group=0, row_group=0   → batches 0-255,  rows 0-543
z=1: batch_group=0, row_group=1   → batches 0-255,  rows 544-1087
...
z=9: batch_group=0, row_group=9   → batches 0-255,  rows 4896-5119
z=10: batch_group=1, row_group=0  → batches 256-511, rows 0-543
...
```

**Option B: Batches inner (batch_group cycles first)**
```
z = batch_group + row_group × num_batch_groups

z=0: batch_group=0, row_group=0   → batches 0-255,  rows 0-543
z=1: batch_group=1, row_group=0   → batches 256-511, rows 0-543
z=2: batch_group=0, row_group=1   → batches 0-255,  rows 544-1087
...
```

### 5.4 DRAM Streaming Analysis

**Within one wave (z=0):**

Weight access pattern:
```
y=0:  W[rows 0-31,    k=0:128]  → base + 0
y=1:  W[rows 32-63,   k=0:128]  → base + 32 × 2880 = +92 KB
y=2:  W[rows 64-95,   k=0:128]  → base + 64 × 2880 = +184 KB
...
y=16: W[rows 512-543, k=0:128]  → base + 512 × 2880 = +1.47 MB
```
DRAM sees: strided access with 92 KB stride, highly predictable.

Activation access pattern:
```
x=0: A[batches 0-31,    k=0:128]  → base + 0
x=1: A[batches 32-63,   k=0:128]  → base + 32 × 10240 = +320 KB
x=2: A[batches 64-95,   k=0:128]  → base + 64 × 10240 = +640 KB
...
x=7: A[batches 224-255, k=0:128]  → base + 224 × 10240 = +2.24 MB
```
DRAM sees: strided access with 320 KB stride, highly predictable.

**Across z-boundary with rows inner (Option A):**

```
z=0 processes: rows 0-543 (all k), batches 0-255
z=1 processes: rows 544-1087 (all k), batches 0-255

Weight stream:
  z=0 ends at:   W[row=543, k=5119]
  z=1 starts at: W[row=544, k=0]
  → Continues sequential row stride! Prefetcher pattern unbroken ✓

Activation stream:
  z=0 loaded: A[batches 0-255, all k]
  z=1 needs:  A[batches 0-255, all k]  ← SAME addresses
  → L2 hit if activations still resident ✓
  → No new DRAM traffic for activations ✓
```

**Across z-boundary with batches inner (Option B):**

```
z=0 processes: rows 0-543, batches 0-255
z=1 processes: rows 0-543, batches 256-511

Weight stream:
  z=0 ends at:   W[row=543, k=5119]
  z=1 starts at: W[row=0, k=0]   ← JUMPS back to beginning!
  → Prefetcher built up state for rows 500+
  → Now resets to row 0 — prefetch prediction broken ✗
  → Weight matrix re-streamed from start ✗

Activation stream:
  z=0 loaded: A[batches 0-255, all k]
  z=1 needs:  A[batches 256-511, all k]  ← Different addresses
  → Fresh DRAM load required
  → But stride pattern is same, prefetcher can re-learn
```

### 5.5 Decision: Rows Inner

Rows inner (Option A) is superior for two reasons:

**1. L2 Persistence:** Activations persist across z-boundary when batch_group stays constant. With 10 row_groups per batch_group, activations are loaded once and reused 10× before the batch advances.

**2. DRAM Prefetch Continuity:** Weights stream continuously through the weight matrix. The prefetcher establishes a stride pattern in z=0 that continues through z=1, z=2, etc. Each weight element is loaded from DRAM exactly once across the entire kernel.

With batches inner, weights would restart from row 0 on each batch_group advance, causing redundant DRAM traffic and prefetch thrashing.

### 5.6 Z-Stride Formula

```
z = row_group + batch_group × num_row_groups

row_group = z % num_row_groups
batch_group = z / num_row_groups
```

### 5.7 Execution Trace (batch=512, M=5120)

```
z=0:  batch_group=0, row_group=0  → batches 0-255,   rows 0-543     [load acts]
z=1:  batch_group=0, row_group=1  → batches 0-255,   rows 544-1087  [acts L2 hit ✓]
z=2:  batch_group=0, row_group=2  → batches 0-255,   rows 1088-1631 [acts L2 hit ✓]
...
z=9:  batch_group=0, row_group=9  → batches 0-255,   rows 4896-5119 [acts L2 hit ✓]
z=10: batch_group=1, row_group=0  → batches 256-511, rows 0-543     [load new acts]
z=11: batch_group=1, row_group=1  → batches 256-511, rows 544-1087  [acts L2 hit ✓]
...
z=19: batch_group=1, row_group=9  → batches 256-511, rows 4896-5119 [acts L2 hit ✓]
```

Activation loads from DRAM: **2 times** (once per batch_group)
Activation L2 reuse: **10×** per batch_group (across row_groups)
Weight loads from DRAM: **1 pass** through entire weight matrix (continuous stream)

---

## 6. Cache Hint Strategy

### 6.1 PTX Cache Operators

PTX provides load hints that influence cache behavior:

| Hint | Name | L1 | L2 | Eviction Priority |
|------|------|----|----|-------------------|
| `.ca` | Cache All | ✓ cached | ✓ cached | Normal (persist) |
| `.cg` | Cache Global | ✗ bypass | ✓ cached | Normal |
| `.cs` | Cache Streaming | ✗ bypass | ✓ cached | Evict-first |
| `.lu` | Last Use | ✓ cached | ✓ cached | Evict immediately after use |
| `.cv` | Cache Volatile | ✗ bypass | ✗ bypass | N/A |

### 6.2 Hint Assignment

| Data | Hint | Rationale |
|------|------|-----------|
| **Weights** | `.cs` | Cache streaming (evict-first priority in L2). Weights are used once per K-block per SM group, then dead. Lower reuse than activations. When L2 is under pressure, weights should evict before activations. |
| **Activations** | `.cg` | Cache global (normal L2 priority). Activations are shared across all SMs in a wave (gridDim.y reuse) and persist across z-waves (row-inner ordering). Highest reuse ratio. Should survive in L2 as long as possible. |
| **Output** | `.cs` | Streaming write, bypass L2. Output tiles are written once and never re-read. Writing to L2 would waste capacity on dead data — at batch=2048, output is 20 MB, which is 31% of L2 wasted if cached. Write directly to DRAM. |

### 6.3 Why Differentiate Weights and Activations

Both weights and activations flow through L2 on their way to shared memory. The difference is **reuse ratio** and **what should survive under pressure**:

```
Activation reuse per L2 load:
  - Spatial: gridDim.y SMs in wave = 128× reuse
  - Temporal: row_groups consecutive waves = 2× additional
  - Total: ~256× reuse per DRAM load

Weight reuse per L2 load:
  - Temporal: gridDim.x blocks on same SM = 8× reuse
  - No spatial reuse (different SMs need different rows)
  - Total: ~8× reuse per DRAM load
```

Activations have 32× higher reuse than weights. Under L2 pressure, evicting a weight tile costs 8 future L2 hits; evicting an activation tile costs 256 future L2 hits. The `.cs` hint on weights tells the L2 to evict weights first.

### 6.4 Why Not `.ca` for Weights

The `.ca` hint caches in both L1 and L2 with normal eviction priority. This is wrong for two reasons:

1. **L1 is unavailable.** At high occupancy with shared memory usage, L1 data cache is ~0 KB. The L1 portion of `.ca` is wasted.

2. **Normal L2 priority is wrong.** Weights should evict before activations. With `.ca` (normal priority) on weights and `.cg` (normal priority) on activations, L2 has no guidance on what to evict under pressure. Random eviction may evict high-value activations to keep low-value weights.

Using `.cs` on weights explicitly marks them as low-priority, ensuring activations survive.

---

## 7. Complete Data Flow

### 7.1 Three-Tier Memory Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SHARED MEMORY (per block)                        │
│                                                                      │
│   ┌─────────────────────┐    ┌─────────────────────┐                │
│   │ Weight pipeline     │    │ Activation pipeline │                │
│   │ (double-buffered)   │    │ (double-buffered)   │                │
│   │ ~4.5 KB             │    │ ~16 KB              │                │
│   └─────────────────────┘    └─────────────────────┘                │
│              │                         │                             │
│              └───────────┬─────────────┘                             │
│                          ▼                                           │
│                    Tensor Core MMA                                   │
│                          │                                           │
│                          ▼                                           │
│                    Accumulators                                      │
└──────────────────────────────────────────────────────────────────────┘
                           ▲                              │
                   cp.async loads                   store (.cs)
                           │                              │
┌──────────────────────────────────────────────────────────────────────┐
│                        L2 CACHE (64 MB)                              │
│                                                                      │
│   Activations (.cg):              Weights (.cs):                     │
│   - Normal priority               - Evict-first priority             │
│   - Spatial sharing (all SMs)     - Temporal sharing (same SM)       │
│   - Persists across z-waves       - Streams through                  │
│   - ~2.6 MB working set per wave  - Evicted after use                │
│                                                                      │
│   Output: BYPASSED (writes go directly to DRAM)                      │
└──────────────────────────────────────────────────────────────────────┘
                           ▲                              │
                      L2 miss                       write-through
                           │                              │
┌──────────────────────────────────────────────────────────────────────┐
│                        DRAM (24 GB, 1 TB/s)                          │
│                                                                      │
│   Source: Weights (88 MB), Activations (5 MB)                        │
│   Sink: Output (5 MB) ← written directly, no L2 pollution            │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Within One K-Iteration, One Wave

```
Weight loading (temporal L2 locality):
  SM 0, Block 0: cp.async W[row=R, k] → L2 MISS → DRAM fetch → L2 → shared
  SM 0, Block 1: cp.async W[row=R, k] → L2 HIT  → shared
  SM 0, Block 2: cp.async W[row=R, k] → L2 HIT  → shared
  ...
  SM 0, Block 7: cp.async W[row=R, k] → L2 HIT  → shared
  
  8 blocks, 1 DRAM fetch, 7 L2 hits → 87.5% L2 hit rate for weights

Activation loading (spatial L2 locality):  
  SM 0:   cp.async A[k, batch=B] → L2 MISS → DRAM fetch → L2 → shared
  SM 1:   cp.async A[k, batch=B] → L2 HIT  → shared
  SM 2:   cp.async A[k, batch=B] → L2 HIT  → shared
  ...
  SM 127: cp.async A[k, batch=B] → L2 HIT  → shared
  
  128 SMs, 1 DRAM fetch, 127 L2 hits → 99.2% L2 hit rate for activations
```

### 7.3 Across K-Iterations (Within One Block)

```
K=0:  Load W[row, 0] → shared.  Load A[0, batch] → shared.  MMA. Accumulate.
K=1:  Load W[row, 1] → shared.  Load A[1, batch] → shared.  MMA. Accumulate.
      (W[row, 0] was .cs → already evicted from L2, correct)
      (A[0, batch] may persist in L2 → doesn't matter, not needed again)
...
K=39: Load W[row, 39] → shared.  Load A[39, batch] → shared.  MMA. Accumulate.
      Store output directly to DRAM (.cs bypass).
```

Shared memory working set per block: **~20 KB** (double-buffered weights + activations). Constant throughout.

### 7.4 Across Z-Waves

```
z=0 (row_group=0, batch_group=0):
    Load activations[batch_group=0] to L2.
    Process all rows 0–127 × batches 0–7.
    
z=1 (row_group=1, batch_group=0):
    Activations[batch_group=0] STILL IN L2. ← Key benefit
    Process all rows 128–159 × batches 0–7.

z=2 (row_group=0, batch_group=1):
    Load activations[batch_group=1] to L2. (New batch group, forced reload)
    Process all rows 0–127 × batches 8–15.
    
z=3 (row_group=1, batch_group=1):
    Activations[batch_group=1] STILL IN L2. ← Key benefit
    ...
```

---

## 8. Block Index Decoding

Inside the kernel, block indices are decoded as:

```
batch_tile = blockIdx.x
row_in_wave = blockIdx.y

row_group = blockIdx.z % num_row_groups
batch_group = blockIdx.z / num_row_groups

row_tile = row_group × gridDim.y + row_in_wave
batch_tile_global = batch_group × gridDim.x + batch_tile

row_start = row_tile × TILE_ROWS
batch_start = batch_tile_global × TILE_BATCH
```

Bounds checking is required for the final wave when row tiles or batch tiles don't divide evenly.

---

## 9. Sizing Algorithm

```
function compute_grid_config(M, K, batch_size, gpu):
    tile_rows = 32
    tile_batch = 32
    
    total_row_tiles = ceil(M / tile_rows)
    total_batch_tiles = ceil(batch_size / tile_batch)
    
    blocks_per_sm = query_max_blocks_per_sm(gpu)      // typically 8
    max_concurrent = gpu.sm_count × blocks_per_sm     // 128 × 8 = 1024
    
    // X: sized to occupancy (temporal L2 unit)
    gridDim_x = min(total_batch_tiles, blocks_per_sm)
    
    // Y: sized to L2 capacity (spatial L2 unit)
    L2_usable = gpu.l2_size × 0.70                    // 64 MB × 0.7 = 44.8 MB
    act_per_x_group = gridDim_x × tile_batch × K × 2  // 8 × 32 × 5120 × 2 = 2.62 MB
    
    y_from_l2 = floor(L2_usable / act_per_x_group)    // 44.8 / 2.62 = 17
    y_from_occupancy = max_concurrent / gridDim_x     // 1024 / 8 = 128
    
    gridDim_y = min(y_from_l2, y_from_occupancy, total_row_tiles)
    
    // Wave size (may be less than max occupancy if L2-bound)
    wave_size = gridDim_x × gridDim_y
    
    // Z: covers remaining work (rows inner, batches outer)
    row_groups = ceil(total_row_tiles / gridDim_y)
    batch_groups = ceil(total_batch_tiles / gridDim_x)
    gridDim_z = row_groups × batch_groups
    
    // Z decode order: row_group inner, batch_group outer
    //   row_group = z % row_groups
    //   batch_group = z / row_groups
    
    return {
        grid: (gridDim_x, gridDim_y, gridDim_z),
        row_groups: row_groups,
        batch_groups: batch_groups,
        wave_size: wave_size
    }
```

### 9.1 Adaptation to Different GPUs

| GPU | SMs | Blocks/SM | Max Concurrent | L2 Size | y from L2 | y from Occ | y actual |
|-----|-----|-----------|----------------|---------|-----------|------------|----------|
| RTX 4090 | 128 | 8 | 1024 | 64 MB | 17 | 128 | 17 (L2 bound) |
| RTX 3090 | 82 | 16 | 1312 | 6 MB | 1 | 164 | 1 (L2 bound) |
| A100 | 108 | 32 | 3456 | 40 MB | 10 | 432 | 10 (L2 bound) |
| H100 | 132 | 32 | 4224 | 50 MB | 13 | 528 | 13 (L2 bound) |

On all current GPUs, L2 capacity is the binding constraint for y, not occupancy. The wave size is smaller than maximum concurrent blocks because fitting activation working sets in L2 is more important than maximizing occupancy.

---

## 10. Performance Analysis

### 10.1 Memory Traffic Reduction

**Qwen 32B (M=5120, K=5120, Q4_K), batch=512:**

| Traffic Source | Without Hierarchy | With Hierarchy | Reduction |
|----------------|-------------------|----------------|-----------|
| Weight DRAM→L2 (temporal) | 88 MB × 1 (no reuse) | 88 MB / 8 (8× L2 temporal reuse) | 8× |
| Act DRAM→L2 (spatial) | 5 MB × 160 (no reuse) | 5 MB × 3 (64× L2 spatial reuse) | 53× |
| Total DRAM reads | ~900 MB | ~93 MB | ~10× |
| Output L2 pollution | 5 MB (cached) | 0 MB (.cs bypass) | ∞ |

### 10.2 Projected Latency

| Batch | Current | Z-Slice Only | Hierarchical Grid | Theoretical Min |
|-------|---------|--------------|--------------------|-----------------| 
| 256   | 2.07 ms | 2.07 ms      | ~1.8 ms            | ~1.6 ms         |
| 512   | 7.17 ms | ~4.2 ms      | ~2.8 ms            | ~2.0 ms         |
| 1024  | 12.04 ms | ~8.2 ms     | ~5.5 ms            | ~4.0 ms         |
| 2048  | 23.54 ms | ~16.4 ms    | ~10.5 ms           | ~8.0 ms         |

### 10.3 Additional Benefits

**Predictable latency:** Performance scales linearly with batch size, eliminating the erratic swings (e.g., 10K→40K→15K GFLOPS) observed in current benchmarks. Critical for SLA guarantees in the Battle Cities NPC inference engine.

**Automatic scaling:** The sizing algorithm queries GPU properties at runtime. Same kernel binary works optimally on 4090, 3090, A100, H100 without manual tuning.

---

## 11. Comparison to Existing Techniques

### 11.1 CUTLASS Threadblock Swizzling

CUTLASS remaps `blockIdx` within a wave to improve spatial locality (e.g., Swizzle<3> groups nearby tiles). This optimizes **intra-wave** L2 access patterns but does not create wave boundaries, does not differentiate cache hints for different data types, and does not control inter-wave data persistence.

**Hierarchical grid** subsumes swizzling by making the grid dimensions themselves encode the L2 access patterns. The block-to-SM mapping is implicit in the dimension assignment rather than requiring explicit index remapping.

### 11.2 CUTLASS Raster Order (AlongM / AlongN)

CUTLASS allows choosing whether to traverse the M or N dimension as the fast axis within a tile grid. This is a one-dimensional choice that optimizes for one access pattern.

**Hierarchical grid** is a multi-level strategy: x for L2 temporal locality, y for L2 spatial locality, z for L2 persistence. It simultaneously optimizes all three L2 access patterns rather than choosing between two traversal orders.

### 11.3 Persistent Kernels with grid.sync()

Persistent kernels launch one block per SM and loop over tiles internally, giving full control over execution order. They can achieve the same cache behavior as hierarchical grids.

**Hierarchical grid** achieves equivalent cache behavior without persistent kernel infrastructure: no cooperative launch API, no grid.sync(), no internal work-scheduling loop. The kernel code remains a straightforward single-tile-per-block design. The scheduler does the work for us.

### 11.4 Split-K / Stream-K (CUTLASS)

Split-K uses `blockIdx.z` to partition the K dimension across blocks, requiring atomic reduction. Stream-K distributes K-work across blocks for better load balancing.

**Hierarchical grid** uses `blockIdx.z` for wave boundaries, not K-partitioning. The K dimension is iterated sequentially within each block (the K-loop), which is where the streaming cache hints (`.cs` vs `.cg`) operate. These are complementary techniques — Split-K could theoretically be combined with hierarchical grid by using a 4th level of indexing, though the added reduction overhead may not be worthwhile.

### 11.5 Simple Z-Slicing (Our Previous Design)

Our earlier z-slicing technique used `blockIdx.z` to create wave boundaries sized to L2 capacity, with optional 2D z-slicing for weight/row partitioning. This was a single-level optimization targeting L2 only.

**Hierarchical grid** extends z-slicing into a complete strategy:
- Adds L2 temporal locality for weights via x-dimension assignment (same-row blocks on same SM)
- Adds differentiated cache hints (`.cs` weights, `.cg` activations) to prioritize activation retention
- Adds z-stride ordering (rows inner, batches outer) for cross-wave L2 persistence
- Subsumes all benefits of z-slicing while adding weight L2 reuse

### 11.6 Summary Comparison

| Technique | L2 Temporal (weights) | L2 Spatial (activations) | Cross-Wave L2 | Kernel Complexity | Extra APIs |
|-----------|:-:|:-:|:-:|:-:|:-:|
| Default CUDA grid | ✗ | ✗ | ✗ | Minimal | None |
| CUTLASS Swizzle | Partial | Partial | ✗ | Low | None |
| CUTLASS Raster | Partial | Partial | ✗ | Low | None |
| Z-Slice (previous) | ✗ | Good | Partial | Low | None |
| Persistent Kernel | ✓ | ✓ | ✓ | High | Cooperative launch |
| **Hierarchical Grid** | **✓** | **✓** | **✓** | **Low** | **None** |

---

## 12. Design Decisions

### 12.1 SM Co-location Assumption

The design relies on CUDA assigning consecutive `linear_id` blocks to the same or nearby SMs. This is not guaranteed by the CUDA specification but follows from the hardware scheduler's round-robin assignment and has been observed consistently across every NVIDIA architecture from Kepler through Ada Lovelace.

**Decision:** Assume consecutive-linear-id co-location holds. The design benefits from this assumption through L2 temporal locality for weights — blocks on the same SM issue loads at similar times, causing L2 hits after the first load. If the assumption doesn't hold, blocks that need the same weight tile would be spread across time, reducing L2 hit rate from ~87.5% to near zero for weights. However, the L2 spatial optimization for activations (all SMs in a wave share activation tiles) and the cross-wave persistence (z-stride ordering) are unaffected — they depend on concurrent execution within a wave, not SM assignment.

Validation is straightforward: read the `%smid` special register in the kernel and log the mapping for a few launches. If any future GPU violates the assumption, the performance impact is limited to weight L2 hit rate.

### 12.2 Blocks per SM (Occupancy)

The tradeoff: higher occupancy (more blocks/SM) means more batch tiles benefiting from L2 temporal locality for weights, and better latency hiding. Lower occupancy means fewer blocks sharing weight loads.

**Decision:** Use maximum occupancy (8 blocks/SM on Ada, queried at runtime on other GPUs). At 8 blocks per SM, all 8 blocks processing the same row issue their weight loads at nearly the same time. The first load fetches from DRAM; the other 7 hit L2. This is 87.5% L2 hit rate for weights.

Higher occupancy also improves latency hiding during the K-loop's memory loads. With 8 blocks × 4 warps = 32 warps per SM, the scheduler has ample warps to keep execution units busy while waiting for memory. The 8× L2 temporal reuse at full occupancy is the sweet spot.

### 12.3 L2 Persistence API

CUDA 11+ provides `cudaStreamAttrValue` to hardware-reserve a portion of L2 for specific address ranges, giving a guarantee that designated data won't be evicted.

**Decision:** Do not use the L2 persistence API. The grid design already achieves activation persistence through two mechanisms: z-stride ordering (rows inner, batches outer) ensures activations are re-accessed by consecutive waves, and the `.cg` cache hint ensures activations are cached at the L2 level. The activation working set per wave is ~64 KB — four orders of magnitude smaller than L2 capacity (64 MB). Even under heavy weight streaming pressure, LRU eviction will never reach the activations because they're accessed by every SM on every K-iteration, keeping them maximally hot.

The persistence API adds per-stream configuration complexity, requires knowing activation buffer addresses at stream setup time (problematic with memory pools), and must be reconfigured when batch sizes change. These costs are not justified when the grid design already provides effective persistence through access pattern control. If profiling ever shows unexpected activation eviction (which would indicate a fundamental misunderstanding of the access pattern), the persistence API can be layered on without any changes to the grid design.

### 12.4 Large Model Scaling (Weights >> L2)

For very large models, the weight matrix far exceeds L2 capacity:

| Model | Weight Size (Q4_K) | L2 Ratio (4090) |
|-------|-------------------|-----------------|
| 7B    | 22 MB             | 34% ✓ fits      |
| 32B   | 88 MB             | 138%            |
| 70B   | 115 MB            | 180%            |
| 405B  | 340 MB            | 531%            |

When weights exceed L2, the K-loop streams weight data through L2. With `.cs` on weights (evict-first), old K-blocks are evicted as new ones arrive. The question is whether the streaming weight pressure can evict activations.

**Decision:** The design handles large models without modification. The activation working set (64 KB per wave) is protected by two factors: extreme recency (accessed every K-iteration by every SM, so always at the head of LRU) and extreme smallness (0.1% of L2, occupying at most a few hundred cache lines out of 512K). The `.cs` hint on weights explicitly marks them as evict-first, so under LRU pressure, weights evict before activations regardless of access pattern.

For models above ~200% L2 ratio (70B+), the weight streaming pressure is high enough that L2 provides diminishing benefit for weights themselves — they load from DRAM regardless. But the critical point is that **L2 activation sharing still functions** (activations are too small, too hot, and higher priority to evict). The hierarchical grid delivers its full activation benefit at every model size.

### 12.5 Interaction with the K-Loop

The K-loop is the innermost loop and is not directly controlled by grid dimensions. It runs sequentially within each block. The grid design and cache hints interact with the K-loop as follows:

The K dimension produces a streaming access pattern for both weights and activations. Each K-iteration loads a fresh weight slice and a fresh activation slice, uses them once for the tensor core MMA, and moves on. The `.cs` hint on weights ensures they have evict-first priority — as each K-block completes, its weight data can be evicted to make room for the next. The `.cg` hint on activations ensures they cache in L2 with normal priority for cross-SM sharing within the wave.

No special handling is needed for the K-loop. The cache hints provide the correct behavior: weights stream through L2 with low priority, activations persist in L2 with normal priority, and shared memory holds the active pipeline stages for the tensor core MMA.