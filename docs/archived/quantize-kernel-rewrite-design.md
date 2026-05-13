# Quantize Kernel Rewrite: API Draft

Status: Draft
Last updated: 2026-04-12

## 1. What I am building

Replace the current transpose+quantize kernel family with one Palette4-first kernel API that all existing reconcile callers can route through.

Core shape:
- One work item = one logical head-chunk conversion.
- Source endpoint has 4 palette lanes.
- Destination endpoint has 4 palette lanes.
- Routing is via the exact same 2-bit maps and ptr/fmt fields already used by decode metadata.

## 2. Hard constraints (non-negotiable)

- `chunk_size == 32`
- `head_dim % 4 == 0`
- Reuse decode metadata object layout exactly (`SlotHeader`, `TokenSlice`, `KvHead` byte layout)
- Use two decode-style metadata roots: `src_header_ptr` and `dst_header_ptr`
- Map domain is exactly palette IDs `0..=3`
- Source dtypes remain current set (float-like + R16 layout code)
- Destination formats remain current GGML quant formats used today
- **f16 SMEM intermediate** — all src values are staged through `__half smem_f16[32][128]` (8 KB). R16/F16 src uses cp.async (lossless). F32 and GGML quant src are dequantized and narrowed to f16 before writing to SMEM. Precision is sufficient: f16's 10-bit mantissa exceeds all quant dst formats (Q4 ≈ 4 bits, Q8 ≈ 8 bits), and F32 dst from a quant src has no extra real precision to preserve.
- **Src and dst may have independent pal_maps** — a merged translation table `xlat[128]` maps each dst dimension to its src dimension across palette boundaries. This is built once per block in Phase 0.
- **Warp-aligned dst palette groups** — dst palette 0 = warp 0 (threads 0–31), palette 1 = warp 1 (threads 32–63), etc. Required for cooperative `__shfl_down_sync` quant encode. (Identity routing from the write path satisfies this.) The src pal_map is unconstrained.
- Each launch converts either K or V, not both. The caller issues two launches.

Caller fit: these constraints match current single-item, scatter-gather, and scatter-gather-multi callers.

## 3. API

### 3.1 KvHead layout (from `slot_types.cuh`)

```text
KvHead(HD) bytes:
  [0        .. HD/4)      k_pal[HD/4]   // 2-bit packed palette indices
  [HD/4     .. HD/2)      v_pal[HD/4]
  [HD/2     .. HD/2+32)   k_ptr[4]      // 4 × u64 pointers
  [HD/2+32  .. HD/2+64)   v_ptr[4]
  [HD/2+64  .. HD/2+68)   k_fmt[4]      // 4 × u8 format codes
  [HD/2+68  .. HD/2+72)   v_fmt[4]

kv_head_byte_size(HD) = HD/2 + 72
```

The caller resolves `SlotHeader → slices_ptr → TokenSlice → KvHead` on the CPU before launch and passes flat arrays of pre-resolved KvHead device pointers. The kernel does no metadata traversal.

### 3.2 Device API

```c
// grid: <<<dim3(num_kv_heads, num_layers), head_dim, 0, stream>>>
// blockIdx.x = kv_head_idx, blockIdx.y = layer_idx, threadIdx.x = dimension index
extern "C" void run_quantize_palette4_convert(
  const uint8_t** src_kvhead_ptrs,  // [num_layers × num_kv_heads] KvHead device ptrs (src), row-major
  const uint8_t** dst_kvhead_ptrs,  // [num_layers × num_kv_heads] KvHead device ptrs (dst), row-major
  int32_t  num_kv_heads,
  int32_t  num_layers,
  int32_t  head_dim,
  int32_t  num_chunks,              // number of 32-token chunks per head to convert
  cudaStream_t stream
);
```

Source format is read from `k_fmt`/`v_fmt` inside each src `KvHead` — no separate dtype parameter. All `SlotHeader`/`TokenSlice` navigation is done by the caller when building the pointer arrays.

### 3.3 Rust API

```rust
pub fn quantize_palette4_convert(
  src_kvhead_ptrs: &CudaSlice<*const u8>,  // [num_layers × num_kv_heads] src KvHead ptrs, row-major
  dst_kvhead_ptrs: &CudaSlice<*const u8>,  // [num_layers × num_kv_heads] dst KvHead ptrs, row-major
  num_kv_heads: usize,
  num_layers: usize,
  head_dim: usize,
  num_chunks: usize,                        // number of 32-token chunks per head to convert
  stream: &CudaStream,
) -> Result<()>;
```

## 4. Grid, occupancy, and launch bounds

### 4.1 Grid shape

```
<<<dim3(num_kv_heads, num_layers), head_dim, 0, stream>>>
```

- `blockIdx.x` = kv_head_idx (fast dimension — heads vary within a layer wave)
- `blockIdx.y` = layer_idx
- `threadIdx.x` = dimension index `d ∈ [0, head_dim)`
- One block handles all chunks for one (layer, head) pair
- For Llama3 / standard models: `head_dim = 128` → 128 threads/block = **4 warps/block**

Inside each block the kernel loops:
1. **Outer loop** over chunks `c ∈ [0, num_chunks)` (passed as a kernel parameter)
2. **Per chunk**: Stage 1 loads all 4 src palettes into `smem_f16[32][128]`, Stage 2 all 4 warps encode simultaneously
   - Stage 1: thread `d` loads src column `d` (= src palette `d/32`, local dim `d%32`) for all 32 tokens into `smem_f16[t][d]`
   - Stage 2: thread `d` reads `smem_f16[t][ xlat[d] ]` where `xlat[d]` remaps dst dim to src SMEM column, then encodes to dst arena

### 4.1a Buffer construction

The caller builds the two pointer arrays on the CPU before launch:

```rust
// Row-major [num_layers][num_kv_heads] — index = layer * num_kv_heads + head
let src_ptrs: Vec<*const u8> = (0..num_layers)
    .flat_map(|l| (0..num_kv_heads).map(move |h| src_kvhead_ptr(l, h)))
    .collect();
let dst_ptrs: Vec<*const u8> = (0..num_layers)
    .flat_map(|l| (0..num_kv_heads).map(move |h| dst_kvhead_ptr(l, h)))
    .collect();
```

The kernel recovers the pointer via:
```c
int job = blockIdx.y * num_kv_heads + blockIdx.x;  // layer * num_kv_heads + head
const uint8_t* src_head = src_kvhead_ptrs[job];
const uint8_t* dst_head = dst_kvhead_ptrs[job];
```

Typical grid sizes: Llama 3 8B = 32 × 8 = **256 blocks**, 70B = 80 × 8 = **640 blocks**, 405B = 126 × 8 = **1 008 blocks**. On a 4090 (128 SMs) the 70B grid gives 5× SM coverage — well above the latency-hiding threshold.

### 4.2 Occupancy and launch bounds

Memory-bandwidth-bound kernel. Single SMEM layout for all format combinations:

**SMEM layout (fixed, all paths)**
- `__half smem_f16[32][128]` — all 4 src palettes staged simultaneously = 32 × 128 × 2 = **8,192 bytes**
- `uint8_t smem_xlat[128]` — merged src→dst dimension translation table = **128 bytes**
- Total: **8,320 bytes/block**

**Launch bounds**
- `__launch_bounds__(128, 12)` — 12 blocks/SM
- 8,320 B × 12 = 99.8 KB — fits within 4090's 128 KB and A100's 164 KB SMEM
- Register budget: ~53 regs/thread (65,536 regs / 12 blocks / 128 threads). Generous — no tight register pressure.
- Grid is always the binding constraint. Llama-3-8B: 256 blocks / 128 SMs = 2 blocks/SM. Even 70B (640 blocks) peaks at ~5 blocks/SM. The launch_bounds hint of 12 is never reached in practice.

| GPU | SMEM/block | Blocks/SM (SMEM limit) | Blocks/SM (warp limit) | Typical grid fill |
|-----|-----------|------------------------|------------------------|----|
| RTX 4090 (128 SMs) | 8.3 KB | 15 | 12 | 2–5 |
| A100 (108 SMs) | 8.3 KB | 19 | 16 | 2–6 |

**cp.async fast path**: when src is R16 or F16, each palette's 32×32 sub-band is contiguous in memory (2 KB). All 128 threads cooperatively DMA all 4 palettes = 4 × 2 KB = 8 KB via `cp.async` directly into `smem_f16`. This is the most common case (R16/F16 arenas as input, quant arenas as output).

**Dequant slow path**: when src is F32 or GGML quant, each thread dequantizes its src value and writes `__float2half()` into `smem_f16`. No cp.async for these formats.

### 4.3 Data flow diagram

One block = one (layer, head) pair. `threadIdx.x` = dimension index `d` in `[0, HD)`. The per-token projection below executes inside the chunk x token loops described in section 4.1.

All 4 src palettes are loaded into SMEM simultaneously. The f16 intermediate means cp.async can DMA directly for R16/F16 src (the common case), while F32 and GGML quant src are dequantized and narrowed to f16 before writing to SMEM. After a single `__syncthreads()`, all 4 warps encode in parallel -- each warp owns one dst palette and reads from SMEM via the xlat remapping table.

```
 src KvHead  (loaded once per block)
 +-------------------------------------------------------------+
 |  k_pal[HD/4]   (2-bit packed, 4 dims per byte)              |
 |  k_ptr[4]      (arena base pointers, one per palette)       |
 |  k_fmt[4]      (ArenaFormat per palette)                    |
 +----------+----------------------------------------------+----+
            |                                              |
            |  Phase 0 (once per block):                   |
            |    Build smem_xlat[128] from INDEPENDENT      |
            |    src_pal_map and dst_pal_map. Each dst      |
            |    thread d computes which src SMEM location   |
            |    holds its value.                           |
            |                                              |
            |  for each chunk c:                           |
            |                                              |
            |    Stage 1: Load ALL 4 src palettes -> smem_f16[32][128]
            |      cp.async for R16/F16 src (4 x 2 KB = 8 KB DMA)
            |      dequant + __float2half() for F32/GGML src
            |      __syncthreads()
            |
            v
 __shared__ __half smem_f16[32][128]    // all 4 palettes at once
 +--------------------------------------------------------------+
 |  smem_f16[0][0..31]   = palette 0, token 0                  |
 |  smem_f16[0][32..63]  = palette 1, token 0                  |
 |  smem_f16[0][64..95]  = palette 2, token 0                  |
 |  smem_f16[0][96..127] = palette 3, token 0                  |
 |  ...                                                         |
 |  smem_f16[31][0..127] = all 4 palettes, token 31            |
 |  (32 tokens x 128 dims, all palettes, dequantized to f16)   |
 +----------+---------------------------------------------------+
            |
            |    Stage 2: ALL 4 warps encode SIMULTANEOUSLY
            |      warp w owns dst palette w
            |      each thread reads smem_f16[t][ xlat[d] ]
            |      where xlat[d] maps dst dim -> src SMEM column
            |      encode val -> dst arena
            |
            v
 4 dst arenas (palette 0..3, each a contiguous quantized block)
```

Notes:
- **Src and dst have independent pal_maps** -- the xlat[128] table merges them. Each dst dimension d (warp_id = dst_pal, lane_id = dst_local_d) looks up its corresponding src location in the 32x128 SMEM buffer via xlat[d].
- All 4 warps are active in Stage 2 simultaneously -- no 75% idle threads. The f16 intermediate makes this affordable at 8 KB SMEM (vs 16 KB for f32).
- Scale reduction for quant formats (e.g. Q4_0 absmax) uses `__shfl_down_sync` within each warp.
- `my_pal = warp_id` (warp-aligned dst palette constraint from section 2), `my_local_d = threadIdx.x % 32` -- precomputed into `smem_xlat[d]` in Phase 0 as a merged src-to-dst mapping.

### 4.4 All-palette pipeline

The block loads all 4 src palettes into SMEM at once, then all 4 warps encode simultaneously. No per-palette iteration in Stage 2.

```
Phase 0  | Build smem_xlat[128] from independent src/dst pal_maps (once per block)
         v
for each chunk c:
  Stage 1  | Load ALL 4 src palettes -> smem_f16[32][128]
           |   For each src palette p:
           |     R16/F16: all 128 threads issue cp.async (2 KB per palette)
           |     F32:     each thread dequants + __float2half() -> smem_f16
           |     GGML:    each thread dequants + __float2half() -> smem_f16
           |   __syncthreads()  // one barrier for all 4 palettes
           v
  Stage 2  | ALL 4 warps encode simultaneously
           |   warp w reads smem_f16[t][ xlat[w*32 + lane_id] ] for each token t
           |   encode to dst palette w using dst_fmt[w]
           v
  __syncthreads()  // SMEM safe for next chunk
```

The key structural change from the old per-palette pipeline: **Stage 2 is now fully parallel across all 4 warps.** The format switch still happens once per chunk per warp (4 switches total across the block), not per token. The inner token loop is a tight scalar/vectorised sequence with no branches.

#### Phase 0 -- build merged translation buffer (once per block)

Each thread d reads BOTH the src pal_map and the dst pal_map and computes the merged mapping. For thread d:
- `dst_pal = threadIdx.x / 32` (= warp_id, guaranteed by warp-aligned dst constraint)
- `dst_local_d = threadIdx.x % 32` (= lane_id within dst palette)
- Look up which global dimension this dst slot corresponds to (from dst pal_map)
- Find that global dimension in the src pal_map to get `{src_pal, src_local_d}`
- `smem_xlat[d] = src_pal * 32 + src_local_d` -- an index into the flat smem_f16[t][0..127] row

After `__syncthreads()`, all Phase 0 temps are dead and reclaimed by the compiler.

Total Phase 0 SMEM cost: 128 bytes (`smem_xlat`).

#### Stage 1 -- load all 4 src palettes into SMEM

All 128 threads cooperate to load all 4 palettes for the current chunk into `smem_f16[32][128]`.

For **R16/F16 src** (the common, fast path): cp.async DMA copies each palette's chunk. Each palette is 32 x 32 x 2 = 2 KB. 4 palettes = 8 KB total. All 128 threads participate in the DMA across all 4 palettes -- this is a block-wide cooperative copy that maximizes memory bandwidth. Thread `d` issues cp.async for src column `d` (palette `d/32`, local dim `d%32`) across all 32 tokens into `smem_f16[t][d]`. The cp.async goes directly into `smem_f16` with no format conversion needed (R16 and F16 are both f16 in memory). After all DMA is issued: `cp.async.commit_group()`, `cp.async.wait_group<0>()`, then `__syncthreads()`.

For **F32 src**: thread `d` reads src column `d` (palette `d/32`, local dim `d%32`) for all 32 tokens from global memory, narrows each with `__float2half()`, and writes to `smem_f16[t][d]`. No cp.async -- the narrowing requires a register-side operation.

For **GGML quant src** (Q4_0, Q8_0, etc.): thread `d` dequantizes src column `d` for all 32 tokens using the existing `dequant_element<T>()` / `ArenaAccessor` infrastructure from `convert_all.cuh`, narrows to f16 with `__float2half()`, and writes to `smem_f16[t][d]`.

Mixed-format note: different src palettes may have different formats. The src format for palette p is read from `src_head.k_fmt[p]`, and each thread dispatches to the appropriate load path for the palette(s) it is assigned to load.

After `__syncthreads()`, `smem_f16[t][0..127]` is valid for all 32 tokens x 128 dims.

#### Stage 2 -- all warps encode simultaneously

All 4 warps are active. Warp w encodes dst palette w. Each thread reads `smem_f16[t][ xlat[w*32 + lane_id] ]` for each token t, converts to float with `__half2float()`, and encodes to the dst arena.

For **scalar dst** (F32, F16): immediate write per token. `__half2float` then store (F32) or direct copy (F16 -- already f16 in SMEM).

For **quant dst** (Q4_0, Q8_0, etc.): cooperative encode within the 32 threads of the warp. The existing `quantize_block_*` functions expect 32 float values distributed one per warp lane, with `__shfl_down_sync` for absmax reduction. Each token's 32 dims across the 32 warp lanes = one quant block.

```c
// Stage 2 -- all 4 warps encode simultaneously
uint8_t src_col = smem_xlat[threadIdx.x];  // merged src location

for (int t = 0; t < CHUNK_SIZE; t++) {
    float val = __half2float(smem_f16[t][src_col]);
    // ... encode val into dst arena using dst_fmt[my_pal] ...
}
```

Format dispatch uses a switch on `dst_fmt[my_pal]`:

```c
int my_pal = threadIdx.x / 32;    // warp_id = dst palette
uint8_t src_col = smem_xlat[threadIdx.x];

switch (dst_head.k_fmt[my_pal]) {
    case FMT_F32:  encode_pal<FMT_F32> (smem_f16, src_col, dst_head, my_pal, c); break;
    case FMT_F16:  encode_pal<FMT_F16> (smem_f16, src_col, dst_head, my_pal, c); break;
    case FMT_Q4_0: encode_pal<FMT_Q4_0>(smem_f16, src_col, dst_head, my_pal, c); break;
    case FMT_Q8_0: encode_pal<FMT_Q8_0>(smem_f16, src_col, dst_head, my_pal, c); break;
    // ... other formats ...
}
__syncthreads();  // SMEM safe for next chunk
```

Key properties:
- The switch executes **once per chunk per warp**. Branch overhead is negligible.
- Each dst palette can have a **different format** -- all handled in a single pass per chunk.
- All 4 warps are active simultaneously -- zero idle threads in Stage 2.
- Register budget is generous at ~53 regs/thread. No register pressure concerns.

### 4.5 Value translation and quant encode

#### Translation buffer (Phase 0)

The src and dst may have independent pal_maps (section 2). The xlat table merges them into a single lookup: for each dst dimension d, xlat[d] gives the column index into the flat `smem_f16[t][0..127]` row where that dimension's src value lives.

```c
// Phase 0: build merged translation buffer (once per block)
__shared__ uint8_t smem_xlat[128];  // HD = 128
__shared__ __half  smem_f16[CHUNK_SIZE][128];  // 32 x 128

const uint8_t* src_pal_map = kvhead_k_pal<HD>(src_head);
const uint8_t* dst_pal_map = kvhead_k_pal<HD>(dst_head);

// Step 1: Build inverse src map -- for each global dim g, find {src_pal, src_local_d}
// The src pal_map assigns each of 128 dims to a palette 0-3.
// Within each palette, dims are stored contiguously in rank order.
// src_col[g] = src_pal * 32 + rank_within_src_pal

// Step 2: For each dst thread d (= dst_pal * 32 + dst_local_d):
//   Find which global dim g this dst slot maps to (via dst pal_map rank counting)
//   Look up src_col[g] from step 1
//   smem_xlat[d] = src_col[g]

// With warp-aligned DST palettes (section 2 constraint):
int my_dst_pal = threadIdx.x / 32;   // warp_id
int my_dst_local_d = threadIdx.x % 32;  // lane_id

// Find the global dim that is the my_dst_local_d-th member of dst palette my_dst_pal
int global_d = find_nth_dim_in_pal(dst_pal_map, my_dst_pal, my_dst_local_d);

// Find where global_d lives in the src layout
int src_pal = (src_pal_map[global_d / 4] >> (2 * (global_d % 4))) & 0x3;
int src_local_d = rank_in_pal(src_pal_map, src_pal, global_d);

smem_xlat[threadIdx.x] = (uint8_t)(src_pal * 32 + src_local_d);
__syncthreads();
```

When src and dst share the same pal_map (the common case), `xlat[d] = d` -- identity mapping. The xlat build still executes but is trivially cheap.

Helper functions for pal_map navigation:

```c
// Find the n-th dimension assigned to palette p in the pal_map
__device__ int find_nth_dim_in_pal(const uint8_t* pal_map, int p, int n) {
    int count = 0;
    for (int g = 0; g < 128; g++) {
        int pal = (pal_map[g / 4] >> (2 * (g % 4))) & 0x3;
        if (pal == p) {
            if (count == n) return g;
            count++;
        }
    }
    return -1;  // should not happen with valid pal_map
}

// Find the rank of global_d within its palette in the pal_map
__device__ int rank_in_pal(const uint8_t* pal_map, int p, int global_d) {
    int rank = 0;
    for (int g = 0; g < global_d; g++) {
        int pal = (pal_map[g / 4] >> (2 * (g % 4))) & 0x3;
        if (pal == p) rank++;
    }
    return rank;
}
```

These loops iterate up to 128 times per thread (128 dims). At ~1 cycle/iteration this is ~128 cycles -- negligible vs the per-chunk memory traffic. They execute once per block lifetime.

#### Src dequant (Stage 1)

All src values are staged through `smem_f16[32][128]` as f16:

| Src format | Load method | Notes |
|-----------|-------------|-------|
| R16 | cp.async DMA -> `smem_f16` | 2 KB per palette, lossless. Most common case. |
| F16 | cp.async DMA -> `smem_f16` | 2 KB per palette, lossless. |
| F32 | Global load + `__float2half()` -> `smem_f16` | 4B read, 2B write. Narrowing is acceptable (see section 2). |
| Q4_0, Q8_0, etc. | `dequant_element()` + `__float2half()` -> `smem_f16` | Uses `ArenaAccessor` from `convert_all.cuh` |

For the cp.async fast path (R16/F16), all 128 threads cooperatively DMA all 4 palettes. The src data is already f16 in memory, so the DMA lands directly in `smem_f16` with no conversion. This covers the most common use case: R16 KV cache (from decode) being converted to quant format for long-term storage.

For F32 and GGML quant src, each thread dequantizes its assigned src elements and writes `__float2half(result)` into `smem_f16`. The f16 narrowing is justified in section 2 (10-bit mantissa exceeds all quant dst precisions).

#### Dst quant encode (Stage 2)

Each warp reads from `smem_f16` via the xlat table and encodes to its dst palette.

For **scalar dst** (F32/F16): each thread reads `__half2float(smem_f16[t][src_col])` and writes directly. For F16 dst, the value is already f16 in SMEM -- direct copy with no conversion.

For **GGML quant dst**: the 32 warp threads cooperatively encode one quant block per token. Each thread reads `__half2float(smem_f16[t][src_col])` to get a float, then the existing `quantize_block_*` functions handle the cooperative encode:

1. Each lane has its float value from SMEM
2. `__shfl_down_sync` reduces across 32 lanes for absmax
3. Each lane quantizes its value using the shared scale factor
4. Lanes cooperatively pack the quantized values into the output block

Since dst palette = warp (section 2 constraint), `threadIdx.x % 32 == lane_id` -- the shuffle topology is correct with no masking issues.


## 5. Migration plan (minimal)

1. Add adapters from current callers to `QuantizeJobRef` over decode-style metadata buffers.
2. Run A/B correctness check: old kernel vs new API on same items.
3. Flip default to new API; keep legacy path behind debug flag for rollback.

## 6. Open review points

- Require bitwise parity for all formats or bounded numeric parity?
