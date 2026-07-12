# Prefill Optimization — INT8 Prefix Attention with GQA-Packed Tiles

> **Status — BUILT (Phases 1–3 + cutover); measured 5.2–18.6×.** The kernel
> is implemented in `candle-kernels/src/paged-prefill/paged_prefill_int8_kernel.cuh`
> (entry `run_paged_prefill_int8`, host `paged_prefill_batched_int8`), and
> production prefills over an existing prefix dispatch to it in
> `batched_layer.rs` (fresh-context prefills keep the FP16 kernel's no-prefix
> async path, per §9.3). Validated by the `prefill_ab` differential harness
> (23 scenarios + real-substrate A/B green) AND the end-to-end duplication
> control (`duplication_replay`: real Qwen3-30B, 13 live turns, 0/6
> duplicated, recall correct, every production prefill through this kernel).
> Measured on RTX 4090 Mobile vs the FP16-dequant path (§13 appendix,
> locked clocks): q64/prefix8k **9.8×**, q8/prefix8k **18.6×**,
> q256/prefix2k **5.2×**, q64/prefix2k **5.6×**.
> Implementation deviations from the design are recorded in §13. One
> amendment to §12 Phase 4: the FP16 prefix path is **retained as the A/B
> harness's reference backend** — the differential oracle needs both kernels
> callable; deletion would destroy the harness. The dead FP8/`write_offset_
> shifts`/F32-no-op cleanups of Phase 4 remain to do.

---

## 1. Abstract

Prefill over a long quantized prefix is the latency-defining operation of
Zen Code: every dialogue turn re-attends over tens of thousands of
provenance-selected, palette-quantized KV tokens before its first token can
decode. Today that path **dequantizes the entire prefix to FP16 in shared
memory and runs FP16 MMA over it** — paying a dequant round-trip, a helper
warp, ~5 block barriers per 32-token tile, and 4× redundant K/V global
traffic across the GQA group, all to feed tensor cores at half their INT8
rate with 96 threads per block.

The decode kernel already proves the alternative on this exact arena: INT8
`m16n8k32` MMA with per-palette scale fixup, V **read through** as int8
with no FP round-trip, and per-palette Q/K quantization — but decode is
M-starved (≤8 query-head rows per MMA, zero-padded to 16). Prefill is the
workload that fills the M dimension.

This design rebuilds the prefix path around four moves:

1. **INT8 compute, quantized-at-rest operands.** QK and PV run int8 MMA.
   V feeds the MMA straight from arena bytes (read-through); K takes the
   minimal FP detour that inline RoPE requires (dequant → rotate →
   requantize), amortized once per staged tile across the whole GQA group.
2. **GQA-packed M.** MMA rows are (query-token × head-in-group) pairs. One
   K/V tile load serves all 8 query heads of a Qwen3-MoE group — ~4× less
   prefix global traffic and 8× more MMA work per staged byte than the
   current warp-per-head / head-block scheme.
3. **Slice-aligned tiling.** A KV tile is one `TokenSlice` (≤ 32 tokens,
   gap-aware), never a cum-token window. Tiles cannot straddle slices, so
   the hoisted two-table palette routing and the ≥3-slice fallback are
   deleted, not ported: the straddle bug class becomes unrepresentable.
4. **Symmetric warp workers + split-KV.** Eight warps per block, all
   staging and all computing (the FP16-postprocess helper warp dissolves);
   block-level split-KV over the prefix with the existing LSE-combine
   kernel fills the GPU when `q_len` is short.

Design target: **3–5× prefix-attention throughput** at HD128 on SM89
(RTX 4090 Mobile), carrying unchanged to SM120 (RTX 5090) whose
hardware block-scaled MMA the staging layer is shaped to feed later (§10).

---

## 2. The Workload

The production configuration this kernel serves:

| Parameter | Value |
|---|---|
| Model | Qwen3-30B-A3B (MoE), later Qwen3-235B-A22B |
| Heads | 32 query / 4 KV (`hpg = 8`), `HEAD_DIM = 128` |
| KV at rest | Palette-quantized arena, `CHUNK_SIZE = 32`, `N_PALETTE = 4` |
| Compression | C4–C5 typical (Q4/Q8-family K and V), C0 boundary sections |
| Hardware | SM89 (RTX 4090 Mobile 16 GB) now; 2× SM120 (RTX 5090 32 GB) mid-2026 |

The hot prefill shape is **short q over long quantized prefix**: a dialogue
turn of 32–512 tokens attending over a 10k–100k-token projection (system
sections + selected turns + utility layers), injected as sealed arena KV.
Fresh-prompt ingestion (long q, no prefix) exists but is not the
bottleneck and keeps its current FP16 path (§9.3).

Secondary shapes that must not regress: section seal prefills (repo_map /
code_read ingestion, `disable_reprojection`), glue prefills (short, handled
by the separate glue kernel — out of scope, same arena), and wave-batched
multi-session prefills.

---

## 3. Current State — What This Replaces

### 3.1 The prefix path today (`paged_prefill_kernel.cuh`)

FlashAttention-2 structure: FP16/BF16 `m16n8k16` MMA, register-resident P,
online softmax with folded beta, 2-stage cp.async ring, reverse tile
iteration. The `HAS_PREFIX` body (lines ~3241–4187) adds:

- **Full dequant to FP16**: quant blocks are cp.async-staged raw, then a
  postprocess pass (`postprocess_quant_tile_prefix`) dequantizes every
  element into FP16 smem before any MMA — on a helper warp when one
  exists (HD128: 2 compute warps + 1 helper = 96 threads).
- **Hoisted palette tables ×2**: per-tile `s_pal_table_k/v` plus the
  straddle twin `s_pal_table_k2/v2` (commit `9eee0ddd`), with a
  per-element `prefill_pal_rank` fallback for tiles spanning ≥3 slices —
  the machinery exists because tiles are cum-token windows that cross
  slice boundaries.
- **Cost structure per 32-token tile**: ~5 `__syncthreads` (validity,
  KV-write scan, K-RoPE, pipeline wait, compute), a full-width FP16
  dequant, and RoPE cos/sin re-fetched per element via `__ldg`.
- **GQA fan-out across blocks**: `head_blocks_per_kv = 4` at HD128 — the
  same prefix K/V tile crosses the L2/global boundary four times per
  group.
- **Worst-case grid**: `grid.x = ceil(max_blocks·32 / 32)` launches no-op
  blocks for every short row in a mixed batch.

### 3.2 What the decode kernel already proves (`int8_decode_kernel.cuh`)

On the **same arena, same palettes, same formats**, decode runs:

- INT8 `m16n8k32` QK with fragment loaders `load_a_frag_m16k32` /
  `load_b_frag_n8k32` and per-`(palette, token)` scale fixup on the int32
  accumulator (`mma_wrappers.cuh`).
- **Palette-as-k-step**: `USE_MMA_QK` requires `SUB_HEAD_DIM == 32` — each
  palette's ~32 dims are one MMA k-step; scales apply per palette after
  the MMA. The palette scheme *is* the k-tiling.
- **V read-through**: when every palette's V format is in the
  int8-readthrough set (Q8_0/Q4_0/Q5_0/Q2_0/Q3_0/Q4_KS/Q8_KS/Q8_1/Q2_S/
  Q1_S/Q1_A/Q0/Q0_M2/Q0_M4/Q0_X), V bytes feed the PV MMA directly with
  per-dim block scales — no FP arithmetic on the V side at all.
- **K's mandatory FP detour**: dequant → RoPE → per-palette int8 requant,
  per tile. RoPE forces this; nothing else does.
- **Gap-aware slice walk**: `slice_eff_len` / `tile_to_slice` — tiles
  live inside one slice, skip sealed-gap tails, and address the write
  slice at its physical position.
- **Split-KV + LSE combine**: grid-Z shards the KV walk;
  `int8_decode_combine_kernel` merges `(ΣwV, m, l)` partials in base-2
  log-sum-exp; a persistent partial pool provides the scratch.
- **The starvation it cannot fix**: the bmma path fills at most `hpg = 8`
  of 16 MMA M-rows and zero-pads the rest. Half the int8 tensor-core
  throughput is structurally unreachable *for decode*.

Everything in that list except the last line is reused by this design.

---

## 4. Research Grounding

Four external results de-risk the design (full brief in the project log):

- **SageAttention 1/2/2++** (thu-ml, ICLR/ICML '25): INT8 Q·Kᵀ with
  per-block/per-thread scales and FP8/INT8 P·V with two-level accumulation
  loses no end-to-end quality on language models and delivers 2.7–3.9×
  attention throughput on RTX 4090-class parts. Their cost — quantizing
  FP16 inputs at runtime, every call — is precisely what our
  quantized-at-rest arena amortizes to zero.
- **FlashInfer / Bifurcated Attention**: packing the GQA group into the
  MMA M-dimension ("grouped" prefill) is established practice; K/V traffic
  amortizes across the group.
- **LeanAttention / FlashInfer stream-K**: equal-work partitioning over a
  flattened tile space beats worst-case grids under ragged batches. This
  design takes the split-KV-Z step now (§8) and leaves the persistent
  worklist as a future document.
- **SM120 (consumer Blackwell)**: warp-level `mma.sync` only — no
  TMEM/tcgen05/WGMMA — plus hardware **block-scaled MMA** (`kind::mxf8f6f4
  .block_scale.scale_vec::1X.m16n8k32`, one UE8M0 scale per 32 elements),
  99 KiB smem/SM. Everything built here on `mma.sync` carries unchanged;
  the 32-element scale granularity matches the arena's quant blocks 1:1
  (§10).

---

## 5. Design — Work Decomposition

### 5.1 GQA-packed M

An **M-row is a (query-token, head-in-group) pair.** For `hpg = 8`,
`BLOCK_M_TOK = 16` query tokens make `16 × 8 = 128` M-rows per block —
eight `m16` MMA row-tiles. Every M-row of the block attends the **same KV
head**, so one staged K/V tile serves all 128 rows.

- Causal masking depends only on the row's token component:
  `row_max_k(row) = min(prefix_len + q_pos(tok) + 1, kv_len)` — identical
  for the 8 head-variants of a token. The mask term is computed once per
  token and broadcast.
- The generic M-row count is `BLOCK_M_TOK × hpg`, with `BLOCK_M_TOK`
  chosen per instantiation so `BLOCK_M_TOK × hpg ∈ [64, 128]`. Degenerate
  q (1 token) yields `hpg` rows — exactly decode's bmma shape. Prefill and
  decode become two points on one continuum.

### 5.2 Warp mapping — symmetric workers, private softmax state

**8 warps, 256 threads, no helper warp.** Per KV tile, the block runs two
roles in sequence, all warps participating in both:

1. **Stage** (cooperative): load the next tile's K/V into int8 smem slabs,
   apply RoPE to K, populate scale tables (§6).
2. **Compute** (independent): each warp owns one `m16` row-tile (16 M-rows)
   and walks all four `n8` column-slices of the staged 32-token tile,
   keeping its online-softmax state (`m`, `l`, O-accumulators) private in
   registers — exactly the current kernel's per-warp flash state, with
   rows redefined.

No cross-warp softmax merge exists (warps own disjoint M-rows). The
decode-stripe pattern of striping warps across KV and merging states is
**explicitly rejected**: it solves M-starvation, which prefill does not
have. Barrier count per tile drops from ~5 to 2 (stage-publish,
ring-swap).

### 5.3 Grid

```
grid.x = ceil(q_len_max_in_batch / BLOCK_M_TOK)   (per-seq early-exit on real q_len)
grid.y = n_kv_head                                 (4 — no head_blocks factor)
grid.z = batch_size × num_splits                   (split-KV shards, §8)
```

Removing the `head_blocks_per_kv` factor from `grid.y` is the GQA-packing
dividend: at HD128 the grid shrinks 4× while each block does 4× the MMA
work per staged byte. `q_len`-aware `grid.x` sizing (host computes the max
over the batch's true `q_lens`, not `max_blocks`) removes the worst-case
no-op blocks.

---

## 6. Design — The INT8 Tile Pipeline

### 6.1 Slice-aligned tiling (straddles become unrepresentable)

A KV tile is **one `TokenSlice`**: `tile_to_slice` is the decode kernel's
monotonic forward cursor (`scan_s`/`scan_base` — the stripe/bmma variant,
not the O(n²) warp=head rescan), and each tile covers
`slice_eff_len(s) ≤ 32` valid tokens. Consequences:

- **One palette table per tile.** `s_pal_table_k2/v2`, the straddle-twin
  selection logic, and the ≥3-slice per-element `prefill_pal_rank`
  fallback are deleted. A tile has exactly one `KvHead`, one set of four
  `(ptr, fmt, scale)` triples per side.
- Partial slices (sealed-gap tails, the final open chunk) mask their
  invalid column slots via the per-tile validity mask — MMA runs full
  `n8` width and masked lanes contribute `-inf` scores pre-softmax, which
  the online softmax already tolerates (current kernel behavior).
- The wasted MMA lanes on partial slices are bounded: one partial slice
  per sealed section plus the open tail — noise against the full chunks
  of a long prefix.

### 6.2 Staging K: dequant → RoPE → requantize (the only FP touch)

Per tile, cooperatively across all 8 warps:

1. **Route & load**: raw K quant blocks are read from the arena. Blocks
   are dim-major (one block = one dim × 32 tokens), palette-regioned; the
   single hoisted palette table (byte-packed `palette≪6 | rank`) routes
   each dim to its region — retained from the current kernel, minus the
   straddle twin.
2. **Dequant to FP32 registers** via the existing `BlockConverter`
   dispatch (`convert_all.cuh`) — per-format, per-element, exact.
3. **RoPE inline**: rotate with cos/sin staged **once per tile into smem**
   (`smem_rope_cs[32][HEAD_DIM]`, FP16 pairs — 8 KB at HD128), replacing
   today's per-element `__ldg`. Position = slice base + within-slice
   offset + `rope_offsets[b]`; K in the arena stays unrotated
   (position-independent storage is mandatory: reprojection re-positions
   tokens and provenance signatures read raw K).
4. **Requantize to int8 per palette**: per-`(palette, token)` max-abs
   reduced with `__shfl_xor` over the palette's lanes (decode's §Q/K quant,
   verbatim), scale into `smem_k_scale[stage][32][N_PALETTE]`, int8 bytes
   into the `[token][dim]` K slab.

This is decode's K path, executed once per tile per *group* instead of
once per tile per *decode step* — and serving 128 M-rows instead of 8.

### 6.3 Staging V: read-through, zero FP arithmetic

When every V palette format of the tile's `KvHead` is int8-readthrough
(the production C4–C5 configuration always is), V staging is **byte
movement only**: raw block bytes scatter into the `[token][dim]` V slab
and per-dim block scales load into `smem_v_dim_scale[stage][HEAD_DIM]` —
`load_head_int8_readthrough_typed` reused directly. Non-readthrough
formats (F16/BF16/FP8/R16 V) take dequant → per-token int8 quant, exactly
decode's fallback. The per-tile `readthrough` flag gates the PV epilogue's
scale application, as in decode.

### 6.4 Staging Q: once per block

Q rows load once at block start (as today), RoPE-rotate in registers, then
quantize **per palette per M-row** to int8 (`smem_q8[128][HEAD_DIM]` +
`smem_q_scale[128][N_PALETTE]` — 16 KB + 2 KB at full packing). The
per-palette granularity matches the MMA k-step so the scale fixup is one
FMA per accumulator tile per palette. Per-thread (SageAttention2-style)
finer granularity is **not** adopted: the accuracy gate in §11 measures
per-palette first, and the decode kernel's identical choice has shipped
without quality regression.

The q-tile's **own new tokens** (the suffix) stage through the same path —
their fresh FP16 K/V quantizes on stage. One MMA type governs the whole
walk, and attending int8 to your own turn matches what every later turn
will see once those tokens seal into the arena.

### 6.5 QK, softmax, PV

Per warp, per staged tile:

- **QK**: for each of the four `n8` column-slices, four palette k-steps of
  `mma.sync.m16n8k32.s8.s8.s32`; after each palette's MMA, fixup
  `acc_f32 += (float)acc_i32 × scale_Q[row][p] × scale_K[tok][p]`.
  Ragged palettes (pal_map not 32/32/32/32) pad their k-lanes with zero
  bytes — zero contributions are exact in integer arithmetic.
- **Softmax**: unchanged from the current kernel — FP32 register-resident
  online softmax, folded beta, `fast_exp`, `__shfl_xor` reductions over
  the 4 column lanes. Scores scale by `softmax_scale` at fixup time.
- **PV**: P quantizes per M-row to int8 using the row max already produced
  by the online softmax (the scale is free); `m16n8k32` against the V
  slab; two-level accumulation — int32 MMA accumulator → FP32 O-registers
  with `β_scale[row] × v_scale` fixup (per-dim block scale when
  readthrough, per-token otherwise). This is decode's PV, M-filled.

### 6.6 KV write — into staging, once

Prefill's fused arena write (sealing the turn's new K/V) moves from the
per-tile 32-row rescan into the **staging phase of the suffix tiles
only**: when a staged tile overlaps the write range, the staging warps
write the new tokens' K (unrotated, R16 Q-capture preserved via
`store_kv_chunk_arena`) and V blocks as they pass through registers.
With GQA packing there is one block per (q-tile, kv_head) — the write
executes exactly once per token per layer with **no** cross-block
duplication; split-KV shards other than `z == 0` skip it (decode's
idempotent-rewrite waste is not inherited).

---

## 7. Shared Memory Budget & Launch (HD128 reference)

| Buffer | Size |
|---|---|
| K int8 slab × 2 stages | 2 × 32 × 128 = 8 KB |
| V int8 slab × 2 stages | 8 KB |
| Q int8 (128 M-rows) | 16 KB |
| RoPE cos/sin tile (FP16) | 8 KB |
| K scales (2 × 32 × 4 F32) + V dim scales (2 × 128 F32) | 2 KB |
| Q scales (128 × 4 F32) | 2 KB |
| Palette table, validity, per-row metadata | ~1 KB |
| **Total** | **~45 KB** |

Two blocks co-resident at 99–100 KB/SM — preserved from the current
design point — now with **512 threads/SM resident** against today's
128–192. `__launch_bounds__(256, 4)` targets ≤ 64 registers/thread;
the O-accumulator footprint (16 rows × 128 dims / 32 lanes × VEC) matches
decode's bmma budget, which ships within that bound.

cp.async stages the raw arena bytes (K quant blocks, readthrough V
blocks); the K FP-detour runs in registers between ring stages. Two
`__syncthreads` per tile (stage publish, ring swap).

---

## 8. Split-KV Scheduling

For `q_len = 32`, the packed grid is `1 × 4 × B` blocks — starved on 76
SMs at small batch. The prefix walk therefore shards across `grid.z`:

- `num_splits = clamp(ceil(SM_count × 3 × 2 / (q_tiles × n_kv_head ×
  batch)), 1, 32)` — decode's heuristic, reused.
- Shards `z > 0` emit un-normalized `(ΣwV, m, l)` partials per M-row into
  the persistent partial pool (`fused_attn_partial_pool`, generalized to
  `q_len × n_head` rows); `int8_decode_combine_kernel`'s base-2 LSE merge
  runs as today, one block per output row.
- `num_splits == 1` (long q, big batch) writes O directly and skips the
  combine — fresh-ingestion shapes pay nothing.

The stream-K persistent worklist (LeanAttention-style) and POD-style
hybrid prefill/decode fusion remain **future documents**; the split-KV-Z
step reuses tested machinery and removes the immediate starvation.

---

## 9. Numerical Accuracy

### 9.1 Why int8 QK/PV holds

The SageAttention line establishes INT8 Q·Kᵀ + quantized P·V with
two-level accumulation as accuracy-neutral end-to-end on language models.
This design is strictly *finer*-grained than their published configuration: scales
per (palette ≈ 32 dims) × token for K, per palette × M-row for Q, per
M-row for P, per 32-element block for readthrough V — and the K operand's
quantization error is **already accounted for** by the compression
policy's cosine-distance thresholds (the arena bytes are the ground truth
this system attends to everywhere else; decode has attended int8 over
them since the int8 decode kernel shipped).

### 9.2 O(1) error framing (paper)

The kernel makes the compressed domain the *compute* domain: attention
error per step remains bounded by the per-block quantization error the
`CompressionPolicy` already certifies at seal time, independent of context
depth — the C-level chosen by provenance selection now bounds the MMA
precision end to end. The requantize-after-RoPE step adds one bounded
int8 rounding per (tile, palette), identical in kind to decode's, and the
suffix quantize-on-stage matches the representation later turns attend
anyway. No new depth-dependent error term is introduced.

### 9.3 What stays FP16

The no-prefix path (fresh prompt, contiguous FP16 K/V, `HAS_PREFIX =
false`) keeps the existing FP16 `m16n8k16` path unchanged: its operands
are not quantized at rest, so int8 would *introduce* the runtime
quantization overhead this design exists to avoid, on a path that is not
the bottleneck. K-smoothing at seal time (per-block K mean subtraction +
`q·mean` correction) is **out of scope** — it modifies the seal format and
interacts with the `PRODUCTION_*` threshold re-derivation; it gets its own
design if the §11 accuracy gate ever demands it.

---

## 10. SM120 Forward Path

Everything here is warp-level `mma.sync` — it runs on SM120 unmodified.
The staging layer is deliberately shaped for the upgrade:

- SM120's block-scaled MMA consumes one scale per 32 elements — the
  arena's quant-block granularity exactly. A seal-time transcode of
  Q4-family blocks to the `mxf8f6f4` container layout (e2m1/e4m3 + UE8M0
  power-of-two scale) lets the K/V slabs feed `kind::mxf8f6f4.block_scale`
  MMA with **hardware** scale application, deleting the software fixup
  FMAs. UE8M0's power-of-two constraint folds the residual (mantissa part
  of the outer scale) into the existing FP32 accumulator fixup — the
  two-level structure of §6.5 is already the required shape.
- The published cautionary result (an SM120 FP4 attention kernel at 0.6%
  tensor-core utilization, 99.9% of instructions in quantization/data
  movement) is the *runtime-quantization* trap; quantized-at-rest storage
  is this design's structural immunity to it.

No code in this document waits on SM120; the transcode is a future design.

---

## 11. Validation Plan

1. **Staging unit tests, raw-byte assertions** (repo policy: never
   error-tolerance for codec paths): K slab bytes + scales after
   dequant→RoPE→requant for every K format × ragged pal_maps × partial
   slices; V readthrough slab bytes verified against arena bytes;
   zero-padded ragged k-lanes byte-checked.
2. **GPU ↔ CPU parity harness**: extend the existing prefill harness with
   a CPU int8 reference (mirroring the sampling-harness pattern) — exact
   int32 QK accumulator match per (row, token, palette); FP32 softmax/PV
   within one ULP-scaled bound; parity across `HAS_PREFIX` × readthrough ×
   split counts × `hpg ∈ {1, 4, 8}` × HD ∈ {64, 128}.
3. **Straddle regression corpus**: the `9eee0ddd` repro fixtures re-run —
   they must pass trivially (tiles can no longer straddle), and the test
   asserts the slice-aligned walk visits every valid token exactly once
   against a position-map oracle.
4. **End-to-end controls**: `duplication_replay` 0/6 at the production
   C5/C4 stress config; the ladder recall scenario; perplexity delta vs
   the FP16-dequant path ≤ the existing C-level acceptance band on
   Qwen3-30B-A3B across C0–C7.
5. **Accuracy gate for Q granularity**: if (4) shows regression traceable
   to Q quantization, the specified escalation is per-thread Q scales
   (SageAttention2 granularity, free in fragment layout) — a bounded,
   pre-decided fallback, not an open question.
6. **Benchmarks**: a `paged_prefill_benchmark` example (mirroring
   `batched_sampling_benchmark`) sweeping q_len × prefix_len × batch ×
   splits; acceptance = ≥3× prefix-attention throughput at (q=64,
   prefix=32k, B=1) and (q=256, prefix=32k, B=8) on the RTX 4090 Mobile;
   TTFT measured end-to-end via the scheduler's decode-start log line.

---

## 12. Implementation Plan

Each phase compiles, passes its tests, and leaves the tree shippable. The
old prefix path survives until the Phase 4 cutover, then is deleted in the
same phase — no dual paths outlive the plan.

**Phase 1 — Shared int8 attention primitives.** Extract from
`int8_decode_kernel.cuh` / `mma_wrappers.cuh` / `convert_all.cuh` into a
shared header (`candle-kernels/src/attention-int8/`): fragment loaders,
`mma_int8_m16n8k32`, readthrough accessors + format predicate, per-palette
quant helpers, the LSE partial format. Decode switches to the shared
header in the same phase (pure move, byte-identical PTX asserted by the
existing decode tests).

**Phase 2 — The kernel.** New `paged_prefill_int8_kernel.cuh`:
slice-aligned walk, staging pipeline (§6.2–6.4), GQA-packed M compute
(§6.5), 8-warp symmetric mapping, KV write in staging (§6.6). Unit
harness + CPU reference from §11.1–11.3. Not yet dispatched from the API.

**Phase 3 — Split-KV + combine.** Grid-z sharding, partial-pool
generalization to `q_len × n_head` rows, combine reuse, `num_splits`
heuristic + the `q_len`-aware grid sizing. Parity across split counts.

**Phase 4 — Cutover and deletion.** `run_paged_prefill_chunks` dispatches
`HAS_PREFIX` to the new kernel. Deleted in the same change: the old prefix
body, the FP8 dead code (~600 lines, unreachable from any wrapper), the
straddle twin tables, the helper-warp machinery, `write_offset_shifts`
end-to-end (Rust FFI → .cu → kernel), and the stale header claims
(smem O-accumulator padding, 2–3-stage pipeline). `q_dtype = F32`
becomes a hard error instead of a silent no-op. Full §11.4 controls run.

**Phase 5 — Tuning + benchmark lock-in.** Sweep `BLOCK_M_TOK`,
`NUM_STAGES ∈ {2, 3}` (int8 slabs are half-size — triple-buffering may
now fit two co-resident blocks; measure, don't assume), splits heuristic
constants. Land `paged_prefill_benchmark` with recorded baseline numbers
in this document's appendix, and update `docs/` cross-references
(`paged_glue_kernel.md` interop note; this document's status line to
"shipped").

---

## 13. Implementation Record (as built)

Built and validated against the `prefill_ab` differential harness
(`candle-conversation/tests/prefill_ab/` — CPU golden + determinism +
A/B over identical arena bytes + real-substrate scenarios).

### 13.1 Measured performance (RTX 4090 Mobile, release, wall-clock per
prefill call incl. host metadata; best of 10)

Recorded in high-performance mode after cool-down (best/mean spread
≤ 5%), on the PRODUCTION-SHAPED harness: device-resident Q/K/V and
prebuilt varlen metadata, as a real prefill call receives them. (The
harness originally re-uploaded Q/K/V from pageable host memory and
rebuilt the metadata tensors inside the timed region — up to 1.3 ms of
measurement artifact per call, −44% on q256's wall once removed.)

| shape | FP16-dequant kernel | INT8 kernel | speedup |
|---|---|---|---|
| q=64, prefix=8192, C5 | 23.92 ms | **1.26 ms** | **19.0×** |
| q=8, prefix=8192, C5 | 23.66 ms | **0.66 ms** | **36.0×** |
| q=256, prefix=2048, C5 | 11.86 ms | **1.46 ms** | **8.1×** |
| q=64, prefix=2048, C5 | 6.10 ms | **0.62 ms** | **9.9×** |

Kernel-only (ncu, locked clocks): 1.47 / 0.71 / 2.03 / 0.43 ms vs the
FP16 kernel's 37.9 / 37.9 / 18.5 / 9.5 ms — 26× / 53× / 9× / 22×. The
prefill call is now ~80–90% kernel: the host-side stall hunt (§13.5)
took per-call metadata from ~0.55 ms to ~0.05 ms by replacing the
owned `SealedChunk` snapshot in the slot-header build with the
zero-clone `visit_live_chunks` path. q256 is compute-bound, paying the
head-dim-split's duplicated QK (§13.3 rounds 6–7) — the accepted trade
of the max-occupancy program.

### 13.5 Host-path stall hunt (post-kernel)

With the kernel 26–53× faster, prefill walls were bounded by the host.
Three stalls found, two fixed, one attributed:

1. **The harness itself** — the timed region re-uploaded Q/K/V from
   pageable host memory (synchronous H2D ×6/call) and rebuilt varlen
   metadata; up to 1.3 ms/call of measurement artifact. Fixed:
   `BuiltCase` carries device-resident inputs + prebuilt prefill_meta
   (the production shape).
2. **`slot:build`** — profiled at 0.51 ms/call @ 8k prefix, 96% of it
   materializing owned `SealedChunk` snapshots (per-chunk Vec/Arc clones
   + O(gids²) `arena_byte_size` walks the build never uses). Fixed:
   `ChunkedKvBacking::visit_live_chunks` yields borrowed `LiveChunkRef`s
   under the state read lock; `TokenSliceHost::from_live_chunk` builds
   slices zero-clone. 0.51 → **0.028 ms** (18×).
3. **Writer-chunk allocation enqueues ~0.33 ms of driver work** (visible
   only under the bench's reset→realloc-per-rep pattern; production
   allocates writer chunks once per turn). Attributed via the
   `prefill:entry` / `harness:ensure` sync spans; left as a known
   per-turn line item — the scheduler can pre-allocate during decode if
   it ever matters.

Instrumentation kept: the profiled bench (`--features profile`) prints
per-span ms/call per shape; `prefill:entry` isolates caller-enqueued GPU
work from the call's own spans. Remaining above the call: the eager MLP
op chain and per-step decode sync (fusion / graph-capture territory,
separate program).

The round-2 staging restructure (per-warp flip-flop + palette-table cache
+ restored pads, §13.3) is worth 3–5% on every shape, and the round-3
compute-phase software pipeline another ~3.5% kernel-side — both only
resolvable under locked clocks with cool-down between runs; unlocked
thermal noise (±8%) swallows effects of this size. Sustained
bench/compile loops heat-soak the laptop even in performance mode:
kernel-only ncu durations are the trustworthy cross-run comparator.

Progression: int8 kernel alone (no split-KV) gave 3.6–4.7×; split-KV
lifted the starved short-q/long-prefix shapes to 8–17× (the FP16 kernel's
q8 ≈ q64 equal-time signature — the §2 starvation — is gone).

Accuracy: A/B vs the FP16 kernel over identical arena bytes on **real
substrate chunks** (827-token production turn, layers 0/24/47):
max_rel 0.2–1.0%, min row-cosine ≥ 0.996. Synthetic uniform-random
scenarios (the quantizer's adversarial case): A/B max_rel < 10%,
row-cosine > 0.99; CPU-golden bands per C-level as coded in the harness.

### 13.2 Deviations from the design sections above

- **Quantization grid is 32-dim natural windows, not palettes** (§6.5's
  palette-as-k-step). The MMA k-window and the requantization scale grid
  are decoupled from the arena's palette routing: K/Q quantize per
  (row/token, 32-dim natural window). Palettes exist only during arena
  *dequant* routing. This removes the ragged-palette zero-padding
  machinery entirely — every HEAD_DIM % 32 == 0 shape gets exactly
  HEAD_DIM/32 full k-steps — and lets Q stage once per block (a per-tile
  palette-aligned Q order would have to re-stage per slice).
- **The V^T slab and O accumulator are natural-dim indexed** — palette
  rank space is per-slice, so a rank-indexed accumulator cannot survive
  across tiles. Every staging decode routes dim→(palette, rank) through
  the forward table and lands directly in natural order; O stores need
  no permute.
- **P uses a fixed 1/127 scale** (P ∈ (0,1] post-softmax), not a per-row
  dynamic scale; measured accuracy is within all bands. A tile-local
  max scale (`exp(m_tile − m_new)` folded into the PV fixup) is the
  known upgrade if a future accuracy gate wants it.
- **Staging is raw-first** (§13.3 round 8): each palette's 32-token
  quant-block span is bulk-copied to smem with 16-byte cp.async, and
  every per-element decode (K dequant, V read-through, V FP fallback)
  extracts from that copy via runtime-format helpers — there is no FP16
  exchange slab and no `ArenaAccessor` load in the tile loop. Each
  thread drains its own cp.async groups before the (pre-existing) tile
  fence barrier, which doubles as the publish (bring-up finding, still
  load-bearing: `__syncthreads` alone does NOT fence cp.async). A true
  double-buffered ring (compute tile N while staging N+1) needs the
  >48 KB dynamic-smem opt-in and remains the Phase 5 candidate.
- **Split-KV is gated on occupancy**: shards only when the unsplit grid
  leaves SMs idle (unconditional splitting regressed the
  already-parallel q256/prefix2k shape 2.8 → 3.2 ms on partial-emit +
  combine traffic). At 4-blocks/SM residency the target is 3 blocks in
  flight per SM, capped at 32 splits (the combine kernel is 13–19 µs —
  noise). Fresh tiles go to the last shard; the arena write pre-pass
  runs on shard 0 only. The partial pool is a grow-on-demand static
  (the decode split-KV pool idiom).
- **v1 head-dim envelope: 64 and 128.** 96 breaks the in-thread RoPE
  pairing (pair (d, d+48) crosses lanes); 256 exceeds the 48 KB
  static-smem budget. Both fall back to the FP16 kernel at dispatch
  (`batched_layer.rs`); production is 128.
- **Read-through V engages at every head dim** (per-element extraction
  has no lane-width constraint) whenever every V palette format is an
  int8 passthrough family; mixed tiles (e.g. C4's Q4_1 V) take the
  FP-fallback dequant→requant path. Both paths are exercised by the
  harness.
- **Fresh tokens never round-trip the arena**: suffix tiles stage from
  the packed q/k/v inputs; the arena write of the same values is an
  independent pre-pass. This removes the write-before-read ordering the
  FP16 kernel's inline writeback carries.

### 13.3 Nsight Compute profile (q64/prefix8k, RTX 4090 Mobile)

Measured with `ncu` on the bench workload (kernel-only duration 2.16 ms of
the 2.7 ms wall):

- **Latency-bound, not throughput-bound**: SM 20%, memory 20%, DRAM 0.7%.
  No unit is near its ceiling.
- **Occupancy 31% achieved / 33% theoretical**, capped by BOTH 128
  registers/thread (block limit 2) and ~42 KB static smem (block limit 2)
  → 16 resident warps of 48. 74.6% of scheduler cycles have no eligible
  warp; top stall is the shared-memory scoreboard (~42% of issue latency).
- **+4-byte row padding** on the int8 MMA slabs (bank-rotation fix) bought
  3.5% kernel time and is kept; the residual smem stall is the staging →
  barrier → fragment-read latency chain itself, not conflicts.

The profile pins the next lever precisely: **more parallelism per SM**, via
(a) a double-buffered staging ring (compute tile N while staging N+1 —
needs the >48 KB dynamic-smem opt-in and an smem diet to keep 2 blocks
co-resident), and/or (b) a register/smem diet to reach 3 blocks/SM
(≤ 85 regs — accepting spills — and ≤ 33 KB smem: union `s_p8` into the
temporally-disjoint `s_fp`, FP16 scale tables). Both are bounded by the
same ceiling: perfect latency hiding at current per-warp throughput caps
the remaining kernel-side gain at roughly 2–3×.

**Round-2 record (profiler-driven; every experiment kept or reverted on
measurement):**

- **3 blocks/SM (route b) — REJECTED.** The full smem diet (scratch union
  + FP16 scales + pads traded away, 33.9 KB) with
  `__launch_bounds__(256, 3)` compiled and fit, but the forced ≤85-reg
  spill measured slightly *slower* than 2 blocks at 128 regs on every
  shape. Reverted to compiler-chosen registers; the diet's union + FP16
  scales are kept (harmless, and they fund the future ring).
- **Per-warp flip-flop staging (the decode kernel's skt-ring trick) —
  LANDED.** Dequant and requant now share one token→warp ownership
  (`warp·TOK_PER_WARP + jj`), making the `s_fp` exchange rows
  warp-private: the K dequant→requant block barrier disappears, each
  staging step issues the NEXT token's arena loads (cp.async on
  dtype-format palettes, USE_TC=true with commit/`wait_group` fencing)
  under the PREVIOUS token's scalar work, and the read-through V
  line/scales are carved from the warp's own row 0 (race-free by
  construction). Sealed-tile barriers: 5 → 2–3. Kernel-time neutral on
  the C5 benches (quant-format palettes extract synchronously — no
  cp.async to overlap); engages on F16/BF16-format prefixes (writer
  chunks, identity seals).
- **Palette-table caching — LANDED.** Consecutive slices sharing pal_maps
  (the common case) skip the table rebuild via a cached-map compare
  (`pal_map_equal`). The compare→rebuild sequence needs its own barrier:
  see below.
- **Compute-phase software pipeline (the q8-matmul pattern) — LANDED
  (round 3).** Both MMA loops are explicitly double-buffered in
  registers: iteration k+1's fragment + scale smem loads issue BEFORE
  iteration k's `mma.sync` + FP32 fixup, so the smem latency drains under
  the tensor-core op. QK pipelines across the flattened (window ×
  n-slice) iteration space (A-fragment rotates on window wrap); PV
  pipelines its V^T fragment + scales across the 16 output-dim slices.
  `__launch_bounds__(256, 2)` pins the register budget at 128/thread —
  2 × 256 × 128 is exactly the 64K regfile, so one register past 128
  would silently halve residency to 1 block/SM; the cap trades that for
  a handful of spills. ncu: kernel 2.25 → **2.17 ms**, issue latency
  14.2 → 13.4 cycles, and the top stall MOVED from smem-scoreboard to
  L1TEX/global — the pipeline drained the smem waits and exposed the
  staging's arena loads as the next link (= the two-hop staging lever
  below).
- **Head-dim-split o_acc restructure — TRIED, REJECTED (round 6).**
  Warp = (row-tile, dim-half): warp pairs duplicate QK+softmax for 16
  shared M-rows, each accumulating half the output dims — o_acc 64 → 32
  registers, M_ROWS 128 → 64, `launch_bounds(256, 3)`. It achieved its
  structural goal — **3 blocks/SM, 50% theoretical occupancy, SM
  throughput 20 → 40%** — and correctness held on first build. But wall
  time LOST on every shape (+2–3% at 8k prefixes, **+17% on
  q256/prefix2k**): the duplicated QK (1.5× MMA per output) consumed
  exactly what the occupancy bought. Conclusion: after the software
  pipeline and ldmatrix rounds this kernel is **no longer purely
  latency-bound** — occupancy purchased with recompute does not pay, and
  the M-preserving alternatives (FP16 o_acc accumulation, cross-block
  dim split with duplicated K staging) are worse trades on accuracy or
  traffic. VERDICT SUPERSEDED (round 7): the split was RESTORED as the
  structural enabler of the max-occupancy program — at 4 blocks/SM with
  the memory system fixed (round 8) the occupancy wins net; only the
  compute-bound q256 shape still pays for the duplicated QK.
- **ldmatrix MMA fragments (from the q8 matmul) — LANDED (round 5).**
  Both A-fragments (Q, P) load via the wrappers' existing
  `load_a_frag_m16k32_ldmatrix` (one `ldmatrix.x4` replaces four strided
  LDS), and the B-fragments (K, V^T) via a NEW
  `load_b_frag_n8k32_ldmatrix` in `mma/mma_wrappers.cuh` (one
  `ldmatrix.x2` replaces two strided LDS ×32 fragment loads per tile per
  warp — B was the dominant fragment-load count). Slab pads follow the
  q8-matmul `KI8_STRIDE` convention (+16: rows 16B-aligned for ldmatrix,
  stride not a multiple of 128 for bank rotation). Kernel 2.17 →
  **2.14 ms**; locked-clock walls improved on three shapes (q256/2k
  2.74 → 2.53 ms — through 100K q-tok/s). The B-side loader is now
  available to the decode kernel too (its "Phase B" note).
- **Two-hop raw-block staging — TRIED, REJECTED (round 4).** Bulk
  cp.async of each palette's quant-block span into smem, extraction
  re-pointed at the copy (the `ArenaAccessor` base is a generic pointer,
  so hop 2 was byte-identical logic). Racecheck-clean and correct — but
  measured kernel-time neutral (2.20 vs 2.17 ms) and q256/prefix2k wall
  +9%: the "L1TEX" stall the round-3 pipeline exposed is substantially
  **L1-hit latency, not DRAM** — the per-element extraction re-reads each
  32-token block ~32×, so L1 serves it near-smem-fast after first touch,
  and the fill + drain barrier cost per tile is pure overhead on
  tile-heavy shapes. Lesson recorded: on this kernel, "global" stall
  reports must be decomposed into DRAM-miss vs L1-hit latency before
  reaching for smem staging. VERDICT SUPERSEDED at the 4-block
  configuration — see round 8, where the raw-first rebuild of the same
  idea (no accessor indirection, no added barriers, occupancy to hide
  the fill) removed 86% of global load sectors and set the kernel-time
  record.
- **Max-occupancy configuration — LANDED (round 7).** The directive:
  drive theoretical occupancy to the hardware max, then optimize until
  achieved meets it. Three pieces close the two budgets at
  `__launch_bounds__(256, 4)` (66.7% theoretical — the max for a
  256-thread block on SM89): (a) the head-dim split restored (o_acc 64 →
  32 regs); (b) Q drained to registers at block scope (the q8-matmul
  trick: a warp's QK A-operand is its own 16 rows = N_WIN ldmatrix
  fragments held all-loop, the Q smem slab dead after one barrier);
  (c) a UNION smem arena — the dead Q prologue's bytes are re-overlaid
  by the per-tile slabs (19.9 KB → fits the 25.6 KB 4-block budget).
  At the 64-reg cap ptxas spills; the spill program cut stack
  344 → 216 B (register V-requant array → two smem passes, row-state
  arrays → recompute lambdas, Q scales kept RESIDENT in smem instead of
  drained, palette loops serialized against 4× accessor inlining) and
  measured local traffic 34.7 M → 26.8 M sectors. Interim verdict was
  NEGATIVE — kernel 2.05 → 2.24 ms, achieved occupancy pinned at ~35%
  regardless of theoretical — because the memory system (79% wasted
  global sectors) could not feed more warps. Round 8 is what cashed the
  configuration.
- **Raw-first staging + splits bump — LANDED (round 8, the payoff).**
  The profiler's dominant finding across every configuration was
  uncoalesced global loads in quant extraction (80.4 M excessive
  sectors = 79% of 102 M total, Est. Speedup 63%). Rebuild of the
  round-4 idea without its overheads: each palette's 32-token block span
  bulk-copies to smem via 16-byte cp.async (perfectly coalesced), and
  ALL per-element decodes — K dequant, V int8 read-through, V FP
  fallback — extract from the copy through runtime-format helpers
  (`i8_dequant_elem` / `i8_rt_elem` / `i8_arena_elem` wrapping the
  `BlockConverter` / `BlockInt8` families per element). The FP16
  exchange slab is GONE (the rank→natural permutation lives in the
  table-indexed reads), which pays for the raw spans inside the same
  scratch union; the inverse V table, the fallback's extra barrier, and
  the read-through SUB == 32 restriction all fall out. Non-hop palettes
  (R16, dtype, unaligned spans — glue slices) decode element-wise from
  global. Split-KV re-targeted for the 4-block config (gate < SMs,
  3 blocks-in-flight/SM, cap 32). Measured (q64/8k): global load
  sectors **101.6 M → 14.6 M (−86%)**, kernel **2.24 → 1.69 ms** (record;
  the 3-block champion was 2.05), achieved occupancy **34.7 → 55.1%**,
  SM throughput 53%. q8/8k kernel-only 0.70 ms — its 1.35 ms wall is now
  host-metadata-dominated. Harness 22/22 twice; racecheck clean (the
  only reports are the two known pre-existing hazards outside this
  kernel). NEW top cost: spill traffic (17.7 M local-ld + 13.3 M
  local-st sectors) now EXCEEDS global — the next lever is
  register-demand reduction (candidate: FP16-pair packing of o_acc).
- **Slab-store packing + wave-exact splits — LANDED (round 9).** Three
  small levers from the post-round-8 profile (top rule: uncoalesced
  shared, 39% excessive wavefronts; top stall: smem scoreboard 33%):
  (a) V^T slab stores pack 4 token-bytes per dim into ONE aligned
  4-byte store (read-through, FP-fallback, and fresh paths) — the byte
  columns are 4-way bank-conflicted by construction (ldmatrix needs
  16 | LD, conflict-free byte columns need LD/4 coprime to 32 —
  mutually exclusive), so the lever is 4× fewer stores; the P tile's
  byte pairs likewise store as u16. (b) Fresh tiles join the sealed
  tiles' round-robin ordinal space instead of pinning to the last shard
  (the profiled ~9% SM imbalance). (c) Split-KV fills toward the
  4-blocks/SM residency limit with FLOOR, never past it. The ceil
  variant was measured first and is the round's lesson: 10 splits put
  320 blocks on 304 residency slots and the 16-block second wave
  near-doubled the makespan — kernel 1.69 → 2.36 ms from ONE block over
  the wave boundary. With floor (288 blocks, 0.95 waves): kernel
  **1.69 → 1.47 ms**, achieved occupancy 59 → **62.8%** (theoretical
  66.7%), and every wall improved: 2.10 / 1.21 / 2.85 / 1.22 ms —
  q256's P/V packing win finally clawed back part of its HD-split
  cost. Harness 22/22.
- **compute-sanitizer racecheck — now part of the gate.** It caught (1)
  the table-cache compare racing the cache rewrite (divergent `tbl_hit` →
  non-uniform barrier; presented as a ~1-in-3 A/B flake) — fixed with a
  fence between compare and rebuild; the int8 kernel now racechecks at
  **0 errors** (11 warnings, all the benign warp-private
  `__syncwarp`-ordered pattern decode also uses). It also surfaced two
  PRE-EXISTING hazards outside this kernel: an Error-level race in the
  FP16 prefill kernel (`load_kv_chunk` smem write vs
  `store_kv_chunk_arena` read, paged_prefill_kernel.cuh:220 ↔
  kv_store.cuh:86) and Warning-level hazards across the seal-time
  quantizer (`select_kv_format.cuh:793` ↔ `quantize_q8_0.cuh`) — a
  plausible candidate for the historical "irreducible" stochastic
  1-fail/run floor in threshold tuning. Both owed a separate
  investigation.

### 13.4 FP16 prefill kernel removal (Phase 4, completed)

The FP16-dequant prefill kernel is REMOVED (kernel, dispatchers, FFI,
and the `int8_backend` dual path): the int8 kernel serves every paged
prefill, including fresh-context ingestion (its fresh-tile path staged
from the packed inputs was validated by `no_prefix_fresh` /
`ab_no_prefix_fresh` before the cutover). Head dims outside {64, 128},
interleaved RoPE, and F32 are hard errors at dispatch — there is no
fallback kernel. `write_offset_shifts` (the pre-position-map SSO
right-pack mechanism) is deleted end-to-end — scheduler →
`forward_batched_with_write_shifts` → `SequenceContext` field → FFI —
every live caller passed shift 0; the position-map/write-slice
machinery handles mid-chunk writer starts natively. The FP16 kernel's
pre-existing `load_kv_chunk` ↔ `store_kv_chunk_arena` race
(racecheck-flagged) is gone with it.

The A/B harness legs were re-based accordingly: every scenario now
checks the CPU golden band (where finite) plus BITWISE reset-rerun
determinism over identical arena bytes — the exact oracle for kernel
nondeterminism. The FP16 wall-clock baselines in §13.1 are historical
(measured immediately before removal).
