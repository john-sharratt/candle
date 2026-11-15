# QSA index keys in Q8_0

**Status:** design, agreed 2026-11-15. Built in the order of §9.

The QSA index cache holds, per sequence and per full-attention layer, one pooled,
RMS-normed, **un-rotated** key per block of `ratio` tokens: 128 channels, `ratio`
4, on the released Flash-Next's 12 full-attention layers. Every key is `F32`
today — on the card in the span's QSA-index arenas, and on disk in a turn's
carried state and its positional index pages. This document moves every
completed key to **Q8_0**, everywhere, with no second format left behind.

## 1. Why

On WhiteWorld (RTX 3090, 2026-11-15, eight characters) the QSA index cache held
**3.87 GB** of the 22.8 GB span — more than the 1.4 GB of live K/V it selects
from, and less than the experts' 5.1 GB only because the experts get what is
left. Experts hit 10–20%; a decode step waited on ~10,000 expert misses a wave,
about a second a step, and a short reply took 40–80 s. At `F32` the index costs
12 × 128 × 4 B ÷ 4 tokens = **1.5 KB per token per live sequence**, and the
characters' contexts run 50–150k tokens.

The reference caches the compressed index key in BF16
(`docs/archived/qwen38_flash_next.md` §3.1); DeepSeek's indexer, the same
design, stores 8-bit keys; `docs/progressive_yarn.md` §7.6 already named
smaller stored keys as the scorer's lever 5. The key only feeds
`Σ_h ReLU(⟨q_h, rot(k̄)⟩)` and a top-512 pick.

## 2. The kernels this changes, as they are

Studied 2026-11-15; every number below is from their own notes.

| Kernel | Shape today | What Q8_0 touches |
|---|---|---|
| `qsa_index_append.cu` `append_kernel` / `flush_kernel` | One block per completing key, `d` threads (128 = **four full warps**), one channel a thread; pool `ratio` rows → RMS-norm by a block tree reduction → `dst[c] = x · k_norm[c]` (`F32`, row-major). One launch for the wave's jobs (descriptor table). | Stores int8 + scales instead of `F32`. |
| `carry_kernel` | Copies a wave's trailing `< ratio` raw rows into the open-block buffer. | Nothing: raw rows stay `F32`. |
| `qsa_page_place.cu` | Transposes a sealed page `[rows, d]` → channel-blocked `[d/4, rows, 4]` through a shared tile; `ncu`: **87–88% of peak DRAM**, 95–97% occupancy, 40 registers. | Transposes Q8_0 planes. |
| `qsa_score_paged.cu` | One thread per candidate, `CPT` candidates a thread, a `TILE_R × H` query tile in shared, `float4` key loads, **rotate-on-load** of the 64 rotary channels (NeoX half-split, pair `i` = channels `(i, i+32)`) from per-warp shared rotation runs, then FMA into `TILE_R × H × CPT` accumulators; `__launch_bounds__(256, 3)` = 80 registers, 50% occupancy; arms `(TILE_R, CPT)` ∈ {(4,2),(2,2),(2,1),(1,1)}. Placed pages are channel-blocked (`cstride = rows`), the live tail row-major (`cstride = 1`). | Reads Q8_0 keys. |

## 3. The format

**Q8_0: blocks of 32 along the key, an `int8[32]` and one `f16` scale
`s = max|x| / 127` each.** A 128-wide key is four blocks:

| Block | Channels | Role in the scorer |
|---|---|---|
| 0 | 0–31 | rotary, low half of pairs 0–31 |
| 1 | 32–63 | rotary, high half (the partners of block 0) |
| 2, 3 | 64–127 | pass-through |

The block boundaries are the scorer's own: block 0 and its partner block 1 are
each one scale, so a pair is rotated from two dequantized values with no
straddle, and blocks 2–3 never rotate.

- **Symmetric, and that is enough.** The keys are RMS-normed with `k_norm`'s
  per-channel weight, not mean-centred, so a block can lean one way; a symmetric
  scale represents it exactly and loses at most one bit of range. Q8_1's block
  sum serves dot products against an offset-quantized operand — the query never
  is one — so it is not used.
- **Rounding:** round-to-nearest-even of `x / s`, clamped to ±127; a block of
  zeros has `s = 0` and all-zero codes. One definition, written once in Rust
  (the reference and the test oracle) and once in CUDA, gated bit-identical.
- **Size:** `d + 2·d/32` = **136 B** a key at `d = 128`, against 512 B
  (**3.76×**). WhiteWorld's 3.87 GB becomes about 1.03 GB.
- **The open block stays `F32`**: at most `ratio − 1` raw rows and its rewind
  copies, pooled into a key only when the block completes.

### Layout

Two planes per page, never interleaved, so every load is a whole vector:

- **Codes:** the live tail row-major, `int8 [rows, d]` (as the append writes);
  a placed page channel-blocked as **`[d/16, rows, 16]`** — a lane's load is one
  16-byte `int4` (16 codes of one row), a warp's is 512 contiguous bytes: the
  `float4` layout's coalescing at a quarter of the bytes.
- **Scales:** `f16 [rows, d/32]` in the tail, **`[d/32, rows]`** placed — a
  warp's scale load is 64 contiguous bytes.

The descriptor table keeps its shape (`{codes, cstride, rstride, delta}`) and
gains the scale plane's base and stride (`QSA_PAGED_PAGE_WORDS` 4 → 6).

## 4. Where a key is quantized: in the append kernel

A key is complete the moment its block closes: pooled, normed, 128 values,
needing nothing outside itself. So `append_kernel` and `flush_kernel` quantize in
registers and store codes and scales; **no `F32` key is ever written** (hot-path
invariant 1).

The kernel already has the right shape for it: one 128-thread block per key is
four full warps, and **each warp is exactly one Q8_0 block**. Each warp takes
its `max|x|` with five `__shfl_xor_sync` rounds (`d` a multiple of 32 — the
released geometry; the generic tree reduction stays for the test geometries),
lane 0 writes the scale, every lane writes its code. The RMS-norm's existing
block reduction is untouched.

**Not in the KV gap** (`docs/kv_quantize_in_the_gap.md`): the gap is for formats
chosen by per-block statistics and a policy, off the forward thread. An index
key needs neither; routing it through the gap would add a store, a read and a
stage.

## 5. The scorer

`qsa_score_paged` keeps its shape — candidates on x, query rows on y, `CPT`
candidates a thread, the query tile and rotation runs in shared — and changes
its loads:

- **Per channel group a lane loads one `int4` of 16 codes**, and per Q8_0 block
  one `f16` scale; it dequantizes into `float4`s in registers. Against today's
  four `float4` loads for the same 16 channels, that is a quarter of the bytes.
- **Rotary blocks (0, 1):** dequantize, rotate on load exactly as today, FMA.
- **Pass-through blocks (2, 3):** no rotation, so the dot product can stay in
  integers: the query's pass-through half is quantized once per row to int8
  with a per-row scale (it is in shared for the block's life), and the 64
  channels go through `__dp4a` — 16 instructions instead of 64 FMAs — with the
  block's two scales and the row scale applied once.

`qsa_page_place` transposes the two planes as it transposes `float4`s today; no
placement converts.

### Tensor cores, and what rotation does to them

**The rotary half cannot go straight through an int8 tensor core.** The scorer
rotates each key at the key's own position before the dot product, and the
rotation is a different 2×2 per pair per candidate; an `mma` multiplies fixed
operands. So:

- **Pass-through half → IMMA.** The query rows of one sequence at decode are 4
  heads × its MTP positions — about 16, the m16 of `mma.sync m16n8k32` on sm_80+
  and the int8 MMA paths on Blackwell; a placed page's `[d/16, rows, 16]` codes
  are the k32 operand two groups at a time.
- **Rotary half → HMMA on a rotated tile.** A block dequantizes and rotates its
  candidates' rotary channels into an `f16` tile in shared, then `mma.sync`
  `f16` against the `f16` query — the rotation is paid once per candidate as
  today, the multiply on tensor cores.

Whether the tensor-core path beats the `dp4a` path at decode widths is the
harness's question (§8), not this document's: at single-digit rows the scorer
is load-bound, not FMA-bound.

## 6. On disk

Two persisted forms hold completed keys, both opaque blobs the model owns:

- **A turn's carried state** (`paged_index::encode_aux`): `AUX_VERSION` becomes
  **4**: each layer's completed rows as the two Q8_0 planes row-major, its open
  block `F32`.
- **A unit's positional index pages** (`seal_positional_tail_span`, placed by a
  slot that borrows the turn): the same encoding, behind a version word.

**A build reads Q8_0 only.** A version-3 aux blob or an `F32` page is refused by
name — "written before the index was Q8_0: run `substrate_inspect
migrate-index-q8`" — never read as before. There is no second reader.

## 7. Migration

A one-time program, `substrate_inspect migrate-index-q8 --log <dir>`, run on a
stopped substrate:

1. `--dry-run` counts every record holding an `F32` index (carried state and
   positional pages) and their bytes.
2. The run decodes each, quantizes its completed rows with the Rust reference
   quantizer (§3 — bit-identical to the kernel's), and appends the re-encoded
   record in its place, as compaction re-emits one; the superseded records are
   dead weight a later compaction reclaims.
3. It ends by counting the `F32` records left, which must be zero.

Run on WhiteWorld's npcd and on BlackWorld's zend after the build reaches main,
each with its daemon stopped, then the new build started. Run first against a
copy of each store, with the count and a resumed conversation checked.

## 8. The micro-harness, and the tuning method

The repository's kernel method (`docs/progressive_yarn.md` §7.6): **attribute
first, from `ncu` and SASS, then make one change and rebuild once.** Each kernel
this touches gets a harness before it is tuned, in `candle-transformers/tests/`,
beside the ones it extends (`qsa_score_rot_harness.rs`,
`qsa_page_place_bench.rs`, `qsa_index_bench.rs`):

- **Correctness first, in the same file.** The quantizer bit-identical to the
  Rust reference; a placed Q8_0 page the transpose it replaces, bit for bit;
  every scorer arm (`TILE_R`, `CPT`, `dp4a`, tensor-core) against an f64 oracle
  scoring the *dequantized* keys within the derived bound — so a tuning change
  can never become a numerics change.
- **A sweep that says which shape is slow:** depth {8K, 32K, 128K, 512K} ×
  query rows {1, 8, 16, 64, 512} × pages {16, 256}, each cell against the `F32`
  scorer in the same run; CUDA events, median and p90; the daemon stopped and
  the card idle.
- **A one-launch-per-shape target for `ncu`** (`…_profile_one`), and the `nsys`
  timeline of a wave for launch counts — the append's history is a 30-launch
  chain found exactly that way.

**What each round reads and must reach**, per kernel, from `ncu`:

| Target | Metric |
|---|---|
| **Full occupancy** | achieved occupancy vs theoretical (`sm__warps_active`), the limiter named (registers / shared / blocks), registers per thread and **zero spills**; `__launch_bounds__` set from it, as the scorer's `(256, 3)` was |
| **Full warps** | warp execution efficiency ~32 threads per instruction (`smsp__thread_inst_executed_per_inst_executed`); no divergent tail in the channel loops; every shuffle and vote under a full mask, warp-uniform loops |
| **Vector math** | SASS shows `LDG.E.128` for codes, `IDP.4A` (or `IMMA`/`HMMA`) in the inner loop, no scalar byte loads; dequantize on `float4`s |
| **Memory** | DRAM and L2 throughput against peak for the bandwidth-bound shapes (the placement's 87–88% is the bar it already meets); L1 hit, bank conflicts < 1% of shared instructions |
| **Stalls** | issue-stall reasons — the scorer's history is a long scoreboard stall on table reads, fixed by shared rotation runs |

Each round is recorded in a results table here, as `progressive_yarn.md` §7.6
records the scorer's.

**Gates for adoption:**

- **Selection:** on recorded contexts and the harness sweep, Q8_0 and `F32`
  select the same top-512 blocks with **≥ 99% overlap**, and the attention
  outputs agree within the KV gate's tolerance; the Flash-Next needle gate with a
  dense control holds.
- **Decode shape** (rows ≤ 16, any depth): the Q8_0 scorer is no slower than the
  `F32` scorer within run-to-run variance — it reads a quarter of the bytes, so
  it should be faster.
- **Wide shapes** (64 rows and up): no slower than `F32`.
- **Append:** no slower than today at decode width, still one launch a wave.
- **On WhiteWorld after migration:** the QSA-index tenant shrinks by ~3.7×, and
  the expert zone grows by what it frees.

## 9. Order of work

1. **The quantizer:** Rust reference + `#[test]`s on raw bytes (rounding,
   clamping, zero block, the four-block key).
2. **Q8_0 on the card:** append and flush storing codes and scales; placement
   transposing the planes; the scorer reading them (`dp4a` pass-through,
   dequantize-and-rotate rotary); descriptor table at six words.
3. **The micro-harness** (§8): correctness, sweep, `ncu` targets — then tuning
   rounds until the occupancy, full-warp, vector and memory targets are met and
   the decode gate holds.
4. **The selection gate** (§8) against `F32` on recorded contexts.
5. **Q8_0 on disk:** aux version 4, the page version word, refusal of the old.
6. **The migration program**, with its dry run, against copies of real stores.
7. **Main, then WhiteWorld and BlackWorld** migrated with their daemons stopped.
8. **The tensor-core scorer** (§5), its own harness rounds, adopted only if it
   beats the `dp4a` path at the decode shape.
