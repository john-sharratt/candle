# QSA index keys in Q8_0

**Status:** design, agreed 2026-11-15; reviewed against the code the same day
(§10 lists what the review changed). Built in the order of §9.

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

## 2. What holds, writes and reads a key today

Studied 2026-11-15. A key has **two writers and four readers on the card**, and
every one of them changes; the first draft of this document named three of the
six.

| Role | Where | Today |
|---|---|---|
| **Writer** | `qsa_index_append.cu` `append_kernel` / `flush_kernel` | One CUDA block per completing key, `d` threads (128 = **four full warps**), a channel a thread; pool → RMS-norm by a block tree reduction → `dst[c] = x · k_norm[c]`, `F32` row-major into a key page. One launch a wave (descriptor table). `carry_kernel` copies raw rows only and does not change. |
| **Writer** | `qsa_page_place.cu` via `ResidentPage::place_host` | A record's rows uploaded into staging slots, transposed `[rows, d]` → `[d/4, rows, 4]` through a shared tile. `ncu`: 87–88% of peak DRAM, 95–97% occupancy, 40 registers. |
| **Reader** | `qsa_score_paged.cu` | One thread per candidate, `CPT` candidates a thread, a `TILE_R × H` query tile in shared, `float4` key loads, rotate-on-load of the 64 rotary channels (NeoX half-split, pair `i` = channels `(i, i+32)`), FMA into `TILE_R × H × CPT` accumulators; `__launch_bounds__(256, 3)` = 80 registers, 50% occupancy; arms `(TILE_R, CPT)` ∈ {(4,2),(2,2),(2,1),(1,1)}; a scalar generic arm for the test geometries. |
| **Reader** | `qsa_shadow.cu` | The shadow scan scores kept queries against unloaded pool rows through **the same page descriptors**, scalar. |
| **Reader** | `qsa_rope_rows.cu`, `RowSource::Paged` | From `GEMM_TAIL_MIN_CELLS` (2²⁵ cells — a prefill chunk over a long live tail) the tail is rotated once into an `F32` scratch and scored by cuBLAS. It reads the key pages through a page table. |
| **Reader** | host readback: `KeyPages::host_rows`, `ResidentPage::host_rows` | What a seal writes to a record. |

Two layouts stand on the card, and **both are permanent**, not "placed versus
tail" as the first draft had it:

- **Blocked** (`ChunkLayout::Blocked`): a page placed from a record.
- **Row-major** (`ChunkLayout::RowMajor`): the live tail's key pages — and every
  page `close_tail_into_page` makes, which takes the tail's key pages **as they
  stand** and never re-places them. A long-lived slot scores mostly row-major
  pages.

Every buffer is a slot of the `SlotTenant::QsaIndex` arenas, `PAGE_BLOCKS` (256)
rows a chunk, an anchored `F32` tensor; the tenant has two strides (a key page;
an open block or its rewind copy) and `compact_index_caches` plans by them.

## 3. The format

**The repository's Q8_0** (`candle-kernels/src/quantize/quantize_q8_0.cuh`),
along the key: blocks of 32 channels, an `int8[32]` and one `f16` scale each.

```
amax = max|x| over the block
d    = f16( amax · (1/127) )                 round-to-nearest-even
id   = (bits(d) == 0) ? 0 : 127 / amax
q[c] = int8( rint( x[c] · id ) )             round-to-nearest-even
x̂[c] = q[c] · f32(d)
```

- **Canonical zero.** A block whose scale rounds to `f16` zero stores all-zero
  codes, so one stored form means one value. (The KV kernel tests `amax != 0`;
  the index tests the stored scale, which is the same for every key a normed
  row can produce and differs only below 3·10⁻⁸.)
- **Any width.** `nb = ⌈d/32⌉` blocks; the last is short when `d` is not a
  multiple of 32. The indexer's oracle runs `d = 16` and the host tests `d` of
  2, 4 and 8, and they keep running.
- **Symmetric, and that is enough.** The keys are RMS-normed with `k_norm`'s
  per-channel weight, not mean-centred, so a block can lean one way; a symmetric
  scale loses at most one bit of range to it. Q8_1's block sum serves a dot
  product against an offset-quantized operand, which the query is not.
- **Block alignment is a nicety, not a requirement.** At `d = 128` block 0 is
  the rotary low half, block 1 its partners, blocks 2–3 pass-through. A pair
  whose halves carry different scales is dequantized and then rotated like any
  other; nothing depends on the alignment.
- **Size:** `d + 2·nb` = **136 B** a key at `d = 128`, against 512 B
  (**3.76×**).
- **The open block stays `F32`**: at most `ratio − 1` raw rows and its rewind
  copies, pooled into a key only when the block completes.

### One definition, two implementations, not bit-identical

The quantizer lives once as a `__device__` inline in
`candle-kernels/src/qsa_key_q8.cuh` (quantize a block from registers;
dequantize a channel, a `float4`, a block), and once in Rust,
`qwen4exp/key_q8.rs` (`quantize_key`, `dequantize_key`), which is the test
oracle and the migration's quantizer.

**The kernel compiles with the rest of the `simple` archive, under
`--use_fast_math`, and is not required to match the Rust bytes.** A key is
quantized once — by the append or by the migration — and from then on only its
stored codes and stored scale are read; no mirror reproduces the kernel, which
is what the KV codecs' bit-exact contract exists for. A code that lands the
other side of a .5 tie, or a scale one ULP off, is a key still within half a
step of the value, read back by the scale that was stored.

So the tests are: `quantize_key` against raw bytes (it is deterministic); the
kernel against it with every code within ±1, the scale within one `f16` step,
and the dequantized key within half a step of the `F32` key the host computes.

### Layout on the card

A chunk is **one slot holding both planes**: the codes at offset 0, the scales
at `PAGE_BLOCKS · d`. One claim, one relocation, one device copy for a fork.
At `d = 128` a slot is 256 × 136 = **34,816 B** (a multiple of `SLOT_ALIGN`),
against 131,072 B; a 16 MiB region holds 481 chunks instead of 128.

- **Codes:** row-major `[rows, d]` where the append writes them (the live tail,
  and a page closed from it); blocked **`[d/16, rows, 16]`** where a record is
  placed — a lane's load is one 16-byte `int4`, a warp's 512 contiguous bytes.
- **Scales: always row-major, `f16 [rows, nb]`**, in both layouts. A key's four
  scales are then one 8-byte load, a warp's 256 contiguous bytes, and placement
  copies the plane instead of transposing it. (The first draft blocked the
  scales as `[d/32, rows]`: four loads a key for no gain.)

The scorer's descriptor grows by one word, to five:
`{codes, scales, cstride, rstride, delta}`, strides in `int4` groups. The shadow
scan's spans carry the same five.

**Open question the harness settles (§8): whether blocking the codes still
pays.** At `F32` a key is four 128-byte lines and a warp's row-major read
scattered across 128 of them; the measured L1 hit rate was 2%. At Q8_0 a key's
codes are **one 128-byte line**, aligned, so a row-major warp touches the same
32 lines a blocked one does. If row-major scores within noise of blocked, the
placement kernel, the staging slots, `ChunkLayout` and both stride words are
deleted and a record uploads straight into its slots.

## 4. Where a key is quantized: in the append kernel

A key is complete the moment its block closes: pooled, normed, `d` values,
needing nothing outside itself. So `append_kernel` and `flush_kernel` quantize
in registers and store codes and scales; **no `F32` key is ever written**
(hot-path invariant 1).

- A job's `dst` word becomes the key's code row; a fifth word names its scale
  row (`QSA_APPEND_JOB_WORDS` 4 → 5, `QSA_FLUSH_JOB_WORDS` 3 → 4). Both come
  from one `key_addr(page_ptrs, k, d)` that replaces `row_addr`.
- With `d` a multiple of 32 — the released geometry — each of the block's warps
  **is** one Q8_0 block: `amax` by five `__shfl_xor_sync` rounds under a full
  mask, lane 0 stores the scale, every lane its code. Otherwise the block
  reduces `|x|` per Q8_0 block through the shared buffer the norm already
  holds. The RMS-norm's own reduction is untouched.
- Still one launch a wave over `AppendGrids`' bound; the carry does not change.

**Not in the KV gap** (`docs/kv_quantize_in_the_gap.md`): the gap is for formats
chosen by per-block statistics and a policy, off the forward thread. An index
key needs neither.

Rewind is unaffected: a snapshot keeps the open block's raw rows and the two
counters, and codes above `n_blocks` are dead rows exactly as `F32` rows were.

## 5. The readers

### The scorer

`qsa_score_paged` keeps its shape — candidates on x, query rows on y, `CPT`
candidates a thread, the query tile and rotation runs in shared. It lands in two
moves, the second a harness round, so that a numerics change and a speed change
are never the same commit:

1. **Dequantize in registers (the baseline).** Per candidate one 8-byte scale
   load, then per 16 channels one `int4`; the channel loop is today's, with
   each `float4` made from four bytes of a register and its block's scale
   instead of loaded. The query is untouched, so the only error is the key's
   rounding and the oracle bound is one line: `|Δ| ≤ ½ Σ_h Σ_c |q_hc| · d_b(c)`.
   The templated arms need `d` a multiple of 32 and `pairs` of 16; everything
   else takes the generic arm, as now.
2. **`dp4a` on the pass-through half (a round, adopted on its numbers).** The
   query's pass-through channels are quantized to int8 with one scale per
   `(row, head)` at tile load; per `(row, head, candidate)` and Q8_0 block the
   dot product is four `__dp4a` into one temporary, converted and scaled once.
   It replaces 64 FMAs and 64 int-to-float conversions a key with 16 integer
   instructions — and it **quantizes the query**, a second error source the
   baseline does not have. So it must hold the selection gate on its own, and
   the shadow scan, which scores in float, would no longer share the scorer's
   arithmetic to the last bit (its cut is a threshold on a ranking, so that is
   tolerable, and is stated here so it is not discovered).

The register budget is the constraint to watch: the `(4, 4, 2)` arm holds 32
accumulators inside 80 registers, and the baseline adds the held `int4`s
(8 words a candidate). The harness reads spills first.

### The shadow scan, and the GEMM tail

- `qsa_shadow.cu` reads keys through `raw(c)`; it becomes the inline's
  dequantize, with the scale row loaded once per row.
- `qsa_rope_rows.cu` gains a Q8_0 paged source: dequantize, rotate, write the
  `F32` scratch the GEMM takes. The scratch stays `F32` and `score_bound_for`
  does not change.

### Placement

`qsa_page_place` transposes `int4` groups instead of `float4`s — a quarter of
the bytes through the same tile — and copies each job's scale rows. A job gains
the two scale addresses.

### Tensor cores, and what rotation does to them

**The rotary half cannot go straight through an int8 tensor core.** The scorer
rotates each key at the key's own position before the dot product, and the
rotation is a different 2×2 per pair per candidate; an `mma` multiplies fixed
operands. So:

- **Pass-through half → IMMA**, on the int8 query tile the `dp4a` round
  introduces; a placed page's `[d/16, rows, 16]` codes are the k32 operand two
  groups at a time.
- **Rotary half → HMMA on a rotated tile.** A block dequantizes and rotates its
  candidates' rotary channels into an `f16` tile in shared, then `mma.sync`
  `f16` against the `f16` query.

Whether either beats `dp4a` at decode widths is the harness's question: at
single-digit rows the scorer is load-bound, not FMA-bound. sm_86 has no int8
MMA rung beyond what the BDP scan already degrades from; the round is measured
on each machine it would ship to.

## 6. The host side

| Type | Becomes |
|---|---|
| `KeyPages` | pages are `U8` slot views of `PAGE_BLOCKS · (d + 2·nb)` bytes; `host_rows` / `write_host_rows` move the two planes as bytes; `fork` copies whole chunks device-to-device; `key_addr` replaces `row_addr`. |
| `ResidentPage` | `place_host` takes `(codes, scales, last_cells)`; `host_rows` returns the planes, un-blocking the codes; descriptors carry the scale address. |
| `SealedIndex` | `{ codes: Vec<u8>, scales: Vec<u16>, dim, last_cells, open: Vec<f32> }`. A seal reads bytes back and converts nothing; `seal_positional_range` narrows both planes. |
| `IndexCache::from_rows` | takes the planes and the open rows. |
| `PagedIndex` / `IndexPage` | the kernel's reference harness holds Q8_0 planes; `score_reference` scores the **dequantized** keys in f64. |
| `compact_index_caches`, `positional_lift_bytes_at` | the key-page stride comes from one `key_page_bytes(d)`. |
| `tensor_ptr` | a byte-typed sibling for `U8` slot views. |

## 7. On disk

Completed keys are persisted only inside the model's aux container
(`paged_index::encode_aux`), which three record kinds carry:

- a `Snapshot`'s and a `BranchCheckpoint`'s `aux` — a turn's carried state;
- a `TurnIndexPage` — a list of pages framed by `index_pages::encode`, each
  page one aux container;
- a section's sealed page — one aux container.

So **one version covers all of it**: `AUX_VERSION` becomes **4**. A layer's
section is `rows, dim, last_cells, n_open` then the codes row-major, the scales
row-major, the open block `F32`. (The first draft added a separate page version
word; pages are aux containers and already have one.)

**A build reads version 4 only.** Version 3 is refused by name — "written
before the index was Q8_0: run `substrate_inspect migrate-index-q8`" — never
read as before. There is no second reader in the daemon; the version-3 decoder
lives in the migration program alone.

A page's `PieceKey` is the digest of its bytes, so a migrated page has a new
key. Keys are taken when a page enters memory and nothing persists one, so
this costs nothing.

## 8. Migration

A one-time program, `substrate_inspect migrate-index-q8 --log <substrate dir>`,
run on a stopped substrate (`persistence/index_q8_migration.rs`):

1. `--dry-run` opens the store read-only — safe beside a running daemon — and
   counts every record holding a version-3 container, by kind, with their
   bytes as they stand and as Q8_0, and **how closely the Q8_0 keys hold the
   `F32` ones**: signal to noise over every key, and the mean and least cosine.
2. The run decodes each, quantizes its completed rows with `quantize_key`
   (§3), re-encodes at version 4 and appends the record again. A
   `TurnIndexPage` holds either one container (a prompt section's page) or a
   framed list of them (a turn's pages); the two are told apart by structure,
   each read strict about accounting for every byte, and a payload that reads
   as both or as neither is reported and left. A framed list is rewritten page
   by page with its token widths unchanged.
3. It commits, opens the store again and counts: none of this node's own
   version-3 records may still be what a reload reads.

**Under a new stamp, and only what this node wrote.** Of two records with one
identity a reload keeps the one it meets first (`stamps::take` is strict), so a
rewrite under the old stamp would be dropped. The rewritten record is a later
write by this node — in the page's own generation, so a section's page is still
the section's — and supersedes the one it replaces, whose bytes maintenance
reclaims. A page belongs to the member that wrote its stream (`StreamKey` is
the stream and its writer), so a record this node stamps cannot supersede
another member's page: those are counted as **foreign** and left, and arrive
rewritten from the member that wrote them once it has run this. Until then a
build reads those turns' K/V with no index. A snapshot's tail is keyed by its
stream alone and is rewritten whoever wrote it.

So on a replicated store **every member is stopped, migrated and started on
the new build together**: a member on the old build cannot read a version-4
page a migrated peer sends it, and a page whose writer has left the cluster
stays `F32` and unindexed.

**WhiteWorld's npcd store, dry run 2026-11-15** (28 GB, the daemon running):

| Records | Q8_0 | `F32` | `F32` MiB | as Q8_0 MiB | foreign | unreadable |
|---|---|---|---|---|---|---|
| turn and section index pages | 0 | 2,917 | 3,487.6 | 928.3 | 0 | 0 |
| snapshots and checkpoints | 1,689 | 28 | 3,164.6 | 3,155.5 | 0 | 0 |

7,162,818 keys held at **45.2 dB**, cosine to the `F32` key mean **0.999985**,
least **0.999956**. The snapshots are mostly recurrent state; their index is
the 9 MiB that shrinks.

**Migrated the same day**, daemon stopped, the store copied first: 2,977
records rewritten in 40 s, the store read back with no `F32` record, and build
5082 (`eaf12e984`) proven by keep with npcd ready in 47 s and no index refusal.

**BlackWorld's zend store, dry run 2026-11-15:** 113,718 pages, 81,947 MiB →
21,828 MiB; 28 snapshots; 0 foreign, 0 unreadable; 167,675,482 keys at 45.0 dB,
cosine mean 0.999984, least 0.999955. Not yet migrated — BlackWorld deploys no
motherd from main until it is.

## 9. The micro-harness, and the tuning method

The repository's kernel method (`docs/progressive_yarn.md` §7.6): **attribute
first, from `ncu` and SASS, then make one change and rebuild once.** Each kernel
this touches gets a harness before it is tuned, in `candle-transformers/tests/`,
beside the ones it extends (`qsa_score_rot_harness.rs`,
`qsa_page_place_bench.rs`, `qsa_index_bench.rs`):

- **Correctness first, in the same file.** The quantizer inline within one code
  of `quantize_key` (§3); a placed page its planes transposed, byte for byte; every
  scorer arm against the f64 oracle over the dequantized keys within the §5
  bound; the shadow scan and the GEMM tail against the same oracle.
- **A sweep that says which shape is slow:** depth {8K, 32K, 128K, 512K} ×
  query rows {1, 8, 16, 64, 512} × pages {16, 256} × layout {blocked,
  row-major}, each cell against the `F32` scorer in the same run — which means
  the `F32` kernel is kept **in the harness's own source** for the life of the
  tuning, not in the daemon. CUDA events, median and p90; the daemon stopped
  and the card idle.
- **A one-launch-per-shape target for `ncu`** (`…_profile_one`), and the `nsys`
  timeline of a wave for launch counts — the append's history is a 30-launch
  chain found exactly that way.

**What each round reads and must reach**, per kernel, from `ncu`:

| Target | Metric |
|---|---|
| **Full occupancy** | achieved occupancy vs theoretical (`sm__warps_active`), the limiter named (registers / shared / blocks), registers per thread and **zero spills**; `__launch_bounds__` set from it, as the scorer's `(256, 3)` was |
| **Full warps** | warp execution efficiency ~32 threads per instruction (`smsp__thread_inst_executed_per_inst_executed`); no divergent tail in the channel loops; every shuffle and vote under a full mask, warp-uniform loops |
| **Vector math** | SASS shows `LDG.E.128` for codes and `LDG.E.64` for scales, no scalar byte loads in the inner loop; `IDP.4A` (or `IMMA`/`HMMA`) once those rounds land |
| **Memory** | DRAM and L2 throughput against peak for the bandwidth-bound shapes (the placement's 87–88% is the bar it already meets); L1 hit rate per layout; bank conflicts < 1% of shared instructions |
| **Stalls** | issue-stall reasons — the scorer's history is a long scoreboard stall on table reads, fixed by shared rotation runs |

**The rounds, in order**, each recorded in a results table here as
`progressive_yarn.md` §7.6 records the scorer's:

1. The baseline reader: registers, spills, occupancy; `(TILE_R, CPT)` and
   `__launch_bounds__` re-derived for the new register mix.
2. Row-major against blocked codes (§3's open question).
3. `dp4a` on the pass-through half.
4. The append at decode width and at a prefill chunk.
5. Placement, if round 2 keeps it.
6. Tensor cores (§5).

### Round 0 — the baseline reader against `F32` (WhiteWorld, 2026-11-15)

RTX 3090, daemon stopped, card idle; `bench_rot_scorer`, median of 50, the
`F32` kernel built from `82e877016` in its own worktree. Milliseconds a launch;
the ratio is Q8_0 over `F32`.

| tokens | rows 1 | rows 8 | rows 64 | rows 512 | rows 2048 |
|---|---|---|---|---|---|
| 8K | 0.014 / 0.015 (0.92) | 0.014 / 0.016 (0.87) | 0.035 / 0.047 (0.74) | 0.221 / 0.159 (1.39) | 0.711 / 0.504 (1.41) |
| 32K | 0.017 / 0.017 (1.01) | 0.033 / 0.034 (0.97) | 0.133 / 0.094 (1.41) | 0.800 / 0.512 (1.56) | 3.04 / 2.07 (1.47) |
| 128K | 0.016 / 0.029 (**0.57**) | 0.044 / 0.066 (0.67) | 0.465 / 0.286 (**1.63**) | 3.40 / 2.95 (1.15) | 13.9 / 12.5 (1.11) |
| 512K | 0.036 / 0.091 (**0.39**) | 0.269 / 0.240 (1.12) | 1.78 / 1.63 (1.09) | 14.5 / 13.0 (1.12) | 56.7 / 52.5 (1.08) |
| 1M | 0.059 / 0.169 (**0.35**) | 0.476 / 0.445 (1.07) | 3.51 / 3.21 (1.10) | 28.6 / 26.8 (1.07) | 115 / 107 (1.08) |

(16 pages; 256 pages reads the same within a few percent.)

- **One query row** — a decoding sequence — is as fast or faster at every
  depth, up to 2.9×: that shape is bound by the key bytes it reads, and reads a
  quarter of them.
- **From 64 rows the reader is slower, 6–63%**, worst where the keys fit L2
  (32K–128K) and the kernel is bound by arithmetic: every key now costs 128
  int-to-float conversions and 128 scale multiplies ahead of the same FMAs.
  Eight rows past 512K is 7–13% slower.

**The decode and wide gates below are not met by the baseline reader.** It was
deployed on WhiteWorld regardless, on the absolute numbers: the worst cell is
+0.18 ms a launch against decode steps that waited about a second on expert
misses the freed span is there to remove. Rounds 1–3 are what close it, and
they need attribution this machine could not give: `ncu` refused with
`ERR_NVGPUCTRPERM` (GPU performance counters are not enabled for this account),
so registers, spills and occupancy are unread. Run them on BlackWorld, or
enable the counters here.

**Gates for adoption:**

- **Selection:** on recorded contexts and the harness sweep, Q8_0 and `F32`
  select the same top-512 blocks with **≥ 99% overlap**, and the attention
  outputs agree within the KV gate's tolerance; the Flash-Next needle gate with
  a dense control holds. Run once for the baseline and again for any round that
  changes arithmetic (`dp4a`, tensor cores).
- **Decode shape** (rows ≤ 16, any depth): no slower than the `F32` scorer
  within run-to-run variance — it reads a quarter of the bytes and adds 128
  conversions a key, and the harness says which wins.
- **Wide shapes** (64 rows and up): no slower than `F32`.
- **Append:** no slower than today at decode width, still one launch a wave.
- **On WhiteWorld after migration:** the QSA-index tenant shrinks by ~3.7×, and
  the expert zone grows by what it frees.

## 10. Order of work

1. **The quantizer:** `key_q8.rs` with `#[test]`s on raw bytes (rounding at
   ties, ±127, the zero block, a short last block, the four-block key) and the
   `qsa_key_q8.cuh` inline.
2. **Q8_0 on the card, in one change** — the format cannot be half in: key
   pages and resident pages as byte slots; append and flush; placement; the
   baseline scorer, the shadow scan and the GEMM tail; `SealedIndex` and the
   host types of §6; aux version 4 and the refusal of 3. Every existing index
   test moves to the planes.
3. **The micro-harness and its rounds** (§9).
4. **The selection gate** against `F32` on recorded contexts.
5. **The migration program** (§8), with its dry run.
6. **Main, then WhiteWorld and BlackWorld** migrated with their daemons stopped.
7. **The tensor-core scorer**, adopted only if it beats what stands at the
   decode shape.

On-disk moved into step 2 from a later step of its own: a seal reads the
card's bytes, so a build with Q8_0 on the card and `F32` on disk would need a
conversion in each direction that the finished design does not have.

`qwen4exp/wave.rs` is touched in step 2 (the seal functions,
`positional_lift_bytes_at`, `restore_aux_state`), not `claim_carried_state`.

### What the review changed

- Three card-side readers and writers the first draft missed: the shadow scan,
  the GEMM-tail rotation, and the host readback.
- Row-major pages are permanent (`close_tail_into_page`), not only the tail.
- The rounding rule is the repository's Q8_0 with a canonical zero; the kernel
  runs under fast math and is gated within one code of the Rust reference,
  since nothing reproduces its bytes.
- Widths that are not a multiple of 32, which the tests run.
- Scales row-major in both layouts; the descriptor at five words, not six; one
  slot a chunk.
- Blocking the codes is now a measured question rather than an assumption.
- `dp4a` quantizes the query, so it is a gated round after a float baseline
  rather than part of the first reader.
- One aux version covers every persisted form; no separate page version word.
- The migration rewrites under a new stamp and only this node's own pages —
  an equal stamp loses on reload, and a page is its writer's.
- On-disk lands with the card, not after it.
