# Persisting the QSA index across a process restart

Flash-Next (`qwen4exp`) carries four classes of state outside the paged K/V. Three
of them were already durable. The fourth — the QSA index — was not, and it is the
one that cannot be rebuilt from the K/V, because its keys are derived from
**hidden states** rather than from stored K.

This document is the design and the contract for making it durable.

---

## 1. What the index is, and why it is not derivable

`project_keys(h, w)` builds one index key per completed block from the layer's
**input hidden state**. Layer `L`'s keys need layer `L`'s input, which needs every
layer below it — attention *and* MoE both write the residual. There is therefore
no "lean prefill" that could rebuild the index from restored K/V: reconstructing
it costs a genuine forward over the whole history.

The system already runs a genuine forward in three of the four places a sequence
can acquire K/V it did not compute:

| Path | Mechanism | Index |
|---|---|---|
| Splice / adopt-by-reference | `catch_memory_up_to` → `MemoryCatchUp` | rebuilt by the forward |
| Reprojection | new view fork, then decode | forked from the parent |
| Branch checkpoint | `BranchCheckpointPass` | built by the pass's prefill |
| **Process-restart resume** | substrate reload | **nothing rebuilt it** |

The fourth row is the gap this work closes.

---

## 2. The ragged boundary

A turn ends where its text ends. The index pools `ratio` tokens per row, so a
turn of `T` tokens leaves `T mod ratio` tokens **carried** in the cache's open
buffer, not yet pooled into a row. The cache's own invariant is

```
n_blocks · ratio + n_open == tokens
```

and every consumer depends on it: `plan()` derives the next append from `n_open`,
and the scorer addresses block `k` as tokens `[k·ratio, (k+1)·ratio)`.

The K/V's partial-chunk padding does not help here — the index counts **real**
tokens while a chunk pads dead ones, so the two remainders are unrelated.

This gives two distinct problems, which want two distinct solutions.

### 2.1 Resume of one continuing sequence — carry the open block

The sealed record stores the completed rows **and** the open block's raw rows, so
a restore rebuilds the cache exactly. `n_blocks · ratio + n_open == T` holds on
the far side, the scorer's uniform arithmetic is untouched, and the resumed
sequence appends and selects bit-identically to one that never stopped.

Persisting only the completed rows is the failure this design exists to prevent:
the restored cache would stand `T mod ratio` behind its own K/V and *stay* there.
Both sides remain internally consistent, every shape matches, and nothing is
raised — the conversation simply attends to the wrong history for the rest of its
life.

### 2.2 Reconstruction from a subset of turns — ragged pages

A window assembled from the turns a projection selected is not one buffer. Each
turn's piece is separately allocated and its last row is short, because that
turn's boundary was ragged. Rows stay ordered by token position, so "wholly below
this query" is still a **prefix** however wide each row is — only the mapping
from a position to that prefix changes, from `(pos + 1) / ratio` to a walk over
the pages' widths.

Two pieces make this work:

- **`IndexCache::flush_open_block`** — closes a turn on a block boundary by
  pooling its carried rows into one **short** block, over the count actually
  present rather than `ratio`. Without it a turn's piece is not self-contained:
  its trailing tokens belong to a block the *next* turn completes, so a window
  built from a subset of turns has a leading block pooled over tokens that are
  not in the window. The flushed block is deliberately a summary of fewer tokens
  and is therefore **not** what a continuous run would have produced for that
  span — that is the trade, and it is the right one, because a short block is
  expressible on the selection side (`pack_entry(block, cells)` already carries a
  2-bit cell count) while a block pooled from another turn's tokens is not
  correctable at all.
- **`qsa_score_paged`** — a ragged, paged scorer that takes a descriptor table of
  page pointers plus a per-row candidate count, and never sees a width. The
  widths are folded into `cnt` on the host, where the page table already lives.

Nothing is concatenated. Materialising the window would copy every key of every
selected turn on the step that reconstructs, which at conversational depth is the
largest copy on that path and buys the scorer nothing it cannot already read
(hot-path invariant 2b).

---

## 3. The record

The index rides the **same** record as the DeltaNet snapshot, in a
model-opaque `aux` blob:

```
SnapshotPayload { timeline_id, turn_index, schedule_hash, layers, aux }
BranchCheckpointPayload { prefix_hash, schedule_hash, layers, aux }
```

`layers` is one shape — a per-layer delta-rule matrix plus a conv tail — and it
is the right shape for the state that owns it. It is not the only recurrence a
model can carry. So `aux` is bytes: persistence carries the blob, checksums it
with the rest of the record, and never decodes it. That is what keeps **one**
seal path serving every architecture instead of growing a branch per model. A
model with no such state writes an empty blob and pays a length word.

Both classes are exported at one instant, from one sequence, and land in one
record. A path that exported the layers and picked the blob up separately could
pair a state from one seal with a blob from another and produce a sequence whose
two halves of memory disagree about how much history they have seen.

Inside the blob, the model's own container (`paged_index::encode_aux`) holds the
PLE window and one sealed index per attention layer. Each layer's section carries
its completed rows, its `last_cells` width, and its open block — the layers seal
at one instant but run at different ratios, so their remainders genuinely differ.

---

## 4. The seeded rule

A slot holding state it did not compute legitimately stands at **offset 0**: a
fork borrows the parent's K/V, and a resume installs its state before the first
wave. So "offset 0 means start over" throws away precisely the state that was
just installed.

`HybridBatched::ensure_recurrent` has carried this rule for the delta-rule store
since the state-persistence work (`RecurrentStateStore::take_seeded`).
`Qwen4ExpBatched::ensure_seq_state` did not, and it resets four classes rather
than one. It now takes the same flag, once per wave, before any class block — a
flag read per class would be true for the first and false for the rest, resetting
three quarters of a restored sequence.

The flag is consumed on the first wave whether or not it fires, and dropped on
`release_sequence`. A flag that outlived its wave would suppress a later, genuine
reset; a flag that survived the slot would let the next sequence handed that id
inherit the previous one's memory.

---

## 5. Measured

### The paged scorer

RTX PRO 5000 Blackwell (sm_120, 110 SM), CUDA-event timed, median of 50 launches.
`head_dim` 128, 4 heads, ratio 4.

**Headline, 128K depth (32,768 rows), 64 query rows:**

| | ms | Gcell/s | limiter after |
|---|---:|---:|---|
| First working version | 0.187 | 11.20 | L2 87.8% |
| Row tile (`TILE_R`) | 0.128 | 16.39 | L1/TEX 87.9%, L1 hit rate **2%** |
| Channel-blocked pages | 0.103 | 20.34 | L1/TEX 40.9% — nothing above 56% |
| Grid-aware tile search | 0.096 | 21.74 | registers 124 → 2 blocks/SM |
| Tile pair `(4, 2)` | **0.094** | **22.20** | compute 67.2%, occupancy 66.7% |

**2.0× overall**, and the whole shape table moved with it:

| depth | rows | pages | queries | before | after | |
|---:|---:|---:|---:|---:|---:|---|
| 8K | 2,048 | 16 | 1 | 0.019 | 0.018 | — |
| 8K | 2,048 | 16 | 8 | 0.029 | 0.018 | **1.61×** |
| 32K | 8,192 | 64 | 1 | 0.018 | 0.018 | — |
| 32K | 8,192 | 64 | 8 | 0.030 | 0.020 | **1.50×** |
| 128K | 32,768 | 256 | 1 | 0.022 | 0.019 | 1.16× |
| 128K | 32,768 | 256 | 8 | 0.036 | 0.028 | **1.29×** |
| 128K | 32,768 | 256 | 64 | 0.128 | 0.094 | **1.36×** |

#### What moved it

1. **Channel-blocked page layout** (`[D/4, rows, 4]`). The record's `[rows, D]`
   puts consecutive candidates `D×4` bytes apart, so each warp's `float4` key
   load scattered over 32 cache lines — **L1 hit rate 2%**. Blocking the channel
   axis outward makes a warp's read 512 contiguous bytes: 4 wavefronts, not 32.
2. **A grid-aware tile search.** Both tiles buy reuse and both shrink the grid,
   and which one starves depends on the shape: at 64 rows `(4, 4)` beats `(4, 1)`
   0.101 → 0.113, and at 8 rows the same pair *inverts*, 0.035 → 0.031, purely
   because `(4, 4)` leaves 64 blocks on a 110-SM part. The launcher now searches
   from most reuse to least and takes the first arm whose grid fills the device.
3. **A branch-free channel loop**, with the tail candidate pointed at a live key
   instead of null, so the loads are not re-serialised by a test between them.
4. **Unrolling only in the arm that could not fill the grid** — the one place
   where latency has to be hidden inside the thread rather than across warps.

#### What was tried and measured worse

Recorded because the reasons are the useful part, and because each of these
looks like an obvious win on paper:

- **Accumulating with four `fmaf`s instead of a summed temporary.** Strictly
  fewer instructions — the measured FFMA count is exactly 0.75× the arithmetic,
  which is the FMUL + 3×FFMA + FADD form — yet **0.096 → 0.101 ms**. The
  temporary keeps three of the four multiply-adds off the accumulator's
  dependency chain, so the critical path is 32 links instead of 128, and at 6.9
  active warps per scheduler the path costs more than the instructions.
- **Spreading the row tile across threads instead of stacking it in one.** Cuts
  the accumulator from `TILE_R×H×CPT` to `H×CPT` and would lift occupancy from
  66.7% to 100%; measured **0.094 → 0.115**. Stacked, a key `float4` is loaded
  once and serves every row from a register. Spread, each of the `TILE_R` threads
  loads it — a broadcast from shared is far cheaper than a redundant load from
  global, so the row axis belongs in registers even at the cost of occupancy.
- **A `minBlocksPerMultiprocessor` floor** to force a fifth block per SM.
  Measured **0.094 → 0.152**: the 51-register budget does not fit the tile's live
  values and it spilled. nvcc's own 64 was already right.

#### What is left, and why it is not taken

The kernel is compute-limited (SM throughput 67.2%, nothing else above 55%), so
the remaining FP32 headroom is roughly 1.9× against the FFMA-issue floor and it
is scheduling, not bandwidth. The only lever that would beat that floor is
**tensor cores**: this is a GEMM (`M = rows·H`, `N = n_cand`, `K = head_dim`)
with a fused per-head ReLU and mask, and TF32 MMA would run it several times
faster.

It is not taken, and the reason is not performance. TF32 carries ~10 mantissa
bits, so scores acquire a relative error near 1e-3 — comfortably inside the
oracle's 2e-3 tolerance, and *not* comfortably inside a top-k. Blocks whose
scores differ by less than that would swap at the selection cut, changing which
history a resumed conversation attends to. That is a retrieval-quality change
this benchmark cannot see and no test here would fail on, in the one path whose
entire purpose is making resumption faithful. The speed is measurable; the cost
is not, so the trade is refused rather than made blind.

### Long-context regression

`test_long_context_scaling`, before and after the whole change. Bulk t/s is the
comparable figure; the single-session column swings by more than this between
runs of an unchanged build, so it is reported but not read as a signal.

| Rung | Baseline bulk | After | Δ |
|---|---:|---:|---:|
| 32K BF16 | 1172.1 | 1261.2 | +7.6% |
| 32K C5 | 1246.3 | 1360.2 | +9.1% |
| 32K C10 | 1313.8 | 1359.6 | +3.5% |
| 128K BF16 | 1274.2 | 1266.1 | −0.6% |
| 128K C10 | 1271.7 | 1259.1 | −1.0% |

Every rung coherent (100% pass) on both runs. The 128K bulk rate matches the 32K
rate on both, which is the depth-flat property holding.

---

## 6. Where each piece lives

| Piece | File |
|---|---|
| Ragged/paged scorer kernel | `candle-kernels/src/simple/qsa_score_paged.cu` |
| Short-block flush kernel | `candle-kernels/src/simple/qsa_index_append.cu` (`flush_kernel`) |
| Page window + container | `candle-transformers/src/models/qwen4exp/paged_index.rs` |
| Cache export/restore | `candle-transformers/src/models/qwen4exp/indexer.rs` |
| Model hooks | `candle-transformers/src/models/qwen4exp/wave.rs` |
| Trait hooks | `candle-transformers/src/models/batched_inference.rs` |
| Record | `candle-conversation/src/persistence/record.rs` |
| Seal + resume | `candle-conversation/src/scheduler/mod.rs` |
| Both classes, one value | `candle-conversation/src/scheduler/exported_state.rs` |
