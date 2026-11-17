# The warm lift: RAM → VRAM, batched and sync-free

A projection that selects history its slot does not hold brings that history hot
before it is injected (`elevate_to_hot`, `candle-conversation/src/persistence/elevate.rs`).
A turn whose K/V sits in the warm tier — host RAM — is copied into fresh VRAM arena
slots. This document replaces how that copy is done. It is the warm half of the tier
design in `docs/archived/kv_tier_migration.md`, whose §10 assumed a pinned warm pool;
that pool is not built and will not be (§2), so this is the design that stands.

## 1. What it costs today

BlackWorld (RTX PRO 5000 Blackwell, PCIe 5.0 ×16), one conversation, 08:42–08:48 UTC
on 17 November 2026, from the `elevate_to_hot phases` log lines:

| | Moved | Time | Rate |
|---|---|---|---|
| Warm → hot, 163 items in 88 lifts | 1,885 MiB | 1,707 ms | **1.1 GB/s** |
| Cold → hot, 483 items | 5,044 MiB | 2,357 ms | 2.1 GB/s |

The reprojection census over the same conversation put the lift at 3,316 ms of 52
rebuilds — 64 ms a rebuild, 21 % of all reprojection time and the largest part of it
after the scan. The link carries more than twenty times the warm rate.

The rate is not the link's. `elevate_to_hot` calls
`ChunkedKvBacking::migrate_sealed_to_gpu_batch_async` (`candle-nn/src/kv_cache/chunked/chunk_ops.rs`)
once per layer — 48 calls a lift — and each call:

1. resolves its sources into fresh `HashMap`s and `HashSet`s;
2. copies every chunk from its pageable CPU arena into a pinned scratch that grows to
   the size of the call;
3. allocates the destination slots, then **`device.synchronize()`** so the slots are
   valid on the copy stream;
4. uploads the staging bytes, and builds a copy plan that `kv_migrate_on` uploads
   through `with_synced_upload`, which **drains the stream** before it releases its
   scratch — after its own **`device.synchronize()`** cross-stream fence;
5. launches `kv_migrate_copy_kernel` — one block per record, one 16-byte load in flight
   per thread;
6. drops the staging generation, which **fences the copy stream**.

That is four host waits a layer, ~192 a lift, each queued behind every wave already on
the primary stream. A 2.8 MB lift is ~58 KB a layer: the cost is the waits, not the
bytes.

## 2. Constraints

- **The source cannot be pinned.** Pinned host memory is the expert weights' tier
  (`docs/expert_cache_design.md`); the warm tier stays in pageable CPU arenas. A DMA
  engine reads only page-locked memory, so every warm byte is copied by the CPU into
  pinned memory before the GPU can reach it. A pageable `cuMemcpyHtoD` is that same
  CPU copy done by the driver, at about half the link and blocking the caller;
  `cuMemHostRegister` on the source pages costs more than copying them and locks the
  memory the weights need. The CPU copy stays, and is made fast (§3.3).
- **The pinned memory the lift uses is bounded and fixed**: allocated once, never grown,
  so the weights' budget is a known number. Today's scratch grows to the size of the
  largest lift.
- **No host synchronisation on any stream** (`AGENTS.md`, hot-path invariants 3–5). The
  host may wait only to reuse a buffer of its own (§3.5).
- **One path.** The per-layer path is replaced, not kept beside it.

## 3. Design

A lift is one plan over every layer and every item, executed through a two-slot pinned
ring: host threads fill one slot from the warm arenas while one scatter kernel drains
the other into the destination arena slots.

```
 warm arenas (pageable)                 pinned ring (2 × SLOT_BYTES)          VRAM arenas
 ──────────────────────   4 host        ┌──────────── slot 0 ────────────┐   scatter kernel,
 layer 0..47, item 0..n ──threads──────▶│ payload bytes │ descriptor table │──▶ W blocks, reads the
                          memcpy        └────────────────────────────────┘   slot at its host address
                                        ┌──────────── slot 1 ────────────┐   (mapped), 4×16 B in
                          (overlapped)  │ payload bytes │ descriptor table │──▶ flight per thread
                                        └────────────────────────────────┘
```

### 3.1 The plan

One pass over all layers and items, under each backing's storage read lock once:

- every unique source band gid with its byte length (from its tag, as now), its
  arena and slot;
- the destination slots, allocated in one `state.write()` per backing — palette-band
  runs contiguous, as `migrate_sealed_to_gpu_batch_async` does today;
- the plan cut into **slot batches** of at most `SLOT_BYTES` payload each, in source
  order, a band never split across two batches.

Each batch's descriptor table is `{src offset in the slot, dst device address, bytes}`
per band — the shape of `MigrationPlan`'s records, and of the expert path's item table.

### 3.2 The ring

Two slots, each `SLOT_BYTES` of payload plus room for its descriptor table, allocated
once with the scheduler's other persistent host buffers and held for the process.
Write-combined pinned memory (`PinnedBuf`, `CU_MEMHOSTALLOC_WRITECOMBINED`): the CPU
only writes it and the GPU only reads it, which is the pattern write-combining is fast
for. Mapped, so the kernel reads it at its host address.

**`SLOT_BYTES` = 8 MiB.** A kernel pulling pinned memory reaches the link's rate with
pieces of ~0.8 MB (`docs/moe_live_dispatch_design.md` §0.10.1: an 819,200 B projection
at ≈ 11.9 GB/s). A slot costs a launch and an event, ~15 µs, against the time its bytes
take to move: ~20 % at 1× the minimum on a 12 GB/s link, ~3 % at 8×. Above 8× the
pipeline only takes longer to fill and the weights lose pinned memory. The constant is
written in the code as that measured minimum times the multiplier, with this reasoning
beside it. 16 MiB pinned in all.

A lift of at most 16 MiB — most of them: this conversation's ran 2.7 MB to 321 MB, the
median under 10 MB — fills both slots at once and never waits for the ring.

### 3.3 The host copy

`HOST_COPIERS` = 4 threads, scoped to the slot and joined before its scatter is
launched — not the global rayon pool, whose queue the belief scan's warm-ups share.
A slot batch is split into four contiguous runs of bands of about equal bytes, one a
thread; each copies its bands from their arena slots into its own byte range of the
slot (`Arena::copy_slot_bytes`), under one storage read of each layer it crosses.
Spawning four threads costs tens of microseconds against a slot's ~0.5 ms of copying.
Four streams of pageable reads outrun a ×16 link; one does not on PCIe 5.0, and on
the 3090 and the 4090 Laptop (~12 GB/s links) the link is the limit with room to spare.

The scatter launch for a slot returns at once, so the copiers fill the next slot while
the GPU drains the last.

### 3.4 The scatter kernel

`warm_lift_scatter` (`candle-kernels/src/simple/`, beside `kv_migrate.cu`), one launch
per filled slot on the device's stream:

- `W` = 8 blocks of 128 threads (`docs/moe_live_dispatch_design.md` §0.10: 4–8 blocks
  saturate the link; more only hold SM slots while waves run);
- blocks take bands from the slot's descriptor table — read from the slot itself, so
  **nothing is uploaded**;
- each thread keeps four 16-byte loads in flight (`ld.global.cg.v4.u32`), then stores
  them — the copy loop of `moe_live_ahead_piece` (`candle-kernels/src/quantized/kernel.cuh`),
  reused, not reinvented. A plain one-load loop over the same memory measured 2.2 GB/s;
- loads are `ld.global.cv`, not `.cg`: the host rewrites a slot in place every time it
  reuses it, and `.cg` may be served from an L2 line an earlier batch's read of the
  same address left behind — the reason the expert path's CPU result ring reads `.cv`;
- a band whose destination or length is not 16 B-aligned is copied bytewise; every
  band starts 16 B-aligned in the slot because the host lays them out that way.

The destination is written directly: no device staging buffer, no second copy.

### 3.5 Ordering, with no host sync

Every caller of `elevate_to_hot` passes the device's own stream, so the lift runs
there and two of the three orders it needs are the stream's own FIFO:

1. **A destination slot is written only after every queued read of its previous
   tenant.** A freed KV slot is reissued, and waves already queued on the stream may
   still read what it held. The scatter is enqueued behind them. The per-layer
   `device.synchronize()` the old path paid for this bought nothing on one stream.
2. **A forward reads a lifted chunk only after it is written.** Every later launch on
   the stream runs after the scatter. The lift returns as soon as the work is enqueued.
3. **A ring slot is refilled only after its scatter has read it.** Each slot carries the
   event recorded after its launch; refilling the slot waits on that event
   (`WarmLiftRing::acquire`). This is the only host wait in the design. It never
   happens for a lift of at most two slots, and otherwise it waits for the GPU to reach
   the scatter, which order 1 may place behind queued waves — once for the lift, where
   the old path paid ~192 waits.

The metadata records stay as they are: `build_meta_records` fills them on the device
from a descriptor table, asynchronously (`candle-nn/src/kv_cache/chunked/backing.rs`).

### 3.6 Failure

A lift that cannot allocate its destinations fails before any byte moves, as now. A
launch or event failure fails the whole lift: `elevate_to_hot` counts every item as
failed and the projection drops them, as it does today when a layer's migrate fails.
No partial lift is installed.

## 4. The cold lift

Cold items are read from the redo log (`cold_load_section_into_hot`, the turn recall
path) and are 2.1 GB/s today, bound mostly by the read. The scatter stage is the same:
a cold read lands in a ring slot instead of a pageable buffer — there is no pageable
hop at all — and `warm_lift_scatter` moves it. The expert stager reads NVMe unbuffered
into pinned memory the same way (`docs/moe_live_dispatch_design.md` §0.10.2). **To check
before it is written:** whether a cold record's on-disk bytes are the arena payload
verbatim, or need decoding on the way in; if they need decoding, that step runs on the
copiers and the scatter is unchanged.

## 5. Where the code goes

| Piece | Where |
|---|---|
| A layer's sources resolved, destinations claimed, sequences rebuilt | `candle-nn/src/kv_cache/chunked/warm_lift/layer.rs` |
| The bands cut into slot batches | `candle-nn/src/kv_cache/chunked/warm_lift/plan.rs` |
| The ring and its events | `candle-nn/src/kv_cache/chunked/warm_lift/ring.rs` |
| The host copiers | `candle-nn/src/kv_cache/chunked/warm_lift/copiers.rs` |
| The lift (plan → fill → scatter → rebuild) | `candle-nn/src/kv_cache/chunked/warm_lift/mod.rs`, `lift_warm_to_gpu` |
| The scatter kernel | `candle-kernels/src/simple/warm_lift.cu`, FFI in `candle-kernels/src/simple/warm_lift.rs` |
| The ring's memory | `PinnedBuf::alloc_owned_mapped` (`candle-core/src/quantized/pinned_staging.rs`) |
| The caller | `elevate_to_hot`'s warm leg, one call in place of the layer loop; the scheduler holds the ring (`Scheduler::warm_lift_ring`), allocated at the first lift |
| Removed | `migrate_sealed_to_gpu_batch_async`, and `elevate_to_hot`'s `copy_stream` and `pinned_scratch` |

`migrate_sealed_to_cpu_batch_async` — hot → warm, on the persistence thread — is not
changed here.

## 6. Tests

In `candle-nn/src/kv_cache/chunked/tests/warm_lift_tests.rs` unless named:

- **Byte for byte against the source** (GPU): three layers × two items lifted through
  a ring of 4 KiB slots — hundreds of batches, every slot refilled many times — and
  every lifted sequence's stored bytes equal its original's.
- **The round trips** in `chunk_ops.rs` and `tests/compress_tests.rs` (hot → warm →
  hot → warm, BF16, F16, quantized, a band narrower than its slot) run through the
  lift and compare raw bytes.
- **No host sync**: with the device's stream held shut by a `StreamGate`
  (`tests/stream_gate.rs`), a lift returns long before the gate would open.
- **The plan** (CPU, `warm_lift/plan.rs`): batches packed 16-byte aligned, cut at the
  slot and at the table's room, never split a band, every band placed once.
- Order 1 of §3.5 is the stream's FIFO, which needs no test of its own.

## 7. What it should reach

- **Rate:** the lower of the link and the host copy — the link's ~12 GB/s on the 3090
  box and the 4090 Laptop; on BlackWorld probably the host copy, several times today's
  1.1 GB/s either way.
- **Waits:** none for a lift of at most 16 MiB; at most one GPU drain to the first
  scatter for a larger one.
- **This conversation's warm lifts:** ~1.7 s → well under 0.2 s.
- **Measured by** the `elevate_to_hot phases` line (`warm_hot_ms`, `warm_bytes`) and the
  reprojection census's `elevate_ms`, before and after, on BlackWorld and the 3090 box.

## 8. Not this design

**Bytes that should not move.** The same conversation's KV region relief evicted 730 MiB
of hot turns at 08:43 and 450 MiB at 08:48 to make room, and ~7 GB was lifted in six
minutes — far more history than one conversation selects. Turns evicted and then lifted
again cost a lift however fast it is. Counting re-lifts by turn, and keeping what the
next rebuild will select, is the larger lever and a design of its own.
