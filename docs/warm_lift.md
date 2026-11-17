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

`HOST_COPIERS` = 4 threads, owned by the lifter and parked between lifts — not the
global rayon pool, whose queue the belief scan's warm-ups share. A slot batch is split
into four contiguous byte ranges, one a thread; each copies its bands from their arena
slots into the slot (`Arena::copy_slot_bytes`, as `read_chunk_into_pinned_bytes` does
now) and writes its share of the descriptor table. Four streams of pageable reads
outrun a ×16 link; one does not on PCIe 5.0, and on the 3090 and the 4090 Laptop
(~12 GB/s links) the link is the limit with room to spare.

The copy runs on the lifter's threads while the scheduler thread builds the next batch
and enqueues the previous one — the scheduler never copies bytes itself.

### 3.4 The scatter kernel

`warm_lift_scatter` (`candle-kernels/src/simple/`, beside `kv_migrate.cu`), one launch
per filled slot on the backing's copy stream:

- `W` = 8 blocks of 128 threads (`docs/moe_live_dispatch_design.md` §0.10: 4–8 blocks
  saturate the link; more only hold SM slots while waves run);
- blocks take bands from the slot's descriptor table — read from the slot itself, so
  **nothing is uploaded**;
- each thread keeps four 16-byte loads in flight (`ld.global.cg.v4.u32`), then stores
  them — the copy loop of `moe_live_ahead_piece` (`candle-kernels/src/quantized/kernel.cuh`),
  reused, not reinvented. A plain one-load loop over the same memory measured 2.2 GB/s;
- a tail shorter than 16 B is copied bytewise; every band starts 16 B-aligned in the
  slot because the host lays them out that way.

The destination is written directly: no device staging buffer, no second copy.

### 3.5 Ordering, with no host sync

Three orders matter and each is an event, never a wait on the host:

1. **A destination slot is written only after every queued read of its previous
   tenant.** A freed KV slot is reissued, and waves already queued on the primary
   stream may still read what it held. The primary stream records `alloc_done` after
   the allocations; the copy stream waits on it (`cuStreamWaitEvent`) before the first
   scatter. This is what today's `device.synchronize()` buys, without blocking the host.
2. **A forward reads a lifted chunk only after it is written.** The copy stream records
   `lift_done` after the last scatter; the primary stream waits on it before the next
   wave is enqueued. The lift returns as soon as the work is enqueued.
3. **A ring slot is refilled only after its scatter has read it.** Each slot carries the
   event recorded after its launch; a copier about to refill a slot waits on that
   event. This is the only host wait in the design. It never happens for a lift of at
   most two slots, and otherwise it waits for the GPU to reach the scatter, which
   order 1 may place behind queued waves — once for the lift, where today pays it ~192
   times.

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
| The plan across layers | `candle-nn/src/kv_cache/chunked/warm_lift/plan.rs` |
| The ring and its events | `candle-nn/src/kv_cache/chunked/warm_lift/ring.rs` |
| The host copiers | `candle-nn/src/kv_cache/chunked/warm_lift/copiers.rs` |
| The lift (plan → fill → scatter → events) | `candle-nn/src/kv_cache/chunked/warm_lift/mod.rs` |
| The scatter kernel | `candle-kernels/src/simple/warm_lift.cu`, FFI in `candle-kernels/src/lib.rs` |
| The caller | `elevate_to_hot`'s warm leg, one call in place of the layer loop |
| Removed | `migrate_sealed_to_gpu_batch_async` and its `pinned_scratch` |

`migrate_sealed_to_cpu_batch_async` — hot → warm, on the persistence thread — is not
changed here.

## 6. Tests

- **Byte for byte against today's path** (GPU): lift the same multi-layer, multi-item
  warm set through both, read every destination slot and every metadata record back,
  assert equal bytes. The old path stays in the test module as the reference until
  this passes, and is deleted with it.
- **A lift larger than the ring** (several slot batches, a band at a batch edge), and
  one of exactly one slot: every byte lands.
- **No host sync**: a test that closes a stream gate on the primary stream
  (`StreamGate`, `candle-nn/src/kv_cache/chunked/tests/decode_slot_resync_tests.rs`)
  and asserts a two-slot lift returns while the gate is shut.
- **Ordering**: a destination slot freed by a queued read is not written before that
  read retires — the same gate, with the read behind it.
- **The plan** (CPU): batches never exceed `SLOT_BYTES`, never split a band, cover
  every source once; the descriptor table's raw bytes for a fixed input.

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
