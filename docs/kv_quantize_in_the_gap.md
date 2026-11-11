# Quantize in the gap: K/V is compressed where it is made, and persistence only copies

**Status:** design, not built. Where this document and the code disagree, the
code is to be brought to this document. It supersedes the hot→warm quantize of
`docs/kv_drain_under_pressure.md` §2a–§2c and §3, and restores the write path of
`docs/archived/kv_tier_migration.md` §3 and §11, which the build departed from.

## 1. What is wrong now

Measured on BlackWorld, 2026-11-11, conversations `01bbad1e…` and `c322c8f2…`.
Both were codebase tours, 9 and 7 model rounds, run on build 4819-e022ff2f.

K/V is written to VRAM native (R16/F16) and stays native until the persistence
thread's hot→warm pass reaches it. That pass quantizes the K/V, copies the
quantized bytes to host RAM, and swaps each residence's hot copy for its
quantized form (`persistence/thread.rs`, `migrate_group_hot_to_warm` and
`install_hot_to_warm`). Four costs follow from that.

| Symptom | Cause | Measured |
|---|---|---|
| Decode stutters: 0.4–0.7 s with no forward in each 2 s window, while each forward takes 30–37 ms at 99.9% expert hits | Each group's work is queued on the device's primary stream, then fenced with a device-wide `device.synchronize()`. The fence exists because the install publishes the swap before the convert has finished, and a reprojection could otherwise borrow half-written V (the "V-only multi-turn duplication corruption"). | `kv_reserve` passes of 65–801 ms every ~2 s during decode, 80–566 MiB each |
| A native backlog stands in VRAM at about 4× its final size | Quantization waits for the drain, which works oldest-first across every conversation and ingest reading on the member | `pending_mib` 1.4–2.2 GB for the whole turn |
| The KV side takes ground from the expert zone | The quant reserve is sized from that backlog (`pending × factor`, capped at 25% of the KV regions) | `cap_mib=3111`, `kv_mib` up to 12,448; "grow class2048 by 24 arenas, reason=backlog" |
| Durability lags by minutes | A turn's commit waits for its turn in the drain's queue | "a finished turn's commit landed … waited_s=319" |

The batched test (`batch_test/utils.rs`, `quantize_and_seal_sequences`) already
quantizes in place, on the forward thread, at the prefill and decode
boundaries. But it does so outside the timed windows, so the sweep has never
seen any of these costs.

## 2. The model

**K/V is quantized on the scheduler thread, in the gap after every admission
pass, where arenas can be created and freed.** Every complete 32-token block
written since the last gap is quantized in that gap, together with every partial
block of a sealed residence. That covers prefill, decode, sections, ingest
readings and every sequence. The only K/V left native at the end of a gap is
each live sequence's **writer chunk**: the partial block a forward will still
append to. Nothing else waits.

**Persistence only copies.** The persistence thread copies blocks that are
already quantized to pinned host memory, on its own stream, and appends them to
the redo log. It never writes device memory and never changes what a residence
points at.

```
 forward ─▶ native writer chunk + newly complete native blocks
   │
 gap   ─▶ select + claim + convert   (one batched launch set, primary stream)
   │       native blocks freed once every holder points at the quantized block
   │       event E_gap recorded
   │
   ├────▶ next forward reads the quantized prefix   (stream order: no fence)
   │
   └────▶ persistence: wait on E_gap on the copy stream ─▶ DtoH into the pinned
          warm buffer ─▶ log append ─▶ commit ─▶ has_warm published
```

The two products then match: the batched test and production run the **same**
gap quantize, inside the timed windows. The test's two boundary calls are
deleted.

## 3. The gap quantize

### 3.1 Inputs, without a readback

The host knows, without asking the device, every block that became complete in
the forwards since the last gap. It scheduled those rows, so it knows each
sequence's offset, and it allocated the native chunks those blocks live in. From
its own bookkeeping it builds a **descriptor table**: one row per block, giving
the source K/V pointers, the (layer, head) layout, the block's token range and
its sequence's position (for attention-sink protection on positions 0–3). That
is invariant 2b's shape. No block is copied together to be quantized.

A sealed residence's partial final block is quantized as a full block. Its dead
slots are zero and the valid range is passed to the selection kernel, as
`kv_tier_migration.md` §3 already specifies. A live sequence's writer chunk is
not in the table.

### 3.2 Device-side destination claim

Today the format selection runs on the device and is then read back, so the host
can allocate each block's destination in the class its chosen format needs. On
the scheduler thread that would be a synchronous readback in the loop, against
invariant 3. Instead, **the host leases slots ahead of time, and the device
chooses among them.**

- **Leases.** In each gap the host claims, for every reachable quant class, a
  lease of free slots from that class's arenas (`ChunkGidPool`, lowest place
  first). It holds their `ChunkGid`s and uploads the lease as a device-resident
  **slot stack** per class: one `{arena ptr, slot}` entry per slot, plus a
  device cursor.
- **Claim.** The fused selection kernel picks each (block, head)'s K and V
  formats exactly as today. It then takes a destination by `atomicAdd` on that
  class's cursor and writes the slot into the convert's descriptor. Nothing comes
  back to the host.
- **A class that runs dry defers the block, never fails it.** If a cursor passes
  its lease, that block stays native and goes into the next gap's table. The
  attempt is counted in the summary (§3.4). The host grows that class's next
  lease. A block that stays native for a gap is still attended correctly, only
  larger, and this is the only way native K/V outlives a gap.
- **Lease size** comes from what gaps actually claim. Per class, it is the
  recent per-gap peak, plus one arena's headroom, read from the summaries. This
  replaces §2a's backlog-sized reserve: the lease covers one gap's demand, not
  the whole backlog. The lease is KV ground and is bought below `weight_floor`
  (invariant 7). On a burst, a class may defer blocks for a gap rather than take
  ground from the experts.

### 3.3 Convert and swap

- **Convert.** One batched launch over every claimed (block, head) descriptor,
  across every layer and sequence (invariant 5). Destinations are allocated
  uninitialised (invariant 6).
- **Swap on the device.** The convert's epilogue rewrites, for each converted
  block, the block-table entry of every live slot that attends it. These are the
  `ArenaTableEntry` / `PerHeadTableEntry` rows: pointer and format tag. The next
  forward on the same stream therefore reads the quantized block, with stream
  order as the fence. Where a block is borrowed by several slots (a shared
  prefix), the descriptor names each table row to rewrite.
- **The host catches up afterwards.** The host's records (`SealedSequence` /
  `SealedChunk`: gid and format per block) are re-pointed when the host reads
  the summary. A native chunk is **freed only after every host-side holder has
  been re-pointed**. A reprojection built from host state before that point
  still names native chunks, which are still valid, so it is correct and merely
  larger for one gap. Native slots then return to their pool with no fence:
  anything that writes them later is queued behind the convert on the same
  stream.

### 3.4 The summary

The kernels write a per-gap summary into a mapped pinned ring, as the MoE
dispatch does with its routing summary (`docs/moe_live_dispatch_design.md`):
- for each block, the class and lease index it took, or "deferred";
- each class's final cursor.

The host reads it in a following gap, with no wait on the device:
- it marks the used gids live and returns the unused ones;
- it re-points records and frees natives (§3.3);
- it sizes the next leases.

No host CUDA call reads or waits on the ring. The ring is bounded, and a summary
the host has not read keeps its leases held. The lease sizing absorbs the
latency.

### 3.5 Cost in the gap

The gap now does work proportional to the K/V produced since the last gap. That
is one selection, one claim and one convert, all batched. A 6,144-row prefill
forward produces 192 blocks per layer. Decode produces about one block per
sequence-layer every 32 steps. The gap is not budgeted: everything complete is
quantized. Its cost is measured, inside the timed windows (§7), and the rate
model's decode step includes it.

## 4. What a forward attends

Every forward attends a quantized prefix plus at most each sequence's writer
chunk in native form. A long prefill's later forwards attend that same
prefill's earlier blocks quantized. Decode attends its own earlier tokens of
the turn quantized once each block completes.

This is a change in what the model sees. Today a turn attends its own tokens
native until the drain reaches them; the gates attend a quantized prefix and a
native decode tail. Under this design the gates and production both attend
per-block quantization. Every gate and the sweep are run on it:
- per C-level fidelity (`batch_test/fidelity.rs`);
- needle and long-context recall;
- the §9.12 battery.

A level that falls outside its validation edge has its `*_KV_FACTORS` row
re-derived by measurement (`sampled_selection/params.rs`). It does not get a
second path that keeps the turn native.

Attention-sink protection is unchanged. Positions 0–3 take the dedicated fine
scale whichever gap quantizes them.

## 5. Persistence

- **A dedicated copy stream.** The persistence thread owns a non-primary
  stream. For each residence it copies, it makes that stream wait on the
  `E_gap` of the gap that quantized the residence's last block
  (`cudaStreamWaitEvent`), then issues one DtoH of the gathered quantized blocks.
  No `device.synchronize()` anywhere on this path.
- **Pinned warm tier.** The DtoH lands in pinned host buffers that are both the
  redo-log write buffer and the warm cache (`kv_tier_migration.md` §3). This
  replaces today's pageable CPU arenas, so the copy runs at full link bandwidth.
  On WDDM, pinned memory is charged to system commit (`docs/driver_model.md`),
  so the pool's size is read from the machine's RAM and commit headroom, never
  fixed. The three dev machines have 32, 64 and 189 GB.
- **Holding while copying.** The copy holds the quantized blocks' `ChunkGid`s
  until its own event completes, so no free or compaction can recycle them under
  it. `HeldChunks` already lets a copy run beside a compaction pass, which steps
  around held chunks. `MigrateFlight`'s quantize-versus-compaction exclusion is
  deleted, because the quantize and compaction now run on the same thread.
- **Publishing.** `has_warm` and the residence's commit are set once the copy's
  event has completed. The persistence thread polls it with `cudaEventQuery`.
  `turn_claims::release` waits only for that.
- **The writer chunk** is persisted as today: a native `Chunk` record that a
  later record supersedes (`kv_tier_migration.md` §5.5).
- **Hot after warm.** The hot copy is already quantized and is left alone.
  Dropping it under pressure, or as an idle demote, is a free, not a copy.

## 6. What is deleted

- From `persistence/thread.rs`: the hot→warm quantize, convert and install
  phases (`quantize_layers_deferred` and `convert_deferred_descs` calls,
  `install_hot_to_warm`), the per-group `device.synchronize()`, the wave
  deferral and gap retries (`wave_deferred`, `await_owed_arenas`,
  `MAX_GAP_RETRIES`), and `GroupCap`.
- The backlog-sized reserve: `pending`, `reserve_targets`' backlog terms, and
  `owe_reserves`' growth with the backlog (`kv_drain_under_pressure.md` §2a,
  §2c). What remains of the reserve is the gap lease (§3.2).
- `MigrateFlight` as a quantize lock (§5).
- `batch_test`'s boundary calls to `quantize_and_seal_sequences`. The test runs
  the scheduler's gap quantize inside its timed loop. The function's selection
  and convert core becomes the gap quantize; there is one implementation.

## 7. How it is verified

Unit tests, written with the code, asserting raw expected bytes:
- the descriptor table built from host bookkeeping for a known wave: complete
  blocks in, writer chunks out, a sealed residence's partial block in;
- the device claim against a lease: the slot each block takes, a dry class
  deferring exactly the blocks past its lease, and the summary's contents;
- the convert's quantized bytes and its block-table rewrites, against the
  per-block reference quantizer;
- the host catch-up: used gids marked live, unused ones returned, records
  re-pointed, natives freed only after the last holder moves;
- persistence: a copy issued before its gap's event has completed reads only
  completed blocks, and `has_warm` is never published before the copy's event.

Gates, run one model per `cargo` process with the daemon stopped:
- every model's fidelity ladder and the sweep, with quantization inside the
  timed windows;
- in production, a decode with the persistence thread active shows forward time
  of at least 95% of each wave window;
- native K/V in VRAM never exceeds the writer chunks plus one gap's deferrals;
- p95 commit wait under 1 s.

The before figures are those in §1.

## 8. Order of work

All of this lands in one change, with no dual path:
1. the descriptor-table builder;
2. the lease and the device claim;
3. the convert's table rewrite;
4. the summary and the host catch-up;
5. the gap call in the scheduler and in `batch_test`;
6. persistence reduced to copy-and-append on its own stream into the pinned
   warm pool;
7. the deletions in §6;
8. `kv_drain_under_pressure.md` and `turn_claims.rs`'s commit-wait comment
   rewritten to match.

The gates in §7 decide whether it is done.
