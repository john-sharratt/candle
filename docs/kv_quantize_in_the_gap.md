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
 gap   ─▶ select + claim + convert + record fill   (one batched launch set,
   │       primary stream), then the host's own header upload behind it
   │       event E_gap recorded; completion word written last
   │
   ├────▶ next forward reads each block through its new record
   │       (stream order: no fence)
   │
   ├────▶ host catch-up, once the completion word is set: gids re-pointed,
   │       native blocks freed
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

**A block is complete only in committed tokens.** The table is built from each
sequence's committed offset, after the step's speculative rollback has settled
(`truncate_sequence_to_tokens`). Rows a verify forward wrote past the committed
position are in no table, so a rollback never has to trim a quantized block,
which cannot take writes.

A sealed residence's partial final block is quantized as a full block. Its dead
slots are zero and the valid range is passed to the selection kernel, as
`kv_tier_migration.md` §3 already specifies. A live sequence's writer chunk is
not in the table.

**Each row also names a record slot.** A block's `KvHead[n_kv_head]` record
(`chunked/meta_pool.rs`: palette maps, formats, scales and band pointers) is
what the attention kernels read the block through. For every row the host
claims one record slot (`MetaGid`) from the record class before the gap's
launches. That claim is exact, one per block, so the host knows each new
record's device address before the device has chosen anything. A block the
record class cannot serve is left out of the table for this gap.

### 3.2 Device-side band claim

Today the format selection runs on the device and is then read back, so the host
can allocate each band's destination in the class its chosen format needs. On
the scheduler thread that would be a synchronous readback in the loop, against
invariant 3. Instead, **the host leases slots ahead of time, and the device
chooses among them.**

- **Leases.** In each gap the host claims, for every reachable quant class, a
  lease of free slots from that class's arenas (`ChunkGidPool`, lowest place
  first). It holds their `ChunkGid`s and uploads the lease as a device-resident
  **slot stack** per class: one `{arena ptr, slot}` entry per slot, plus a
  device cursor.
- **Claim.** The fused selection kernel picks each (block, head)'s K and V
  palettes and formats exactly as today. For each band the record will name
  (up to `N_PALETTE` a head, for K and for V), it then takes a destination by
  `atomicAdd` on that band's class cursor. Nothing comes back to the host.
- **A class that runs dry defers the block, never fails it.** If any of a
  block's claims passes its lease, the whole block stays native and goes into
  the next gap's table; the slots its other bands took are reported unused. The
  attempt is counted in the summary (§3.4). A block that stays native for a gap
  is still attended correctly, only larger, and this is the only way a complete
  block misses its gap.
- **Lease size is bounded by this gap's own demand.** The host built the table,
  so it knows the band count exactly: blocks × heads × `N_PALETTE` × 2. Only
  the split across classes is unknown. Each class's lease is that count times
  the class's recent peak share of claims (read from the summaries, kept
  separately for prefill and decode gaps), plus one arena's headroom, and never
  more than the count itself. The share is a fraction, not a volume, so a
  prefill arriving after an idle period is leased for its own size. Before any
  summary exists the count is split evenly across the reachable classes.
  This replaces §2a's backlog-sized reserve: the lease covers one gap's demand,
  not the whole backlog. The lease is KV ground and is bought below
  `weight_floor` (invariant 7). When the ground cannot cover a lease, the class
  defers blocks for a gap rather than take ground from the experts.

### 3.3 Convert, record and swap

- **Convert.** One batched launch over every claimed band, across every layer
  and sequence (invariant 5). Destinations are allocated uninitialised
  (invariant 6).
- **Record fill.** The convert's epilogue writes each block's whole record into
  the record slot its row names: palette maps, format tags, scales, and the
  claimed slots' band pointers. This is `backing::fill_records_on_device`'s job
  with the claim as its source, held byte-for-byte against
  `serialize_kv_heads`. **A deferred block's record is filled too**, naming its
  native bands with their float format. Every record slot in the table
  therefore holds a valid record once the gap's launches have run, whichever
  way the claim went.
- **The host swaps the header, in the same gap.** A sequence's slot-state
  buffer (`chunked/gpu_chunks.rs`) holds one 16-byte header per block, whose
  `kvheads_ptr` names the block's record. The host rewrites that pointer to the
  new record slot in its own copy of the buffer and uploads the header through
  the ordinary staged path, queued behind the gap's launches on the same
  stream. The next forward on that stream therefore reads the block through the
  new record, with stream order as the fence. The device never writes a slot
  buffer, so **the host's copy stays the authoritative one**: a later
  `rebuild_decode`, `extend_decode` or `snapshot_into_generation` serialises
  the swapped pointer, and a resident record is not rebased by a snapshot.
- **Borrowers.** A block borrowed by several slots (a shared prefix) shares one
  record slot, as sealed chunks do today. The host swaps the header in every
  slot buffer that names the block.
- **What still names the native bands.** After the swap, no header reaches the
  native bands except through a deferred block's record. The host's chunk
  records (`ChunkWindow`, `SealedChunk`) still hold the native `HeadGids`
  until the catch-up (§3.5), and so does any `ChunkPin` a launch or a
  generation snapshot took before the gap.

### 3.4 The summary

The kernels write a per-gap summary into a mapped pinned ring, as the MoE
dispatch does with its routing summary (`docs/moe_live_dispatch_design.md`).
One gap's summary is:
- a header: the gap's sequence number and its row count;
- one row per block: converted or deferred, and the block's record with each
  band pointer replaced by the class and lease index it claimed. That carries
  the palette maps, format tags and scales the host needs for its own records
  and for the redo log;
- each class's final cursor;
- a **completion word**, written by the gap's last launch, holding the gap's
  sequence number.

The host reads a summary only once its completion word matches. It looks at the
head of every gap and at the end of every scheduler step, by reading the mapped
word. No host CUDA call reads or waits on the ring.

The ring is sized at start for two gaps of the largest table a gap can build:
the widest forward's blocks across every layer, at one row each. When the unread
part of the ring leaves no room for a gap's whole table, the table is cut to
what fits and the remaining blocks stay native for a gap, as a deferral does. A
summary the host has not read keeps its leases held.

### 3.5 The host catch-up and the native bound

When the host reads a summary it:
- marks the claimed band gids live and returns the unused ones to their pools;
- re-points every holder of each converted block (`ChunkWindow`, `SealedChunk`)
  from its native `HeadGids` to the claimed ones, with the formats and scales
  from the row;
- puts each deferred block into the next gap's table, where it is given a new
  record slot;
- updates each class's share for the next leases.

**A native band is freed by its last reference.** The catch-up drops the
host records' references; a `ChunkPin` taken before the gap drops its own when
its launch or generation is done. By then the convert that read the band has
finished, because the completion word is set, so the slot is free for any
stream to reuse and no fence is needed. A reprojection built between the gap
and the catch-up clones a chunk whose header already names the new record. It
attends the quantized block and merely holds the native gids a little longer.

**The bound.** Native K/V standing in VRAM is at most:
- every live sequence's writer chunk;
- the blocks the latest forward completed, until their gap;
- the blocks of gaps whose summary the host has not yet read, normally one gap;
- the deferred blocks.

In steady decode that is the writer chunks plus at most two blocks per
sequence-layer. In a chained prefill it is the output of the two most recent
forwards. A forward is admitted against ground that counts the previous gap's
natives as still held.

### 3.6 Cost in the gap

The gap now does work proportional to the K/V produced since the last gap. That
is one selection, one claim, one convert and one record fill, all batched. A
6,144-row prefill forward produces 192 blocks per layer. Decode produces about
one block per sequence-layer every 32 steps. The gap is not budgeted: everything
complete is quantized. Its cost is measured, inside the timed windows (§7), and
the rate model's decode step includes it.

The cost is real device work that now sits between forwards, so it is measured
before anything is built (§8 step 1). `quantize_and_seal_sequences` runs the
same selection and convert core today. Its device time per MiB of native K/V,
at the rate production produces K/V in decode, decides whether §7's 95% gate
can be met by the kernels as they are. If it cannot, the kernels are made faster
first. The gate is not loosened.

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
  stream. It copies a residence once the host catch-up has re-pointed the
  residence's last block, so it reads the gids, formats and scales from the
  host's records. It makes its stream wait on the `E_gap` of the gap that
  quantized that block (`cudaStreamWaitEvent`), then issues one DtoH of the
  gathered quantized blocks. No `device.synchronize()` anywhere on this path.
- **Pinned warm tier.** The DtoH lands in pinned host buffers that are both the
  redo-log write buffer and the warm cache (`kv_tier_migration.md` §3). This
  replaces today's pageable CPU arenas, so the copy runs at full link bandwidth.
  On WDDM, pinned memory is charged to system commit (`docs/driver_model.md`),
  so the pool's size is read from the machine's RAM and commit headroom, never
  fixed. The three dev machines have 32, 64 and 189 GB.
- **A full pool.** When the pool has no free buffer, it drops its least recently
  used blocks that the log already holds. Those blocks are then cold and are
  read back from the log when wanted. A block the log does not yet hold is
  never dropped. If every buffer holds such a block, the copy waits for the
  log's append to release one. That wait is the persistence thread's alone: the
  blocks stay hot and quantized, and the gap never waits on the pool.
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
- The selection's host readback of palettes, scales and maps
  (`sampled_selection/gpu.rs`). The summary row carries them (§3.4).
- `batch_test`'s boundary calls to `quantize_and_seal_sequences`. The test runs
  the scheduler's gap quantize inside its timed loop. The function's selection
  and convert core becomes the gap quantize; there is one implementation.

## 7. How it is verified

Unit tests, written with the code, asserting raw expected bytes:
- the descriptor table built from host bookkeeping for a known wave: complete
  blocks in, writer chunks out, a sealed residence's partial block in, and rows
  past the committed offset out;
- the lease for a known table: never more than the table's band count, split by
  the recorded shares, and split evenly when there are none;
- the device claim against a lease: the slot each band takes, a dry class
  deferring exactly the blocks past its lease and reporting their other slots
  unused;
- the convert's quantized bytes and the record it fills, against the per-block
  reference quantizer and `serialize_kv_heads`; a deferred block's record
  naming its native bands;
- the header swap: the slot buffer's host copy and its uploaded bytes name the
  new record, and a rebuild and a generation snapshot taken afterwards keep it;
- the summary: its rows, cursors and completion word, a summary whose word is
  not yet set being left unread, and a table cut to the ring's free room;
- the host catch-up: used gids marked live, unused ones returned, records
  re-pointed with the row's formats and scales, and a native band freed only
  when its last reference drops;
- persistence: a copy issued before its gap's event has completed reads only
  completed blocks, `has_warm` is never published before the copy's event, and
  a full pool drops only blocks the log holds.

Gates, run one model per `cargo` process with the daemon stopped:
- every model's fidelity ladder and the sweep, with quantization inside the
  timed windows;
- in production, a decode with the persistence thread active shows forward time
  of at least 95% of each wave window;
- native K/V in VRAM never exceeds §3.5's bound: the writer chunks, the two
  most recent forwards' blocks, and the deferred blocks;
- p95 commit wait under 1 s.

The before figures are those in §1.

## 8. Order of work

The measurement comes first and changes no behaviour:
1. time the selection and convert core in `batch_test`, per MiB of native K/V,
   and set it against the decode K/V rate (§3.6).

Everything after it lands in one change, with no dual path:
2. the descriptor-table builder, with the record-slot claim;
3. the lease and the device claim;
4. the convert's record fill and the host's header swap;
5. the summary and the host catch-up;
6. the gap call in the scheduler and in `batch_test`;
7. persistence reduced to copy-and-append on its own stream into the pinned
   warm pool;
8. the deletions in §6;
9. `kv_drain_under_pressure.md` and `turn_claims.rs`'s commit-wait comment
   rewritten to match.

The gates in §7 decide whether it is done.
