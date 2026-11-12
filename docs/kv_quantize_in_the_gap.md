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
| A native backlog stands in VRAM at about 4× its final size | Only a sealed residence is quantized, and only when the drain reaches it, oldest-first across every conversation and ingest reading on the member. A live sequence's K/V is never eligible: it stays native for as long as the sequence lives. | `pending_mib` 1.4–2.2 GB for the whole turn |
| The KV side takes ground from the expert zone | The quant reserve is sized from that backlog (`pending × factor`, capped at 25% of the KV regions) | `cap_mib=3111`, `kv_mib` up to 12,448; "grow class2048 by 24 arenas, reason=backlog" |
| Durability lags by minutes | A turn's commit waits for its turn in the drain's queue | "a finished turn's commit landed … waited_s=319" |

### 1.1 The same on WhiteWorld, with the NPCs running

Measured on WhiteWorld (24 GB, WDDM), 2026-11-12 19:45–19:59 UTC, build
4915-1e749473, with npcd's characters live and zend ingesting. The card had
1.96 GB of driver headroom.

| What | Measured |
|---|---|
| Native K/V standing | `pending_mib` 590–776 for the whole window, never lower. The two native-sized classes held about 677 MB live against about 538 MB in every quantized class together. |
| What the drain moved | 0 or 1 residence and 0–3 MiB a pass. The backlog did not fall. |
| How long a pass took | 1.0–5.1 s for passes that moved nothing; 181 passes over 1 s in three hours. |
| The reserve | 227–840 MB held free against that backlog, `cap_mib` 2,055–2,803, arenas grown and released every few seconds. |
| Expert ground | The weight zone conceded from 3,746 to 1,909 MiB in two minutes. Five decode forwards were then discarded on a cold expert not staged within 1,000 ms, and the learned decode step rose from 244 to 1,345 ms. |
| Work lost | About 25 ingest readings failed with "the turn made no progress for 60 s". |

The backlog here is mostly K/V the drain cannot reach. `pending` counts every
live native slot (`chunked/alloc.rs`, `owe_reserves`), and the characters hold
long-lived live sequences. A faster drain would not move it. The reserve is
then sized against a backlog that never drains, so it stands free indefinitely.

**What this design does not fix.** K/V was not the largest tenant on the KV
side of that card: recurrent state held 2.79 GB and the QSA index cache
1.03 GB, against about 1.2 GB of live K/V, and a promote's relief took 464 MiB
from the experts on its own. This design returns the native backlog's excess
and the standing reserve, roughly 0.7–1.3 GB there. The other tenants are
outside it.

### 1.2 The test does not see it

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

**The device converts; the host installs when it knows.** A gap launches the
selection, the claim and the convert and reads nothing back. The host learns
what was chosen from mapped memory, in a later gap, and only then points its
records and the block tables at the quantized blocks. Until then everything
still names the native blocks, which are still valid.

```
 forward ─▶ native writer chunk + newly complete native blocks
   │
 gap N ─▶ select + claim + convert   (one batched launch set, primary stream)
   │       summary and completion word written to mapped memory
   │       event E_gap recorded
   │
 forward ─▶ still reads those blocks native   (nothing has been re-pointed)
   │
 gap N+1 ─▶ catch-up: the completion word is set, so the host installs gap N's
   │         blocks (records, block tables) and drops the native ones
   │       then its own select + claim + convert
   │
   └────▶ persistence: wait on E_gap on the copy stream ─▶ DtoH into the pinned
          warm buffer ─▶ log append ─▶ commit ─▶ has_warm published
```

**One gap, however many tenants.** A process has one `EngineCore`, so one
scheduler thread, one primary stream and one set of KV pools, shared by every
attached substrate (`docs/shared_model_host.md` §3.2). The gap is that
scheduler's. Its one table holds every tenant's blocks, and its one summary
serves them all. Each attachment keeps its own persistence thread, which only
copies (§5).

The two products then match: the batched test and production run the **same**
gap quantize, inside the timed windows. The test's two boundary calls are
deleted.

## 3. The gap quantize

A gap does three things, in this order:

1. **Catch-up** (§3.5), if a gap is outstanding and its completion word is set.
2. The arena creation and the compaction the gap already does
   (`scheduler/run.rs`).
3. **Quantize** (§3.1–§3.4), if no gap is outstanding and there are blocks to
   take.

**At most one gap is outstanding.** A gap whose predecessor's completion word is
not yet set quantizes nothing; its blocks wait for the next gap. The summary is
therefore one buffer, a lease is never held by two gaps, and the host never
reasons about two unread results.

### 3.1 Inputs, without a readback

The host knows, without asking the device, every block that became complete in
the forwards since the last gap. It scheduled those rows, so it knows each
sequence's offset, and it allocated the native chunks those blocks live in. From
its own bookkeeping it builds a **descriptor table**: one row per block, giving
the source K/V pointers, the (layer, head) layout, the block's token range, its
sequence's position (for attention-sink protection on positions 0–3) and its
layer's thresholds. That is invariant 2b's shape. No block is copied together to
be quantized.

- **The builder is proportional to what changed.** Each sequence keeps a cursor:
  the first block not yet handed to a gap. The builder visits the sequences a
  forward touched and takes the blocks from the cursor to the writer chunk. It
  never walks a sequence's history. An empty table launches nothing.
- **A block is complete only in committed tokens.** The table is built from each
  sequence's committed offset, after the step's speculative rollback has settled
  (`truncate_sequence_to_tokens`). Rows a verify forward wrote past the
  committed position are in no table, so a rollback never has to trim a
  quantized block, which cannot take writes.
- **A block is taken only when every layer has it.** A wave that died mid-sweep
  leaves one layer a filled chunk ahead of the others. The row for a block
  exists only when the block is complete on every layer, which is the rule
  `assert_sealed_layers_aligned` enforces at a seal today.
- **Thresholds ride in the row.** A layer can seal at its own level (the MTP
  draft head, `layer_seal_cap`). Today that makes one selection run per policy
  group. The row carries its thresholds, so the gap is one launch whatever the
  mix.
- **A sealed residence's partial final block** is quantized as a full block. Its
  dead slots are zero and the valid range is passed to the selection kernel, as
  `kv_tier_migration.md` §3 already specifies. A live sequence's writer chunk is
  not in the table.
- **A table is cut to what the gap's scratch can carry.** The selection runs in
  held scratch (`SELECTION_BUDGET_BYTES`). Blocks past the cut wait for the next
  gap.

### 3.2 Device-side band claim

Today the format selection runs on the device and is then read back, so the host
can allocate each band's destination in the class its chosen format needs. On
the scheduler thread that would be a synchronous readback in the loop, against
invariant 3. Instead, **the host leases slots ahead of time, and the device
chooses among them.**

- **Leases.** For every reachable quant class the host keeps a standing lease:
  free slots claimed from that class's arenas (`ChunkGidPool`, lowest place
  first), held as `ChunkGid`s, and mirrored on the device as a **slot stack**
  of `{arena ptr, slot}` entries. In each gap the host tops the stack up and
  uploads only the entries it added. Slots a gap does not take stay in the
  stack for the next one.
- **Claim, in table order.** The fused selection kernel picks each (block,
  head)'s K and V palettes and formats exactly as today and writes each band's
  class. A claim kernel then gives every band its index in its class: the count
  of bands of that class before it in the table. The band's destination is that
  entry of the class's stack. The claim is a prefix count, not a race on an
  atomic cursor, so the same table against the same lease always takes the same
  slots, and the lowest places fill first.
- **A class that runs dry defers the block, never fails it.** A block is
  deferred when any of its bands' indices falls past its class's lease. The
  whole block stays native and goes into the next gap's table. The slots its
  other bands would have taken are not written and stay leased. A block that
  stays native for a gap is still attended correctly, only larger, and this is
  the only way a complete block misses its gap.
- **A band that stays as it is needs no slot.** Where the selection keeps a band
  in its native format (C0's K), nothing is claimed or converted for it.
- **Lease size is bounded by this gap's own demand.** The host built the table,
  so it knows the band count exactly: blocks × heads × `N_PALETTE` × 2. Only
  the split across classes is unknown. Each class's lease is topped up to that
  count times the class's recent peak share of claims (read from the summaries,
  kept separately for prefill and decode gaps), plus one arena's headroom, and
  never more than the count itself. The share is a fraction, not a volume, so a
  prefill arriving after an idle period is leased for its own size. Before any
  summary exists the count is split evenly across the reachable classes.
  This replaces §2a's backlog-sized reserve: the lease covers one gap's demand,
  not the whole backlog. The lease is KV ground and is bought below
  `weight_floor` (invariant 7). When the ground cannot cover a lease, the class
  defers blocks for a gap rather than take ground from the experts.

### 3.3 Convert

One batched launch over every claimed band, across every layer and sequence
(invariant 5), reading its destinations from the claim. Destinations are
allocated uninitialised (invariant 6). The convert is launched once for the
whole table, never once per layer.

Nothing else is written on the device. No record is filled and no block table is
touched in the gap: both are the host's to change, at the catch-up.

### 3.4 The summary

The kernels write the gap's summary into one mapped pinned buffer, as the MoE
dispatch writes its routing summary (`expert_lre/dispatch.rs`, `SummaryRing`).
It holds:
- a header: the gap's sequence number and its row count;
- one row per block: converted or deferred, and for each head the palette tags,
  scales and palette map the selection chose, and each band's lease index;
- a **completion word**, written by the gap's last launch, holding the gap's
  sequence number.

That is what the selection reads back to the host today, delivered through
mapped memory. The host reads it only once the completion word matches, by
reading the mapped word at the head of a gap. No host CUDA call reads or waits
on the buffer. The buffer is sized at start for the largest table a gap can
build (§3.1's cut).

### 3.5 The host catch-up and the native bound

When the completion word is set, the head of the next gap installs the
outstanding gap's blocks. For each converted block the host:
- takes the claimed slots out of the lease as the block's bands, with the
  formats, scales and maps from the row;
- claims a record slot and fills it with the existing device fill
  (`fill_records_on_device`);
- replaces the block in its owner, in place: the live sequence's chunk at that
  index, or the sealed residence's chunk. No sequence is truncated and
  re-injected;
- re-serialises that block's header in the owner's slot buffer
  (`update_chunk`), uploaded with the gap's other slot uploads in one scatter;
- drops the owner's reference to the native bands.

A deferred block is put into this gap's table. Each class's share is updated for
the next leases. This is the install the persistence thread does today, moved to
the thread that owns the sequences, with nothing left to fence: the convert has
finished, and the header upload is queued on the stream ahead of the next
forward.

**A native band is freed by its last reference.** The owner's reference goes at
the catch-up. A `ChunkPin` a launch or a generation snapshot took earlier drops
its own when it is done. The convert that read the band has finished by then,
so the slot is free for any stream to reuse and no fence is needed.

**A clone made while its gap is outstanding stays native.** A seal or a
reprojection that clones a block between its gap and its catch-up holds the
native bands, and is not an owner the catch-up re-points. It reads correctly and
is merely larger until it is dropped. A sequence is not sealed while it has
blocks in an outstanding gap; the seal runs after that gap's catch-up, one step
later. That leaves only a reprojection of a sealed residence's partial block,
one block a residence at most.

**The bound.** Native K/V standing in VRAM is at most:
- every live sequence's writer chunk;
- the blocks the latest forward completed, until their gap;
- the blocks of the one outstanding gap;
- the deferred blocks, and the clones above.

In steady decode that is the writer chunks plus at most two blocks per
sequence-layer. In a chained prefill it is the output of the two most recent
forwards. A forward is admitted against ground that counts the outstanding
gap's natives as still held.

### 3.6 Cost in the gap

The gap now does work proportional to the K/V produced since the last gap: one
selection, one claim and one convert, all batched, and at the next gap one
record fill and one scatter of headers. A 6,144-row prefill forward produces
192 blocks per layer. Decode produces about one block per sequence-layer every
32 steps, so most decode gaps have an empty table and launch nothing. The gap
is not budgeted: everything complete is quantized. Its cost is measured, inside
the timed windows (§7), and the rate model's decode step includes it.

The cost is real device work that now sits between forwards, so it is measured
before anything is built (§8 step 1). `quantize_and_seal_sequences` runs the
same selection and convert core today. Its device time per MiB of native K/V,
at the rate production produces K/V in decode, decides whether §7's 95% gate
can be met by the kernels as they are. If it cannot, the kernels are made faster
first. The gate is not loosened.

### 3.7 What the gap logs

The scheduler writes one `kv_gap` line for each second in which a gap ran,
summed over that second's gaps: blocks converted, blocks deferred and the class
that ran dry, gaps that quantized nothing because one was outstanding, native
bytes standing, lease bytes held, and the gaps' device time. It replaces the
`kv_reserve: pass` line, which each attachment's persistence thread writes
today about the one shared pool, so that every pass is logged once per tenant.
§7's production gates are read from this line.

## 4. What a forward attends

Every forward attends a quantized prefix plus, in native form, each sequence's
writer chunk and the blocks of the outstanding gap. A long prefill's later
forwards attend that same prefill's earlier blocks quantized. Decode attends its
own earlier tokens of the turn quantized from the step after each block's gap.

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

- **A dedicated copy stream for each persistence thread.** Every attachment's
  persistence thread owns a non-primary stream of its own, so two tenants'
  copies never queue behind each other or behind a forward. Today both
  threads run their hot→warm pass on the shared primary stream and each fences
  it. A thread copies a residence once the catch-up has installed the
  residence's last block, so it reads the gids, formats and scales from the
  host's records. It makes its stream wait on the `E_gap` of the gap that
  quantized that block (`cudaStreamWaitEvent`), then gathers and copies the
  quantized blocks with the existing batch copy
  (`migrate_sealed_to_cpu_batch_async`). No `device.synchronize()` anywhere on
  this path.
- **Pinned warm tier.** The DtoH lands in pinned host buffers that are both the
  redo-log write buffer and the warm cache (`kv_tier_migration.md` §3). This
  replaces today's pageable CPU arenas, so the copy runs at full link bandwidth.
  On WDDM, pinned memory is charged to system commit (`docs/driver_model.md`),
  so the pool's size is read from the machine's RAM and commit headroom, never
  fixed. The three dev machines have 32, 64 and 189 GB. The pool is the
  machine's and is one budget: each attachment's thread holds its share of it
  through the `WarmLedger`, as it does of the warm tier today.
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
  §2c). What remains of the reserve is the gap lease (§3.2). The
  `kv_reserve: pass` line and `persistence/reserve_log.rs` go with it (§3.7).
- `MigrateFlight` as a quantize lock (§5).
- The selection's host readback of palettes, scales and maps
  (`sampled_selection/gpu.rs`). The summary row carries them (§3.4).
- `batch_test`'s boundary calls to `quantize_and_seal_sequences`, and the
  function's seal by truncate and re-inject. The test runs the scheduler's gap
  quantize inside its timed loop. The function's selection and convert core
  becomes the gap quantize; there is one implementation.

## 7. How it is verified

Unit tests, written with the code, asserting raw expected bytes:
- the descriptor table built from host bookkeeping for a known wave: complete
  blocks in, writer chunks out, a sealed residence's partial block in, rows
  past the committed offset out, a block one layer lacks out, and nothing
  before a sequence's cursor;
- the lease for a known table: never more than the table's band count, split by
  the recorded shares, split evenly when there are none, and topped up by only
  the entries it lacks;
- the device claim against a lease: the slot each band takes, the same on every
  run; a dry class deferring exactly the blocks past its lease and writing none
  of their other bands; a band kept native taking no slot;
- the convert's quantized bytes, against the per-block reference quantizer;
- the summary: its rows and completion word, and a summary whose word is not
  yet set being left unread;
- a gap that finds one outstanding quantizes nothing and keeps its blocks;
- the host catch-up: the block replaced in place at its index, its record equal
  to `serialize_kv_heads`, its header re-serialised, a rebuild and a generation
  snapshot taken afterwards naming the quantized block, and a native band freed
  only when its last reference drops;
- persistence: a copy issued before its gap's event has completed reads only
  completed blocks, `has_warm` is never published before the copy's event, and
  a full pool drops only blocks the log holds.

Gates, run one model per `cargo` process with the daemon stopped:
- every model's fidelity ladder and the sweep, with quantization inside the
  timed windows;
- in production, a decode with the persistence thread active shows forward time
  of at least 95% of each wave window;
- native K/V in VRAM never exceeds §3.5's bound: the writer chunks, the two
  most recent forwards' blocks, the deferred blocks and the clones;
- gaps that quantize nothing because one is outstanding are under 1% of the
  gaps with a table;
- p95 commit wait under 1 s.

The before figures are those in §1 and §1.1. The production gates are run on
BlackWorld and on WhiteWorld, the second with the characters live.

## 8. Order of work

The measurement comes first and changes no behaviour:
1. time the selection and convert core in `batch_test`, per MiB of native K/V,
   and set it against the decode K/V rate (§3.6).

Everything after it lands in one change, with no dual path:
2. the descriptor-table builder and the per-sequence cursor;
3. the lease and the device claim;
4. the convert reading its destinations from the claim;
5. the summary and the host catch-up;
6. the gap call in the scheduler and in `batch_test`;
7. persistence reduced to copy-and-append on its own stream into the pinned
   warm pool;
8. the deletions in §6;
9. `kv_drain_under_pressure.md`, `AGENTS.md`'s tiering note and
   `turn_claims.rs`'s commit-wait comment rewritten to match.

The gates in §7 decide whether it is done. §9 gives each step its code.

## 9. Tasks, with the code they touch

Line numbers are as of commit `dc04b62ac`.

**1. Measure the core (§8 step 1).**
- `candle-transformers/src/models/batch_test/utils.rs:2056` and `:2195`: the two
  boundary calls. Time them, and report device time per MiB of native K/V.
- `candle-transformers/src/models/batched_inference.rs:2052`,
  `quantize_and_seal_sequences`. Its convert is issued inside a loop over every
  backing with the whole descriptor list each time, where
  `convert_deferred_descs` says any one backing serves. Confirm whether that
  repeats the convert once per layer before trusting the timing, and time a
  single convert.

**2. The descriptor table and the cursor (§3.1).**
- New `candle-nn/src/kv_cache/chunked/gap_table.rs`: the row type and the
  builder. One concern, tested alone.
- `candle-nn/src/kv_cache/chunked/sequence_ops.rs`: the per-sequence cursor
  beside `writer_start_idx`; `truncate_sequence_to_tokens` (`:2134`) keeps the
  cursor at or below the committed block.
- `candle-nn/src/kv_cache/chunked/types.rs:216`: `ChunkWindow` (`gids`, `meta`),
  the source of a row's pointers.
- `batched_inference.rs:2172`, `assert_sealed_layers_aligned`: the every-layer
  rule the builder applies per block.
- `batched_inference.rs:645`, `layer_seal_cap`: the per-layer thresholds a row
  carries.
- `candle-nn/src/kv_cache/chunked/compress.rs:69`, `SELECTION_BUDGET_BYTES`: the
  cut.

**3. The lease (§3.2).**
- New `candle-nn/src/kv_cache/chunked/gap_lease.rs`: the standing per-class
  lease, its device stack, the top-up and the share.
- `candle-nn/src/kv_cache/chunked/gid_pool.rs:1428`, `ChunkGidPool`: claiming
  free slots lowest place first; `set_reserve_target` (`:1465`) goes.
- `candle-nn/src/kv_cache/chunked/alloc.rs:509`, `owe_reserves`, and
  `reserve_target.rs:45`, `reserve_targets`: replaced by the lease's sizing.
- `alloc.rs:664`, `create_deferred_arenas`: still makes the arenas a lease is
  short of, in the gap.

**4. The kernels (§3.2, §3.3).**
- `candle-kernels/src/quantize/select_kv_format.cuh:2285` and `:3199`,
  `select_kv_format_palette4_paged`: thresholds per row; each band's class
  written for the claim; outputs written to the mapped summary.
- New `candle-kernels/src/quantize/kv_gap_claim.cu`: the per-class prefix count
  and the deferral rule. Registered in `candle-kernels/build.rs`, bound in
  `candle-kernels/src/lib.rs`.
- `candle-nn/src/kv_cache/chunked/compress.rs:1413`, `convert_deferred_descs`
  and `convert_descs_batched`: destinations read from the claim, one launch for
  the table.
- `candle-nn/src/kv_cache/chunked/sampled_selection/gpu.rs:741`, `run_palette4`,
  and the readback at `:1497–1538`: removed.

**5. The summary and the catch-up (§3.4, §3.5).**
- New `candle-nn/src/kv_cache/chunked/gap_summary.rs`: the mapped buffer, its
  row layout and the completion word. Pattern:
  `candle-transformers/src/models/expert_lre/dispatch.rs:156`, `SummaryRing`.
- New `candle-nn/src/kv_cache/chunked/gap_install.rs`: the catch-up.
- `candle-nn/src/kv_cache/chunked/backing.rs:1307`, `fill_records_on_device`,
  and `meta_pool.rs:146`, `serialize_kv_heads`: the record, unchanged.
- `candle-nn/src/kv_cache/chunked/gpu_chunks.rs:692`, `update_chunk`, and
  `types.rs:1458` / `:1507`: the header re-serialisation and its upload.

**6. The gap call (§3, §3.7).**
- `candle-conversation/src/scheduler/run.rs:1363`: the gap. Catch-up before
  `create_deferred_arenas`; the quantize after `compact_kv_if_fragmented`
  (`scheduler/prefill.rs:1249`).
- `candle-conversation/src/scheduler/decode.rs` (the rollback, around `:860`
  and `:1051`): the table is built after it.
- `batched_inference.rs:2361`: the session's gap entry beside
  `create_deferred_arenas`.
- The `kv_gap` line, in a file of its own under `candle-conversation/src/scheduler/`.
- `batch_test/utils.rs:2056`, `:2195`, `batch_test/fidelity.rs:147` and
  `quantized_qwen38_moe.rs:1867`: every caller of `quantize_and_seal_sequences`
  moves to the gap call.

**7. Persistence (§5).**
- `candle-conversation/src/persistence/thread.rs`: `run_pass` (`:1036`) keeps
  the copy and the append. Deleted: `await_owed_arenas` (`:694`),
  `install_hot_to_warm` (`:710`), `migrate_group_hot_to_warm` (`:771`), the
  syncs at `:1100`, `:1334` and `:1414`, `MAX_GAP_RETRIES` (`:1253`).
- `candle-nn/src/kv_cache/chunked/chunk_ops.rs:1297`,
  `migrate_sealed_to_cpu_batch_async`: the copy, on the thread's own stream,
  into the pool.
- New `candle-conversation/src/persistence/warm_pool.rs`: the pinned pool, its
  sizing and its full-pool rule. `persistence/warm_ledger.rs:27` gives each
  thread its share.
- `candle-nn/src/kv_cache/chunked/migrate_flight.rs:131` (`MigrateFlight`,
  deleted) and `:212` (`HeldChunks`, kept).
- `candle-conversation/src/persistence/group_cap.rs` and `reserve_log.rs`:
  deleted.

**8. Documents and comments (§8 step 9).**
- `docs/kv_drain_under_pressure.md` §2a–§2c and §3.
- `AGENTS.md`, the "Tiering status" note under the KV cache subsystem, which
  describes the hot→warm quantize and the pageable warm arenas.
- `zend/src/turn_claims.rs`: the commit-wait comment.

## 10. Things to be careful of

**Addresses on the device.**
- A slot stack holds raw arena addresses. Any move of the ground under it
  invalidates them: a compaction, a vacate (`vacate_epoch`), a frontier or
  `weight_floor` move (invariant 7). The stack is rebuilt after any of them,
  and never inferred to be unchanged from the geometry.
- While a gap is outstanding the host does not know which leased slots it
  wrote. Compaction and the empty sweep treat the whole lease as held until the
  catch-up, and never release an arena a lease stands in.
- A record names its bands by raw address, so it must hold their gids
  (`meta_pool.rs`). A record filled at the catch-up holds the claimed gids, not
  the native ones.
- A `ChunkPin` holds both the bands and the record (`gpu_chunks.rs`). Holding
  only the first has already cost a night.

**The sequence's layout.**
- The catch-up replaces a chunk in place. It does not truncate and re-inject:
  re-injecting shared chunks under a lone sequence has corrupted the decode
  read before (`batched_inference.rs`, the float-mode comment in
  `quantize_and_seal_sequences`).
- The empty writer chunk and any pad after it are not blocks
  (`drop_empty_tail`). A sealed chunk with room left in it can end up where
  writes land.
- A sliding ring drops front tokens (`base_pos`). Block indices, the cursor and
  positions 0–3 are in absolute positions, not resident ones.
- A block whose geometry the kernels do not quantize, and a float mode with no
  policy, give the gap an empty table. Neither is an error.

**Timing and order.**
- The catch-up runs before anything in the step that seals, snapshots or
  reprojects, so no new clone is made of a block it is about to replace.
- Check that a prefill-only step waits for its forward before the next gap. If
  it does not, the completion word is unset at that gap, the gap quantizes
  nothing, and natives build up across forwards. §7 gates this at 1%; the
  `kv_gap` line counts it.
- The header upload at the catch-up goes through the slot-buffer staging path,
  which never reads the live host copy and is aware of recorded waves
  (`gpu_chunks.rs`). No other path writes a slot buffer.
- A copy stream's wait on `E_gap` needs an event that carries no timing.
  cudarc's `new_event(None)` already makes one. Events come from a small pool,
  one per outstanding gap and one per copy in flight.
- The persistence thread is its substrate's one owner thread. Waiting on a full
  pool stops its other duties for that substrate, so the wait is bounded by the
  log append it is waiting for and nothing else.

**Memory on the small cards.**
- The selection's scratch is held ground. WhiteWorld declined 27 scratch takes
  in three hours, the largest 113 MB. The table's cut is sized from the scratch
  actually held, not from a constant.
- The persistence copy gathers into device staging before the DtoH
  (`chunk_ops.rs`). That staging is transient VRAM and stays small.
- The lease never buys ground from the experts. Deferral is the release valve,
  and a run of deferrals is a finding to log, not a reason to raise the lease.
- The pinned pool is charged to system commit under WDDM. WhiteWorld already
  commits 119 of 206 GB.

**Testing and measuring.**
- An instrument that fences hides exactly what this change is meant to remove.
  No `synchronize` is added to see what the gap does. The `kv_gap` line and the
  spans are the instruments.
- The claim is deterministic by design. A test that needs a retry to get the
  expected slots has found a bug.
- The model gates run one model per `cargo` process with the daemon stopped,
  and the device tests need the card alone.
- `elevate.rs` keeps its own `device.synchronize()` on the warm→hot path. It is
  outside this change and will still show in the wave windows.
