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

### 1.2 The test sees it

The batched test (`batch_test/utils.rs`) runs the scheduler's gap quantize on
the forward thread, inside its timed windows (§7), so its rates carry these
costs; its `ex-gap` columns take the gap's host time back out. A boundary seal
outside the timed windows, which is what the test ran before, hid every one of
them from the sweep.

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

A gap does these things, in this order:

1. **Catch-up** (§3.5), if a gap is outstanding and its completion word is set:
   install the converted blocks, record the summary's demand, and return every
   leased slot no installed block took to its pool. The lease is then empty.
2. **Demand** (§3.2): the classes and slot counts this gap's table needs.
3. The arena creation and the compaction the gap already does
   (`scheduler/run.rs`), with this gap's demand set aside as ground, so that
   nothing is released that the lease would grow straight back.
4. **Lease** (§3.2): allocate the demand from the pools, the lowest-placed
   arena first.
5. **Quantize** (§3.1–§3.4), if no gap is outstanding and there are blocks to
   take.

**At most one gap is outstanding.** A gap whose predecessor's completion word is
not yet set quantizes nothing; its blocks wait for the next gap, and the
outstanding gap's lease stays held, untouched, until its catch-up. The summary
is therefore one buffer, a lease is never held by two gaps, and the host never
reasons about two unread results.

### 3.1 Inputs, without a readback

The host knows, without asking the device, every block that became complete in
the forwards since the last gap. It scheduled those rows, so it knows each
sequence's offset, and it allocated the native chunks those blocks live in. From
its own bookkeeping it builds a **descriptor table**: one row per block, giving
the source K/V pointers, the (layer, head) layout, the block's token range, its
sequence's position (for attention-sink protection on positions 0–3) and the
policy group it seals under. That is invariant 2b's shape. No block is copied together to
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
- **One selection launch per policy group.** Thresholds and candidate formats
  are fixed by the compression level and the model's `*_KV_FACTORS`, so every
  block sealed under one policy shares them; they never vary block to block.
  Policies differ only where a layer seals at its own level (the MTP draft head,
  `layer_seal_cap`) or a conversation overrides its level or formats
  (`effective_turn_policy`). The table is grouped by policy and the selection
  kernel runs once per group, with the group's thresholds as its scalar launch
  arguments, exactly as the seal groups layers today. A gap normally holds one
  group, so it is one launch; a handful at most. The selection kernel's
  threshold interface is unchanged.
- **A sealed residence's partial final block** is quantized as a full block. Its
  dead slots are zero and the valid range is passed to the selection kernel, as
  `kv_tier_migration.md` §3 already specifies. A live sequence's writer chunk is
  not in the table.
- **A table is cut to what the gap's scratch can carry.** The selection blocks,
  the convert buffer and the launch inputs run in held scratch allocated once,
  with the gap's state, for the widest table that fits `GAP_SCRATCH_BYTES`.
  Nothing is allocated between forwards: a buffer grown there comes from held
  ground behind the forward's fence and waits the forward out — measured on
  the RTX 3090 at 260–430 ms for a ×8–×16 prefill gap. Every byte the launch
  uploads is packed into one pinned host image, so each copy is queued behind
  the forward instead of staged once the stream reaches it. Blocks past the cut
  wait for the next gap.

### 3.2 Device-side band claim

Today the format selection runs on the device and is then read back, so the host
can allocate each band's destination in the class its chosen format needs. On
the scheduler thread that would be a synchronous readback in the loop, against
invariant 3. Instead, **the host leases slots ahead of time, and the device
chooses among them.**

- **No existing kernel changes.** The selection and convert kernels run exactly
  as today. The selection already leaves its results (palette tags, outer
  scales, per-channel palette slots) in device memory before today's readback
  copies them out; the claim reads them there. The convert already reads its
  per-head source and destination records from one device buffer; the host
  writes the source half and the claim writes the destination half. The gap
  adds two kernels of its own in `kv_gap_claim.cu`: the claim, which also
  writes the destination records, and the completion word.
- **A lease lives for one gap.** Each gap the host allocates a lease for that
  gap's demand alone: free slots claimed from each class's arenas
  (`ChunkGidPool`: the lowest-placed arena with a free slot first, and within
  an arena its own free-slot order), held as `ChunkGid`s on the host, and
  handed to the device as a **slot stack** of slot addresses, one run per
  class, built from the slots just allocated and uploaded with the gap's other
  inputs. The stack is about one eight-byte entry per band of this gap — some
  3 MB for an 8K prompt's prefill gap, a few KB for a decode gap — against the
  hundreds of bytes per band the convert moves. At the next catch-up every slot
  the gap did not install goes back to its pool, so no lease stands between
  gaps: compaction and the empty sweep see every free slot, the next lease is
  packed into the lowest arenas first, into the holes that truncates, evictions and freed
  natives opened, and the addresses on the device are never older than the gap
  that reads them. The host is the only owner of a leased slot; the device is
  told addresses and reports, through the summary, which of them it used.
- **Claim, in table order.** A band's format is the selection's palette tag,
  or the policy's uniform override; a host-built table maps each format code to
  its size class and its record format code. The claim gives every band its
  index in its class: the count of bands of that class before it in the table,
  bands ordered job, head, palette, K before V. A gap with several policy
  groups claims once per group against a per-class count the groups share, so
  the gap's indices are one sequence. The claim is a prefix count, not a race
  on an atomic cursor, so the same table against the same lease always takes
  the same slots, and the lowest places fill first. It is one launch per group:
  a single-pass scan with decoupled look-back over tiles of whole jobs, sixteen
  class counters two to a word. Its look-back status words are scratch zeroed
  once at allocation; each launch carries a new epoch, so nothing is cleared
  between launches. A job has at most 64 KV heads, and the head dimension is
  128 or 256 — the two the convert has a record for, and the only two the
  claim is built at; the gap refuses any other at setup.
- **A class that runs dry defers the block, never fails it.** A block is
  deferred when any of its bands' indices falls past its class's lease. The
  whole block stays native and goes first into the next gap's table. Its bands
  still hold their indices, so no later block's slot moves; the slots they name
  are not written, and go back to their pools at the catch-up with the rest of
  the unused lease. The convert runs over every row of the buffer,
  so a deferred block's destinations are set to one **sink** slot as wide as
  the widest class, which nothing ever reads: the convert writes a deferred
  band there and nowhere else. A block that stays native for a gap is still
  attended correctly, only larger, and this is the only way a complete block
  misses its gap.
- **Every band takes a slot.** The selection's candidates are all quantized
  formats (R16 among them at C0), so every band is converted into a slot of its
  class.
- **The demand comes from the summaries, not from a model.** The claim counts
  every band of the table into its class, deferred or not, so each summary
  carries the exact per-class demand of the blocks it saw. A gap's demand is
  two parts:
  - **Retried blocks, exactly.** Each deferred block's rows name the format of
    every band, so its demand per class is known to the slot. The claim counts
    every group of a gap in table order, against counts the groups share, so
    the deferred blocks lead the next table **in groups of their own** — one
    per policy among them — ahead of every group of new blocks, whatever the
    new blocks' policy; a policy may hold a retry group and a new group in the
    same gap. A lease that covers their exact demand is then certain to place
    them: the selection is deterministic on native bytes that have not
    changed, and chooses the same formats again. Were a retried block to sit
    in a group behind new blocks, those blocks could take its slots first and
    defer it again, gap after gap.
  - **New blocks, by measured share.** Their band count is exact — blocks ×
    heads × `N_PALETTE` × 2 — and only the split across classes is not yet
    known. It is taken from each class's peak share of claims over the recent
    summaries, kept apart for prefill and decode gaps, and capped at the band
    count. The share is a fraction, not a volume, so a prefill arriving after an
    idle period is leased for its own size. Before any summary of its kind
    exists, the split is unknown and real K/V concentrates in a class or two,
    so an even split alone defers most of the first gap — a prompt's worth of
    K/V native for a gap more. Each reachable class leases its even part of
    the count **firm** and the rest of the count **spare**; the over-lease is
    returned at that gap's catch-up, once.

  An over-estimate costs only slots allocated and returned within one gap; an
  under-estimate defers the blocks past it, whose exact demand the summary then
  reports, and the next gap places them. Nothing is held against a backlog:
  this replaces §2a's backlog-sized reserve. The lease is KV ground below
  `weight_floor` (invariant 7), from free slots in existing arenas first and
  then new arenas. **Firm demand** — retried blocks' exact demand, the measured
  shares, a cold gap's even split — may buy ground from the weight side when
  the free ground runs out: it is measured, the blocks it converts give back
  more native ground than they take, and in all it buys at most one count's
  worth. **Spare demand** takes only ground standing free and never buys
  (`region_pool::without_purchase`): a purchase blocks on the expert pipeline
  and evicts resident experts, and a guess whose slots are mostly returned one
  gap later is worth neither. Measured on the RTX 3090 before the split, a cold
  prefill gap at C10 ×16 that bought its whole guess spent 577 ms in the lease
  and left decode with 2.4 GB less expert residency. When the ground cannot
  cover a class's firm demand, the class defers blocks for a gap.
- **Ground, not slots, is what is kept.** Compaction and the empty sweep run
  with the gap's demand per class set aside, so an arena the lease is about to
  fill is not released and grown again within the same gap. Which slots fill
  that ground is decided fresh each gap, the lowest-placed arena first: packing
  is by arena, because an arena is released only when every slot in it is
  free.

### 3.3 Convert

One batched launch over every claimed band, across every layer and sequence
(invariant 5), reading its destinations from the claim. Destinations are
allocated uninitialised (invariant 6). The convert is launched once for the
whole table, never once per layer: the table is cut to the convert's single
launch (65,535 jobs), which §3.1's scratch cut is already well inside.

The claim writes each head's destination record (palette maps packed two bits a
channel, record format codes, outer scales, slot addresses) into the convert's
buffer and into the summary row (§3.4), in the same launch that counts. It packs
the maps, the bulk of its bytes, while its tile's predecessors resolve, and
assembles each tile's rows in shared memory so they leave as two contiguous
spans. Under a uniform override a side's map is the identity and its scales are
one, as today. On the RTX 3090 it runs at the device copy roofline: 88 µs for an
8K prompt's 12,288 jobs, 446 µs for a full 65,535-job table, against 87 µs and
457 µs for a device copy of the same bytes (`candle-core/tests/kv_gap_claim_bench.rs`).

Nothing else is written on the device. No record is filled and no block table is
touched in the gap: both are the host's to change, at the catch-up.

### 3.4 The summary

The kernels write the gap's summary into one mapped pinned buffer, as the MoE
dispatch writes its routing summary (`expert_lre/dispatch.rs`, `SummaryRing`).
It holds:
- one flag per block: converted or deferred;
- for each head, its destination record exactly as the convert read it, and
  each of its eight bands' lease index;
- a **completion word**, written by the gap's last launch (after the convert),
  holding the gap's sequence number.

That is what the selection reads back to the host today, delivered through
mapped memory. The record carries the palette maps packed, two bits a channel
as `PalMapBytes` holds them (`head_dim / 4` bytes per head for K and for V), not
one byte a channel as today's readback widens them (`sampled_selection/gpu.rs`),
so a head is `head_dim / 2 + 136` bytes. The host reads it only once the completion word matches, by
reading the mapped word at the head of a gap. No host CUDA call reads or waits
on the buffer. The buffer is sized at start for the largest table a gap can
build (§3.1's cut).

### 3.5 The host catch-up and the native bound

When the completion word is set, the head of the next gap installs the
outstanding gap's blocks. For each converted block the host:
- finds the native chunk the gap converted: at the job's index, or — when a
  front eviction or a splice moved it while the gap ran — wherever it moved to
  in its sequence, found by identity. The search runs only for a job whose
  index no longer holds its chunk. A chunk truncated or evicted away is gone,
  and its row is discarded. Matching by index alone would leave a moved block
  native for good: it already sits behind the sequence's cursor, so no later
  gap takes it;
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

Then the rest of the lease goes back: every slot no installed block took — the
slots past every class's demand, the slots deferred blocks were given and
never wrote, the slots of rows discarded as stale — returns to its pool, and
the lease is empty. A deferred block leads this gap's table in a retry group
ahead of every new block's group (§3.2), its exact demand
taken from its rows, and each class's share of the summary's claims is recorded
for the new blocks' demand (§3.2). This is the install the persistence thread does today, moved to
the thread that owns the sequences, with one fence left and no wait: the
convert has finished, the compute stream is set to wait on the gap's stream
(§3.8 rule 2), and the header upload is queued on the compute stream ahead of
the next forward.

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
before anything is built (§8 step 1). The boundary seal the batched test ran
then (`quantize_and_seal_sequences`, since deleted) ran the same selection and
convert core. Its device time per MiB of native K/V,
at the rate production produces K/V in decode, decides whether §7's 95% gate
can be met by the kernels as they are. If it cannot, the kernels are made faster
first. The gate is not loosened.

**Measured: the kernels as they are meet it.** Qwen3.8-Flash-Next's batched
forwarding gate on WhiteWorld (RTX 3090, card alone), 2026-11-13, at
`f4717577c`, from each row's `seal core` lines (`models/seal_timing.rs`):

| Row | Prefill seal | Decode seal | Decode K/V rate | Share of decode |
|---|---|---|---|---|
| C0 ×2 | 34.9 MiB, 0.25 ms/MiB | 13.0 MiB, 0.24 ms/MiB | 3.63 MiB/s | 0.09% |
| C5 ×2 | 34.9 MiB, 0.27 ms/MiB | 13.0 MiB, 0.31 ms/MiB | 3.63 MiB/s | 0.11% |
| C5 ×8 | 137.3 MiB, 0.26 ms/MiB | 52.0 MiB, 0.26 ms/MiB | 7.67 MiB/s | 0.20% |
| C8 ×2 | 34.9 MiB, 0.50 ms/MiB | 13.0 MiB, 0.62 ms/MiB | 3.47 MiB/s | 0.21% |
| C10 ×2 | 34.9 MiB, 1.47 ms/MiB | 3.2 MiB, 3.52 ms/MiB | 3.04 MiB/s | 1.07% |
| C10 ×8 | 137.3 MiB, 1.47 ms/MiB | 13.0 MiB, 1.55 ms/MiB | 5.82 MiB/s | 0.90% |

The cost per MiB is the selection plus the convert, the selection's host
readback included, which the gap removes (§3.4). The selection dominates and
grows with the level's candidate count (C10 searches the most formats); the
convert stays at about 0.07 ms/MiB at every level. The worst row spends about
1% of decode, a fifth of what §7 allows, so no kernel is made faster before
the gap is built.

### 3.7 What the gap logs

The scheduler writes one `kv_gap` line for each second in which a gap ran,
summed over that second's gaps: blocks converted, blocks deferred and the class
that ran dry, gaps that quantized nothing because one was outstanding, native
bytes standing, lease bytes held, and the gaps' device time. It replaces the
`kv_reserve: pass` line, which each attachment's persistence thread writes
today about the one shared pool, so that every pass is logged once per tenant.
§7's production gates are read from this line.

### 3.8 Ownership: why the gap cannot race

Every piece of state the gap touches has exactly one owner at any moment, and
every hand-over between owners is ordered by the stream or by a word the
receiver checks. These six rules are the change's safety argument. Each step of
the build is checked against them, and a change that needs an exception to one
of them is redesigned rather than excepted.

1. **One thread changes the bookkeeping.** Chunk lists, the gid pools, the
   leases, the gap cursors and the records are changed only on the scheduler
   thread. The persistence thread copies sealed bytes it was handed and appends
   them to the log. It changes no topology and takes no lock the scheduler
   needs (§5). With one writer there is no host-side race to lose, which is why
   the `MigrateFlight` lock is deleted rather than kept (§6).
2. **The gap runs on its own stream, fenced to the compute stream at both
   ends.** Selection, claim and convert are queued on the gap's stream, which
   first waits on the compute stream — so the gap reads its blocks only after
   the forward that completed them, and writes its lease only after the region
   cleans the lease's claims queued there — and then runs beside the next
   forward instead of ahead of it. The catch-up, once it has seen the
   completion word, has the compute stream wait on the gap's stream, so a
   forward reading an installed block is queued behind the gap's writes.
   Between the two, nothing a forward touches is the gap's: the native bands it
   reads are pinned until its catch-up (rule 4), the slots it writes are its
   lease's (rule 3), and its scratch, summary and status words are its own. On
   the RTX 3090 at C10 this took the gap's kernels off the forward's critical
   path: decode 748 → 888 t/s at ×8 and 1,024 → 1,130 at ×16, prefill
   4,606 → 5,566 and 4,663 → 5,692.
3. **A slot has one owner: the free pool, a lease, or a sealed block.** A lease
   takes its slots out of `ChunkGidPool` before the gap that may write them is
   launched, and nothing else hands them out while they are leased. The lease
   is the host's: the device is handed addresses and owns none of them. It
   writes only to `stack[class][i]` with `i` below the class's lease length, so
   no address it can write lies outside the lease. A slot leaves the lease only
   at the catch-up, which empties it: into a block, or back to its pool.
4. **A native band an outstanding gap reads is not freed until its catch-up.**
   The gap holds a `ChunkPin` on every native band in its table, and the
   catch-up drops it. A truncate, an eviction or a seal in the window drops
   only its own reference, so the band cannot return to the pool and be
   rewritten while the convert may still read it (§3.5, "a native band is
   freed by its last reference").
5. **The host reads only a finished summary, and installs only rows that still
   match.**
   - The gap's last launch writes the completion word after a system-scope
     fence; the host reads the word first and the rows only once it holds the
     gap's sequence number.
   - Each job holds the native gids it converted. The catch-up installs a row
     only into the chunk that still holds those gids — at the job's index, or
     where a front eviction or a splice moved it. A chunk that is gone has its
     row discarded and its slots go back to their pools; a deferred block that
     moved is retried at its new index.
   - At most one gap is outstanding. A gap that finds one quantizes nothing, so
     no buffer, lease or row is ever shared by two gaps.
   - Arming a gap clears the summary's error and completion words. Completion
     words repeat once the sequence number wraps `u32`, so the word a gap is
     polled against is never one an earlier gap left behind.
   - Everything that can refuse a gap — a token window outside its chunk, a
     band that is not a GPU float or R16, a gid its arena no longer backs, a
     code the tables do not place — refuses it on the host, before the first
     launch. A launch the driver refuses part way is waited out before the
     error returns, and the gap's device state waits out an outstanding gap
     before it frees the summary, so no kernel writes memory that has gone.
6. **Persistence sees a block only after its install.** The persistence thread
   is handed a quantized block at the catch-up, after the convert has finished.
   A sealed slot freed while a copy reads it goes through the reclaim queue, as
   every device free off the scheduler thread does today
   (`drain_reclaimed`).

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
- **The warm tier stays where it is.** The copy already moves at the link's
  full bandwidth: `migrate_sealed_to_cpu_batch_async` gathers the bands on the
  device and copies them into pinned staging on its own stream, and only the
  scatter from staging into the warm tier's host arenas runs on the host. The
  warm arenas therefore stay pageable host memory, budgeted as today — one
  `WarmLedger` share per attachment of the machine's KV warm budget
  (`KV_WARM_FLOOR`), purged to its cold copy past it. A pinned pool would buy
  no bandwidth and charge the machine's commit with every byte it held.
- **A full tier.** Past its budget the tier drops its least recently used
  blocks that the log already holds, which are then cold and are read back
  from the log when wanted. A block the log does not yet hold is never
  dropped. The gap never waits on the tier: a block that cannot be copied yet
  stays hot and quantized.
- **Read from cold into warm.** `PersistenceTrigger::warm_turns` asks the
  persistence thread to bring a list of turns from the log into the warm tier,
  without placing them hot: best-effort, skipping a turn already warm or hot,
  and dropping what does not fit the thread's share. The hit-driven pool
  (`docs/hit_driven_projection.md` §7.2) warms its pieces this way, so a round
  the slot later chooses lifts at the link's speed rather than the disk's.
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
- **The persistence thread touches no arena topology.** Its pass opens with a
  `device.synchronize()` that orders more than the migrate: the deferred arena
  creation and the idle demote that follow it in the pass (`run_pass`) rely on
  it too, and taking it away once raised `CUDA_ERROR_ILLEGAL_ADDRESS` from 0
  to 199 in a run. Both leave the persistence thread. The gap already creates
  the arenas a refusal owes (§3), and the idle demote runs in the gap after
  compaction, between forwards, where every free is ordered behind the
  forward that last read the block. With the copy on its own stream and
  nothing else on the device, the persistence thread has nothing left that
  the fence ordered.
- **The backlog gauge measures the copy.** `pending_warm_bytes` is the
  quantized bytes installed hot and not yet copied warm, stamped by the
  persistence thread as today. `on_backlog_fall` still wakes the scheduler,
  and `flush_blocking` / `flush_at_least` still wait for warm copies, because
  a hot block is evictable only once a warm copy or the log holds it. The
  figure is about a quarter of today's for the same K/V. `pinned_undrainable_bytes`
  keeps its meaning.

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
- `BatchedInferenceSession::quantize_and_seal_sequences`, its seal by
  truncate and re-inject, and its `SealTiming` (`models/seal_timing.rs`).
  Deleted. The batched test runs the scheduler's gap quantize inside its timed
  loop; a caller that needs every complete block quantized now — the fidelity
  gate, the drafter test — calls `gap_settle`. There is one implementation.

## 7. How it is verified

Unit tests, written with the code, asserting raw expected bytes:
- the descriptor table built from host bookkeeping for a known wave: complete
  blocks in, writer chunks out, a sealed residence's partial block in, rows
  past the committed offset out, a block one layer lacks out, and nothing
  before a sequence's cursor;
- the lease for a known table: the retried blocks' exact demand plus the new
  blocks' band count split by the recorded peak shares (the whole count to each
  reachable class when there are
  none), never more than the table's band count per class, allocated lowest
  place first; and the catch-up emptying it, every slot either in an installed
  block or back in its pool;
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
  a full pool drops only blocks the log holds;
- the ownership rules (§3.8), as a randomised model test with no device: the
  lease, the cursor, the pins and the catch-up's match check driven through
  thousands of random interleavings of forwards, truncates, evictions, seals,
  gaps and catch-ups against a reference model, asserting after every step that
  each slot has exactly one owner, that no gid is freed twice or while pinned,
  that no block is both sealed and deferred, and that no complete native block
  is left behind a cursor without being retried. A failing interleaving is kept
  as a fixed regression test;
- on the device, a truncate and an eviction landing while a gap is outstanding:
  the stale rows are discarded, their slots return to their pools, and the
  surviving blocks hold exactly the bytes the per-block reference quantizer
  gives.

Under `tensor-assert`, at production throughput:
- `readonly_regions` declares the free pool and every other tenant's ground
  immutable while a gap is outstanding, so a stray write names its writer;
- `SlotIntegrity` watches installed blocks for any change after their install;
- the run's throughput is checked against production's. An instrumented run far
  below it proves nothing (the `tensor-assert` harness's first danger).

Gates, run one model per `cargo` process with the daemon stopped:
- every model's fidelity ladder and the sweep, with quantization inside the
  timed windows;
- in production, a decode with the persistence thread active shows forward time
  of at least 95% of each wave window;
- native K/V in VRAM never exceeds §3.5's bound: the writer chunks, the two
  most recent forwards' blocks, the deferred blocks and the clones;
- gaps that quantize nothing because one is outstanding are under 1% of the
  gaps with a table;
- p95 commit wait under 1 s;
- a soak at full load on WhiteWorld with the characters live, clean for well
  past the five minutes the last production corruption took to show.

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
7. persistence reduced to copy-and-append on its own stream into the warm
   tier;
8. the deletions in §6;
9. `kv_drain_under_pressure.md`, `AGENTS.md`'s tiering note and
   `turn_claims.rs`'s commit-wait comment rewritten to match.

The gates in §7 decide whether it is done. §9 gives each step its code.

## 9. Tasks, with the code they touch

Line numbers are as of commit `dc04b62ac`.

**1. Measure the core (§8 step 1).**
- `candle-transformers/src/models/batch_test/utils.rs:2056` and `:2195`: the two
  boundary calls. Time them, and report device time per MiB of native K/V.
- `candle-transformers/src/models/batched_inference.rs`,
  `quantize_and_seal_sequences`. Its convert was issued inside a loop over every
  backing with the whole descriptor list each time, where
  `convert_deferred_descs` says any one backing serves: it ran the convert once
  per KV layer. It now runs once, and the call answers its `SealTiming`
  (`models/seal_timing.rs`), which both boundaries print as a `seal core` line:
  native MiB, selection and convert time, ms per MiB and, at the decode
  boundary, the share of decode the gap would take.

**2. The descriptor table and the cursor (§3.1).**
- New `candle-nn/src/kv_cache/chunked/gap/table.rs`: the row type and the
  builder. One concern, tested alone.
- `candle-nn/src/kv_cache/chunked/sequence_ops.rs`: the per-sequence cursor
  beside `writer_start_idx`; `truncate_sequence_to_tokens` (`:2134`) keeps the
  cursor at or below the committed block.
- `candle-nn/src/kv_cache/chunked/types.rs:216`: `ChunkWindow` (`gids`, `meta`),
  the source of a row's pointers.
- `batched_inference.rs:2172`, `assert_sealed_layers_aligned`: the every-layer
  rule the builder applies per block.
- `batched_inference.rs:645`, `layer_seal_cap`: the per-layer level that puts a
  row in its policy group.
- `candle-nn/src/kv_cache/chunked/compress.rs:69`, `SELECTION_BUDGET_BYTES`: the
  cut.

**3. The lease (§3.2).**
- New `candle-nn/src/kv_cache/chunked/gap/lease.rs`: the one-gap lease, its
  demand (retried blocks exactly, new blocks by peak share), its device stack,
  the take at the catch-up and the return of the rest.
- `candle-nn/src/kv_cache/chunked/gid_pool.rs:1428`, `ChunkGidPool`: the lease
  allocates through `allocate_n_for`, which already takes the lowest-placed
  arena with a free slot (a find-first-set over the capacity bitmap); each
  slot's address is computed as it is allocated, so building the stack is a
  copy. `set_reserve_target` (`:1465`) goes.
- `candle-nn/src/kv_cache/chunked/alloc.rs:509`, `owe_reserves`, and
  `reserve_target.rs:45`, `reserve_targets`: replaced by the lease's demand.
- `alloc.rs:664`, `create_deferred_arenas`, and the compaction: run with the
  gap's demand per class set aside as ground, and make the arenas a lease is
  short of, in the gap.

**4. The kernels (§3.2, §3.3).**
- `candle-kernels/src/quantize/select_kv_format.cuh`,
  `select_kv_format_palette4_paged`: unchanged. One launch per policy group
  (§3.1), its results left in the selection block for the claim.
- `candle-kernels/src/quantize/palette4_convert.cuh`: unchanged. Its buffer's
  destination half is written by the claim.
- New `candle-kernels/src/simple/kv_gap_claim.cu`: the claim (the per-class
  prefix count, the deferral rule and the destination records, one launch) and
  the completion word. Listed in `SIMPLE_KERNELS`
  (`candle-kernels/build_utils.rs`), bound in
  `candle-kernels/src/simple/kv_gap_claim.rs`. Its harness,
  `candle-core/tests/kv_gap_claim_bench.rs`, checks every output byte against a
  sequential reference and holds the benchmark and the `ncu` target.
- `candle-core/src/quantized/cuda.rs`, `quantize_palette4_convert_buffered`:
  split into building the buffer and launching on it, so the gap launches the
  same convert on a buffer whose destination half the device wrote.
- New `candle-nn/src/kv_cache/chunked/gap/codes.rs`: the per-code class and
  record-format tables the claim reads.
- New `candle-nn/src/kv_cache/chunked/gap/launch.rs`: one gap on the device —
  every group's selection tables and the claim's inputs in one staged upload,
  per group the selection and the claim, one convert over the table, the
  completion word. It keeps, across gaps, the summary, the claim's status words
  and epoch, the sink and held scratch for the selection blocks and the convert
  buffer, and refuses a gap while one is outstanding. The candidates and
  thresholds come from `compress.rs`'s `policy_selection`, each head's source
  bands from its `head_source`, and the selection's gid order from
  `HeadGids::extend_selection_gids` — all three shared with today's seal, and
  the token-window check with today's convert (`check_token_window`). `gap/launch_tests.rs` holds it to today's seal: the same native
  chunks, through `quantize_layers_deferred` and through one gap, give every band
  the same format, outer scale, palette map and payload bytes — at C5 and C10,
  under a K override, split across two groups — and a short lease defers
  exactly the jobs past it. It also refuses a bad window and a gap over no
  layers before launching, drops the device state mid-gap cleanly, and does
  not see the gap after a sequence wrap complete early.
- `candle-nn/src/kv_cache/chunked/sampled_selection/gpu.rs:741`, `run_palette4`,
  and the readback at `:1497–1538`: removed.

**5. The summary and the catch-up (§3.4, §3.5).**
- New `candle-nn/src/kv_cache/chunked/gap/summary.rs`: the mapped buffer, its
  row layout and the completion word. Pattern:
  `candle-transformers/src/models/expert_lre/dispatch.rs:156`, `SummaryRing`.
- New `candle-nn/src/kv_cache/chunked/gap/install.rs`: the catch-up over live
  sequences — the summary read only once `GapDevice::finished` hands it out,
  each block found by identity, its slots taken block by block, every record
  built in one batch, each layer's chunks swapped in under one write of its
  block table with their headers re-serialised in one upload per sequence and
  the writer boundary moved past them, the rest of the lease returned, and the
  deferred blocks handed back with their exact demand. `gap/assemble.rs`
  rebuilds a block's slots in grid order and its chunk state from its rows, as
  today's seal lays it out. `gap/install_tests.rs` holds the installed chunks
  to today's seal band for band, and covers a truncate under the gap, the
  natives' return, the lease's surplus and a deferred block installed by the
  next gap. `gap/ownership_tests.rs` is the randomised model test of §7.
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
- `candle-nn/src/kv_cache/chunked/gap/cycle.rs`, `GapCycle`: the gap's state
  from one gap to the next and its two calls, `catch_up` and `quantize`
  (`gap_take` cuts each sequence's take to the gap's room).
- `candle-transformers/src/models/batched_inference/gap.rs`: the session's
  `gap_catch_up` and `gap_quantize` over every live sequence, each layer under
  `layer_seal_policies`, resolved through `seal_policy` — and `gap_settle`, gaps
  run until the cycle is quiet, for a caller that must see every complete block
  quantized now; no gap for a session that
  does not quantize, a head_dim the convert has no record for, or the
  single-latent layout. The gap's state is made with the session, before any
  forward; a table holds the jobs whose scratch fits `GAP_SCRATCH_BYTES`
  (64 MiB, `gap_jobs_within`): 3,866 at 4 KV heads of 128, 4,761 at 2 of 256.
- The `kv_gap` line, `candle-conversation/src/scheduler/kv_gap.rs`: summed per
  second and written while gaps run.
- `batch_test/utils.rs`, `batch_test/fidelity.rs` and `quantized_qwen38_moe.rs`:
  every caller of `quantize_and_seal_sequences` moves to the gap — the batched
  gate's loop to `gap_catch_up` and `gap_quantize`, the other two to
  `gap_settle` — and the function is deleted.

**7. Persistence (§5).**
- `candle-conversation/src/persistence/thread.rs`: `run_pass` (`:1036`) keeps
  the copy and the append. Deleted: `await_owed_arenas` (`:694`),
  `install_hot_to_warm` (`:710`), `migrate_group_hot_to_warm` (`:771`), the
  syncs at `:1100`, `:1334` and `:1414`, `MAX_GAP_RETRIES` (`:1253`), and the
  pass's deferred arena creation (`:1146`). The idle demote (`:1162`) moves to
  the gap (§5). `BacklogGauge` stays, stamped with the copy backlog.
- `candle-nn/src/kv_cache/chunked/chunk_ops.rs:1297`,
  `migrate_sealed_to_cpu_batch_async`: the copy, on the thread's own stream,
  through its pinned staging into the warm arenas.
- `candle-conversation/src/persistence/`: `PersistenceTrigger::warm_turns`, a
  cold-to-warm read on the persistence thread. `persistence/warm_ledger.rs`
  gives each thread its share of the warm budget, as today.
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
  `weight_floor` move (invariant 7). The stack is built each gap from slots
  allocated after that gap's compaction, so it is never older than the gap
  that reads it; nothing keeps it across gaps.
- While a gap is outstanding the host does not know which leased slots it
  wrote. Compaction and the empty sweep treat the whole lease as held until the
  catch-up, and never release an arena a lease stands in. That is at most one
  gap's lease, and only while that gap is outstanding.
- A record names its bands by raw address, so it must hold their gids
  (`meta_pool.rs`). A record filled at the catch-up holds the claimed gids, not
  the native ones.
- A `ChunkPin` holds both the bands and the record (`gpu_chunks.rs`). Holding
  only the first has already cost a night.

**The sequence's layout.**
- The catch-up replaces a chunk in place. It does not truncate and re-inject:
  re-injecting shared chunks under a lone sequence has corrupted the decode
  read before (the float-mode comment the deleted boundary seal carried).
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
  in three hours, the largest 113 MB. The gap takes its 64 MiB once, with the
  session, and the table's cut is sized from that scratch, so no take of the
  gap's is declined or grown later.
- The persistence copy gathers into device staging before the DtoH
  (`chunk_ops.rs`). That staging is transient VRAM and stays small.
- The lease never buys ground from the experts. Deferral is the release valve,
  and a run of deferrals is a finding to log, not a reason to raise the lease.
- Pinned staging is charged to system commit under WDDM. WhiteWorld already
  commits 119 of 206 GB, so the copy's staging stays bounded and the warm tier
  stays pageable.

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
