# Live MoE Dispatch — the expert forward without a host round trip

> **Status — Revision 2, implemented (§0); measured on the qwen36 gate (§0.7.2).**
> Implementing revision 1 on the
> RTX 3090 (WDDM) found that a gate GEMM which waits on a *host-driven* release
> can deadlock against any driver call that synchronizes with the device (§0.1).
> Revision 2 takes demand misses off the host and the driver entirely: worker
> blocks inside each expert launch copy every miss from pinned memory into VRAM
> scratch and compute it, cold experts are staged from NVMe by a thread that
> makes no CUDA call, and every host↔GPU signal is a mapped word (§0.2–§0.10).
> Measured end to end on the 3090 in simulation (§0.10.1–2). §0 is the design;
> §1–§18 are revision 1, retained where §0 builds on them (see the note before
> §1).
>
> **Status of revision 1 — Superseded in part.** Replaces the host-orchestrated expert path
> (`route_indices` → `submit_moe_work` → pipeline-thread compute) with one
> device-side path for every MoE model, paged or all-resident. The pointer table
> the device path already uses becomes *live*: entries for non-resident experts
> are zero, the expert pipeline thread fills them as copies land, and the gate
> GEMM waits on the entry instead of the host waiting on the GPU.
>
> Everything here is grounded in the tree at `8d061e4fc` + `33dac8cf2`. Line
> references are to that tree. `docs/archived/gpu_native_moe_dispatch.md` §Phase B
> sketched this direction ("residency table with NULL sentinels, doorbell
> miss-service"); this document is the design that was missing from it.

---

## 0. Revision 2 — demand misses fetched by the expert kernels

### 0.1 What revision 1 met on hardware

Measured on the RTX 3090 box (WDDM, driver 560.94), with unit tests in
`candle-core/src/quantized/cuda_tests.rs` (`cuda_waiting_gate_*`) and traced
runs of the qwen36-35B gate:

1. **Any implicitly synchronizing driver call made while a gate block spins
   blocks until the gate ends.** The one that fired in practice is **lazy kernel
   loading**: the first launch of a kernel loads it, and the load waits for the
   device to go idle. Behind a spinning gate it blocks the thread for the full
   spin limit (1.5 s), and other threads' driver calls stall with it — the
   pipeline thread's 8-byte table clear sat in `cuMemcpyHtoDAsync` for 1.51 s.
   The forward thread launches kernels behind every gate (up, SwiGLU, down, the
   next layer), so any first use of a kernel can deadlock the release.
2. **Copies are not starved by the driver.** With the successor's kernel loaded
   beforehand, a copy submitted *after* work was queued behind a spinning gate
   runs beside it and releases it (67 ms for 32 × 2 MiB out of a 2 GiB pinned
   tier), on the legacy null stream and on a non-blocking stream alike. An
   earlier conclusion that copies starve behind a queued successor was this
   lazy-load effect: the test thread never reached the copies.
3. **A host store into mapped pinned memory reaches a spinning kernel with no
   driver call.** A gate spinning on a table in mapped memory, with work queued
   behind it, was released by plain host stores 0.3 ms after they were made,
   bit-identical to the launch over the filled table.
4. **Host waits on device events are unreliable while a gate spins** — a
   pipeline thread blocked on a compute-stream event that the device had long
   passed. The summary hand-off now uses a sequence word `bucketize` writes into
   mapped memory, polled by the host (§0.3), not an event.

The rule revision 2 is built on: **once a gate block can wait, nothing it waits
for may need the driver.** Lazy loading stays on deliberately — it is the
sharpest test of that rule, and the acceptance gate runs with it.

### 0.2 The shape

Every expert the router can pick is readable by the GPU at all times, from one
of three places, and the live table says which:

| Live-table entry (per row × expert, gate / up / down) | Where the weights are | The GPU |
|---|---|---|
| a **VRAM** address (weight-zone slot) | hot — cached in VRAM | reads VRAM |
| a **pinned host** address (warm-tier slot or pad slot) | warm — a slot image in pinned RAM | a worker block copies the slice it needs into VRAM scratch and computes it; nothing waits |
| **0** | cold — on the NVMe pack only (or in a pageable warm slot) | a worker spins on the entry until the stager publishes a pad address, then as above |

```text
forward thread (null stream)            stager thread (new)              pipeline thread (off the critical path)
────────────────────────────            ───────────────────              ───────────────────────────────────────
router → route
bucketize ── reads the live gate row,   polls summary word n             polls summary word n
   classifies VRAM / pinned / cold,     cold experts of row n:           stats, scores, Markov observe
   orders tiles pinned → VRAM → cold,     pad slot ← pack read           ring: offers slots (empty, lazy
   claims ring slots for its misses,      publish pad address              victims); collects claims, lands
   then for later rows' predicted         (host stores: up, down,          them once their invocation is done
   warm experts (read-ahead, §0.7.3);     fence, gate)                   predictions: warm → the ring's
   writes summary[n] + word n           then: speculative staging          read-ahead lists; cold → stager
gather                                    requests from the pipeline       (pack → pad, ahead)
gate GEMM ── W worker blocks copy each                                    VRAM eviction: retarget entry to its
   miss slice and compute it, then                                          pinned copy (or 0)
   copy the read-ahead images and
   publish them; hit blocks compute
   from VRAM
up, SwiGLU, down (same), scatter
next layer …
```

- **Nothing the GPU waits on needs the driver.** VRAM and pinned tiles wait for
  nothing. A cold tile waits for a pack read and a host store, both made by the
  stager, which issues no CUDA call on that path. A lazy kernel load or any
  other synchronizing call on any thread can therefore delay a release but
  never block it (§0.1).
- **A demand miss never touches the VRAM cache.** No eviction, no slot, no copy
  on the critical path. Demand misses feed the cache only through the pipeline
  thread's asynchronous promotion — a later pass hits.
- **The pipeline thread leaves the critical path entirely.** It owns policy —
  scores, the Markov transition matrix, which experts live in VRAM — and moves
  no bytes itself: the GEMM workers fill the slots it offers, for misses and
  for read-ahead alike, and no GPU wait depends on anything it does.
- **Bit-identical results.** A warm slot and a pad slot hold the pack record,
  which is the VRAM slot image (`slot_offsets`, `pipeline.rs:666`): the kernel
  reads the same bytes whichever address the entry names, and the scatter's
  canonical order (§4.3) is unchanged.

### 0.3 Memory

| Store | What | Allocation | Size | Mutable |
|---|---|---|---|---|
| Weight zone | VRAM cache slots | the span's right side (unchanged) | elastic (§9) | yes — pipeline thread |
| Warm tier | slot images, pinned + pageable parts | `cuMemAllocHost` once at startup (`pinned.rs:464`), pageable `PagedSlots` beyond it | sized by `warm_sizing_from` (`handle.rs:224`) | **no** — filled once (`pinned.rs:1-15`) |
| **Pad** (new) | slot images staged from the pack | one `cuMemAllocHost` block at startup | §0.8 | yes — stager |
| Live table | `[rows × E]` × 3 u64 | `cuMemHostAlloc(DEVICEMAP)` | 240 KiB (35B) | host stores |
| Summary ring | `[RING × (E + 1)]` u32 | `cuMemHostAlloc(DEVICEMAP)` | 66 KiB | `bucketize` |
| Abort word | u32 | `cuMemHostAlloc(DEVICEMAP)` | 4 B | host store |

**GPU access to the warm tier and the pad.** Both are `cuMemAllocHost`
allocations. Under unified addressing (every 64-bit platform CUDA supports,
WDDM included), memory from `cuMemAllocHost` is mapped into the device address
space at the same address, so the host pointer of a slot *is* its device
address; no `DEVICEMAP` flag and no `cuMemHostGetDevicePointer` is needed. This
is general CUDA behaviour that nothing in the product code exercised before (the
warm tier was only ever a copy source); it is confirmed on the 3090 — a grouped
GEMM reading `cuMemAllocHost` copies through their host addresses is
bit-identical to the VRAM run (§0.10.1, test 1 of §0.15).

**Pageable warm slots are not GPU-readable** (`WarmTier`'s `PagedSlots`,
`warm_tier.rs:95-116`). An expert whose warm copy is pageable is treated as cold:
its entry is 0 and the stager stages it into the pad by a host `memcpy` (0.59 ms
per 13.5 MiB slot, `warm_tier.rs:21-23`) instead of a pack read.

### 0.4 The live table

**Initial state**, written at load before the first forward: VRAM-resident →
VRAM address; pinned warm slot → its address; everything else → 0. Pinned
layers (`PINNED_LAYERS`, `cache.rs:73`) are always VRAM.

**Every write is a host store** to the mapped table (`[3][rows][E]`: gate, up,
down planes). A publish writes up, down, full fence, gate; a clear writes gate,
full fence, up, down.

**The snapshot.** `bucketize` reads all three entries of every *routed* expert
and copies them into a per-invocation VRAM snapshot (`u64[3][E]`); the layer's
gate, up and down GEMMs take the snapshot as their weight table and never read
the live table — except a cold expert's workers, which wait on its live gate
entry and then read its live up and down entries. So an entry the host
retargets after `bucketize` cannot reach a block that was ordered by the old
value: that block keeps reading the old address, whose slot the reclaim rule
(§0.5) holds until the invocation is done. An expert with **any** of its three
entries 0 is classified cold, so a clear caught half-way (gate already 0, or
up / down already 0) is a cold expert, never an address.

This replaces the revision-1 assumption that a hit block may read the live
entry: with the entry able to change between `bucketize` and the block — and
able to become 0 — a hit block reading it could have dereferenced 0.

**One writer at a time per entry: the `Residency` lock.** Two threads change
entries — the pipeline thread (VRAM promotion and eviction) and the stager (pad
publish and pad eviction) — and an entry's next value depends on state both
own (an expert evicted from VRAM falls back to its pad copy, if it has one). A
single `Mutex<Residency>` serialises every entry transition and holds, per
`(row, expert)`: VRAM slot (if any), pad slot (if any), warm slot (if pinned),
and the entry's current value. It is never touched by the forward thread and
never held across a CUDA call or a pack read. The entry's value is a pure
function of the three locations: VRAM if present, else pad, else pinned warm,
else 0.

**Transitions** (all under the lock):

| Event | Who | Entry becomes |
|---|---|---|
| promotion copy observed complete | pipeline | VRAM address |
| VRAM eviction | pipeline | pad address, else warm address, else 0 |
| stager publishes a staged expert | stager | pad address (only if the entry is not VRAM) |
| pad eviction | stager | warm address, else 0 (only if the entry is the pad address) |

### 0.5 Reclaim — when a slot's bytes may be overwritten

Retargeting an entry away from a slot does not free the slot: a kernel that read
the old address may still be reading it. A slot (VRAM or pad) is reclaimable
once every kernel that could hold its old address has finished. The kernels
that read row `r`'s entries are row `r`'s `bucketize`, gate, up and down, all
enqueued before row `r + 1`'s `bucketize`.

**Tickets.** Every invocation gets a ticket (`seq + 1`). The forward thread
records each row's latest ticket, then a full fence, *before* it enqueues the
row's `bucketize`; the host threads record the highest ticket whose summary word
they have seen. A summary word is `bucketize`'s last store and the compute
stream is in order, so observing ticket `T` means every kernel of every ticket
below `T` has completed (`expert_lre::reclaim`).

**Rule R.** Two rules, both on tickets:

- **Retargeting an entry to another address is allowed at any time.** The old
  slot is reused once the row's ticket, read *after* the retarget's stores and a
  full fence, is below the observed ticket — otherwise it goes on a retire list
  keyed by that ticket and is freed when a later word arrives. The two fences
  (forward: store ticket, fence, launch; host: store entry, fence, load ticket)
  guarantee that either the new invocation's `bucketize` saw the new value or
  its ticket is in the key.
- **An entry goes to 0 only while its row is quiet** — no invocation of it in
  flight (row ticket below the observed one, checked before the store). A cold
  expert's workers read the live entry, so a 0 written under an invocation that
  had already classified the expert cold would never be undone. An invocation
  enqueued between the check and the store is harmless: its `bucketize` reads
  either the old address (into its snapshot; the slot waits on the retire key)
  or 0 (cold; the stager stages it).

In practice the pipeline thread evicts only from quiet rows (rows behind the
GPU in this pass, and rows the forward thread has not yet reached), so its
victims are reusable at once and the retire list is the rare race.

**Rule R′ — the stager's demand window.** While row `n` has a cold tile the
stager has not yet released, the GPU is inside row `n`'s gate launch and cannot
be past it: every row before `n` of this pass has finished, and every row after
`n` (and every row of an earlier pass) has no kernel running. So during a demand
row any slot not holding one of row `n`'s routed experts is reclaimable at once —
including rows ahead of `n`, whose entries are retargeted before any of their
kernels exists. The window closes when the stager publishes row `n`'s last cold
expert, so it takes every victim it needs before that publish.

Neither thread ever waits for a reclaim on a path a GPU wait depends on.

### 0.6 The stager thread — NVMe and pageable reads into the pad

One dedicated thread owns the pad and is the **only reader of the pack**. It
replaces `ColdStaging` and every pack read on the pipeline thread
(`load_expert`, `load_experts_batched`, `pipeline.rs:1627, 1734`) and the
streamer's own pack reads (`streamer.rs:195-256`).

**It never calls CUDA on the demand path.** Its work is: poll mapped summary
words, read the pack (Windows `ReadFile`, Linux `pread` — `direct_io.rs`), copy
pageable warm slots with `memcpy`, and store to the mapped live table. That is
what makes a cold tile's wait driver-free.

**Per routed row `n`**, in order:

1. Poll summary word `n` (the same mapped word the pipeline thread polls; two
   readers are fine).
2. Read the summary's **cold set**: experts `bucketize` found with a zero gate
   entry (summary bit 30, §0.9). Ascending expert id — the order `bucketize`
   gives the cold tiles, so the first staged expert releases the first waiting
   tile.
3. For each cold expert: take a pad slot (free, else a reclaimable victim, §0.8);
   fill it — pack read into the slot (4 KiB-aligned, one stride, the pad's slot
   stride is the pack stride), or `memcpy` from its pageable warm slot; then
   publish (under the `Residency` lock): up, down, fence, gate.
4. **Publish each expert as its read lands**, not after the batch: reads are
   issued to a small pool of persistent reader threads, one `DirectFile`
   handle each (`read_at_with_handle`, `direct_io.rs:232`), at a queue depth of
   `NVME_QD` (8 by default, sized from the drive — "QD8 saturates" the
   dev-box Gen4, `direct_io.rs:66`). Completions publish in order of
   completion. `read_stripes_concurrent`'s per-call thread spawn and all-or-
   nothing return (`direct_io.rs:265`) are not used on this path.
5. Speculative staging requests (pipeline thread → stager channel) are served
   only when no demand row is outstanding, and are cancelled if their row has
   passed.

**Demand before everything.** A demand row is serviced to completion before any
speculative read is issued; speculative reads already in flight are allowed to
land (at most `NVME_QD` of them, ~1–2 ms).

**Gate projections first** is a later refinement: the gate launch needs only
gate projections, so splitting each record into gate / up+down reads would
release the gate sooner. The record layout allows it (`PROJECTION_ALIGN = 256`
offsets; a 4 KiB-aligned split needs the up offset rounded to 4 KiB in the pack,
a format change) — not in this revision.

### 0.7 The pipeline thread in revision 2

Kept: per-row stats and scores, the Markov transition matrix
(`transition.rs`), prediction-precision accounting, the boundary moves (§9),
profile snapshots.

Changed:

- **Promotion of misses is done by the workers** (§0.7.1): the expert crossed
  the link anyway, so the workers write it into a VRAM slot the pipeline thread
  provided, and the pipeline thread points the entry there once the invocation
  is done. No copy-engine traffic of its own.
- **Speculative promotion is read-ahead by the workers** (§0.7.3): the
  pipeline thread writes each row's predicted experts into the promotion ring,
  and a later launch claims offers for the predicted ones with a pinned copy
  the host vetted (warm, or a pad slot it pinned) with the link
  time its own misses leave; its gate launch's workers copy them into the slots
  and publish their entries. There is no copy-engine prefetch: a copy issued
  from the host could not be published before the next row's bucketize, so
  66–97% of them landed late, after the workers had fetched the same expert
  (Qwen3.8-Flash-Next, RTX 4090 Laptop, decode ×1–×8).
- **Prediction** (Markov chain over adjacent rows; a row routing most of its
  experts predicts the next rows' scored ones): a predicted warm expert goes
  into its row's read-ahead list; a predicted **cold** expert is sent to the
  stager as a staging request (pack → pad, ahead of the router).
- **Eviction** from VRAM retargets the entry to the expert's pinned or pad copy
  (or 0) and frees the slot by Rule R. Demand eviction (`demand_eviction`,
  `cache.rs:839`, Rule D) is deleted: demand never takes a slot.
- **No host wait on the GPU anywhere**: `ColdStaging::acquire`'s
  `event.synchronize()` (`pipeline.rs:1118`) goes with `ColdStaging`; the
  summary wait is the mapped-word poll.
- **Whole-layer streaming** (`streamer.rs`) is deleted, thread and staging ring.
  What replaced it — the pipeline thread promoting the next row's scored experts
  for a wide row — survives only for decode-width launches that route most of a
  row (many sequences decoding together), where the link has room. A
  prompt-prefill launch's workers saturate the link and are not given
  speculation to compete with (§0.7.1).

### 0.7.1 Promotion by the workers

Measured on the qwen36 gate with copy-engine promotion of every miss: the
cold-start prefill (BF16×1, 3,050 misses) moved each missed expert twice — once
through the workers (~5.5 GB) and once more as a promotion copy (5.36 GiB),
both on the one 12 GB/s link — and ran at 787 t/s against the baseline's
1,287. The workers' copy is the one the layer needs; the promotion is the same
bytes a second time.

So the workers promote. The pipeline thread keeps a **promotion ring** in
mapped memory (`expert_lre::promo`) of VRAM slot **offers** — `u64 slots[cap] |
u64 log[cap] | u32 head | u32 tail | u32 marks[rows][E] | u32 reserve | u32
sweep | u64 victims[cap] | u64 retarget[cap][3]`, the host writing `tail`, the
offers, `reserve` and `sweep`, the device `head` and the log:

1. The pipeline thread offers slots at `tail`. **Every empty slot is offered**
   — the zone's free list plus the empty offers already standing, to half the
   ring — because an empty slot evicts nothing to stand there and a miss that
   finds one is promoted for the copy it made anyway; stocked only to the
   prediction, a 16 GB card's zone stayed ~1,400 slots short of full for a
   whole Qwen3.6-35B-A3B run. Past the empties the stock is **predicted**: the
   next two rows' non-resident experts at the share of experts the current row
   routed, ×1.25, at least 32, at most `2E`; the ring holds `4E` indices, the
   rest room for offers the device has taken and the host not yet collected.
   **The predicted stock is lazy victims, not evictions.** An offered victim is
   a resident expert — ranked by the eviction policy, rows with an invocation
   enqueued last — that **stays resident and hittable** until a miss claims its
   slot, named by its gate-plane index `row · E + expert` and the three entries
   (gate, up, down) it falls back to — exactly what an eviction would publish:
   its pad slot, else its pinned warm slot, else 0 (cold). **Every resident
   expert is offerable**, and the fallback is held for the life of the offer:
   a pad slot is pinned against the stager, which owns pad eviction (at most a
   quarter of the pad, so cold staging always has slots); a cold expert is
   marked *offered cold* in the residency, because a later launch that routes
   it after a claim reads its zeroed entries as cold before the pipeline thread
   has booked the eviction — the stager stages a cold expert only when its
   entry reads 0, so it books the eviction itself on the first cold summary of
   an offered-cold expert (`Residency::device_evicted`, idempotent with the
   pipeline thread's booking). Restricting offers to warm-backed experts
   starved Qwen3.8-Flash-Next's ring — its zone is filled largely through the
   pad — leaving up to 71,060 decode misses a config unslotted and the decode
   hit rate 6–13 points down, and evicting dearer experts when the cheapest
   were cold-only. An offered slot is the ring's: the host evicts, relocates
   and offers nothing standing in it until a claim or a withdrawal returns it. Stocking by eviction emptied every stocked slot ahead of need —
   a standing stock of 256 cost C10×16 1,692 misses where 32 cost 278 — which
   is the cost a lazy offer does not pay: an unclaimed victim never left.
   Without room (`PROMPT_ROOM`) the predicted decode stock is the reserve, so a
   prompt-only expert takes only stock past it. Stock past the target is
   withdrawn (`tail` set back) when no bucketize can be reading the ring —
   under the pass lock, with every begun invocation observed — an empty offer
   going back to the zone and a victim back to plain residency.
2. **A prompt-only expert takes only an empty offer** — a prompt fills the
   zone's holes and never evicts; when the next offer holds a victim it stays
   in scratch. Letting prompts claim victims cost Qwen3.5-35B-A3B's prefill
   15–18% (its hit rate 66% → 55%) while its decode gained 37–56%.
   `bucketize` decides first whether the launch claims at all: **a launch whose
   decode-scored claiming experts outnumber the `sweep` word is a sweep and
   claims nothing**
   — a prompt passing over the table, its misses left in its workers' scratch.
   The host sets `sweep` to the predicted stock (decode's target without room,
   all misses' with it), never counting empty slots in, so a cold prompt cannot
   claim victims because the zone had a hole. Decided per layer launch on the
   device from that launch's own misses — distinct missed experts, which is
   what crosses the link — so a decode row co-batched with a prompt costs at
   most a claim deferred to the next narrow launch. Otherwise it gives each
   remote expert, in list order, the next offer at `head` (`remote_dst[i]`,
   else 0 when the ring is empty). A victim **of this row that this launch
   routes** is skipped — one of the launch's own tiles reads it — logged with
   the expert `PROMO_SKIP` and handed back; the miss takes the next offer. A
   claimed victim's entries are retargeted to its warm copy — up and down, then
   gate — before the launch's workers write a byte into the slot. Every launch
   enqueued earlier has completed in stream order, and every later one reads
   the entries fresh in its own bucketize, so no tile reads the slot under the
   victim's name again. No fence per claim: nothing reads a VRAM victim's entry
   concurrently, and the host learns of the claim only through `head`, which
   is published behind a system fence — a fence per claim would stall the
   serial claim loop on a PCIe drain for every victim of a decode launch. It logs `summary_word << 32 | row << 16
   | expert` at the offer's index and stores `head` before the summary word.
3. A worker whose expert has a slot copies every 16-byte unit to the same
   offset of the slot's projection (`dst_offset` per launch) **instead of** its
   scratch, and computes the row tile from there, as a resident expert's tile
   is computed (§0.10, worker step 4): the promotion costs no write the compute
   did not already need. After the gate, up and down launches the slot holds
   the whole slot image. A sweep's workers copy into scratch and the zone is
   left to the experts decode reuses; nothing re-sends those misses afterwards
   (step 6). Measured against copying every miss to both scratch and slot, with
   the copy engine promoting prompt misses into empty slots: decode +9.4% on
   Qwen3.6-35B-A3B's ladder and +5.8% on Qwen3.5-35B-A3B's (RTX 4090 Laptop,
   alternating binaries), prefill within 0.5%.
4. The pipeline thread collects the log up to `head`, in log order. A skipped
   offer comes back with its expert still resident. A claimed victim's eviction
   is booked at once — the device already made it — ahead of any later claim
   that could promote the same expert again. Each claimed slot lands once its
   invocation's ticket (from the logged word) is below the observed one — the
   invocation, all three launches, has completed: install the views, set the
   entry to VRAM. An expert promoted twice (two invocations claiming it before
   the first landed) keeps one slot; the other goes back to the zone, never
   having been named by any entry.

5. **One promotion in flight per expert.** `bucketize` sets the expert's mark
   (`u32 marks[rows][E]`, mapped, after the counters) when it hands out a slot
   and skips a marked expert; the host clears the mark when the slot lands or
   is dropped. Without it a prefill that visits a row again before the host
   has landed the first visit's slots promotes the same experts twice.
6. **A miss that took no slot is not re-sent.** Its workers computed it from
   scratch, and the next launch that routes it claims an offer. A copy-engine
   promotion of it would be a second crossing of the link for bytes the workers
   had just read: on Qwen3.8-Flash-Next (16 GB card) re-sending a prefill
   launch's decode-scored misses was 0.6–1.1 GiB per prefill row, read while
   the launch's workers saturated the link, and prefill ran 2–10% behind
   workers that promoted into ring slots once. What else crosses the link is
   read-ahead (§0.7.3), within the link time a launch's own misses leave.

7. **The owner check** (`tensor-assert`). Each zone slot carries a mapped tag of
   the expert last installed there (`expert_lre::slot_owners`), written on an
   install or a relocation's destination and never on an eviction — an
   eviction retargets the entry while a bucketize may already hold the old
   value, so clearing the tag then would race the checker against its subject.
   Bucketize checks every VRAM entry it snapshots for a GEMM against its slot's
   tag and traps on a mismatch, naming the slot: a tile about to read one
   expert's weights under another's name is caught at the read.

A boundary move drains the ring with the device synchronized: lands what was
taken, takes back what was not — an empty offer to the zone, a victim to plain
residency — and sets `tail = head`.

**Worker count by launch width.** A worker's slice copy is a few PCIe round
trips followed by its compute, so one worker is latency-bound and the link fills
only with many. Decode launches take 32 (`dispatch::WORKERS_DECODE`): on
Qwen3.8-Flash-Next (RTX 4090 Laptop) 8 → 16 raised decode by a third (×8 68 →
89 t/s, ×1 22 → 29), 32 a few percent more, and 64 measured level with 32;
eight loads in flight per thread instead of four measured level too. (The
§0.10.1 measurement of `W` = 8 saturating the link was a pure copy with no
compute between slices.) A prefill launch's remote
experts carry several token tiles each, all computed by the workers, so 8 made
the cold prefill compute-bound: 814 t/s at 8, 1,083 at 32, no better at 64 —
while every surplus worker is a block launched to exit on every decode launch
(single-context decode ~123 → ~117 t/s at 64). So launches over more than 64
tokens take 64 workers, the rest 8 (`dispatch::workers_for`). 64 rather than 32
for the model whose misses are most of a launch: Qwen3.8-Flash-Next at ×8–×16
prefill (a working set 3× the zone) ran C10 ×8 at 969 t/s with 32, 1,031 with 64,
1,044 with 128 — and 128 cost the ×16 decode rows ~4%. The workers take their
own grid rows ahead of the tiles (`⌈W / row_tiles⌉` of them): a single worker row
set the grid's width for every tile row, which at 128 workers over a 16-row-tile
projection launched 112 blocks per tile row only to exit.

**No speculation on a prompt-prefill launch** (over 256 tokens,
`dispatch::PREFILL_LAUNCH_TOKENS`). Its misses are pulled by enough workers to
saturate the link, so a speculative copy for the next row takes bandwidth from
them and saves nothing — late, and the workers moved the same bytes; on time,
and they would have moved them at the same cost. On Flash-Next ×16 prefill the
look-ahead issued 40,823 copy-engine promotions, 52.6 GiB, nearly all late:
845 t/s with it, 1,075 without. A many-sequence decode step leaves the link room
and keeps it.

### 0.7.3 Read-ahead by the workers

One copy engine moves experts into VRAM: the promotion ring and the GEMM
workers. Besides a launch's own misses (§0.7.1) it reads ahead for later rows,
with no copy issued by the host and no landing waited for.

1. **Predictions.** When the pipeline thread serves row `r`'s summary it
   predicts rows `r + 1 … r + 5` (`transition::HOPS`), each hop's candidates
   in order of evidence: the **router look-ahead** — a decode launch's stacked
   router projection also applies rows `r + 1 … r + 5`'s routers to row `r`'s
   input (their own stacked launches of at most three, `forward_parts`), and
   `moe_predict_votes` counts, per hop, the tokens whose top `k` names each
   expert into a mapped vote ring beside the summary (`votes`), which the
   pipeline thread copies out with the summary — then, for a narrow launch, a
   direct Markov table from row `r`'s own decode routing to row `r + h`'s
   arrivals (one table per hop, learned from decode launches only), then the
   target row's non-resident experts decode routed recently; up to 32 experts
   a row (rows `r + 1, r + 2` only stage — too near to read ahead). Measured
   look-ahead precision by hop 1 → 5 on Flash-Next: 65/54/43/39/34% at ×1,
   84/75/64/60/55% at ×8. Each row's prediction, filtered
   to experts whose device-readable copy is pinned host memory — a **warm**
   slot, or a pad slot the stager has staged it into — replaces
   that row's list in the ring (`u32 window | u32 depth | u32 ahead_n[rows] |
   u32 ahead_list[rows][AHEAD_CAP]`, after the retarget entries) — while a
   launch that reads the list has yet to begin: a launch at `L` reads the lists
   of `L + 2 … L + depth`, so a row's last reader is the launch two rows before
   it, and the list is written only while that row lies past the served row
   and the device has begun no invocation of it this pass — its started word
   is at most the served ticket (`read_ahead::listing_readable`). Predicted
   cold experts go to the stager, which stages them into the pad ahead of
   their row and reports each as it lands (`landed_ahead`); the pipeline
   thread, which drains the reports while it waits for the next summary, then
   lists the row again from its latest prediction, now with the staged expert
   from the pad, while a launch can still read it.
   Each listed expert carries the slot image the host **vetted** as its source
   (`u64 ahead_src[rows][AHEAD_CAP]`, after the lists): a warm slot as it is,
   since it never changes; a pad slot only once the host has **pinned** it for
   the listing (`ahead_pins`), since the stager may otherwise evict and reuse it
   under another row's launch — its own reuse rules (Rule R′, the reclaim
   clock) guard the pad's row's invocations, not the reader's. A relisting
   keeps the pins it lists again; a pin it drops is released once every
   invocation that could have read the old listing has finished — the latest
   invocation begun on any row when the list was rewritten (`readers_key`: the
   host writes, fences, reads the started words; a bucketize stores its started
   word before it reads a list). Read-ahead pins hold at most an eighth of the
   pad, beside the lazy victims' quarter.
2. **The window.** A launch at row `L` reads ahead for rows `L + 2 … L +
   depth` (wrapping into the next pass), spending `window − n_remote` offers,
   at most `AHEAD_MAX` (64): `window` is slot images the link moves in one
   layer at a share of ½ — the link's rate, measured at startup by a pinned →
   VRAM copy on the copy stream (`link_rate`, ~12 GB/s on the 4090 Laptop's
   Gen4 ×8, ~half that on the 3090 box's PCIe 3.0 host), times the median
   interval between consecutive rows' summaries over the previous pass. The
   share bounds the feedback of read-ahead lengthening the layers its window is
   measured from: with `a` read-ahead images, `n` misses, base layer time `T0`
   and image time `c`, `a = (s·T0/c − n)/(1 − s)`, finite for any `s < 1`. A
   prefill launch's misses use up its window; a sweep reads nothing ahead.
3. **The claim** (bucketize, thread 0, after its own misses' walk): for each
   listed expert whose three entries still point at its vetted image, in
   pinned memory, and whose mark is clear, take the next offer exactly as a
   miss does — a victim of this row that this launch routes is skipped, any
   other victim's entries retargeted first — set its mark, log it with its
   **target row** and `AHEAD_FLAG` set in the expert field, and write an item
   (`moe_read_ahead.cuh`): its image, the claimed slot, the image's bytes, the
   three entries' addresses and the values to publish, and the slot's owner
   tag. An expert whose entry has moved since it was vetted — staged into the
   pad after a warm listing, promoted, evicted — is passed over: the device
   reads only what the host vetted. A list read mid-rewrite may pair an expert
   with the source listed beside the one it replaced; a source still readable
   through a list is a warm slot or a pinned pad slot, neither of which can
   come to hold another expert, so the mismatch is passed over the same way.
4. **The copy.** The gate launch's workers — the layer's first launch over its
   remote experts, whose workers wait on its cold experts while the link is
   free — take the items' `AHEAD_CHUNKS` (16) pieces after their own items.
   A piece is a byte copy of the warm image into the slot; no layer geometry,
   so a row of another dtype or layout is copied the same way.
5. **The publish.** The worker that counts an item's last piece writes the
   slot's owner tag (with the owner check), the up and down entries, a system
   fence, then gate — the host's order. Every later bucketize then finds the
   expert in VRAM; nothing reads the target row's entries concurrently (later
   rows run after this launch in stream order).
6. **The host books it** when it collects the claim from the log: like a
   miss's claim, the victim's eviction at once and the slot's install once the
   invocation is done, when `set_vram` publishes the value the device already
   wrote. The row's routing judges the prediction (precision). A boundary move
   drains the ring with the device synchronized, as for any claim.

Measured on Qwen3.8-Flash-Next (RTX 4090 Laptop, 16 GB, warm tier 26% of the
experts, 2026-10-08): every config passes; decode read-ahead claims 3,700–4,600
per config at 61–90% precision, with windows of 16 (×1) to 64 images per layer.
Decode throughput is level with the copy-engine prefetch it replaces (×8 64.3
t/s): volume, not timing, limited it — the chained Markov predictor proposed at
most eight experts a row, about 1.5 claims a layer against 26 misses, which is
what the per-hop tables, the look-ahead and the recency candidates above widen.

**What read-ahead buys, measured (2026-10-09, same card, 32 workers).** The
copies run inside the gate launch, on the link the launch's own misses are
already using, so an image read ahead costs its link time now and saves it
later only if its row routes it — at 30–55% precision that is a loss unless the
link would otherwise be idle (a worker waiting on a cold expert). A share of 1
(a window twice today's) raised the decode hit rate to 73% and cut decode to
×1 19 / ×8 80 t/s; a share of 0 (no read-ahead) decoded ×1 28.3 / ×8 95.7
against ×1 27.4 / ×8 94.8 at ½ — level within this laptop's noise. The share
stays at ½; the copies that would pay are ones placed in idle link time only.

### 0.7.4 Victim ranking — evict behind the wave, by expected miss cost

Every eviction — a demand miss's claim and a read-ahead copy's alike — takes a
lazy victim the pipeline thread ranked into the promotion ring, and the device
retargets the victim's entries before it writes the slot (§0.7.1), so *safety*
never depends on where the victim sits. *Which* victim is taken does. The
ranking (`ExpertCacheInner::rank_victims`) supersedes revision 1's Rule S
(§6.2):

- **Behind the wave, fading, with the wrap.** Each resident's frequency is
  scaled by how far behind the wave its layer lies, `f + (1 − f)·(n − dist)/n`
  for wrapped forward distance `dist` and floor `f = 0.25`
  (`cache::behind_protection`, `cache::FADE_FLOOR`). L−1, the layer just
  executed and the one whose next use is a full pass away, is protected least;
  each layer further back a little more; the fade wraps, so at the start of a
  pass the layers just behind are the previous pass's last, and as the wave
  advances the eviction front moves onto the early layers it has left. The
  layers about to run keep nearly their whole score, so a read-ahead copy for
  a row ahead is bought from the layers the wave has left.
- **Distance steers, frequency decides.** The fade spans about `1/f` (4×): a
  hot expert just behind outlasts a cold one about to run, because the next
  pass needs it. The full fade (`f = 0`, `n×`) evicted exactly those and raised
  the share of victims routed again at their next visit by 3–4 points at ×8
  (`FADE_FLOOR` records the measurement).
- **Passes by eviction class.** The residents fall into four classes — reload
  tier (warm-backed or pack-only) × whether the row routed the expert on its
  last decode visit (`ExpertCacheInner::mark_visit`) — and the ranking runs
  over one class at a time, a later class only for the shortfall. Decode
  routing carries over step to step — 37–63% of a row's just-routed experts are
  routed on its next visit, against 10–21% of the rest — which a decayed
  frequency score cannot see: ranked by it, just-routed experts were 54–82% of
  all victims.
- **Cheapest expected miss cost first** (`ExpertCacheInner::set_eviction_costs`):
  each class's running regret rate (`regret::ClassRates`, the share of its
  victims routed again at their row's next decode visit) times its restore
  cost — the copy over the link for a warm-backed expert, and the measured
  pack read the layer then waits on plus that copy for a pack-only one.
  Restoring a pack-only expert costs about an order of magnitude more, so
  pack-only classes are protected by that ratio and evicted only when that
  much less likely to come back. Putting recency above the reload tier
  outright halved regret at ×1–×4 (25 → 15%, 37 → 18%) but turned cheap warm
  misses into pack reads (+23–58%) and decoded no faster; the measured costs
  decide that trade per class, per width, per drive. Repriced at every pass
  boundary; until a pack read is measured the passes run warm-backed first,
  stale before just-routed. The pinned head layers are never candidates.
- **Learned from decode only.** A prompt launch routes most of a row, so it
  neither judges claims nor marks a visit: it would count nearly every victim
  routed again and mark nearly every resident just hit.

Tests: `cache.rs` (fade values, the walk backwards from L−1, the front
wrapping with the wave, frequency within the fade, recency passes, cost-ordered
passes), `regret.rs` (judging by ticket, class rates, pricing, tags).
Measured (`regret`'s counters in the gate's decode table): regret at ×8 55 →
50% with the recency passes, BF16×8 decode ~+7%.

### 0.7.2 Measured — the qwen36 gate (RTX 3090, PCIe 3.0, WDDM)

`test_parallel_batched_forwarding_36_35b`, prefill / decode t/s, every session
matching, against the baseline at `a9889aeca` (host-readback dispatch):

| Config | Baseline | Revision 2 |
|---|---|---|
| BF16×1 | 1,287 / 34.4 | 1,384 / 48.3 |
| BF16×4 | 3,494 / 342.7 | 4,175 / 404.7 |
| Q8_0×1 | 3,513 / 72.8 | 4,270 / 101.2 |
| C0–C7 ×1 | 3,397–3,578 / 96–107 | 4,225–4,274 / 99–124 |
| C8×5 | 4,271 / 410 | 5,128 / 473 |
| C9×2 | 3,791 / 189 | 4,766 / 229 |
| C10×8 | 3,771 / 476 | 4,955 / 624 |
| C10×16 | 3,174 / 738 | 4,044 / 877 |

Two runs, reproducible to within noise (±1 % on the BF16×1 row; single-context
decode rows occasionally dip ~20 % in one run). The cold-start prefill (BF16×1)
moves ~3,400 experts over the link — the cache's first contact with the model —
and is within ~10 % of the link's floor for those bytes.

### 0.8 The pad

**A mutable pinned tier with LRE eviction**, owned by the stager. Slot images,
one pack stride each, allocated once at startup in one `cuMemAllocHost` block and
booked as pinned host memory like the warm tier
(`note_host_pinned_alloc`, `pinned.rs:470`).

**Size: one full layer of experts at least** (`E × stride`): a layer's cold set
is then always stageable without reusing a slot of the same row, so the demand
path never needs a "consumed" handshake from the GPU. Above that, the pad is
cache: every additional slot keeps a cold expert pinned-readable for its next
use. It comes out of the same pinned budget as the warm tier
(`warm_sizing_from`, `handle.rs:224`) — the warm tier is sized after the pad.

| Machine / model | One layer | Notes |
|---|---|---|
| 3090 (64 GB), Qwen3.6-35B | ~0.6 GB | every evictable expert fits the warm tier already (`docs/performance.md:1037-1040`): the pad sees no demand here |
| 4090 Mobile (32 GB), Flash-Next | ~E × 14.2 MB | warm tier covers ~30 % (`docs/expert_cache_design.md:822-836`): the pad is the hot set of the other 70 % |
| any, DeepSeek-V4-Flash | ~3.4 GB | 147 GB of experts: the pad is the main cold-miss cache |

**Eviction (LRE).** The stager keeps the pad's own score table: it reads every
row's summary anyway, so it credits each pad-resident expert the summary routes
to (decode rows +1.0, prefill rows +0.1 — `cache.rs`'s weights) and decays the
table by 0.85 at each pass boundary (`decay_scores`). The pipeline's VRAM score
table stays private to the pipeline thread, unlocked. Victims: on the demand path, any slot outside the current row's routed set
(Rule R′); for speculative staging, only a slot whose expert is also in VRAM —
an eviction that changes no entry — reused under Rule R's retire key. Never a
slot pinned by an in-flight promotion copy. A slot whose expert is also
VRAM-resident is the cheapest victim — its entry already names VRAM. A demand
row's cold set takes free slots first, then victims; with the pad at least one
layer, Rule R′ always leaves enough (the current row routes at most `E`
experts, and its cold ones are not in the pad).

### 0.9 The forward thread and `bucketize`

- **`bucketize`** classifies each *routed* expert by its entries: any of the
  three 0 → **cold**; else its gate entry against the two pinned allocations'
  fixed address ranges (warm tier, pad — both allocated once at startup, so the
  ranges never move, unlike the weight zone, whose boundary moves): inside
  either → **pinned**, anything else → **VRAM**. It snapshots the three entries
  (§0.4). The summary word per expert becomes `count | pinned << 29 |
  cold << 30 | decode << 31`; an unrouted expert's word is 0. Its kernel
  outputs are §0.10.
- **The token-tile width is chosen per launch, as the host tile builder chooses
  it** (`grouped_int8_n_sub`): Bm 32 at decode, 64 or 128 at prefill. The tile width
  is the grouped GEMM's weight-reuse factor, and it has to be decided without the
  routing readback, so it is read off `n_tokens·k / E` — what uniform routing gives,
  a lower bound on an active expert's rows and close to it at prefill widths, where
  nearly every expert is active. Bucketize builds `16·n_sub`-wide tiles and the three
  projections launch at that mode (one tile table between them, so the wide modes
  only where both dtypes are KO). Fixed at 32, every prefill re-streamed each expert
  2–4× per projection, which a model that had run the host path's wide tiles paid
  for directly: Qwen3-30B-A3B prefill at ×10 −11%, Qwen3.8-Flash-Next at ×8–×16
  −25–33% (RTX 3090).
- **No per-row VRAM bounds reach the device**, so a boundary move (§0.13) never
  races a classification.
- **Run-ahead bound.** The summary ring slot of invocation `n` is rewritten by
  invocation `n + RING`. The pipeline channel's bound covers the pipeline thread;
  the stager publishes a consumed sequence number (an atomic) and the forward
  thread holds invocation `n` until both readers have consumed `n − RING + 2`.
  This is a host-side check against counters; it waits on neither the driver nor
  the GPU's progress on any tile that could be spinning.
- Nothing else on the forward thread changes: no host wait on the MoE.

### 0.10 Kernel changes

The int8 impl (`grouped_matmul_impl_int8`, `kernel.cuh:2126` — STAGES 2, a
one-slot per-warp weight ring, static shared memory sized for occupancy) is **not
changed**. Everything below is in `bucketize` and in the grouped entry
(`quantized_matmul_grouped_entry`, `kernel.cuh:2569`) that calls the impl.

#### `bucketize`

- **Tile order: remote experts first, then VRAM experts.** A remote expert is a
  pinned or cold one (§0.9). Within each class, ascending expert id; the row
  layout (`tok_ids`, `perm`, …) stays ascending by expert id as now, so the
  scatter's canonical order and every output bit are unchanged (§4.3).
- **New outputs**: `remote[]` — per remote expert, `{expert, first_tile,
  n_tiles, cold}`, pinned before cold — and `header[3]` = the number of remote
  experts, `header[4]` = the number of tiles they own (the first tiles of the
  tile list). Sized `E` entries; written by thread 0 in phase 2 next to the tile
  prefix it already builds. And `snap[3][E]`, the routed experts' entries as
  read (phase 1b, after the histogram, one thread per routed expert).
- The summary word gains the pinned and cold bits (§0.9).
- **Promotion offers** (§0.7.1): thread 0 counts the launch's claiming experts
  against the mapped `sweep` word before its first claim, then, in list order,
  takes offers from `head`, skipping a victim of this row the launch routes and
  retargeting a claimed victim's three live entries before the slot is handed
  out. With an owner tag table (`tensor-assert`), phase 1b checks every VRAM
  entry it snapshots against its slot's tag and traps on a mismatch.

#### The grouped entry: `W` workers, then hits

The live launch's grid is `(row_tiles, worker_rows + launch_tiles)` with
`worker_rows = ⌈W / row_tiles⌉` — `row_fast = 1`, the order the live gate already
uses (`cuda.rs` `grouped_qmatmul_dev_q8a128`); grid rows `y < worker_rows` are the
**workers**, numbered row by row, and the rest are today's tile blocks with
`tile = y − worker_rows`. Blocks dispatch in linear order, so the workers lead the
grid — the order §0.10.1 showed overlaps and the reverse serialises.

- **A tile block** (`y ≥ worker_rows`) whose tile is one of the first `header[4]` (a remote
  expert's) exits at once — workers own it. Every other tile block is a hit: it
  reads its address from the snapshot (VRAM, a plain load) and runs the impl
  exactly as today. **Hit blocks never wait.**
- **A worker** (`y < worker_rows`, number `y·row_tiles + x < W`; the last worker
  row's blocks past `W` exit) loops over
  *items* `(remote expert r, row tile j)`, `n_items = header[3] × row_tiles`,
  pulled off a per-launch device counter:
  1. **Source**: thread 0 takes a pinned expert's address from the snapshot. A
     cold expert's it reads from the projection's **live** row
     (`MoeLive::live_row`) with a system-scope acquire load, spinning while it
     is 0: `__nanosleep` backoff to 32 µs, abort word and spin limit checked
     each poll; only workers ever spin. In the up and down launches the stager
     published the entry before the gate entry, so it is already there.
  2. **Mini loop**: all 128 threads copy row tile `j`'s slice — `K/128` pieces of
     `4 × chunk_bytes` (3,200 B for Q6_KO), piece `k` at chunk `k·(nrows/8) + 4j`
     of the source — 16-byte loads, four in flight per thread. With a promotion
     slot (`remote_dst[r]`) each unit lands at its own offset in the slot's
     projection, the source's layout; without one, in the worker's VRAM scratch
     slot, which then holds row tile `j` as a **32-row KO matrix**
     (`[K block][4 row groups]`).
  3. **One `__syncthreads()`**: the stores are visible to the block's own
     `cp.async.cg` reads (both through L2).
  4. **Compute**: for each of `r`'s `n_tiles` token tiles, call the unmodified
     impl — over a promotion slot with `weights = slot projection`,
     `nrows = nrows`, `row_tile_idx = j`, exactly a resident tile's call, which
     reads only the chunks this worker just wrote; over scratch with
     `weights = scratch`, `nrows = 32`, `row_tile_idx = 0` and `dst + 32j` — the
     impl indexes chunks `k·(nrows/8) + warp` (= the scratch layout) and stores
     `dst[token·dst_stride + warp_row_base + …]` (`store_tile_output`). Either
     way the output lands in row tile `j`'s columns with the same per-row
     arithmetic, bit for bit (`cuda_live_launch_*`, including a ring that runs
     short mid-list).
  5. `__syncthreads()`, next item.
- **`W` = 8** (§0.10.1–2): 8 workers reach the link's 12.2 GB/s; a ninth adds
  nothing and holds an SM slot. A launch with no remote expert pays 8 blocks
  that exit at once (a prefill launch 64 — see §0.7.1).
- **Scratch**: `W` slots of the largest `(K/128) × 4 × chunk_bytes` over the
  three projections — 51.2 KB for qwen36's gate/up, so 410 KB in all; one device
  allocation at load. The gate, up and down launches of a layer reuse it in
  stream order.
- **Up and down** use the same snapshot, workers and scratch, each projection
  copying its own slice: a remote expert crosses PCIe once per projection per
  layer, never once per token tile. A cold expert's up and down entries were
  published before its gate entry (§0.4), so they are set before the gate
  launch could finish, and they cannot go to 0 while the invocation is in
  flight (§0.5). Up and down launches never spin.

#### `MoeLive` and the gate launch

- `MoeWait` becomes `MoeLive { abort, live_row, remote, header, counter,
  scratch, slot_bytes, stall, spin_limit_ns, workers }`
  (`candle-kernels/src/quantized/moe_live.cuh`), read by workers only;
  `moe_wait_for_weights` (the per-block wait) is deleted.
- **Gate slicing** (`GATE_SLICE_BYTES`, `tile_offset`) is deleted: it existed so
  no launch outlived the display watchdog while *every* miss tile could wait. Now
  a launch waits only on cold experts in its `W` workers, bounded by the
  stager's disk time — a full DeepSeek layer of gate projections (~1.1 GB) is
  ~0.4 s at this drive's ~3 GB/s (§0.10.2), inside the 2 s limit with room; the
  per-wait spin limit (1.5 s) remains the backstop.
- **`row_fast` is fixed at 1** for every live launch, including up and down,
  which today pick their axis order for L2 (`grouped_grid_row_fast`). The cost
  for hit tiles is measured in acceptance (§0.16); the grid needs the tile axis
  on `y` for the worker rows.

#### Registers and occupancy

The worker branch adds a copy loop (four `uint4` in flight) whose live range does
not overlap the impl's, so the kernel's register maximum should not move; static
shared memory is unchanged (the worker reuses the entry's buffers; two shared
words for the item and the source). Both are checked against the current
build's `ptxas -v` numbers in acceptance, because occupancy is what the hit path
is tuned on.

#### 0.10.1 Measured (RTX 3090, PCIe 3.0 ×16, 2026-10-04)

`bench_grouped_gemm_weights_from_pinned_host` (`candle-core/src/quantized/cuda_tests.rs`):
the **unmodified** grouped GEMM, qwen36-35B gate shape (512 × 2048, Q6_KO,
819,200 B per expert), with table entries holding the host addresses of
`cuMemAllocHost` copies.

| Question | Result |
|---|---|
| Can the GEMM read `cuMemAllocHost` memory at its host address (`cp.async` from host memory)? | **Yes — bit-identical to the VRAM run** |
| GEMM read rate from pinned memory, 16 → 1024 blocks | **5.0–5.5 GB/s, flat** |
| …at 1 / 2 / 4 / 8 / 16 blocks (32-row experts) | 2.17 / 4.30 / 4.99 / 5.28 / 5.42 GB/s — **saturated by 4–8 blocks** |
| A plain reduction over the same pinned memory | 2.20 GB/s |
| Copy engine H2D, same memory (bulk / one 0.8 MB copy) | **12.75 GB/s** / 5.2 GB/s |
| One pinned expert across T token tiles (T = 1, 2, 4, 8) | time × 1.00, 2.01, 4.02, 7.96 — **L2 does not absorb re-reads of host memory** |
| VRAM tiles (151 µs) + pinned tiles (1217 µs) in one launch, pinned first | 1325 µs (sum 1368, max 1217) — overlapped, ~100 µs not hidden with 128 pinned blocks resident |
| Same, pinned tiles last | 1338 µs — serial |
| VRAM GEMM rate, 64 experts | ~770 GB/s |

Then `bench_pinned_host_read_bandwidth` (same file): probe kernels compiled at
test time (NVRTC) reading a 256 MiB `cuMemAllocHost` buffer.

| Probe | GB/s |
|---|---|
| Flat 16 B loads, 256 threads/block, 4 in flight — default / `.nc` / `.cg` / `.L2::128B` / `.L2::256B`, 32 → 1312 blocks | **11.7–12.75** (all variants, from 32 blocks) |
| Same, base misaligned by 16 / 32 / 64 / 96 B | 11.35–12.09 |
| One 800 B chunk per warp via `cp.async` (the GEMM's unit), not / 128 B-aligned, any hint | 10.7–11.0 |
| Same with 896 / 1024 B chunks | 12.2–12.6 |
| The GEMM's exact address walk (row tile × K block, 800 B per warp), 1 or 4 K blocks in flight | 10.6 |
| **Wide copy pinned → VRAM, 128 threads/block, 4 loads in flight/thread, `W` = 4 / 8 / 16 / 32 / 82 blocks** | **12.77 / 12.43 / 12.27 / 12.29 / 12.05** |
| Wide copy of one projection (819,200 B), `W` = 4 … 32 | **~69 µs** (≈ 11.9 GB/s) |
| Copy engine | 12.81 |

What follows:

- **A kernel pulls pinned memory as fast as the copy engine** (12.3–12.8 GB/s)
  with as few as 4 blocks of 128 threads. Request size, cache hints and
  alignment are second-order (≤ 12 %).
- **The GEMM pulling its own weights reaches only ~5.3 GB/s**, half what the
  same address walk gets as a bare probe (10.6). The loss is in the GEMM's
  per-K-block structure, not the access pattern; it is not root-caused, and it
  does not need to be: the miss path does not pull through the GEMM.
- **So a miss is a wide copy into VRAM, then the unchanged GEMM from VRAM**:
  ~69 µs per 819 KB projection, ~200 µs per whole qwen36 expert — the copy
  engine's rate, with no driver call, no shared memory and no change to the
  hit path. `W` = 4–8 copy blocks; more only hold SM slots.
- **A miss spanning T token tiles crosses PCIe once** — the GEMM re-reads VRAM.
  (Pulled directly it cost T crossings: L2 does not keep host-memory lines.)
- **Pinned work must lead the grid**; trailing it serialises the launch.

#### 0.10.2 The worker launch, simulated end to end

`bench_moe_worker_launch_simulation` (same file): one kernel with the proposed
grid — `W` worker blocks first, pulling miss items `(expert, 32-row tile)` off a
counter; each reads the expert's mapped live-table entry (spinning gently while
it is 0), copies the item's slice (16 × 3,200 B) into its VRAM scratch slot with
wide loads, `__syncthreads`, then runs a **fake GEMM** with the real one's
access structure (per K block: `cp.async` 800 B per warp, wait, block sync; ×
`reps` token tiles) over the slot as a 32-row matrix. Hit blocks follow, one
tile each, fake GEMM over VRAM. Every item's checksum is verified against the
source bytes. Cold experts are read from a real file on NVMe
(`DirectFile`, unbuffered) by reader threads that make no CUDA call and publish
with a host store.

| Experiment | Result |
|---|---|
| A. Warm misses only, 1 / 2 / 8 / 32 experts, `W` = 8 | 76.8 / 143 / 535 / 2158 µs — **12.2 GB/s, ~67 µs per 819 KB projection** (`W` = 4: 11.0 GB/s; `W` = 16: no better than 8) |
| B. 32 VRAM experts (hits) + 8 warm misses, 4 token tiles, `W` = 8 | hits alone 121 µs, misses alone 661 µs; **together hits 129 µs (+7 %), misses 738 µs (+12 %), kernel 762 µs** — overlapped |
| C. + 8 cold misses from NVMe, reader QD 1 / 4 / 8 | published 0.55 → 2.7 ms at every QD (~3 GB/s from this drive at 819 KB reads); hits still 127–128 µs; kernel ~3.0 ms |
| D. Forward thread, while workers spin: 8 transfers queued behind, then **load + launch a module never loaded before** | the forward thread **blocked 29.2 ms — until the spinning launch ended**; the readers (no CUDA calls) published meanwhile; the launch finished and the late kernel then ran. **No deadlock.** |
| E. Hits with 8 workers spinning on cold entries for 30 ms | hits 126 µs vs 121 µs alone (+4 %) |

What follows:

- **The design works on this box as a whole**: workers copy misses at the copy
  engine's rate, hits overlap them at a few percent cost, cold experts flow
  from NVMe with no driver call, and the lazy-load hazard of §0.1 becomes a
  stall of the forward thread, never a deadlock.
- **`W` = 8.**
- **Disk throughput here is ~3 GB/s at whole-expert reads, independent of
  queue depth** — one 819 KB read already saturates the drive. `NVME_QD` matters
  for smaller reads; 8 costs nothing.
- **A spinning worker costs the hit tiles ~4 %**, at the 32 µs backoff cap.
- Not yet shown: the real `grouped_matmul_impl_int8` over a scratch slot as a
  32-row matrix (bit-identity and rate), and the real GEMM's occupancy with the
  worker branch present.

### 0.11 Profiling — stall versus overlap

The question §16 asked — *is the GPU waiting on bytes, or computing while they
move?* — is answered by three places, each recording what only it sees.

| Where | Counter (per MoE row, profile build only) | Answers |
|---|---|---|
| Workers (device, `MoeWait::stall`) | items, bytes copied, copy ns, compute ns, cold-wait ns, launches with a cold wait | how much of a launch the misses took, and how much of that was the disk |
| Gate / up / down launch (device clocks, as §0.10.2's `clocks`) | hits-done and misses-done time from launch start | **overlap**: misses-done ≫ hits-done means the misses are the layer's long pole |
| Stager (host) | cold experts staged, bytes, read µs (issue → land), publish lag behind its summary word | disk rate and latency per machine |
| Pipeline (host) | promotions, promotion bytes, promotion copy µs, prefetch / promotion hit rate | whether the async side keeps up |

The summary table prints, per config: hits / pinned misses / cold misses per
layer, worker GB/s, cold-wait share of MoE time, overlap ratio (misses-done ÷
hits-done), stager GB/s. §16–§17's counters for the waiting gate
(`stall_ns`, `wait_blocks`, …) are replaced by these; the plumbing they designed
(per-row device counters folded by the launch's last worker, read at snapshot)
carries over.

### 0.12 Failure and abort

- **The abort word** (mapped) is raised by the pipeline thread's and the stager's
  `DeadFlagGuard`s and by any error on either thread. A spinning worker sees it
  on its next poll and gives its expert up.
- **A pack read that fails** (I/O error, short read) is an abort, not a retry:
  the worker waiting on that expert cannot be released any other way.
- **The spin limit** (1 s per wait) is the backstop for a stager that is alive
  but stuck (a drive that stopped answering); a wait past it gives its expert
  up the same way.
- **Giving an expert up is a failed forward, not a trap** (`expert_lre::fault`).
  The worker claims a mapped fault word — aborted or timed out, the row, the
  expert, the microseconds waited — and skips the item, writing neither its
  output nor its promotion slot. Any other cold wait that sees the word set
  gives its expert up at once; the word is read only inside a wait, so an item
  whose expert is already readable pays nothing for it. Read-ahead pieces never
  wait on a cold expert and run to the end. `drive_wave` synchronises after
  every segment, residual or logits, and reads the word
  (`WaveSweep::take_device_fault` — required of every sweep, so a model gaining
  an expert cache cannot skip it); a fault fails the segment with an error
  naming the expert and rolls it back as a failed sweep, so a faulted segment
  never commits its KV or hands its residual on. Until the word is taken,
  nothing builds on the faulted forward: the dispatch refuses every launch
  (`FaultWord::check`; the wave driver takes the fault on that error path
  too, so a refusal never outlives the fault), and the pipeline thread lands
  no demand promotion — a given-up item's slot was never written. Taking the
  fault sends the drop of every pending demand promotion down the pipeline
  thread's channel and clears the word only once it is done, so a landing
  still queued behind the faulted forward never reads the word clear; a
  boundary move with a fault pending drops them as well, before the zone can
  retract under their slots. The stager's eviction stores the entry and then
  checks the row's retire key even under Rule R′, which a forward that
  faulted and moved on no longer satisfies; a demand whose `T` the GPU has
  left is abandoned. A trap was a sticky error that poisoned the CUDA
  context for the whole process; the context now survives, and the
  scheduler's retry is a retry.
- **A hit tile whose entry changed under it** (VRAM-evicted between `bucketize`
  and its block's start) still reads the VRAM address from the snapshot, and
  Rule R keeps that slot unreclaimed until the invocation is done.
- **The stager never blocks on CUDA**, so a dead driver context cannot wedge it;
  it exits on the abort word or when its channels close.

### 0.13 Boundary moves (§9)

The elastic weight/KV boundary moves only when quiet (no wave live, every
summary consumed — `reserved == served` under the pass lock, as §9). A
retraction retargets the conceded VRAM slots' entries to their pinned copies or
0 and frees the slots after a device-wide quiesce (`ctx.synchronize()`, as
`quiesce_before_handover` does) — allowed there because no gate can be waiting
on anything but the stager, which needs no driver. A growth adds free slots.
`bucketize` never sees VRAM bounds (§0.9), so neither move races a
classification. Relocation copies keep the copy engine and publish on observed
completion like promotion.

### 0.14 Porting

Every expert-cache MoE path already goes through `ExpertCache::forward_routed`
→ `Dispatch::forward` (qwen3-MoE / 3.5 / 3.6 via `SparseMoeBlock`, qwen4exp,
latent_moe / DeepSeek), so revision 2 changes no model file beyond what
revision 1 did. `dspark_experts` keeps its own all-resident VRAM tables (no live
table, no workers). The latent_moe readback counter test asserts zero
readbacks.

### 0.15 Tests (written first)

Kernel side (`candle-core`):

1. **`cuMemAllocHost` memory is device-readable at its host address** — done:
   `bench_grouped_gemm_weights_from_pinned_host` §1 (bit-identical).
2. **The impl over a scratch slot as a 32-row matrix** — copy one row tile's
   slice of a KO projection into a VRAM buffer, run the impl with `nrows = 32` and
   `dst + 32j`; output bit-identical to the normal launch's row tile `j`, for
   every KO dtype and for `T = 1, 3` token tiles.
3. **`bucketize` remote ordering** — raw expected `remote[]`, header, tile order
   and summary bits for a hand-built case with VRAM / pinned / cold experts
   (extends `cuda_moe_bucketize_live_table_orders_resident_first`); row tables
   unchanged against the all-resident reference.
4. **Worker launch, real impl** — the §0.10.2 simulation with the real entry:
   warm and cold remote experts, output bit-identical to an all-VRAM launch.
5. **Cold release with the forward thread in a module load** — §0.10.2 D with the
   real launch; must finish, bit-identical.
5a. **Read-ahead claims** (§0.7.3) —
   `cuda_moe_bucketize_reads_ahead_into_offers_within_the_window`: only the
   vetted (a pinned pad slot included), unmarked predictions of rows `L + 2 …
   L + depth`, in list order, within the window; a stale vetting passed over;
   a victim the launch routes skipped, another retargeted; every item word, the
   log entries and the marks raw; a window of 0 reads nothing.
5b. **Read-ahead copy and publish** — `cuda_live_gate_launch_reads_ahead_and_publishes`:
   images landed byte for byte, entries and owner tag published, every piece
   counted once, the launch's own output bit-identical to the all-VRAM one.

Host side (`candle-transformers`, host-only, raw expected values):

6. **`Residency` transitions** — every row of §0.4's table and the races (pad
   eviction against a VRAM eviction falling back to the pad; promotion from a
   pad slot being evicted).
7. **Rule R / R′** — reclaim behind / ahead of the observed row; the retire list
   draining on a later word; R′ reclaiming a row ahead during a demand row.
8. **Stager** against a real pack: a cold set staged and published in completion
   order, bytes equal to the record; pageable warm slots staged by `memcpy`; a
   one-layer pad serving a row whose whole routed set is cold.
9. **Pad scores** — credit, decay, victim order.
10. **Listing readability** — `read_ahead::listing_readable`: a row's list is
    written only while the launch two rows before it has yet to begin, judged
    by that row's started word against the served ticket, across a pass
    boundary too.
11. **A lost cold expert fails the forward, not the context** —
    `candle-core/tests/moe_live_abort.rs`: an aborted wait and a wait past the
    spin limit each claim the fault word (abort bit, row, expert, wait — raw
    bits) and leave the next synchronise and later work running;
    `expert_lre::fault` decodes the word into the forward's error.
12. **Router look-ahead votes** — `votes::ranked` (most votes first, ties by
    expert, zero votes left out), and `qkv_segmented_refuses_a_fourth_segment`:
    a stacked launch that does not split K takes at most three weights.
13. **Residency references** — `belady`, `routing_trace`: Belady's MIN and LRU
    over a recorded routing, raw hit counts, and the trace's file format byte
    for byte; `replay` replays scored policies over a written trace.
14. **Read-ahead pins** — `ahead_pins`: pad candidates pinned under the cap and
    dropped past it, a relisting carrying the pins it keeps, dropped pins
    released only once their readers' key has passed, a pending release still
    counting against the cap, and a row listing a pad expert from its commit
    until a relisting drops it.

### 0.16 Acceptance

- The qwen36-35B gate (`test_parallel_batched_forwarding_36_35b`) passes, **lazy
  loading on**, plain and `--features profile`, against the baseline captured
  at `a9889aeca` (`BF16×1` 1287 / 34.4 t/s prefill / decode … `C10×16` 3173.7 /
  737.6), every session matching.
- No regression on any row of the baseline beyond noise; the profile shows the
  overlap ratio and cold-wait share per layer.
- `ptxas -v` registers and static shared memory of the grouped entry unchanged
  against `a9889aeca`; hit-tile throughput with `row_fast = 1` forced on up and
  down within noise of today's choice.
- The full sweep (`/sweep`) green, including the Flash-Next gate and engine probe
  on the 16 GB card, where the pad and stager do real work.

### 0.17 Deleted by revision 2

`ColdStaging` and its event waits; the streamer thread and its staging ring;
pack reads outside the stager; demand loading, `demand_eviction` and Rule D;
the copy-engine table fills and clears and `PointerConstants`; the routing
stream, the summary DtoH and `summary_ready`; the `spec_clear` event and Rule
P's compute-stream wait (clears are host stores, visible at once); the device
abort word raised by a copy; `moe_wait_for_weights`; gate slicing
(`GATE_SLICE_BYTES`, `tile_offset`); the per-source prefetch fence ring
(`CopyBatchFence`, `FenceSource`); the evicting victim scans
(`allocate_slot`, `demand_eviction`, `evict_for_prefetch_batch`), replaced by
the non-evicting `rank_victims` the reclaim rule filters; and
`ExpertPack::read_many_unverified`. The pipeline is split by concern into
`slot_image.rs`, `startup.rs`, `boundary.rs` and `pipeline.rs`, beside the new
`reclaim.rs`, `residency.rs`, `pad.rs` and `stager.rs`.

Read-ahead (§0.7.3) deleted the copy-engine prefetch that preceded it: the
copier thread (`copier.rs`) and its promotions, `take_slots`, the retire list
the pipeline and the stager parked evicted slots on (`RetireList`), the
adaptive prefetch depth and its late-load and copy-bandwidth signals
(`adapt_prefetch_depth`, `late_loads`), and the `promotions` /
`promotion_bytes` counters.

### 0.18 Documentation to update

As §18's audit, plus: `docs/expert_cache_design.md` (warm tier no longer the
only pinned tier; demand misses no longer load), CLAUDE.md hot-path invariant 3
(the MoE routing readback is gone — propose the edit, do not commit it),
`docs/performance.md` (new rows from acceptance), and the module docs of
`expert_lre` (`pipeline.rs`, `dispatch.rs`, `streamer.rs` — deleted —
`live_table.rs`).

### 0.19 Decided in this revision (formerly open)

| Question | Decision |
|---|---|
| Kernel side | §0.10: workers in the same launch, mini-loop copy + one sync, the unmodified impl over a scratch slot |
| `W` | 32 for a decode launch, 64 over 64 tokens (§0.7.1) |
| Promotion of misses | by the workers into ring slots; a miss with no slot is claimed on its next launch, never re-sent (§0.7.1) |
| Prefetch | read-ahead by the gate launch's workers into ring slots, published by the device, within the link window a launch's misses leave; from host-vetted sources — warm slots, and pad slots pinned for the listing (§0.7.3) |
| Scratch | `W` slots of the largest projection slice — ~410 KB for qwen36 |
| Pad size | one full layer, the floor that never needs a GPU→host "consumed" signal; anything above it is a pinned-budget split left at zero in this revision |
| Promotion policy | every miss is a promotion candidate, admitted by the pipeline's existing score policy (today's behaviour minus the critical path) |
| Pad eviction scores | the stager's own table, from the summaries it reads |
| VRAM classification | against the pinned allocations' fixed ranges, not the moving zone |
| `NVME_QD` | 8 (one 819 KB read saturates this drive; 8 covers smaller reads) |

### 0.20 Residual risks

1. **The real GEMM's rate over a scratch slot** has not been measured — only the
   fake GEMM's (§0.10.2). Test 4 measures it; the impl is the same code that
   runs from VRAM today, so a large difference would itself be a finding.
2. **`row_fast = 1` for up and down** may cost hit tiles some L2 reuse; measured
   in acceptance. If it does, the worker row moves to the tile axis's end of a
   `row_fast = 0` grid instead — same scheme, different index arithmetic.
3. **Block dispatch order** is relied on for performance (workers lead), never
   for correctness: no block waits on another block.
4. **The disk on other machines**: the 3 GB/s here is one drive; the 16 GB box
   and DeepSeek will report their own through §0.11's stager counters.
5. **A display-attached GPU's watchdog** bounds a launch's cold waits at ~2 s
   total; the largest case (a full DeepSeek layer cold) is ~0.4 s here.

---

> **Sections 1–18 are revision 1.** They are kept because §0 builds on them and
> refers into them, and they remain correct for: the motivation (§1), the
> routing summary's content (§5.3), passes (§6.4), tile order not changing
> numerics (§4.3), the scatter, porting (§10), the boundary rule's quiet
> condition (§9), failure semantics (§8) and the documentation audit (§18).
> Everything they say about the demand path — copy-stream fills and clears,
> the waiting gate, the routing stream, Rule D, gate slicing, demand eviction —
> is superseded by §0.

## 1. Why

On every machine whose expert cache streams — the 16 GB and 24 GB cards for
every MoE model, and DeepSeek-V4 everywhere — each MoE layer of each forward
does this today (`quantized_qwen3_moe.rs:629-856`, `expert_lre/handle.rs:1266-1316`):

1. the forward thread enqueues attention, the shared expert, the router and
   `moe_route`, then **`e2.synchronize()`** (`quantized_qwen3_moe.rs:751`) —
   which, because the routing stream waits on the compute stream, drains the
   whole null stream;
2. it sorts assignments on the host, sends a `Work` request to the pipeline
   thread and **blocks on `recv`**;
3. the pipeline thread classifies, evicts, issues miss DMA, then enqueues the
   gather and grouped GEMMs itself, and answers.

The GPU is idle from the end of step 1 until the pipeline thread's first launch,
and can never be more than one layer ahead of the host. On Qwen3.8-Flash-Next the
profile puts `fwd_routing_wait` at 30.9% and `submit_roundtrip` at 14.8% of span
time (`docs/archived/qwen38_flash_next.md:1641`). DeepSeek does the same through
a synchronous `indices.to_vec2()` (`latent_moe/engine.rs:556`). Where every
expert is resident the device path already exists and removing this round trip
took decode from 26–28 ms to 18–19 ms per step
(`docs/archived/gpu_native_moe_dispatch.md:3-11`); that path is refused the
moment one expert is not resident (`handle.rs:713`, `:1036-1044`) and, for the
rest of the process, the first time the weight zone concedes ground
(`handle.rs:1665-1683`).

## 2. The shape

```text
forward thread (null stream)                      pipeline thread            copy stream
──────────────────────────────                    ───────────────            ───────────
router → route (top-k, device)
bucketize  ── reads live gate row, orders
             resident tiles first, writes
             routing summary[n]
record e_route(n); routing stream: wait,
  DtoH summary[n] → pinned[n], record C(n)
send Routed(n) ─────────────────────────────────▶ wait C(n)
gather                                            classify vs bookkeeping
gate GEMM  ── resident tiles run at once;         clear victims' gate entries ──▶ 8 B zero
              a miss tile spins on its entry      load misses ────────────────▶ record H2D
up GEMM, SwiGLU, down GEMM                        fill up, down, then gate ───▶ 3 × 8 B
scatter (unchanged, deterministic)                stats, transition, prefetch
next layer …                                      next message …
```

- **The forward thread never blocks on the MoE.** Every launch has a
  data-independent bound (as the device path already does,
  `quantized_qwen3_moe.rs:359-363`), so it enqueues the whole layer and moves on.
- **The GPU waits only on a real miss**, and only the tiles of that miss: hit
  tiles compute while the copy is in flight — the overlap the determinism fix
  gave up (`pipeline.rs:2556-2572`) comes back without touching determinism,
  because each tile writes its own rows and the scatter sums them in canonical
  expert order regardless of which tile ran first (§4.3).
- **The host is told, not asked.** The routing summary goes to the pipeline
  thread over the routing stream it already owns (Option A); the forward thread
  only records events and sends a message.
- **One path.** An all-resident cache is the case where every entry is non-zero
  and no message ever produces a load. A conceded zone is a few cleared entries.
  The host path, inline mode and the reader path are deleted (§10).

## 3. The live table

### 3.1 Layout

`GpuDispatchTables` keeps its three `[rows × n_experts]` u64 arrays
(`gate_ptrs`, `up_ptrs`, `down_ptrs`, `gpu_dispatch.rs:40-46`) and its per-row
dtypes. What changes:

- **Built for every cache**, at construction, over **every** MoE row the cache
  serves (`num_moe_layers`, which already includes the qwen4exp/qwen35 MTP head
  row, `qwen4exp/engine.rs:444-456`, `qwen35/quantized_weights.rs:340-351`).
  `min_layer` is 0. Per-row dtypes and shapes come from `layer_geometries`, not
  from resident slots, so a sparse grid is expected rather than refused.
- **A zero gate entry means "not resident".** The gate entry is the only one
  that is ever cleared and the only one anything waits on. Up and down entries
  are written before the gate entry on every fill (§3.3) and are never read for
  an expert whose gate entry was not observed non-zero first, so stale up/down
  values behind a zero gate entry are unreachable.
- **The build is a load-time contract, not a probe.** Every condition
  `GpuDispatchTables::build` checks today becomes an error from `ExpertCache::new`
  naming the model and the condition: KO weights in every row, uniform shapes
  across rows, one dtype within a row, `n_experts ≤ MOE_MAX_EXPERTS` (512),
  `k ≤ 16` (`moe_route`) / `≤ 32` (bucketize), N % 32 and K % 128 for both
  projections, compute stream is the legacy null stream
  (`gpu_dispatch.rs:358-369`). There is no host path to decline to.
- **The `hidden % 1024` condition is removed** (`gpu_dispatch.rs:241`). It cites
  "the q8a1024 byte-row gather", which is now tile-granular and serves any
  multiple of 128 (`candle-core/src/quantized/cuda.rs:7278-7281`); it is what
  refuses Flash-Next (hidden 2560) today.
- **Deleted with it:** `built_capacity`, `built_floor`, `built_concede_epoch`,
  `zone_moved` and the permanent refusal (`gpu_dispatch.rs:60-85, 567-586`;
  `handle.rs:1632-1697`). Their job — never dispatch through an address the zone
  no longer owns — is now done by clearing entries (§6), which is exact where the
  epoch was conservative.

### 3.2 Pointer constants — fills need no staging

A fill writes three u64 that are a pure function of `(slot, row geometry)`:
`slot_base(slot) + slot_offsets(geom)` (`pipeline.rs:749-762, 850-888`). They
are precomputed once into a **pinned, immutable** array
`PointerConstants[geom_class][slot] = {gate, up, down}` for every slot up to
`zone.limit()`, plus one pinned zero word. Every table write is then a
`cuMemcpyHtoDAsync` of 8 bytes **from an address that never changes**, so there
is no ring to manage, no event to wait before reuse, and no host write racing an
in-flight copy. Geometry classes are the distinct offset triples across rows —
two on the dynamically quantized 3.5/3.6 checkpoints (Q5_KO/Q6_KO down,
`gpu_dispatch.rs:93-107`), one elsewhere.

### 3.3 Writes

All table writes are copy-engine copies on a non-blocking stream — never a
kernel and never a memset (`cuMemsetAsync` can be implemented as a kernel, which
cannot be scheduled while spinning blocks hold the SMs).

| Operation | Stream | Sequence |
|---|---|---|
| **Fill** expert `e` of row `r` into slot `s` | copy stream (demand, prefetch, relocation) or streamer stream (whole-layer stream) | record H2D into `s` → `up[r,e]` → `down[r,e]` → `gate[r,e]` |
| **Clear** the tenant `(r', e')` of slot `s` | copy stream | `gate[r',e'] ← 0`, **before** any byte of the new tenant is written to `s` |

Copies on one stream execute in order, so the gate entry cannot become non-zero
before the expert's bytes and its up/down entries have landed.

`cudarc` 0.17.8's `CudaContext::new_stream` creates `CU_STREAM_NON_BLOCKING`
streams (`core.rs:454-469`), so the copy, routing and streamer streams do not
implicitly serialize with the null stream: a fill proceeds while a null-stream
kernel spins.

## 4. Kernel changes (existing kernels only)

### 4.1 `moe_bucketize` (`candle-kernels/src/simple/moe_bucketize.cu`)

Three new arguments: `const uint64_t* gate_row` (the live gate table at
`row × n_experts`), `int decode_tokens`, `uint32_t* summary`.

1. **Phase 1** (`:110-120`): alongside the histogram, a grid-stride
   `ld.volatile` of `gate_row[e]` into a new `__shared__ uint8_t sh_res[512]`
   (+512 B, ~40.5 KB of the 48 KB static cap), and an `atomicOr` of a decode bit
   into `sh_dec[e]` when assignment `i` belongs to a token `< decode_tokens`.
2. **Phase 2** (`:123-147`, thread 0): `sh_offsets` — the **row** layout — is
   computed exactly as now, ascending expert id. `sh_tile_pref` — the **tile**
   order — takes two passes: resident experts with tokens first, then the rest,
   each ascending. `num_tiles` is unchanged.
3. **Summary**: `summary[e] = count | (decode << 31)` for every expert, written
   grid-stride after phase 1. 512 words — 2 KB at the widest model.
4. Phases 3–5 (`tok_ids`, `weight_ids`, `perm`, `rw_ids`, `token_starts`) are
   untouched: the row layout is identical, so the gather, the GEMM rows and the
   scatter are bit-identical to today's.

The kernel stays one block of 256 threads on the compute-stream handle
(`:317-346`). It is not timing-sensitive: a stale read only reorders tiles (a
zero read for an expert that has just been filled makes it a spin that ends at
once; a non-zero read cannot be stale — §6.2).

### 4.2 `grouped_qmatmul_dev_q8a128` — the wait

In `quantized_matmul_grouped_entry` (`candle-kernels/src/quantized/kernel.cuh:2468-2532`),
between `expert = tile_expert[tile]` (`:2493`) and the pointer dereference
(`:2494-2495`) — before the first `cp.async` of the weights (`:2513`), so
nothing upstream depends on the pointer:

```text
if (tid == 0) {
    p = ld.acquire.sys(weight_ptrs[expert])          // not the __restrict__ const path,
    while (p == 0 && wait != nullptr) {              // which may compile to ld.global.nc
        if (ld.volatile(wait->abort) || now() - t0 > SPIN_LIMIT_NS) __trap();
        __nanosleep(ns); ns = min(2 * ns, 4096);
        p = ld.acquire.sys(weight_ptrs[expert]);
    }
    if (p == 0) __trap();                             // a zero pointer is never computable
    s_w = p;
}
__syncthreads(); weights = s_w;
```

- **New arguments**: `const MoeWait* wait` (`{abort}` in device memory; null for
  the up and down launches, which never wait) and `int tile_offset` (§4.4).
  Plumbed through `INSTANTIATE_KERNEL_GROUPED_INT8` and its m4/m8 forms
  (`kernel_instantiate.cuh:467-521`), `run_grouped_quantized_matmul`
  (`dispatcher.cu:1576-1657`), the FFI (`quantized/api.rs:282`) and the wrapper
  (`cuda.rs:6686-6781`). The FP grouped instantiation (`:352-364`) gains the same
  parameters so the ABI stays uniform; the host-table int8 path
  (`grouped_matmul_gemx_q8a128`, still used by `latent_moe/attention.rs:482`)
  passes null/0.
- **Cost on the resident path**: one acquire load and one barrier per block.
- **Weights are read through `cp.async.cg`**, which bypasses L1
  (`kernel.cuh:1970`), and the first read of the slot follows the acquire of
  its pointer, so a block cannot hold stale cached bytes of a previous tenant.
- **The fault word is the error path** (§0.12). An abort or an overrun gives
  the expert up and claims the word; the host fails the forward once it has
  synchronised, before any token computed from the layer is returned, and the
  context survives. §8.

### 4.3 Why tile order changes nothing numerically

The grouped kernel's blocks are independent — no atomics, counters or
inter-block synchronization (`kernel.cuh:2014-2258`) — and every output row has
exactly one writer with a fixed K order. `silu_mul_q8a128` and
`fused_deterministic_scatter` index rows and `perm`, never tiles
(`moe_scatter.cu:182-246`). So reordering tiles is invisible in the output bits.
The only things that assumed ascending tile order are a test reference
(`cuda_tests.rs:9417-9431`, compared exactly at `:9507-9509, 9547-9561`) and a
doc comment (`cuda.rs:6683-6684`); both change with the kernel.

### 4.4 Gate launch shape

- **`row_fast = 1` for the gate launch.** The tile index then rides
  `blockIdx.y`, so all row tiles of the resident tiles are dispatched before
  any miss tile (`dispatcher.cu:1633`, `kernel.cuh:2485`). `row_fast` does not
  affect numerics. Up and down keep `grouped_grid_row_fast` (`cuda.rs:95-97`).
- **Sliced against the watchdog.** A spinning launch is a long launch, and the
  Windows 2 s TDR resets the device under one (`cuda.rs:2571-2582` banded a
  dequant for exactly this). The gate GEMM is issued as
  `⌈launch_tiles / gate_slice_tiles⌉` launches over `tile_offset` ranges.
  `gate_slice_tiles` is fixed at load from the slot size:
  `max(1, GATE_SLICE_BYTES / slot_bytes)` with `GATE_SLICE_BYTES` sized so that
  even a slice of all-cold experts is served well inside the watchdog at the
  pack's measured read rate. On Flash-Next (≈1.4 MB slots) decode and most
  prefill are one launch; DeepSeek's ≈12.6 MB MXFP4 experts at prefill width are
  the case it exists for. The count depends only on `launch_tiles`, so it stays
  data-independent.
- `SPIN_LIMIT_NS` is the backstop inside a slice, below the TDR, and is never
  meant to fire.

### 4.5 Routers

Unchanged. `moe_route` (`moe_scatter.cu:285-422`) serves Qwen3, 3.5, 3.6 and
Flash-Next. DeepSeek's `Gate::route` (`latent_moe/moe.rs:127-264`, sqrt-softplus
scores, selection bias, `route_scale`, device `tid2eid` for hash layers) already
produces `(weights, indices)` on the device and feeds bucketize directly — the
all-resident DSpark drafter already chains exactly this
(`latent_moe/dspark_experts.rs:45-324`).

## 5. The routing summary channel (Option A)

### 5.1 Per layer, on the forward thread

After bucketize:

1. record `e_route(n)` on the null stream;
2. the routing stream waits `e_route(n)`, copies `summary_dev[n % RING]` →
   `summary_host[n % RING]` (pinned), records `C(n)`;
3. `cuEventQuery(e_route(n))` — non-blocking; it makes WDDM submit the queued
   batch, so the bucketize the pipeline thread is about to wait on is not left
   sitting in an unflushed command buffer (§11);
4. **send `Routed { seq: n, row, pass, slot: n % RING, done: C(n) }`** — before
   the gate GEMM is enqueued, so that every enqueued gate launch has its message
   already in the FIFO (§7);
5. enqueue gather, gate (sliced), up, SwiGLU, down, scatter.

**This is not the experiment that was reverted.** DeepSeek once moved its
routing readback onto a side stream into a pinned buffer and lost 8% on the cfg8
gate (`docs/deepseek/deepseek_decode_launch_overhead.md:161-163`,
`latent_moe/engine.rs:541-545`). That change kept the forward thread *waiting*
for the copy, so it paid the extra event and stream calls and removed nothing:
the wait still drained the stream. Here the forward thread never waits for the
summary — the GPU keeps executing the layer while it travels — so the same
calls buy the removal of the drain. The per-layer host cost is still real
(§6.5); §15 measures it.

`summary_dev` is a dedicated device ring `[RING × n_experts] u32`, separate from
the bucketize workspace, so the next layer's bucketize cannot overwrite it while
the DtoH reads it. `RING = 64`: 128 KB per side at 512 experts.

### 5.2 Back-pressure is the channel bound

The pipeline channel becomes `sync_channel(RING − 2)`. Message `n + 1` can only
be dequeued after message `n` is fully processed, so when `send(n + RING − 1)`
returns, message `n` has been processed — `C(n)` waited and the host slot read.
Bucketize `n + RING` and DtoH `n + RING` are enqueued after that `send`. So
neither ring slot is overwritten before it is consumed, with no other bookkeeping.
A full channel blocks the forward thread, never the GPU, and the pipeline thread
can always drain it (it waits on nothing the forward thread holds, §7).

The single `PinnedRoutingBuffer` (`handle.rs:489-534`), whose correctness relied
on the forward thread blocking, is replaced by this ring.

### 5.3 What the summary carries

Count and decode bit per expert. That is the whole of what the host consumed
from the readback: the routed set (classify, `transition_matrix.observe`,
speculative validation), the per-expert decode attribution
(`record_hit` vs `record_prefill_hit` / `record_prefill_elevate`,
`pipeline.rs:2458-2467, 1541-1549, 1683-1685`), and hit/miss counts. It does not
carry the table bit: the host decides misses from its own bookkeeping, which §6.2
shows is always at least as current as anything bucketize read.

## 6. The pipeline thread

### 6.1 What it does per `Routed(n)`

1. **Pass boundary** if `pass` advanced: `reset_pass`, `drain_streams`,
   `adapt_prefetch_depth`, end-of-pass decay — as `process_request` does today
   (`pipeline.rs:2386-2401`, `:2699-2701`), keyed on the forward thread's pass id
   instead of "layer index went backward".
2. Wait `C(n)` (host). Read the summary slot.
3. Speculative validation and `transition_matrix.observe` — unchanged.
4. `join_stream_for(row)` — unchanged, before classify, so failed streamed jobs
   classify as honest misses (`pipeline.rs:2443-2447`).
5. **Classify** — `classify_and_load` (`pipeline.rs:1496-1754`) with the routed
   set and decode set from the summary: hit = bookkeeping says resident (in VRAM
   *or* an in-flight speculative/streamed install whose fill is queued); miss =
   otherwise. Demand eviction and `allocate_slot` as today.
6. On the copy stream: wait `e_route(n)` (edge 1, §6.2), **clear** each
   victim's gate entry (§3.3), **load** the misses (`load_experts_batched`, cold
   reads then warm H2Ds, unchanged) and **fill** each one.
   `cuStreamQuery(copy_stream)` to flush.
7. Stats, then speculative prefetch for the next rows and, at prefill width,
   `stream_next_layer` — under the rules of §6.3.

It computes nothing. `process_request`'s compute half, `post_compute`'s response
plumbing, `compute_experts_grouped`, the `Work` response channel, `Hint`, and
every compute-stream wait it issued are gone (§10).

### 6.2 Ordering — the async side keeps its shape

The expert copies have produced corruption through races before (the fence-ring
key collision, `pipeline.rs:3602-3612`; the retraction that let KV memset bytes a
GEMM was reading, `docs/expert_cache_design.md` §12.4). So this design does not
re-architect the copy side. **No stream is added, no stream is removed, and no
new kind of ordering primitive is introduced.** The copy stream, the streamer
stream and its thread, the cold staging rings, `load_experts_batched`,
`load_expert`, the warm/cold sourcing and the per-plan done fences all stay as
they are. What changes is listed exhaustively here, edge by edge:

| # | Ordering edge today | Here | Why |
|---|---|---|---|
| 1 | copy stream waits a **fresh** compute event before every load batch (`order_copies_after_compute`, `pipeline.rs:1804-1812`, at 1983) | **kept, re-pointed**: it waits `e_route(n)`, the event the forward thread recorded after bucketize(`n`) and carried in the message | a fresh record now lands *behind* the spinning gate GEMM, and the copy that would release the spin would wait on it. `e_route(n)` covers exactly the readers the fresh event covered for every invocation `< n`; readers `≥ n` are §6.2's Rule D |
| 2 | streamer stream waits a **fresh** compute event `after` (`pipeline.rs:2225`, `streamer.rs:170-184`) | **kept, re-pointed**: it waits `clears_done`, recorded on the copy stream after edge 1's wait and the plan's victim clears | same deadlock; `clears_done` implies `e_route(n)` transitively, so the streamer is ordered after every reader edge 1 covers *and* after its victims' entries are zero |
| 3 | compute waits `classified.fence` and the prefetch fence ring before computing (`pipeline.rs:2497-2554`, `types.rs:290-301`) | **removed** — the fill-after-bytes order (§3.3) plus the spin does this job | the forward thread enqueues the GEMM before the pipeline thread has recorded these fences; there is nothing to wait on yet. The ring stays, for `is_complete` late-load accounting only |
| 4 | compute waits the streamer's done fence at the target layer (`pipeline.rs:2273-2277` → 2498) | **kept, re-pointed**: the **copy stream** waits it at `join_stream_for`, which runs before classify | same reason as 3 for compute. The copy stream must be ordered after a streamed install before it may clear or re-tenant that slot (otherwise a late streamer write lands on the new tenant, or a late streamer fill resurrects an evicted pointer). Placing it before classify also closes an existing race, §18 |
| 5 | compute waits copy after relocations (`order_compute_after_copies`, `pipeline.rs:1834-1843`, 2943-2945) | **kept as is** | relocation runs only with no pass in flight (§9) |
| 6 | routing stream waits `e1` on compute before the routing DtoH (`quantized_qwen3_moe.rs:697-704`) | **kept**: it waits `e_route(n)`, recorded after bucketize instead of after `moe_route` | it copies the summary rather than the indices |
| 7 | host waits copy-stream events in `ColdStaging::acquire` (`pipeline.rs:1197-1204`) | **unchanged** | — |
| 8 | `quiesce_before_handover` `ctx.synchronize` (`pipeline.rs:3071-3080`) | **unchanged**, reached under §9's entry rule | — |
| 9 | the hint path's fresh-event wait (`pipeline.rs:3185-3192`) | **removed with the hint** | the forward thread has no host routing to hint from |
| new | — | compute waits `clears_done` at the start of a pass (Rule P below) | the one new edge, and of a kind that exists today (compute waiting on a copy-stream event, as edge 3 did) |

New *operations* on the copy stream are only the 8-byte clears and fills of §3.3,
issued from the same thread into the same FIFO as the loads they bracket. The
streamer's jobs gain the fill copies, issued by the streamer thread on its own
stream after each job's bytes; its read, staging and upload logic is untouched.

The rules below show that, with edges re-pointed this way, no slot is overwritten
while a kernel can still read it.

**Fact F.** When the pipeline thread holds `C(n)` — and on the GPU, wherever the
copy stream has passed its wait on `e_route(n)` — bucketize(`n`) has completed,
and, the null stream being serial, so has every kernel enqueued before it,
including every GEMM of every invocation `< n`. Edge 1 makes this a GPU-enforced
property of every overwrite, not only a host-side inference.

**Rule D (demand evictions may take any unprotected, unpinned slot).** Processing
`Routed(n)` with at least one miss, the victim of a load may sit in any row. The
clear is queued before the load, the load before its fill, and the gate GEMM of
`n` cannot complete until that fill lands. Readers of the victim's old pointer:

- invocations `< n` — complete (Fact F);
- invocation `n` itself — reads only experts routed in row `r`, which are
  protected (the classify protect set, `pipeline.rs:1575-1589`);
- invocations `> n`, in this pass or any later one — their bucketize and GEMMs
  run after gate(`n`) completes, hence after the clear has landed; they read
  zero and the expert is a miss for them, loaded on their own message.

So demand eviction needs no lock and no knowledge of where the GPU is. This
covers every victim choice the current policy can make, including the
ahead-of-the-wave ones (`cache.rs:704-712, 885-916`).

**Rule S (speculative evictions only from rows behind, in the current pass).**
*Superseded by §0.7.4: revision 2 has no host speculative clears — every
eviction is a ring claim the device retargets before its write, ranked behind
the wave with the wrap.*
Prefetch and whole-layer streaming are issued while processing `Routed(n)` whether
or not `n` had misses, so the GPU may be anywhere past bucketize(`n`). Their
victims must therefore come from rows whose readers are known complete and which
nothing will read again before the clear lands:

- rows already passed **in this pass** (`row' < row`, invocation `< n`) —
  complete by Fact F, and not re-invoked in this pass by definition of a pass
  (§6.4);
- **not** rows ahead, and **not** the wrap into the previous pass's tail that
  `PREFETCH_EVICT_WINDOW` allows today (`cache.rs:219, 628-634, 832-848`). The
  window keeps its depth and its ranking; it loses the wrap. When `row < 5` it
  is simply shorter.

**Rule P (a pass may not start over un-landed speculative clears).** Rule S stops
speculative clears from racing *this* pass; the next pass re-reads every row. A
**pass lock** (a `Mutex<PassState>` in `ExpertCache`) closes that:

- the forward thread, at the first invocation of a new pass, takes the lock,
  advances `pass`, takes `spec_clear_event`, releases, and if there was one
  enqueues `null_stream.wait(spec_clear_event)` — a GPU-side wait on the copy
  stream, which never waits on compute, so it always completes;
- the pipeline thread, before a speculative eviction for pass `P`, takes the
  lock; if `pass != P` it skips the eviction (the forward has moved on; it may
  still prefetch into free slots); otherwise it queues the clears, records
  `spec_clear_event` (`clears_done`, edge 2) on the copy stream after them,
  releases, then queues the loads and fills. A whole-layer stream plan carries
  that event as its `after`.

The lock is held for a few asynchronous enqueues on one side and three field
updates on the other. A free-slot prefetch needs no lock: a free slot has no
reader, and a fill landing during the next pass is just an expert becoming
resident.

**Rule J (cross-stream clears).** A streamed install's fill is on the streamer
stream. Until its plan is joined it is in `stream_loads` and protected from
demand eviction exactly as today (`pipeline.rs:1575-1589`, `cache.rs:684-688`);
at the join the copy stream waits the plan's done fence (edge 4), so every later
copy-stream clear of that entry is ordered after the streamer's fill.

**Why bucketize never reads a pointer that is about to be overwritten.** A
non-zero entry bucketize reads was written by a fill, after the bytes. It could
only be stale if a clear for it was queued but not landed. Rule D clears land
before any later bucketize (above); Rule S clears are of rows the current pass
has finished; Rule P makes the next pass wait for them. So bookkeeping and the
table agree for every row a bucketize can be reading, apart from one benign case:
a fill queued but not landed reads as zero, the expert is reported routed, the
host sees it resident and does nothing, and the spin ends when the fill lands.

**Why the gate GEMM never spins on a fill that is not coming.** A routed expert
whose entry reads zero is, in bookkeeping, either not resident — the host loads
it on this message — or an in-flight install whose fill is already queued. A
streamed job that fails is uninstalled by `join_stream_for` *before* classify,
so it classifies as a miss and is loaded. A load that fails is the abort path
(§8). There is no fourth case.

### 6.3 The demand-slot guarantee

`minimum_resident_slots(E) = 3E + 1` (`cache.rs:156-158`) prices the pinned rows
(`PINNED_LAYERS = 2`) plus one working row plus one. Rule D shows that only
**one** row is ever being read when a miss is serviced — the GPU is held at that
row's gate GEMM and nothing behind it is reading — so the existing floor is
sufficient, and the concern that several in-flight layers need protecting does
not arise. The protect set is left exactly as it is. That keeps one existing
edge: at the floor, a wide row plus protected in-flight installs can leave no
victim (`cache.rs:717-719`), which today fails the wave and here is the abort
path (§8). Closing it would mean letting demand eviction take an in-flight
install, which changes how the copy and streamer sides interact; it is left as
Q5 rather than folded into this change.

### 6.4 Passes

A **pass** is a run of invocations with strictly increasing row. The forward
thread starts a new pass when an invocation's row is `≤` the previous one: every
trunk forward; each sub-forward of a slab-split prefill (`wave_driver.rs:398-437`);
each MTP draft step, which re-invokes the head row (`qwen4exp/draft.rs:467`,
`qwen35/mtp.rs:327-332`). This is decided inside the `ExpertCache` call the MoE
block already makes, so no driver changes. The current end-of-pass decay keys on
`moe_layer_idx + 1 == num_moe_layers` (`pipeline.rs:2699-2701`), which on models
with an MTP head fires on draft steps only; this design keeps that behaviour
unchanged (§13, Q3).

### 6.5 Does the outcome survive the conservative stream rules?

The goal is that **the forward thread never waits on the MoE, and the GPU waits
only for the bytes of a real miss.** Every wait this design keeps or adds,
checked against that:

| Wait | Who waits | Cost |
|---|---|---|
| copy stream on `e_route(n)` (edge 1) | copy stream | **none** — the pipeline thread only reaches it after holding `C(n)`, which follows `e_route(n)`, so it is already signalled |
| streamer on `clears_done` (edge 2) | streamer stream | the stream plan's uploads queue behind this layer's demand uploads on the bus. Bus throughput is unchanged — one H2D direction is shared either way — and demand, which the GPU is actually waiting on, goes first. The streamer's NVMe reads are host-side and still overlap, since `run_plan` only makes its *stream* wait (`streamer.rs:170-184`) |
| copy stream on the plan fence at join (edge 4) | copy stream | this row's demand uploads queue behind the rest of the row's stream plan — the same total the compute stream waited for today (edges 3 and 4), and only at prefill width, where plans exist |
| compute on `clears_done` at a pass start (Rule P) | GPU, once per pass | only copies queued *before* the last speculative clear: earlier rows' demand uploads (already consumed by their own spins) and prefetch uploads for rows the pass has passed. Expected to be already signalled when the GPU reaches it; measured as its own `gpu_span` (§15) |
| gate GEMM on a miss's fill | GPU, miss tiles only | the irreducible part: the bytes have to arrive. Hit tiles of the same row compute meanwhile |
| forward thread on `send` | host | only with `RING − 2` = 62 layers outstanding — more than a whole forward |
| forward thread on the pass lock | host | three field updates on the pipeline side; a boundary move only under §9's rule |

What the design deliberately gives up: speculative eviction loses the
wrap-around into the previous pass's tail (Rule S). That only affects prefetch
for the first rows after the pinned ones (rows 2–4 with
`PREFETCH_EVICT_WINDOW = 5`), whose victims must now come from free slots or the
rows already passed in this pass. Those rows still prefetch; they have a
smaller pool to take from. If the hit rate on rows 2–4 drops measurably, there
is a refinement that keeps the stream rules intact: when the message that
issues the prefetch had demand misses, queue its speculative clears *before* its
demand fills. The GPU is then held at that row's gate GEMM until those clears
have landed, so the tail rows' next bucketize must see them. That is a change
to the order in which one thread enqueues on one stream, nothing more.
(Revision 2 restores the wrap, §0.7.4.)

What it does not change, and so does not regress:
- **Prefetch lead time.** Today a prefetch for row `L+1` overlapped the forward
  thread's next attention pass. Here it is issued while the GPU runs row `L`'s
  expert GEMMs and then `L+1`'s attention, which is at least the same window.
- **Host enqueue cost per layer.** The forward thread issues the same number of
  driver calls per layer that `route_indices` does today (two event records, a
  stream wait, a DtoH, plus one query in place of the synchronize). Run-ahead
  hides them for as long as the host enqueues a layer faster than the GPU
  executes one. Where it does not, the bound is launch overhead — the existing
  question `docs/decode_graphs.md` addresses. It is
  neither caused nor worsened by this design.

## 7. Deadlock freedom

The design is deadlock-free if nothing the gate GEMM waits for can itself wait on
the gate GEMM. Its only wait is a fill on the copy or streamer stream. Every wait
those streams can encounter:

| Waiter | Waits on | Can that wait on a spinning GEMM? |
|---|---|---|
| copy stream | `e_route(n)` (edge 1) | no — recorded before gate(`n`) was enqueued, after gate(`n−1`), whose fills are earlier in the same copy-stream FIFO |
| copy stream | streamer done fence (edge 4) | no — the streamer stream waits only on `clears_done`, a copy-stream event recorded earlier |
| streamer stream | `clears_done` (edge 2) | no — copy stream, see above |
| pipeline thread | `C(n)` | no — it follows bucketize(`n`), which precedes the gate GEMM |
| pipeline thread | `ColdStaging::acquire` (`pipeline.rs:1197-1204`) | no — copy-stream events |
| pipeline thread | NVMe reads, `join_stream_for` `recv` | no — the streamer depends only on the above |
| pipeline thread | `quiesce_before_handover` `ctx.synchronize` (`pipeline.rs:3071-3080`) | yes, transiently — every gate launch enqueued so far has its `Routed` message ahead in the FIFO (§5.1 step 4), so its fills are already queued; it completes |
| pipeline thread | KV pool mutex (in `renegotiate_boundary`) | only if another thread holds it across a device synchronize while a pass is open — §9 refuses before taking it |
| forward thread | channel `send` | no — the pipeline drains it unconditionally |
| forward thread | pass lock | no — the pipeline holds it only for enqueues, or for a boundary move with no pass in flight to wait on |

And nothing on the pipeline thread is ever enqueued on the null stream: it issues
no kernels and no allocations against the wave arena. Wave arenas reset in
stream order (`bump_arena.rs:83-105, 721-728`), which stays correct because every
expert kernel is now enqueued by the forward thread, in program order — the
`as_foreign_lease` leases and ticketed allocations that relied on "the submitter
blocks" (`quantized_qwen3_moe.rs:312-333`, `types.rs:409-419`,
`wave_buffers.rs:381-430`, `cuda.rs:5684-5688`) are deleted with the host path.

## 8. Failure

There is no host path to retreat to, so a miss that cannot be served fails the
forward that needed it (§0.12):

- **A load fails** on the stager (NVMe error, short read) → it raises the abort
  word (a plain host store to mapped memory) and stops serving. Every waiting
  worker gives its expert up and claims the fault word; the wave driver reads
  it after the segment's synchronise and fails the segment. No token computed
  from the aborted layer is returned.
- **The pipeline thread or the stager dies** → its guard raises the abort word
  in its drop, before setting the flag. `pipeline_dead` stops gating dispatch.
- **A spin overruns `SPIN_LIMIT_NS`** (a bug, a stalled drive) → the same give-up,
  well before the TDR would reset the device.

Errors that are recoverable today because they surface synchronously through
`submit_moe_work` were all one of these. The one that was not — "cannot evict
(all pinned)" — is made unreachable (§6.3).

## 9. Boundary moves

`renegotiate_boundary` (`pipeline.rs:2771-3023`) moves and relocates slots, so it
must not run under a bucketize that has read the old layout. Today it runs the
whole retraction and only then lets `set_weight_floor`'s `wave_is_live` latch
refuse, keeping the evictions (`region_pool.rs:2432-2439`, `pipeline.rs:3006-3009`).
Under this design it decides at entry, holding the pass lock:

- **Refuse at once** (answer 0, edit nothing) if a wave is live, or if the
  requester is not the forward thread and a pass is open. A pass opens at its
  first invocation and closes when the forward thread reaches phase 0
  (`reclaim_spare_ground` / `request_kv_ground`, `batched_model.rs:781, 813`,
  `qwen4exp/wave.rs:2264`, `latent_moe/wave.rs:1373`, `qwen35/forward.rs:781`).
- **Otherwise proceed.** Either no pass is open, or the requester *is* the
  forward thread and is blocked in the round trip, so nothing new is enqueued
  until it returns. The quiesce is safe (§7). Evictions clear entries;
  relocations copy D2D then fill the destination's entries; all of it lands
  before the closing quiesce, ahead of the floor publish.

That keeps every current caller working: phase 0, phase-1 KV claims before the
first MoE invocation, and the forward thread's own claims during a draft walk.
What it refuses that runs today is a boundary move requested by *another* thread
while a pass is in flight — today that edits the cache under the forward. And it
fixes the existing wart that a refused retraction still evicted.

Growing never touches the table: new slots are free.

## 10. Porting — every ExpertCache model

The device chain moves out of `SparseMoeBlock::forward_gpu_native`
(`quantized_qwen3_moe.rs:370-625`) into one function on the cache —
`ExpertCache::forward_routed(acts, weights, indices, decode_tokens, out_dtype, wave)`,
in its own file under `expert_lre/` — which does bucketize through scatter and
§5.1. Each model supplies its router and calls it.

| Model | Router | Acts at the MoE | Change |
|---|---|---|---|
| Qwen3-30B-A3B (`quantized_qwen3_moe.rs`) | `moe_route` | Int8 | `forward_dynamic` → `moe_route` → `forward_routed`; host fork, `route_indices`, `forward_with_indices` deleted |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B incl. AntiLoop/StyleTune (`qwen35/quantized_moe.rs`) | via `SparseMoeBlock` | Int8; **Float** when a LoRA forces `Int8Mode::Off` on a DeltaNet layer (`qwen35/quantized_delta_net.rs:67-71`) | Float acts are quantized with `to_dynamic(Precision, SumScale::Raw)` at entry — the same step the host path's Float-KO arm takes (`compute.rs:523-587`). Quantization is per row and per 128-tile, so quantize-then-gather equals gather-then-quantize bit for bit |
| Qwen3.8-Flash-Next incl. MTP head (`qwen4exp/wave.rs:3370`, `draft.rs:467`) | via `SparseMoeBlock` | Int8 | none beyond the shared block; the head row is an ordinary row. Unblocked by §3.1's `hidden % 1024` removal |
| DeepSeek-V4-Flash (`latent_moe/engine.rs:505-586`) | `Gate::route` | Int8 (Performance quantize, `:529`) | `indices.to_vec2`, the host sort and `submit_moe_work` → `forward_routed(q8, weights, indices, …)`; the readback-budget test drops the routing term (`latent_moe/wave.rs:3413-3424`) |

`ExpertCache::new` already refuses `Int8Mode::Off` (`handle.rs:604-611`), so no
cache-backed model can reach the MoE with Float acts except the LoRA case above.

Every routing constraint holds for all four: experts 128 / 256 / 512 / 256,
top-k 8 / 8 / 10 / 6, hidden 2048 / 2048 / 2560 / 4096, expert widths that
satisfy N % 32 and K % 128, KO weights throughout (MXFP4_KO for DeepSeek, which
the int8 grouped kernel serves, `dispatcher.cu:1517-1563`).

**Routing capture** (`routing_capture`, `quantized_qwen3_moe.rs:819-831`) becomes
an observer on the one path: when enabled it reads the route's indices and
weights back after `moe_route` and records them. The compute path is identical
with it on or off; the device fork's refusal under capture (`:286`) goes.

## 11. WDDM

All three dev machines run WDDM (`docs/performance.md:215-224`), which batches
submissions and drains at sync points (`docs/decode_graphs.md` §1).
Today the per-layer synchronize is such a point; this design removes it. So the
two hand-offs flush explicitly with a non-blocking query: the forward thread
after enqueuing the summary copy (§5.1 step 3), the pipeline thread after
enqueuing fills (§6.1 step 6) and after an abort. Each is one driver call per
layer or per miss batch.

## 12. What is deleted

- `expert_lre/handle.rs`: `PipelineMode::Inline`, `submit_inline`,
  `new_prepopulated`, `submit_moe_work`, `classify_and_load` / `with_inner` /
  `wait_for_copies` / `is_threaded` / `gpu_dispatch()` (no external callers),
  `send_hint`, `set_prev_layer_experts` / `get_prev_layer_experts`,
  `routing_pinned_ptr` and `PinnedRoutingBuffer`, the `live_gpu_dispatch` chain.
- `expert_lre/compute.rs`: `compute_experts_grouped` (all three arms),
  `compute_expert_contribution_gpu_weights`. `extract_weight_info` stays.
- `expert_lre/types.rs`: `MoeInput`, `MoeWorkRequest`, `ClassifiedExperts`,
  `PipelineMessage::{Work, Hint}`; `CopyBatchFence` loses `wait` (it stays as the
  `is_complete` probe for late-load accounting, which no longer gates compute).
- `expert_lre/assignment_sort.rs` and its tests.
- `expert_lre/pipeline.rs`: the compute half of `process_request`,
  `process_hint`, and the compute-side `fence.wait` calls in
  `wait_prefetch_fences_through` / `drain_prefetch_fences` / the classify fence
  (edge 3). `order_copies_after_compute` stays and takes the event to wait on
  as an argument (edge 1); `order_compute_after_copies` stays as is (edge 5).
- `gpu_dispatch.rs`: the all-resident and complete-grid requirements,
  `zone_moved` and the concede-epoch refusal, `hidden % 1024`.
- `quantized_qwen3_moe.rs`: `route_indices`, `forward_with_indices`, the host
  fork, and the reader path `from_gguf` (`:1428-1651`), which has no callers.
- `streamer.rs`: nothing. Its `after` field carries `clears_done` instead of a
  fresh compute event (edge 2) and its jobs carry their fill copies; the thread,
  its stream, staging and fence are unchanged.
- Profile spans that measured the round trip: `fwd_routing_wait`,
  `fwd_cpu_assign`, `submit_roundtrip`, `submit_inbound`, `submit_outbound`,
  `pipe_worker_total`, `pipe_compute_experts`, `pipe_fence_wait`, `pipe_hint*`;
  DeepSeek's `moe:sort`, `moe:sort_readback`, `moe:submit`. New:
  `pipe_routed_wait` (`C(n)`), `pipe_fill`.

Kept: host-table `grouped_qmatmul` (`latent_moe/attention.rs:482` and the kernel
tests use it), `grouped_matmul_gemx` and the float `fused_moe_gather` (core
tests), `moe_bucketize`, `fused_deterministic_scatter`.

## 13. What else changes

- **Stats settle.** Counters the scheduler reads right after a forward
  (`admit_ground.rs:504-531`, `scheduler/decode.rs:57-83`, `prefill.rs:1144-1157`,
  `WeightPlan::from_stats`) were complete when the forward returned because the
  forward waited for the pipeline. `ExpertCache::expert_stats()` now first sends a
  `Settle` message and waits for its answer, which the FIFO delivers after every
  earlier `Routed`. Read after the sampler's sync, that costs one channel round
  trip. Gauges read after phase 0 are settled by the phase-0 round trip itself.
- **Forward timing.** `decode.rs:637-664` times the forward around
  `decode_forward_cobatched`, which with a non-blocking MoE measures enqueue,
  not execution, and trains the admission planner's `layer_secs` on it. It is
  timed through the sampler's synchronize instead. (This is already wrong on the
  all-resident device path.)
- **Shared-expert order.** Qwen3.5/3.6/Flash-Next enqueue the shared expert
  before the router (`qwen35/quantized_moe.rs:136`); with no sync left between
  them it no longer matters where it sits, so it stays.
- **CLAUDE.md hot-path invariant 3** sanctions the MoE routing readback
  "because the streaming ExpertCache schedules pinned→VRAM uploads by expert
  id". It becomes: the MoE routing *summary* goes to the host on a side stream
  and **nothing on the device waits for the host to read it**.
- **Docs and comments** — the full audit is §17.
- **Profiling** — §16.
- **`SlotIntegrity`** (`tensor-assert` only) fingerprints a grid that no longer
  stands still. Its baseline for a slot is taken from the host record it was
  loaded from, and its checks — each a stream synchronize — run with the pass
  closed, at phase 0.

## 14. Tests (written first)

Kernel, `candle-core/src/quantized/cuda_tests.rs`:

1. **bucketize resident-first** — `bucketize_ref` gains the residency input and
   the two-pass tile order; `tile_expert` / `tile_b_start` / `tile_b_cnt` compared
   exactly, and `tok_ids` / `perm` / `rw_ids` / `token_starts` unchanged against
   the all-resident run of the same routing.
2. **bucketize summary** — raw expected `count | decode << 31` words for a fixed
   routing with a known decode split, sentinels included.
3. **device GEMM order-invariance** — the existing
   `cuda_grouped_qmatmul_dev_matches_host_tables` (`:9659-9772`) over a
   resident-first table set, bit-exact against the ascending order.

Integration, each in its own test binary so a launch left waiting cannot hold
up another test's device:

4. **spin released by a copy** — a gate launch over a table whose entries are
   zero, released by fills on a non-blocking stream after a host delay; output
   bit-exact against the pre-filled launch; no synchronize issued between launch
   and fills (this is also the WDDM flush test).
5. **abort gives up** — the abort word set while a launch spins; the launch
   completes, the fault word names the row and the expert, and the context
   stays usable (`candle-core/tests/moe_live_abort.rs`).
6. **ordering stress** — a free-running forward over a cache sized to the floor
   with demand and speculative eviction on every layer, run for minutes with
   `readonly_regions` armed over resident slots; it is the instrument for Rules
   D, S, P and J and must run at production throughput (CLAUDE.md, the fence
   that suppresses the race it hunts).

Pipeline, `expert_lre`:

7. Rule S window: victims never at or ahead of the row, no wrap.
8. Rule P: a speculative eviction is skipped when the pass has advanced; the
   forward side receives the event otherwise.
9. Boundary refusal at entry: nothing evicted when refused.
10. Edge table (§6.2): one test per re-pointed edge asserting the event a load
    batch, a stream plan and a join order against — `e_route(n)`,
    `clears_done`, the plan fence on the copy stream — so a later change that
    re-introduces a fresh compute event fails a test rather than a gate.
11. The §18 streamer race: a stream plan whose last job is still in flight at
    join, a deficit that selects that job's slot, `readonly_regions` armed over
    it — the copy stream's write must be ordered after the streamer's.

Model gates: the sweep (`.claude/skills/sweep`) — every gate and both engine
probes, story intact, on the 3090 (paged for every MoE model) and the 4090
Mobile; plus the existing bitwise repeatability gates
(`qwen3_moe_decode_replay_is_bitwise_repeatable`, `quantized_qwen3_moe.rs:2771`),
which must stay bitwise.

## 15. Acceptance

- Every gate and engine probe green on the 3090 and the 4090 Mobile.
- Decode output bit-identical to the current build on the same residency
  (tile order is the only change in the arithmetic path, and §4.3 shows it is
  invisible).
- Flash-Next single-stream warm decode and ×16 aggregate decode on the 3090
  against the 2026-10-04 baseline (64.4 / 234.6 t/s,
  `docs/results/baseline_rtx_3090_24gb_2026-10-04.md`), and `fwd_routing_wait`
  gone from the profile.
- The gate-launch `row_fast = 1` choice measured against `grouped_grid_row_fast`
  at decode width on an all-resident cache (the 72 GB card), where it is the only
  change.
- **Proof the stall is gone, by direct measurement rather than by a missing
  span:** a `gpu_span` from each row's bucketize to its gate GEMM's start,
  and from the gate GEMM's start to its end, on the 3090 at ×1 and ×16. On an
  all-hit row the first must be the gather's duration and the second the
  GEMM's; on a miss row the second carries the miss service time and nothing
  else. The pass-start wait (Rule P) gets its own span, expected ≈ 0.
- The forward thread's per-layer enqueue time (host) against the GPU's
  per-layer time at ×1 decode on an all-resident cache — the configuration the
  reverted −8% experiment (§5.1) would show up in, since there was no stall
  there to remove.
- Expert hit rate on rows 2–4 against the current build (the Rule S wrap
  loss, §6.5); the refinement there is applied only if this moves.

## 16. Profiling — stall versus overlap

After this change the copies run in parallel with inference, and a profile has
to say which of three things moved when decode gets slower or faster:

1. **The GPU waited on a copy** — streaming did not keep up (hit rate, prefetch
   lead, bus, NVMe).
2. **The copies ran but were hidden** — streaming is working; the bus is busy
   and costs nothing.
3. **The expert kernels themselves got slower** — nothing to do with streaming.

Today's `profile` feature cannot separate these. Its GPU spans time only the
null stream (`profile/gpu.rs`, `gpu_span(name, &Device)` → `dev.cuda_stream()`),
every copy-stream event is `DISABLE_TIMING`, `dma_h2d` times the host enqueue of
an async copy rather than the transfer (`pipeline.rs:1010, 1057`), no kernel
records time or counters, and `moe:gate_up` (`quantized_qwen3_moe.rs:473-532`)
merges the one GEMM that will wait with one that never does.

### 17.1 What is measured

**M1 — miss stall, measured inside the gate GEMM.** `MoeWait` (§4.2) gains a
pointer to a per-row device counter block, non-null only in `profile` builds,
so the kernel code is the same in both and the cost off is one null check.
Thread 0 reads `%globaltimer` around its wait; a block that waited adds to:

| counter (per row, u64, monotonic) | meaning |
|---|---|
| `stall_ns` | per launch, the **longest** wait of any block, summed over launches — the delay the stream saw, since the launch cannot end before its longest-waiting block |
| `wait_block_ns` | every waiting block's wait, summed — how much of the GPU sat spinning |
| `wait_tiles` | tiles that waited at all |
| `stalled_launches` / `launches` | launches with any wait, and all gate launches |

The per-launch maximum is folded by the **last block to finish**, the pattern
the split-K path already uses (`kernel.cuh:2335-2341, 2447-2449`): each block
`atomicMax`es into a launch scratch word and increments a done counter, and the
block that brings the counter to `gridDim` adds the maximum into `stall_ns[row]`
and resets both words itself. One scratch pair per device suffices because
launches on the null stream do not overlap; a sliced gate (§4.4) folds per
slice, and since slices run in sequence the sum is still the stream's delay. No
reset kernel and no memset — both would be unschedulable under spinning blocks
(§3.3) — and the counters are never reset: the host subtracts readings.

`stall_ns` is an upper bound on the wall time waiting cost: a block whose wait
ends while other blocks of the same launch are still computing cost less than
its wait. It is exact for the case that matters, a launch held open by its
misses.

**M2 — copy busy time, measured on the copy engine.** Timing-enabled event
pairs on the stream that carries the bytes:

| span | stream | brackets |
|---|---|---|
| `copy:demand` | copy | one message's clears through its last fill |
| `copy:prefetch` | copy | one speculative prefetch batch |
| `copy:stream` | streamer | one whole-layer plan, `after` wait excluded |

with the bytes moved counted alongside (`demand_bytes`, `prefetch_bytes`,
`stream_bytes` in `PipelineStats`), so achieved GB/s per kind is a division,
not an estimate. This is the first direct measurement of bus time in the tree;
`adapt_prefetch_depth`'s `achieved_gbps` divides bytes by a host-clock *pass*
duration (`pipeline.rs:2348-2378`) and is never reported.

**M3 — overlap, derived.** `copy hidden % = 1 − stall_ns / (demand + prefetch +
stream busy)`. A copy the GPU did not wait for was hidden; every stall is a copy
it did wait for (a speculative copy that landed late shows up as a stall on the
row that needed it). No timeline reconstruction is needed for this number.

**M4 — expert kernel time without the stall.** `moe:gate_up` splits into
`moe:gate` and `moe:up`. The table reports `moe:gate − stall_ns` as the gate's
own time. `moe:up`, `moe:silu`, `moe:down` and `moe:scatter` never wait, which
makes them the control: if they slowed too, the kernels slowed; if only the gate
moved and by the stall, streaming did.

**M5 — miss service latency.** For a message with misses, a cross-stream span
`moe:service` from `e_route(n)` (null stream) to the event after its last gate
fill (copy stream) — the GPU-clock time from "routing known" to "the last missing
expert is computable". `cuEventElapsedTime` accepts two timing-enabled events on
different streams of one context. Its host-side breakdown, all on the pipeline
thread: `pipe_inbound` (send → pickup), `pipe_routed_wait` (`C(n)`),
`cl_classify`, `cold_acquire`, `cold_read`, `pipe_issue` (enqueue of clears,
loads, fills), `stream_join`.

**M6 — pass-start wait.** `moe:pass_wait`, a null-stream span around Rule P's
wait on `clears_done`. Expected ≈ 0 (§6.5); non-zero says speculative clears
are landing late.

`e_route(n)` and the copy-side bracket events are timing-enabled only in
`profile` builds; the fences are otherwise unchanged (`record_event(None)`).

### 17.2 Profile plumbing

- **`gpu_span_on(name, &Arc<CudaStream>)`** and a cross-stream
  `gpu_span_between(name, start: &CudaEvent, stop_stream)` in `profile/gpu.rs`.
  The event-pair pool is already stream-agnostic (`open`/`close` take a stream);
  only the public API is device-only.
- **The pipeline and streamer threads drain their own pools** — the pool is per
  thread — non-blocking after each message and each plan, blocking on `Settle`.
- **The pipeline thread records into the process-wide store** (`pipeline_record`)
  instead of its private `ProfileAccumulator` (`pipeline.rs:1473`), so `pipe_*`,
  `cl_*`, `cold_*`, `copy:*` reach `pipeline_snapshot*`, `/v1/profile`
  (`zend/src/api/profile.rs`) and `kv_fragmentation`'s tables, which today never
  see them. `ExpertCache::forward_profile` and `record_profile`'s second write go
  (it double-counts every forward-thread span, `handle.rs:1916-1922`).
- **`snapshot_profiles`** becomes: `Settle` (the pipeline thread drains its spans
  blockingly, reads the M1 counters with an async copy on the routing stream it
  waits for, and returns their deltas as entries). **`Qwen4ExpBatched` overrides
  it**; today it inherits the empty default (`batched_inference.rs:5464`), so
  Flash-Next's pipeline spans appear in no table at all.
- **`ProfileAccumulator` / `ProfileSnapshot` gain `max` and a `device` flag.**
  Miss stalls are a tail; a total and a count hide it. The device flag is set by
  the GPU drain, which replaces `kv_fragmentation`'s hand-maintained
  `GPU_SPANS` list (`candle-conversation/tests/kv_fragmentation.rs:303-350`).
- **`HIGH_WATER`.** With no per-layer sync, a forward opens on the order of a
  thousand GPU spans before anything drains them, and the high-water mark of
  4096 (`gpu.rs:80, 159-164`) would block the forward thread mid-forward in a
  profile build — perturbing exactly the run-ahead being measured. It is raised
  to cover several forwards, the scheduler's per-wave non-blocking drain
  (`scheduler/run.rs:687`) keeps it down, and a blocking drain that does happen
  records `profile:drain_block` so a self-perturbed profile says so.
- **Printers that skip `gpu_drain_blocking`** — the `decode_profile` test
  (`quantized_qwen3_moe.rs:3483-3612`) and the DeepSeek wave printers
  (`latent_moe/wave.rs:4011, 4204`) — call it, or they report no GPU spans.

### 17.3 What is printed

A new table in `batch_test/utils.rs` beside `"=== Expert Pipeline Stats ==="`,
one column per config as that table has, printed by `validate_and_print`:

```text
=== MoE Streaming ===
Metric                          #1 BF16×1   #2 C10×16 …
Miss stall (GPU)                 ms/tok, % of decode step
Stalled launches                 n / launches
Wait block-time                  ms/tok
Gate GEMM excl. stall            ms/call
Up / down GEMM (control)         ms/call
Copy busy — demand               ms/tok  @ GB/s
Read-ahead                       images / GiB (worker counters)
Copy busy — stream               ms/tok  @ GB/s
Copy hidden                      %
Miss service (GPU)               avg / max ms
  of which routed wait           avg / max ms
  of which cold read             avg / max ms
Pass-start wait                  ms/tok
```

`kv_fragmentation`'s `print_pipeline_profile` and `/v1/profile` get the same
entries through the snapshot (the derived rows as computed entries), and a
per-row detail — the rows with the most `stall_ns`, from the per-row counters —
follows the table when any row stalled.

`"=== Expert Pipeline Stats ==="` (`batch_test/utils.rs:2791-2968`) changes with
the path: **"MoE dispatch"** (`device` / `host (readback)`) goes — there is one
path; **"Hint loads"** goes with the hint; **"Work requests"** becomes
**"Routed messages"**; **"Fence stalls"** goes — it counted "a load batch had an
event" (`pipeline.rs:2547-2553`), never a stall, and M1's stalled launches are
the real figure. **"Hit rate"** becomes correct: `classify_and_load` returns
before its telemetry on a layer with no misses (`pipeline.rs:1616-1622` vs
`1733-1736`), so today all-hit layers are not counted and the hit rate — which
also trains the admission planner (`admit_ground.rs:504-529`) — is biased low.
Counting from the routing summary fixes that by construction.

### 17.4 Reading it

| What moved | Means |
|---|---|
| stall ↑; gate excl. stall, up, down flat | streaming fell behind — look at hit rate, read-ahead claims and precision, copy GB/s, cold reads |
| stall flat; gate excl. stall and up/down ↑ | the expert kernels got slower |
| copy GB/s ↓ | the bus or the source — `cold_read` and `cold_acquire` say which |
| miss service ↑, copy busy flat | the pipeline thread's host side — `pipe_inbound`, `pipe_routed_wait`, `cl_classify` |
| copy hidden ↓ with stall ↑ at equal hit rate | prefetch lead time shrank — the run-ahead is outpacing the predictor |
| pass-start wait > 0 | Rule P clears are landing late |

### 17.5 Tests

- Kernel: a gate launch whose fills are released after a known host delay —
  `wait_tiles` and `stalled_launches` exact, `stall_ns` at least the delay;
  a pre-filled launch leaves every counter but `launches` unchanged.
- `ProfileAccumulator` `max` and `device` flag: raw expected entries.
- `gpu_span_on` on a non-null stream and `gpu_span_between` across two streams
  return a duration bounded below by a timed copy.

## 17. Documentation to update

> **Applied 2026-10-07:** the §18.1 READMEs (`candle-transformers`, `candle-kernels`,
> `docs/README.md`, `docs/archived/README.md`), the §18.2 design docs, the §18.3
> superseded-by pointers, and the §18.5 code comments are updated. `docs/decode_graphs.md`
> already described the live dispatch and needed no change. The paper (§18.4) is still
> flagged, not edited. `CLAUDE.md` (invariants 3, 4 and 7's example) and the `bed` and
> `sweep` skills are updated too.

Audited across `docs/`, `CLAUDE.md`, the skills, the READMEs and every Rust and
CUDA doc comment. **A** = states behaviour this change makes false; rewrite with
the code. **B** = a historical record; leave it, add a superseded-by pointer
where noted. **C** = paper text; flagged, not edited here. Comments sitting on
code §12 deletes go with that code and are not repeated.

### 18.1 Repository guidance (A)

- **`CLAUDE.md`** — invariant 3 (lines 193-196) and invariant 4's "the two
  transfers in #3" (line 200), per §13; invariant 7's MoE-table example
  (lines 232-237) cites the concession count this design deletes — keep the
  rule, replace the example with clear-before-re-tenant; the `SlotIntegrity`
  row of the tensor-assert table (§13).
- **`.claude/skills/bed/SKILL.md:83-85`** — "two sanctioned readbacks": mirror
  invariant 3.
- **`.claude/skills/sweep/SKILL.md:161-169`** — says the `qwen4exp` wave loop
  never calls `reclaim_spare_ground()`; it does (`qwen4exp/wave.rs:2264`), and
  §9 relies on it. Already false today.
- **`candle-transformers/README.md:82-93, 134`** and
  **`candle-kernels/README.md:87-88`** — "two modes — threaded … and inline";
  links to `docs/gpu_native_moe_dispatch.md`, which has moved to `archived/`.
  Point at this document.
- **`docs/README.md:65`** — the `gpu_native_moe_dispatch.md` row is a broken
  link marked "Built and verified"; replace it with this document.

### 18.2 Design docs (A)

- **`docs/expert_cache_design.md`** — §5.7 line 376 ("in front of a forward that
  is waiting for it"), §6.3 line 479 ("the routing buffer" → the summary ring),
  §12.7 lines 890-892 (`post_compute`'s response plumbing).
- **`docs/vram_span_partition.md`** — §4 lines 205-210 and §7 rule 5
  lines 475-476: the cached-slot-address rule and concession count.
- **`docs/deepseek/deepseek_hot_path_invariants.md`** (authoritative per
  CLAUDE.md) — lines 243-247, 258-260, 276-281, and the invariant 3/4 tables at
  348, 349, 360: the routing readback is no longer sanctioned-and-load-bearing.
- **`docs/decode_graphs.md`** — rewritten for every model with the readback and
  the pipeline round trip gone: the whole MoE layer is one capturable region, with the
  dispatch host protocol run per layer between graph launches (§4.5).
- **`docs/deepseek/deepseek_decode_launch_overhead.md:147-163`** — "why the
  readback cannot simply be removed". The rest of the doc is a measurement
  record (B), including the −8% lesson §5.1 answers.
- **`docs/deepseek/deepseek_hot_path_optimization_findings.md`** — lines 49-51,
  264-267 ("routing must be host-visible").
- **`docs/deepseek/deepseek_v4_speculative_decode.md`** — §1 52-59, §4.1 168-172,
  §4.2 208-209, §7 433-437: the "routing readback is the WDDM wall" premise.
- **`docs/deepseek/deepseek_v4_flash.md:486-499`** and
  **`docs/deepseek/deepseek_batched_paged_attention_plan.md:446, 535, 685`** —
  lower priority; the first is partly stale already.

### 18.3 Historical (B)

`docs/wave_feeder.md:1126-1128`; `docs/unified_wave_inference_engine.md` §3.3;
`docs/deepseek/deepseek_perf_optimization_report.md:78-83, 310-319` (its
"`gpu_dispatch` today requires `all_resident`" is exactly the gap this closes —
add a pointer); `docs/deepseek/deepseek_decode_reproducibility.md:38-40,
171-178`; the "MoE dispatch … host (readback)" rows of
`docs/results/performance_rtx_3090_24gb_rows.tsv`; and in `docs/archived/`:
`gpu_native_moe_dispatch.md` (**add superseded-by this document** — its Phase B
is this design), `expert_pipeline_dataflow.md` (still cited by
`expert_lre/mod.rs:9`), `qwen38_flash_next.md`, `qwen36_prefill_profile.md`,
`qwen35_qwen38_models.md` (incl. the `hidden % 1024` row),
`tiered_cache_paper.md`, `arena_unification_results.md`,
`elastic_vram_partition.md`, and single lines in `npc_api_gui_design.md`,
`overnight_int8_and_kernels_report.md`, `qwen36_performance_plan.md`.

### 18.4 Paper (C)

- **`docs/unbounded_agents.md` (v2, in preparation)** — abstract line 11 and §1
  line 72 claim stall-free / zero-idle MoE inference under partial residency;
  the current profile contradicts that, and this design is what makes it closer
  to true — the claim should be re-measured, not assumed. §4.1 line 172; §4.2
  line 191 and §4.3 line 207 ("speculative prefetch never evicts" — already
  untrue, and Rule S changes it again); §4.4 lines 225-238, the "Phase 1 / Fence
  / Phase 2 / Join" dispatch and "the CPU never blocks", which this design
  replaces with per-tile waits.
- **`docs/unbounded_agents_v1.md`** — the same passages (lines 15, 73, §4.1,
  §4.2, §4.4 at 222-235). Published and served; an erratum is the only route.

### 18.5 Code comments that survive the change (A)

- **`expert_lre/`**: `mod.rs` module doc (1-32, 99-106 wrap, 137-175 modes and
  tables, link at 9); `handle.rs` (1-9, 102, 430-480, 1021-1034, 1755-1799);
  `gpu_dispatch.rs` (1-17, 60-130, 155-165, 236-241, 341-362, 443-447, 662-669,
  and the "fell to the host path" test messages at 842-951); `zone_geometry.rs`
  (whole module doc — it exists for the deleted `zone_moved`; either it goes or
  it is re-scoped); `cache.rs` (114-119, 213-222, 259-277, 485-505); `types.rs`
  (62-63, 86-87, 117-131, 257-258, 270-289, 428-460); `pipeline.rs` (1-6, 117,
  164-170, 1402-1458, 1814-1825, 2250, 3289-3318, 3498-3511); `streamer.rs`
  (4-11, 24-34: protocol now `clears_done`, joined on the routed message);
  `compute.rs` module doc; `slot_integrity.rs:6-11`; `weight_plan.rs:9-13`.
- **Models**: `quantized_qwen3_moe.rs` (216-276, 345-363 → `forward_routed`,
  592, 2197 dangling "reader path above", 3475); `qwen35/quantized_moe.rs:1-10`
  and `qwen35/expert_loader.rs:13-15`; `qwen4exp/engine.rs:20-23` (already
  wrong: "bucketize declines >256"); `qwen4exp/wave.rs:3202-3207, 3359-3365`;
  `latent_moe/readback.rs:1-9`, `engine.rs:498-501`, `wave.rs:929-930,
  1044-1047, 1089-1090, 2953-2956` and the readback-budget test 3413-3424;
  `latent_moe/dspark_experts.rs:1-43, 212-216, 335` (lower priority);
  `batched_inference.rs:4743, 5490-5491` and
  `candle-conversation/src/models/builder.rs:257-258` (fixed cost attributed
  to the routing readback).
- **Elsewhere**: `candle-conversation/src/scheduler/decode.rs:639-658` (forward
  timing and "counters this forward just moved", §13);
  `candle-nn/src/kv_cache/chunked/region_pool.rs:40-45` (when `W` moves, §9);
  `candle-core/src/quantized/cuda.rs` 5567-5571, 5760-5762, 6669-6684 (device
  GEMM: "every expert VRAM-resident", "same ascending-expert tile order"),
  7490-7498 (bucketize); `candle-kernels/src/simple/moe_bucketize.cu` 4-9,
  34-45, 158-159; `batch_test/utils.rs:2856-2867` (the "MoE dispatch" row,
  §16.3).

## 18. Open questions

- **Q1 — scope.** "Every MoE model" here means every `ExpertCache` model: Qwen3-30B,
  3.5, 3.6 (and variants), Flash-Next, DeepSeek-V4. Two MoE forwards in the tree
  never used this path and are not touched: Mixtral through `quantized_llama.rs`
  (`MlpOrMoe`, its own CPU routing at `:106-175`, non-KO experts, no cache), and
  the upstream float models (`mixtral.rs`, `qwen2_moe.rs`, `qwen3_moe.rs`,
  `deepseek2.rs`, `granitemoehybrid`). Bringing Mixtral onto the cache is a KO
  repack plus a cache build — a port of its own.
- **Q2 — boundary moves from other threads during a pass** are refused (§9).
  Whether any caller depends on them succeeding today needs one look at
  `claim_region`'s refusal handling in the persistence path.
- **Q3 — end-of-pass decay** fires only on draft steps for models with an MTP
  head (§6.4). This design preserves it; it reads like a defect.
- **Q4 — `GATE_SLICE_BYTES` and `SPIN_LIMIT_NS`** need a measured cold-read rate
  per machine to size. The slicing is the mechanism; the numbers are a
  measurement.
- **Q5 — "cannot evict (all pinned)" at the floor** (§6.3) is an existing edge
  that becomes fatal rather than a failed wave. Closing it touches the
  copy/streamer protection, which this change deliberately leaves alone.

**A probable existing copy race, closed by edge 4.** `join_stream_for` removes
the joined layer's installs from `stream_loads` (`pipeline.rs:2291`) *before*
classify computes its protect set (`:1575-1589`), and `join` waits only for the
streamer to finish *enqueueing* (`:2250-2262`), not for its copies to land. A
streamed expert of the current row that the route did not select is then
unprotected, and `demand_eviction` admits the current row at factor 1.0
(`cache.rs:895-903`, distance 0). Its re-tenant goes out on the copy stream
ordered only after a fresh compute event — and the compute stream has not yet
waited this plan's fence (that wait is at `pipeline.rs:2498`, after classify).
So the new tenant's bytes and the streamer's still-landing bytes can hit the same
slot unordered. It needs a prefill-width wave whose stream plan mispredicted an
expert and a deficit that picks exactly that slot, which fits the
"intermittent, wide-wave only" shape of past expert corruption. Not reproduced;
found by reading. Edge 4 orders the copy stream after the plan before classify,
which closes it, and it is worth a regression test of its own (§14, test 11).

Pre-existing defects found on the way and out of this scope: `router_topk`
indexes out of bounds when every score is `-inf` (`router_topk.cu:104-123`);
DeepSeek's routed experts skip the `swiglu_limit` clamp the reference applies
(`latent_moe/moe.rs:65-89`); the Float non-KO arm of `compute_experts_grouped`
cannot have worked for BF16 activations (the scatter requires F32,
`cuda.rs:7348`) — moot once deleted.
