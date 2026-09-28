# The VRAM Span Partition — weights, transients and experts

**Status: current.** This is how device memory is laid out and handed out today,
written for someone arriving at the code with no history. It describes the
mechanism as built, names the one place the mechanism does not yet keep its own
promise, and states the fix that closes it.

Related documents, and how they differ from this one:

| Document | What it is |
|---|---|
| [`vram_governor_design.md`](vram_governor_design.md) | The **startup** authority: how capacity `C` is measured and the partition is first sized. |
| [`expert_cache_design.md`](expert_cache_design.md) | What lives in an expert slot, how it is chosen, and what a miss costs. |
| [`archived/elastic_vram_partition.md`](archived/elastic_vram_partition.md) | The original design argument and its design-versus-build ledger. Historical. |
| [`archived/arena_unification.md`](archived/arena_unification.md) | Why KV moved to one reservation with shared size classes. Historical. |
| [`vram_partition_behavioural_tests.md`](vram_partition_behavioural_tests.md) | The tests that hold the invariants below. |

`CLAUDE.md`'s **Hot-Path Invariant 7** is the normative statement of the rules
here. This document explains the machine those rules protect.

---

## 1. One span, four tenants
<!-- Five kinds of allocation: the KV side holds both band arenas and the KvHead
     record arenas of §8, which share its region pool and its boundary. -->


At model load the governor measures real resident capacity and reserves **one
contiguous device span**. Every tenant lives inside it, at a known offset. There
is no second allocator competing for the card.

```
span_base                                                              span_end
    |                                                                      |
    v                                                                      v
    +----------+--------------+---------------------+----------+------------+
    | persist  | dense weights|     KV regions      | transient|  expert    |
    |  staging |   (if any)   |   (size-class       |   tier   |  weights   |
    |  (fixed) |   (fixed)    |    arenas)          | (per fwd)|  (slots)   |
    +----------+--------------+---------------------+----------+------------+
               ^              ^                     ^          ^
          region_base    region_base           live_end    weight_floor
          (after the                        (arena frontier)  (ELASTIC)
           fixed blocks)
    <------------- grows right ------------>   <---- grows left ------------
```

Reading it left to right:

- **`persist` staging** — the persistence thread's hot→warm copy buffer.
  `PERSIST_SPAN_BYTES`, four regions' worth, at the far **left** and fixed for the
  process lifetime. It is left rather than floating precisely so the elastic
  boundary can move without anyone reasoning about a copy stream that is not
  synchronised to the wave.
- **dense weights** — on a streamed-dense model, the non-expert weights. Also
  fixed, also below `region_base`.
- **KV regions** — the paged KV cache. `REGION_BYTES` each (`= TARGET_ARENA_BYTES`,
  16 MiB), indexed `[0, total)`, handed out one region per arena. **Fills from the
  left.** Two *kinds* of arena draw from this pool: the band arenas holding K/V
  payload, and the record arenas holding the `KvHead` records the paged kernels
  dereference (§8). Both are inside `[0, total)` and so below `weight_floor` by
  construction; they differ in what may move them.
- **wave transient tier** — activations for the forward currently running. Exists
  *only* during a forward. **Placed per forward**, not reserved.
- **expert weights** — equal-sized expert slots. **Fills from the right.**

Two boundaries matter, and they are not alike:

- **`region_base`** is fixed. The blocks below it never move.
- **`weight_floor`** is **elastic**. It is the line between the KV side and the
  weight side, and it moves at runtime. `total` is *derived* from it:
  `total = (weight_floor − region_base) / REGION_BYTES`, recomputed by
  `layout_span` on every path that moves it. So the weight side's ground lies
  outside `[0, total)` **by construction** and can never appear in a count of
  regions the KV side owns.

**Nothing in CUDA enforces any of this.** Every address in the span is mapped, so
a tenant that walks past its boundary reads and writes a neighbour's live data and
raises nothing at all. It surfaces as a wrong *number*, many layers later, never
as a fault. That is why the rules below are checked in arithmetic rather than
trusted.

---

## 2. The symmetry the moving boundary rests on

The two sides of `weight_floor` are deliberately built from the same parts, and
this is the load-bearing idea of the whole partition:

> Both sides hand out **fixed-size units** from one end of the span, and both keep
> live data packed **away from the frontier**, using a **lowest-index-first free
> list** to do it. The KV side packs left because its frontier is on the right;
> the weight side packs right because its frontier is on the left.

That symmetry is *why* a boundary between them can move: whichever way it moves,
the data it would disturb has already been pushed out of the way.

Both free lists are `BinaryHeap<Reverse<usize>>` — a min-heap, so a release pushes
the index back and the next claim takes the **lowest** free index, never the most
recently freed.

`weight_zone.rs` states the weight side's version of the property plainly: every
slot is `max_expert_size` bytes, so "the rightmost free spot" *is* "the lowest
free index", a retraction is a **suffix of the index space**, and relocating a slot
is a memcpy between two addresses of identical length rather than a compaction.
There is no fragmentation to reason about because there is none.

**The KV side does not achieve this, and §5 is about why.**

---

## 3. Flow: the transient tier (per forward)

The tier is the tenant with the shortest life and the strictest placement rule.

1. **Admit.** Every KV claim a forward needs is made *before* the forward, in the
   admit phase (`wave_admit`). A region claim creates an arena, and an arena may
   only be created between forwards.
2. **Place.** The tier is anchored at the **arena frontier** and sized to what
   *this* forward needs (`plan_wave_transient`), not to the worst case.
3. **Run.** Every sweep of the forward uses fixed offsets inside the placed tier.
4. **Release.** The tier's lifetime ends with its forward.

### The placement rule, and why it is a refusal rather than a clamp

`tier_fits(base, len, live_end, floor)` is the whole of it. The tier must sit
**above every live arena** and **below `weight_floor`**:

```rust
let short = live_end.saturating_sub(base).max(top.saturating_sub(floor));
```

Both directions are reported because either can bind. If it does not fit, the
answer is to **refuse** — never to place it anyway (invariant 7: refuse rather
than corrupt).

`live_end()` is the hard floor for the tier, and it is a **high-water mark, not a
count**:

```rust
fn live_watermark(&self) -> usize {
    let free: HashSet<usize> = self.free.iter().map(|Reverse(i)| *i).collect();
    (0..self.next).rev().find(|i| !free.contains(i)).map_or(0, |i| i + 1)
}
```

It scans downward from the top for the highest **non-free** index. A region is not
preemptible — an arena keeps its address for as long as it lives — so this is a
hard floor, not a preference. **One live arena at a high index holds the tier up,
no matter how much free ground lies beneath it.**

### Three geometry defects already paid for

Recorded here because each was found only on a GPU, and none needed one:

- A tier measured **down from `weight_floor`** landed **on top of live regions**,
  because a region keeps its address for the life of its arena.
- A **placed tier makes the region ceiling deaf to the boundary**.
  `ceiling_regions` answers with the tier's own base when one is placed, because
  that is real memory a running wave is writing into. So a tier left standing past
  its forward caps the KV side wherever it was put, and **no concession the weight
  side makes can lift that cap** — measured, as a daemon conceding itself to its
  floor over thousands of retries while the ceiling answered 293 every time. The
  tier's lifetime ending with its forward is what closes this.
- Counting fresh ground from `next` alone **double-counted** regions `claimable`
  had already returned. `blocked` takes `next.max(ceiling)` for this reason.

---

## 4. Flow: the weight side and the experts

The expert cache leases equal-sized slots from the right end of the span.

**Filling.** Slots are claimed lowest-index-first, which means packed hard against
`span_end` — away from the frontier at `weight_floor`.

**Retracting.** When the KV side needs ground the weight side holds, the weight
side *concedes*. `weight_zone` owns bytes, not experts: it returns a
**`RetractPlan`** — what to move and what to drop — and the expert cache above it
executes. Keeping policy out is what lets the whole module test without a GPU, a
model, or a routing trace.

**When a concession is legal.** Only **between forwards**. `set_weight_floor`
refuses a floor that would strand a live region, and refuses to move at all while
a wave generation is open. A refusal is correct: mid-wave, the transient tier
stands flush against the KV frontier and the ground above it belongs to that wave.

**The floor the weight side may never cross** is `MIN_ELASTIC_RESERVE`, and it is
*derived*, not chosen — the point at which a warm daemon can still serve a wave
without evicting a single sealed chunk:

| term | bytes |
|---|---|
| wave transient span (from the three phase spans) | 912 MiB |
| `MIN_FIRST_WAVE_KV` | 384 MiB |
| **`MIN_ELASTIC_RESERVE`** | **1,296 MiB** |

Deriving it means a change to the wave tier moves the floor with it, which a
constant could not. It covers a *first* wave only; growth beyond it is handled by
retraction. (Halving the old flat 2 GiB to 1,024 MiB is the obvious move and is
wrong: it lands below `912 + 320`, so every forward would retract the weight side
to place its tier and then regrow it — layer slots traded for boundary churn on
the hot path.)

**A captured device address does not survive a boundary move.** The MoE dispatch
tables cache one slot address per expert, on the reasoning that an all-resident
cache's weights never move. They do: a concession evicts the slots at the frontier,
and the zone then *grows back* — so capacity and floor read exactly as they did at
load while the conceded slots hold something else. **Compare a monotonic concession
count, never the geometry.**

---

## 5. Where the mechanism does not keep its promise

§2 says both sides keep live data packed away from the frontier. The weight side
genuinely does. **The KV side only does so for new claims.**

A lowest-index-first free list decides where the *next* arena goes. Nothing ever
moves an arena that already exists. `release_empty_arenas()` — the per-wave sweep
in the scheduler loop — returns a region only when its arena's **last** chunk is
gone; it performs **no data movement**. So a sparse arena holding one live chunk
pins its region exactly as firmly as a full one, and if that region sits at a high
index it pins `live_watermark()`, and therefore the tier, and therefore the
pressure on `weight_floor`.

The result is a ratchet:

1. KV arenas end up spread across the region index space.
2. `live_watermark()` sits near the top of that space.
3. The tier must be placed above it, and needs its full width there.
4. That width overshoots `weight_floor`, so `tier_fits` reports a shortfall.
5. The weight side concedes; `weight_floor` moves right; expert slots are evicted.
6. Free regions *below* the watermark are reported as available KV budget — and
   they are, for new arenas — but they are **unusable by the tier**, which is the
   tenant that forced the concession.

Nothing lowers the watermark again, so step 5 is one-way.

### Measured, on the 5-hour daemon run of 2026-09-25

| | 15:58 | 18:21 | |
|---|---|---|---|
| weight zone | 53,582 MiB | 33,685 MiB | **−19,897 MiB (−37%)** |
| expert slots | 21,452 | 13,527 | **−7,925 (−37%)** |
| decode, 16 sequences | **131 ms/fwd** | **~680 ms/fwd** | **5.2× slower** |
| KV per forward | 400,423 | 553,012 | 1.38× |

56 concessions, `relocated=0` on every one. Decode degradation tracked the
concession window and **plateaued when concessions stopped** at 18:21, staying at
~680 ms until the process died at 20:59. KV volume grew 1.4× against a 5.2×
slowdown, so KV size does not account for it; a 37% loss of expert residency is
the right order of magnitude.

The end state shows the inflation directly: **25,264 MiB of region space holding
9,488 MiB of arenas, holding 5,960 MiB of live KV** — 4.2× inflation, with 428
free regions (6,848 MiB) stranded below the watermark.

> A caution for anyone reading this evidence: the per-wave
> `kv-vram budget=… used=…` line does **not** show an overrun. `budget` is
> `(free + blocked) × REGION_BYTES` — *remaining* KV headroom — while `used` is the
> CUDA pool allocator's live bytes across everything it serves. Neither bounds the
> other. What the line is for is watching `budget` fall toward the setpoint
> (eviction firing) and `used` cap out rather than climb without bound.

---

## 6. The fix, as built: continuously compact the left side

**The KV side earns the packed-left property that the weight side already has.**
Compaction runs from the wave loop, so live KV stays dense at low region indices,
`live_watermark()` stays as low as the live byte count allows, the tier is placed
low, and `weight_floor` can sit as far left as possible — **making the weight side,
and so expert residency, as large as it can be.**

The code is `candle-nn/src/kv_cache/chunked/compact_plan.rs` (what to move),
`compact_map.rs` (who holds the old identities), `compact.rs` (the pass), and
`scheduler/prefill.rs::compact_kv_if_fragmented` (when). The harness that specified
it, and still gates it, is `candle-conversation/examples/kv_fragmentation.rs`.

### What the shape is, and why each part is that shape

- **It relocates, it does not merely release.** `release_empty_arenas()` returns an
  arena whose *last* chunk has gone; the defect is a *sparse* arena at a high index,
  which that can never reach. `plan_pool` is a two-cursor walk over one pool's slots
  in physical address order: every gap is filled with the last live chunk above it,
  so when the cursors meet the pool is a gapless prefix. The move count is the
  minimum for a perfect pack.
- **Highest arena first.** The pass is time-budgeted, and the census — an occupancy
  bitmap walk per arena, per rung — is most of its cost. Walked in ladder order it
  spends the budget on whichever rungs come first and can never reach the pool that
  owns the topmost arena, which is the only pool whose position costs the weight
  side anything. `pool_top_rank` answers the ordering question without the census,
  so deciding where to spend the budget does not spend it. Measured before the
  ordering: nine seconds at 52–59% with the frontier pinned and 64 free regions
  under it, every pass running and clipping on the low rungs.
- **A fresh low arena for every pool, highest pool first.** Packing is per pool, so
  each pool converges onto *its own* lowest arenas — and a pool whose lowest arena sits
  high in the span packs perfectly and stays where it is. Before planning, every
  censused pool is given one fresh arena, in the census's order (the pool whose top
  arena is highest first). The region free list is lowest-index-first, so that order
  hands the lowest hole to the pool standing highest, the next hole to the next, and
  the walk then fills each fresh arena first because it ranks lowest in its pool. Only
  while a hole exists below the frontier: with none, a claim would land above it and
  raise it. A fresh arena that lands above its own pool's top receives nothing and is
  released as soon as the claims are in — one O(1) claim and one O(1) release, which
  falls to nothing once the pools sit in the lowest regions.
- **Every empty arena goes back, including the pass's own.** An arena's creation window
  — which keeps the empty sweep off it — is always closed by its creator: at the first
  claim for an allocate-on-demand creator, and explicitly right after its claim for a
  stamp (`claim_fresh_region_open`, then `finish_creation`), so no arena is left
  unreclaimable because nothing happened to land in it. Measured on the 30B with windows
  left open: 11 record arenas standing where 1 was in use.

  The pass protects what it creates ahead of demand (`FreshArenas`) and releases it
  itself: the per-pool low arenas the plan put nothing in go back right after the
  claims — before the holder sweep, when nothing can name an empty arena, and before the
  record reservation looks for a hole — and whatever is still empty goes back when the
  pass ends, on every exit path. The arenas that took relocations stay protected until
  then, because releasing one mid-sweep would free an `arena_idx` an unreached holder
  still names.
- **Two CUDA calls for the whole pass.** The claims are a host walk producing three `i64`
  arrays, and one launch of the migration scatter/gather kernel copies every relocated
  slot; one further launch fills the records minted for them. A per-chunk
  `memcpy_dtod_async` measured ~8 µs of launch overhead each, which put 1,024 moves in an
  8 ms budget and left every pass clipped with the frontier exactly where it started. No
  record is ever *rewritten* — see the record bullets below.
- **It runs only between forwards, and it refuses rather than waits.** The pass takes
  the arena window, for exactly the reason `set_weight_floor` refuses while a tier is
  placed. It also takes an exclusive hold on chunk locations (`migrate_flight`),
  because the arena window says nothing about the persistence thread — see below.
- **It is incremental and budgeted.** A quarter of the budget plans, the rest claims,
  and the first batch of claims is unconditional so a pass that planned always moves
  something. A clipped pass is not a broken one: everything below the cursor is
  packed and nothing moved upward, so the next pass resumes closer.
- **Considered every wave, run on a cheap signal.** The gate is an interval floor,
  then the region pool's hole count, then a sparsity sum from the refcount tables'
  live counters — no bitmap walked. Both halves are needed: holes self-correct under
  allocation, so a steadily-loaded pool reads zero holes with tens of sparse arenas
  beneath it (gating on holes alone: 16 passes over 110 s, pools at 82%), and
  sparsity alone misses the burst.
- **Lowering the watermark is followed by lowering the floor.** A non-empty pass
  calls `reclaim_spare_ground()` in the same method, while no wave generation is
  live. The two are one method and not two precisely because either alone buys
  nothing.
- **Every holder of a relocated identity is rewritten, structurally.** A chunk's gid
  *is* its location, so moving bytes changes identity. The holders span three crates
  — the backings' block tables, the substrate's residences, the projection caches —
  and `compact.rs` enumerates them and refuses the pass if any sweep fails. It does
  not try to *discover* holders from a gid: there is no reverse index, and the
  refcount cannot stand in for one, because `HeadGids` is `Arc<Vec<ChunkGid>>` with a
  derived `Clone` and every sharing path shares the allocation. A prior branch tried
  that and corrupted conversations.
- **The device records are NOT rewritten, and a record owns the bands it names.** A
  chunk's location is recorded twice — in the gid, which refcounts the slot, and in the
  `KvHead` record's band-pointer word, which is what the kernels dereference. The pass
  rewrites only the first. A record holds a clone of the `HeadGids` its words were
  serialized from (§8), so it cannot outlive the slots it addresses: they keep a
  refcount for as long as it lives, still hold the bytes the copy read out of them, and
  are read correctly through it. A sealed chunk is read through its record and never
  written; a live writer window has no record and is addressed from its own gids, with
  its decode buffer invalidated — so the two cannot diverge.

  **It used to rewrite them, and that is what corrupted K/V.** A chunk has exactly one
  record, shared by every holder of that chunk, so a pass that rewrote some of a chunk's
  holders and missed others had no correct value for that word: whichever slot it named
  was kept alive only by the holders naming the same slot, and when those went the rest
  were still reading through it into re-tenanted ground. Patching to the destination,
  patching to the source and leaving it alone were all wrong — leaving it alone became
  right only once the record started holding a refcount. `unwitnessed` is the proof the
  holder list is incomplete (53–1,240 on two passes of every Flash-Next run), and the
  `tensor-assert` boundary named the side: zero orphaned gids against 148 orphaned
  records already present when a pass *began*. A record-arena walk keyed on the address
  each record held was built and measured — the Flash-Next probe answered 8/8, 5/8, 8/8,
  5/8 against 8/8 twice with the pass off — and then deleted, because the domain was
  never the problem.
- **A relocated chunk is given a freshly MINTED record, and that is what reclaims the
  ground.** `compact_mint.rs`: one record per rewritten `HeadGids` allocation that had a
  record, claimed during the sweep (host-only, so the holder installs it in the same visit
  and the sweep stays one traversal) and written by **one** batched device launch at the
  end of the pass. Installing it drops the old record; when that was the chunk's last
  holder, the old record's slot frees and with it the clone of the *old* gids it was
  holding — the last reference to the source bands — so the source frees and
  `release_empty_arenas` hands the region back.

  Leaving the records alone instead is safe but reclaims nothing: measured on the
  Flash-Next probe, 1.5–1.8 M relocations for 3–11 regions, because a source pinned only by
  its record is still *occupied*, so the census re-plans it every pass and no holder names
  the destination. With minting: 566 k moves, 79 regions, efficiency 92%.
- **Minting moves a RECORD's address, and every cached copy of one must be dropped.** This
  is the second half of the fix and it is not optional. A band's address is owned — by the
  record — but a record's own address is cached as a bare `kvheads_ptr` word in every
  `TokenSlice` header of every slot's decode buffer, and that buffer is reused whenever the
  chunk *count* agrees, which a compaction never changes. `rewrite_for_compaction` clears
  the buffer of each slot whose own chunks moved or whose record was followed to a copy.
  After the holder sweep, every other slot's buffer is checked against what the sweep
  replaced, using the `ChunkPin`s the buffer holds for each chunk it serialised — one per
  header, naming its bands and its record — and any buffer naming replaced ground is
  dropped (`invalidate_decode_buffers_naming`). A buffer naming nothing replaced is exact
  and is kept. Zero is the expected count: a non-zero one names a slot whose buffer
  reaches a chunk through something the sweep does not rewrite. The pins keep that ground
  alive, so it is not a wrong read — the pin must name the record the header names, which
  is why `update_chunk` replaces a pin when either the bands or the record changed.

  Without it the probe answered 7/8, 5/8, 3/8, 2/8 — a stale `kvheads_ptr` dereferencing a
  record slot the pool had reissued, which reads another chunk's band pointers and does not
  fault. With it, 8/8. **And `kv_integrity::report_boundary` could not see any of it**: zero
  orphans and zero content changes on a run that answered 2/8, because nothing was orphaned
  (every reference named a live slot) and the content hash covers only bands that live
  slots' block tables name. The engine probe's story gate is what caught it.

### The defect this work found, which is the one to remember

**A pin keeps an arena alive and says nothing about which slot of it a chunk
occupies.** The persistence thread's hot→warm migrate captures a device address per
band from gids it has pinned, off the scheduler thread, with no arena window. A
compaction relocating those chunks underneath it makes it copy whatever now sits in
the vacated slot into the warm tier. It does not fault — every address in the
reservation is mapped — and it surfaces much later as a sequence answering from
another sequence's KV.

`migrate_flight` is the exclusion. Both sides use `try_`, and **neither ever blocks**:
each holds substrate and block-table locks while it works, so either side waiting on
the other would need those lock orders to agree, and nothing waits, so there is
nothing to agree about. The migrate takes and fences the guard *per group* rather
than per batch, because a mass eviction is what produces both the fragmentation and
the hot→warm work that must precede it — held batch-wide it starved compaction
exactly when compaction was most needed. A refused pass sets a flag that makes the
next migrate step aside once, because the long holder otherwise wins nearly every
contest (78 refusals in 107 attempts).

This is the general rule, of which §4's expert-slot caution is the other instance:
**a captured device address is invalidated by anything that moves what it names, and
a reference count is not a location.**

### Where it stands

On the 30B-A3B, driving the real engine through overlapping conversations with
pinned residents and stragglers (`--churn-secs 90 --concurrency 24 --pinned 6
--straggler-every 4 --batch 20`):

| figure | before | after |
|---|---|---|
| VRAM efficiency, steady state | 82–89% | 97–99% |
| VRAM efficiency, worst sustained | 48% | ≥ 90% (often unjudgeably good) |
| frontier after a full drain | 1,727 of 1,999 | 13–120 |
| ground handed back by one drain | 0 | up to 9,680 MiB |
| story rewrites correct | 20/20 | 20/20 |

The forward gate is unmoved: 10,210 t/s prefill and 565 t/s decode against a
recorded 10,239.6 / 572.4, with C10 compression identical at 5.50×.

**What remains, and it is the parked item.** The residual loss is a single-sample
dip at a mass eviction: 24 conversations retiring at once frees ~85 regions below
the frontier, and the top *live* arena keeps its high rank because an arena's rank is
its region's. Chunk compaction cannot fix that — the chunks are live and packed. What
would is **arena relocation**: swapping a high arena's region for a lower free one,
which moves 16 MiB and changes no gid at all, only resolved addresses. The harness
charges only losses that persist across two samples, for the reason stated there, so
it passes without this; the burst dip is visible in its `worst single sample` row.

---

## 7. The rules, as a checklist

Each has been violated in production at least once.

1. The tier may not be placed above `weight_floor` (`tier_fits`).
2. The tier may not stand on a live region — it sits above `live_end()`.
3. `weight_floor` may not move below a standing tier's top. A floor approved
   against the geometry of a moment ago can retroactively put a placed tier inside
   the weight zone.
4. An arena layout may not cross `weight_floor`. A wider forward arriving behind a
   live one must not walk its plan from a base chosen for a smaller purchase.
5. A raw device address captured from one tenant is invalidated by any boundary
   move. Compare a monotonic concession count, never the geometry.
6. A boundary check must consult the **reservation**, never live occupancy. A check
   against a mid-wave `region_stats().transient_bytes` snapshot, or one comparing
   two figures derived from the same array, passes while the invariant is broken.
7. A captured chunk address is invalidated by a compaction, and a pinned gid does not
   protect it — the pin holds the arena, not the slot. Anything acting on captured
   addresses off the scheduler thread must hold `migrate_flight`'s guard for as long
   as it acts on them, and fence before releasing it. See §6.

When a symptom looks like bad arithmetic — a NaN in a GEMM, an implausible
magnitude — but the operands and weights are individually finite, **suspect the
partition before the kernel.** `candle::readonly_regions` (behind `tensor-assert`)
exists for this: declare a tenant's ground immutable and the guard names the writer
at the moment of the write, instead of leaving a wrong number to be found
downstream.

---

## 8. `KvHead` records: the second arena kind

### Why they move into the span

A `KvHead` record is what a paged kernel actually dereferences: per `(head, palette,
K/V)` band it holds the band's absolute device address, plus the palette tag bytes
that say how to decode it. One record describes one chunk, and every slot referencing
that chunk resolves to the same record.

They begin outside the reservation, in `CudaSlice<u8>` slabs taken straight from the
CUDA allocator. That contradicts §1's first claim — *there is no second allocator
competing for the card* — and it costs three things that matter here:

- **A host serialize and an upload per write.** `serialize_kv_heads` builds each
  record into a `Vec<u8>` on the host, and `write_records_batched` coalesces those
  into runs and issues one `memcpy_htod` **per run, under a lock**. The record's
  entire content is derivable on the device from data the device already has, so
  every one of those bytes crosses the bus needlessly.
- **No walk.** A slab is reachable only through the handles pointing into it, so
  "every live record" is not a question the pool can answer, and neither is "does
  this record still name ground something holds".
- **Ground the partition cannot see.** The slabs are real VRAM that `weight_floor`
  arithmetic knows nothing about, so the two sides of the boundary are sized against
  a capacity that is already spoken for.

### Shape

One arena kind, distinct from the band arenas, drawing regions from the same pool:

- **Stride is the record size rounded up to a `RECORD_STRIDES` rung.** A record is
  `n_kv_head × (head_dim / 2 + 26 × n_palette)` bytes. The property that matters is that
  the set of strides is *finite and known*, because that is what lets the gid pool
  preallocate every record pool it could be asked for and stay eager and lock-free.
  The rungs interleave 1.5× steps between the powers of two, every one a multiple of 8 so
  each slot base is 8-aligned for the `uint64_t` band-pointer stores.
  **Derived per call, not cached.** `n_palette` changes after the backing exists —
  `set_single_latent` gives the single latent four times GQA's bands — so a record size
  taken once at construction would be the GQA one and every record would overrun its
  slot. `build_meta_records` therefore recomputes it from the live `n_palette()` on
  every call and picks the key from that, leaving nothing to keep in step.
  > **Why not powers of two.** They were, and the pad is not free: a record's slot is
  > pure padding above its size and nothing recovers it. The 1,344 B GQA record sat in a
  > 2,048 B slot — 34%, one record per chunk per layer, ~144 MiB at 48 layers and 128K
  > context, all inside the reservation and subtracted from `weight_floor`'s arithmetic.
  > The justification offered for powers of two — that slot decode becomes a shift and a
  > mask — is not cashed in anywhere: nothing decodes a record slot from an address,
  > because the kernel is handed absolute destinations and `record_slot_addr` multiplies.
  > Interleaving 1.5× rungs bounds the pad at a third of a slot instead of a half and puts
  > that record in 1,536 B (12.5%), for the same finite-pool property.
- **The handle is the existing arena gid.** A record is an arena slot, so `ChunkGid`
  and the arena refcount tables give refcounting, cloning across every holder of the
  chunk, and free-on-last-drop with no new lifetime machinery. This is the one place
  the existing design is reused rather than re-derived: the semantics wanted for a
  record — shared by every referencing slot, released when the last one goes — are
  exactly `ChunkGid`'s.
- **A record owns the bands it names, and that is what makes a compaction safe.**
  `MetaGid` carries a clone of the `HeadGids` its pointer words were serialized from, so
  a record can never outlive the slots it addresses: the allocator cannot reissue a band
  slot while a record still points at it, because the record is one of that slot's
  refcount holders. One `Arc` bump per record, eight bytes on the handle, nothing per
  band.

  Before this, a record was a *dead copy* of a fact the gid owned — the address
  `base_ptr + chunk_idx · chunk_byte_stride`, stored with no reference to its subject —
  and that is the general rule `CLAUDE.md` states as "a captured device address is
  invalidated by anything that moves what it names, and a reference count is not a
  location". It is why a compaction that rewrote holders' gids and left records behind
  read another chunk's K/V, and why the pass can now leave them alone instead. See §6.
- **A record slot is an ordinary arena claim, which is what makes minting cheap.** A
  compaction mints one record per relocated chunk — 800–1,400 a pass — and each is a
  free-list pop plus an address, with the bytes written by one batched launch. That is the
  other half of the reason for moving records here, and it is what the band arenas' reclaim
  now depends on (§6).

  Two hazards found paying for it. A record claim must **never** be promoted: when
  `stamp_region_promoting` cannot get a region it widens `key.class` and rebuilds the key
  with `ArenaKey::new`, which is a `Band` key, so the record was handed a slot in a band
  arena and filled on top of live K/V — and every check passed, because a 1,344 B record
  fits a 2 KiB band slot. The allocator now refuses for any non-`Band` kind. And a mint
  moves a record's address, so every cached `kvheads_ptr` must be dropped (§6).
- **Records are enumerable, and nothing needs it any more.** Being arena slots, the
  arena table lists every record arena's base, stride and slot count, so a walk can
  reach every record whether or not anything reachable names it. That was the original
  reason given for moving records here; it was built, measured, and deleted once the RAII
  above made rewriting records unnecessary. `kv_integrity::check_records` still walks the
  block tables per holder and reads each record back over the bus, exactly as it did when
  records lived in slabs — a diagnostic comparing boundaries, not a publish.

### Filling them: one batched launch, nothing on the host

The write becomes a scatter/gather kernel, not a serialize-and-copy. Per batch the
host uploads **one** descriptor table — the gid grid, the palette tags, and the
destination record addresses — and the arena extent table; the kernel is grid-strided
over a work space flattened across records, palette bytes and bands, and each band's
thread computes its address as `base[arena] + slot × stride` and stores the 8-byte
pointer at `band_ptr_offset`.

> One qualification on "nothing on the host": the descriptor is smaller than the records
> for a *batch*, but the extent table is the dense arena-indexed vector, so a
> single-record seal at a few hundred arenas uploads several KB to write ~1.3 KB of
> record, across nine `memcpy_stod` calls. Batching amortises it; a persistent extent
> buffer patched on arena change would remove it.

**The layout is defined in four places and the kernel must not become a fifth.**
`serialize_kv_heads` and `band_ptr_offset` in `meta_pool.rs`, `kv_head_size` in
`models/slot_state.rs`, the hand-rolled record writer in `latent_moe/paged.rs`, and
the device-side accessors in `paged-decode/slot_types.cuh` all encode the same byte
offsets independently. A fill kernel that derived them a fifth time would be one more
copy to drift, so it computes its offsets from the same arithmetic and is held against
the host serializer byte-for-byte by test — which is what makes the host serializer
worth keeping as the reference implementation once it is off the hot path.

Four properties this is required to have, and the reason each is not an optimisation:

1. **One launch for the whole batch**, per invariant 2b: the kernel takes a
   descriptor table, so no caller has to pack records together to satisfy it.
2. **No host loop over records or bands.** The host builds the descriptor and stops;
   addresses are computed where the data already is.
3. **No avoidable host→device copy.** The descriptor is strictly smaller than the
   records it produces, and the records themselves never cross the bus.
4. **No synchronisation.** The upload and the launch are both async on the wave's
   stream and nothing is read back. A fill that fenced would put a device-wide wait
   between every pair of chunk allocations.

### Packing the record arenas

**The record pool is compacted by the same pass, with the same walk.** A record arena
is a region like any other, so a sparse one high in the span holds the frontier up
exactly as a sparse band arena does. The pass ranks the record pool beside the band
pools, gives it a fresh low arena the same way, plans it with `plan_pool`, and copies
its moves in the same batched launch as the bands.

What differs is how a holder follows the move, and it is the minting argument again:

- **A copy is a correct record.** A record's bytes are band *addresses*, never its own
  slot, so the bytes at the new slot describe exactly what the old ones did. No fill
  kernel runs; the copy is byte-exact.
- **Holders get a new handle; the old one is never touched.** The sweep looks up each
  holder's `meta` by its slot's raw id and installs a `MetaGid` for the copy carrying the
  same `bands` clone. A holder the sweep does not reach keeps the old handle, whose slot
  stays occupied — and whose address stays valid — for as long as it does. Nothing is ever
  relocated under a handle.
- **Bands first, then the record.** A chunk whose bands moved in the same pass is given a
  freshly minted record, and its record's copy is left unused: the copy names the
  *source* bands. `Sweep::follow_record` skips anything the sweep minted.
- **Every retired record is held until the sweep ends.** Records are followed by raw id,
  and a record whose last holder was replaced frees its slot — which a mint in the same
  sweep can then be handed. A second visit to a holder of that mint would find its raw id
  in the record map and be given a copy of the old chunk's record. Holding the retired
  records keeps every raw id the sweep resolves meaning one thing, which is the same fix
  `Sweep::originals` makes for band gids.
- **The cached `kvheads_ptr` words are dropped.** A followed record moves the address a
  slot's decode buffer holds, so the slot counts as moved and its buffer is thrown away;
  any other buffer still naming a record the sweep retired is caught by the pin check
  (§6).
- **Mints land low, and never on a record source.** A mint takes a slot from the record
  arenas lowest in the span first, rather than by arena index — which is unrelated to
  address, and would put fresh mints in the very arena the pass is draining. It never
  takes one of the pass's planned record sources, whose record may have died after the
  census: that slot is ground the pass has declared it is reading.

## 9. Recurrent state: the third arena kind

A DeltaNet layer's state — the `s` accumulator plus its conv tail — is a fixed-size
block of model geometry (3.12 MiB on Flash-Next, 2.09 MiB on Qwen3.5), and a sequence
holds two per recurrent layer: the live state and the half a wave writes. They are
**state arenas** (`chunked::state_arena`): one 16 MiB region cut into slots of one
stride, the block rounded up to 256 B, shared by every sequence of that geometry. A
sequence holds slots, not regions. Five fit a region on Flash-Next (2.6 % unused),
seven on Qwen3.5 (8.4 %).

- **Device-global, not the KV backing's.** A model's recurrent stores outlive every
  session, while a KV backing and its arena storage belong to one; so the arenas are
  keyed by `(device, stride)` and each holds its region as a `SpanRegion`, claimed
  through `SpanClaims` between forwards like every other tenant of the span. The
  window is opened only when a new arena is needed.
- **One slot per layer state.** The live and write halves need not be adjacent — the
  kernels take a pointer per buffer — so each is its own slot, which is what lets a
  slot be a fifth of a region instead of two-fifths.
- **RAII.** A `StateSlot` returns itself on drop, and the drop that empties an arena
  releases its region. A recycled slot holds its last tenant's state, so a store zeroes
  its `live` slots (read at zero, invariant 6's exemption) and leaves the write halves
  alone.
- **Priced as slots, accounted as regions.** Admission prices a store at two slots per
  recurrent layer at the stride — an arena's unused tail is shared and charged to no
  one sequence. The whole-card report counts every state arena's regions instead
  (`RecurrentStateStore::arena_reserved_bytes`), because that is what recurrent state
  denies the rest of the span: a sum over stores would leave out every free slot and
  unused tail, and any slot a handle kept past its store.
- **An empty half is refused.** A geometry whose `s` or conv tail is zero bytes
  (`conv_kernel = 1`) has no address to give that buffer that is not the next slot's,
  so the store refuses to build rather than hand the kernels an aliasing pointer.
- **The tensors own their slots.** Each state tensor's storage holds an `Arc` of its
  slot (`LeaseAnchor`), and every clone, view and re-lease shares it, so a slot returns
  to its arena only when nothing can read it any more. A slot is returned on the host
  while kernels may still be in flight: its next tenant in the arena works on the same
  primary stream, and a region released by an emptied arena is stamped dirty, so its
  next claim synchronises before zeroing it — the same fence every arena relies on.
- **Not compacted yet.** The arenas are enumerable, every slot has an address and every
  arena a position — the census a compaction plans from — but no pass moves them, so
  their regions still count as span tenants in the efficiency report.
