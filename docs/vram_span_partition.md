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
  left.**
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

## 6. The fix: continuously compact the left side

**The KV side must earn the packed-left property that the weight side already
has.** Compaction runs continuously, so that live KV is always dense at low region
indices, `live_watermark()` stays as low as the live byte count allows, the tier is
placed low, and `weight_floor` can therefore sit as far left as possible —
**making the weight side, and so expert residency, as large as it can be.**

The target is the inflation ratio in §5: region space should approach the live KV
bytes it holds, not 4.2× them.

Design constraints any implementation has to satisfy:

- **It must relocate, not merely release.** Emptying arenas is what
  `release_empty_arenas()` already does, and it is not enough — the defect is a
  *sparse* arena at a high index. Live chunks have to move down.
  [`archived/arena-compact-kernel-design.md`](archived/arena-compact-kernel-design.md)
  drafts the slot-level copy (a raw `memcpy` of `stride_bytes`, no dequantisation,
  since equal-length moves within a size class); note that its description of
  `compact_arenas()` predates the current code, where that function no longer
  exists.
- **It may only run between forwards.** An arena may not be created or moved while
  a tier is placed, for exactly the reason `set_weight_floor` refuses then. The
  scheduler loop's one legal window is the same one guest models use.
- **It must be incremental.** A stop-the-world compaction on the scheduler thread
  is a stall paid by every sequence in flight. Bounded work per wave, continuously,
  is the shape.
- **Lowering the watermark must actually be followed by lowering the floor.** The
  weight side *taking back* ground the KV side no longer needs is listed as **open**
  in the original design ledger; compaction that frees high regions without a path
  for `weight_floor` to move back left buys nothing.
- **Address capture must be re-checked.** Moving a chunk invalidates any cached
  device address for it, by the same rule §4 states for expert slots.

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

When a symptom looks like bad arithmetic — a NaN in a GEMM, an implausible
magnitude — but the operands and weights are individually finite, **suspect the
partition before the kernel.** `candle::readonly_regions` (behind `tensor-assert`)
exists for this: declare a tenant's ground immutable and the guard names the writer
at the moment of the write, instead of leaving a wrong number to be found
downstream.
