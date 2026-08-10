# The Elastic VRAM Partition

**Status: partially built.** Supersedes the static-partition parts of
[`vram_governor_design.md`](vram_governor_design.md) §7 and §11 (`kv_floor`,
`expert_budget`, both deleted) and extends the single reservation of
[`docs/archived/arena_unification.md`](archived/arena_unification.md) §3 to cover
the expert cache as well as KV.

§14 is the honest ledger of design-versus-build, and it is worth reading before
anything else: the first build got the **central mechanism backwards**, and the
correction changed the shape of the whole document.

## What is built, and what is not

| | state |
|---|---|
| One span, dense weights outside it, `kv_floor` / `expert_budget` deleted | **built**, measured (§12) |
| Balloon: one reserve on both paths, chunk refinement | **built**, `C` +944 MiB |
| Expert slots as leased views into the span; mirrored free lists | **built** |
| Copy-stream ordering for fixed-address slots (§8) | **built** |
| Transient tier placed per forward between arenas and weights, released after | **built** — mechanism proven |
| Tier *sized* per wave (the reclaim itself) | **built and measured on the daemon**: +53 KV regions / +848 MiB over baseline, a 19% increase (§13c) |
| Retraction bounded by what the pinned pool can absorb | **built** — it could previously destroy the expert cache (§13c) |
| Weight side blocked from taking ground while KV is under pressure | **built** — history and occupancy both miss a monotone ingest (§13c) |
| **Phase-locked forward**: admit all KV → place tier at `A` → move `W` → lock | **not built** — §7 is the design, §7a is what stands in for it today |
| **Admit phase**: every KV claim made before the forward, for every layer | **built** (`wave_admit`, §13b) |
| Tier reserved for the **forward**, not per phase — every sweep at fixed offsets | **built** (`plan_wave_transient`, §13b) |
| Admission capped by KV room as well as compute and tier: `R = min(8192, transient-fits, KV-fits)` | **built** (`prefill_width_cap`) |
| Tier anchored at the arena frontier `A` | **refuted** — the quantized path claims regions mid-forward at the same order as the KV itself (§13b) |
| Boundary set exactly instead of by a decaying estimate | **refuted** — the boundary is set against the *next* wave's demand, which no exactness reaches (§13b) |
| The weight side *taking* the ground the tier gives back | **open** — needs a refused claim that can block (§13) |

---

## 1. What is wrong with the partition today

Every byte of the card is assigned to one of two owners at model load, by a
constant, and never reassigned:

```
expert_budget()  = usable − kv_floor − scratch_margin
kv_floor         = 4352 MiB + 0.15 × (C − weights)
```

`kv_floor` is the reserve the expert cache must leave. It is sized against the
**cold-boot high-water mark** — 284 live regions (4,544 MiB) while the system
prompt's collections prefill — and the daemon then spends the rest of its life
at **70 regions (1,120 MiB)**. So in steady state roughly 3.4 GiB of the card is
reserved for a peak that happened once, at boot, and the expert cache — the only
thing that pays for decode — never sees it.

The measured cost of that reservation is not small. From
`candle-core/src/vram/budget.rs`, on the 16 GiB card:

| `kv_floor_abs` | KV span | expert slots | decode |
|---|---|---|---|
| 3 GiB | 218 regions | 2618 | dies: retry storm |
| 4 GiB | 274 regions | 2267 | — |
| 5 GiB | 328 regions | 1917 | 67 ms/fwd |
| 6 GiB | 384 regions | 1566 | 80 ms/fwd |

1024 MiB of the term buys 56 KV regions and costs 351 expert slots. The constant
has to be set high enough for the *worst* moment the workload ever produces, and
is therefore wrong at every other moment — which is all of them.

The failure is not that the constant is mistuned. It is that a single number is
being asked to answer a question whose answer changes every wave. When a wave
carries little KV, the right answer is "almost all weights". When a wave carries
an 8192-token prefill, the right answer is "as few weights as it takes to fit" —
and that is *also* the wave where the extra weights matter least, because a
prefill of that width has enough arithmetic per expert load to hide the misses.

This document replaces the constant with a boundary that moves.

---

## 2. Shape

One reservation covers the entire card. It is divided by **two frontiers** that
move between waves and are frozen during them:

```text
                     ┌─ the reservation ───────────────────────────────────┐
  dense weights      │ span_base                     W            span_end │
  (CUDA pool,        │     ├─────────┬─────────┬─────┼──────────────────┤  │
   outside the span) │     │ persist │transient│ KV  │   expert slots   │  │
                     │     │ staging │ (wave)  │arenas                  │  │
                     │     │  fixed  │  fixed  │ ───►│ ◄──────────────  │  │
                     │     └─────────┴─────────┴─────┴──────────────────┘  │
                     └─────────────────────────────────────────────────────┘
```

- **Dense weights** are loaded **before** the span is reserved, from the ordinary
  CUDA pool, by the ordinary path. They are permanent and immovable, so nothing
  is gained by placing them inside the span — the span is simply claimed after
  them, and takes what is left. This is what keeps the load path unchanged and
  removes the dense extent from every piece of arithmetic below.
- **The persistence staging block is the one fixed address.** Its copy stream
  runs on the persistence thread's schedule, so its ranges can be live at any
  moment a forward begins — it sits at the far left, where nothing reaches it.
- **Arenas** grow **left to right** from a fixed base. Unchanged: whole 16 MiB
  regions, lowest-index-first, live data left-packed
  (`chunked/region_pool.rs`).
- **The wave transient tier is the one variable-size block, and it lives between
  the arenas and the weights.** Its size is *per wave* — `WavePlan` prices it
  from the model's geometry, and a twenty-session decode measures at ~50 MiB
  where the widest prefill the gate runs measures at ~465 MiB. It is safe to be
  variable because **it vanishes between
  forwards** (§3): at the moment its extent changes it holds nothing, so a
  resize leaves no hole and moves no data.

  That is also *why* it must sit where it does. A variable-size block at a fixed
  address is a choice between two failures: leave the arena base fixed and every
  resize strands a hole between them — **fragmentation**; or pack the arenas
  straight after it and every resize moves the arena base, relocating every KV
  region — **thrashing**. The only position where a size change disturbs nothing
  is adjacent to the boundary that is already designed to move.
- **Expert slots** grow **right to left** from `span_end`, **lowest slot index
  first**, so live weights stay right-packed.
- **W** is the only moving boundary: the leftmost byte the weight side holds.

The two allocators are mirror images and use the same data structure — a
lowest-index-first free list — for the same reason: **keep live data away from
the frontier, so the frontier can move.** `region_pool` already does this
(`BinaryHeap<Reverse<usize>>`, "principle 5: keep live data left-packed"). The
weight zone gets the same, mirrored.

`A ≤ W` at all times, with `W − A` at least the transient span. That inequality
is the whole invariant.

### The floor weights never cross

The weight zone may never grow so far left that the elastic middle drops below
`MIN_ELASTIC_RESERVE`, initially **2 GiB**. The arithmetic behind that number:

| term | bytes |
|---|---|
| transient span (`W_attn + W_ffn + W_forward`) | 912 MiB |
| steady-state KV (measured, 70 regions) | 1,120 MiB |
| **total** | **2,032 MiB** |

So 2 GiB is the point at which a warm daemon can still serve without evicting a
single sealed chunk. It is a floor, not a target: a wave that needs more takes
more, by evicting weights. It exists so that a pathological weight fill at load
time cannot leave the engine unable to run its first wave, and so that the
minimum-viable configuration is a stated number rather than an emergent one.

### There is no second number — the floor is the only one

**`MIN_ELASTIC_RESERVE` is the only constant in the partition.** Where the
boundary sits at load does not matter, because it is recomputed exactly on every
forward (§7) — at 57–80 ms a sweep, the opening position survives one decode step
and is then irrelevant. Fill the weight side to the floor at load, as this
section originally said, and let the first forward correct it.

`INITIAL_KV_RESERVE` (5,376 MiB) currently exists in the code and should not.
It is a **crutch for the missing phase lock**, and the story is worth keeping
because the failure it patched looks like a tuning problem and is not one:

Filling to the floor was built as designed and the gate killed it — the twenty
concurrent Q8_0 contexts exhausted a 2 GiB KV side outright, `every region of the
KV reservation is occupied (67 live)`. The temptation is to read that as "2 GiB
is too small". It is not. Load time is the least informative moment the engine
ever has — the arenas are empty, so every byte the weight side could take looks
free — and **nothing was going to correct the boundary afterwards**, because the
built give-back path costs a pass (§7a) and a failing arena claim has no pass to
wait. The constant did not fix the partition; it hid the fact that the partition
never moved.

Under §7 the correction arrives before the first claim can fail, so the crutch
comes out and the floor is all that is left: how far the weights may be squeezed
before the expert cache is worth more than the context depth. That is a real
policy question with a derived answer (above); the opening position never was.

---

## 3. The invariant that makes this safe

**Before a wave there are zero live transient buffers.**

This is already true and already enforced. `Generation` refuses to reset while
any guard is live, and `LiveTensor<'w>` makes a wave buffer unnameable outside
its generation's scope — the borrow checker rejects the program rather than the
allocator catching it at run time (`chunked/bump_arena.rs`, commit `b91d4011`).
So between waves the elastic middle holds nothing at all.

That is what makes the boundary movable without a fragmentation story. There is
no compaction, no relocation of live intermediates, no address that has to stay
put across the move — because at the moment the boundary moves, nothing in the
middle exists. The tier is re-placed and re-sized every forward, and a
`BumpRange`'s `'w` is bounded by the generation, which is bounded by the wave.
A tier base changing between waves is invisible to every rule that governs a
bump range.

**This is the load-bearing sentence of the whole design, and the first build
read it backwards.** The reasoning that went wrong: *a `BumpRange` is a bare
pointer, therefore the tier's base must be fixed.* That confuses **fixed for the
process** with **fixed for the wave**. `'w` only ever bounds a range inside its
own generation; between generations there is nothing left to invalidate. The
vanish is not a constraint the tier has to survive — it is the licence that lets
the tier be the movable, resizable block, and therefore the licence for the whole
partition to be elastic.

Read the two consequences together and the position falls out with no freedom
left in it:

- the tier must be variable-size, because a per-wave price is the entire source
  of the reclaim (912 MiB reserved against ~4 MiB needed for a decode);
- a variable-size block can only sit adjacent to the moving boundary (§2);
- and it can only be *safely* variable because it vanishes.

The only thing left to enforce is that the tier does not move while a guard is
open, and the `live` counter that already refuses a reset refuses a rebase for
the same reason and in the same place.

---

## 4. Startup

Order, with the change from today marked:

1. **Balloon → `C`.** (§5.) Install the governor.
2. **Load the dense weights.** Unchanged — the ordinary pool path, in the
   ordinary order. The only edit is that the whole per-layer loop now completes
   before anything else happens.
3. **Reserve the whole span.** ← *new; this is the "something else" that happens
   just before the expert weights.* `Reservation::reserve(usable − pool_cushion)`,
   then map every granule with the touch. What the driver refuses is the honest
   ceiling and the span ends there.

   **The span is `C` less the dense weights, and the subtraction happens once.**
   `usable()` is `headroom.min(C − spent_by_us)`, where `spent_by_us` is the
   *drop in headroom since `C` was measured* — and the dense weights are now
   resident when it is called, so they are already inside that drop. Nothing
   further is subtracted for them. In particular the span must **not** also
   subtract `class_reserved(Weights)`: that is the same bytes booked twice, and
   it is the exact mistake this codebase has already made twice — once when
   `balloon_headroom_abs` reserved the transient peak that `expert_budget` was
   also reserving (cost: 1,104 MiB nobody could touch), and once when
   `kv_span_from` subtracted `scratch_margin` against a transient tier that was
   then added back on top ("same bytes, two places, opposite signs"). The
   ordering change makes the weights *visible* to the measurement, which is
   precisely why the second subtraction now looks tempting and is wrong.
4. **Fill expert slots leftward** from `span_end`, stopping at
   `min(all experts resident, span_bytes − MIN_ELASTIC_RESERVE)`. The floor is
   the only bound: the first forward recomputes the boundary exactly, so where
   it opens does not matter (§2). *Today the code stops at
   `INITIAL_KV_RESERVE` instead — the crutch §2 describes, which comes out with
   the phase lock.*
5. **Everything else** — gallery arena slabs, sampler / provenance / MoE-routing
   scratches, the threaded pipeline's combine target — allocates from the CUDA
   pool, outside the span, from `pool_cushion`.

Loading the dense weights first is what makes the rest of this document short.
The alternative — right-aligning them inside the span — needs the dense extent
known before the first byte is placed, needs the load path to route through a
zone allocator, and buys nothing: they are immovable either way, so they occupy
the same bytes whichever side of the reservation boundary they sit on. Ordering
them first replaces all of that with a subtraction that has already happened.

The one real code change is an inversion in `quantized_qwen3_moe.rs`: the expert
cache is built at ~line 2077, *before* the per-layer dense loop at ~line 2103.
Those swap. The combined progress denominator (`total_units`) survives the swap;
only the order of the two phases it spans changes.

### Why the pool cushion survives

The instruction was to allocate every buffer during load so the span can take
literally everything else. Two of those buffers are not sizeable at model load:
the gallery arena scales with the corpus, which is ingested after the model, and
the KV backings are constructed when a session picks its KV config. So the
cushion stays, and it is not a guess — it is the one term in the current budget
that has been measured directly and found correct
(`candle-core/src/vram/budget.rs`): the CUDA pool reserves **once** during load
(30 MiB → 7,232 MiB, three distinct values in 60 samples) and never grows again,
through cold ingest or six concurrent conversations. At 512 MiB the process
peaks exactly 512 MiB below `capacity_c` — the cushion, untouched at peak.

Keep 512 MiB. Driving it to zero by eagerly constructing every pool consumer is
a later step, and `forbidden_alloc` is the instrument for it: arm it for the
whole post-reservation process lifetime rather than only around the decode loop,
and every remaining pool allocation reports itself with a byte count. Read the
counts, not the labels — the third backtrace frame is unreliable under release
inlining, and a whole diagnosis has been built on a misattributed one before.

---

## 5. Balloon: claim more, leave exactly 512 MiB

Two independent leaks, both giving capacity away for nothing.

**The fast path never applies the reserve.** `run_balloon` skips the touch when
the card is already free — the normal case — and sets `C = headroom.min(total)`,
which is *everything the driver reports*. There is no reserve at all on the path
that runs in practice.

**The slow path stops on the first refusal.** `balloon_measure` grows in 256 MiB
chunks and `break`s the moment one fails. If the driver would accept 200 MiB but
not 256, that 200 MiB is never claimed, and the same is true at every step of
the fractional target.

The change:

```rust
/// Capacity target: everything except a fixed absolute reserve. One expression,
/// used by both the fast path and the growth loop, so they cannot disagree
/// about what "as much as possible" means.
fn capacity_target(total: u64, reserve: u64) -> u64 {
    total.saturating_sub(reserve)
}
```

- `balloon_target_frac` is **deleted**. A fraction of the card is not a fact
  about anything; the reserve is.
- `balloon_headroom_abs` becomes `capacity_reserve`, default 512 MiB, and is
  applied on **both** paths: the fast path clamps to
  `headroom.min(capacity_target(total, reserve))`.
- On refusal the growth loop **halves the chunk** and continues, down to the
  reservation's granule size, instead of breaking. The claim then ends within
  one granule of the true ceiling rather than within one 256 MiB chunk of it.
- `balloon_floor` is deleted: it was a second, deeper expression of the same
  reserve, and two of them cannot both be "the amount we leave".

On the 16 GiB dev card the fast path currently reports `C = headroom`; under the
change it reports `min(headroom, 15,864 MiB)`. Whether that is a gain or a loss
depends on what DXGI reports free, and the answer is measured in step 1 of the
build, not predicted here. The chunk refinement is an unambiguous gain wherever
the touch balloon actually runs.

---

## 6. The weight zone

### Geometry

Every expert slot is `max_expert_size` bytes — the max over all layers, already
computed at load. Equal-sized slots make the zone an **array**, not a heap:
there is no fragmentation to reason about, and "the rightmost free spot" is
"the lowest free index".

```rust
/// Slot `i` occupies [span_end − (i+1)·slot_bytes, span_end − i·slot_bytes).
/// Index rises leftward, so slot 0 is the rightmost slot in the span and the
/// highest live index is the frontier.
fn slot_base(&self, i: usize) -> u64 {
    self.span_end - ((i + 1) * self.slot_bytes) as u64
}
```

- `capacity` — how many slots fit between the current `W_max` and `span_end`.
  Dynamic; recomputed between waves.
- `free` — `BinaryHeap<Reverse<usize>>`, so `pop` yields the lowest free index,
  which is the rightmost free spot.
- `W = span_end − highest_live_index·slot_bytes`, or `span_end` when empty.

The zone's right edge is the end of the reservation itself — there is no dense
block inside the span to bound it, because the dense weights were placed before
the span existed (§4).

`ExpertCacheInner::free_slots` is a `Vec` used as a stack today, seeded
`(0..n).rev()` so the first `pop` is index 0 — it already allocates rightmost
first. What breaks the order is eviction, which `push`es the freed index onto
the top. Changing the type to the same `BinaryHeap<Reverse<usize>>`
`region_pool` uses restores the invariant on every path.

### Placement is not a replacement for the eviction policy

Position decides **where** an expert lands and **which** slots a retraction
reaches. It never decides **who** gets evicted: that stays the existing
temperature-based choice in `ExpertCacheInner::allocate_slot` and
`end_of_pass_eviction`, unchanged, including layer-awareness and the pinned
early layers.

(One correction of the record, because the doc has to be true about what it is
preserving: that policy is not LRU. `slot_eviction_score` is a decayed access
*frequency* — `+1.0` on a hit, `×0.85` per pass — multiplied by a mild
positional factor, with `last_used` only as a tiebreak. `cache.rs` says so
directly: "effectively LFU with a recency decay". The thing being preserved is
"eviction is by temperature, not by address", and that is preserved exactly.)

The two rules compose in one direction only:

- **Allocation always takes the lowest free index**, i.e. the rightmost free
  spot — whether the space came from an eviction, from a retraction that
  released a neighbour, or from the zone growing. If any slot is free, no
  eviction happens; the free list is drained before the policy is consulted.
- **Newly-gained space is therefore used last and lost first.** When the zone
  grows, the new slots appear at the highest indices and sit at the back of the
  heap. Nothing is wasted — every free slot is still used before any eviction —
  but the volatile margin stays empty for as long as the working set does not
  need it, which is what keeps the next retraction cheap.

### Retraction: relocate the hot, evict the cold

Retracting to a smaller capacity clears every occupied slot at index ≥ the new
capacity. There are two ways to clear one, and the choice is where the
temperature gradient comes from:

```text
doomed = occupied slots with index ≥ new_capacity, sorted by score, hottest first
relocate the hottest  min(|doomed|, free slots below new_capacity)  of them
evict the remainder through the existing eviction path
```

Relocation is a device-to-device copy of one slot — 2–3 MiB, a few microseconds
at card bandwidth — plus rewriting the one pointer that names it
(`key_to_slot` / `slot_to_key`, and the `gate/up/down` device pointers in the
dispatch table). It runs in the between-wave window, where §3 guarantees nothing
is reading the zone, so it needs no fence of its own.

Eviction of the remainder is always safe: expert bytes are read-only and a clean
copy exists on the host, in the pinned pool and in the mmap behind it. This is
the existing `ExpertCacheInner::evict` plus a zone-side free.

### Why the gradient emerges

The trend — hot experts drifting right into stable ground, churn collecting at
the left where the frontier moves — is not a property of the allocation order on
its own. In a full, stable cache, allocate-rightmost-free just refills the slot
its predecessor vacated, so the layout stays frozen at whatever fill order
produced it. Position only sorts itself when something moves, and two things do:

- **Retraction sorts directly.** Every retraction touches the leftmost occupied
  slots and asks one question of each: hot enough to carry, or cold enough to
  drop? Hot ones move right by construction. Cold ones leave. Each pass makes
  the left margin colder than it was.
- **Refill sorts by survival.** An expert that *is* evicted at the frontier and
  is genuinely hot comes back on the next miss — and lands at the rightmost free
  spot, which is further right than where it died. One that is not hot does not
  come back at all. Over many cycles the difference accumulates in one
  direction, because the frontier only ever destroys leftward.

Neither mechanism runs when the frontier is still, and neither needs to: with a
stable boundary no slot is at risk, so position carries no cost. The gradient
establishes itself exactly when it starts to matter, which is when the boundary
starts to move.

### Growth

When `A` retracts — sequences complete, regions return to the free list — the
zone's capacity rises and the new slots join the free list at the back, where
the next miss takes them only after every closer hole is used. The expert
cache's slot vectors (`slots`, `slot_to_key`, `last_used`) resize with it.
`ExpertCacheInner::new(num_slots, …)` gains a `resize(capacity)` sibling;
nothing else in the eviction policy changes.

### Ownership

The zone is owned by the expert cache (`ExpertCacheInner`), and it **is** that
cache's free list — there is no second one. That matters for more than tidiness:
`free_slots` was a `Vec` used as a stack, seeded `(0..n).rev()` so the first
`pop` was index 0, but eviction `push`ed the freed index onto the top, so the
rightmost-first ordering survived exactly until the first eviction.

The region pool holds only `W`, the address. It never reaches into the cache.

---

## 7. The cycle: the phase-locked forward

**Not yet built** — §7a records what stands in for it today, and why that is a
placeholder rather than an alternative.

A wave is a complete layer sweep, so this cycle runs **once per decode step,
every 57–80 ms** on the measured numbers. That cadence is what makes an exact
recomputation the right mechanism: the partition is recomputed from scratch
twelve to seventeen times a second, which tracks demand at the rate demand
actually changes.

```text
unlock
  1. ADMIT      every chunk slot the forward needs, at slot granularity:
                  the wave's KV for R rows, the elevations it will wait on,
                  and an estimate for persistence's quantize destinations.
                Growing right; on contact with W, evict weights.
  2. PLACE      A is now final, so T follows from `WavePlan` (§13a).
                Tier base = A. If A+T crosses W, evict more weights.
  3. SETTLE     W ← A+T. Everything above is known free — hand it to the
                weight side now, in the same operation.
lock
  4. RUN        forward and persistence proceed in parallel. Neither can
                expand the arena count; both allocate slots freely inside
                arenas already claimed.
unlock
  5. RELEASE    persistence gives back what it did not use.
```

### Why each phase is where it is

**Admit is exhaustive, and bounded.** `R` is the min of three things all known
before a byte is claimed: `MAX_PREFILL_TOKENS` (8192, where compute saturates),
`WavePlan::max_rows_within(T_budget)` (what the tier can hold), and **the KV
bytes claimable once the weights are allowed down to `MIN_ELASTIC_RESERVE`**.
The third term does not exist yet and is the missing piece.

Because `R` is capped by what is claimable, **phase 1 cannot overflow** — the
thing that would have made it fail was already the cap. The byte cap and
`MIN_ELASTIC_RESERVE` are the same constraint from two ends: the floor says how
far the weights may retract, which says how much KV is claimable, which caps `R`.
One number, one direction, checked once.

**Placing at `A` is the optimal moment and the optimal position**, and both
halves of that matter. After admit, `A` is final and `T` follows from the plan,
so `A+T` is the smallest `W` this forward can need — the partition is *computed*,
not estimated. ("Exact" would overstate it: the plan charges the union of the
chains a phase can run and bounds the routing tables rather than transcribing
them, so `T` is a tight upper bound, not an equality — §13a.) Placing the tier hard against `A` leaves the entire remainder in one
contiguous run adjacent to the weight side, which is what lets phase 3 hand it
over immediately. Placed anywhere else the same bytes are free but stranded in
the middle, and the weights can only discover them later, through a control loop
(§13).

**Evictions happen only in phases 1–2**, before the lock, while the pipeline
thread is idle — which is what makes a synchronous retraction safe there and
unsafe anywhere else. During phase 4 nothing is evicted for space; the expert
cache still swaps experts within its own fixed extent, exactly as it does today.

### The lock freezes the boundary, not the allocator

The distinction is the whole reason the lock is cheap:

| operation | gated? |
|---|---|
| `claim_region` — expanding the arena count | **yes**: this is the only thing that can move `W` |
| slot allocation inside an existing arena | **no**, for anyone, at any time |
| the wave's own slots | already allocated in phase 1; it needs neither during the run |

**The tier's ceiling binds *both* of `claim_region`'s paths**, and missing that
is a silent corruption. A region freed while the tier was small — or absent,
between forwards — keeps its index, and the next wave's tier can be placed below
it; handing it out then puts KV writes inside the wave's own intermediates, and
nothing downstream notices until the output is wrong. The first build gated only
the fresh-region path and left `free.pop()` unconditional (§14). One comparison
closes it: the free list is ordered lowest-index-first, so if the lowest free
region is above the ceiling then every free region is.

**The ceiling is only half the rule, and the other half is at the placement.**
A ceiling governs claims made *after* the tier is placed. It says nothing about
claims made **before** it and still live when it lands — and those are the same
corruption arriving from the opposite direction. `place_transient` measures the
tier down from `W`, so it occupies the *highest* regions of the KV side: exactly
the ones a KV side approaching full is using. The original fit test bounded the
tier below by the start of the KV side, which only refuses a tier wider than the
entire span; every partial overlap passed. A region cannot be recalled — an arena
holds its address for as long as it lives — so nothing downstream can repair it,
and the wave writes its intermediates over live KV.

The placement therefore checks the **live watermark**, not `region_base`, and
refuses rather than overlaps (principle 7). `set_weight_floor` already guarded
the identical invariant from the weight side, for the identical reason; the two
frontiers that can reach the arenas now both do.

Two things keep that refusal rare rather than routine:

- **The ceiling survives the release.** `tier_reserve` holds the last tier's
  width back from the KV side even while no tier stands, so the persistence
  thread cannot take the ground the next placement needs during the gap between
  forwards. This is not the fixed 912 MiB reservation returning: it is the width
  of the tier that actually ran, which for a decode wave is a few regions.
- **`spare_regions` already prices the tier into demand**, so the weight side
  cannot take ground the next wave's tier will need either.

Neither is a guarantee — a decode wave followed by a wide prefill widens the tier
against a reserve sized for the narrow one — which is why the check at the
placement is the thing that has to be right, and the reserve is only what keeps
it quiet.

So the chunk allocator is not frozen — it is prevented from *expanding arenas*
until the sweep finishes. A region is 16 MiB of many slots, so admitting the
wave's KV claims regions that arrive with spare slots inside them, and the
persistence thread seals into those throughout phase 4 without touching the lock
at all.

### The persistence thread aligns with the wave, then decouples

It claims what it needs in the unlocked window and does its work in parallel
while the forward runs. Two kinds, split along a line that already exists:

- **Elevations the wave will wait on** (warm→hot for provenance-selected chunks)
  are part of phase 1. The forward blocks on these via `migrate_flight`, so they
  must have their regions before the lock — that is what removes the deadlock
  rather than managing it.
- **Background seals and hot→warm demotions** are not waited on by anything. In
  the common case they never reach the lock, because spare slots cover them. On
  genuine exhaustion the request blocks until the next unlock — at most one
  forward, ~60 ms, for work nothing is waiting on.

This is why the estimate in phase 1 is worth carrying: it makes the spare-slot
headroom deliberate instead of whatever region rounding happens to leave.

---

## 7a. What stands in for it today

The built code moves the boundary at the expert pipeline's **end of pass**,
driven by a feedback signal rather than an exact computation. That was not a
design decision so much as the consequence of one constraint discovered during
the build, and it is worth recording because the phase lock is what dissolves it:

**`claim_region` runs on whichever thread needed an arena. The expert cache is
owned by the pipeline thread, and its slots may be under read by kernels still
in flight.** A synchronous retraction from the claim path is a cross-thread call
into a cache that is mid-wave, to evict memory a GEMM may be reading.

With no phase in which the pipeline thread is *known* idle, the two halves had
to be separated:

### The KV side records; it does not act

`claim_region`, on exhaustion, increments a demand counter and returns `None` —
the caller's existing pressure path. Nothing else changes on that thread.

### The weight side acts, at its own safe point

`renegotiate_boundary` runs at the expert pipeline's **end of pass**, after the
last MoE layer, where the pipeline thread owns the cache and no GEMM for the pass
is still being issued. It drains the counter and moves `W`:

- **Concede** (`wanted > 0`): retract by that many regions. The zone relocates
  the hottest doomed occupants into free slots below the new frontier and evicts
  the rest to the pinned pool — the worst case is a reload, never a loss.
- **Take** (`spare > 0`): grow into regions the KV side has not needed *lately*
  (below), by at most `KV_GROW_STEP` (8) per pass.

The cost is **one pass of latency** on the concede direction. That is what buys
an eviction that cannot race a kernel, and it is why the boundary opens at a
known-good position (§2) rather than at the floor: a failing arena claim does not
have a pass to wait.

### "Lately" is wall-clock, and that was measured the hard way

The first build decayed the KV high-water mark per pass (×0.9). The gate's
single-context configs run for seconds and leave the KV side almost entirely
free, so the weight side took **224 of 291 regions**, and the twenty-context
config that followed failed outright. Per-pass decay asks "has KV been idle for a
few forwards?" — which a benchmark answers yes to constantly, and which says
nothing about the next workload.

What governs KV demand is sessions arriving and leaving, on a scale of minutes.
So the mark rises to the live count immediately and falls only on a **five-minute
wall clock**. Fast to concede, slow to take: being short of KV fails a forward,
being short of experts is a slowdown, and the two are not worth trading
symmetrically. A daemon converges toward maximum residency over its first hour; a
benchmark that sweeps sixteen configs in two minutes never moves the boundary,
which is the correct answer for it.

### The latch

One rule, checked in one place: `set_weight_floor` refuses while any wave
generation is open. The call site already guarantees it — `renegotiate_boundary`
runs only from `post_compute` — so this is the structural property made
checkable rather than trusted (principle 7).

The broader latch — region claims refused during a wave — is **the phase lock of
§7**, and it is not built. While the boundary moves only at end-of-pass it is
genuinely unnecessary: a region claim during a wave touches nothing the weight
side owns. Under the phase lock it becomes the central mechanism, because the
whole point is that the partition the wave was locked against cannot change
underneath it.

Which is the same reason §7's admit phase has nothing enforcing it yet, and will
not stay that way. Arena expansion has exactly **one** path —
`alloc::claim_slab`, reached from `create_arena`, reached from
`alloc_chunk_with_arenas` — so the gate itself is one comparison in one place.
What is not one place is the *claiming*, and the two halves of the wave are in
very different states:

- **Decode already admits exhaustively.** `ensure_for_batch_entries_all` runs
  once per decode step, before the layer loop, over every layer's backing. It
  was hoisted for an unrelated reason — 48 lock acquisitions per token where the
  answer is almost always "nothing to allocate" — but it is exactly phase 1, and
  `A` is already final before layer 0 on that path.
- **Prefill claims per layer, and hoisting it means hoisting a partner.** The
  ensure lives inside `paged_prefill_batched`, and immediately before it
  `forward_attn_batched_multi` calls `truncate_caches_to_offset`, which discards
  the stale tail chunks a re-prefill at the same offset left behind. The order
  matters: truncate, then allocate. Lifting only the allocation above the layer
  loop would ensure capacity for layer 0, then have layer 0's truncation free
  chunks the pre-pass had just claimed — so both have to move together, as one
  all-layers pre-pass, and that is the shape phase 1 takes here.

This is worth stating because "make admit exhaustive" reads like a matter of
moving one call, and on the decode side it already is. On the prefill side it is
a small refactor of the prefill path's entry sequence, and the partner it has to
bring with it is not visible from the allocation site.

---

## 8. The hazard the fixed addresses introduced

Not in the original design, and the most serious thing the build turned up.

A slot used to be a fresh CUDA-pool allocation per load (`stream.alloc` in
`load_repacked`), and the old buffer was returned with `cuMemFreeAsync`. **The
pool supplied an ordering guarantee for free**: `cuMemAllocAsync` returns memory
whose free has retired in stream order, so a reused buffer could not be written
before its last reader finished.

A fixed-address slot has no such guarantee, and the eviction policy makes the
collision likely rather than rare: `allocate_slot`'s behind-layer scan prefers
experts from layers already executed *this pass* (`PINNED_LAYERS <= layer <
current_layer`), and "already executed" means **issued**, not retired. So the
previous layer's GEMM is exactly the kernel most likely to still be reading the
slot the next layer's H2D is about to overwrite.

The fix is the cheap half of what `region_pool::claim_region` already does for
regions: before a batch of loads, `order_copies_after_compute` records an event
on the compute stream and has the copy stream wait on it. GPU-side only — the
host does not block, exactly as it did not under the pool.

**It costs throughput**, because it serialises the copy stream behind compute
already issued, which is the overlap the copy stream exists for. Measured at the
first attempt as ~14 % on `F16×1` and 3–5 % elsewhere. §14 records the cheaper
scheme that keeps one layer of overlap, and why it was not built tonight.

---

## 9. What this deletes

Per the repo's standing rule, these are replaced rather than kept alongside:

| Deleted | Why |
|---|---|
| `GovernorConfig::kv_floor_abs`, `kv_floor_pct`, `VramGovernor::kv_floor()` | The partition is no longer a number. |
| `VramGovernor::expert_budget()` | Slot count *is* the zone's capacity. Nothing divides a budget by an expert size any more. |
| `num_slots` derivation in `quantized_qwen3_moe.rs` (~line 2046) | Same. |
| `pending_dense_bytes` pre-declaration (~line 1983) | It exists to stop the KV side paying for weights the loader has not reached yet. Under the new order there are none: every dense tensor is resident before the span is sized, so the measurement sees them instead of a forecast of them. |
| `GovernorConfig::balloon_target_frac`, `balloon_floor` | One reserve, one expression (§5). |
| `region_pool::kv_span_from` / `kv_span_target` / `floor_deficit` | The span is everything; there is no partition arithmetic left to get wrong. |
| `TRANSIENT_SPAN_BYTES` as a fixed carve | The transient extent is per-wave. |

`scratch_margin` survives as `pool_cushion`, renamed to say what it is: the
memory left *outside* the reservation for the CUDA pool, not a cushion held
back from anything.

`TEST_KV_SPAN_BYTES` survives — a test binary with no governor still needs a
fixed span, and the reason it is a constant rather than a fraction of free
memory has not changed.

---

## 10. Why this is stable under load

The mechanism has a property worth stating plainly, because it is the reason to
prefer it over a better-tuned constant.

**Light KV → more weights → higher hit rate → faster decode.** A warm daemon at
70 live regions hands the expert cache ~3.4 GiB the old floor was holding for a
boot transient. On the measured curve that is on the order of a thousand extra
slots.

**Heavy prefill → fewer weights → more misses.** But a wave carrying thousands
of prefill tokens does far more arithmetic per expert load than a decode step
does, so the misses it pays for are the cheapest misses the engine ever takes,
and they are amortised across every token in the wave.

So the system does not have a worst case where it falls over; it has a worst
case where it gets *slower per weight load* at exactly the moment weight loads
are cheapest per token. Instead of an OOM at the boundary, there is a smooth
exchange of residency for width. That is a better failure shape than any
setting of `kv_floor`, and it is why the boundary is worth making elastic even
if a perfectly-tuned constant existed.

---

## 11. Verification

### Arithmetic, no GPU

- **Zone geometry.** `slot_base(i)` is `span_end − (i+1)·slot_bytes`; slots are
  disjoint and ascending leftward; capacity from a frontier is exact.
- **Allocation order.** Repeated alloc/free yields the lowest free index every
  time, including after out-of-order frees — the property `free_slots` as a
  `Vec` breaks today.
- **Retraction.** Retracting to capacity `n` clears exactly the slots at index
  ≥ `n` and no others; `W` afterwards equals the new frontier. With free slots
  available below `n`, the **hottest** doomed occupants are relocated into them
  and the coldest are evicted — asserted against scores, not against indices,
  since the whole point is that position is not the eviction criterion.
- **The free list is drained before the policy.** A load with any slot free
  never evicts, whatever the scores say, and takes the lowest free index —
  including when the free slot came from the zone growing, in which case the
  closer hole wins over the newly-gained one.
- **The gradient.** A scripted sequence of retract/grow cycles over a mixed
  workload (a few experts hit every pass, the rest sampled) ends with the hot
  set's mean slot index strictly lower than the cold set's, from a randomised
  initial layout. This is the property §6 claims and the one most likely to be
  quietly false.
- **The tier ceiling binds recycled regions, not just fresh ones.** Every index
  at or above the ceiling addresses memory inside the placed tier, whichever
  path handed it out (§7). This one is worth more than the others: the bug it
  pins produced no error, only wrong output.
- **The frontiers never cross.** Over a randomised sequence of region claims,
  region frees, slot loads and retractions: `A + transient ≤ W`, and the elastic
  middle never falls below `MIN_ELASTIC_RESERVE`.
- **The dense weights are subtracted exactly once.** Against a scripted probe:
  measure `C` on an empty card, consume `D` bytes, then assert the span comes
  out `C − D − pool_cushion` — not `C − 2D − pool_cushion`. Drive it through a
  governor with `set_class(Weights, D)` *also* recorded, so a future change that
  reaches for the tally fails here instead of silently halving the span.
- **Balloon target.** `capacity_target` applied identically on both paths;
  chunk halving reaches within one granule of a scripted ceiling that a fixed
  chunk misses by up to `chunk − 1` bytes (`FakeBalloonAllocator` already
  scripts a ceiling).
- **The plan covers every chain that can run in its phase.** Stated against
  per-row constants read off `wave_census` on the real model — not recomputed
  from the declaration, which is a test that would have passed while the plan was
  wrong by 1.8× (§13a). The union's margin over the widest single chain is pinned
  too, so an over-bound cannot grow unnoticed either.

### GPU

- **The latch.** `set_weight_floor` refuses while a wave generation is open —
  the one rule that must hold for a moving boundary to be safe, checked in the
  one place every move goes through.
- **The transient tier does not move.** Its base and end are identical at every
  boundary position (`the_transient_tier_is_fixed_wherever_the_boundary_is`),
  which is what lets a `BumpRange` stay a bare pointer.
- **Ground becomes regions one for one.** Every megabyte the weight side gives
  up is exactly that many regions of KV, modulo the slack the rounding accounts
  for.
- **Round trip.** `test_parallel_batched_forwarding` (Qwen3-30B-A3B) as the
  gate — 16 configs, all valid, against a back-to-back baseline (§12).
- **The gradient, live.** The score-weighted mean slot index
  (`WeightZone::score_weighted_mean_index`) after a run that has actually moved
  the boundary. §6 predicts it falls over time; on a real workload it is the
  only evidence that the sorting mechanism survives contact with the real
  routing distribution rather than the scripted one. **Not yet observed** — the
  five-minute decay means no benchmark short enough for CI moves the boundary
  at all (§13).

---

## 12. What was measured

`test_parallel_batched_forwarding` (Qwen3-30B-A3B, 16 configs), release + cuda,
on the RTX 4090 Mobile 16 GiB. Baseline is the same test on `a8ed408e` with the
change stashed, run back to back on the same machine.

| config | baseline t/s | built t/s | Δ |
|---|---|---|---|
| BF16 ×1 | 513.5 | 518.6 | +1.0 % |
| BF16 ×10 | 1975.8 | 1967.9 | −0.4 % |
| Q8_0 ×20 | 2498.8 | 2447.9 | −2.0 % |
| Q4_0 ×4 | 1791.7 | 1812.9 | +1.2 % |
| C0 ×2 | 958.5 | 971.9 | +1.4 % |
| C6 ×2 | 956.3 | 975.3 | +2.0 % |
| Q4_0 ×20 | 2484.9 | 2406.0 | −3.2 % |
| **whole run** | **127.7 s** | **122.0 s** | **−4.5 %** |

All 16 configs valid on both. The `F16×1` row is omitted: it is the first config
of the run and swings 380–437 across repeats of the same build.

**`C` rose 14,592 → 15,536 MiB (+944 MiB)** from the balloon change alone, on
this card, measured at the reservation line.

### After the per-wave tier (§13a)

| config | fixed-constant tier | plan-sized tier |
|---|---|---|
| BF16 ×1 | 507.7 | 515.5 |
| BF16 ×10 | 1888.6 | 2143.5 |
| Q8_0 ×20 | 2318.1 | 2210.1 |
| Q4_0 ×4 | 1794.2 | 1812.1 |
| whole run | 134.9 / 148.6 s | 130.4 / 146.9 / 153.3 s |

**Read the whole-run column as noise, not as a result.** Repeats of an unchanged
binary spanned 134.9–158.3 s, and the plan-sized build's three runs span the same
range and include the fastest of the lot. A wall-clock difference of a few percent
is not measurable here without many more repeats, and no claim in this document
should rest on one.

Per config the two are within their own repeat spread, with `BF16×10` the only
row that moves by more than the spread — and in the right direction. The result
worth recording is not the throughput at all: it is that the tier stopped
reserving 912 MiB per wave and started reserving what the wave uses.

---

## 13a. Completing `WavePlan`, and the instrument that had to be built first

This was the precondition for §7 and it is now closed, but *how* it was closed
matters more than the numbers, because the same trap is set for anyone who
changes the attention or FFN chain again.

### The instrument named in the design was the wrong one

The plan's doc claimed "a site that is not a variant here is a site still
reaching the driver", and offered `candle::forbidden_alloc` as the check. That
was true when it was written and **operand provenance made it false**: an op
reading a wave-backed operand now carves its output from the same generation, so
an undeclared buffer never reaches the driver, never appears in a
forbidden-allocation report, and costs the span exactly as much as a declared
one. Arming `forbidden_alloc` over the wave path reports nothing at all.

So the check had to be built: `candle-nn/src/kv_cache/chunked/wave_census.rs`.
A phase span is reset when its generation drops, so one generation is one
layer's phase and its final cursor is that layer's cost. The census records the
ranges handed out within a generation and prints them **when that generation sets
a new high-water mark for its chain** — keyed by carve count, so the attention
span's twelve-carve prefill chain and its nine-carve decode chain each report
their own worst case rather than the wider one hiding the narrower. Two modes:
`KV_WAVE_CENSUS=1` gives sizes, which is enough to *track* an inventory that is
already written down; `KV_WAVE_CENSUS=labels` symbolises the caller of each
carve, which costs a fifth of the gate's wall clock and is what you need to
*establish* one.

Sizes alone nearly sufficed — every buffer is `rows × cols × width` for known
dimensions, so most fall out by arithmetic — but not entirely, and the two that
did not were the ones worth getting right. Two different buffers came to the same
byte count, and guessing which was which would have put a name in the plan that
the code did not have.

### What it found

The attention phase priced at 33,024 B/row and allocated 58,624 on the prefill
chain — **1.8×**, and the failure was the one the gate saw:

```
wave-attn: transient span exhausted — 6094848 B at offset 37521408
           exceeds the 38300928 B budget
```

The undeclared buffers were not exotic. Q and K each come out of the fused QKV
projection as a **strided narrow**, so every reshape around the head-wise RMSNorm
copies: split, flatten, norm, transpose back — four `attn_cols`-wide buffers for
Q and four `kv_cols`-wide for K where the plan declared none, plus V's own
contiguous copy. Meanwhile `OProjAccum` and `OProjCast` were declared and **do
not exist**: on the prefill chain `o_proj` takes a `Float` context and the int8
override quantizes it at the matmul, and that quantize breaks the provenance
chain, so prefill's `o_proj` output lands on the pool. On the decode chain it is
a single compute-dtype buffer.

Two more, both structural:

- **The accumulate dtype was declared BF16 and measured F32.** The expert GEMM
  outputs came back at four bytes an element. Gate, up and down are the three
  largest buffers a MoE layer allocates, and every one of them was priced at half
  its size.
- **The FFN span was silently spilling.** At the widest wave the gate runs, the
  FFN generation sat at 536,869,760 B of a 536,870,912 B span — 1,152 bytes
  short of the cap — and the tail of the second expert batch simply fell back to
  the pool, because `resolve_wave_alloc` returns `None` on exhaustion and a pool
  allocation is always a correct answer. The measurement was clipped by the
  constant it was supposed to justify.

### What the plan says now

Fourteen attention variants and thirteen FFN ones, every one read off the census
rather than inferred, with the chain that allocates it named in its doc comment.
The pinning test states the measured per-row cost of each chain as a constant and
asserts the plan covers it — deliberately not by recomputing the declaration,
which is a test that would have passed throughout.

Two judgement calls are recorded in the code as such. The plan charges the
**union** of the chains in a phase rather than their maximum, because
`forward_layer_batched_mixed` opens one attention generation and runs every group
inside it — a mixed wave really does allocate both chains' buffers. That costs a
pure-prefill wave 14.9%, which the test pins so it cannot drift unnoticed, and
closing it means passing admission's per-group split rather than a total row
count. And `ROUTING_U32_PER_ASSIGNMENT = 8` is a **bound rather than a
transcript** — the expert pipeline's per-batch table uploads measured 3.5 u32 per
assignment and eight is that doubled, costing 0.19% of the phase, because being
short here fails a forward and being generous is invisible.

### What it bought, and what it has not

The tier is now sized per wave: a twenty-session decode places about 50 MiB
across its three spans where the fixed constants reserved 912 MiB, and a wide
prefill places about 465 MiB — *less* than the constants, and without the pool
spill. Per-config throughput is flat to slightly better.

**The reclaim is still unspent.** The tier gives the ground back on every
forward, but the boundary only moves at the expert pipeline's end of pass on a
five-minute decay (§7a), so within a benchmark nothing takes it. That is §7 phase
3's job, and it is the next thing to build.

---

## 13b. Admit is exhaustive; anchoring the tier at `A` is not the next line of code

Phase 1 of §7 is **built** (`candle-transformers/src/models/wave_admit.rs`) and
phase 2 was built, measured, and reverted. Both halves are worth recording,
because the second one's obvious diagnosis is wrong and someone will reach for it
again.

### Phase 1: built, and the partner it had to bring

Decode already admitted exhaustively — `ensure_for_batch_entries_all` runs over
every layer's backing before the forward is entered — and glue claims nothing,
because a glue row reprojects tokens that are already resident. Only prefill
claimed per layer, and hoisting it meant hoisting `truncate_caches_to_offset`
with it: the truncation runs immediately before the claim and discards the stale
tail chunks a re-prefill left, so lifting only the claim would have had layer 0's
truncation free what the pre-pass had just made. The order is preserved exactly —
truncate, reset a sequence starting from zero, claim — and only the *when*
changed.

Two counters were added to `RegionStats` to prove it rather than assert it:
`fresh_claims_during_wave` (claims taken after a wave began) and
`refusals_during_wave` (claims the tier's ceiling blocked while room existed).
Both read zero across the gate.

**Read that carefully, because the first version of this section over-claimed
it.** Those counters only observe the window in which `transient_base` is set,
and the tier is *released between every layer phase* — `release_if_last` clears
the base when the last guard of a phase drops. So they say the wave's own claims
are all made up front. They do **not** say the arena frontier is stationary for
the duration of a forward: a persistence claim landing in the gap between two
phases is invisible to both counters and still advances `next`.

That blind spot sits exactly where the §13b failure lives, and closing it is what
[`plan_wave_transient`] now does — the tier is reserved for the **forward**, not
for each phase, so the ceiling holds across the gaps and the counters see
everything.

### Phase 2: refuted, in two attempts

**First attempt.** Anchored at the frontier, `Q8_0 x20` produced silently wrong
output on every session, reproducibly. The counters said the region pool was
untouched — identical arena creations (577, max index 163), zero class
promotions, zero refusals, zero fresh claims taken mid-wave — and three
controlled variations narrowed it further: poison-filling the tier at every
placement passed, a device-wide quiesce when the base moved failed, and stamping
the released footprint so `claim_region`'s wait covered a fresh claim on it
failed. What the poison had that the others lacked was a **synchronous** memset
on **every** placement, roughly a hundred per forward.

That pointed at the tier's base walking forward *between phases*, which the
counters could not see (above). So the fix was to stop it walking:
`plan_wave_transient` now reserves the tier for the **forward**, once, and
`begin_wave` lays the three spans out inside a reservation whose address it does
not choose. That is a good change on its own — it closes the blind spot, holds
every phase of a sweep at the same offsets, and is kept.

**Second attempt**, on top of that. The silent corruption became a loud refusal,
and the refusal said what the first attempt could not:

```
[tier-refusal] region claim refused with the tier placed at 0xb21e00000
               (ceiling 16, next 16, total 332) — count 1025
```

**1025 refusals inside a single forward, at one placement.** The gap between the
frontier and the tier would have to be hundreds of regions, not a headroom.

### Why, and why no admit phase fixes it

The quantized KV path creates size-class arenas **as it compresses**, on the
persistence thread, throughout the forward. §7's premise for phase 2 is that
persistence takes spare *slots* inside arenas already claimed and only rarely
needs a region of its own. On the quantized path that is false, and it is not
false by an amount an admit phase could close: the destination arenas are not
knowable until the compressor has chosen formats, which it does from the data.

So the tier stays measured down from `W`. **The reclaim does not depend on this**
— it comes from `bytes` being this wave's price rather than the widest wave's —
and the placement's other job, handing the freed run to the weight side, has to
be done by moving `W` itself. Which brings its own surprise.

### The exact boundary is untested, not refuted — and the gate cannot test it

An earlier revision of this section claimed the exact boundary was disproven,
citing a monotone watermark that made **every** gate config fail validation. That
claim is **withdrawn**, and the reason it was wrong matters more than the claim.

`test_parallel_batched_forwarding` lives in candle-transformers, which cannot
depend on candle-conversation, so it drives `BatchedInference` directly and
admits its twenty contexts **unconditionally**. There is no `admit_budget`, no
ceiling, no throttle. The daemon is a different system entirely
(`scheduler/prefill.rs`):

```text
admit_budget_ceiling = (free_regions − setpoint) × region_bytes
```

read live from `region_stats()` on every admission decision. Session admission
already tracks the moving boundary.

So a harness with no admission control failing when the partition shrinks under
it is the *expected* result and says nothing about the design. What was measured
was the harness, not the boundary.

Two lessons, and the second is the general one:

- **The gate is the wrong instrument for anything about the partition moving.**
  It proves correctness and throughput at a fixed partition. Questions about
  admission have to be asked of the daemon.
- **A refused claim is the correct failure when the budget is sound.** Reaching
  for a mechanism to absorb refusals — a claim that waits for the next boundary
  move — was treating the symptom. The question is always why something was
  admitted that could not fit.

Which points at the actual defect, below.

### The real gap: the compressor's arena demand is not budgeted

§7 phase 1 says admit claims "the wave's KV for R rows, the elevations it will
wait on, **and an estimate for persistence's quantize destinations**". The first
term is built (`wave_admit`). The third is not, and that is what the 1025
refusals were: the compressor creating size-class arenas as it seals, against a
budget that never charged for them.

Every token admitted eventually needs somewhere to put its *compressed* copy, so
the KV-fits term has to price the raw chunk and its quantize destination
together. Admitting on the raw cost alone over-commits the partition by whatever
the compressor will later ask for — and then the refusal is not the partition
protecting itself, it is the bill arriving late.

**Built**: `admission::per_block_seal_bytes` charges the active pair *plus* the
sealed pair, because sealing allocates the destination, writes it, and only then
releases the source — the two coexist, and the sealed classes are different
classes that start with no arena. Note the symmetry with the mistake already
recorded on `Scheduler::per_block_kv_bytes`: pricing the sealed pair alone
understated a live sequence by ~3.7x. Both errors are the same shape — charging
for one of two states a block passes through — and the fix for both is to charge
for the moment they overlap.

### The frontier anchor still cannot be tested here, and now for a clear reason

With the compressor priced, the anchor was re-run behind `KV_TIER_FRONTIER=1`
and failed on the first forward:

```
every region of the KV reservation is occupied (4 live) —
nothing left to stamp for class 4096 B
```

The gap between frontier and tier has to be **seeded**, not learned: it starts at
zero, the first claim after the reservation is refused, and the forward dies
before the measurement that would widen it.

So a seed was built. `set_claim_reserve` carries a per-forward prediction into
the placement, and the tier lands at `frontier + max(seed, observed peak)` —
prediction for the cold case, observation for the case where a backlog makes the
compressor run ahead of the wave. The forward bounds its own prediction by what
it *writes*, since a sealed copy is never larger than the active block it
compresses.

Two defects surfaced on the way and both are fixed, because both were real
regardless of the anchor:

- **An unplanned `begin_wave` pinned the tier for the process.** Reservations
  became forward-scoped in the fix above, but a caller that never priced itself
  has no forward to hand one back — so a helper opening a span at load left the
  tier standing at whatever the frontier was then. Invisible at `W − T`, fatal at
  the frontier. Reservations now record whether a forward owns them.
- **The anchor applied where no seed existed.** Those same unplanned callers
  predicted nothing, so the anchor gave them a zero gap. It now applies only
  where a reserve has been set, which is exactly the signal that a forward is
  behind it.

With both fixed the anchor still does not survive the gate, and the trace says
why:

```
[arena-create] index=0..3          <- the gap, consumed
[tier-refusal] ... (ceiling 4, next 4, total 332)
```

The seed was four regions and the first forward claimed four before asking for a
fifth. Not a wrong shape — a cold start allocates its backing lazily, so demand
arrives that no prediction from *this wave's* token count covers, and the
observation that would correct it only exists from forward two onward.

**This is where the gate stops being able to help.** Its harness sets caches up
lazily inside the first forward; the daemon establishes them through admission,
which is also where the seed's real source (`per_block_seal_bytes` against the
in-flight backlog) lives. Both remaining questions — can the anchor sit at `A`,
can the boundary be exact — are questions about admission, and **the gate has
none**. The anchor is left switchable rather than deleted so that asking again
costs one environment variable.

---

## 13c. The daemon run, which is where this was finally measured

Four runs of `zend --wipe-substrate` against the candle workspace, plus one of the
pre-change baseline (`a8ed408e`) for comparison. The rebuild is the intended
workload: small prefills, large prefills, decodes, interleaved.

### A correction to how this section first read

An earlier draft compared against `a8ed408e` — the commit *before this whole
branch* — found the rebuild failing on both, and concluded the workload exceeds
the hardware. **That inference is invalid.** The branch is where the iteration
lives, so the baseline failing carries no information about the current state;
all it shows is that the branch has moved. The comparison below is kept because
the *KV capacity* numbers are a real measurement of what the per-wave tier buys.
The conclusion drawn from it was not, and is withdrawn.

What the run actually ended in was a **stall, not a ceiling**: `(no forwards)`
against a 64 MiB admission budget with a 14,314-token backlog. Nothing was
admitted, so nothing completed, so nothing was freed, so the budget never
recovered. Two causes, both introduced by this work and both now reverted or
fixed — see §13d.

### What it is worth

| | baseline `a8ed408e` | this branch |
|---|---|---|
| KV regions | 275 (4400 MiB) | **328 (5248 MiB)** |
| transient tier | 912 MiB, fixed | **64 MiB** on a decode wave, 496 MiB on a wide prefill |
| expert slots at open | 2377 | 2377 |

**+53 regions, +848 MiB of KV — a 19% increase — bought entirely by pricing the
tier per wave instead of reserving the widest case.** That is the reclaim, on the
real workload, and it is the number this document exists to produce.

### The constraint this uncovered, which belongs to another document

The retraction failure below is not really a bug in this design. It is this
design meeting a **structural limit in the expert cache**: its two tiers are
mutually exclusive, so an expert leaving VRAM must be *copied* to pinned RAM, and
pinned has room for `total_experts − vram_slots` plus about two layers of churn.
Measured, that leaves **0–109 usable slots** for the boundary to move — which is
why a retraction asking for 4436 regions delivered `relocated=0 evicted=0`.

No tuning in this document reaches that. The boundary cannot move while eviction
needs a destination slot. `docs/expert_cache_design.md` is the fix — a cold tier
that always holds a copy, so eviction becomes a drop — and it is a **prerequisite
for this design, not an optimisation of it**. Everything below is what was built
to stop the limit doing damage while it stands.

### Three defects the gate could never have found

**The retraction could destroy the expert cache.** The pinned pool is the expert
cache's *backing store*, not an overflow area: it holds fewer slots than the model
has experts, so VRAM residency has a hard floor at `total_experts − pinned_slots`
— 6144 − 4004 = 2140 slots here. Below it, some expert exists only in VRAM.
Nothing enforced that. Under KV pressure the boundary retracted past it, every
eviction failed with `pinned pool full`, **the failures were logged and the
boundary moved anyway**, and the experts were gone from both tiers. Every
subsequent forward reported `Expert cache full, cannot evict (all pinned)`,
forever. Now the concession is capped at the free pinned slots, and a failed
eviction fails the retraction instead of being narrated.

**The weight side stole ground from a starving KV side.** Twice, for two
different reasons, and the second is the interesting one. A history-based mark
does not see the present, so it was paired with a live free-region count — and
that still gave eight regions away ninety seconds into a rebuild that then
saturated, because during an **ingest demand climbs monotonically**: the window
has not caught up, this instant has room, and a moment later there is none.
Neither history nor occupancy catches that. The third guard is the KV side's own
voice: a refusal is it asking for more and being told no, and a side refused
within the last window has no spare ground whatever the other two say. With it,
the weight side took **nothing** during the ingest, and the error count fell from
21,059 to 6,920.

**The five-minute decay was replaced by a sixty-second sliding-window maximum.**
Convergence is one window rather than three halvings, and there is no rate
constant to choose — only how long quiet must last to count as quiet.

### What the run does not fix, and why that is not this document's problem

The rebuild does not complete, on this branch **or** on the baseline. It ends
with the KV side full, `backlog=14314tok`, and relief reclaiming nothing:

```
turns_compressed=0 turns_evicted=0 arenas_released=0 relieved=false
```

Nothing is compressible because the resident KV is almost entirely **sections** —
the base conversation's prelude, tool catalog and ingested content — which are
permanent by design, not turns that can be sealed and evicted. The workload's
working set exceeds the card.

The shape of the card is worth stating because it bounds what the boundary can
ever do, even though it was **not** the cause of this run's failure:

```
6144 experts x 2.9 MB   = 17.8 GB of expert weights
pinned RAM backing      = 11.6 GB   (host RAM 31.5 GB, 0.2 GB free)
=> VRAM must hold       >= 6.2 GB of experts   (the pinned-pool floor)
=> KV ceiling           ~= 5.7 GB
```

---

## 13d. Two stalls this work introduced, and what they teach

Both were found by the daemon and both are gone. They are recorded because they
share a shape: *a conservative-looking change that removes the system's ability
to make progress at all.*

**Admission was charged twice for every block.** §7 phase 1 asks admit to account
for "persistence's quantize destinations", and the obvious reading is that a
block occupies its active slot **and** its sealed destination while the
compressor copies between them — so charge both. Built as
`per_block_seal_bytes`, and wrong: the overlap lasts one copy, the charge lasted
the block's whole life. It doubled the price of every block in every admission
decision, every in-flight accounting, and the decode reserve — so admission
cleared roughly half the work it should, and under a cut budget, none.

The lesson is about *shape*, not magnitude: a transient double-occupancy is a
**reserve** — a fixed pool the compressor draws on — not a per-block tariff.
§7's third term is still unbuilt, and it should be built as a reserve.

**A width cap could reach zero.** `prefill_width_cap`'s KV term divides free
regions by per-row cost, and at zero free regions it returned zero rows. That is
not a narrow wave, it is *no* wave, and it is self-perpetuating: no forward runs,
so nothing completes, so nothing is freed, so the cap stays at zero. The
partition's answer to genuine exhaustion is a refused claim and the relief pass
behind it — and both need a forward to have been attempted. The term is now
floored at one row.

Both faults are invisible to the gate, which admits unconditionally (§13b) and
never runs a KV side to exhaustion.

---

## 13. Open

### The phase-locked forward (§7), now that the plan is trustworthy

Concretely:

1. Gate `claim_region` on the lock — arena *expansion* only; slot allocation
   inside existing arenas stays open to everyone throughout.
2. ~~Make admit exhaustive~~ — **built** (`wave_admit`, §13b). ~~Add the KV-bytes
   term to the width cap~~ — **built** (`prefill_width_cap`).
3. ~~Place the tier at `A`; move `W` to `A + T`~~ — **refuted** (§13b), twice, on
   two independent grounds. What survives of the intent is the per-forward
   reservation, which is built.
4. ~~Delete the feedback loop~~ — **refuted** (§13b): the estimate cannot be
   replaced by an exact figure, because the quantity it estimates is in the
   future. `KV_REGION_SLACK` was tried as a measured quantity and is worse than
   the constant on the quantized path.

### What is actually left

**One mechanism, and everything else follows from it: a region claim that can
block.** Today a claim the partition cannot satisfy is refused, and a refused
claim fails its forward — so the weight side can only ever be given ground it is
provably safe to lose, which is why the opening position has to be generous and
why the decay has to be slow. If a claim could instead *wait* for the next
boundary move (~60 ms, one pass), three things fall out at once:

- `INITIAL_KV_RESERVE` deletes: the weight side may open at
  `MIN_ELASTIC_RESERVE` because a cold-start claim that arrives too early waits
  rather than fails.
- The decay may be aggressive or go entirely, because taking too much costs a
  pass of latency instead of a failed request.
- §7's "instead of getting OOM, it just slows down the decodes" becomes literally
  true, which it is not today.

That is the same deferral §7 specifies for persistence's background seals,
generalised to the wave path — and it is the single highest-value piece of work
left in this document.

**The layer-window question is settled: the lock spans one window.**
`forward_wave_contexts` takes a layer range, so a co-batched forward can be
sliced into windows with the residual handed back between them, and the concern
was that a residual living on a wave span would forbid the boundary moving
between slices. It does not. The function returns `(WavePhase::Residual(x),
None)` — no guard — and the guard is exactly the thing a caller must hold to keep
span-backed values alive (the `Logits` arm returns `Some(head_span)` for that
reason). The residual is the embedding's pool allocation mutated in place by
`add_mut`, and the one op that reallocates it, `to_dtype_mut`, inherits from `x`
itself and so stays on the pool.

So each window prices, places and releases its own tier — which is what the built
code already does, since `plan_wave_transient` is called per call — and the lock
has the same scope. Nothing extra is needed to make the sliced case safe.

**The copy-stream ordering costs 2–3 %.** §8's fix serialises the copy stream
behind compute already issued, which removes the DMA/compute overlap on the
expert path. The cheaper scheme, not built: record an event per layer, keep the
last two, and wait on the one from **two** layers back — combined with excluding
layer `current − 1` from the eviction windows (`allocate_slot`'s behind-layer
scan and `evict_for_prefetch_batch`), that is equally sound and keeps one layer
of overlap. It was not built tonight because it changes the eviction policy, and
a policy change validated on one benchmark at 1am is how hit rates quietly
regress.

**The logits copy is still on the pool.** 2,734,848 B per forward, 97 % of what
the engine allocates outside the arenas. Unchanged by this work.

**`MIN_ELASTIC_RESERVE` and `pool_cushion` are untuned.** Both are still the
derived-then-rounded values of §2 and §4. The cushion in particular is a
measurement from the *previous* load order and is worth re-taking now that the
span is claimed at a different moment — `residual` in the reservation line reads
2,289 MiB, and where that goes is not currently attributed.

**The reclaim is real and unspent.** The tier now gives its ground back on every
forward (§13a), but the five-minute decay means no benchmark short enough to run
in CI moves the boundary, so the gate proves only the *conceding* direction and
nothing takes what was freed. `INITIAL_KV_RESERVE` still opens where the fixed
blocks used to be. **Neither is worth tuning.** Both are artefacts of a boundary
that only moves on a control loop; under §7 it is recomputed exactly on every
forward and the remainder is handed over in the same operation. Tuning the crutch
would only make it harder to remove.

**The gate's wall clock is too noisy to read a few percent from.** Runs of an
unchanged binary spanned 134.9 s to 158.3 s — a 17% spread with no code change
between them — so a single total is not evidence about a change of this size.
Per-config throughput is the steadier signal and it is what §12 should be read
against; a wall-clock claim needs the run repeated. The `−4.5 %` in §12's first
table was a single pair and should be treated the same way.

---

## 14. Built differently, and why

### The one that was a mistake, not a finding

**I pinned the wave transient tier at a fixed address on the left**, and wrote a
justification for it: *the tier is a constant size, so floating it buys nothing
and costs a rebase of every wave domain's address.*

Both halves of that are wrong.

- **The tier is only constant because I kept it constant.** Its size is per-wave
  and `WavePlan` prices it — though not correctly until §13a, which is a separate
  story. Making it variable is the entire source of the reclaim: 912 MiB reserved
  against the ~50 MiB a twenty-session decode measures at, ≈430 expert slots at
  ~2 MiB each — the same order as everything else this change is for.
- **"Bases must be fixed" confused fixed-for-the-process with fixed-for-the-wave**
  (§3). `'w` bounds a range inside its generation; between generations there is
  nothing to invalidate.

And the consequence of pinning it is exactly the failure the position exists to
avoid: a variable-size block at a fixed address either strands a hole
(fragmentation) or moves the arena base (thrashing). I had quoted the vanish
invariant in §3 and then used it to argue for *not* moving the thing it licenses
moving.

Corrected in the build: the tier floats and is priced per wave. The remaining gap
is *where* it lands (below).

### Found reviewing the build against this document

Three things, and the third is why the review was worth doing at all.

**The tier ceiling did not bind recycled regions.** `claim_region` gated the
fresh-region path (`next < ceiling`) and left `free.pop()` unconditional, so a
region freed when the ceiling was higher could be handed out while sitting inside
the placed tier — KV writes into the wave's own intermediates, silent until the
output was wrong. Pinned by `the_tier_ceiling_binds_recycled_regions`; the rule
is stated in §7.

**`live_generations` was incremented before a fallible call.** The count is what
`release_if_last` decrements on guard drop, so an error from `arena.generation()`
would have left the tier placed with no guard in existence to release it —
stranded for the process lifetime. Counted after the guard exists now.

**The per-wave plan was consumed with `.take()`, so it was barely being used.**
Only the first phase of each forward read it; every phase after fell back to the
fixed constants. Removing that made the sizing real for the first time, and it
failed immediately on the model — which is how `WavePlan`'s incompleteness
(§13a) surfaced. Worth noting the shape of this one: the feature was written, was
computing the right thing, passed the gate, and was inert. A bug that makes a
mechanism *quietly not run* survives every test that only asks whether the
program still works.

### The check the design specified could not work

§13a has the detail; the entry belongs here because it is a design claim that the
build overturned rather than a bug. `WavePlan`'s doc named
`candle::forbidden_alloc` as the instrument that would catch a missing variant,
and under operand provenance that instrument reports nothing — an undeclared wave
buffer does not reach the driver, it carves from the arena like a declared one.
The completeness check had to be a new thing (`wave_census`), and the plan was
wrong by 1.8× on attention, by 2× on the accumulate dtype, and declared two
buffers that do not exist, with every test green throughout.

Two of those were only visible *because* a measurement was taken rather than
reasoned about, and one of them — the FFN phase resting 1,152 bytes under its
512 MiB cap while its tail spilled to the pool — was a span sized from a
measurement the span itself had clipped.

### The rest, which were forced by the build

| Designed | Built | Why |
|---|---|---|
| Tier placed at the arena frontier `A`, after all KV is loaded | Placed hard-right, at `W − T`, at the first `begin_wave` | Built at `A` first and the gate killed it: `every region occupied (2 live)`. **Arenas are created throughout the layer loop, not before it**, so anchoring at the live watermark froze KV growth for the whole wave. `W − T` gives the arenas every region below the tier. Same reclaim, but the freed run is stranded mid-span instead of adjacent to the weights — which is what §7 phase 3 fixes. |
| Boundary moves at wave start, KV side retracts on contact | Moves at the expert pipeline's end of pass; KV side records demand | No phase exists yet in which the pipeline thread is known idle, so a synchronous cross-thread eviction is unsafe (§7a). Dissolved by the phase lock. |
| Fill the weight side to `MIN_ELASTIC_RESERVE` at load | Open at `INITIAL_KV_RESERVE` instead | **The design was right and this is a crutch.** It fails the gate only because nothing corrects the boundary afterwards — the give-back costs a pass a failing claim does not have (§7a). Under §7 the constant deletes and the floor is the only number left (§2). |
| — | `order_copies_after_compute` | Not designed at all: fixed-address slots lost the ordering guarantee `cuMemAllocAsync` was silently providing (§8). |
| Per-pass decay of the KV high-water mark | Five-minute wall clock | Per-pass decay let the weight side take 224 of 291 regions during the gate's single-context configs. **Both are placeholders** — the phase lock recomputes the partition exactly, per decode step, and needs no decay at all (§7). |

Two design claims survived contact unchanged and are worth recording as such:
**the dense weights must be subtracted exactly once** (§4 — the load reorder makes
the second subtraction look like prudence, and the test that pins it deliberately
records the tally), and **the reserve must apply on the balloon's fast path**
(§5 — that path is the one that runs, and it was applying no reserve at all).
