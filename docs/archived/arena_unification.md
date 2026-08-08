# Unified Arena Memory — One Reservation, Two Directions, No Defragmentation

> **Status**: **FINAL** — design approved, audited against the code (A1–A15),
> probed on the target hardware, scoped, and with **no open questions
> remaining** (§10). Implementation staged in seven gated steps; step 1 is
> unblocked. What is left is measurement, and every measurement has a named
> hook and a threshold.
> **Gate** (run after every step, all configs must pass, perf recorded in
> `docs/arena_unification_results.md`):
>
> ```
> cargo test --release --features cuda,verbose --lib --package candle-transformers \
>   quantized_qwen3_moe::tests::test_parallel_batched_forwarding -- --ignored --nocapture
> ```
>
> The daemon (`zend`) is only touched in the final step, after the model is
> proven in the gated test.

---

## 0. Abstract

All device memory the inference engine will ever use is claimed **once at
startup** as a single contiguous virtual span and never returned. Inside it,
allocation is two-directional: long-lived KV regions pack **left→right**
through a fixed-stride size-class allocator, and wave-scoped transient buffers
grow **right→left** as a generational bump allocator whose cursor resets once
per wave — with a region-aligned boundary between them that floats under a
slow regulator. Nothing is ever relocated: reclaim is eviction to the warm
tier, rightmost-first. The CUDA allocator, the defragmenter, the compactor,
the VRAM budget gate, and the runtime memory-pressure subsystem are all
deleted — allocation becomes pointer arithmetic, reclaim becomes the tiering
path the cache already needs, and pressure becomes a comparison against exact
counters we own. Memory safety is proven by the borrow checker where the type
system can reach (generation-lifetime tensors and buffers) and enforced by
fail-loud runtime checks where it cannot; corruption is structurally
impossible in both cases.

### 0.1 Principles

1. **Claim once, free never.** Every steady-state byte comes from a
   reservation mapped at startup. No CUDA allocation, no CUDA free, on any
   inference or maintenance path.
2. **Lifetime determines allocator, and there are only three.**
   Wave-scoped → the bump side. Turn-or-longer → the region pool.
   Process-lifetime → the static shelf. Mixing lifetimes in one pool is the
   root fragmentation mechanism (§3.10) and is forbidden by layout.
   **Corollary — anything that outlives a wave is arena-managed, never
   buffer-allocated.** The small reserve is for odd-shaped *transients* we
   chose not to convert; it is not a home for long-lived state. A long-lived
   consumer left on the CUDA pool means recurring `cuMemAlloc`/`cuMemFree`
   in steady state, which is principle 1 violated by another name — and if
   it also *grows*, it drags §1.3's pool-slack mechanism back in with it
   (audit A13).
3. **Fixed strides only.** Every allocator hands out fixed-size units
   (class slots, whole regions, or a bump cursor that resets wholesale).
   Variable-size allocation — the precondition for fragmentation that needs
   relocation — exists nowhere.
4. **Relocation does not exist.** Reclaim is eviction through the existing
   hot→warm tiering path. The hot tier is a cache; demotion is its defining
   operation. No GPU→GPU moves, no gid remaps.
5. **One positional order: pack left, reclaim right.** Lowest-first packing
   keeps the right edge cheap to reclaim; rightmost-first eviction serves
   both cross-class reclaim and boundary movement.
6. **Pressure signals are exact and ours.** Free-region counts and plan byte
   totals — never driver-reported headroom, never feedback against WDDM.
7. **Safety by refusal, not by ceremony.** The generation refuses to reset
   while any lease is live — fail loud, never scribble. One counted check
   beats a type-system campaign (§9, S1).
8. **Formats are chunk metadata, not arena identity.** The attention path
   already reads per-band formats from per-chunk records; the selection path
   is re-indexed to match (§2), and the allocator stops caring what a slot
   holds.
9. **Every step lands behind the gate test.** The daemon converts last, after
   the model is proven.

---

## 1. Motivation

### 1.1 Goal

Replace the dynamic per-format arena system — and every subsystem that exists
to police it — with the reservation of §0, so that:

- fragmentation requiring relocation is **structurally impossible**, not
  merely mitigated,
- no inference or maintenance path ever calls the CUDA allocator (allocation
  is infallible pointer arithmetic until the reservation itself is exhausted),
- KV chunks of every quant format, the gallery arena, the meta pool, and the
  inference loop's transient/intermediate buffers all draw from the one
  reservation — long-lived allocations from its left region pool,
  wave-scoped ones from its right bump side,
- the memory-pressure problem reduces to exact counters (free regions, plan
  bytes) instead of a feedback loop against driver-reported headroom.

### 1.2 Why this is the right altitude

Every reclaim mechanism in the current system serves exactly two consumers:

1. returning 16 MiB slabs to the CUDA pool so a **different format's** pool
   can allocate them, and
2. leaving room for the forward pass's **transient activations** to grow.

Shared size classes eliminate (1) — cross-format reclaim becomes "pop an
empty region, stamp it with a new class" — and the bump side behind the
floating boundary eliminates (2). With no one to hand bytes *back* to, a hole
in a region is simply a free slot the O(1) recycle stack reuses. The
defragmenter, the compactor, the empty-arena sweeps, the VRAM budget gate,
and the runtime pressure ladder all lose their reason to exist.

### 1.3 Measured motivation (2026-08-06 overnight run, 16 GiB RTX 4090 Mobile)

Peak arena slack during workspace ingest — **2142 MiB, 16 % of the card**:

```
float:  79 arenas / 1264 MiB reserved,  435 MiB live  →  829 MiB waste (66 %)
quant: 257 arenas / 4112 MiB reserved, 2799 MiB live  → 1313 MiB waste (32 %)
```

Three mechanisms produce it:

1. **Per-format last-arena tails** — adaptive selection scatters chunks across
   ~10–15 live formats; each format's newest 16 MiB slab is partially full.
2. **Within-format hole fragmentation** (dominant) — turn drops free scattered
   slots; a slab is only reclaimable when *wholly* empty, and free slots of
   format A are invisible to format B.
3. **CUDA-pool slack on top** — freed 16 MiB slabs leave pool holes reusable
   only by another contiguous 16 MiB request, while contiguous activation
   allocations force pool growth (`pool_reserved − pool_used` gap).

The same fragility produced the wedge class of failures
(`deepest_rung=Critical freed_mib=0`, 20 s `compact_forced` stalls, the
compaction-livelock and drain-plan pathologies now papered over with bounded
plans and forced sweeps). This design removes the failure class, not the
symptom.

---

## 2. Current state (survey summary)

The allocators that exist today, all independently re-implementing the same
slab-pool pattern:

| System | Granularity | Location |
|---|---|---|
| KV chunk arenas | 16 MiB slabs ÷ per-**format** chunk slots (32 B–4096 B) | `candle-nn/src/kv_cache/chunked/gid_pool.rs`, `arena.rs`, `alloc.rs` |
| Gallery arena (belief scan) | 16 MiB slabs ÷ 6144 B pages | `candle-conversation/src/provenance/gallery_arena/` |
| Meta pool (`KvHead` records) | device slabs ÷ fixed records | `candle-nn/src/kv_cache/chunked/meta_pool.rs` |
| Forward activations / kernel metadata | ad-hoc tensors per forward | cudarc stream-ordered pool |
| Grow-only device scratch | `ProvSignScratch`, sampler scratch, … | various |

Load-bearing facts the design builds on:

- **Chunk geometry.** `CHUNK_SIZE = 32` tokens; one arena chunk slot is one
  *(head, palette-band, side)* of `32 × sub_head_dim(32)` = 1024 elements. A
  logical block holds `GIDS_PER_HEAD(8) × n_kv_head` slots.
- **Gid arithmetic.** `raw = arena_idx × stride + chunk_idx`. The *form*
  survives; the stride constant changes to a fixed power of two (§3.5).
- **Two read paths, and they differ on formats — the single most important
  fact for this design.**
  - **Attention (decode / prefill / glue): per-chunk, per-band.** Each
    `TokenSlice` carries a `kvheads_ptr` to a `KvHead[n_kv_head]` record whose
    fields are `k_ptr[4] / v_ptr[4] / k_fmt[4] / v_fmt[4] / k_scale[4] /
    v_scale[4] / pal` — **one format tag per palette band**
    (`slot_state.rs:84-106`). Formats already travel with the *chunk*, so
    mixed-format regions need **no attention-kernel change**. Only the host
    builder (`KvHeadHost::from_gids`, `slot_state.rs:147-153`) derives those
    tags from `arena_info[arena].k_format_tag` and must be inverted.
  - **Selection / probe (`select_kv_format_palette4_paged*`,
    `sample_quant_errors_*`, `reduce_head_format_stats`): per-ARENA, and it
    discards the palette dimension.** `resolve_band_source`
    (`select_kv_format.cuh:1461-1463`) computes `arena_idx = gid /
    arena_chunks` and looks up `load_per_head_entry(table, arena_idx, head,
    n_kv_head)` — which bottoms out in `per_head_lookup`, returning
    **`.palette[0]`** (`arena_table.cuh:114`), *ignoring* the `palette`
    argument `resolve_band_source` was given. The row is built per-arena from
    `arena.format()` in `per_head_table_host` (`backing.rs:1549-1655`), which
    fills all four sub-entries **identically**.

    So today's per-band format variety is produced entirely by
    **`arena_idx` varying per band** — format ⇒ arena is load-bearing, and
    the `Palette4` sub-entry structure is a present-but-unused capability
    carrying four identical copies. Size classes put bands of different
    formats in one region, collapsing `arena_idx`, so this path needs the
    sub-entry dimension *activated*: index rows by `(chunk, head)` and read
    `.palette[p]`. The row layout already provides it (see
    `tests/kv_stats_tests.rs:930`, which builds per-chunk rows), so no struct
    or launch change — see §5 step 1.
- **Per-band format metadata already exists on two of three paths.** The
  cold-load path carries per-band `k_formats`/`v_formats`
  (`BlockAllocSpec`), and the substrate log persists per-chunk format tags
  (`KvFormat::to_tag`). Only the "derive format from the arena key" sites
  need inverting (§5, step 1).
- **The lock-free fast path is already right.** `ArenaRefcounts` (overlapped
  u16 refcount/recycle-link words, Treiber stack + high-water mark),
  `CapacityBitmap` find-first-set, single `alloc_gate` per pool. All retained.

### 2.1 Latent defect fixed by this design

`ArenaRefcounts.counts` is `Vec<AtomicU16>`; a **free** slot's word holds the
recycle-stack *link* (next free chunk index), documented as safe because
"arena chunk counts and indices are both far below 65536". That is false for
**seven** formats at 16 MiB slabs — enumerated by running the real
`arena_chunks_for_format` table (experiment E1, §8):

```
Q0     524,288   Q0_V   262,144   Q0_X  262,144   Q0_M2 174,762
Q1_S   104,857   Q1_A    87,381   Q0_M4  65,536   ← exactly at the boundary
```

Any recycled slot above 65,535 in those pools silently truncates its link —
free-list corruption. `Q0_M4` is the subtle one: 65,536 chunks means the
empty-stack sentinel (`arena_chunks`) is itself unrepresentable in `u16` and
wraps to 0, aliasing slot 0. The 320 B class minimum (§3.4, invariant 6 of
§4) caps chunks-per-region at **52,428** and removes the hazard structurally.

---

## 3. Target architecture

### 3.1 Layout

```
┌──────────────────────────────── VRAM (capacity C, balloon-measured) ───────────────────────────────┐
│ dense weights │ expert cache (slots) │ ██ RESERVATION ██████████████████████████████ │ small       │
│  (static)     │  (static, governor-  │  KV regions ──────▶   boundary k  ◀── transients │ reserve  │
│               │   sized at startup)  │  ┌─────────┬─────────┬────────┐╎┌───────────┬──┐│ (CUDA pool│
│               │                      │  │ class   │ class   │ free   │╎│ per-wave  │SS││  remnant) │
│               │                      │  │ 1152 B  │ 4096 B  │ region │╎│ bump ◀──  │  ││           │
│               │                      │  └─────────┴─────────┴────────┘╎└───────────┴──┘│           │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
   regions[0..k) allocate lowest-first (left-packed)  ╎ = floating, region-aligned   SS = static shelf
```

Allocation is **two-directional** (the process stack/heap model): KV regions
grow left→right and pack lowest-first; the **transient bump side** grows
right→left from the reservation's fixed end. The boundary `k` is
region-aligned and **floats** under a slow regulator (§3.6). Long-lived KV
allocations and ~50 ms transient allocations never share a region — mixing
the two lifetimes in one pool is the dominant fragmentation mechanism this
layout exists to prevent (§3.10, problem 1) — yet no static split has to be
sized at startup: the boundary adapts to the workload.

### 3.2 Reservation mechanics (probed on the target machine)

- **One contiguous virtual span, mapped via the CUDA VMM API**
  (`cuMemAddressReserve` + `cuMemCreate`/`cuMemMap` in 256 MiB physical
  granules): a single base pointer for all region arithmetic and the floating
  boundary, with **granule-level residency** so WDDM eviction pressure can
  never hit the whole reservation at once. Probed on the target machine
  (RTX 4090 Mobile, driver 596.08, WDDM): VMM supported, min granularity
  2 MiB, and a boundary-straddling write across independently-mapped granules
  verified contiguous.
- **No fallback. VMM is a hard requirement, and its absence is a named
  failure.** This section originally specified a runtime capability probe
  choosing between VMM and a single giant `cuMemAlloc` (probed and working:
  14 GiB in 15.6 ms with the desktop resident, full touch at 38.5 GiB/s). That
  is not built, and should not be — the second path is not a smaller version of
  the first, it is a different one on three axes:

  1. **Eviction granularity.** VMM's unit is the 2 MiB granule, so WDDM pressure
     sheds cold granules. One giant allocation's unit is the whole buffer: the
     entire reservation goes to host and faults back over PCIe on the next
     touch. On a 16 GiB card driving a display that is the difference between a
     hiccup and a stall.
  2. **No partial release.** The probed release-then-reuse trick
     (`vmm_release_probe.py`) has no `cuMemAlloc` equivalent — a giant
     allocation must be sized right once and can never give a byte back.
  3. **Capacity measurement changes shape.** Map-and-touch per granule is
     self-terminating and doubles as the region tier's zero-fill. `cuMemAlloc`
     is all-or-nothing, so extent would have to be binary-searched over whole
     allocations, each failed probe costing an allocate/free cycle.

  So `Reservation::reserve` queries
  `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` first and fails with
  a message naming the capability, the reason there is no fallback, and where
  VMM is commonly absent (WSL2, vGPU/MIG, older drivers). A device test asserts
  the attribute, so an unsupported target reports it in the suite rather than at
  the first KV cache of a benchmark run. Superslabs are never needed either way.
- **Physical VRAM fragmentation from other processes is a non-issue by
  construction**: WDDM2 GPU memory is page-table virtualized per process, so
  allocation success depends on total commit only, and VidMm evicts other
  tenants to make touched memory resident (the balloon's proven mechanism).
- **The balloon becomes the reservation — but the stop signal is the TOUCH,
  not the create.** Probed (`vmm_overcommit_probe.py`, 32 GiB VA span on the
  16 GiB card): **`cuMemCreate` never refuses** — it succeeded for every
  granule out to the full 32 GiB span. The real limit surfaces at the first
  *write* to a granule, as an unhelpful `invalid argument` at 15,360 MiB
  (driver-free having fallen 15,074 → 738 MiB on the way). Mapped-and-touched
  granules are genuinely resident, not host-backed: re-touching granule 0
  after the span was full ran at **528 GiB/s**.

  So "map until the driver refuses" is only valid if *refusal* means a failed
  **touch**, checked per granule, with the failing granule unmapped and
  released. The reservation extent is therefore
  `min(policy_cap, first_touch_failure)` — keeping the existing balloon's
  back-off policy (`C = min(frac × total, total − headroom_abs)`), because
  mapping to the touch-failure point leaves the desktop with zero headroom.
  Do **not** treat `cuMemCreate` success as evidence of capacity.
- **Startup sequence — as built (2026-08-08), and the order is inverted from
  what this section originally specified.** The plan was: reserve the VA span →
  map + **touch** granules under the policy cap → **release** the granules the
  partition assigns to dense weights, the expert cache and the small reserve →
  load those through the CUDA pool into the freed memory → the granules still
  mapped are the reservation. That sequence was probed end-to-end and works
  (`vmm_release_probe.py`: after releasing 8 of 64 granules an ordinary
  `cuMemAlloc(1920 MiB)` succeeded into the freed space **and** the surviving
  14,336 MiB read back intact).

  It is not buildable, because **the partition is not knowable before the
  load**. `expert_budget()` is by construction a live measurement taken *during*
  the load, and the dense-weight total is only known once the loader has walked
  the GGUF — predicting either means duplicating the loader's tensor walk and
  keeping the duplicate in sync forever.

  So: the governor's balloon measures `C` and frees (unchanged) → the model
  loads (unchanged) → the reservation is claimed at the first KV cache, sized
  `usable − scratch_margin` and filled granule by granule, each **written**
  before it counts. What the original order was protecting — that measuring `C`
  and claiming it are one act, with no window to lose it in — is preserved
  exactly, because the refusal point of that fill *is* the reservation's extent.
  Only the position relative to the model load moved, and the thing that moves
  into that window is our own weight loading, which is what is supposed to
  happen there.

### 3.3 Region tier

- `region[i]` base = `reservation_base + i × 16 MiB` (a clean multiple of the
  2 MiB granularity). `TARGET_ARENA_BYTES` stays 16 MiB.
- **The free-region list is its own — this section originally claimed
  otherwise, and that claim is wrong.** It said `GidPoolState.free_arenas`
  under the `metadata` mutex *is* the free-region list, so the region tier
  reuses it verbatim and adds no lock. It cannot: `register_arena` pops that
  queue **regardless of `ArenaLocation`**, so a tombstoned CPU arena's index is
  handed to the next GPU arena and vice versa. An arena index therefore says
  nothing about position in the span, and using it as a region index would
  place GPU regions at addresses that were never mapped.

  As built, `region_pool` keeps its own free list — a min-heap, lowest-first
  per principle 5 — behind its own mutex. Audit A5's conclusion survives: that
  mutex is a **leaf**, taken by nothing else and taking nothing, so the
  documented lock order (`alloc_gate → tables`, `metadata` outside both) still
  gains no edge. A region is claimed by `create_arena`, held by the `Arena` for
  its lifetime, and returned when the arena drops — release, truncate, or the
  backing going away all work, because ownership does the bookkeeping.
- **Recycling a region needs an explicit quiesce.** A region returns to the
  free list when no host-side gid names it, which does not mean no kernel is
  still reading it. The allocator used to supply that ordering for free —
  `cuMemFreeAsync`/`cuMemAllocAsync` on one stream — and a free-list push does
  not, so a device-wide synchronise precedes the zero-fill of a *recycled*
  region. Device-wide because the persistence thread's copy stream is one of
  the readers. Found by the candle-nn suite as `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Regions allocate lowest-first (the capacity bitmap's find-first-set),
  keeping live data left-packed per principle 5.
- `ResolvedArenaInfo.base_ptr` becomes pure arithmetic; the
  "arena not found" / dangling-base-pointer error class disappears.

### 3.4 Class tier

- `ArenaKey` becomes `(SizeClass, ArenaLocation)`. Every chunk slot in a
  region has stride = class bytes; a chunk of format F occupies one slot of
  the smallest class ≥ `chunk_bytes(F)`, trailing pad unread.
- The per-class pool machinery is today's `ArenaPool` verbatim: refcount
  tables, capacity bitmap, gated claim walk, HWM runs for palette locality.
- Formats become **chunk metadata**, not arena identity: `ChunkWindow` /
  `SealedChunk` carry per-band format tags parallel to `gids` (§5, step 1).

Chunk bytes per 1024-element slot, by format, and the chosen ladder:

| Class (B) | Formats (chunk bytes) | Rounding waste |
|---|---|---|
| **320** | Q0 (32), Q0_V/Q0_X (64), Q0_M2 (96), Q1_S (160), Q1_A (192), Q0_M4 (256), Q2_S (288), Q2_0/Q2_A (320) | high in % for the tiny formats, negligible in absolute bytes |
| **448** | Q2_1 (384), Q3_0 (448) | 14 % / 0 % |
| **640** | Q3_1 (512), Q4_0 (576), Q4_1/Q4_KS (640) | 20 % / 10 % / 0 % |
| **768** | Q5_0 (704), Q5_1 (768) | 8.3 % / 0 % |
| **1152** | Q8_0 (1088), Q8_1/Q8_KS (1152) | 5.6 % / 0 % |
| **2048** | F16, BF16 (active V) | 0 % |
| **4096** | R16 (active K), F32 | 0 % |

**Coverage is complete and deliberate.** Seven classes cover **all 22
`QuantFormat` variants and all four float dtypes** — the ladder is
infrastructure over the *format space*, not over whatever the current policy
happens to select. Sizing it to today's candidate lists would be a mistake in
three ways: the `PRODUCTION_*_QREL_*` threshold tables are explicitly
provisional and re-derived per model (a Qwen3 re-tune or a new model can pull
any format in); `override_k_quant`/`override_v_quant` can force any format at
any time; and **cold-loaded chunks carry whatever format tag was persisted**,
so a substrate written under an earlier policy can hand back formats the
current level would never pick. A class costs one row in a `const` table; a
missing class costs 30–40 % pad on every chunk of that format, forever.

Measured against the real format table (experiment E1, §8): every one of the
22 `QuantFormat` variants and all four float dtypes maps to a class, none
uncovered. `F8E4M3` KV (1024 B/chunk) rounds into 1152 at **11.1 %**; if a
model ever configures F8 KV as its working format, give it a 1024 class
rather than eat that. The 6144 gallery-page class is absent only because the
gallery arena stays out of scope (§7, S5) — it joins the ladder if that
changes.

**The small end is a deliberate trade, and it is not free.** Rounding waste
at the bottom of the ladder is large in percentage terms — Q0 32 → 320 is
**90 %**, Q0_M2 96 → 320 is 70 %, Q1_S 160 → 320 is 50 % — and those formats
are exactly the C9/C10 candidates, i.e. they appear when bytes matter most. A
finer low end ({64, 160, 320, …}) would cut that, at the cost of splitting one
maximally-fungible pool into three with their own partial tails. The bet is
that fungibility wins because the *absolute* bytes are small and a chunk's 32
bands rarely all pick the smallest format — but it is a bet, not a fact.
**Step 6 must measure the realised per-class occupancy at C9/C10 and revisit**;
the E1 harness (re-created as a permanent test in step 1) recomputes the whole
table from one constant.

Weighted by the production mix (R16/F16 active + Q8/Q4 families sealed
dominate), rounding waste is **~2–4 % of live bytes**, replacing the measured
32–66 % slab waste. The ladder is one `const` table; re-derive it if the
format histogram shifts (per-class occupancy logging keeps the histogram
observable). Meta-pool records get their own class row when they fold in
(record size is fixed per model geometry; timing is open question 3).
Transients are **not** a size class — they live on the bump side (§3.6),
never in class regions.

**Scarcity-only class promotion.** When a class has no free slot AND the
free-region list is empty, a chunk may be placed in the next class up (more
pad; per-band reads use format bytes, so correctness is untouched). This
stops a trickle of rare formats (a handful of Q1_S chunks) from stamping a
whole 16 MiB region for a class that will never fill it. Strictly
scarcity-gated — under any free-region availability a class gets its own
region — so promotion cannot become a background mixing vector.

**Rejected: variable quantum runs.** A quantum-unit design (chunks occupy
`ceil(bytes/U)` contiguous units) reintroduces external fragmentation
*within* regions (a freed 2-unit run cannot serve a 3-unit request) — the
disease this design exists to eliminate, and a violation of principle 3.
Fixed-stride classes have zero external fragmentation, O(1) alloc/free, and
leave the kernel addressing model untouched.

**Rejected: a single stride for every class.** At stride 4096 the dominant
sealed formats waste 73 % (Q8_0) to 84 % (Q4_KS) — worse than the slab waste
being eliminated. The drift-safety a compiled single constant would buy does
not apply to KV: class strides are consumed **host-side only** (allocation
and sub-entry serialization) — the kernels read band pointers and strides as
*data* from the pre-resolved slot-state buffers and `KvHead` records, never
as compiled arithmetic. The class ladder has no CUDA twin to drift from.

### 3.5 Gid encoding and shared compile-time constants

The gid stride becomes a fixed power of two, **65,536**
(`chunk_idx = raw & 0xFFFF`, `region_idx = raw >> 16`) — headroom above the
52,428 max chunks/region of the 320 B class, and gid decode becomes
shift/mask instead of div/mod. (Today's stride is 524,288, diluted by the Q0
layout that the class minimum retires.)

Constants compiled into both Rust and CUDA (one header each side, locked by
`static_assert`, exactly like `CHUNK_SIZE` and the block structs in
`blocks.cuh` today):

| Constant | Value | Used for |
|---|---|---|
| `CHUNK_SIZE` | 32 | tokens per chunk (already shared) |
| `REGION_BYTES` | 16 MiB | region carve, base-pointer arithmetic |
| `GID_STRIDE` | 2^16 | gid decode as shift/mask on both sides |
| `PAGE_BYTES` | 64 KiB | the paged-output contingency only (§5 step 3): if a converted buffer must outlive its wave, kernel page math is shift/mask with immediate operands |

### 3.6 The transient tier: a contiguous bump side with a floating boundary

Inference-loop transients (kernel argument blobs, staging blobs, intermediate
activations in the bounded set of §5 steps 2–3) do **not** allocate from the
region pool. They occupy the reservation's **right side** — regions `[k..N)`,
adjacent in VRAM and therefore one contiguous span — run as per-domain
**bump allocators** growing right→left from the fixed end:

- **Contiguous by construction, through every boundary move.** The span is
  `[k..N)`; growth decrements `k` (space joins adjacent-left), shrinkage
  returns the span's left edge to the region pool. The bump base (the
  reservation end) never moves — only the limit. Every transient buffer is
  an ordinary contiguous buffer: no paging, no page tables, no kernel
  addressing changes. Small blobs pack densely by bump instead of costing a
  page each.
- **Generational — nothing is ever deallocated.** A generation bump-allocates
  and later resets its cursor to zero — one store, no per-buffer RAII, no
  frees, **zero allocator traffic on the wave path in steady state**. The
  wave domain is double-buffered (A/B halves) so wave `N+1`'s assembly can
  allocate while wave `N`'s kernels drain. That double-buffering is also what
  makes the reset free: the two halves share one stream, so a whole wave's
  work separates a half's last read from its next write and no fence is
  needed. Cross-stream domains (persistence staging) do fence — the
  `PinnedStager` sync-then-reset discipline — but the wave path does not.
- **The wave buffer set — no mid-wave allocation, ever.** Pure bump-no-free
  would turn peak usage from O(max concurrent) into O(total allocated during
  the wave) — a ~`n_layers`× blowup for per-layer intermediates. Instead,
  wave assembly allocates its **whole buffer set once, at wave start**: every
  shape is known when the wave is admitted (batch, token counts, layer count
  are fixed; layers are shape-identical), so it is a ping-pong pair for the
  inter-layer hidden state (layer `i` reads A, writes B; `i+1` swaps) plus one
  reused workspace buffer per within-layer temporary (attention scores, MoE
  dispatch — liveness never crosses a layer boundary). These are ordinary
  locals held across the layer loop, not a data structure: **a bump allocator
  returns disjoint ranges by construction**, so no slot table, no
  disjointness bookkeeping. Peak = the true concurrent working set; the set's
  byte total is the admission input. After wave start, allocating from the
  wave generation is a debug assert.
- **Domains and span sizing.** The scheduler's wave loop (A/B halves) and the
  persistence thread (migration/quantize staging) own disjoint sub-ranges
  with independent generations, so neither resets the other's live buffers;
  process-lifetime grow-only scratches (`ProvSignScratch`, sampler scratch,
  MoE routing buffers) keep growing from the CUDA pool, which `scratch_margin`
  already covers. The span requirement is the sum of the domain budgets:
  `S = 2·W_wave + W_persist` = **192 MiB**.

  **`W` is not always a watermark (2026-08-08).** For the wave it is: 30.75 MiB
  measured, 64 MiB taken. For the persistence domain it is a **declared
  budget** — `MIGRATION_STAGING_CAP_BYTES` — because a migration batch *bisects
  itself* to fit whatever the span is. Size a domain from its watermark only
  when the domain cannot shrink to fit.

  **The span is priced in KV regions, so a declared budget still has to be
  argued (2026-08-08).** `S` was 704 MiB — 44 regions taken off the KV side
  before one is carved, on a card where those regions decide whether the expert
  cache is fed. Two of the three terms were unearned:

  - The **shelf** held 64 MiB for an allocator that was never built. Removed;
    reinstate it when something allocates from it, priced in regions like
    everything else.
  - **`W_persist` was 512 MiB.** "It bisects to fit" argues the span can be
    *small*, not that it should be large — the cost of a big one is paid in KV
    regions on every boot, while the cost of a small one is paid in DtoH syncs
    on the hot→warm path only. Sized instead from the floor the batch-halving
    retry already handles (a single ~30 MB layer): **64 MiB**, two layers plus
    headroom, ~22 syncs for a ~1.4 GiB pass against the per-layer 48 that made
    `copy_ms` the bottleneck.

  **The `W_persist` cut was reverted on review; only the shelf removal stands,
  so `S = 128 + 512 = 640 MiB`.** "It bisects to fit" is true of exactly one of
  the three staging sites. `migrate_sealed_to_gpu_batch_async` (warm→hot
  elevate, issued **once per layer across every warm item**) and the hot→warm
  per-layer gather each do a single `bump.alloc(total_bytes)` with no batching,
  so for them the span is a hard ceiling and 64 MiB turns a deep elevate into a
  failed forward. The 29,696 B peak that argued for the cut was measured on runs
  where the elevate path never executed. **Make those two sites batch like the
  third and this term becomes tunable**; until then it stays at the size the
  elevate was written against.
- **Two time scales.** A **fast per-domain gate**: wave assembly gates width
  against its half's *current* capacity using the plan's byte total — exact
  by construction, not a forecast — hard, never blocks on eviction; a wave
  that does not fit waits (the persistence domain likewise caps its staging
  batch). A **slow envelope regulator**: `k_target = N − ceil(S /
  REGION_BYTES)`, moved with hysteresis between waves. Overflow of transients
  into KV regions is **forbidden** in both directions and at all times
  (§3.10, problem 1).
- **Boundary movement is evict-only — relocation never returns.** Pressing
  `k` left claims free regions first (the free-region setpoint keeps a
  hysteresis gap at the frontier, so this is the common case), then
  **evacuates occupied regions rightmost-first** by demoting their sealed
  chunks to warm — the existing tiering path, never a GPU→GPU move, never a
  gid remap. Un-evictable active writer chunks (mid-turn R16/F16) delay a
  press at turn granularity; admission throttling drains them toward seal,
  and the regulator's time constant absorbs the wait. Keeping the free list
  below `k` is the existing `drain_free_arenas_above` mechanism.
- **Fixed addresses enable CUDA graph capture.** Because a wave's kernels all
  run against plan-fixed addresses, the steady decode wave becomes capturable
  as a CUDA graph — the standing answer to WDDM per-launch overhead. Not in
  scope for this initiative, but the design deliberately preserves the
  precondition; a per-wave allocator forecloses it.

Consumers: chunk-meta rows, per-head tables, head-gid uploads, selection
tables, migration descriptors, migration staging (`copy_stream.alloc`
today), inter-layer hidden states, attention and MoE combine outputs.

**Paged buffers are a reserved option, not a deliverable** (audits A10/A11 found no kernel output that outlives its wave, so none is built). Were one ever to appear: a buffer that
must *outlive its wave* cannot live in the bump side and, if it also should
not occupy the small reserve, allocates `PAGE_BYTES = 64 KiB` pages from a
region plus a device page table. With the compiled page constant the
kernel-side page math is shift/mask (`page = row >> LOG2_ROWS_PER_PAGE`); a
`PagedTensor` (or a paged `QTensor` storage variant) wraps
`(page_table, logical shape)` for the Rust side.

### 3.7 Memory ownership in the type system

Cross-cutting machinery used by both sides of the reservation.

- **One bump abstraction, two backings.** The device bump side and the host
  `PinnedStager` (pinned bump arena + `Generation` guards, already used for
  zero-copy kernel-argument staging) become the same `BumpArena` construct
  with host-pinned and device instances sharing the generation lifecycle and
  its invariants. The pinned instance keeps its role (zero-copy PCIe reads
  of small descriptors); the unification is the allocator and its safety
  rules, not the memories.
- **Regions are byte slabs, not tensors.** `Arena::Float{Tensor}` /
  `Arena::Quantized{QTensor}` collapses to a single byte-slab variant with a
  class stride: under size classes the storage no longer knows (or needs to
  know) what format a slot holds — that lives in chunk metadata. This removes
  62 match sites across 6 files and makes the primitives *simpler than
  today's*: `zero_chunk_at` becomes a memset (was `Tensor::zeros` +
  `slice_set` / `write_bytes_at`), and the pinned read/write paths become
  memcpy (was `narrow`/`flatten_all`/`to_vec1`). The `PagedKvArenas` trait and
  `float_arenas()` / `quantized_arenas()` / `k_arenas()` / `v_arenas()` are
  consumed **only by tests** (verified) and are deleted rather than ported.
- **`Tensor`/`QTensor` backing, for the paths that genuinely need tensors.**
  Storage gains a second backing (the `from_blob` pattern):

  ```rust
  enum Backing {
      Owned,              // pool allocation, freed on drop (today's behavior)
      Lease(ArenaLease),  // memory owned by the reservation; drop releases
  }                       // the lease, never frees
  ```

  `ArenaLease` = `{generation_id, Arc<AtomicUsize> live_count}` —
  incremented at construction, decremented on storage drop.

  **As built (2026-08-07): `Backing::Lease` is a bare marker, and the count
  lives on the allocator instead.** Region-tier leases need no count — a
  region is never reset or reclaimed under a live chunk, so there is nothing
  for a count to refuse. The bump side does need one, and `BumpArena` carries
  it directly: its `Generation` guard is counted, and the cursor refuses to
  rewind while any guard is live. Same refusal, one owner instead of two.

  Wave intermediates (step 3) do not change this. They are handed to candle ops
  as `Tensor`s and outlive the scope that allocated them, so they cannot pin the
  cursor themselves — **the wave holds one generation guard for its whole
  duration** instead. That is not a weaker guarantee but a stronger one: a
  per-lease count would let the cursor rewind mid-wave the moment the last
  intermediate happened to drop, while a kernel enqueued earlier was still
  draining. The wave-scoped guard is exactly the lifetime every intermediate
  needs, and it is what the A/B-half buffer set already implies.
  **Step 3 must re-check this**: wave intermediates handed to candle ops as
  `Tensor`s will outlive the scope that opened the generation, so either the
  lease starts carrying the count as originally drafted, or the wave's
  buffer-set guard is held for the whole wave (which the wave-domain design
  already implies). Do not assume the current shape suffices.

  Two consumers only: the legacy contiguous `KvCache` façade
  (`read_contiguous`/`write_contiguous`, which need on-demand tensor views
  over region bytes) and, from step 3, wave intermediates that candle ops
  must consume. Views/reshapes share the `Arc<Storage>` and the lease travels
  with them, so the inference loop keeps its Tensor-based code.

  **Interception is mandatory, not an optimization.** A region pointer is an
  offset into the VMM reservation, never a pool allocation, so letting
  `CudaSlice::drop` reach `cuMemFreeAsync` on it is an *error*, not a leak.
  The mechanism is measured, built, and reverted in experiment E4 (§8.2): a
  tombstone variant on `CudaStorageSlice` plus

  ```rust
  impl Drop for CudaStorage {
      fn drop(&mut self) {
          if self.backing == Backing::Lease {
              let slice = std::mem::replace(&mut self.slice, CudaStorageSlice::Empty);
              slice.leak();   // per-variant CudaSlice::leak()
          }
      }
  }
  ```

  `CudaSlice::leak` takes `self` **by value** and `mem::forget`s it, so it is
  unreachable from `&mut self` without the `mem::replace`. Calling it — rather
  than merely suppressing the drop — is load-bearing: `leak` waits on the
  slice's read/write events, destroys them, and decrements the
  `Arc<CudaStream>`. Bare suppression strands two `CudaEvent`s and a stream
  refcount **per lease**, thousands per second.

  **No lifetime parameterization.** An earlier draft added
  `TensorG<'a>(Arc<Tensor_>, PhantomData<&'a ()>)` with
  `type Tensor = TensorG<'static>` to make use-after-reset uncompilable. It
  is cut: the measured surface is **1311 `-> Result<Tensor>` signatures
  workspace-wide and 33 `impl` blocks on `Tensor` in candle-core alone**, and
  the counted reset below already catches strictly more (it sees erasure
  seams a lifetime cannot type), fails without corrupting, and runs every
  wave — so a leak surfaces on the first gated run either way (§9, S1).
- **The safety stack** (each layer catches what the layer above is blind to):
  1. **Counted reset, fail-loud-never-scribble.** The generation refuses to
     reset while `live_count > 0`: log loudly, quarantine that half for the
     wave, keep running. Corruption is impossible; the failure mode is a
     *detected* leak. Wave buffers drop at end-of-scope, so zero is the
     steady state.
  2. **Event fence.** Host bookkeeping cannot observe stream asynchrony: the
     reset fences on the completing wave's stream event before the cursor
     moves.

     **As built (2026-08-07): the fence is a per-domain policy, and the wave
     domain does not need it.** A domain declares how its reset is made safe.
     The persistence domain fences, because it stages on the copy stream while
     the compute stream runs and nothing else orders the two. The wave halves
     do not, because they are double-buffered on a single stream: by the time a
     half is handed out again, an entire other wave's work sits between the
     reads and the writes *on that stream*, and same-stream launches complete
     in issue order. Fencing there would add a full device sync to every
     forward — a stall on the wave path, which is the cost §3.6 exists to
     remove. The A/B structure is not merely an overlap optimization; it is
     what makes the wave reset free.
  3. **Debug canaries** (optional). Rust cannot observe device writes:
     canary words between buffers catch a kernel overrunning its output.
     This guards a hazard that exists identically with today's pool
     allocations — a new net, not a mitigation for anything this design
     introduces — so it is a debug feature, not a required deliverable.

  Bump arithmetic itself is one audited function; disjointness needs no
  bookkeeping because a bump allocator returns disjoint ranges by
  construction.

### 3.8 Budget model after the switch

- The VRAM governor's *runtime* regulator role ends. It keeps its **startup**
  role: balloon-measure `C` (mapping the reservation as it goes, §3.2), size
  the expert cache, then freeze the partition. Built 2026-08-08: `relief.rs` is
  deleted entire, along with `Criticality`, the five `LadderTier` trip points,
  the sync and reuse hooks, `available()`, `forecast_units`, the OOM-retry
  `allocate`, and the Windows budget-change watcher thread. What remains is the
  balloon, `usable()`/`expert_budget()`/`kv_floor()`, the per-class tallies
  (reporting only), and the starvation signal.
- Pressure = `free_regions < setpoint`.

  **As built (2026-08-08), with two corrections.** The setpoint is
  `max(span/8, 24 regions)` under load and `max(span/16, 8)` in decode, clamped
  to half the span; it scales to the *reservation*, not the card, so one set of
  constants holds across machines. Step 6 tunes both terms.

  First correction: this section's list omits **compress-to-free**, which is a
  real response and survives as the step between the gallery and evacuation. It
  is a *shrink in place* rather than a move — the turn stays resident and
  attended-over, only its float working set goes — so it is strictly cheaper
  than eviction, which has to be reloaded if the turn is re-attended. It sits
  where the governor's ladder had it, one rung above eviction.

  Second: the gallery arena sheds **before** all of this. Its pages rebuild from
  the substrate blob on demand, so dropping one costs only the rebuild, which
  makes it cheaper than anything involving model KV. §5 already specifies the
  mechanism (call `evict_lru` directly rather than registering a relief rung);
  this is where it lands in the order.

  So, in full, each step run only if the one before it left pressure standing:
  1. steal an empty region from any class (O(1)),
  1b. evict resident gallery pages,
  1c. compress-to-free: bring the float→quant conversion forward,
  2. **evict-as-evacuation**: demote sealed chunks to warm via the existing
     `migrate_sealed_layers_to_cpu_batch` + install path (the hot tier is a
     cache; demotion is its defining operation — this replaces GPU→GPU
     defragmentation entirely).

     **As built (2026-08-08): no rightmost scan, and none is wanted.** This
     originally specified demoting the *rightmost* occupied regions. But
     lowest-first packing already makes the high end the least-populated, so
     the scheduler's existing budget-aware eviction — now driven by the exact
     free-region count instead of three disagreeing driver estimates — empties
     the right edge without needing to know that is what it is doing. §3.10
     states the same identity ("rightmost ≈ emptiest"); a separate positional
     scan would be a second mechanism for what the packing policy already
     does. If boundary motion is ever built (step 6), *that* needs an explicit
     rightmost order, because it must empty a specific region rather than
     merely enough of them,
  3. throttle admission (the existing regulated setpoint, now driven by an
     exact, latency-free counter instead of driver headroom).

     As built, the admission *ceiling* is `(free_regions − setpoint) ×
     REGION_BYTES` — the setpoint is subtracted because those regions are the
     relief pass's working room, not admission's to spend. It carries neither of
     the two corrections the driver-derived version needed: pinned KV holds live
     regions, so it is excluded by construction rather than by discount, and
     evictable KV shows up as free regions the moment the relief pass ahead of
     admission evicts it. Measured, not forecast.
- The transient side is governed by §3.6's two time scales: the fast
  per-domain gate against current capacity, and the slow watermark regulator
  positioning the boundary to maximize admitted work while minimizing
  evictions and keeping the span tight. The failure mode for an oversized
  wave moves from mid-forward WDDM spill to a wait; a persistent demand
  shift moves the boundary instead of failing forever.

### 3.9 What is knowingly given up

- **Runtime elasticity between experts and KV.** Already fictional — the
  expert cache has no relief hook and never shrinks. The split becomes an
  explicit startup decision (step 7 tunes it).
- **Returning VRAM to other processes mid-run.** The reservation is held for
  the process lifetime; the balloon already excludes the OS reserve and
  desktop working set from `C`, and a hot reservation has maximal WDDM
  residency priority. This is the intended posture.

### 3.10 Fragmentation model

The allocation streams, modeled by lifetime — the axis fragmentation actually
cares about:

| Stream | Where | Lifetime | Free correlation |
|---|---|---|---|
| Active writer KV (R16 K / F16 V) | classes 4096 / 2048 | one turn (sec–min) | turn-correlated: seal frees a turn's active chunks together |
| Sealed quant KV | classes 320–1152 | until demotion (min–hours) | conversation/LRU-correlated |
| Transients | bump side (§3.6) | one wave (~tens of ms) | perfect: cursor reset at wave end |
| Gallery pages | class 6144 | turn, LRU-evicted | good |
| Meta records | record class (at fold-in) | tracks its chunk | tracks quant |

**Problem 1 — transient/KV interleaving (severe; designed out).** If
transient allocations drew from the shared region pool, a wide wave would
claim ~30 regions and release them ~50 ms later; any concurrent seal pass
hole-fills into them, and a single long-lived quant chunk pins a region the
transient tide just vacated. Over hours, long-lived chunks diffuse into every
region transients ever touched: the free-region list runs dry while aggregate
free space is huge. No allocation-order policy fixes mixing ~50 ms and ~1 h
lifetimes in one pool — so the design removes the sharing entirely (§3.6):
the two sides are strictly segregated across the floating boundary, and the
region pool only ever holds turn-or-longer lifetimes. The two-directional
layout adds the key synergy: lowest-first packing was chosen (problem 2,
below) to keep the high frontier clean — which makes the rightmost regions
precisely the cheapest for the boundary to claim. The two policies form one
invariant: **KV packs left, so the right edge is always the cheapest to
reclaim** (principle 5).

**Problem 2 — hole-fill scatter of long-lived quant chunks (moderate;
bounded).** Within a class, a seal pass's bulk allocation fills slots freed
by many earlier turns across many regions, destroying turn-clustering; later
demotions then leave regions half-full rather than empty. Within-class this
costs *nothing* — free slots are perfectly fungible and allocation stays
O(1). It costs only when another class needs a region and none is free.
Lowest-first packing shapes the damage: mixing concentrates in low regions
that are permanently in use and never need reclaiming, the high-water
frontier stays clean, and empty regions surface at the top whenever a
class's demand contracts. The residual is the **evacuation tax**: when the
free list is empty, demote the rightmost region's live chunks to warm — at a
typical 10–30 % live that is ~1.6–4.8 MiB of DtoH per 16 MiB region
reclaimed, spent on chunks that were bound for warm eventually. Turn sizes
make clean frees realistic: one ~1 K-token turn seals ≈ 49 K quant slots
(~3.4 regions of the 1152 class) — multi-region scale, so correlated frees
do empty regions outright when scatter is low.

**Problem 3 — rare-class region stranding (small; fixed by promotion).** One
Q1_S chunk would stamp a 16 MiB region for the 320 class that never fills.
Scarcity-only class promotion (§3.4) places trickle demand into a larger
class's free slots instead; the stranding bound drops from "one region per
class with any demand" to zero.

**Interaction that remains by design**: the active (4096/2048) ↔ sealed
quant (320–1152) flow is the engine's metabolism — prefill grows active
classes, sealing converts active bytes to quant bytes continuously. Active
frees are turn-correlated and empty regions readily; the free-region list
plus the evacuation tax buffer the quant side's slower contraction. This
flow is the free-region setpoint's whole job (§3.8), and it operates on
exact counters.

---

## 4. Invariants that must hold throughout

1. **Kernel addressing**: `data = ptr + byte_offset + chunk_idx ×
   chunk_byte_stride` with per-band format tags from `PerHeadTable`
   sub-entries. Classes change only which values get serialized, never the
   formula.
2. **Position-agnostic sealed chunks**: un-rotated K, RoPE at read from the
   destination slot's cumulative usage. Untouched.
3. **COW / refcount semantics** of `ChunkGid` clone/drop, single-writer
   partial tails, `writer_start_idx` discipline. Untouched.
4. **Zero-on-recycle**: a recycled slot is zeroed to the *class* stride
   before reuse (persist quantize reads past `token_count`).
5. **Byte-identical persistence**: warm/cold images and redo-log records are
   unchanged — format tags were already per-chunk on disk. Serialization
   tests assert raw bytes, per repo policy. **This is not free under size
   classes**: the persist path is stride-driven end to end, so it requires
   the payload/stride split of invariant 8 (audit A1, §8).
6. **u16 recycle links**: chunks-per-region ≤ 65,535 for every class
   (min class 320 B ⇒ 52,428).
7. **Lifetime segregation** (principle 2): no transient allocation in a
   class region, no class chunk in the bump side, at any time.
8. **Payload bytes ≠ address stride.** Two lengths, and they have *different
   owners*. The **stride** is the arena's: `ResolvedArenaInfo.chunk_byte_stride`
   is the class stride, used for address arithmetic only
   (`base + idx × stride`) and for zeroing on recycle (invariant 4 — the next
   tenant may be any format). The **payload** is the *chunk's*: it comes from
   the band's own format tag via `payload_bytes_for_tag`, and it is every copy
   length, `arena_byte_size`, and the persist blob. Conflating them makes every
   hot→warm migration move pad over PCIe and silently changes on-disk image
   sizes and their Fletcher goldens. **Seven** sites consume the length as a
   copy/extent rather than an address step, and they span three crates —
   enumerated in §5 step 1; A1 found the class, and the sweep that followed
   found the rest.

   **As built (2026-08-07):** an earlier draft of this invariant put *both*
   lengths on `ResolvedArenaInfo`. That was wrong in the same way the whole
   initiative is about — an arena holds whatever fits its slots and cannot say
   what format a given slot is, so it cannot supply a payload. The field was
   removed along with the struct's two format tags; the payload now comes from
   the tag, which is also the byte the substrate persisted.

---

## 5. The seven steps

Each step ends with the gate test (all configs pass; tokens/s and per-phase
timings recorded in `docs/arena_unification_results.md` alongside the step).
Capture the **baseline** run before step 1.

### Step 1 — Unify allocation across quants and arena types (size classes)

*Scope*: `candle-nn/src/kv_cache/`, one `candle-core` storage variant
(`Backing::Lease`), one CUDA function (`resolve_band_source`), and **one
`candle-conversation` persist site** — `seal_to_chunk_images_cpu`
(`transfer.rs:269`) is a payload/stride consumer and cannot be left behind
(see the split below). The CUDA allocator is still called per region;
defrag/compaction still exist (they fire far less).

- Introduce the **`Backing::Lease` storage variant** (§3.7) — needed here
  only so the legacy `read_contiguous`/`write_contiguous` façade can take
  on-demand tensor views over region bytes. No lifetime parameterization.
  The candle-core surface is **measured, not estimated** (E4, §8.2): 4 new
  declarations (`Backing`, `CudaStorageSlice::Empty`, `CudaStorageSlice::leak`,
  `Drop for CudaStorage`), **18** exhaustive match arms, **43** struct
  literals (35 candle-core + 8 candle-nn), 2 imports — 115 insertions total,
  workspace **and tests** compiling clean. Every one is compiler-enumerated
  (E0004 / E0063), so there is no silent-failure surface; budget it as an
  afternoon of mechanical edits, not a refactor.
- Add per-band format tags to `ChunkWindow` and `SealedChunk` (parallel to
  `gids`; `Arc`-shared per block like pal/scale), then invert every
  format-from-arena derivation.

  **Corrected by the 1.4 sweep — there are thirteen readers, not the nine
  listed here originally.** Eleven are inverted in 1.4; two are structurally
  blocked on later sub-steps and are recorded with their reason rather than
  papered over. Two further sites turned out to be dead code and are deleted.

  | # | Site | Disposition |
  |---|---|---|
  | 1 | `KvHeadHost::from_gids` (`slot_state.rs`) | takes per-head `k_fmt`/`v_fmt`; arena supplies the address only |
  | 2 | `serialize_kv_heads` (`meta_pool.rs`) | the resident-record twin of #1; **absent from the original list** |
  | 3 | `per_head_table_host` (`backing.rs`) | **deferred to the re-index** — the row *is* the arena, so inverting it and re-keying it are the same edit |
  | 4 | `kv_formats_for_gids` (`chunk_ops.rs`) | **deleted.** Its three consumers all mapped the result through `KvFormat::to_tag`, which is defined as `ArenaFormatTag::from_kv_format(..).as_u8()` — i.e. the chunk's tag bytes already. They now copy `SealedChunk::format_tags()` |
  | 5 | `ensure_writable_tail` (`sequence_ops.rs`) | **deleted — zero callers.** See the note below |
  | 6 | `bucket_quant_chunks` (`compress.rs`) | format from `SealedChunk::bands()`, location still from the arena |
  | 7 | `gpu_format_stats` (`gid_pool.rs`) | **deferred to class-occupancy logging** — it aggregates over *pool keys*, not chunks, so it has no chunk to read from until `ArenaKey` carries a class |
  | 8 | `PalHeadDesc` source probe, quantize (`compress.rs`) | `k_is_r16`/`v_is_r16` from the source chunk's tags |
  | 9 | `build_meta_records` (`backing.rs`) | takes `ChunkRecordSrc`, feeds #2 |
  | 10 | `sealed_has_compressible_chunk` (`backing.rs`) | same predicate as #6 |
  | 11 | `dequantize_sealed_in_place` bucketing (`compress.rs`) | **new** — same shape as #6, separate call site |
  | 12 | `resolve_src` closure, dequantize (`compress.rs`) | **new** — resolved `(ptr, GgmlDType)` from `arena.format()`; now from the band tag via `KvFormat::from_tag` |
  | 13 | The `[pal4]` verbose grid (`batched_inference.rs`) | **new** — a diagnostic, but the one that *displays* per-band format variety. Built from arena state it would print every band of a shared region identically, i.e. it would lie about exactly the property this design introduces |

  Two functions in the sweep turned out to be dead and are deleted rather than
  ported: **`ensure_writable_tail`** (`sequence_ops.rs`, zero callers — and its
  `any_quantized` probe was already vacuous on GPU, where the active K arena is
  `Quantized(R16)`, so it would have pushed a fresh block on every call) and
  **`Cache::chunked_per_head_table_and_sync`** (`cache.rs`, zero callers, with a
  doc comment claiming it feeds the decode kernel — false, and plausibly the
  origin of the belief that attention depends on the per-head table).

  Also deleted: **`ArenaFormatTag::from_ggml_index`** — zero callers, and a
  byte-for-byte duplicate of the tag-decode table, under a name and doc comment
  that describe GGML type indices it does not actually map (its own doc says
  `Q4_0=2` while the body maps `2 => BF16`). The surviving decoder is
  `ArenaFormatTag::from_u8`, the declared inverse of `as_u8`.

  **The gate is the proof, not the assert.** 1.3's `debug_assert_tags_match_arenas`
  is compiled out under `--release`, so it can never fire during the gate. What
  makes 1.4 safe is stronger: after the inversion the model's attention and
  selection paths read formats *only* from chunk tags, so a wrong or missing tag
  produces mis-decoded KV and a failed validity check. Sixteen green configs
  spanning `F16`/`BF16`/`Q8_0`/`Q4_0` and `C0–C7`, `C9` are therefore direct
  evidence that the tags are correct on the real workload — which is what §1.5
  needs before it deletes the arena's copy.
- **Re-index the selection table per `(chunk, head)` and activate the
  palette sub-entry** (§2) — the **one kernel-visible edit** size classes
  require. It must land in the same commit as the class switch, or selection
  silently reads wrong formats. Three parts, all local:
  1. **Kernel**: inside `resolve_band_source` alone, drop
     `arena_idx = gid / arena_chunks` as the row key and index by the
     `chunk_idx` it is already passed, then take `.palette[palette]` instead
     of `per_head_lookup`'s hard-coded `.palette[0]`. `chunk_in_arena =
     gid % stride` still comes from the gid for the address offset. **All
     five `resolve_band_source` call sites already pass `chunk_idx` and stay
     unmodified**; no struct, grid, or launch-config change.
  2. **Host**: rebuild `per_head_table_host` as one row per `(chunk, head)`,
     populating each of the four sub-entries from *that band's own* gid →
     region pointer + the chunk's own format tag. This is the same loop
     `KvHeadHost::from_gids` already runs for the attention path — the two
     builders converge, and the per-layer `arena_offset`/`gid_off` rebasing
     in `from_head_gids_multi` disappears (rows simply concatenate).
  3. **Delete** `ArenaEntry`'s per-arena `k_format_tag`/`v_format_tag` and
     `actual_kv_format_tags`: audit Q1 confirms their only readers are
     `arena.rs:928-929` and the two table builders at `backing.rs:1613-1616`
     / `:1713-1716` — precisely the code being rewritten here.

  **Corrected: "must land in the same commit as the class switch" is
  one-directional.** It forbids the class switch from *preceding* the re-index;
  it does not force them together. Populating each palette sub-entry from its
  own band's gid is behaviour-neutral while bands are still one-arena-per-format
  — the four sub-entries carry exactly what the single palette-0 row carried —
  so the re-index lands and gates on its own, and the class switch then arrives
  into a selection path that already reads per-band data. That ordering was
  taken, and it paid: it isolated two defects (a table not narrowed alongside
  its gids in `select_chunks`, and eleven stale hand-built table fixtures, three
  of them still on the pre-palette4 7-column layout) that would otherwise have
  landed tangled with the allocator rewrite. See
  `arena_unification_results.md` §1.5a.
- Introduce `SizeClass` and the ladder table (§3.4); `ArenaKey` →
  `(SizeClass, ArenaLocation)`; `arena_chunks_for_format` →
  `chunks_for_class`; `arena_gid_stride()` → the fixed 65,536; scarcity-only
  class promotion in the allocator.

  **This bullet and the byte-slab bullet below are one change, not two.** Once
  `ArenaKey` carries no `KvFormat`, `create_arena` has nothing to construct an
  `Arena::Float { dtype }` or `Arena::Quantized { format }` with — the storage
  variant is selected by the key's format today. Carrying a "representative
  format" on the arena so the constructor still type-checks would be
  optionality-as-a-feature-flag, and it would resurrect the second answer to
  "what format is this?" that step 1's reader inversion exists to remove. The
  class key and the byte-slab collapse land together.

  The fixed stride is a bigger win than "cheaper gid decode in the serialize
  paths" implies. `arena_gid_stride()` re-derives the maximum over **all 22
  `QuantFormat`s plus 3 float dtypes** through a `strum` iterator on every
  call, and it is called inside `ChunkGid::clone`, `drop`, `arena_idx`,
  `chunk_idx`, and `strong_count` (`gid_pool.rs:462/495/500/517/539`) — i.e.
  on **every refcount operation in the system**. It is not a `const fn` and
  nothing guarantees the fold. A compiled `1 << 16` makes it shift/mask by
  construction.
- **Port `validate_selection_gids`, do not delete it**
  (`backing.rs:1499-1541`, audit A14). It bounds-checks every selection gid's
  `chunk_idx` against `arena_chunks_for_format(arena.format())` before the
  table upload, and its second arm exists for a sanitizer-confirmed OOB read
  at exactly slab end. Classes make its failure mode *rarer*, not impossible:
  a region freed and re-stamped to a different class has a different stride,
  which is the same re-tenancy hazard under a new name. Rewrite the bound as
  `chunks_for_class(class)` and keep both arms.
- Collapse `Arena::Float`/`Arena::Quantized` into **one byte-slab variant**
  with a class stride (§3.7): `zero_chunk_at` → memset, the pinned
  read/write paths → memcpy, pointer resolution → `base + idx × stride`.
  Delete the `PagedKvArenas` trait and its impl — empirically dead
  (experiment E2, §8: deleting both leaves `candle-nn` lib **and** tests
  compiling clean). The inherent `float_arenas()`/`quantized_arenas()`/
  `k_arenas()`/`v_arenas()` methods and their ~19 test references are
  rewritten to byte-slab equivalents or deleted with the tests they serve.
- **Split payload from stride** (invariant 8): add `chunk_payload_bytes` to
  `ResolvedArenaInfo` and switch **all seven** copy-length sites to it. Leave
  every `base + idx × stride` addressing site on `chunk_byte_stride`, and
  leave `zero_chunk_at` on the stride. The verified inventory:

  **Corrected by the step-0 pre-flight — there are nine sites in two
  opposite-facing categories, and one of the seven listed here was
  misclassified.** The verified inventory (all landed in 1.2):

  **Category A — payload inferred from stride** (→ `chunk_payload_bytes`):

  | # | Site | Role of the length |
  |---|---|---|
  | A1 | `chunk_ops.rs:1987` | DtoH gather copy length (and the `src_ptrs` len half) |
  | A2 | `chunk_ops.rs:2206` | cross-layer gather copy length |
  | A3 | `head_gids.rs:185` | `arena_byte_size` → `SealedChunk.byte_size` |
  | A4 | `migrate.rs:199` | `resolve_sealed_chunk_ptrs` → `(ptr, len)` |
  | A5 | `migrate.rs:234` | `resolve_sealed_chunk_ptrs_per_gid`, same |
  | A6 | `migrate.rs:423` | `resolve_block_ptrs_from_hgids` — **missed**; feeds cold-load via `pipeline.rs:589` |
  | A7 | `transfer.rs:269` | `seal_to_chunk_images_cpu` blob slot reservation |
  | A8 | `chunk_ops.rs:2597` | `chunk_byte_size_of(arena)` in the HtoD scatter — **missed**, and not a `chunk_byte_stride` expression at all |

  **Category B — stride inferred from payload** (→ the slot stride, via the
  new `slot_stride_of` helper). A category the original sweep did not have:

  | # | Site | Expression |
  |---|---|---|
  | B1 | `chunk_ops.rs:166` | `read_chunk_into_pinned_bytes`: `byte_offset = chunk_idx * dst.len()` |
  | B2 | `chunk_ops.rs:2498` | `write_chunk_from_pinned_bytes`: `byte_offset = chunk_idx * bytes.len()` |

  Category B fails in the *opposite* direction from A: these derive the
  **address step** from the **payload length**, so under classes they address
  slot `n` at `n × payload` and read or write a neighbour. That is data
  corruption, not a size mismatch, and no length assertion would catch it.
  They take their stride from the arena inside the callee rather than from a
  caller-passed value, so there is no way to pass a wrong one.

  **`chunk_ops.rs:2755` is addressing-only and must be LEFT on the stride** —
  the previous listing had it as a payload site. Its length comes from
  `gid_byte_range`, populated at A8, so fixing A8 fixes the scatter
  transitively and "fixing" 2755 would break it.

  **A7 is the dangerous one.** `seal_to_chunk_images_cpu` appends a slot to the
  blob per unique gid, then splits that blob by `sc.byte_size`. Today the two
  agree by construction. The moment `byte_size` goes payload-summed while the
  gather still reserves stride, they disagree — visibly as
  `seal_to_chunk_images_cpu: blob underrun`, invisibly as every chunk after the
  first landing on shifted boundaries in persisted data. **All ten must move in
  one commit**; a partial conversion is worse than none, because sites that
  still agree mask the ones that don't.
- Land four **permanent** invariant tests. Three are the E1 harness,
  promoted: every `QuantFormat` and float dtype maps to a class; every class
  yields ≤ 65,535 chunks/region; `GID_STRIDE` exceeds the max chunks/region.
  The fourth is **new and load-bearing**: assert that a sealed chunk's
  `byte_size` equals the sum of its bands' **format** bytes, independently
  computed — because the existing round-trip tests cannot catch a
  payload/stride confusion (audit A7, §8: they read
  `chunk_byte_size_of(arena)` on *both* sides, so they compare equal even if
  both sides copy pad).
- Delete the vestigial `gpu_quant_kv` constructor and update the stale
  `docs/kv_cache_unification.md` references in `arena.rs` to this document.
- Class-occupancy logging (per-class regions / live / free) replaces
  `gpu_format_stats`'s float/quant split, keeping the slack observable.

**Move `GpuChunks` device buffers to the region tier** (audits A10, A13) —
neither bump-side nor the pool remnant. The per-`(layer, batch_idx)`
decode slot-state buffer is **cached across waves** —
`sync_decode_gpu_chunks` returns `DecodeGpuChunksSyncKind::Reuse` with the
existing `raw_device_ptr()` whenever the chunk count still matches — so a
per-wave cursor reset would recycle it under a live sequence. By principle 2
it is turn-or-longer, therefore arena-managed:

- **A doubling class family** (4 KiB, 8, 16, … 1 MiB), one slot per
  `(layer, batch slot)`. Growth is a **promotion**: claim the next class up,
  copy, release the old slot — exactly what `resize` does today, minus the
  allocator and minus the stream sync. A 1 MiB slot holds 65,536 chunks
  ≈ 2 M tokens per sequence per layer; a 16 MiB region caps at ~33 M.
- **Contiguity is preserved.** The kernel reads the buffer linearly from
  `raw_device_ptr`; a region slot is contiguous, so it stays a plain pointer.
  No paging, and the paged-output contingency remains unbuilt.
- **The cost is a rounding error**: 16 B of slot-state per chunk against
  ~37 KB of KV for that same chunk (32 bands × ~1152 B) = **0.04 %**.

The host-side `PinnedBuf` half has the same problem and the same answer, but
its home is the pinned host reservation (S7, follow-on); until then it stays
as-is. This lands in step 1 because it needs only the class machinery, and it
retires the decode-path churn A13 measured — a win the step-1 gate should
show directly.

*Gate*: correctness across all configs; a measurable drop in arena count and
reserved-vs-live gap; **decode-path allocator traffic falls to ~zero** (A13's
per-32-token alloc/free/sync cycles disappear — check `KV_ARENA_STATS` and
pool counters, and watch for a decode-latency improvement); no perf
regression. Run the migration byte-identity round-trip tests alongside the
model test (audit A2's second gate), plus the new `byte_size` assertion.

Two numbers to record specifically, because both are predictions this step
makes and neither is visible in tokens/s:

- **Fused selection-table bytes** (E5): expect a fall at typical drain
  widths — 17.7 MiB → ~1.7 MiB at 336 arenas / a 1000-token turn — and the
  scaling to switch from arena count to chunk count.
- ~~**Where the `GID_STRIDE` win lands** (A15)~~ — **answered: nowhere.** The
  swap measured no change at all (115.70 s vs a 119.37 s baseline, every config
  at or above it), so LLVM was folding `arena_gid_stride()`'s `strum` iterator.
  The constant stays — it is a precondition for the class switch and it retires
  an unproven assumption — but it buys no speed. Recorded so nobody re-derives
  the expectation later.

---

#### Step 1 as built (2026-08-07) — where the plan above and reality diverged

The plan above is left as written; this records what actually landed and every
place it differs. Measurements are in `docs/arena_unification_results.md`.
**Gate: 16/16, 124.47 s** against a 114–120 s band.

**Predictions that held.**

- The class ladder covers the whole format space; the gid pool table collapsed
  from ~58 per-format pools to **14** (7 classes × 2 locations).
- The contiguous-run eligibility test became a question about *keys*, so bands
  in different formats that share a class can still form one run — it fires
  strictly more often, as predicted.
- `PagedKvArenas` and the `Arena::Float`/`Quantized` duality deleted clean, as
  E2 said they would.
- Every payload/stride site moved together, and none needed a second pass.

**Predictions that did not.**

- **`arena_gid_stride()` → `GID_STRIDE` bought nothing.** Measured at parity;
  LLVM was folding the `strum` iterator. The constant stays — it is a
  precondition for the class switch — but it is not a speedup. (Already
  corrected in the step-1 text above.)
- **`GpuChunks` on the region tier is currently a ~6 % *cost*, not a win.**
  A13's alloc/free/sync cycles are gone, but its slabs have nowhere to live
  until step 4 and so compete with KV for the same pool. Kept rather than
  reverted because the mechanism is right and step 4 removes the contention by
  construction; the residual 6 % is not visible in per-config throughput and
  needs a profile. See §1.8 of the results log.

**Things the plan did not know about.**

- **`QTensor::from_leased_cuda_ptr` had to be built.** The byte-slab arena
  removes the typed `QTensor` that `quantize_into` writes *through*, and
  1.5b(i)'s `Backing::Lease` covered plain `Tensor` only. Two details are
  load-bearing: `Clone` must return `Backing::Owned`, because
  `CudaSlice::clone` is a device-to-device **copy**; and the view carries no
  matrix-row padding, so it is a block-quantize operand and never a matmul one.
- **`migrate_chunk` cannot convert formats, and never needed to.** Every
  production caller — hot→warm demote, warm→hot elevate, fork — is
  format-preserving; the converting arms were reachable only from tests. It is
  now a byte-verbatim slot relocation that *requires* equal classes.
- **Scarcity promotion must reuse, not re-stamp.** Every class's region is the
  same size, so walking up the ladder stamping regions fails identically at
  every rung while paying `ensure_vram_budget`'s global compaction each time.
  Promotion takes a free slot from a class that already has a region, or it
  does not happen.
- **Two allocation-order dependencies surfaced** once slots were bounds-checked:
  `read_raw_sealed_chunk` sized a read as a whole head and issued it against
  palette 0's slot, reaching the rest only because runs are *usually*
  contiguous; and `write_contiguous`'s quantized arm wrote into a
  `data.clone()` — a deep copy — and dropped it.

**Deletions beyond the inventory.** `ArenaEntry` and its tensor-row codec, the
CUDA `ArenaTableEntry` struct and its eight accessors, `actual_kv_format_tags`,
`count_quantized_arenas`, `chunked_live_chunks_as_sealed_with`,
`tensor_ptr_at_offset` / `qtensor_ptr_at_byte_offset`, and
`convert_chunk_data_static` — all dead once formats left the arena.

**Naming.** `chunks_for_class` is `SizeClass::chunks_per_region`;
`slot_stride_of(arena)` is `Arena::slot_stride`; `gpu_format_stats` is
`gpu_class_stats` returning per-rung `ClassOccupancy`. `ResolvedArenaInfo` lost
`chunk_payload_bytes` and both format tags — a band's payload comes from its
own tag via `payload_bytes_for_tag`, which is the single path from a persisted
byte back to a byte length.

**A12 is now answerable.** `ChunkedKvBacking::class_histogram` reports live
slots per class and, separately, those in formats narrower than the 320 B
floor — the numerator of the ~2 % decision rule. Step 6 makes the call.

### Step 2 — Port transient buffers to the bump side

*Scope*: identify and convert the inference loop's transient device buffers
to the bump side (§3.6). Buffers stay contiguous — no kernel changes in this
step. **Provisional backing**: until step 4's reservation exists, the bump
side is a standalone device allocation sized generously from measured peaks
(~1.5 GiB); step 4 swaps its backing for the reservation's right side and
the floating boundary — the `BumpArena` API and every call site are
unchanged by the swap.

Candidates, in order of confidence:

- kernel argument/metadata blobs: chunk-meta rows, per-head tables, head-gid
  uploads, selection tables (`PagedSelectionGpuInputs`), migration
  descriptors;
- migration staging slices (`copy_stream.alloc`, ≤ 512 MiB cap) — allocated
  from the persistence domain's sub-range;
- **logits** (audit A11: wave-scoped — produced by the forward, consumed by
  sampling in the same wave; `BatchedSampler` holds *no* device state at
  all, so nothing else from sampling needs placement);
- grow-only scratches: `ProvSignScratch`, `KvSamplerGpu`, MoE routing
  buffers — moved to the static shelf.

**`GpuChunks` is deliberately absent from the candidate list above** — it is
cross-wave state and moved to the region tier in step 1 (A10/A13). Any buffer
whose lifetime is not provably within one wave gets that same classification — the mid-wave-allocation debug assert is what catches a
misclassification.

Build the unified `BumpArena` abstraction by generalizing `PinnedStager`
(host-pinned + device instances, shared `Generation` lifecycle), the
per-wave double-buffered generations with counted, event-fenced reset, and
the per-domain sub-ranges. Establish the **wave buffer set** (§3.6):
allocate-at-wave-start, the no-mid-wave-allocation debug assert, and the
set's byte total as the admission input.

*Gate*: correctness; per-wave buffer-set size and per-domain peaks logged
(these fix the boundary position in step 4); record cudarc pool
`reserved`/`used` — both should fall.

### Step 3 — Intermediate buffers in the inference loop; paged writes only if needed

*Scope*: bounded to the inference loop. Move the inter-layer hidden state,
attention outputs, and MoE combine outputs (`ys`) onto wave-plan slots in
the bump side — contiguous fixed addresses, so QMatMul and the attention
kernels take them as ordinary output pointers, and the per-layer `Tensor`
creations become lease-backed views over the ping-pong buffers (step 1's
`Backing::Lease`; the layer code stays a single code path).

**Baseline (a) vs optimization (b) for interior op outputs.** Candle ops
that allocate their outputs internally (`matmul` etc.) can't take a plan
slot without out-variants. Baseline (a): our custom kernels take
preallocated leased Tensors; interior op outputs stay on the pool remnant.
Deferred optimization (b, step 6, gated on the leak counter staying clean):
a `WaveAllocScope` RAII that routes `device.alloc` to the wave generation
for its extent, landing every interior output in the plan with zero
call-site changes (the PyTorch stream-scoped-mempool precedent, and the
allocation shape CUDA graph capture wants). Anything allocated inside the
scope that must outlive the wave becomes a leak the counted reset detects
loudly.

**The paged-output contingency is not built.** The audit it was waiting on is
done (A10/A11, §8): the only buffer that outlives its wave is `GpuChunks`,
and that is host-built metadata uploaded by memcpy, not a kernel output — it
goes to the region tier as a contiguous slot (§5 step 2), so it needs no
paging either. No kernel output needs paging, so no
`page_table`/`PagedTensor`/paged-`QTensor` work happens. `PAGE_BYTES` stays a
reserved constant (§3.5) so the option survives if a future buffer needs it;
nothing implements against it today.

*Gate*: correctness (bit-identical where the op is deterministic);
per-forward latency within noise of baseline on decode and wide-prefill
shapes.

### Step 4 — Static reservation: consume the memory at startup

> **Built 2026-08-08.** Gate green: `pool_reserved` flat at 8,858,370,048 B
> across configs 1–15 of the MoE gate, 562 arena creations at 0.029 ms each,
> every region returned. Five corrections to this document came out of it — the
> startup order (§3.2), the free-region list (§3.3), `W_persist` (§3.6),
> positional evacuation (§3.8), and the unbuilt `cuMemAlloc` fallback (§3.2) —
> each recorded inline above. Full record in `arena_unification_results.md`.

- Startup sequence per §3.2: reserve the VA span; the balloon maps granules
  to the driver's refusal (measuring `C` and claiming it in one act);
  release the partition's granules for dense weights + expert cache + small
  reserve; load them; the remaining mapped granules are the reservation.
  Runtime probe falls back to the giant-`cuMemAlloc` order. Carve regions,
  seed the free-region list.
- Region creation/release stop touching the CUDA allocator entirely;
  `create_arena` → carve, `release_arena` → push free list.
- Swap the bump side's provisional backing (step 2) for the reservation's
  right side, at a **fixed, region-aligned boundary** sized from step 2's
  logged peaks. `drain_free_arenas_above(k)` keeps the free list on the KV
  side. The boundary does not move in this step — see §9 S6: the *layout*
  carries the segregation and contiguity benefits; *motion* is a step-6
  optimization only if measurement shows the fixed split wastes real memory.
- Wire the free-region counter as the pressure signal; add rightmost-first
  evict-as-evacuation (§3.8) to the persistence thread's demotion pump.

*Gate*: correctness; startup time; confirm zero CUDA allocations during
steady state (`KV_ARENA_STATS` / pool counters flat).

### Step 5 — Rip out everything no longer needed

> **Built 2026-08-08.** Every item below is deleted. Gate green: MoE
> `test_parallel_batched_forwarding` 120.32 s, candle-nn 423/0, candle-core
> `vram::` 30/0, candle-conversation 950/0, both `cargo check` branches clean,
> clippy 227 (below the 235 baseline). Step 5 as a whole: 77 files,
> 6,395 insertions / 10,974 deletions. Full record in
> `arena_unification_results.md`.
>
> Nine symbols beyond the inventory came out with it, each because its last
> caller was inside the deleted set: `VramGovernor::{allocate, reserve,
> forecast_units, available, do_sync, with_sync_hook, with_reuse_hook,
> spawn_budget_watcher}` and `CudaDevice::trim_pool`'s runtime callers (the
> method survives for the startup balloon alone). The pressure signal and the
> admission ceiling were *rewritten*, not merely stripped — see §3.8 and the
> results doc.
>
> Two defects surfaced while deleting, neither related to what was being
> deleted. Every `#[cfg(feature = "cuda")]` block in `candle-conversation` is
> dead code — the crate has no such feature, it forces the feature on its
> dependencies — which had silently disabled all of step 4's `kv-regions`
> telemetry. And `demote_cold_ingest_if_pressured` was gating on `pool_used`
> against a fraction of `C`, a reading that stopped describing KV when KV left
> the pool, so it would have fired on every wave.

Deletion inventory (all now dead by construction):

- **Budget gate**: `ensure_vram_budget`, `vram_gate`, `VramGateFacts`,
  `gate_decide`, `vram_has_room`, `kv_alloc_headroom`, `vram_reserve_bytes`,
  `eviction_reserve_bytes`, `EvictionScope` and the eviction reserve.
- **Reclaim**: `request_global_compact`,
  `release_empty_arenas{,_forced,_inner}`, `defragment_arenas`, `DrainPlan`,
  `allocate_avoiding`/`allocate_for_avoiding`, `CompactMove` +
  `arena_compact_copy_async` + `apply_gid_remap`, `try_tombstone`'s
  force/headroom split, `can_reclaim_arena`, `needs_defragmentation`,
  `defragmentable_ratio`, protected-arena bookkeeping,
  `compact_arenas{,_forced}`, `defragment_bounded`.
- **Pressure subsystem** (`candle-conversation/src/scheduler/`):
  `relieve_vram_pressure` and its ladder, `vram_under_pressure_for`,
  `relieve_compression_starvation`, the **18** wave-boundary
  `defragment_bounded`/`release_empty_arenas_forced` call sites in
  `prefill.rs` plus the unforced sweep in `run.rs:484`, driver-headroom
  admission terms (replaced by the region counter + the plan-byte width
  gate), the `migrate_guard` relief windows that fence compaction.
- **Governor runtime role**: relief rungs, `evictable_estimate`-driven
  admission; the balloon (now reservation-mapping, §3.2) and startup
  partitioning remain.
- **The arena-topology guard** (`migrate_guard.rs`): `MigrateGuard`,
  `ReliefGuard`, `enter_migrate`, `try_enter_relief`, and the process-global
  `ARENA_TOPOLOGY` `RwLock`. Its entire contract is protecting captured arena
  base pointers from free / relocate / truncate / `cuMemPoolTrimTo` — and
  under a permanent reservation **a base pointer can never be invalidated**:
  "freeing" a region is a free-list push that unmaps nothing (audit A4, §8).
  This also removes a process-global read-lock from every migrate and
  elevate, so it is a latency win as well as a deletion. Keep
  `migrate_in_flight` as a plain advisory counter — it drives a non-safety
  deferral (avoiding double-conversion), not a correctness guarantee.

**The gallery arena's relief registration** goes with the ladder, but its
*eviction* must not. Today `scheduler/mod.rs:2351-2371` registers
`evict_lru(want)` under `AllocClass::Kv` at `Criticality::Cheap`, so the
governor sheds resident galleries before touching model KV. Replacement: the
scheduler already owns `gallery_arena`, so the free-region pressure response
calls `arena.evict_lru(want)` **directly, ahead of** KV evacuation — the same
priority, expressed as call order instead of rung numbers. Delete the
registration block; keep `evict_lru` and `resident_bytes`.

Every deletion lands with its tests either deleted (tests of removed
behavior) or rewritten against the region model (tests of surviving
invariants).

*Gate*: correctness; the log must show zero relief/compaction activity under
sustained load.

### Step 6 — Optimize steady state in the gated test

> **Built 2026-08-08. Gate green: 16/16 configs at or above the step-0
> baseline** (table in `arena_unification_results.md`). Two changes carried it,
> both found by measurement rather than by the list below:
>
> - **The recycled-region quiesce cost 2,837 ms of a 120 s run** — 7.18 ms
>   average over 395 claims. Step 4 had measured it as free; step 5 invalidated
>   that by removing the per-wave `device.synchronize()` it had been riding on.
>   Fixed with a **quiesce epoch** rather than the per-region release event this
>   section proposed: one `cuCtxSynchronize` retires every kernel on every
>   stream, so it discharges every region released before it, and regions are
>   released and re-claimed in bulk. 395 waits → 15, 2,837 ms → 0.5 ms. The
>   release event stays unbuilt, now for a measured reason.
> - **The class ladder padded `Q8_0` (5.6 %) and `Q4_0` (10.0 %)** — the two
>   fixed formats, derived after the C-level ones. `Q8_0×20` and `Q4_0×20` were
>   the gate's only two configs below baseline, five runs of five. Ladder
>   7 → 13 rungs on the rule **coarse where region stranding dominates, exact
>   where read bandwidth does**: one catch-all rung absorbing everything ≤320 B,
>   an exact rung for every format above it. Both configs moved above baseline.
>
> **Boundary motion is now never built** (§9 S6's condition): KV peaked at 167 of
> 226 regions, so the fixed split strands nothing. The A12 low-end split also
> stays unbuilt — zero class promotions fired, so no class was starved.

- Profile the allocation fast path (claim, run claim, region pop), the
  transient generation reset, and eviction cadence under the widest gate
  config.
- **Boundary motion, only if earned** (§9 S6): if the fixed boundary from
  step 4 measurably strands memory (transient span idle while KV starves, or
  vice versa), add the hysteretic watermark regulator and rightmost-first
  boundary evacuation. If the fixed split holds, this is never built.
- ~~Evaluate `WaveAllocScope`~~ (step 3's deferred option (b)) — **evaluated
  2026-08-08, not adopted.** The leak counter is clean, so the gate opened; the
  second condition is what failed. The CUDA pool reserves once during load
  (30 → 7,232 MiB, flat across the cold-ingest peak *and* six concurrent
  conversations) and never grows again, with `used` swinging ~370 MiB inside it.
  Interior op outputs cost no VRAM, so routing them to the wave returns nothing
  to the expert cache or the KV side. Same verdict, same evidence, for the
  threaded pipeline's `ys` — which additionally needs the combine target to stop
  crossing a thread boundary by channel. Both become earned if `scratch_margin`
  must go under the ~370 MiB working swing (it is at 512 MiB), or if CUDA graph
  capture is pursued, which is the case the scope was really for.
- Tune the class ladder against the observed histogram; tune the free-region
  setpoint; verify the stride-65,536 decode shows up as a win in the
  serialize paths.

*Gate*: the test's tokens/s must meet or beat baseline on every config;
record the final table.

### Step 7 — Switch to the daemon; tune the partition

> **Partly built 2026-08-08.** The conversion was a no-op: `zend` has no
> allocator of its own, which is what one hopes from a change made at the right
> altitude. Two partition defects found and fixed, both by measurement:
>
> - **`scratch_margin` was double-booked** — subtracted when sizing the KV span
>   *and* the transient tier added on top of the result. A forward's activations
>   have come from the transient tier since step 3, so the cushion and the tier
>   were the same memory, reserved twice with opposite signs. The reservation is
>   now exactly `kv_floor`, both sides, with the cushion left outside it for what
>   still allocates from the CUDA pool.
> - **`expert_budget()` reserved against a card that had not finished loading.**
>   It runs before the per-layer dense tensors, so `usable` included memory
>   already promised to them and the KV side paid the difference. Declaring the
>   weights early was necessary but *insufficient* — `kv_floor` is
>   `abs + pct × (C − Weights)`, so declaring them lowers the floor and hands the
>   experts more (measured: KV 182 → 176 regions, the wrong way). The pending
>   bytes had to come off the **budget** at the loader's call site, which is the
>   only place that knows what has not loaded yet.
>
> **KV capacity +19.8 %** on the 16 GiB card (182 → 218 regions), `usable` at
> first KV cache 4,640 → 5,216 MiB, zero truncation, MoE gate green.
>
> **The partition is now measured, and the knob is retired.** Cold-booting the
> real `mind` corpus, KV demand is **70 regions (1,120 MiB) in steady state** and
> **284 regions (4,544 MiB) at the cold-boot peak** — boot needs 4× what running
> needs, and it is the peak that decides whether the daemon comes up. The
> `kv_floor` sweep:
>
> | `kv_floor_abs` | KV span | expert slots | residency | decode | boot |
> |---|---|---|---|---|---|
> | 3 GiB (old default) | 218 regions | 2618 | 42.6 % | — | dies |
> | 4 GiB | 274 regions | 2267 | 36.9 % | 57 ms/fwd | ready |
> | 5 GiB | 328 regions | 1917 | 31.2 % | 67 ms/fwd | clean |
> | 6 GiB (the workaround) | 384 regions | 1566 | 25.5 % | 80 ms/fwd | clean, 100 regions unused |
>
> (measured with the 704 MiB transient tier; trimming it to 192 MiB moved
> 3.5 GiB to **278 regions / 2443 slots / 39.8 % / 57 ms**. **Shipped default is
> 3840 MiB** — interpolated between the 3.5 and 4 GiB points, not measured
> directly.)
>
> **1024 MiB of `kv_floor_abs` buys 56 KV regions and costs 351 expert slots,
> worth ~13 % of decode throughput.** The old 6144 workaround was the single
> worst point on this curve — it was sized against pre-unification arena slack
> that no longer exists, and it starved the expert cache to hold regions nothing
> ever occupied. Default is now 4 GiB and the workaround is deleted, not carried:
> **decode is 29 % faster per forward than the configuration this repo has been
> running.**
>
> Two things the sweep exposed, both fixed: `kv_floor` is routinely **not
> achieved** (the expert budget is taken before the dense weights finish loading,
> so the KV claim comes up ~1,079 MiB short) and *nothing reported it* —
> `shortfall` measures only the granule touch refusing. `[reservation]` now
> carries `floor_deficit` and warns. And the post-priming quantize drain ran
> without relieving first, so at a cold build's high-water mark it failed for want
> of one slot while relief stood ready to free 41 regions.
>
> **Still not done**: aggregate ingest t/s and warm-restart/concurrency probes
> against the pre-unification baselines.

- Move `zend` onto the unified allocator; re-run the daemon measurement
  suite (cold ingest, warm restart, concurrency probes with distinct
  `conv_id`s).
- Tune the startup partition — expert slots vs reservation vs small reserve
  — to the measured optimum on the 16 GiB card (the transient split needs no
  tuning: the floating boundary self-adjusts). The slack reclaimed in §1.3
  goes to expert residency and admission width; prior fixes took residency
  3065-broken → 2847 slots at 46.3 %; this step targets the next increment
  from the ~2 GiB recovered. Tune the watermark regulator and free-region
  setpoint against daemon workloads (ingest vs interactive mix).
- `CANDLE_VRAM_KV_FLOOR_MB` and friends collapse into the partition knobs.

*Gate*: daemon suite green; expert residency, decode t/s, aggregate ingest
t/s, and zero budget-exceeded/no-forward waves recorded against the
pre-unification baselines.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Transient allocations interleaving with long-lived KV across regions (§3.10 problem 1) | Designed out: the two-directional layout strictly segregates the sides across the region-aligned floating boundary; transient overflow into KV regions is forbidden (a wave that doesn't fit waits, the watermark regulator moves the boundary between waves). |
| Cross-class region stranding (many partially-full regions of one class while another class starves) | Lowest-region-first packing concentrates mixing in permanently-used low regions and keeps the frontier clean (§3.10 problem 2); rightmost-first evict-as-evacuation (~1.6–4.8 MiB DtoH per 16 MiB reclaimed) through the existing tiering path — no new mechanism. Rare-format stranding is closed by scarcity-only class promotion (§3.4). |
| Activation peak exceeds the current transient span on very wide waves | The fast gate degrades to narrower waves, never OOM; a persistent demand shift moves the boundary instead of failing forever. Initial position seeded from step 2's logged peaks (measured 666–1005 MiB prefill scratch). |
| Fixed boundary (step 4) mis-sized — one side strands memory | Sized from step 2's *measured* per-domain peaks, not a guess; step 6 adds boundary motion if measurement shows real stranding (§9 S6). Until then the failure mode is bounded slack, not a wedge. |
| Boundary thrash / press stalled by un-evictable writers — **only if motion is built** | Region-aligned quantum (16 MiB) + regulator hysteresis + the free-region gap at the frontier make free-region claims the common case; admission throttling drains writers toward seal; the regulator's time constant absorbs turn-granularity waits (§3.6). Deferred to step 6 precisely so this interaction is faced with data. |
| Buffer aliasing / use-after-reset (no allocator left to catch them) | Counted reset refuses while any lease is live (log + quarantine the half, never scribble); no-mid-wave-allocation debug assert; bump arithmetic is one audited function and returns disjoint ranges by construction; optional debug canaries for device-side overruns (§3.7). |
| Cursor reset races an in-flight kernel of the prior wave | A/B halves on one stream: a whole wave's work separates the reads from the writes, and same-stream launches complete in order. Domains that *are* cross-stream (persistence staging) fence instead — see §3.6's `Reclaim` note. |
| VMM path fails on a future driver/platform | **Accepted, not mitigated** — the allocator requires VMM and says so. `Reservation::reserve` probes `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` and fails with a message naming the capability and the reason there is no second path (§3.2); a device test asserts it. The giant-`cuMemAlloc` variant differs in eviction unit, partial release and capacity measurement, so it is a distinct implementation to be written against a real unsupported target, not carried untested. |
| Silent over-commit while mapping the reservation (`cuMemCreate` never refuses — §3.2) | Stop signal is a failed **touch**, checked per granule, failing granule unmapped+released; extent additionally capped by the existing balloon back-off policy so the desktop keeps headroom. Never treat `cuMemCreate` success as capacity. |
| Selection path reads a wrong format tag under mixed-format regions | The per-`(chunk, head)` re-index of the selection table (§5 step 1) must land in the *same commit* as the class switch; the gate test exercises the quantize-selection path on every config. |
| Class ladder mismatch with a future model's format mix | Ladder is one constant table; per-class occupancy logging makes drift visible; re-derivation is a constants change. |
| Warm-tier pinned reservation pressure on host RAM | Host reservation sized by the existing host-RAM budget admission; pageable fallback remains a config choice. |
| Losing the CUDA pool's implicit service of odd-shaped tensors | The small reserve keeps a working cudarc pool for everything outside the bounded conversion scope. |

## 7. Scope discipline — what was cut, and why it costs nothing

Each cut below was taken *after* the design converged, by checking the code
for what the mechanism would actually cost against what it would actually
buy. None erodes a §1.1 benefit.

| # | Cut | Measured cost avoided | Why no benefit is lost |
|---|---|---|---|
| **S1** | `TensorG<'a>` lifetime parameterization | **1311** `-> Result<Tensor>` signatures workspace-wide; **33** `impl` blocks on `Tensor` in candle-core | The counted reset catches strictly more (erasure seams a lifetime cannot type), never corrupts, and runs every wave — a leak surfaces on the first gated run either way |
| **S2** | `Arena::Float`/`Arena::Quantized` duality + the `PagedKvArenas` trait | **62** match sites across 6 files; trait deletion verified to compile clean (E2) | Under size classes the storage need not know the format (it is chunk metadata). Resulting primitives are *simpler than today's*: memset instead of `Tensor::zeros`+`slice_set`, memcpy instead of `narrow`/`flatten_all`/`to_vec1` |
| **S3** | The 6144 gallery-page class only | 1 of 8 classes | It exists solely for the gallery arena, which is out of scope (S5). **Quant coverage is kept in full**: the remaining seven classes cover all 22 `QuantFormat` variants and all four float dtypes. Sizing the ladder to the *current* candidate lists was rejected — those tables are provisional/per-model, overrides can force any format, and cold-loaded chunks carry persisted tags (§3.4) |
| **S4** | `PlanBuf<'gen>` + slot tables + disjointness bookkeeping | a whole handle-type layer | A bump allocator returns disjoint ranges by construction; the buffer set is just locals held across the layer loop. Canaries survive as an optional debug feature for device-side overruns |
| **S5** | Gallery-arena and meta-pool fold-in | two subsystem migrations | Both work today and are small. Crucially both **already obey principle 2's corollary** — they are arenas (slab allocators), just parallel implementations — so folding them in is *consolidation*, not correctness. Contrast `GpuChunks`, which genuinely buffer-allocates and therefore moves now (A13). The only coupling — the gallery holding the sole `register_relief` — is answered by driving its LRU from the free-region counter (a few lines), not by absorbing it |
| **S6** | Floating boundary *motion* in step 4 (layout still lands) | regulator + hysteresis + boundary-press-vs-writers interaction | The **layout** carries the segregation (§3.10 problem 1) and contiguity benefits; **motion** only removes a tuning knob. Fixed boundary from measured peaks first; add motion in step 6 only if it strands real memory |
| **S7** | Pinned host (warm-tier) reservation | a second reservation subsystem | Different memory (host RAM), different benefit (PCIe bandwidth), zero coupling to the VRAM wins. Follow-on initiative |

Net: the initiative touches `candle-nn/src/kv_cache/` plus one narrow
`candle-core` storage variant, one CUDA row-index change, one
`candle-conversation` persist-gather site (§5 step 1, site 7), and the
scheduler's signal wiring — instead of a workspace-wide type refactor.

## 8. Audits and experiments (2026-08-07)

Fifteen audits and five throwaway experiments run before implementation.
Four findings changed the design (A1 + its amendment, A10/A13, A14); one added
a deletion (A4); two re-priced a change (A15, E5); one overturned an earlier
experiment's conclusion (E4 vs E3); the rest confirmed it.

### 8.1 Audits

**A1 — Persistence boundary. FINDING: the persist path is stride-driven end
to end.** `HeadGids::arena_byte_size` sums `chunk_byte_stride` per distinct
`(arena, chunk)` slot; `seal_to_chunk_images` slices the gather blob by that
`byte_size` and stamps a **Fletcher `golden` per chunk over those bytes**; the
DtoH gather (`chunk_ops.rs:1987`), the cross-layer gather (`:2206`) and the
HtoD scatter (`:2755`) all use `info.chunk_byte_stride` as the **copy
length**. So under class strides, on-disk images would silently grow by the
pad, goldens would cover pad, and every migration would move pad over PCIe —
invariant 5 violated. **Resolved**: split payload from stride (invariant 8);
open question 2 is closed in favour of trim-to-format.

*Amended after a full sweep of the length's consumers*: A1 named four sites;
there are **seven**, and they span three crates. The three it missed are
`resolve_sealed_chunk_ptrs` (`migrate.rs:199`), its per-gid variant
(`:234`) — both returning `chunk_byte_stride` as the `len` half of a
`(ptr, len)` pair that becomes the GPU gather's `split_sizes` — and
`seal_to_chunk_images_cpu` (`transfer.rs:269`), which reserves a
stride-sized blob slot per gid and then splits that blob by
`sc.byte_size`. The last one sits in **candle-conversation**, outside the
crate scope step 1 originally declared; the scope line and the step-1
inventory are corrected accordingly. The lesson generalises: A1 audited the
*persist path* and found the class of bug, but the length escapes through
`(ptr, len)` return tuples that read like addresses. Audit the **consumers of
a value**, not the paths you expect it to travel.

**A2 — Gate coverage. Better than assumed.** The gate test spans `F16`,
`BF16`, `Q8_0`, `Q4_0` **and adaptive `C0–C7`, `C9`** (only `C8` is
absent) at 1–20 contexts (the baseline run corrected this from "1–48"; there is
also no `C10` config — 16 configs total), and drives `record_turn` → `quantize_sealed_in_place`
in-session (`batched_inference.rs:1254-1259`). It therefore exercises seal,
per-`(chunk, head, palette)` selection across nearly the whole format space,
multi-format arena allocation, and wave concurrency. **Blind spots**:
hot→warm migration, cold load, persistence/goldens, the gallery arena, and
multi-turn fork/CoW. The existing
`gpu_cpu_gpu_round_trip_{,_f16,_r16,_quantized}_is_byte_identical` tests in
`chunk_ops.rs` cover migration byte-identity and are the **second gate** —
run them alongside the model test from step 1, since A1's payload/stride
split lands there.

**A3 — Test blast radius: small; total surface larger than first counted.**
Re-measured across `candle-nn` + `candle-transformers` + `candle-conversation`
as `(total references, of which in test files)`: `*_arenas()` **19 / 13**,
`ArenaKey::` **61 / 28**, `arena_gid_stride` **55 / 8**,
`arena_chunks_for_format` **37 / 4**, `PagedKvArenas` **4 / 1**,
defrag/compact **48 / 1**. The original "~45 test-side references" counted
constructor calls in tests only; the *test* blast radius survives that
correction (≈55 test-side references, two thirds of them `ArenaKey::`
constructions in fixtures), but the **production** surface for
`arena_gid_stride` and `ArenaKey::` is 3–5× what was recorded, which is why
the fixed-stride change is graded as a hot-path win rather than a tidy-up
(see step 1). Classify each test reference as delete / rewrite / must-pass
before step 1 starts.

**A4 — Deletion safety. FINDING: one more subsystem dies.** See the
`migrate_guard` entry in step 5 — a permanent reservation makes base-pointer
invalidation impossible, so the process-global topology `RwLock` and its
guards go, removing a global read-lock from every migrate/elevate.

**A5 — Lock order: no new lock.** `GidPoolState.free_arenas` already is the
free-region list under the existing `metadata` mutex (§3.3). No new edge in
the `alloc_gate → tables` order.

**A6 — Pad reads: safe.** The selection kernel derives its read extent from
`blocks_per_head` (a launch parameter) × `quant_block_bytes(fmt)` — the
*format's* block size — never from the arena stride
(`select_kv_format.cuh:1556-1599`). In kernels `chunk_byte_stride` appears
only as an address step. Trailing pad is never read as data.

**A7 — Second-gate sensitivity. FINDING: the round-trip tests are blind to
the A1 bug.** `gpu_cpu_gpu_round_trip_*_is_byte_identical` compares byte
vectors built by `bytes_of_cpu_sealed`, which reads
`chunk_byte_size_of(arena)` bytes per slot — the *arena-derived* length — on
**both** sides of the round trip. A payload/stride confusion changes both
sides identically, so the test passes while pad is copied and persisted.
These tests prove round-trip *fidelity of whatever length is copied*, not
that the length is correct. Step 1 therefore adds a direct assertion on
`byte_size` against independently-computed format bytes (§5 step 1). This is
the one place the existing suite would have handed us a false pass.

**A8 — Palette sub-entry is vestigial today.** `per_head_lookup` returns
`.palette[0]` unconditionally, and `per_head_table_host` writes four
identical sub-entries; per-band format variety comes solely from `arena_idx`
differing per band (§2). The selection change therefore *activates* an
existing structure rather than adding one — but it also means the sub-entry
path has **never executed with non-identical entries**, so step 1's gate run
is its first real exercise. Treat the C-level configs as the acceptance
signal for it specifically.

**A10 — `GpuChunks` is cross-wave. FINDING: it must not go bump-side.** The
decode slot-state buffer is cached per `(layer, batch_idx)` and reused
whenever the chunk count matches (`types.rs:946-957`, the `Reuse` arm), with
a `PinnedBuf` + `CudaSlice<u8>` pair resized on demand
(`gpu_chunks.rs:104-130`). It was listed as a step-2 bump candidate; that was
wrong and is corrected — a per-wave cursor reset would recycle it under a
live sequence. **This is the answer to "does any buffer outlive its wave?" —
yes, but it is host-built metadata uploaded by memcpy, not a kernel output,
so it does not force the paged-output contingency.** That contingency stays
unbuilt. (A first revision parked it on the pool remnant; A13 overturns
that.)

**A13 — …and the pool remnant was the wrong home: it is a live decode-path
cost.** `GpuChunksGuard::clear` (`gpu_chunks.rs:275-287`) performs a **full
`stream.synchronize()`**, then drops the `PinnedBuf` (`cuMemFreeHost` —
page-unpinning) and the `CudaSlice` (`cuMemFree`). `clear` is invoked by
*every* structural mutation — `push_chunk`, `truncate_chunks`,
`extend_chunks`, `split_off_chunks`, `drain_front_chunks`,
`prepend_chunks`, `replace_chunks`, `invalidate_gpu_chunks` — and
`push_chunk` fires **every time a sequence crosses a 32-token boundary**. The
following decode then hits `rebuild_decode` → `resize` → a fresh
`PinnedBuf::alloc_owned` (`cuMemHostAlloc`) plus `stream.alloc`.

Steady-state decode therefore pays, per sequence per 32 tokens across 48
layers: 48 stream syncs + 48 host-pinned frees + 48 device frees, then 48
host-pinned allocs + 48 device allocs — **≈ 3,000 alloc/free/sync cycles per
32 decoded tokens at batch 64**. Full stream syncs serialise the pipeline and
`cuMemHostAlloc` pins pages in the OS. Leaving this on the pool remnant would
preserve a decode-path performance bug *inside* a design whose stated purpose
is deleting exactly this, and the buffer grows with context depth — unbounded
by the engine's premise — so a "small reserve" sized for it would drag
§1.3's pool-slack mechanism back in. Resolution: the region tier with a
doubling class family (§5 step 1), which removes every one of those calls
(promotion is a copy between preallocated slots). **Generalised as principle
2's corollary: anything outliving a wave is arena-managed, never
buffer-allocated.**

**A11 — Sampling holds no device state; logits are wave-scoped.**
`BatchedSampler` (`batched_sampler.rs:362-377`) carries only `device`,
`vocab_size`, `max_recent_len`, a host `TokenBuffer`, and a log path — no
device buffers. Logits arrive as a forward-produced `Tensor` and are consumed
in the same wave (`flatten_to_2d` → `index_select` → `sample_full_vocab`), so
they belong on the bump side like any other intermediate. The only persistent
sampler-adjacent device memory is `KvSamplerGpu`'s grow-only scratch → static
shelf. Note `index_select` allocates its output through candle, i.e. the
step-3 baseline-(a) case: interior op outputs stay on the pool remnant unless
`WaveAllocScope` lands.

**A12 — Small-end granularity has a break-even, not just a measurement.**
Splitting the 320 class into {64, 160, 320} saves 256 B/slot on Q0/Q0_V/Q0_X
and 160 B/slot on Q0_M2/Q1_S (Q1_A, Q0_M4, Q2_S, Q2_A unchanged), at the cost
of two extra classes whose steady-state partial tails run ≈ ½ region each
≈ 16 MiB total. Break-even is therefore ≈ 16 MiB ÷ ~200 B ≈ **65–84 K live
slots** in sub-320 formats — roughly **2 % of a ~4.8 M-slot pool** on this
card. **Decision rule: split the low end only if sub-320 formats exceed ~2 %
of live slots.** Expected to fail at C4/C5 (production default, which never
selects them) and to be worth re-checking at C9/C10. The instrumentation is
nearly free: `compression_bpe` (`backing.rs:1874-1892`) already walks every
`(head, palette, K/V)` slot — add a per-format tally to that walk.

**A9 — `arena_bytes_per_chunk`'s `CHUNK_SIZE²` coupling dissolves.** Today
chunk bytes are computed as `CHUNK_SIZE × CHUNK_SIZE` elements
(`types.rs:22-32`) while arenas are *shaped*
`(chunks, CHUNK_SIZE, sub_head_dim)` — equal only because
`head_dim / N_PALETTE = 32 = CHUNK_SIZE`, so any other `head_dim` silently
mis-sizes every arena. Size classes remove the hazard structurally rather
than papering it: region capacity becomes `REGION_BYTES / class` (no
geometry at all), and geometry enters only when *selecting* a class for a
format, where `sub_head_dim` is passed explicitly. Keep an assert that
`sub_head_dim × CHUNK_SIZE` matches the class-selection input.

**A14 — The selection-gid validator survives classes. FINDING: it was absent
from the step-1 inventory.** `validate_selection_gids`
(`backing.rs:1499-1541`) runs before every selection table upload and rejects
two host-state corruptions the kernel cannot detect: a gid whose arena is
**absent** from storage (freed under a live gid → zeroed row → near-null
deref), and a gid whose `chunk_idx` exceeds its arena's per-format capacity
(the arena index re-tenanted under a live gid, so `old_chunk_idx × new_stride`
walks past the slab — the code comment records this as sanitizer-confirmed at
exactly slab end). Both failure modes get *rarer* under size classes and
neither disappears: a region returned to the free list and re-stamped with a
different class has a different stride, which is bit-for-bit the same hazard.
The validator is therefore **ported** (bound becomes `chunks_for_class`), not
deleted with the per-format machinery around it. Recorded because it is
exactly the shape of thing a deletion sweep removes by association — it reads
like format bookkeeping and is actually a memory-safety net.

**A15 — `arena_gid_stride()` is on the refcount hot path.** Not a defect in
the design, a correction to how the fixed-stride change was *valued*. The
function iterates all 22 `QuantFormat` variants (via `strum`) plus 3 float
dtypes, computing a division per variant, and is called from
`ChunkGid::clone` / `drop` / `arena_idx` / `chunk_idx` / `strong_count` — the
five operations every COW share, every window drop, and every gid walk go
through. It is not `const fn`; the fold is plausible under LTO but unproven.
`GID_STRIDE = 1 << 16` removes the question. Watch for this in the step-1 gate
numbers: if the fold was *not* happening, the win shows up as a broad
host-side drop, not a serialize-path one.

**Answered: the fold was happening.** The swap landed with the u16 clamp in
`1.5b(ii-a)` and measured **115.70 s against a 119.37 s baseline, every config
at or above baseline** — i.e. no win at all. LLVM was folding the `strum`
iterator. The change stands, because it is a precondition for the class switch
and it removes an unproven assumption, but it must **not** be carried as a
performance improvement. This is the one place A-series pricing was wrong in
the optimistic direction; §5 step 1's "two numbers to record" is updated
accordingly.

### 8.2 Experiments (throwaway; reverted)


**E1 — Size-class harness driven by the real `QuantFormat` table.** Three
assertions, all passing: full coverage (22 quant formats + 4 float dtypes,
none uncovered), every class ≤ 65,535 chunks/region, and `GID_STRIDE`
headroom. **Two corrections to this document**: the u16 hazard list has
**seven** formats, not six — `Q0_M4` sits exactly at 65,536 and was missing
(§2.1) — and `F8E4M3` rounds at **11.1 %**, not 12.5 %. Max chunks/region
across the ladder is 52,428, leaving 13,108 of stride headroom. Also
surfaced the honest small-end cost (Q0 at 90 %), now recorded in §3.4.
Promoted to three permanent tests in step 1.

**E2 — `PagedKvArenas` deletion, for real.** Removing the trait produced
**exactly one** error (its own `impl`); removing that too left `candle-nn`
lib *and* tests compiling clean. Confirms the "test-only consumers, delete
don't port" claim empirically rather than by grep. Reverted.

**E3 — `Backing::Lease` primitives.** cudarc 0.17.3 provides
`CudaStream::upgrade_device_ptr<T>(ptr, len) -> CudaSlice<T>` (unsafe) and
`CudaSlice::leak() -> CUdeviceptr`. The lease is therefore
**upgrade-on-construct + leak-on-drop**, with no vendored-cudarc change.
cudarc's stated contract ("memory may not be valid for `T`; memset it") is
already satisfied by zero-on-recycle (invariant 4). Closes open question 5.

*Corrected by E4*: E3 also claimed "no `ManuallyDrop` hack", which was wrong
in both directions. `CudaSlice::leak` takes `self` **by value** and ends in
`mem::forget(self)`, so it cannot be called from `Drop::drop(&mut self)` at
all — *something* must move the slice out. But `ManuallyDrop` is not that
something: a tombstone variant plus `mem::replace` is smaller and leaves
owned storage's drop semantics untouched. E3 named the primitives correctly
and then guessed at the assembly; E4 built it.

**E4 — `Backing::Lease`, built end-to-end and reverted.** Three phases, each
compiled separately so the cost of each is attributable:

| Phase | Question | Result |
|---|---|---|
| 1 — add `CudaStorageSlice::Empty` | how many exhaustive matches break? | **18**, one round, no nested reveals; `cuda_backend/{mod,utils}.rs` only |
| 2 — add `backing: Backing` | how many struct literals break? | **35** candle-core + **8** candle-nn; **only** E0063, nothing else |
| 3 — add `Drop for CudaStorage` | does anything move fields out? | **zero errors** — nothing destructures a `CudaStorage` |

`cargo check --workspace --features cuda --tests`: clean. Total **115
insertions / 30 deletions across 6 files**. Reverted.

Phase 3 was the real risk — implementing `Drop` forbids moving fields out of
a type, and `CudaStorage`'s two fields are `pub`. Nothing does. Had anything
destructured it, the tombstone route would have died and the
`ManuallyDrop`-per-variant route (**142** `ManuallyDrop::new` sites, measured
separately) would have been forced. It is worth knowing that the fallback
exists and what it costs, because nothing else in the design has a
three-times-larger plan B hiding behind a one-line assumption.

**E5 — selection-table sizing. FINDING: the fused table is 48 identical
copies today.** `ChunkedKvBacking::new_layer` does `inner:
Arc::clone(&self.inner)` (`backing.rs:908`), so every layer of a group shares
**one** `BackingInner` and therefore one arena set. But
`from_head_gids_multi` calls `backing.per_head_table_host()` once per layer
and concatenates the results, relabelling each by `arena_offset`
(`gpu.rs:965-980`). All 48 calls return byte-identical data. At `PerHeadEntry`
= 288 B/row and the §9 geometry (48 layers × 4 KV heads):

```
TODAY  rows = n_layers × num_arenas × n_kv_head      ← the 48 copies
   336 arenas  →  64,512 rows = 17.72 MiB per fused selection

AFTER  rows = total_chunks × n_kv_head
   1,536 chunks (a ~1000-token turn) →  6,144 rows =  1.69 MiB
   crossover at 336 arenas: 16,128 chunks = 10,752 tokens/layer
```

The ratio is workload-dependent; the **exponent** is not, and that is the
point. Today the table scales with *arena count* — it grows with exactly the
fragmentation this initiative exists to delete, so the drain gets more
expensive as the pool gets sicker. After the re-index it scales with *chunks
actually being selected*, i.e. with work. **Record the fused-table bytes in
the step-1 gate**; expect a fall at typical drain widths and confirm the
crossover only bites above ~10 K tokens/layer.

## 9. Verification status

Every structural claim above was checked against the code on 2026-08-07;
platform claims were probed on the target machine (RTX 4090 Mobile, driver
596.08, WDDM) with the daemon stopped. Probe scripts live in the session
scratchpad (`vmm_probe.py`, `vmm_release_probe.py`, `vmm_overcommit_probe.py`).

**Proven against code**

| Claim | Evidence |
|---|---|
| `GIDS_PER_HEAD = 8`, `N_PALETTE = 4` | `head_gids.rs:19`, `arena_table.rs:359` |
| Sub-entry carries ptr/offset/stride/format/scale per (head, palette, side) | `arena_table.rs:367-405`; CUDA twin `arena_table.cuh` (`static_assert(sizeof(PerHeadTableEntry) == 72)`) |
| Attention formats are per-chunk, per-band | `slot_state.rs:84-106` (`k_fmt: [u8; N_PALETTE]`) |
| Selection formats come from the arena row | `select_kv_format.cuh:1461-1463`, `backing.rs:1549-1655` |
| Every class-ladder byte size | `blocks.cuh` static asserts (q4_ks 20, q8_ks 36, q2_0 10, q3_0 14, q0 1, q0_v 2, q0_x 2, q0_m2 3, q0_m4 8, q1_s 5, q1_a 6, q2_s 9, q2_a 10, q2_1 12, q3_1 16, r16 128, f16 64/32elem, f32 128/32elem) + `k_quants.rs` const asserts (q4_0 18, q4_1 20, q5_0 22, q5_1 24, q8_0 34, q8_1 36) |
| u16 recycle-link overflow hazard (§2.1) | `gid_pool.rs:89-102` (overlapped refcount/link word) + the verified block sizes |
| Per-chunk format tags already persisted | `transfer.rs:221` (`to_tag`), `pipeline.rs:529-542` (`from_tag`) |
| Cold-load carries per-band formats | `BlockAllocSpec.k_formats/v_formats`, `chunk_ops.rs:183-193` |
| `Tensor` is an `Arc` newtype (so a lease travels with views/reshapes) | `tensor.rs:68` `pub struct Tensor(Arc<Tensor_>)` |
| cudarc pool is CUDA's stream-ordered mempool | `cuda_backend/device.rs:358-393` (`cuMemPool*`) |
| Gallery = 16 MiB slabs ÷ 6144 B pages | `gallery_arena/pool.rs:18`, `pages.rs:13-20` (`PAGE_TOKENS × wpt(24) × 8`) |
| Exactly one production relief closure — the gallery, under `AllocClass::Kv`; `Expert` gets a tally only | `scheduler/mod.rs:2363` vs `quantized_qwen3_moe.rs:1922-1926` (`set_class`) |
| All 30 symbols in step 5's deletion inventory exist | grep sweep across the four crates |
| 18 relief call sites in `prefill.rs`, 1 unforced in `run.rs:484` | grep count |
| Attention reads per-band ptr/fmt/scale with no arena involvement | `slot_types.cuh:187-221` (`kvhead_k_ptr/k_fmt/k_scale` index by palette `p`) |
| Format ⇒ arena is created in exactly one closure | `compress.rs:728-738` (`alloc_side` → `ArenaKey::gpu_quant(fmt)`), run per `(chunk, head, side)` |
| Seven payload/stride consumers, spanning three crates | `chunk_ops.rs:1987/2206/2755`, `head_gids.rs:156-186`, `migrate.rs:199/234`, `transfer.rs:269` |
| `seal_to_chunk_images_cpu` reserves by stride, splits by `byte_size` | `transfer.rs:269-289` (`blob.resize(start + stride)` vs `let n = sc.byte_size`) |
| The selection-gid validator bounds `chunk_idx` per format (A14) | `backing.rs:1523-1534` (`arena_chunks_for_format(arena.format())`) |
| `arena_gid_stride()` is called from all five `ChunkGid` refcount ops (A15) | `gid_pool.rs:462`, `:495`, `:500`, `:517`, `:539`; definition iterates `QuantFormat::iter()` at `types.rs:50-68` |
| `alloc_side` already prefers a contiguous `N_PALETTE` run per uniform group | `compress.rs:729-730` — classes are coarser than formats, so run eligibility *rises* |
| **All layers of a group share one `BackingInner`** (⇒ one arena set) | `backing.rs:908` `inner: Arc::clone(&self.inner)` in `new_layer`; load-bearing for E5 and for `per_head_table_host`'s dense-over-storage sizing |
| Nothing destructures a `CudaStorage`, so `Drop` may be added | E4 phase 3: zero errors workspace-wide with `impl Drop for CudaStorage` present |
| `CudaSlice::leak` is by-value and `mem::forget`s | cudarc 0.17.3 `core.rs:1915`; waits on read/write events, destroys them, decrements the stream `Arc` |

**Proven by probe**

| Claim | Result |
|---|---|
| CUDA VMM available; one VA span, granule-mapped, contiguous | attr 102 = 1, min granularity 2 MiB; write straddling a granule boundary verified |
| Single giant allocation fallback | `cuMemAlloc(14 GiB)` in 15.6 ms, full touch 38.5 GiB/s, desktop resident |
| Partial release then reuse (the §3.2 startup sequence) | released 8/64 granules → `cuMemAlloc(1920 MiB)` OK into the freed space → surviving 14,336 MiB read back intact |
| Mapped granules are truly resident, not host-paged | re-touch of granule 0 at a full span: **528 GiB/s** |
| **`cuMemCreate` never refuses** — corrected §3.2 | succeeded across a 32 GiB VA span on a 16 GiB card; the limit appears at first *touch* (`invalid argument` at 15,360 MiB) |

**Assumed, not re-verified here** (carried from the 2026-08-06 measurement run
and prior sessions; re-confirm in the baseline gate run): the 2142 MiB slack
breakdown of §1.3, the 666–1005 MiB prefill-scratch peak, and the 2847 expert
slots / 46.3 % residency baseline. The §3.10 turn arithmetic ("≈49 K quant
slots ≈ 3.4 regions") assumes the Qwen3-30B-A3B geometry of 48 layers ×
4 KV heads × head_dim 128; it is derived, not measured.

**Known latent coupling** (worth an assert during step 1):
`arena_bytes_per_chunk` sizes a chunk as `CHUNK_SIZE × CHUNK_SIZE` elements
(`types.rs:22-32`), while the arena is *shaped*
`(arena_chunks, CHUNK_SIZE, sub_head_dim)`. These agree only because
`head_dim / N_PALETTE = 128 / 4 = 32 = CHUNK_SIZE`. Any model with a
different `head_dim` silently mis-sizes every class. The class table should
derive from `sub_head_dim` explicitly and assert the identity.

## 10. Open questions

**None.** The three that survived the §8 sweep were closed by audits A10-A12:

1. ~~Logits/sampling buffer placement~~ -> **bump side** (A11). Sampling holds
   no device state; logits die with the wave. `KvSamplerGpu` scratch -> static
   shelf.
2. ~~Does any buffer outlive its wave?~~ -> **yes, `GpuChunks`** (A10), which
   therefore goes to the **region tier** under principle 2's corollary, not
   the pool remnant (A13 — leaving it there would have preserved ~3,000
   alloc/free/sync cycles per 32 decoded tokens). Because it is host-built
   metadata rather than a kernel output, the **paged-output contingency stays
   unbuilt**; `PAGE_BYTES` remains a reserved constant with no implementation
   behind it.
3. ~~Small-end class granularity~~ -> **a decision rule, not a question**
   (A12): split the low end only if sub-320 formats exceed ~2 % of live slots.
   Instrument by adding a per-format tally to `compression_bpe`'s existing slot
   walk; evaluate at step 6 against C9/C10.

One item is deferred *by scope*, not unresolved: the host-side lease for
`CpuStorage`'s `Vec` travels with the warm-tier reservation (S7, follow-on) -
the same upgrade/leak question with no cudarc equivalent, likely
`ManuallyDrop` over `Vec::from_raw_parts`.

Everything the design needs to decide is decided. What remains is measurement,
and every measurement has a named hook and a threshold.

*Resolved by §7 (scope)*: gallery/meta-pool fold-in (S5 — out of scope),
size-class count (S3 — seven, full quant coverage retained), buffer-plan
formalism (S4 — none), tensor lifetimes (S1 — none), warm-tier reservation
(S7 — follow-on).

*Resolved by §8 (audits/experiments)*: trim-vs-stride on migration copies
(A1 — trim, via invariant 8); region-tier locking (A5 — no new lock);
kernel pad reads (A6 — safe); second-gate sensitivity (A7 — blind, so step 1
adds a direct `byte_size` assertion); `CHUNK_SIZE²` coupling (A9 — dissolves
under classes); `ArenaEntry` per-arena format tags (Q1 — only readers are the
three sites being rewritten, so they are deleted in step 1);
`read_contiguous`/`write_contiguous` reachability (Q4 — **production**, via
`batched_inference.rs:2749/2772` and `prefill_utils.rs:2557`, so
`Backing::Lease` stays in step 1); gallery relief replacement (call
`evict_lru` ahead of KV evacuation); `Backing::Lease` mechanics
(E3 — `upgrade_device_ptr` + `leak`).
