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
- **Fallback, also probed**: a single giant `cuMemAlloc` (14 GiB succeeded in
  15.6 ms with the desktop resident; full touch at 38.5 GiB/s). Same
  single-span design, coarser (whole-buffer) eviction unit. A runtime
  capability probe picks the path at startup. Superslabs are never needed.
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
- **Startup sequence** (probed end-to-end, `vmm_release_probe.py`): reserve
  the VA span → map + **touch** granules under the policy cap, stopping on a
  touch failure (this both measures `C` and claims it) → **release** the
  granules the partition assigns to dense weights, the expert cache, and the
  small reserve → those load through the CUDA pool into the freed memory →
  the granules still mapped are the reservation. Probe result: after
  releasing 8 of 64 granules, an ordinary `cuMemAlloc(1920 MiB)` succeeded
  into the freed space **and the surviving 14,336 MiB of mapped reservation
  read back intact**. The measurement and the claim are one act; there is no
  release-and-hope-to-reclaim window. (On the giant-`cuMemAlloc` fallback,
  partial release is impossible, so the classic order applies — measure,
  release, load weights, then claim the remainder — accepting a brief claim
  race.)

### 3.3 Region tier

- `region[i]` base = `reservation_base + i × 16 MiB` (a clean multiple of the
  2 MiB granularity). `TARGET_ARENA_BYTES` stays 16 MiB.
- **The free-region list already exists — no new lock.**
  `GidPoolState.free_arenas: VecDeque<usize>` under the existing `metadata`
  mutex *is* the free-region list today: `register_arena` pops it,
  `next_tombstone` pushes, `drain_free_arenas_above` trims it. The region
  tier reuses it verbatim, so the documented lock order
  (`alloc_gate → tables`, `metadata` outside both) gains **no new edge**
  (audit A5, §8). A region is **stamped with a size class** when popped and
  returns to the list when its live count reaches zero (the existing
  `creation_pending` / `live == 0` tombstone logic, minus the storage release
  and the CUDA free).
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
  allocate while wave `N`'s kernels drain; the half being reset fences on the
  completing wave's stream event (the `PinnedStager` sync-then-reset
  discipline).
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
  MoE routing buffers) sit on the **static shelf** at the reservation's
  absolute right tip — the only address the boundary never moves — outside
  every reset. The span requirement is the sum of the domain budgets:
  `S = 2·W_wave + W_persist + shelf`, where each `W` is that domain's
  watermark (max plan/staging bytes over a recent window + margin).
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
  incremented at construction, decremented on storage drop. Two consumers
  only: the legacy contiguous `KvCache` façade
  (`read_contiguous`/`write_contiguous`, which need on-demand tensor views
  over region bytes) and, from step 3, wave intermediates that candle ops
  must consume. Views/reshapes share the `Arc<Storage>` and the lease travels
  with them, so the inference loop keeps its Tensor-based code.

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
  the expert cache, then freeze the partition.
- Pressure = `free_regions < setpoint`. Responses, in order:
  1. steal an empty region from any class (O(1)),
  2. **evict-as-evacuation**: demote the **rightmost** occupied regions'
     sealed chunks to warm via the existing
     `migrate_sealed_layers_to_cpu_batch` + install path (the hot tier is a
     cache; demotion is its defining operation — this replaces GPU→GPU
     defragmentation entirely). With lowest-first packing, rightmost ≈
     emptiest, so one *positional* eviction order serves both cross-class
     reclaim and boundary movement (§3.6),
  3. throttle admission (the existing regulated setpoint, now driven by an
     exact, latency-free counter instead of driver headroom).
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
8. **Payload bytes ≠ address stride.** `ResolvedArenaInfo` carries *two*
   lengths: `chunk_byte_stride` (the class stride — address arithmetic only,
   `base + idx × stride`) and `chunk_payload_bytes` (the format's real bytes
   — every copy length, `arena_byte_size`, and the persist blob). Zeroing
   uses the **stride** (invariant 4: the next tenant may be any format);
   copying and persisting use the **payload**. Conflating them makes every
   hot→warm migration move pad over PCIe and silently changes on-disk image
   sizes and their Fletcher goldens. **Seven** sites consume the length as a
   copy/extent rather than an address step, and they span three crates —
   enumerated in §5 step 1; A1 found the class, and the sweep that followed
   found the rest.

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
  on-demand tensor views over region bytes. No lifetime parameterization; no
  other candle-core change.
- Add per-band format tags to `ChunkWindow` and `SealedChunk` (parallel to
  `gids`; `Arc`-shared per block like pal/scale). Invert every
  format-from-arena derivation — the verified site list:
  `KvHeadHost::from_gids` (`slot_state.rs:147-153`, `arena_info[..].k_format_tag`
  → chunk metadata) and `build_meta_records`; `per_head_table_host`
  (`backing.rs:1549-1655`, both `k_tag`/`v_tag` and the `arena.format()`-derived
  `chunk_byte_stride`); `kv_formats_for_gids` (`chunk_ops.rs:1630`); migration
  source resolve; `ensure_writable_tail`'s any-quantized check
  (`sequence_ops.rs:1791`); `bucket_quant_chunks` eligibility
  (`compress.rs:409-423`); `gpu_format_stats` (`gid_pool.rs:1588`);
  `PalHeadDesc` build in `compress.rs`.
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
- Introduce `SizeClass` and the ladder table (§3.4); `ArenaKey` →
  `(SizeClass, ArenaLocation)`; `arena_chunks_for_format` →
  `chunks_for_class`; `arena_gid_stride()` → the fixed 65,536; scarcity-only
  class promotion in the allocator.

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

  | # | Site | Role of the length |
  |---|---|---|
  | 1 | `chunk_ops.rs:1987` | DtoH gather copy length |
  | 2 | `chunk_ops.rs:2206` | cross-layer gather copy length |
  | 3 | `chunk_ops.rs:2755` | HtoD scatter copy length |
  | 4 | `head_gids.rs:156-186` | `arena_byte_size` → `SealedChunk.byte_size` |
  | 5 | `migrate.rs:199` | `resolve_sealed_chunk_ptrs` → `(ptr, len)` pairs |
  | 6 | `migrate.rs:234` | `resolve_sealed_chunk_ptrs_per_gid`, same |
  | 7 | `transfer.rs:269` | `seal_to_chunk_images_cpu` blob slot reservation |

  Sites 5–7 are the ones the first pass missed, and **7 is the dangerous
  one**. `seal_to_chunk_images_cpu` appends a **stride**-sized slot to the
  blob per unique gid, then splits that blob by `sc.byte_size`. Today the two
  agree by construction. The moment `byte_size` becomes payload-summed while
  the gather still reserves stride, the blob and the split disagree — the
  visible outcome is `seal_to_chunk_images_cpu: blob underrun`, the invisible
  one is every chunk after the first landing on shifted boundaries in
  persisted data. Site 5 feeds `seal_to_chunk_images_gpu`'s `split_sizes` the
  same way. **All seven must move in one commit**; a partial conversion is
  worse than none, because sites that still agree mask the ones that don't.
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

- Profile the allocation fast path (claim, run claim, region pop), the
  transient generation reset, and eviction cadence under the widest gate
  config.
- **Boundary motion, only if earned** (§9 S6): if the fixed boundary from
  step 4 measurably strands memory (transient span idle while KV starves, or
  vice versa), add the hysteretic watermark regulator and rightmost-first
  boundary evacuation. If the fixed split holds, this is never built.
- Evaluate `WaveAllocScope` (step 3's deferred option (b)) — adopt if the
  leak counter stays clean and it measurably shrinks pool-remnant traffic.
- Tune the class ladder against the observed histogram; tune the free-region
  setpoint; verify the stride-65,536 decode shows up as a win in the
  serialize paths.

*Gate*: the test's tokens/s must meet or beat baseline on every config;
record the final table.

### Step 7 — Switch to the daemon; tune the partition

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
| Cursor reset races an in-flight kernel of the prior wave | A/B halves + event-fenced reset (the `PinnedStager` sync-then-reset discipline, ported to the device instance). |
| VMM path fails on a future driver/platform | Fallback probed and working: a single giant `cuMemAlloc` (14 GiB verified on the target machine) — same single-span design, coarser (whole-buffer) WDDM eviction unit. Runtime capability probe picks the path at startup. |
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

Fifteen audits and three throwaway experiments run before implementation.
Four findings changed the design (A1 + its amendment, A10/A13, A14); one added
a deletion (A4); one re-priced a change (A15); the rest confirmed it.

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
`BF16`, `Q8_0`, `Q4_0` **and adaptive `C0–C7`, `C9`, `C10`** (only `C8` is
absent) at 1–48 contexts, and drives `record_turn` → `quantize_sealed_in_place`
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

**E3 — `Backing::Lease` mechanics.** cudarc 0.17.3 provides
`CudaStream::upgrade_device_ptr<T>(ptr, len) -> CudaSlice<T>` (unsafe) and
`CudaSlice::leak() -> CUdeviceptr`. The lease is therefore
**upgrade-on-construct + leak-on-drop** — no vendored-cudarc change and no
`ManuallyDrop` hack. cudarc's stated contract ("memory may not be valid for
`T`; memset it") is already satisfied by zero-on-recycle (invariant 4).
Closes open question 5.

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
