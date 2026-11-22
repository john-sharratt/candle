# VRAM Governor — Design Document

**Status:** The startup half is built (`candle-core/src/vram/`): the
measurement probes (§6), the balloon-and-measure bootstrap that fixes capacity
`C` (§5), the `AllocClass` tags with their loose per-class tallies and the
`BudgetTable` diagnostics (§10), and the capacity arithmetic the span is sized
from (`VramGovernor::usable`). **The runtime half is removed.** The criticality
ladder with its registered relief closures (`Criticality`, `register_relief`,
`relief.rs`, §4 and §8), the variable KV band and `kv_floor` (§7), the managed
`reserve`/`allocate` retry and `forecast_units` (§9), and `expert_budget()` (§11)
went when KV moved onto a static reservation: the memory is claimed once and the
scheduler reads an exact free-region count (`docs/vram_span_partition.md`;
`docs/archived/arena_unification.md` §3.8, §5). Those sections are kept as the
design record of the removed machinery. §8.1 (idle demote) and §8.2 (splice
sources) describe code that stands, now driven by the scheduler.
**Scope:** A single, cross-platform module that owns GPU VRAM residency for the
`zend` inference engine. It **measures real free VRAM** and lets the budget
**evolve as allocations happen**, classifying each allocation by type so it knows
what is evictable, and relieving pressure through a **criticality-ranked,
threshold-triggered callback ladder**. Replaces the current split-brain budgeting
(`init_free − pool_used` in the KV path + an independent one-shot expert-slot
formula).

**Not** Blackwell-specific and **not** Windows-specific in its logic. Windows/WDDM
and Linux are both first-class; only the measurement backend differs.

---

## 1. Core principle (read this first)

Everything flows from one decision: **the real free-VRAM measurement is the single
source of truth. We do not do byte accounting; we do not predict footprints.**

- We **cannot** reliably predict what the model, experts, scratch, and KV will use
  — but we **know it is all allocated before inference begins**. So we don't need
  the numbers up front: the budget **evolves** as allocations land, and by the
  time inference starts the measurement already reflects reality.
- Because the gate is a live measurement (not a running `ceiling − Σcommitted`
  tally), it **cannot drift out of sync** the way a virtual accounting scheme
  does. This is the resilience win: no real-vs-virtual reconciliation, no
  fragmentation blind spot, no async-free lag corrupting the counter.
- Allocations still carry a **type tag** (`Weights | Expert | Scratch | Kv`) — not
  to sum bytes for a gate, but to know **what is evictable**, to **forecast**
  concurrency, and to **report** a budget table. Classification, not accounting.

The governor's job is therefore: measure honestly, keep headroom above a laddered
set of thresholds by relieving pressure cheapest-first, and tell the scheduler how
much room it really has (including what eviction can reversibly reclaim).

---

## 2. Goals & non-goals

### Goals
- **Measure physically-resident free VRAM** on WDDM and Linux — not the driver's
  polluted `free`.
- A **fast balloon-and-measure** bootstrap that clears squatters and fixes the
  true capacity `C` — fast in the *general* case (strided touch), no
  platform-specific fast path required.
- A budget that **evolves with allocation**, driven purely by measurement; each
  allocation carries an **`AllocClass`** so the governor knows evictable vs fixed.
- A **variable band** for KV with a **floor** = absolute + percentage (≈ 3 GiB +
  %), defended against external VRAM theft without paging.
- A **threshold ladder**: cheap/non-destructive relief kicks in **early**;
  hit-rate-damaging KV eviction is held until **really needed**. Relief is **async
  (no per-alloc/free sync)**; a **GPU-level (stream) sync** happens **only high up
  the ladder**.
- A **forecast** function so the concurrency loops (prefill parallelism) size
  themselves to `measured_free + reversibly-evictable KV`.
- **Diagnostics**: structured event logging + a **budget table dump** invokable
  from unit tests.
- **Minimal dependencies** — reuse what's already in the tree; add the smallest
  possible platform shim.
- **Extensive unit tests** with a mockable probe + allocator.

### Non-goals
- **No byte accounting as the availability gate.** Per-type tallies exist only for
  reporting/forecasting/evictability.
- **No footprint prediction** — not for scratch, not for the model. Scratch is
  "whatever it needs"; it must simply be allowed to allocate.
- **No device-memory pinning** — CUDA offers none (host-pinned only). We design
  *for* eviction.
- Not a general allocator; tensors still allocate through cudarc's pool.

---

## 3. Background — what exists today (condensed from code study)

**Citations in this section are the original design record.** The `file:line`
positions are those of the code when the design was written and mostly no longer
resolve. The functions `mem_get_info`, `pool_used_bytes`, `pool_reserved_bytes` and
`trim_pool` stand in `candle-core/src/cuda_backend/device/device_info.rs`;
`release_empty_arenas` is in `candle-nn/src/kv_cache/chunked/backing/`; `evict_hot_to_free`,
`evict_hot_except` and `demote_turns_to_warm` are in
`candle-conversation/src/substrate/eviction.rs`; `quantize_sealed_in_place` is in
`candle-nn/src/kv_cache/chunked/compress/mod.rs`; `relieve_vram_pressure` is in
`candle-conversation/src/scheduler/prefill/relief.rs`; `DEVICE_INIT_FREE` is in
`candle-core/src/gpu_memory.rs`. `request_global_compact`, `compact_forced`,
`EvictionScope` and `purge_warm_to_target` no longer exist under those names: the
compaction pass is `compact/pass.rs` (`docs/vram_span_partition.md` §6), and the
quantize is the gap's (`docs/kv_quantize_in_the_gap.md`).

### 3.1 Single allocation chokepoint
Every device allocation funnels through `CudaDevice::alloc`/`alloc_zeros`
([`device/memory.rs:46-58`](../candle-core/src/cuda_backend/device/memory.rs)) → cudarc
`cuMemAllocAsync` on the default stream-ordered pool. Free is RAII (drop →
`cuMemFreeAsync`, returns to the **pool**, not the OS). Present measurement
primitives: `mem_get_info` (`device/device_info.rs:318`), `pool_used_bytes`/`pool_reserved_bytes`
(`device/device_info.rs:333/345`), `trim_pool` (`device/device_info.rs:384`), `synchronize` (`device.rs:726`).

### 3.2 Allocation categories (→ the `AllocClass` tags)
| `AllocClass` | Category (Qwen3-30B-A3B) | Evictable? |
|---|---|---|
| `Weights` | mandatory tensors (embeddings, attn q/k/v/o+norms, ln1/ln2, router gate, final norm, lm_head), ~1–2 GB | No — permanent |
| `Expert` | MoE slots `ExpertSlot{gate,up,down: QMatMul}`, pool of `num_slots × max_expert_size`; total ~15–30 GB by quant | Only at `Critical` (slot→pinned), and only if `!all_resident` |
| `Scratch` | forward activations + grow-once pools (`ProvSignScratch`, `KvSamplerGpu`, `meta_pool`, RoPE tables) | No — live/needed |
| `Kv` | 16 MiB arenas, one shared `Arc<BackingInner>` across all 48 layers | **Yes — the flex region** |

### 3.3 The nine KV relief mechanisms (wired into the ladder in §8)
Cheap→expensive / reversible→lossy: `release_empty_arenas` (`backing.rs:873`),
`compact` (`backing.rs:856`), `compact_forced` (`backing.rs:863`),
`evict_hot_to_free` (`substrate/eviction.rs:1716`, oldest hot→warm, reversible),
`evict_hot_except` (`substrate/eviction.rs:1646`), `demote_turns_to_warm` (`substrate/eviction.rs:1788`),
`quantize_sealed_in_place` (`compress/mod.rs:157`, **lossy**), warm→cold
(`thread.rs:570`), `purge_warm_to_target` (`substrate.rs:1553`). The existing
cascade `relieve_vram_pressure` (`scheduler/prefill/relief.rs:74-128`) returns "still pressured?" as
a bool; `request_global_compact` (`backing.rs:49-84`) broadcasts to all backings;
`EvictionScope` (`alloc.rs:93-124`) grants a compress-to-free op a privileged
slice. These are the closures the ladder registers — subsumed, not rewritten.

### 3.4 Measurement surface (from API research)
- **Windows/WDDM (primary):** `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` →
  per-process `Budget`/`CurrentUsage`; `Budget − CurrentUsage` = real headroom
  before the OS pages *us*. `RegisterVideoMemoryBudgetChangeNotificationEvent` →
  push re-measure. `SetVideoMemoryReservation` = minimum-working-set hint (not a
  lock). Adapter matched to CUDA device by **LUID** (`cuDeviceGetLuid` ↔
  `DXGI_ADAPTER_DESC1.AdapterLuid`).
- **TCC and Linux:** `cuMemGetInfo` is accurate (no WDDM virtualization). The
  driver model is read from the device (`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`), never
  inferred from the OS — `docs/driver_model.md`.
- **Balloon:** `cuMemAlloc` + `cuMemsetD8` (touch) forces residency and evicts
  other processes' cold VRAM. No CUDA lock exists.
- **Pool return:** `cuMemPoolTrimTo` reclaims *settled* memory without a sync but
  can't reclaim outstanding async-frees until their stream op retires. → reinforces
  "measure, don't account, and only sync high on the ladder."
- **NVML:** board-wide only, per-process N/A on WDDM. **Dropped** from rev 2 to
  minimize deps (see §6, §14).

---

## 4. Architecture overview

```
                         ┌──────────────────────────────────────────────┐
   register_relief() ───▶│              VramGovernor (per GPU)            │
   (KV / experts, by tier)│  ─ measure() → real headroom (SOURCE OF TRUTH)│
                          │  ─ evolving budget: classify by AllocClass    │
   allocate(class,…) ────▶│  ─ threshold ladder → escalating relief (§8)  │
   reserve(class,…)       │  ─ forecast(per_seq) for concurrency (§9)      │
                          │  ─ budget_table() / log events (§10)          │
   forecast_prefill() ───▶│                                              │
                          └───────────────┬──────────────────────────────┘
                                          │
                     ┌────────────────────▼─────────────────────┐
                     │  VramProbe:  DxgiProbe(win) | CudaProbe    │
                     │  Balloon:    claim + strided-touch + measure│
                     └───────────────────────────────────────────┘
```

One `VramGovernor` per physical GPU, stored process-globally keyed by ordinal
(mirroring `DEVICE_INIT_FREE`, `gpu_memory.rs:38`), reachable from candle-nn (KV)
and candle-transformers (experts) without threading a handle everywhere.

```rust
/// What an allocation is FOR — drives evictability, forecasting, reporting.
/// NOT summed as an availability gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllocClass { Weights, Expert, Scratch, Kv }

/// Ascending cost / descending reversibility. Each tier has a headroom THRESHOLD
/// (§8); relief runs low→high, and only `Critical` takes a GPU sync.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Criticality {
    Trivial  = 0, // release empty arenas, trim settled pool — no data loss, early
    Cheap    = 1, // lossless compact/defrag — early
    Moderate = 2, // compress-to-free COMPLETED turns early — net shrink, stays
                  //   resident, no extra loss (they quantize on seal anyway)
    Costly   = 3, // reversible hot→warm evict — MOVES off-card, reload if re-attended
    Critical = 4, // GPU sync + remeasure + drop-to-cold / expert-pool shrink
}
```

---

## 5. Balloon-and-measure bootstrap

**Purpose:** clear transient squatters and fix the true resident capacity `C`
(what we can actually hold on *this* machine now), which anchors the variable
band and the KV floor. `C` is not `cuMemGetInfo total`.

**Algorithm:**
1. Grow a chain of chunks (`balloon_chunk`, 256 MiB, `cuMemAlloc`), touching each
   with a memset (`memset_d8_async`) so residency is forced and others' cold pages
   are evicted. Stop when **any** of these holds: reserved reaches
   `capacity_target = total − capacity_reserve` (`capacity_reserve`, 256 MiB, the one
   number the card holds back); the probe's live headroom falls to the **wobble
   margin** (on a DXGI reading, a `WOBBLE_DIVISOR`th (32) of the headroom, capped at
   `MAX_WOBBLE_MARGIN` (512 MiB) and floored at `capacity_reserve`; under TCC and
   Linux just `capacity_reserve`); or the driver refuses a chunk already as small as
   `balloon_min_chunk` (2 MiB). A refusal halves the chunk and asks again.
2. `C = reserved_high_water`.
3. **Free the balloon.**

The post-balloon `SetVideoMemoryReservation` of the first design is not built.

**Fast in the general case (requirement):** the same balloon runs everywhere (on
Linux the touch simply forces allocation, there's nothing to evict). The
tunables are `GovernorConfig`'s `capacity_reserve`, `balloon_chunk`,
`balloon_min_chunk` and `scratch_margin`.

**Runs before model load**, on the emptiest card. **Circuit breaker:** if it can't
reach a sane floor, log loudly and fall back to `C = cuMemGetInfo total − margin`
(retain `device_init_free` as this fallback). Never hang.

---

## 6. Measurement layer

```rust
pub struct VramReading {
    pub headroom: u64,   // bytes we may still make resident (WDDM: Budget−Usage; Linux: cuMemGetInfo free)
    pub total:    u64,   // device total
    pub source:   ProbeKind,
}
pub enum ProbeKind { Dxgi, Cuda }

pub trait VramProbe: Send + Sync {
    fn read(&self) -> Result<VramReading>;
    fn budget_change_event(&self) -> Option<BudgetWatch> { None } // Windows push; None elsewhere
}
```

- `DxgiProbe` (`#[cfg(windows)]`) — LUID-matched `IDXGIAdapter3`, LOCAL segment,
  wires the budget-change event. **Primary on WDDM.**
- `CudaProbe` — `cuMemGetInfo`. **The probe under TCC and Linux.** Never a
  fallback for WDDM: a WDDM device whose DXGI probe cannot open gets no governor,
  named in the log (`docs/driver_model.md` §2).

**Dependency posture (minimal):** `CudaProbe` uses cudarc (already a dep).
`DxgiProbe` needs DXGI COM — proposed as the `windows` crate gated to
`#[cfg(windows)]` with only `Win32_Graphics_Dxgi` + `Win32_System_Threading` +
`Win32_Foundation` features (the crate is already present transitively). If we want
to avoid even that, the alternative is hand-rolled vtable calls through
`windows-sys` (lighter, more code). **NVML is dropped** — DXGI (Windows) +
`cuMemGetInfo` (Linux) cover the truth we need; the board-level cross-check isn't
worth a new dependency. (Open question §14.)

The probe is a trait object → tests inject a `FakeProbe` returning a scripted
sequence of `VramReading`s.

---

## 7. The evolving budget & the variable band

*Removed: `kv_floor` and the variable band (the `kv_floor_abs`/`kv_floor_pct`
fields are gone from `GovernorConfig`), and with them the external-theft
relief below. The span is sized once from `C`, and the boundary between the KV
and weight sides moves by concession (`docs/vram_span_partition.md`). The
per-class tallies remain, for the budget table only.*

No partition is computed up front. Instead:

- Each managed allocation is tagged `AllocClass`. The governor keeps a **loose
  per-class running total** — for the budget table and forecast only, explicitly
  **not** an availability gate and allowed to drift (measurement corrects it).
- As `Weights` then `Expert` allocations land, `measure().headroom` falls
  naturally from `C`. Whatever headroom remains once the fixed classes are in
  place **is** the KV budget — observed, not predicted.
- The **KV variable band**:
  ```
  kv_floor = KV_FLOOR_ABS (3 GiB)  +  KV_FLOOR_PCT (15%) × (C − Weights)
  variable.max  = current measured headroom + reversibly-evictable KV
  variable.min  = kv_floor          # never evict KV below this
  ```
  Worked: 16 GiB card (Weights≈2) → 3 + 0.15×14 ≈ **5.1 GiB**; 73 GiB card →
  3 + 0.15×71 ≈ **13.7 GiB**. `kv_floor` being **absolute + percentage** keeps a
  small card viable (the 3 GiB floor dominates) while a big card scales the floor
  up with the percentage. Base = `C − Weights` is known right after weight load and
  avoids the expert/KV circularity.

**External-theft resilience without paging:** the DXGI budget-change event (or the
periodic remeasure on Linux) lowers `headroom`. The governor drives relief to hold
KV inside `[kv_floor, …]`, shedding KV **reversibly** rather than being paged — and
never below `kv_floor`. On Windows the post-balloon `SetVideoMemoryReservation`
(floor = `Weights + Expert + kv_floor`, set once `C` is known) makes the OS trim
*other* apps toward *their* reservations before touching ours.

---

## 8. The relief ladder — thresholds, gentle-early, async, sync-only-high

*Removed: the ladder, its thresholds and `register_relief` are gone
(`candle-core/src/vram/mod.rs` module doc). The scheduler's pressure gate and
relief cascade (`vram_under_pressure`, `relieve_vram_pressure`,
`scheduler/prefill/`) read the KV side's exact free-region count, and relief is the
scheduler's, not a registered closure set. §8.1 and §8.2 below stand.*

Relief is triggered by **where `measure().headroom` sits** relative to a ladder of
thresholds, so **cheap relief starts early** and **KV eviction is withheld until
really needed** (protecting hit-rate). Each threshold is an **absolute + percentage
offset above the floor** — same hybrid shape as `kv_floor`, so the margins stay
sane on a 16 GiB card *and* scale on a 73 GiB card:

```
T_i = kv_floor + ABS_i + PCT_i × C          (headroom trip point for tier i)

tier        ABS_i     PCT_i×C     engaged when            KV mechanism
──────────────────────────────────────────────────────────────────────────────────
Trivial     2.0 GiB   4.0% × C    headroom ≤ T_trivial    release_empty_arenas, trim settled pool
Cheap       1.5 GiB   3.0% × C    headroom ≤ T_cheap      compact / compact_forced
Moderate    1.0 GiB   1.5% × C    headroom ≤ T_moderate   quantize_sealed_in_place on COMPLETED
                                                            float turns (compress-to-free) + reclaim
Costly      0.5 GiB   0.5% × C    headroom ≤ T_costly     evict_hot_to_free / demote_turns_to_warm
Critical    0         0           headroom ≤ kv_floor     warm→cold drop, evict_hot_except,
                                     (GPU SYNC)             expert-pool shrink (!all_resident)
```

**Why compress precedes evict.** The ladder orders by *future penalty*, not by
reversibility. Compressing a **completed** turn is a net shrink that keeps it
resident and attended-over, and it is **no extra loss** — the persistence thread
quantizes completed turns on seal regardless, so pressure only pulls the same
work forward. That makes its incremental penalty ≈ 0. Eviction, by contrast,
*moves* a hot copy off the card and pays a reload if the turn is re-attended, so
it sits one rung higher. Compress-to-free also feeds the reuse accounting
directly: the freed source is float working-set arenas (`reserved − used`),
which the budget counts as immediately reusable.
Worked (73 GiB card, kv_floor ≈ 13.7 GiB): Trivial trips at ≈ 13.7+2.0+2.9 = **18.6
GiB** headroom, Cheap ≈ 17.4, Moderate ≈ 15.8, Costly ≈ 14.6, Critical at the 13.7
floor. On a 16 GiB card the `PCT×C` terms shrink and the absolute offsets dominate,
keeping the tiers from collapsing on top of each other. `ABS_i`/`PCT_i` are the
tunables re-derived per card class.

- **Gentle-early:** `Trivial`/`Cheap` (release/compact/trim) reclaim fragmentation
  and dead arenas with **zero hit-rate cost**, and they engage while there's still
  gigabytes of headroom, so most pressure never reaches an evicting tier.
- **KV eviction withheld:** `Costly`+ (dropping hot copies off-card) only engages
  close to the floor — "don't be aggressive until really needed." `Moderate`
  compresses completed turns first, which stays resident, so eviction is a later
  resort.
- **Async everywhere except the top:** tiers `Trivial..=Costly` issue only
  `cuMemFreeAsync`/compaction — **no `synchronize`, no per-alloc/per-free barrier**.
  The measurement will reflect the reclamation once the stream retires the frees;
  because gentle relief keeps a buffer, the lag is harmless.
- **GPU-level sync only high on the ladder:** entering `Critical` does a **stream
  sync** (retire pending frees → measurement becomes ground truth), **remeasures**,
  runs the aggressive relief, then **syncs + remeasures again** to confirm. This is
  the one place we pay a sync, and it's a per-episode circuit breaker, not a
  per-op cost.

**Phase-aware trigger (scheduler), as built.** The scheduler's
`vram_under_pressure_for` gate — which decides *when* to run the relief cascade —
reads one exact counter, the KV side's free regions (`stats.free + stats.blocked`),
and compares it with a setpoint that depends on the phase the caller is in
(`VramPhase`, `scheduler/prefill/vram_phase.rs`). The earlier byte band derived from
driver headroom (`max(C/10, 2 GiB)` for load, `max(C/20, …)` for decode) and the
footprint qualifier built on `pool_reserved`/`used` (evict near C, defrag when a
whole arena is reclaimable, `reclaim_footprint`) went with the reservation: KV comes
from regions claimed at startup, so no driver reading enters the question.
- **Load** (prefill upload, section/scope ingest, warm→hot elevation — bringing KV
  into VRAM *before* attention): `LOAD_TARGET = FreeRegionTarget::new(8, 24, 8)` —
  span/8, at least 24 regions, relief frees 8 past it. A wide ragged forward claims
  regions fast, so the setpoint is wide enough that a seal never finds the free
  list empty.
- **Decode** (stable working set, ~1 chunk per sequence per 32 steps):
  `DECODE_TARGET = FreeRegionTarget::new(16, 8, 8)` — span/16, at least 8 regions.
  A thin setpoint keeps the maximum KV resident (the point of unbounded context).

The weight side is told the load target (`set_kv_free_target`) so its growth leaves
the relieved count free (`docs/vram_span_partition.md` §4).

### Registration (callers own their relief)
```rust
impl VramGovernor {
    /// Register a relief closure for `class` at `tier`. Closure attempts to free
    /// toward `want` and returns bytes it queued (async). Also reports an
    /// `evictable_estimate` used by the forecast (§9).
    pub fn register_relief(
        &self, class: AllocClass, tier: Criticality,
        relief: impl Fn(ReliefRequest) -> ReliefOutcome + Send + Sync + 'static,
        evictable: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> ReliefHandle;
}
```
KV registers `Trivial..=Critical` wrapping §3.3. Experts register `Moderate`
(slot→pinned) and `Critical` (shrink `num_slots`), active only when
`!all_resident`. Cross-class: a KV allocation relieves KV first; only `Critical`
reaches into `Expert` (and vice-versa) — the single protocol through which the two
budgets finally negotiate.

The relief loop escalates tier by tier, re-reading `measure().headroom` between
tiers (and after each `Critical` sync), stopping the moment headroom clears the
tier's threshold. If the whole ladder is exhausted and headroom is still below
floor → return `Exhausted` (circuit breaker: **no spin**), surfaced as a typed OOM.

### 8.1 Idle demote — the counterpart to elevate

**Sections are three-tier residences, exactly like turns.** `SequenceResidence`
carries `hot` / `warm` / `cold` regardless of whether it holds a turn or a
section, and `section_tier_state` reports all three. A section is prefilled hot,
quantized to C4 and offloaded per branch during the catalog build
(`OffloadCollectionMembers` → `quantize_section_batch(mark_evict = true)`, which
bounds the native catalog to one branch), then lifted back on demand:
`elevate_to_hot(sections, turns)` takes **both** lists.

**The gap this closes: elevate had no counterpart.** Every rung of the ladder
above that can reclaim — compress-to-free, `demote_turns_to_warm`,
`evict_hot_to_free` — walks *turns*. Nothing demoted a section once it was
elevated, and nothing demoted an idle conversation's turns either. So under
sustained projection churn the hot set only grew, every shortfall was met by
`request_kv_ground` conceding expert-weight ground, and that concession is
one-way: giving it back needs a floor move, which `live_watermark` refuses while
any region above the floor is live. Measured on a 16 GiB card: 234 concessions,
expert zone 5,440 → 1,225 slots, decode 26 → **3 t/s**, KV side 8.6 GiB against
2.8 GiB of weights. Every relief line read
`turns_compressed=0 turns_evicted=0 arenas_released=0 conceded_mib=<all of it>`.

**Mechanism — a last-touch epoch, not a live set.** Each `SequenceResidence`
carries `last_used_epoch`. `set_working_set_pins` — which already computes the
active set under the write lock — **stamps** it; the demote scan **advances** the
clock. Residences where `epoch − last_used > IDLE_DEMOTE_GRACE_EPOCHS` **drop
`hot`**, and only where `warm` already holds the bytes: the demote sheds a copy,
so it is never the operation that loses the last one.

**It sets no flag.** The obvious wiring — latch the existing `evict_when_cold`
and let the persistence thread's `residence_evict_when_cold` → `install_cold`
path do the freeing — is wrong twice over, and the shape of the flag is why.
`evict_when_cold` is a statement about a residence's *kind*: this KV belongs to a
transient timeline (a `code_read` fork, a calibration exemplar) whose cold copy
is disposable, so drop it the moment cold lands. Idleness is not that; a dormant
conversation's KV is as durable as any other. And the flag is **latched** —
every write in the tree is `= true` and nothing ever clears it — so one quiet
minute would permanently condemn ordinary KV to be dropped rather than persisted.
Dropping `hot` directly says exactly what is meant, is undone by the next
`elevate_to_hot`, and leaves the flag to mean the one thing it means.

**A slot's dropped units are held for one of its projections**
(`SlotState::previous_working_set`). Each slot keeps the working set its latest
projection replaced, and the pin publication holds it beside the current one
until that slot projects again. Another slot's projection cannot end the hold:
a global count of publications would, and on a busy member every other slot's
elevate ticked it within the same boundary. Without the hold, a recall
was shed on the first pass after it left the working set, so a projection that
re-selected the same `code_read` turn one boundary later read it off the log
again. Measured on BlackWorld on 2026-11-22: 15–36 turns and 120–236 MiB
recalled cold at every 64-token boundary of one thinking block, which was
75–180 ms of each rebuild. The hold is one projection, not the idle grace, so
VRAM holds a recall no longer than the selection that might want it back.

**A residence with a cold copy takes no warm copy** (`Substrate::pending_warm`,
`pin_defers_warm`). Its hot copy is already the stored form and the log holds
the same bytes, so a hot→warm copy is a device-to-pageable transfer on the
primary stream that buys nothing a cold read does not. Once recalls stopped
going straight back to cold, that copy cost 145–382 MiB per pass at ~0.7 GB/s,
100–540 ms of the stream the decode shares. The demote's safety condition is
therefore "warm *or* cold holds the bytes".

**The clock is one scan**, ticked by `Substrate::demote_idle_hot`, which the
scheduler calls each gap between forwards (`Scheduler::demote_idle_hot`,
`scheduler/gap_residence.rs`). Getting this unit right is the whole of the
calibration: an epoch tracks real work, and the window shortens under load, which
is when the VRAM is wanted back. A projection *publish* is far too coarse (a
whole session may do only a handful, which makes any usable grace unreachable and
the demote silently inert — measured).

The counter is **per-`Substrate`, not a global**: a process-wide clock lets one
engine's activity age out another's residences, which showed up immediately as
tier tests that passed alone and failed in parallel. Keeping it beside the
residences it stamps also removes the static, the atomic, and the epoch argument
every caller would otherwise thread through. Residences are stamped born-now at
`alloc_residence`, so none is ever stale from birth and eligible for demotion
before its first use.

Three properties make this the shape to build rather than a live "active set":

- **The active set is excluded arithmetically, not by a filter.** Anything the
  current forward touched is stamped at the current epoch and cannot satisfy the
  predicate. There is no check to forget and no set to get wrong.
- **No snapshot window.** A set is built and then acted on, and reality can move
  in between; a stamp is written by the consumer at the moment of use.
- **No hot-path allocation.** One store per touch, one integer compare per
  candidate — versus assembling every active conversation's KV per scan.

**The grace window is load-bearing.** With a grace of zero, a conversation's KV
becomes evictable the instant a *different* conversation forwards, which at 64
concurrent sessions is continuous thrash. The grace keeps a conversation resident
across its own turn while letting a genuinely idle one age out. It is sized
against the tick above, never guessed — the failure directions are asymmetric, so
too generous only reclaims later while too aggressive charges a warm→hot lift on
every turn of a live conversation.

**Why it cannot demote live KV.** Five barriers, the first structural: (1) the
epoch predicate excludes the current forward by construction; (2) the working-set
pins are checked as well, so a residence the current wave attends is skipped even
if its stamp were somehow stale; (3) the stamp is written by the consumer, so
"inactive" cannot go stale between decision and action; (4) the scan runs only on
the scheduler's between-forward seam — after the gap's quantize, where every free
is ordered behind the forward that last read the block, with no wave in
flight; (5) it only ever drops a **redundant** copy — `has_warm()` is part of
the predicate, and a residence still flagged `PENDING_QUANTIZE` (whose
`hot` is the interim native form the scheduler still owes a quantize) is excluded too.

### 8.2 Splice sources are exempt from every hot-drop

A `code_read` file is ingested as parallel scope forks: each scope round-trips on
a throwaway timeline, the file conversation then re-records the fork's sealed
turns **by reference** (`adopt_turn`), and the fork is tombstoned. `adopt_turn`
reads the fork turn's **hot** copy and nothing else. Between the fork's seal and
its splice the persistence thread lands a warm copy — which makes that hot copy
redundant by every definition above, and so a candidate for `evict_hot_to_free`,
`evict_hot_except`, `demote_turns_to_warm`, `demote_cold_ingest` (forks are ingest
timelines), the idle demote, and the migrate's own `install_warm_and_evict_hot`.
Measured 2026-09-13 on a fresh ingest at `--max-depth 2`: 20 of 131 files failed
their splice with `source K/V not hot` in under five minutes, the first one second
after a relief pass evicted three turns; every root file, `README.md` among them,
was lost, and the ingest stopped at its failure cap.

So `fork_scope` marks each fork's timeline a **splice source**
(`Substrate::mark_timeline_splice_source`), and every residence on it carries
`splice_source`. Every automatic hot-drop honours the flag. The fork's
`tombstone_timeline` clears it and drops the hot copy — the only release a fork
needs, since it lives for one scope chunk (`SCOPE_PARALLELISM` scopes), and the
file conversation's cloned chunk handles keep the shared K/V alive. Clearing it
at the tombstone, before the drop, also keeps a migrate install that lands after
the tombstone from re-adding a hot copy nothing will free.

The flag is deliberately **not** `no_cold_persist`. That one marks every transient
timeline, and its other user — calibration exemplars — *wants* its hot copies
shed: exempting all transient timelines would hold ~3,000 exemplars' K/V on the
card through a calibration.

---

## 9. Managed allocation, retry, and the forecast

*Removed: `reserve`, `allocate`, `forecast_units` and `evictable_estimate` are gone
(`candle-core/src/vram/managed.rs` keeps `is_oom` and `usable` only). Nothing
called the managed path once KV stopped allocating from the pool. Admission width
is the scheduler's own (`scheduler/admit/`).*

### 9.1 Managed allocation (progressive, measurement-gated)
```rust
impl VramGovernor {
    /// Permanent classes (Weights, and the fixed part of Scratch/Expert): record
    /// the class tag and run `alloc`. No prediction — the class just tags it.
    pub fn reserve<T>(&self, class: AllocClass, alloc: impl FnOnce() -> Result<T>) -> Result<T>;

    /// Variable/expert allocation: run `alloc`; if it OOMs, relieve (escalating
    /// the ladder) and retry, up to Critical; then typed OOM.
    pub fn allocate<T>(&self, class: AllocClass, alloc: impl FnMut() -> Result<T>) -> Result<T>;
}
```
- **Progressive (requirement 10):** only the *major* sites call these (KV arena
  create, expert slot load, big scratch, weight load). Small transient tensors
  allocate directly — the live measurement catches them; there is no counter to
  keep exact.
- **Retry on failure (requirement 11):** `allocate` catches `KV_DEVICE_OOM_MARKER`
  or raw driver OOM (`is_device_oom`, `backing/refusals.rs:39`), invokes relief, retries,
  escalates a tier per round, gives up after `Critical`. Subsumes today's
  `ensure_vram_budget` force-compact retry (`alloc.rs:244`) and `handle_prefill_oom`
  requeue (`scheduler/prefill/wave_run.rs:652`).
- **Scratch is never predicted:** scratch allocations just call `reserve(Scratch,…)`
  and are allowed to proceed; if they need room, the ladder (KV is the flex) makes
  it. "It just has to be allocated or everything breaks" — so scratch always wins
  over KV, which evicts for it.

### 9.2 Forecast — sizing the concurrency loops (requirement)
Prefill parallelism must account for KV we *will* evict once prefill runs:
```rust
impl VramGovernor {
    /// Max concurrent units of `per_unit_kv_bytes` that fit given current real
    /// headroom PLUS what KV can recoverably free (up to `Costly` — compress
    /// completed turns + reversible hot→warm evict; no drop-to-cold, no sync).
    /// Feeds the scheduler's admit_window instead of the AAIMD guess.
    pub fn forecast_units(&self, per_unit_kv_bytes: u64) -> usize {
        let headroom  = self.measure().headroom;
        let evictable = self.evictable_estimate(Criticality::Costly); // recoverable only
        ((headroom.saturating_add(evictable)) / per_unit_kv_bytes.max(1)) as usize
    }
}
```
**Forecast as ceiling, AIMD smooths the ramp.** The forecast does not replace the
AIMD `admit_window` (`mod.rs:4205-4225`) — it **caps** it:
```
admit = min(forecast_units(per_seq_kv_bytes), aimd_window)
```
AIMD still ramps additively (+1/loop) and backs off multiplicatively, but never
past the measured ceiling. This keeps the proven anti-thrash ramp (no snapping to
full width and re-tripping) while grounding the ceiling in real headroom +
recoverably-freeable KV. The forecast can legitimately exceed raw free headroom
*because* prefill will free KV under pressure — but it counts only up to `Costly`
(compress-completed-turns, which quantize on seal anyway, + reversible evict; no
drop-to-cold, no sync), so it never plans on permanently damaging the cache.
`evictable_estimate` sums the registered `evictable()` reporters up to the given
tier.

---

## 10. Diagnostics & logging (requirement)

*As built* (`candle-core/src/vram/diag.rs`): `BudgetRow { class, reserved }` and
`BudgetTable { capacity_c, total, headroom, source, rows: [BudgetRow; AllocClass::COUNT],
pool_cushion }`. The design as first written, with the removed floor, forecast and
ladder fields:

```rust
pub struct BudgetRow { pub class: AllocClass, pub reserved: u64 }
pub struct BudgetTable {
    pub capacity_c: u64, pub total: u64, pub headroom: u64,
    pub rows: Vec<BudgetRow>,          // loose per-class tallies
    pub kv_floor: u64, pub variable_max: u64,
    pub evictable_reversible: u64,     // forecast input
    pub thresholds: [u64; 5],          // ladder trip points
    pub last_relief: Option<(Criticality, u64)>,
}
impl VramGovernor {
    pub fn budget_table(&self) -> BudgetTable;  // structured, for asserts + rendering
    pub fn log_budget(&self, whence: &str);     // tracing::info! a rendered table
}
```
- **Event logging:** every balloon result, every relief episode (tier, bytes
  queued, headroom before/after), every `Critical` sync/remeasure, and every
  budget-change event logs on a `candle_core::vram` target — the diagnostic trail
  that was missing when the sawtooth was hard to explain.
- **Diag mode / table:** `budget_table()` returns the structured snapshot and
  `log_budget()` renders it as a table. **Unit tests call `budget_table()` and
  assert on it** (capacity, per-class rows, floor, thresholds, evictable), so the
  same view a human debugs with is the view tests pin.

---

## 11. Expert budget — a point-in-time computation at expert-load

*Removed: `expert_budget()` went with `kv_floor`. The resident-expert count is the
weight zone's capacity, `(span − MIN_ELASTIC_RESERVE) / slot_bytes`, computed once
against the span (`candle-core/src/vram/managed.rs` module doc;
`docs/vram_span_partition.md` §4).*

The expert module needs a concrete resident count (`num_slots`) **up front** to
drive its two-tier GGUF→VRAM/pinned repack. So the governor exposes a **single
computation, evaluated at the instant experts are about to be allocated** — after
the mandatory weights are already resident, so the live measurement already
reflects them:

```rust
impl VramGovernor {
    /// Bytes available to hold MoE experts resident, computed NOW from the real
    /// measurement (weights already loaded), leaving the KV floor + scratch
    /// cushion free. The expert module divides this by max_expert_size to pick
    /// how many slots to keep resident.
    pub fn expert_budget(&self) -> u64 {
        let headroom = self.measure().headroom;         // reflects resident weights
        headroom.saturating_sub(self.kv_floor())
                .saturating_sub(SCRATCH_MARGIN)
    }
}
```
The expert loader then:
```
budget      = governor.expert_budget()
num_slots   = min(round(budget / max_expert_size), total_experts)
all_resident = num_slots >= total_experts
```
- **Point-in-time, measured, not predicted.** It reads real headroom at the exact
  moment of the decision, so it's accurate to what the card actually has after
  weights — no `cuMemGetInfo`-free guessing, no double-counting. This replaces
  today's `min(max(free−5GB, free/2), total_expert_bytes)`
  (`quantized_qwen3_moe/loader.rs:1467-1536`).
- **Experts can never starve KV below its floor** — the budget subtracts
  `kv_floor` (+ `SCRATCH_MARGIN`) before experts get a byte, by construction.
- **All-resident preferred when it fits** — if `total_expert_bytes ≤ budget`,
  every expert stays resident → the static-index fast path (`handle.rs:287`), zero
  DMA stalls; KV gets the rest.
- **Partial residency degrades gracefully** — the score-based cache churns the
  remainder (`expert_lre/cache/bookkeeping.rs:180-270`), the 69% Markov hit-rate hides the DMA, and
  experts register `Moderate`/`Critical` relief so KV can borrow expert VRAM only
  under genuine overload.
- **`SCRATCH_MARGIN`** is a small fixed cushion (not a scratch prediction — §9.1)
  so the first forward's scratch lands before any KV eviction is needed.

`num_slots` and `all_resident` remain exactly the values the expert cache already
consumes; only *how they're derived* changes — from a free-VRAM guess to one
measured computation at the load instant.

---

## 12. Module layout

Per AGENTS.md ("one concern per file", subfolder). In **candle-core** (owns the
device; reachable by both consumers):

```
candle-core/src/vram/
  mod.rs        VramGovernor, per-GPU global registry, AllocClass
  reading.rs    VramReading, VramProbe trait, ProbeKind
  probe_dxgi.rs #[cfg(windows)] DXGI backend + LUID match + budget-change event
  probe_cuda.rs cuMemGetInfo backend (TCC and Linux)
  driver_model.rs WDDM / TCC / Linux, read from the device
  balloon.rs    balloon-and-measure, wobble margin
  budget.rs     GovernorConfig, per-class tally
  managed.rs    capacity arithmetic (usable), is_oom
  diag.rs       BudgetTable, log_budget, event targets
  gpu_addressable.rs  the DXGI NON_LOCAL host budget (docs/driver_model.md §3)
  host_probe.rs, process_ram/, process_paging/  host RAM and paging probes
  tests.rs
```
(`relief.rs` and the `Criticality` ladder it held no longer exist.)
Deps: `cudarc` (existing); `windows` crate (`#[cfg(windows)]`, minimal features) or
`windows-sys` hand-roll. **No NVML.**

### 12.1 Integration points (what changes, where)

*These were the planned changes for the removed runtime half; the file and line
citations below are those of the code as it was when the design was written and
no longer resolve. Of them, the balloon at session construction
(`ensure_vram_governor`, `batched_model/model_core.rs`) and `device_init_free`
as the circuit-breaker fallback (`BALLOON_FALLBACK_MARGIN`, `vram/mod.rs`) stand.*

- **Replace** `vram_has_room`/`vram_budget_available` (`alloc/budget.rs:140/210`) with
  governor queries; `ensure_vram_budget` (`alloc.rs:234`) → `governor.allocate(Kv,…)`.
- **KV registers relief** wrapping §3.3; `relieve_vram_pressure` (`scheduler/prefill/relief.rs:74`)
  and `evict_to_fit_incoming` (`projection_turns.rs:6470`) call the governor.
- **Scheduler admit width** ← `governor.forecast_units(...)` (`mod.rs:4205-4225`,
  `prefill.rs:139`).
- **Expert fill** (`quantized_qwen3_moe/loader.rs:1467-1536`) → measured fill (§11);
  expert cache registers slot relief.
- **Weight load** (`quantized_qwen3_moe/loader.rs:1543-1694`) → `governor.reserve(Weights,…)`.
- **Balloon** at session construction (`batched_inference.rs:560`), before load.
- **`device_init_free`** kept only as the balloon circuit-breaker fallback.

---

## 13. Failure modes & circuit breakers

| Situation | Behavior |
|---|---|
| Balloon can't reach floor / card < model | log, fall back to `cuMemGetInfo total − margin`; never hang |
| Ladder exhausted, still below floor | `Exhausted`; `allocate` → typed OOM; **no spin** |
| DXGI init fails on a WDDM device | no governor, named in the log — a `cuMemGetInfo` reading there would size `C` into demotion |
| Device under TCC (or Linux) | `CudaProbe` by design; DXGI does not see a TCC adapter (`docs/driver_model.md`) |
| External app collapses our budget mid-decode | budget-change event → relieve toward `kv_floor`, hold; never below floor; don't thrash |
| `Critical` sync storm | rate-limit `Critical` entry (min interval); hold at floor between |
| Async frees not yet retired | measurement lags but gentle-early relief kept a buffer; if a physical alloc still fails, escalate to `Critical` (sync forces retirement) then retry |
| `all_resident` experts | expert relief registry empty; only KV tiers run; pool never shrinks |
| Scratch needs room | always allowed; KV (flex) evicts for it |

---

## 14. Test plan

**This is the original design record; most of it is not in the tree.** The tests
for the removed runtime half — the ladder, `kv_floor`, `forecast_units`, the expert
budget and external-theft relief — went with that code. The startup half's tests
are in `candle-core/src/vram/tests.rs`: `budget_evolves_with_allocations`,
`balloon_undersized_falls_back` (the plan's `balloon_undersized_fallback`),
`balloon_stops_at_the_real_ceiling`,
`balloon_stops_at_the_residency_budget_when_the_driver_never_refuses`,
`both_balloon_paths_target_total_less_the_reserve`,
`refusal_refines_the_chunk_to_within_a_granule_of_the_ceiling`,
`the_wddm_margin_is_capped_at_half_a_gib`, the `usable_*` capacity tests,
`budget_table_shape`, and `startup_partitions_evolve_and_report`, plus the
`real_cuda_*` tests behind the `cuda` feature. The plan below is kept as the
record of what was specified.

All non-GPU tests inject a `FakeProbe` (scripted `VramReading` sequence) +
`FakeAllocator` (scripted failures). **Every test asserts via `budget_table()`** —
the same diag view humans use.

**Budget & measurement**
- `balloon_measures_capacity`: declining headroom → stop at floor → `C` = high-water.
- `budget_evolves_with_allocations`: tag Weights then Expert; `budget_table` rows +
  headroom track the scripted measurement, no upfront prediction.
- `kv_floor_abs_plus_pct`: floor = `KV_FLOOR_ABS + KV_FLOOR_PCT × (C − Weights)`;
  small-card floor dominated by abs, big-card by pct.

**Relief ladder**
- `gentle_relief_early`: headroom just under `T_trivial` → only Trivial/Cheap fire;
  Moderate+ untouched (hit-rate protected).
- `escalates_to_moderate_only_near_floor`: eviction tiers engage only below their
  thresholds.
- `critical_syncs_and_remeasures`: entering Critical → exactly one pre- and one
  post- sync+remeasure (fake counters).
- `async_tiers_never_sync`: Trivial..=Costly perform zero syncs.
- `ladder_exhausted_no_spin`: all closures return 0 → `Exhausted`, bounded, typed OOM.

**Expert budget**
- `all_experts_resident`: experts fit before floor → `all_resident`, empty expert
  relief, KV gets remainder.
- `some_experts_resident`: fill stops at `kv_floor + SCRATCH_MARGIN` → `num_slots <
  total`, expert relief registered.
- `experts_never_cross_kv_floor`: measured fill never drops headroom below floor.

**Forecast & external pressure**
- `forecast_counts_reversible_evictable`: `forecast_units` = `(headroom +
  evictable≤Moderate)/per_unit`; excludes lossy/critical.
- `external_theft_holds_floor`: budget-change drops headroom → relieve to floor,
  never below; `budget_table` shows the clamp.
- `budget_recovers`: headroom restored → forecast width grows back.
- `dxgi_unavailable_falls_back`: probe errors → `CudaProbe`, run continues.
- `balloon_undersized_fallback`: fallback capacity, no hang.

**Diagnostics**
- `budget_table_shape`: rows per class, thresholds, floor, evictable populated;
  `log_budget` renders without panic.

**Integration (GPU-gated, `#[cfg(feature="cuda")]`)**
- `balloon_fast`: init-time delta under threshold (guards "must not move init").
- `kv_growth_respects_band`: `pool_reserved` never exceeds resident capacity; no stall.
- `end_to_end_no_paging`: the 14–115 s repro stays single-digit-second, `headroom`
  never crosses zero.

---

## 15. Decisions & remaining questions

Resolved in review:
1. **`kv_floor` = 3 GiB + 15% × (C − Weights).** ✔ (§7)
2. **Ladder thresholds = absolute + percentage** above the floor (same hybrid as
   `kv_floor`), tunable per card class. ✔ (§8)
3. **DXGI via the `windows` crate**, minimal features, `#[cfg(windows)]`. ✔ (§6, §12)
4. **Forecast caps AIMD** (`admit = min(forecast, aimd_window)`); AIMD still smooths
   the ramp. ✔ (§9.2)

Open / defaulted (flag if you disagree):
5. **`SCRATCH_MARGIN`.** Now `GovernorConfig::scratch_margin`: the cushion left
   outside the KV reservation for allocations that still come from the CUDA pool
   (the gallery arena's slabs, the grow-only scratches, a streamed model's layer
   pack). **Defaults to 512 MiB** (`CANDLE_VRAM_SCRATCH_MARGIN_MB`; a forward's
   activations come from the reservation's transient tier, not from it). (§9.1, §11)
6. **`ABS_i`/`PCT_i` ladder constants.** The specific `{2.0/4%, 1.5/3%, 1.0/1.5%,
   0.5/0.5%}` values are a starting shape to validate against real runs and
   re-derive per card class (AGENTS.md: measure, don't guess). (§8)

---

## 16. Why this finally fixes it

- The gate is the **real, physically-resident measurement** (balloon-anchored,
  DXGI/cuMemGetInfo-tracked) — it **cannot lie or drift**, because there is no
  virtual counter to desync.
- The budget **evolves with allocation** and needs no prediction — the model
  allocates everything before inference, and by then the measurement is the truth.
- Pressure is relieved **cheapest-first on a threshold ladder**: fragmentation and
  dead arenas go early at zero hit-rate cost; **KV eviction is withheld until the
  floor is near**; the only GPU sync is the top-of-ladder circuit breaker.
- The **abs+pct KV floor** + post-balloon OS reservation convert external VRAM
  theft into bounded, reversible KV eviction — resilient to squatters, immune to
  paging — while the **forecast** lets prefill safely use evictable headroom for
  parallelism without hurting the cache.
