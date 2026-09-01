# VRAM Governor — Design Document

**Status:** Draft for review (rev 2 — measurement-driven)
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

### 3.1 Single allocation chokepoint
Every device allocation funnels through `CudaDevice::alloc`/`alloc_zeros`
([`device.rs:46-58`](../candle-core/src/cuda_backend/device.rs)) → cudarc
`cuMemAllocAsync` on the default stream-ordered pool. Free is RAII (drop →
`cuMemFreeAsync`, returns to the **pool**, not the OS). Present measurement
primitives: `mem_get_info` (`device.rs:318`), `pool_used_bytes`/`pool_reserved_bytes`
(`device.rs:333/345`), `trim_pool` (`device.rs:384`), `synchronize` (`device.rs:726`).

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
`evict_hot_to_free` (`substrate.rs:1716`, oldest hot→warm, reversible),
`evict_hot_except` (`substrate.rs:1646`), `demote_turns_to_warm` (`substrate.rs:1788`),
`quantize_sealed_in_place` (`compress.rs:157`, **lossy**), warm→cold
(`thread.rs:570`), `purge_warm_to_target` (`substrate.rs:1553`). The existing
cascade `relieve_vram_pressure` (`prefill.rs:74-128`) returns "still pressured?" as
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
- **Linux:** `cuMemGetInfo` is accurate (no WDDM virtualization).
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
1. Grow a chain of large chunks (1–2 GiB `cuMemAlloc`), touching each with a
   **strided** `cuMemsetD8` (one write per WDDM segment granule is enough to force
   residency and evict others — we don't memset every byte). Stop when **either**
   reserved reaches `BALLOON_TARGET_FRAC` (default 0.90) of total, **or** the probe
   headroom hits `BALLOON_FLOOR`, **or** an allocation fails.
2. `C = reserved_high_water`.
3. **Free the balloon.**
4. On Windows, **now that `C` is known**, `SetVideoMemoryReservation` to the floor
   we intend to defend (§6) so the OS trims *others* first if contended.

**Fast in the general case (requirement):** strided touch makes this bandwidth-
trivial (a handful of writes per GiB), so even 0.9 × 73 GB is well under model-load
time. No DXGI-specific fast path; the same balloon runs everywhere (on Linux the
touch simply forces allocation, there's nothing to evict). `BALLOON_TARGET_FRAC`,
`BALLOON_FLOOR`, `BALLOON_TOUCH_STRIDE`, chunk size are tunables.

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
- `CudaProbe` — `cuMemGetInfo`. **Primary on Linux**, and the Windows fallback if
  DXGI init fails.

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

**Phase-aware trigger band (scheduler).** The scheduler's `vram_under_pressure`
gate — which decides *when* to invoke this ladder — sizes its reserve band by the
phase the caller is in (`VramPhase`, `scheduler/prefill.rs`). The availability
number is phase-independent (`headroom + reuse`, so the WDDM false-pressure fix
always holds); only the band it is compared against changes:
- **Load** (prefill upload, section/scope ingest, warm→hot elevation — bringing KV
  into VRAM *before* attention): wide band `max(C/10, 2 GiB)`. Here the freed-float
  free-list (`pool_reserved − pool_used`) is the *destination* of the incoming KV —
  the forward will consume it — and a wide ragged forward has a large transient
  activation peak, so the free-list is **not** spare admission capacity; relieve
  early to make real headroom before the load competes for it.
- **Decode** (stable working set, ~1 chunk/token/step): thin band
  `max(C/20, CANDLE_VRAM_DECODE_BAND_MB=1.5 GiB)`. The free-list *is* genuinely spare,
  so keep the maximum KV resident (the point of unbounded context).

This is why "R16/F16 availability is irrelevant when loading KV before attention,
and the opposite in decode" — encoded as a wider vs thinner reserve, not by
excluding the reuse term (which would reintroduce false pressure).

**Actionable-footprint qualifier.** Beyond the phase band, the gate also trips on
the raw pool footprint — but only in a state it can *act* on, so it never churns a
no-op reclaim:
- **Evict** — resident `used` within `max(8%C, 1 GiB)` of C. This is the real
  paging signal (the working set), independent of the reserved gap.
- **Defrag** — `reserved` over `C − band` **and** a whole arena is reclaimable
  (`can_reclaim_arena`). The reserved-but-free gap is compactable only while it is
  fragmentation *inside live arenas*; once free has fallen to the CUDA pool's own
  free-list with no whole arena recoverable, `trim` can't return it and compaction
  has nothing to move. Firing there churns a `shed=0` reclaim every wave, so with
  `used` far below C the gate treats that gap as free headroom (not a paging risk)
  and holds. When defrag *does* run, `reclaim_footprint` ramps the bounded
  compaction budget with the overshoot (×1 at the ceiling → ×4 near C).

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

**The clock is one persistence pass**, ticked at the scan. Getting this unit
right is the whole of the calibration: the pass wakes on every seal/flush trigger
and otherwise on its own 5 s tick, so an epoch tracks real work with a wall-clock
floor — a dormant conversation retires even with zero traffic, and the window
shortens under load, which is when the VRAM is wanted back. The alternatives are
both wrong: a projection *publish* is far too coarse (a whole session may do only
a handful, which makes any usable grace unreachable and the demote silently
inert — measured), and a per-*wave* tick would need a hook the scheduler does not
expose to the substrate.

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
the persistence thread's between-forward seam — the same one
`create_deferred_arenas` depends on, past `device.synchronize()`, with no wave in
flight; (5) it only ever drops a **redundant** copy — `warm.is_some()` is part of
the predicate, and a residence still owed a quantize (`pending_quantize`, whose
`hot` is the interim native form the drain is about to replace) is excluded too.

---

## 9. Managed allocation, retry, and the forecast

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
  or raw driver OOM (`is_device_oom`, `backing.rs:39`), invokes relief, retries,
  escalates a tier per round, gives up after `Critical`. Subsumes today's
  `ensure_vram_budget` force-compact retry (`alloc.rs:244`) and `handle_prefill_oom`
  requeue (`prefill.rs:652`).
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
  (`quantized_qwen3_moe.rs:1467-1536`).
- **Experts can never starve KV below its floor** — the budget subtracts
  `kv_floor` (+ `SCRATCH_MARGIN`) before experts get a byte, by construction.
- **All-resident preferred when it fits** — if `total_expert_bytes ≤ budget`,
  every expert stays resident → the static-index fast path (`handle.rs:287`), zero
  DMA stalls; KV gets the rest.
- **Partial residency degrades gracefully** — the score-based cache churns the
  remainder (`cache.rs:180-270`), the 69% Markov hit-rate hides the DMA, and
  experts register `Moderate`/`Critical` relief so KV can borrow expert VRAM only
  under genuine overload.
- **`SCRATCH_MARGIN`** is a small fixed cushion (not a scratch prediction — §9.1)
  so the first forward's scratch lands before any KV eviction is needed.

`num_slots` and `all_resident` remain exactly the values the expert cache already
consumes; only *how they're derived* changes — from a free-VRAM guess to one
measured computation at the load instant.

---

## 12. Module layout

Per CLAUDE.md ("one concern per file", subfolder). In **candle-core** (owns the
device; reachable by both consumers):

```
candle-core/src/vram/
  mod.rs        VramGovernor, per-GPU global registry, AllocClass, Criticality
  reading.rs    VramReading, VramProbe trait, ProbeKind
  probe_dxgi.rs #[cfg(windows)] DXGI backend + LUID match + budget-change event
  probe_cuda.rs cuMemGetInfo backend (Linux primary, universal fallback)
  balloon.rs    strided balloon-and-measure
  budget.rs     evolving budget, variable band, kv_floor (abs+pct), per-class tally
  relief.rs     criticality ladder, thresholds, relief loop, evictable_estimate
  managed.rs    reserve()/allocate()/forecast_units(), pressure-retry
  diag.rs       BudgetTable, log_budget, event targets
  tests/        (see §14)
```
Deps: `cudarc` (existing); `windows` crate (`#[cfg(windows)]`, minimal features) or
`windows-sys` hand-roll. **No NVML.**

### 12.1 Integration points (what changes, where)
- **Replace** `vram_has_room`/`vram_budget_available` (`alloc.rs:140/210`) with
  governor queries; `ensure_vram_budget` (`alloc.rs:234`) → `governor.allocate(Kv,…)`.
- **KV registers relief** wrapping §3.3; `relieve_vram_pressure` (`prefill.rs:74`)
  and `evict_to_fit_incoming` (`mod.rs:6470`) call the governor.
- **Scheduler admit width** ← `governor.forecast_units(...)` (`mod.rs:4205-4225`,
  `prefill.rs:139`).
- **Expert fill** (`quantized_qwen3_moe.rs:1467-1536`) → measured fill (§11);
  expert cache registers slot relief.
- **Weight load** (`quantized_qwen3_moe.rs:1543-1694`) → `governor.reserve(Weights,…)`.
- **Balloon** at session construction (`batched_inference.rs:560`), before load.
- **`device_init_free`** kept only as the balloon circuit-breaker fallback.

---

## 13. Failure modes & circuit breakers

| Situation | Behavior |
|---|---|
| Balloon can't reach floor / card < model | log, fall back to `cuMemGetInfo total − margin`; never hang |
| Ladder exhausted, still below floor | `Exhausted`; `allocate` → typed OOM; **no spin** |
| DXGI init fails (no D3D / RDP) | fall back to `CudaProbe`; DXGI disabled this run |
| External app collapses our budget mid-decode | budget-change event → relieve toward `kv_floor`, hold; never below floor; don't thrash |
| `Critical` sync storm | rate-limit `Critical` entry (min interval); hold at floor between |
| Async frees not yet retired | measurement lags but gentle-early relief kept a buffer; if a physical alloc still fails, escalate to `Critical` (sync forces retirement) then retry |
| `all_resident` experts | expert relief registry empty; only KV tiers run; pool never shrinks |
| Scratch needs room | always allowed; KV (flex) evicts for it |

---

## 14. Test plan

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
5. **`SCRATCH_MARGIN`.** The cushion left above `kv_floor` when computing
   `expert_budget()` so the first forward's scratch lands before any KV eviction.
   **Defaulting to 1 GiB** (tunable, `CANDLE_VRAM_SCRATCH_MARGIN_MB`) — sized to
   clear a wide prefill's transient activations + the grow-once scratch pools. (§9.1, §11)
6. **`ABS_i`/`PCT_i` ladder constants.** The specific `{2.0/4%, 1.5/3%, 1.0/1.5%,
   0.5/0.5%}` values are a starting shape to validate against real runs and
   re-derive per card class (CLAUDE.md: measure, don't guess). (§8)

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
```
