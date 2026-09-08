# The Wave Feeder — a fat pipe, a smart valve, and a simulator to tune it

**Status:** §4.11 is the design as built. §4.1–§4.10 are the record of what was
tried before it and why each attempt was falsified; they are kept because the
same wrong ideas keep looking right. **They describe code that no longer
exists** — §4.11.6 lists what was deleted once the fill became the only
admission path, including the trace this document's early sections were written
from. Every number here is measured, not estimated.

---

## 1. The shape of the problem

Work reaches a forward through three stages, and today only the middle one is
designed:

```
   producer                 feeder                    engine
   (ingest pool)  ──queue──▶ (admission)  ──wave──▶  (forward)
```

The **feeder** — then `scheduler/admission.rs`, now `scheduler/interleave.rs` —
was thought good. It regulated in bytes rather than sequence counts, priced
prefill as a stock and decode as a rate, and its module header contained what
looked like the decisive argument for why counts cannot work: *"a width of 9 is
idle headroom for nine short scopes and an out-of-memory forward for nine long
ones."* The argument is sound and the conclusion was still wrong — the answer is
neither a count nor a byte budget but a claim through the real allocators
(§4.11). The module is deleted (§4.11.6).

The **producer** is not designed at all. It is a worker pool whose width is a
count-based constant, and every attempt to tune that constant has failed in both
directions — five settings in one session, each either starving the queue or
over-filling the card. That is the signature of a proxy variable, not a control.

The intent of this document: make the producer a **fat pipe** that keeps the
queue deep and varied, make the feeder a **smart valve** that degrades
gracefully under whatever it is handed, and build a **simulator** so the valve's
control law can be fitted against load the real machine cannot conveniently
produce.

## 2. Measured baseline

### 2.1 What the model can do

`test_parallel_batched_forwarding_36_35b`, Qwen3.6-35B-A3B, 16 GB RTX 4090
Mobile, speculative decode on:

| contexts | aggregate | per context |
|--:|--:|--:|
| 1 (BF16) | 19.8 tok/s | 19.8 |
| 1 (Q8_0 / C0–C7) | 34–47 tok/s | 34–47 |
| 4 | **138.9** | 34.7 |
| 8 | **120.7** | 15.1 |
| 16 | **148.2** | 9.3 |

Aggregate throughput is roughly flat from 4 contexts upward and **5–7× the
single-context figure**. Width is worth a lot, and only up to a point.

### 2.2 What the daemon achieves

From `wave_trace.jsonl` over a repo-map ingest (226 waves, 17 admission
decisions):

```
QUEUE
  candidates offered    min=1  med=1  mean=1.0  p90=1  max=1
  candidates admitted   min=1  med=1  mean=1.0  p90=1  max=1
  backlog (tokens)      min=55 med=202 mean=303 p90=571 max=768
  passes admitting nothing: 0 of 17 (0%)

ENGINE
  decode width          min=1  med=1  mean=1.0  max=1
  decode tok/s          min=0  med=7  mean=5.9  max=11
  weight zone MiB       min=6318 med=8918 mean=8909 max=10398
  free KV regions       min=25 med=56 max=322
  transient tier MiB    med=64 p90=64 max=832
```

### 2.3 The three findings

**(a) The pipe is empty.** The queue never held more than one candidate, across
the entire run. Admission refused nothing — not once. Every downstream number is
therefore a measurement of the producer, not of the feeder or the engine.

**(b) The width formula fights the thing that makes decode fast.**
`repo_scan::scan_width` computes `for_kv = capacity − scratch − fixed − baseline`
where `fixed = pool_used − kv` — which **includes the expert cache**. So the
healthier the expert cache, the smaller `for_kv`, and the narrower the pool. On
the run above, with the weight zone in good shape at 8,918 MiB, the pool sized to
`n_workers=1`. The producer is throttled hardest exactly when the engine is in
its best state.

**(c) Creates are serialised.** A directory mints three conversations — folder
chain, generation, probes — each through `new_conversation_with_projection` under
the global engine mutex, and each a synchronous scheduler round-trip that blocks
until the current wave ends. The calibration path already solved this with
`new_conversations_with_projection_batch`, whose comment records the same
symptom: *"the whole window's cases prefill together in a wide forward instead of
trickling in one per wave-latency (which starved the batch to 2–4 wide)."* The
ingest pool never adopted it.

**What this retires.** Earlier in this work the slowness was attributed, in turn,
to PCIe limits, host RAM starvation, expert-cache thrash, and the elastic
partition. The trace refutes all four: the weight zone is healthy at 8.9 GiB,
concessions run at ~1/min, host RAM is unremarkable, and the engine is idle
waiting for work. **The whole gap is producer-side.**

## 3. Instrumentation (built, then deleted — §4.11.6)

> This section describes the trace the §4.5 controller was to be fitted against.
> The controller was falsified and the trace went with it: its `adm` half was
> written from inside the byte-budget path, so once the fill took over a GPU run
> emitted only `wave` records — half of a format whose entire purpose was the
> join. The live equivalents are the `wave fill` DEBUG line
> (`scheduler::interleave` target) and the memory report's admission section.

`candle-conversation/src/scheduler/wave_trace.rs` — always on, JSONL, inert until
`set_path`. Order one record per second; no flag, because a trace you have to
enable is a trace you do not have when the interesting run happens.

**`adm`** — one per admission decision: `ceiling`, `in_flight`, `reserve`,
`live_width`, `backlog`, and every candidate with `{kind, tokens, kv_bytes,
context_tokens, admitted}`. `context_tokens` is carried separately from `tokens`
because they differ by four orders of magnitude for a decode (1 token fed,
25,213 attended) and they drive different terms of the transient cost.

**`wave`** — one per wave window: per-channel forwards / width / tokens / attended
KV / ms; region occupancy (`live`, `free`, `blocked`, `total`); `weight_bytes`;
and the transient tier split into **attention / FFN / forward** peaks from
`wave_domain_stats`.

`zend/examples/wave_trace_report.rs` summarises both sides and joins them —
decode tok/s banded by the weight zone it ran under.

## 4. Proposed design

### 4.1 Fat pipe — the producer

Two changes, both with precedent in the calibration path:

1. **Batch the creates.** `new_conversations_with_projection_batch` for a whole
   window of directories, so slot allocation costs one wave boundary rather than
   one per conversation.
2. **Stop sizing the pool from residency.** The producer's width should be
   bounded by *queue depth*, not by VRAM: keep the queue between a low and high
   water mark and let the feeder decide what runs. Concretely, replace
   `SCAN_CONV_KV_MIN` with a target queue depth in **tokens** — the same unit the
   feeder spends — and let workers block on the queue being full rather than on a
   VRAM estimate they cannot evaluate.

The pool then has one job: never let the queue run dry, never let it grow
unboundedly.

### 4.2 Smart valve — the feeder's control law

Admission already answers *"does this fit?"*. It should answer *"what mix
maximises tokens per second?"*. Three signals, all now traced:

| signal | source | why it matters |
|---|---|---|
| weight-zone bytes | `RegionStats::weight_bytes` | a forward that streams experts costs ~4× one that does not |
| attended context per candidate | `Candidate::context_tokens` | drives the attention tier independently of fed tokens |
| achieved tok/s at width *w* | `wave` records | the objective, measured rather than assumed |

The law to fit has the shape:

```
value(candidate)  = expected tokens contributed this forward
cost(candidate)   = kv_bytes + transient(tokens, context_tokens, width)
constraint        = Σ cost ≤ free_regions·R − reserve(width)
                    AND  weight_bytes ≥ W_min
```

`W_min` is the new term and the one the current design lacks entirely: **a floor
below which admission stops buying KV room**, because past it the marginal
sequence costs more in expert streaming than it returns in batching. §2.1 says
that floor exists — throughput is flat from 4 contexts up, so width past the knee
buys nothing while continuing to cost residency.

The transient model replaces `width × per_seq`, which cannot be right: the traced
split shows attention and FFN moving independently, so the reserve needs
`f(Σ context_tokens, Σ tokens, width)` fitted on the `wave` records.

### 4.3 Simulator

`candle-conversation/src/scheduler/sim/` — a discrete-event model of the
partition and the wave loop, calibrated on the traces, with no GPU.

* **State:** span layout (persist / KV regions / transient tier / weight zone),
  free regions, weight-zone bytes, in-flight work.
* **Cost model:** forward time as a function of `(width, Σ context, weight-zone
  residency)`, fitted to `wave` records; expert streaming charged when the
  working set exceeds the zone.
* **Input:** a queue generator — replay a captured `adm` stream, or synthesise a
  much fatter one than the real producer can currently make.
* **Parameter:** card capacity, so the same controller can be evaluated at 16 GB,
  24 GB and 72 GB.

Validation gate: replaying a captured trace must reproduce the measured decode
width and tok/s within a stated tolerance before any tuning result is believed.

## 4.4 Phase 2/3 result — the pipe is fat, and the feeder is now the constraint

Built: the producer paces on `queued_prefill_tokens` (published on
`AdmissionSection`) between a low and high water mark, bounded by a fixed count
of open conversations. The residency-derived width survives only as a cold-start
bound, consulted until the scheduler publishes its first backlog.

**Queue depth, before and after:**

| | offered per decision |
|---|---|
| VRAM-sized pool | `min=1 med=1 mean=1.0 max=1` |
| queue-paced pool | `min=1 med=12 mean=14.4 p90=36 max=40` |

With a real queue for the first time, the feeder's behaviour is unambiguous:

```
candidates offered    med=12  mean=14.4  p90=36
candidates admitted   med=1   mean=0.8   max=2
passes admitting nothing: 55 of 218 (25%)
prefill width         min=1 med=1 mean=1.0 max=1
```

**It admits 0.8 of 14.4 and never runs prefill more than one sequence wide.**
That is the constraint now, and it is the one this document exists to fix.

Two rules were tried for the ceiling and both fail, in opposite directions:

| ceiling | admitted / offered | prefill width | weight zone (med) |
|---|---|---|---|
| `free + blocked` (shipped) | — | up to 3 | 8,918 MiB |
| `free` alone | 0.8 / 14.4 | 1 | 6,910 MiB |

`free + blocked` over-promises ground the wave itself will need;
`free` alone collapses to zero under exactly the load that most needs batching.
Neither is a control law, which is the finding — the choice cannot be made
statically, and the shipped rule stands until a fitted one replaces it.

**A note on generality.** Nothing in the producer names an expert cache, a
partition or a residency figure: it paces on a token backlog and bounds a
conversation count, both of which mean the same thing on a dense model with room
to spare as on a streaming MoE on a tight card. Only the drain rate differs. The
same must hold of the fitted law — see §6.1.

## 4.5 The learning controller — built, run, and falsified on its lever

A learned controller was built and wired: a hill climb on measured token rate,
with a fast spiral detector for the coupling the design has to survive —

> admit more KV → the weight floor moves down → experts offload → every forward
> streams → throughput falls

— answered asymmetrically (retreat faster than you advance, then hold while
residency recovers), because that descent is fast and the recovery is slow and a
symmetric controller loses ground every cycle. It passed a closed-loop test over
a toy model of exactly that coupling.

**On hardware it cut the budget to the floor and residency kept falling anyway:**

```
move_=Backoff rate=538.8  residency_mib=9054  budget_mib=3072
move_=Backoff rate=23.3   residency_mib=7961  budget_mib=256
move_=Backoff rate=22.7   residency_mib=4933  budget_mib=256
```

Three backoffs, budget 3072 → 256 MiB, and the weight zone continued 9054 →
7961 → 4933 with the budget pinned at its floor.

**The controller was right and its lever was wrong.** The admission budget does
not control resident weights. What consumes the weight zone here is the K/V of
*conversations the producer holds open* — resident before admission ever sees
them, and unaffected by refusing to admit more. Admission governs what enters a
forward; it has no authority over what is already resident.

That is falsifiable and it was falsified, which is the useful outcome. It also
names the right lever: **the producer's open-conversation count**
(`SCAN_MAX_OPEN_CONVERSATIONS`) is what sets resident K/V, so a residency-aware
climb belongs there, not on the admission budget. The controller has been
removed rather than left wired to a lever it does not move.

Two constraints for the rebuild:

* **Match lever to observable.** A climb may only steer a quantity whose changes
  it can actually cause. Rate-vs-budget and residency-vs-open-conversations are
  two different loops and must not share one controller.
* **The spiral guard is still right**, and its asymmetry with it — that part of
  the design survives intact and should be carried over unchanged.

> **Corrected by §4.6.** "What consumes the weight zone is the K/V of
> conversations the producer holds open" is wrong: the K/V arenas hold ~1 GiB
> across the whole run. What consumes it is the **span tenants** — recurrent
> state and the verify stash — which scale on the same axis (open conversations)
> and so made the wrong cause fit the evidence. The conclusion that the lever is
> the producer's open-conversation count survived one more falsification (§4.6)
> for the same reason: right axis, wrong quantity.

## 4.6 The third lever, and the actual root cause

The controller was rebuilt on the lever §4.5 identified — the producer's
open-conversation count — validated against the calibrated model that then
lived in `scheduler/sim.rs` (retired with the controller, §4.11), and run. **It
failed the same way:** weight zone 6,318 → 1,417 MiB, 237 directories refused,
prefill still one sequence wide.

Three levers, three failures, so the premise was wrong. The measurement that
seemed to explain it:

```text
kv-regions: live=642 peak=642 free=0 of 642 (10272MiB) | weights=1417MiB
kv-pool classes: 5a/80MiB 3a/48MiB 1a/16MiB 46a/736MiB 29a/464MiB
```

**642 regions claimed against 84 arenas**, `total` growing `of 336` → `of 642`,
`free` at 0 throughout. That reads as a leak, and this document said so: a
reclamation problem, ~8.9 GiB unaccounted, no controller can steer a quantity
nothing gives back.

**That reading was wrong in all three of its parts.** Instrumenting the
accounting — which is what the section correctly said to do — refuted it.

### 4.6.1 Two counters that read as their opposite

`free` is the free-**list** length: regions handed back and awaiting reuse. It is
not headroom, which is `total − live`. So `free=0` means "nothing returned yet",
not "exhausted", and it does reach 21, 75 and 156 later in the same runs.
Reclamation was working the whole time.

`total` is the KV zone's **capacity**, `(weight_floor − region_base)/REGION_BYTES`
— a figure derived from the boundary, not a count of anything claimed. It grows
*because* the weight zone yields. Watching it climb while `weights` falls is the
elastic partition doing its job, not KV leaking.

### 4.6.2 Regions have two kinds of tenant; the report saw one

A region is claimed either by `claim_slab`, which makes an `Arena` and appears in
`kv-pool classes`, or by `claim_span_region` / `SpanClaims::claim`, which makes a
`SpanRegion` and appeared **nowhere**. The class line also read only
`backings.first()`, so it described one pool's share of a process-wide
reservation with no sign that it was doing so.

With both fixed (`all_pools_gpu_arenas`, `span_tenant_counts`), the run reports
`all 1 pools` — a single `BackingInner`, its arena count exact
(`registered == stored`, 43/43 samples). The remainder is held by three span
tenants, all legitimate, none of them ever leaked:

| tenant | where | scales with |
|---|---|---|
| `Gallery` | `provenance/gallery_arena/storage.rs` | the **ingested corpus** |
| `RecurrentState` | `delta_net/state_store.rs` | concurrent sequences |
| `VerifyStash` | `qwen35/spec.rs` | concurrent sequences × draft budget |

All three exist only on DeltaNet-lineage models, which is why this never
appeared on a non-DeltaNet target — and why the throughput law may not carry
across models unchanged (§6.1).

### 4.6.3 The actual root cause

There is no leak. Width is **paid for in weight residency**: the two
sequence-scaled tenants take regions, every region pushes `weight_floor` down,
and the weight zone's collapse from 10,398 to 1,417 MiB is what puts a resident
model onto the streaming path. The fat pipe bought queue depth and paid for it
here.

That is exactly the local minimum the design was warned about — *weights
offload, so it slows down, so the controller scales up, so it slows down more* —
reached through a tenant nothing was accounting for. And it is why all three
levers failed: each regulated the symptom. The lever that matters sets width
against residency, or bounds the span tenants directly.

**A process note, since it cost most of a day.** Each lever took a ten-minute
hardware run to disprove, and the simulator was built after two of them.
Building it first would have caught none — the model shared the producer's wrong
premise. Instrument the accounting before tuning the controller; and when the
accounting itself is the suspect, check what each counter *means* before
building a theory on its value. Two of the three wrong conclusions here came
from reading a field's name rather than its definition.

## 4.7 The re-fit model, and a controller that prices a conversation

### 4.7.1 What the measurements gave

Two quantities, both measured on a repo-map ingest of this tree against
Qwen3.6-35B-A3B (they were pinned by the simulator's tests until it was retired
in §4.11; the measurements stand):

**Displacement is exactly one-for-one, and the span is fixed.** Across 57 wave
records, `total × 16 MiB + weight_bytes` held at **11,690 MiB with a spread of
7 MiB** — from 81 regions against 10,398 MiB of weights all the way to 642
against 1,417. There is no slack in the partition: a region taken is a region
the weight zone does not have.

**A sequence costs ~8 regions.** One recurrent state store claims a bump's worth
across the model's DeltaNet layers, ~126 MiB. The peak observed was 576 regions,
about 72 concurrent stores against a producer holding up to 64 conversations
open with roughly three forks per turn.

Eight regions is **128 MiB of weight zone per sequence**. Widening from 8 to 64
sequences therefore spends ~7 GiB of residency — on a 16 GB card, all of it.
That is the whole of §4.6.3 in one number.

### 4.7.2 Why the cost is learned rather than configured

`Machine::span_regions_per_conversation` is a field and the controller's
`cost_regions` is learned online, because the tenants are per-model: a dense
checkpoint has none, its counts read zero, the term vanishes, and no law may
throttle it. `a_model_without_span_tenants_never_loses_residency` and
`a_model_without_span_tenants_learns_a_zero_cost_and_no_ceiling` pin both halves.
The quantity that makes the law necessary is the same one that switches it off,
so it cannot be forgotten in one direction only.

The learned cost is **rise-fast, decay-slow**, matching the retreat's asymmetry
and for the same reason: under-estimating the price is what walks a climb into
the spiral, so evidence that a conversation is expensive is taken at once and
evidence that it is cheap is taken slowly.

### 4.7.3 The ceiling, and what it is not for

The climb now stops at `residency_ceiling()` — the most conversations the
learned price says fit above a floor set at 75% of the best residency this
machine has actually reached. Not a byte budget: the mark is observed, so a card
that never concedes has a floor it cannot breach and the ceiling never binds.

**The ceiling does not defend residency, and a test asserting that it did failed
correctly.** Driven against the calibrated machine, the reactive guard alone ends
with *more* weight zone (10,359 vs 9,728 MiB) and less throughput — because its
only answer to a spiral is to halve the target and hold. It protects residency by
surrendering the width residency was for. Knowing the price instead buys the most
width the machine can afford and stops, so the comparison that counts is the
operating point, and `the_ceiling_finds_a_better_operating_point_than_the_guard_alone`
asserts tokens per second with a residency floor as a side condition.

This is the answer to the local-minimum trap: a purely reactive loop cannot
avoid a descent it can only detect afterwards, and on this engine the descent is
fast and the recovery slow. Pricing the trade converts the loop from reactive to
predictive while leaving the spiral guard in place for everything the price does
not predict.

## 4.8 The fourth falsification: the lever has no authority over the tenants

The capped controller was run on hardware. **The weight zone still collapsed to
1,417 MiB**, and the trace says plainly why:

```text
09:22:38  target moved from=4 to=2   residency_mib=6884  span_regions=162  ceiling=64
09:22:48  span 271
09:23:02  span 288
   peak   span 577
```

The producer's open-conversation target was at **2** — its floor — and the span
tenants went on growing from 162 to 577 regions afterwards. The cap never bound
(`ceiling=64` throughout) because it never needed to: the quantity it caps was
not what was taking the ground.

**§4.5's own rule, unapplied.** That section ended with "match lever to
observable: a climb may only steer a quantity whose changes it can actually
cause." It was applied to rate-vs-budget and then never checked for
span-regions-vs-open-conversations, which is the pair this phase was built on.
The check is one line of trace and would have cost ten minutes before any of
the three controller designs in §4.7.

**What this means.** Recurrent state stores are created per *sequence in the
wave engine*, and per fork (`state_store.rs`, the fork path, ~3 per turn) — not
per conversation the producer holds open. Closing conversations neither prevents
them nor returns them. So the lever for phase 9 is the wave engine's concurrent
sequence count, or the store's own lifetime, and **not** anything the producer
controls.

A second dead signal showed up in the same trace: `rate=0.0` on every feeder
window. `completed_tokens` is `PREFILL_OK_TOKENS`, which advances only when a
prefill lands, so a window with no completed prefill reports zero throughput and
the spiral guard reads it as "slower". Every retreat above fired on that.

**Standing conclusion.** Four levers have now been falsified on hardware —
admission budget (§4.5), open-conversation count (§4.6), a learned climb on it
(§4.6), and a residency cap on it (here). Three of the four failed the same way:
the controller was sound and its lever did not move the quantity. Before a fifth
is built, the lever must be shown to have authority — vary it, and measure that
the tenants respond — and that demonstration belongs in this document as a
measurement, not an argument.

## 4.9 The real lever: recurrent stores

§4.8 said the next step was to show a lever has authority before building on it.
This is that measurement.

### 4.9.1 The identity

`recurrent_memory_count()` — the model's own leak gauge, which existed but was
reported nowhere — printed beside the span-tenant count over a full ingest:

| stores | 1 | 2 | 13 | 17 | 30 | 48 | 60 |
|---|---|---|---|---|---|---|---|
| span regions | 9 | 18 | 117 | 153 | 270 | 432 | 540 |

**`regions = 9 × stores`, exactly, in every sample.** Not a fit — an identity.
Each store claims one bump across the model's recurrent layers, so the ratio is
integral and constant: **144 MiB per store** on Qwen3.6-35B. The stores also
release cleanly (17 → 2 as work drained), so nothing here leaks.

### 4.9.2 Why every earlier lever missed

The recurrent map is keyed by **sequence id** and owned by the model. A scan
directory mints conversations; a conversation carves a view per turn
(`fork_recurrent` runs at *every* turn's view carve, not only on an explicit
fork); each view is a sequence with its own store. So the producer's
conversation count reaches the stores through a multiplier nothing bounds —
measured, 60 stores stood against a producer target of 2.

That is the whole of §4.5–§4.8 in one sentence: four controllers steered a
quantity upstream of the one that spends the ground, separated by a gain nobody
had measured.

### 4.9.3 The gate

`stores_are_at_their_ceiling()` prices the stores directly:

```text
span   = weight_bytes + span_regions × 16 MiB     (one fixed span, §4.7.1)
floor  = span × 0.68
max    = (span − floor) / (regions_per_store × 16 MiB)
gate   = stores >= max
```

Three properties worth stating, because the earlier designs each failed one:

* **Nothing is divided by the quantity under control.** `regions_per_store` is
  an observed integer ratio, not headroom over a moving target (§4.7's pricing
  failure).
* **Nothing is extrapolated from a single event.** The bound is recomputed from
  what is held right now, so lag cannot poison it (§4.7's bracket failure).
* **It cannot throttle without a price.** No stores, or no headroom to measure,
  returns `None` and the gate is open — so a model whose sequences carry no
  recurrent state is never touched, by measurement rather than by a model check
  (§6.1).

## 4.10 The queue holds live sequences, which is why no valve could work

The throttle was moved to admission — `plan_admission`'s `max_count`, bounded by
`store_room()` — and run. **It did not hold either:** peak 64 stores, weight zone
still 1,417 MiB.

The trace says why, in one coincidence that is not a coincidence:

```text
queue backlog   57 .. 67
recurrent stores      peak 64
```

`PrefillWork` carries a `sequence_id`, and the slot is allocated **before** the
work is pushed (`scheduler/mod.rs`, `sequence_id: slot`). A queued turn is
already a live sequence, so it already holds its 144 MiB recurrent store. The
ground is claimed at *submit* time.

**That is why every lever failed, this one included.** All of them — admission
budget, open-conversation count, a learned climb, a residency cap, and finally
admission width — sit *downstream of the allocation*. None of them can release
what queuing has already taken. The five failures are one failure.

It also prices the pipe: queue depth ~60 × 144 MiB ≈ **8.6 GiB**, which is the
entire collapse. The fat-pipe design is not wrong — a fat pipe is free only when
a queued item is *cheap*, and right now a queued item is the most expensive
object in the system.

### 4.10.1 What has to change

Queuing must stop claiming ground: **carve the view and create the recurrent
store at admission, not at submit.** A queued turn becomes a work descriptor;
the slot and store are claimed when the valve admits it. Stores then track
in-flight width — the quantity `store_room()` already meters — and the throttle
built in §4.9 starts guarding a door the work has not already walked through.

Two routes, and they are not exclusive:

* **Lazy creation.** Ingest conversations are new, so their recurrent state is
  fresh and deferring costs nothing. Small, and it targets the measured 8.6 GiB
  directly.
* **Idle eviction.** For a resumed conversation with real state, the
  `export` / `import` / `restore_recurrent` machinery already exists, so an idle
  store can go to host and come back on admission. The general answer, and the
  larger change.

The cost of the move is that a queued item stops being tied to a slot, which the
seal and fork paths currently assume.

## 4.11 What was actually wrong, and the design as built

§4.10's route was taken — idle recurrent stores are parked to pageable host RAM
and come back on admission (`qwen35/batched.rs`, `parked`) — and it worked for
what it targeted: recurrent state fell from 5,120 MiB to 640 MiB, span refusals
from 140 to 0.

> **Superseded — host parking of recurrent state has been removed.** The `parked`
> map, the per-wave park sweep, the unpark-on-materialise branch and the host
> arms of fork/move/export are gone; `restore_recurrent` now creates a device
> store and imports into it. What bounds residency instead is the turn seal:
> `evict_recurrent` drops the device copy once the substrate snapshot is durable,
> and the next turn restores from that snapshot, so a conversation between turns
> holds no recurrent state at all. The substrate is now the only place recurrent
> state is recovered from, rather than a host cache shadowing it.
>
> This also collapsed `has_recurrent` into `recurrent_resident`: the two only
> differed for parked sequences, so with nothing parked they cannot disagree.
> Everything below in §4.10/§4.11 describing the parked map is history, kept for
> the measurements. The residency numbers it reports still stand; the mechanism
> that produced them does not.
>
> The case seal-eviction does not cover is a sequence idle *within* a turn — one
> queued behind others — which stays resident. Measured before: a backlog of
> 57–67 turns held 64 stores. The write-buffer release still sheds the ping-pong
> half of those. The ingest still did not complete. One run finished 14 of 353
directories; every run after it, each fixing the loudest line in its
predecessor's log, finished 0. The run that ended this section (P) was read
whole rather than for its loudest line, and it said three things.

### 4.11.1 The decode side was gated by the weight point, and that is the wedge

The fill (§4.9's "real allocation" loop) alternated decode and prefill and
stopped the moment the weight zone stood at or below `HOLD × high-water`,
admitting only its opening pair unconditionally. Run P:

```text
wave fills                          ~900
  decodes=1 prefills=1 stopped_on_weights=true    547
weight zone, first fill / last fill  8,639 MiB / 4,952 MiB   (hold point 7,818)
```

Once the zone was under the mark — which K/V from 95 open conversations put it
under, permanently — every fill stopped after one decode and one prefill. One
decode a wave across 95 conversations each needing hundreds of steps is a rate
of completion indistinguishable from zero, and completion is the only thing that
gives ground back. The zone could not recover because recovery needs decodes and
decodes were what the mark withheld. **Gating decodes on the weight point is
self-defeating**: a decode row's K/V is already claimed and its state already
resident, so admitting it costs the partition nothing it is not already paying,
and refusing it is what keeps the memory open.

### 4.11.2 Tier refusals were the co-batched wave, not the budget

Every one of run P's 15 tier refusals read identically:

```text
tier needs 6,320 MiB   gap between arena frontier and weight floor 6,070 MiB
```

A 6.3 GiB tier against a `prefill_width_cap` priced to a 912 MiB guarantee. The
cap was correct and irrelevant: the engine's slab packer bounds only a *pure*
prefill wave (`drive_wave`'s slicing branch requires no decode rows), and a wave
carrying a decode row takes its prefill group whole. The scheduler admitted up
to `MAX_PREFILL_WIDTH` prefills at up to ~6k tokens each into a wave that
nothing bounded. Run L had a scheduler-side tier bound and finished 14; run N
removed it as redundant and finished 0. It was never redundant — it was the only
bound the co-batched wave had.

### 4.11.3 The producer's limit was not a limit

The learned open-conversation target (§4.7) oscillated between 3 and 20 across
150 epoch decisions in run P, and 95 conversations were open against a target of
7. `SCAN_POOL_WAIT_CAP` — twenty seconds, after which a worker took its slot
regardless — turned every hold into a delayed admission. The pool was the
thread count with a twenty-second ramp.

### 4.11.4 The design

Five regulators shared one pool of bytes — the learned target, the queue
watermark, the cold-start residency estimate, the interleave's weight hold, the
growth policy's spare — on different time scales, and fixing any one moved the
failure to another. The rebuilt shape has the engine answer one question and the
producer read the answer.

**The fill is decode-first, and decodes are gated only by the real allocators**
(`scheduler/interleave.rs`). Every eligible decode is offered; each admission
claims its K/V chunk (`ensure_capacity`) *and* its per-sequence model state
(`ManagedBatchedModel::admit_recurrent`, which on the DeltaNet lineage is
`HybridBatched::reserve_recurrent` — a quiet probe that unparks or creates the
store and takes its write buffers, returning `false` when the span refuses). A
refusal skips that sequence and is counted; the offers rotate from the last
admitted id so a refused tail is offered first next wave. The wave then runs
**exactly the admitted set** (`wave_decode_set`), not a prefix of a list. The
weight zone is not consulted for decodes.

**Prefill takes what is left.** The opening prefill is unconditional — a wave
that never starts new work makes no progress once the decodes drain — and every
one after it is admitted only while the real weight zone stands above the hold
point. Each prefill also claims its recurrent state, and **the co-batched wave is
bounded in tokens**: `prefill_width_cap(act_dtype, head_rows)` now takes the
rows already in the wave (decodes, and their verify blocks at `1 + draft` rows
each) and returns what the tier has left; the fill subtracts the prefills already
in flight and refuses a prefill that would exceed it, unless it is the only one
(a lone oversized prefill still travels, as in the slab packer). A refusal here
is a quiet `false`, never a wave-time fault. The tier budget the cap prices
against is the **measured gap** between the arena frontier and the weight floor
less a four-region margin, never below the 912 MiB guarantee — the guarantee
alone prices to ~1,000 tokens on this geometry (run R sliced two-sequence
calibration waves into two forwards on it), where the gap is several GiB and is
what the placement actually measures against. The fill's claims are already in
the frontier the gap reads, and the growth policy withholds the last tier's
ground from the weight side, so the margin only has to cover an arena the
compressor creates between forwards.

**The engine publishes two numbers** in the memory report's admission section:
`decode_carried` (decodes the last fill admitted) and `decode_starved` (eligible
decodes it could not). Starved is the real allocation's answer that the device is
full — a decode is refused only when there is no ground for it.

**The producer reads them and models nothing** (`repo_scan::gate_decision`).
A worker opens a conversation when `live == 0`, or when a fresh report shows
`decode_starved == 0` and the prefill backlog under the watermark (with the
existing low/high hysteresis). Otherwise it holds, and **a hold ends only when
the engine says so** — the wait cap is gone. `FeederControl`, the simulator it
was fitted against, the residency estimate (`scan_width`,
`per_conversation_kv`, the K/V baseline anchor) and the process-wide live gauge
are deleted; nothing in `zend` names VRAM, expert residency or a partition.

**Two loops close without a controller.** Under pressure the weight zone drops
below the hold point, the prefill side narrows to one a wave, the backlog grows,
the producer holds, the decodes — every one of them, every wave — finish, their
ground comes back, the zone recovers, prefill widens. And if resident
conversations cannot be stepped, `decode_starved` goes non-zero, the producer
holds, and the allocators' refusals, not an estimate, set the width.

**The hold point is measured when the engine is idle, from the span identity.**
The first build of this seeded the mark from the weight zone at model load and
held it for the process. Run Q showed that mark was stale within two minutes:
the tool catalog's permanent KV took ~1.2 GiB the zone could never regain, so
calibration — a phase with no decodes at all — ran with the zone 2% under the
mark and the prefill side at one a wave (135 of 200 fills). Now
`reseed_achievable_weight` runs at entry and whenever the engine falls idle,
when everything resident is by definition permanent, and computes
`span − live_kv − MIN_ELASTIC_RESERVE` rather than reading the zone, which lags
because the boundary only grows back between forwards. `HOLD` is 0.70 of that:
a wave's own working set is 1–2 GiB and four fifths left no room for it.

**Run T (2026-09-04) found three more things, each fixed in place.**
(1) *The burst.* All 96 workers passed the gate in the first second of the
pass: the report's backlog is a wave behind, so every worker read "empty". The
report now carries `open_slots`, the pool records the engine's count when the
pass opens, and a worker holds (`Hold::Unreflected`) while more than
`SCAN_OPEN_SLACK` (4) of this pool's openings are not yet in the report. The
pool still opens as fast as the engine reflects it; it just cannot run a wave
ahead of it blind. (2) *The tier budget was priced once per fill.* Each admitted
prefill then claimed its KV chunks and its 144 MiB recurrent store, moving the
frontier the budget had been read from; 29 identical refusals read `needs
1,264 MiB against a 1,206 MiB gap`, four regions short, and each refusal failed
its wave's sequences. The budget is now re-read at every offer. (3) *The hold
gated only prefill, the cheapest tenant.* Fifteen resident decode stores and the
tier took the zone 2 GiB under the 0.70 mark regardless, so the pipeline ran
prefill-bound at one admission a wave with 88 queued while the residency the
mark defended was already gone, and by the end of the run the zone stood at
its 1,417 MiB floor with 559 of 642 regions live. `HOLD` is 0.45; with the
burst bounded, far fewer conversations hold KV at once and the zone has
somewhere to stand.

**Run U (burst bounded, HOLD 0.45) named the tenant.** With the herd gone the
queue was short, no tier was refused, and completions began three minutes into
the pool — but by twelve minutes it had settled at one every two minutes.
Twenty-four resident DeltaNet stores at 144 MiB each had taken the zone to
2.2 GiB, under any hold, and prefill was back to one per fill. The store is the
tenant that costs, and it was the one tenant nothing gated. **A decode whose
state is already resident always runs; one whose state would have to be made
resident runs only while the zone stands above the hold** (`WaveFill::admit`,
`Kind::Decode`). That bounds the stores standing at once to what the zone
affords without touching the resident ones — the distinction §4.11.1's wedge
lacked, where every decode was gated on the absolute level and nothing could
finish. A deferred decode counts as starved, so the producer holds while the
device is full by the hold's own definition. Run U also reproduced the
recurrent-resume race (a sealed snapshot indexed before its bytes reached the
file, read back as zero bytes by the next turn); `LogFile::read_at` now
flushes the staging buffer when asked for a record only it holds.

**Run V (new-store gate) found the lone-prefill exemption.** A prefill wider
than the tier was allowed through when it was the only prefill — the slab
packer's own rule, so a lone oversized sequence is refused by the placement
rather than starved by the fill. Beside decode rows the wave is not lone: one
4,000-token read turn admitted on that exemption into a wave of eight decodes
built a 4,080 MiB tier against a 3,542 MiB gap, and the 21 refusals that
followed failed 18 directories. The exemption now holds only when the wave
carries nothing else.

**Run W showed draining the decodes for a blocked head does not work.** A
2,493-token head sat behind three remaining decodes for five minutes with 57
queued and 221 hold events: the weight side could not grow back past
scattered live KV regions (`requested boundary move failed … region N is
live`, ~400 times), so the room the drain was waiting for never came. The fill
now **yields** instead (`Ground::yield_decodes`): a head the tier turns away
beside the decodes runs alone in a decode-less wave — the slab path, which
buys its ground at placement — and the resident decodes sit that one wave out
and are offered again next fill. The transient tier on this hybrid costs on
the order of a megabyte per prefill row, so a mid-sized read turn wants more
tier than the zone above the hold plus the frontier gap can supply beside a
decode set; that per-row cost is the engine-side ceiling left.

**Run X: the tier's ground is bought at fill time, or not counted.** Two
more refusals (`needs 4,064 MiB against a 3,830 MiB gap`, the weight side
"could not concede") showed that pricing the zone's standing above the hold
into the budget and leaving the purchase to the placement does not work: the
placement runs inside the forward, where the boundary may not move. The fill
runs between forwards, where it may, so `WaveFill::tier_tokens_left` now
prices the head's tier against the measured gap, and when it would exceed the
gap by no more than the zone stands above the hold, asks the weight side for
exactly that shortfall (`request_kv_ground`) and re-measures. What the weight
side does not concede is not a budget. The yield also fired for a 65-token
head held by prefills already in flight, idling eighteen decodes for a wave
in which nothing could land; it now yields only when the head would fit alone.

**Run Y: the purchase must be partial, and unbounded for a lone head.** With
the purchase only made when the whole shortfall fit under the hold, a
2,855-token head whose tier wanted 3 GiB against a 1 GiB gap and a 2.4 GiB
standing above the hold bought nothing and held the queue at 57 with
prefill at zero. The fill now asks for `min(shortfall, above_hold)` every
time and re-measures; and for an item that would be the wave's only prefill
it asks for the whole shortfall — the hold governs how much runs at once, not
whether a single item may ever run — and the weight side concedes down to
its own floor or refuses.

**Run Z: the concession is all-or-nothing, and the queue cannot be strictly
FIFO.** A 70-region ask for a lone head came back as nothing eighteen times
while eight-region asks were granted beside it, so an ask the weight side
refuses is now halved and repeated down to one region. And with the head
unable to run, strict FIFO held 57 items behind it and the prefill side at
zero. The prefill arm now passes over an item the tier cannot hold for the
next that fits, with the head held in place by the decode arm — no new store
is admitted while the head is blocked, so the resident decodes finish and the
room the head needs is made while the items behind it keep the pipeline busy.

**Run AA: a refused tier is a wave composed too wide, never a failure.** One
wave priced 26 MiB over a 4,054 MiB gap — two regions, the compressor's
arenas in the window between fill and placement plus the placement's own
rounding — was refused, and the refusal failed 18 directories. Nothing had
run: the placement fails before any compute and the KV side rolls back. The
scheduler now recognises the refusal (`is_tier_refusal`), requeues the wave's
unstarted prefills at the front in order, doubles the margin the fill holds
back from the tier budget (four regions, doubling to a cap of 64), and lets
the next fill compose the wave narrower. The same run then idled with 23
queued: the one eligible new decode was deferred by a hold the zone undershot
by 16 MiB, and with no decode running nothing could free anything. The
opening decode is now unconditional, as the opening prefill is.

**Run AB: the hold was watching a number that does not move.** Twenty-five
stores were admitted through a hold that read the zone above the mark every
time, because the hold compared the zone's *extent*, and a region claimed
from the free list does not move the boundary. The boundary caught up when
the tier bought ground and the zone fell 2 GiB under the mark in one step.
The hold, the purchase headroom and the fill's weight reading now use the
**effective** zone — span less live regions less the reserve, the idle
identity evaluated on the live count — which charges every claim the instant
it is made.

**Run AC: the per-fill opening prefill was the leak to the floor.** With the
hold finally watching a live number, 24 stores were still resident within
minutes and the extent reached 1,417 MiB again. The "opening prefill is
unconditional" rule fired on every fill, and fills run about thirty times a
minute (loop top and mid-wave), so thirty new sequences a minute entered under
the hold, each bringing its 144 MiB store and later decoding as a resident
that no gate touches. The guarantee is that **the wave is never empty**, not
that every fill admits a prefill: with no decode running the first prefill is
unconditional, and with decodes running the prefill side may admit nothing
under the mark — the decodes finishing are what brings the zone back.

**Run AD: the purchase never goes below the hold, and a head that fits
nowhere fails once.** Asking the weight side for a lone head's whole shortfall
drove it to its 1,417 MiB floor, where every remaining expert slot is pinned by
the wave that needs it and forwards fail with `Expert cache full, cannot
evict (all pinned)`. And a head still too wide after the purchase was admitted
on the alone exemption, refused by the placement, requeued, and admitted again
— 43 refusals of one 4,144 MiB tier. The purchase is now capped at the zone's
standing above the hold, and an item that does not fit even alone with all of
that bought is failed explicitly, once, with the numbers
(`prefill_too_wide`): the model's per-row tier cost exceeds what this
partition can offer it, which is an engine-side fact the scheduler cannot buy
its way past.

**Run AE: the first run in which the shape works.** 18 directories in the
first eight minutes of the pool, 2–6 a minute, zero failures, zero refusals
that cost anything, the producer paced on `decode_starved` with the queue at
one or two. It then blocked on a 4,549-token head whose tier wanted 4 GiB
against a zero frontier gap (384 of 434 regions live, the free 32 scattered):
the head held new stores while the 13 resident decodes drained, and would have
failed as too wide only once they had. An item that would not fit with
*nothing* else in the wave and everything above the hold bought now fails at
its first hold, so the pipeline never drains for a head it cannot run.

**Run AF: the drafter's forward refuses too.** Eleven refusals failed ten
directories through `decode draft failed`: the speculative drafter runs a
forward of its own and places a tier of its own, and that error path still
failed the sequences. Every forward error site in the decode step — draft,
verify setup, the forward, verify readback — now takes a refused tier as the
wave-too-wide answer and requeues.

**Run AG: the burst bound's base went stale.** The pool compared its own open
count against `open_slots − base`, with `base` the engine's slot count when the
pass opened — taken while the calibration conversations were still open, so
once they closed the difference read zero and the pool was held to about five
conversations for the rest of the run, at 1–2 completions a minute with the
zone at 8.6 GiB. The bound now compares against the engine's total open slots;
slots this pool did not open only loosen it, which is the safe direction.

**A starved fill runs no forward.** `batch_decode_step` returns `false` when
every eligible decode was refused; the decode quantum yields for 20 ms and
returns to the outer loop's relief and promotion passes rather than forcing a
refused sequence through to a loud wave-time refusal that would fail the wave.

**Run AH: the tier's per-row cost is the ceiling, and whole-turn prefill is
what pays it.** With the burst bound loosened the run carried 20–22 decodes a
wave for thirteen minutes of pool and ingested nothing: eight items failed as
too wide (2,096–4,615 tokens against alone budgets of 1,346–2,056), and every
fill was decode-only because no queued turn's tier fitted beside the decodes.
The tier on the 35B hybrid prices at roughly a MiB per prefill row, so a turn
is a tier of its own length in MiB, and the fill was pricing a turn's *whole*
token set into one wave — the one shape a 16 GiB partition cannot place beside
a working decode set, however the ground is bought.

The dialogue prefill now rides the wave the way a section ingest already
did: in chunks. `WaveMember::Prefill` carries an `advance`, and a group
covers `[offset, offset + advance)` of each prefill's tokens; the head commits
the chunk (advance the sequence, record the slot tokens, `PrefillProgress`)
and only the chunk that reaches the end of the tokens takes the logits as the
turn's first-token distribution and emits the staged projections. The fill
prices an item at its next chunk — `min(remaining, max_prefill_pass_tokens)`,
and it is offered at least `PREFILL_MIN_ADVANCE` (128) rows or all it has
left — and the group formation hands the wave's real prefill room
(`prefill_width_cap` against the budget the fill set on the session) out in
admission order, each member taking the smaller of its remaining tokens, the
chunk, and the rows still unassigned; the first member always gets its least
chunk so a wave carrying a prefill always advances one. In-flight prefills
count for their next chunk, not their remainder, so a long turn no longer
blocks admissions behind it for its whole length. "Too wide" now means the
partition cannot place a tier for one 128-row chunk with nothing else in the
wave, which is a partition that cannot run a prefill row at all.

**Run AI: the resume path put the stores on the device.** Chunked prefill
ran the calibration at 6–9 prefills a fill, then the pool opened and the run
ingested nothing: the weight zone stood at its 1,417 MiB floor from the first
pool minute, `kv-regions` reported **43 stores (5,200 MiB)** for 14 decodes
carried, and one wave was refused 1,641 times with nothing requeued. Two
causes and a loop:

- Every pool conversation forks the tool-catalog root and *restores* the
  recurrent snapshot at open (`restore_recurrent_state` at `create_sequence`),
  and `restore_recurrent` created a device store to scatter it into. That is a
  144 MiB store per open conversation, taken at open rather than at admission,
  on the one path the parking never covered — the fill's new-store gate never
  saw them because they were resident before any fill ran. The restore now
  lands in the parked host map, validated against the model's layout at that
  moment (`validate_snapshot`), and is scattered into a store by the admission
  probe when the sequence is admitted, seeded as before. Fork and move handle
  a parked parent or child on the host; export answers from the parked copy;
  `has_recurrent` (state anywhere — the fork/move/restore question) is now
  distinct from `recurrent_resident` (on the device — the fill's question).
  *(Superseded: the parked map is gone, `restore_recurrent` creates the store
  directly, and `has_recurrent` has been collapsed into `recurrent_resident` —
  see the note at the head of §4.11. `validate_snapshot` still runs first, so a
  foreign or torn snapshot is refused before any device memory is spent.)*
- With the zone at the floor and live regions scattered to the top of the
  span, the frontier gap was 150 MiB, and `prefill_width_cap` floored the
  budget at the 912 MiB guarantee — ground the placement cannot use while
  arenas stand in it — so the fill and the group priced ~1,000 rows into a
  wave the placement refused. The cap prices the budget it is given; a budget
  that holds nothing prices to one row.
- The refused wave's prefills had started (chunks committed), so nothing was
  requeued and the same group re-formed at the same width every wave. A
  refusal at the widest margin now fails the started prefills with the
  numbers, and a wave with decode rows and no room for a least chunk carries
  no prefill member — the decodes run and free the ground.

**Run AJ: the queue's length starved the chains.** With restores parked the
store count tracked the carried width exactly (19–23 stores for 18–21
decodes), the zone sat at the hold rather than the floor, zero refusals, zero
failures — and still nothing completed in 16 minutes and 191 waves, where run
AE completed its first directory 83 waves after the pool opened. The pool had
opened 114 conversations: none of the producer's holds bit, because the engine
was carrying every resident decode (`decode_starved` 0) and the 77 queued
prefills were summary and tool-result prompts of a few hundred tokens each,
20,717 tokens against a 24,576 mark. A directory ingest is a *chain* — decode,
tool call, prefill the result, decode again — and each chain's next turn
entered a FIFO queue behind 76 fresh first turns, admitted one or two a fill
as the zone allowed. No chain reached its next turn; the 20 decodes carried
were the first turns of conversations whose chains would then wait in the same
queue. Two changes, both general:

- The producer holds on the queue's **length** as well as its width:
  `Hold::Queued` while queued prefills exceed the carried decode width plus a
  wave's worth (`SCAN_QUEUE_SLACK` = 8), draining to half that. A queue longer
  than the width the engine steps is not a deeper choice for the feeder; it
  is K/V waiting on the device.
- The scheduler admits **continuations before first turns**: at each fill the
  prefill queue is stably partitioned so turns on sequences that already hold
  K/V (a chain between tool call and result, a dialogue mid-reply) are offered
  ahead of first turns on empty sequences. Finishing what is open is what
  frees the ground for what is not.

**Run AK: the tier bought the zone to its floor, and the hold could not see
it.** The producer was bounded (25 open at pool start, 59 at most, holds
biting), stores tracked width, and the run still ingested nothing: the weight
zone stood at 1,417 MiB — its floor — for the first minutes of the pool with
3.6 GiB of frontier gap free above it, decodes at six seconds a step, 111 tier
refusals in one second, and two 200-token prompts failed as "too wide even
alone". The calibration fill had admitted 15 prefills and bought 3 GiB of tier
ground in four asks, each ask judged against `effective_weight_zone_bytes`
(span − live regions − reserve = 6,623 MiB, far above the 3,554 hold) while
the extent it was actually spending stood at 1,487. The two measures each
miss a tenant: the extent misses stores and chunks claimed from the free list;
the live count misses the tier, which is not a live region. Fixes:

- The hold and the purchase cap watch `defended_weight_bytes`, the **minimum**
  of the extent and the live-region view, so both tenants are charged.
- When the extent stands under the hold, the frontier gap is not the tier's:
  the tier budget is the gap less what the weight side is owed back to the
  hold, so the weight side grows back into it between forwards instead of
  being bought out of it every wave.
- "Too wide even alone" is gone. It was measured true for a 213-token prompt
  in the minute the zone sat on its floor and false a minute later; an item
  the tier cannot take waits, and a partition that genuinely cannot place a
  chunk fails at the wave through the refusal path's own limit.
- A refusal's text is logged with the requeue, so the placement's own numbers
  are in the record rather than only the fill's.

**Run AL: the first completions, and the open count that throttled them.**
Three directories in the pool's first thirteen minutes, zero failures, zero
refusals, the zone never below the hold — and the hold is why only three: the
pool had 83 conversations open for 14 carried and 36 queued, and the 33 in
neither state (decode done, tool result being built, prefill in flight,
closing) held their K/V like the rest. 267 regions of K/V for 83 conversations
put the live-region view of the zone under the 4.2 GiB hold permanently, so
the fill admitted no prefill for minutes at a stretch and every chain's next
turn waited for a decode to finish and free a store. The burst path was the
"unreflected" slack: a slot shows in the report the moment it is created, so
the slack of four bounded nothing but the rate — four per report interval, 80
a minute.

The producer now bounds what it holds **open** against the carried width:
`Hold::Open` while this pool's live count reaches the carried decodes plus
twice the queue's worth (`SCAN_OPEN_WAVES` = 16), draining to the
carried width plus one queue's worth. It is not a width — it grows as the
engine carries more — and it exists because the engine cannot yet put a live
sequence's idle K/V off the device (§4.11.5); when it can, this bound is the
one to retire.

**Run AM: 22 directories in 26 minutes, zero failures, zero refusals** — the
first run of the day that ingests steadily rather than in a burst and then
blocks. The pool held 31–32 conversations open for 15–25 carried, the queue at
8–16, the zone at the hold throughout. Per-wave wall-clock in the pool phase:
decode 2,945 ms, drain 477 ms (p50 0, p90 818, max 22,450), promote 177 ms,
unaccounted 107 ms. Two ceilings, both measured in this run:

- **Boundary churn.** 440 waves, 440 weight-side grows, 764 concessions,
  90,017 expert slots evicted — every wave the weight side took the free gap
  and the next fill bought it back. The cause is the tier margin: a refusal
  doubles it and nothing shrank it, so from the calibration minute on it stood
  at its 1 GiB cap, and every fill priced the tier a gigabyte under the gap,
  bought that gigabyte, and placed a tier that did not use it. The margin now
  decays one region per fill. That alone did not end the churn (run AN's pool
  opened at two concessions a wave with the margin at its base): the growth
  policy leaves the last placed tier's ground alone — `Occupancy::tier_planned`
  — but prices its grant on `free_below_ceiling`, every free region wherever
  it lies, while the weight side takes ground at the frontier and the tier is
  placed against that same frontier. Free regions scattered below live ones
  are spare to the policy and useless to the tier. `spare_regions` now caps the
  grant at the frontier gap less the slack and the tier term, whatever the
  free list says. Run AO then showed the *last* tier is the wrong term on a
  co-batched loop: a decode-only wave places a few regions of tier, the weight
  side takes everything above it, and the next wave's prefill chunk buys it
  back — 90 grows against 83 concessions in 102 steady-state waves. The term
  is now the widest tier of the last 16 placements (`TIER_RECENT_WAVES`),
  which spans the decode-only waves between one chunk and the next.
- **The staged-log flush on the reader's thread.** `LogFile::read_at` flushed
  the whole group-commit buffer when a read reached into it, and the reader
  was the scheduler resuming a conversation's recurrent snapshot after an
  ingest chain's 24 turns had been sealed: three 16–22 s stalls in seventeen
  minutes, each a wave with nothing on the device. Reads now copy from the
  staging buffer.

GPU-native MoE dispatch tables were never built in any of these runs — the
cache streams experts on this card, and the tables are for an all-resident
cache — so every MoE layer takes the host path's blocking routing readback.

**The batched-forwarding gate, and what it says about the operating point.**
`test_parallel_batched_forwarding_36_35b` on this card, after run AO, every
row at 100%:

| sessions | KV | decode tok/s (aggregate) | bulk prefill t/s |
|---|---|---|---|
| 1 | BF16 | 21 (40 warm) | 560 |
| 4 | BF16 | 104 | 810 |
| 5 | C8 | 77 | 720 |
| 8 | C10 | 72 | 596 |
| 16 | C10 | 53 | 361 |

Weight zone 8.7 GiB cedeable, 10.1 GiB resident, expert hit rate 55–65%,
speculation accepting 2.25 tokens a step. The ingest ran 15–20 sessions wide
against a 4.2 GiB zone at 5–7 tok/s aggregate — at equal width an 8–9× gap,
and every part of it residency: the hold at 0.45 let the stores and the idle
K/V take the zone to the point where every step streams most of the expert
set. The curve peaks at four to eight sessions with the zone whole, so the
hold moves to 0.85 (`interleave::HOLD`): the decode width it admits is what
the curve says, not what the span can be made to fit.

**Runs AP–AR: the hold is a floor; the width is the bound.** At 0.85 the
zone held at 8.1–8.3 GiB and the decodes ran at 190–280 ms a step (from 3 s),
20–40 tok/s aggregate at width 4–5 — and the runs stalled anyway: AP's
calibration sat 5 MiB under the mark and admitted one chunk a wave (the fill
was stopping on its own previous wave's tier; the stop now reads the
live-region view), and AR ran five fast decodes for six minutes with a queue
behind them and admitted no prefill, because the decodes' own growing K/V had
taken the live-region view under the mark. A residency fraction cannot be
both the width bound and the safety floor. The fill now has two: the model
answers `decode_width_target` — the width its measured curve peaks at on the
card it was calibrated on (8 for the 35B on the 16 GB card) — and the wave is
filled to that, decodes, in-flight prefills and new prefills together; the
hold drops to 0.60 and is only the floor beneath. The producer's slacks come
down with it (queue 4, open 8 beyond the carried width).

**Run AS: 22 directories in 13 minutes at the width target** — zone 6–7 GiB,
7–8 decodes, zero failures — with prefill getting a row only when a decode
finished, because the decodes filled the width. **Every wave now carries at
least one prefill and one decode** when both are queued: the first prefill of
a pass lands before the floor and the width are consulted unless one is
already riding from an earlier pass, and the decode side reserves that row
(a store that would fill the width while a prefill waits and none rides,
waits itself). The cost of the row is bounded by the width target; the floor
at 0.60 is far below nine stores.

**Run AT: 12 directories in 9 minutes with the prefill row guaranteed** —
prefill 242 tok/s in the first minutes (from 66–95), zero failures — and the
rate decayed with the zone: the carried sequences' K/V took the live-region
view to the 0.60 floor (5.6 GiB) within nine minutes, and at that residency
the step is about a second at width 8–10. The floor fraction is the wrong
instrument for the second time: it is a proxy for the thing that actually
sets the step time, which is the expert cache's hit rate, and that is
observable. **New stores are now gated on the measured hit rate**
(`WaveFill::residency_ok`): the fill reads the cache's cumulative hits and
misses each pass, folds the interval's rate into a short exponential average,
and admits a new store only while it stands above the model's knee
(`expert_hit_rate_knee`, one half — the gate's best rows ran at 55–65%, the
collapses well below one half). The rate is dimensionless and reads the same
on every card: a card that holds every expert never gates. The fraction stays
beneath it as a hard floor only.

**Run AU: 15 directories in 8 minutes, and the prefill row grew the width.**
The hit-rate gate held new stores at the knee as designed, but the guaranteed
prefill row bypasses every gate by construction, each prefill becomes a
resident decode, and resident decodes were never bounded: 12 decodes and 13
stores by minute eight, the zone on its floor, the hit rate through the knee.
Resident decodes are now bounded by the width target as well (less the row a
waiting prefill holds); the ones past it sit the wave out and are offered
first next fill, so each steps within a few waves. A decode that sits out
long enough is parked by the store's idle lag and costs a copy to return — a
bounded cost where the alternative was an unbounded width.

**The fill, restated as one bound.** By run AU the fill carried three gates
(the residency fraction, the hit-rate gate, the width target) and two
exceptions (resident decodes always run, the first prefill row is
unconditional), and the interaction between the last two is how the width
ran away. Special cases on shared infrastructure are the sign the fix is not
deep enough, so the fill is now one rule: **a wave carries `wave_width`
sequences** — decodes take at most `wave_width − 1` by rotation, prefill takes
the rest and always at least one — **and the width follows the measured expert
hit rate**: one narrower per fill while the smoothed rate stands under the
model's knee, one wider while it stands a tenth clear above it, never past the
model's ceiling (`decode_width_target`) and never under two. Width is the one
quantity that bounds how many stores exist, so nothing else gates a store; the
residency fraction remains only as an emergency floor a governed run never
reaches. The ceiling and the knee are the model's two numbers, both from the
gate, both to be given per-card rows.

**Run AW: 18 directories in the pool's first 8 minutes** — 9 in the first
three, the best opening of the day — zero failures, hit rate 0.59–0.63 with
the width at its ceiling of 8, zone 6.1–6.5 GiB. The queue then ran empty:
the producer's slacks were the fixed 4 and 8, and with 8 carried the pool held
16 open and nothing waiting. The engine now publishes its `wave_width` in the
admission report and the producer paces on it — a wave's worth queued, two
waves' worth open — so the pool's depth follows the engine's width as the
width follows the hit rate. The constants remain only as floors for an engine
that reports no width.

**Run AX: 13 directories in 8 minutes, and two things the log named.** The
producer held 95 times on "no fresh engine report": the report publishes once
a wave, waves ran 3–5 s with drain spikes, and the staleness test was a fixed
five seconds — so the queue ran empty with work available. The report now
carries its own cadence (`publish_interval_ms`) and the producer measures
staleness against two of those plus slack. And fifteen decodes were alive with
seven stepping: the guaranteed prefill row admitted a new sequence every wave,
each became a decode, and the width cap made them take turns, parking and
unparking stores as they rotated. **Live decodes are now bounded at the width
less the prefill row at promotion time**: a finished prefill becomes a decode
only when a decode slot is free, and waits (finished, its store parked) until
one is. The prefill row is always free because live decodes never fill it;
every live decode steps every wave; a completed prefill's turn is a queue,
not a store.

**Overnight runs AZ–BH (2026-09-05, 00:30–07:00), each a build on the last:**

- **AZ** — relief no longer buys weight-side ground (260 concessions in 441
  waves had been the setpoint's, not a claim's); width steps once per wave;
  the producer logs holds continuously. Zone held at 7.2–7.5 GiB, hit rate
  0.67, zero relief purchases — and the pool sat at 15 open with the queue
  empty, the prefill row idle.
- **BA** — an empty queue overrode the open bound; 41 conversations opened in
  a burst (a first turn takes seconds to reach the queue, so the queue read
  empty throughout). Now an empty queue lifts a held worker to the full mark
  and no further.
- **BB** — surviving KV pressure no longer closes prefill admission (with
  relief not buying, the KV side never grew: 13 free of 316 regions and no
  prefill for a hundred fills). Then `room` was found counting *finished*
  prefills waiting for a decode slot as in flight — zero room again.
- **BC** — 2/min steady at width 8, zero failures, and the ceiling was the
  bound: 7 decodes filled it at a hit rate of 0.59, the prefill row idle, 16
  conversations finished-and-waiting. Ceiling raised to 16.
- **BD/BE/BF/BG** — 13–14 directories in the pool's minutes 3–8 (2.6–2.8/min,
  the best segments of the day) each time the width ran at 8–11 with the
  zone still large, then each time the width controller hunted: the hit rate
  on these long-context chains sits at 0.45–0.55 at *any* width from 5 to
  11 — it is set by the routing diversity of the contexts more than by the
  width — so a knee at one half walked the width from 15 to 2 and back on
  noise. Steps slowed to one per 5 s; the width floored at 4 (the gate's
  peak); the knee lowered to 0.45, under the band, so only a genuine collapse
  narrows the wave; the ceiling back to 8, where the steady rate was.

**The configuration shipped at 07:00 on 2026-09-05 (runs BH–BJ):** one wave
width following the expert hit rate — ceiling 10, floor 4, knee 0.45, one step
per 5 s — decodes bounded at width−1 by promotion, one prefill row per wave;
the residency fraction 0.50 as an emergency floor only; relief that never buys
weight-side ground; the producer pacing on the engine's published width (a
wave queued, one and a half waves open, staleness relative to the report's
own cadence). Measured on the 16 GB card: calibration prefill 630–750 tok/s;
pool opening 6–10 directories in the first 3–4 minutes, 14–17 by minute 8–9,
hit rate 0.53–0.72, decode 7–9 wide at 300–400 ms a step (30–40 tok/s
aggregate with speculation); zero failures, zero refusals, zero relief
purchases in every run from AZ on. The second cohort settles at about one
directory a minute, decode-bound: the chains' long summaries run 7–9 wide
while the finished prefills wait for decode slots, and the KV side (not the
weight zone) is the tight resource — the idle K/V of §4.11.5 is now the one
tenant left to move.

**The night's ledger, pool phase only** (directories at +3/+8/+13 minutes;
"decode≥" is forwards × sequences per second, a floor under speculation):

| run | change | +3 | +8 | +13 | prefill tok/s | decode≥ | hit rate | width | zone GiB |
|---|---|---|---|---|---|---|---|---|---|
| AW | one width bound following the hit rate | 9 | 18 | 23 | 232→43 | 5–7 | 0.63→0.56 | 8 | 6.5→5.8 |
| AX | producer paced on `wave_width`; floor 0.50 | 7 | 13 | – | 247→70 | 5 | 0.60→0.54 | 8 | 7.0→5.9 |
| AY | cadence-relative staleness; promotion into free slots | 2 | 12 | 19 | 247→0 | 6→18 | 0.57→0.62→0.55 | 8 | 7.0→5.0 |
| AZ | relief no longer concedes; width one step per wave | 1 | 11 | – | 183→35 | 5→12 | 0.67 | 8 | 7.5→7.2 |
| BA | empty queue lifts the open bound | 0 | 11 | – | 243→191 | 4→7 | 0.61 | 8 | 7.3→6.6 |
| BB | KV pressure no longer closes admission | 6 | 7 | – | 192→53 | 8→15 | 0.66→0.62 | 8 | 7.3→6.4 |
| BC | `room` counts running prefills only | 3 | 13 | – | 280→81 | 5→14 | 0.60→0.59 | 8 | 6.2 |
| BD | ceiling 16 | 2 | 15 | – | 185→89 | 8 | 0.57→0.58 | 11→4 | 4.8→6.0 |
| BE | dead band 0.50–0.55 | 0 | 10 | – | 211→114 | 7 | 0.54→0.53 | 15→5 | 4.7→5.8 |
| BF | ceiling 12, steps every 5 s | 2 | 16 | 21 | 250→63 | 6–8 | 0.53→0.46→0.54 | 11→8→2 | 4.7→5.3 |
| BG | width floor 4 | – | 9 (+5) | 13 (+10) | 288→36 | 6–7 | 0.49→0.51 | 11→5 | 4.5→5.7 |
| BH | ceiling 8, knee 0.45 | 9 (+4) | 17 (+9) | 20 (+14) | 209→20 | 7→16 | 0.62→0.47→0.54 | 8 | 6.4→5.3 |
| BI | open bound 1.5 waves | 10 (+4) | 14 (+9) | 15 (+14) | 197→25 | 8→21→16 | 0.72→0.64→0.61 | 8 | 7.3→6.0 |
| BJ | ceiling 10 | 6 | 16 | 21 (25 at +18) | 281→50 | 6→12 | 0.54→0.57 | 10 | 5.9→6.3 |
| BK | final build (resident branch, cap 64) — shipped | 19 (+6) | 28 (+11) | 31 (+16) | 346→210→0 | 2→6→15 | 0.53→0.47 | 10 | 6.3→5.0 |

**On a card that holds every expert.** Checked on 2026-09-05 at the user's
request: as shipped at 07:00 two ceilings would have bound such a card — the
model's width row (10) and the prefill backstop (24). Now, while the smoothed
hit rate stands at or above the model's *resident mark* (`resident_hit_rate`,
0.9 — a streaming card's best rows measured 0.55–0.72, a resident cache
reports one), the width's ceiling is the engine's hard cap
(`WAVE_WIDTH_HARD_CAP` = 64, the gates' widest rung) rather than the model's
row, and the prefill backstop matches it; the producer's burst slack is half
a wave per report interval. Everything else already scaled: the knee is
dimensionless, the floor is a fraction of achievable residency, the producer
paces on the published width, the tier and chunk sizes come from the wave
plan on the machine. **Not yet measured on the 72 GB card**; the gate must be
run there and the model's rows given per-card values (plan row 27).

**Every constant in this section was measured on one card and one model**
(RTX 4090 Mobile 16 GB, Qwen3.6-35B-A3B) and scales with neither: `HOLD`
(a fraction, so partly), `SCAN_QUEUE_SLACK` and `SCAN_OPEN_WAVES`
(absolute conversation counts at the 4–8-session peak), `PREFILL_MIN_ADVANCE`
(rows, against a ~1 MiB/row tier), `TIER_RECENT_WAVES` (placements, about a
minute of waves here), `KV_REGION_SLACK` and `TIER_MARGIN_CAP_REGIONS`
(regions). A 72 GB card holds the 35B's experts whole and the curve peaks far
wider; a 24 GB card on PCIe 3.0 streams slower and it peaks narrower. The
work to make them scale is to derive them from what the engine can measure
on the machine it is on — the gate's aggregate-vs-width curve per model, the
per-row tier cost from `WavePlan`, the region count from the span — or to
give them per-model, per-card rows the way `draft_ladder.rs` does.

### 4.11.5 The remaining VRAM tenant: idle conversations' K/V

Run P also answered the question §4.10 left for K/V. The gentle-early ingest
demotion (`demote_cold_ingest_if_pressured`) ran against 5,920 MiB of hot ingest
K/V across 100 idle timelines with a 3,520 MiB watermark and freed **8–9 MiB per
pass** (`nudged=true` on every line). The rolling hot window is not what holds
it: a live slot's block table references its own sealed turns' chunks, so
setting `hot = None` on them frees nothing until the slot is freed or reprojected
(`evict_cold_tail` says as much). Idle K/V for a *live* sequence can only leave
the device by demoting the sequence's block table itself to the warm tier and
re-elevating it on admission — the same shape as the recurrent parking, on the
other tenant. Not built. The producer gate bounds how many such conversations
exist, which is what makes the tenant bearable rather than what makes it right.

### 4.11.6 What the fill replaced, and what was still standing

The fill (§4.11.4) is the only admission path there is. Everything below it had
survived as a second path taken when `optimal_weight_bytes()` returned `None` —
that is, when there is no device reservation to defend, so on unit tests and CPU
sessions and on no GPU run at all. Zero is the honest `optimal` for that case:
the residency stop cannot bind, the wave is bounded by its width and by what the
allocators will actually give, and one path serves both. So the second path and
everything only it reached are deleted:

* **`scheduler/admission.rs` (928 lines)** — `plan_admission`, the AIMD byte
  budget (`cut_budget` / `raise_budget` / `budget_notches` / `admit_quantum`),
  the backlog and evidence controllers (`backlog_admit_action`,
  `evidence_admit_grow`), the width-dependent reserve (`BandParams`,
  `reserve_for_width`, `decode_reserve_bytes`) and the per-block cost model
  (`per_block_kv_bytes`, `prefill_cost_bytes`). This is the "estimate a cost,
  compare against a setpoint, admit what fits" family that §4.5–§4.11 falsified
  five times; the fill's claim-and-read replaced its judgement, and this was its
  remains.
* **`Scheduler::admit_budget` and its actuators** — `cut_admit_budget`,
  `cut_admit_budget_leveled`, `raise_admit_budget`, `max_admit_budget`,
  `log_throttle`, the AIMD reopen in the run loop, `regulate_ingest_admission`,
  and the evidence state (`admit_grow_streak`, `admit_ok_tokens_seen`,
  `promote_ok_tokens_seen`, `promote_last_progress`, `last_level_cut`). Once
  `plan_admission` went, **nothing read the budget**: it was still computed,
  throttled on four signals, logged in both directions and published in the
  memory report, and no decision anywhere consulted it. A control loop with no
  actuator reads exactly like a working one.
* **`scheduler/wave_trace.rs` + `zend/examples/wave_trace_report.rs`** — the
  paired `adm`/`wave` JSONL built in §3 to fit the controller of §4.5. The
  controller is gone, `flush()` had no caller (so the tail of every trace was
  lost), and the `adm` half was emitted from inside the byte-budget path, which
  means a GPU run wrote only half of a format whose whole point was the join.
* **`Ground::residency_ok`** — a hook from the abandoned hit-rate-gate-on-stores
  arc, whose trait default and only implementation had both decayed to
  `weight_bytes() > optimal`. The fill asks the zone directly again.

**One behaviour goes with them.** `regulate_ingest_admission` was the *gentle*
hot→warm backpressure: warm tier over its host budget, or the drain backlog over
`ingest_warm_backlog_pct`, cut the setpoint. It has been unactuated since the
fill landed, so nothing changes by deleting it, but the gap is now visible rather
than hidden behind a busy control loop. What still applies backpressure is
`sync_if_backlog_critical` (a device sync above `ingest_sync_ceiling_pct`, the
harder stop) and `demote_cold_ingest_if_pressured`. If a gentle throttle is
wanted back it needs a lever the engine actually reads — the wave width — not a
byte setpoint.

### 4.11.7 Every buyer but admission, and every gate but the fill

A full `repo_map` ingest with the fill of §4.11.4 in place (2026-09-08, run 4)
still drove the weight zone from 10,398 MiB to 3,709 and ingested one directory
in twenty-eight minutes. The gate was working — `stopped_on_weights=true` on 565
of 942 fills — and the zone fell anyway, because the fill was one of several
things moving the boundary and the only one that knew the hold. The log, ANSI
stripped, attributes it:

| window | what ran | admission | weight zone | stores |
|---|---|---|---|---|
| 04:17:26–48 (22 s) | 350 section forwards, up to 12 × ~2k tokens | **none** — first `wave fill` line is 04:18:25 | 10,398 → 5,171 MiB (the hold) | 1 → 23 |
| 04:31:01–04:32:15 | decode only, 13 → 8 seqs | `stopped_on_weights=true`, `prefills=0` | 5,628 → 4,863 MiB | 37 → 58 |

Sixty-three concessions, all stopped only by `floor_slots=769` — the expert
cache's hard floor, 1,415 MiB — never by the hold. Four things were buying:

1. **Stores at conversation open.** `InstallRecurrentState` (the prompt-branch
   checkpoint), `create_sequence` (the timeline snapshot) and
   `NewSequence { parent }` (a fork) each created a device store when a
   conversation *opened*. Every one of 994 creations was followed within 10 ms
   by `recurrent stores packed considered=N+1`, with `prefills=0`; the store
   count tracked `queued + decoding` (31 + 13 against 37) and peaked at 74 —
   6,000 MiB. §4.11's move of the *view's* store to admission had left the
   parent's untouched.
2. **Sections outside admission.** `IngestSection` pushed straight into the
   active set and `build_section_batch` packed to the token cap. One startup
   minute of sections took the zone to the hold before any admission pass had
   run.
3. **Claims and the tier buying for themselves.** `claim_region` bought
   `KV_BUY_STEP` on exhaustion (the twenty `wanted=8` concessions), the
   transient tier's placement bought its shortfall (`wanted=96`, `wanted=1..4`),
   and `forward_wave` asked for `32 + contexts − free` regions on every forward.
   All three moved the boundary from inside code that knows only the hard floor.
4. **Loops beside the admission pass.** `relieve_vram_pressure` ran 865 times
   from four sites and relieved twice; `demote_idle_slots` ran on every due pass
   and re-admitted 1,611 of the slots it had just demoted, 241–650 ms later; the
   producer's scan pool held 9,237 times on the weight zone — the same signal
   the engine's own gate was already refusing on.

**The design as built (S13–S16).**

* **A store is materialised at admission, never at open.** Opening records a
  `RecurrentSeed` — the checkpoint payload, the parent, or `Neutral` — and
  `claim_recurrent` resolves it through `materialise_recurrent`: the slot's own
  timeline snapshot first, then the seed chain (a live parent is copied; an
  evicted parent's checkpoint is reached through it). A queued conversation
  holds nothing on the device. The seal evicts, as before.
* **Admission is the only buyer.** `claim_region` refuses on exhaustion,
  `place_transient` refuses when the tier does not fit, the pre-wave purchase is
  gone, and so is the ground-broker registry they went through.
  `Scheduler::buy_kv_ground`, called from `WaveFill::admit` for every item the
  gate passes (and from `resume_parked`), buys the larger of two shortfalls —
  the price against the free list, and the tier against the frontier gap —
  between forwards, bounded by the same accounting that admitted the item.
* **Sections pass the fill.** `IngestSection` queues a `PendingSectionIngest`;
  `Kind::Section` is a seventh band, Low, ahead of Low prefill (the caller is
  blocked on the seal before it can submit the turn that attends over it) and
  behind every interactive band. Priced like a prefill without a lease, gated
  and counted like one; the setup and the claims run at admission. A running
  section is an active slot and its finish is a completion.
* **One eviction pass, run for a reason.** `demote_idle_slots` runs when the
  head of the queue does not fit the headroom the fill prices with, or when
  the engine is idle so the weight side grows back to what the card can hold.
  Decodes never ask; they are continuations. The relief ladder, its setpoints,
  `evict_cold_tail`, `compress_pending_turns` and the gentle-early ingest demote
  are deleted, and the producer's gate no longer reads the weight zone.

The rules of §4.11.4 are unchanged. What changed is that nothing else in the
engine can now move the boundary or place a store without passing them.

### 4.11.8 The refused wave that never narrowed

Run 11 (2026-09-08, S25) carried §4.11.7 through a clean calibration — one
buyer, 3 refusals in the first minute, none after, 367–557 tok/s — and wedged
the moment the repo_map ingest began. In the first two minutes of ingest: 199,704
placement refusals of one wave, 157 waves with no forward, seven decodes admitted
and never stepping, 87 directories queued, zero ingested. Every refusal read the
same: `needs 335544320 B … is 4–7 regions into ground live KV arenas hold …
gap_mib=213..256 least_mib=176 bought_mib=0`. The margin the fill held back had
doubled to its 1 GiB cap and the budget read zero; the wave asked for 320 MiB
anyway.

Three things, each visible in the log:

1. **The refused wave was never re-formed.** A creep group lives across waves —
   members, layer cursor, held residual — and `decode_forward_cobatched` forms a
   fresh group only when none is held. `note_tier_refusal` requeued the group's
   unstarted prefills but left the group standing; the next fill re-admitted
   them (`re-admits=577`), `build_wave_group_inputs` found them under the same
   ids, and the same 320 MiB (seven verify blocks beside a 287-token turn)
   went back to the same placement, at 7 Hz, for as long as the daemon ran. The
   margin ratchet bounded nothing, because the thing it was meant to narrow was
   never composed again.
2. **An admission guarded only its own rows.** `buy_kv_ground` kept the gap at
   `max(this admission's tier, least)`. With a 300-row creep held, one fill
   admitted a prefill whose prompt-branch checkpoint install took eight regions
   off the gap (384 → 256 MiB), and the held wave — which the fill had not
   counted — was refused by four. The wave's tier is one quantity over every
   row it carries; a purchase that guards an increment lets the next increment
   eat the last one's ground.
3. **The least chunk never fit beside a decode.** The fill bought the least
   wave's tier into the *gap*; the group former reads the *budget*, the gap
   less the margin. So in steady state the gap held exactly `head + 128 rows`
   and the budget held one margin less: `budget=141..190 MiB` against a 192 MiB
   least wave, seven decodes stepping alone while eight admitted prefills sat
   unstarted for a minute. Prefills ran only when a completion happened to
   widen the gap.

**The design as built (S26).**

* **A refusal drops the wave.** `note_tier_refusal` resets the held group
  (members, cursor, residual) before requeueing, so the next build reads the
  gap as it stands and composes to it; the layers the creep had done are redone
  from zero, which is idempotent. Nothing is bought at the refusal — the fill's
  purchase runs on the next pass, every iteration.
* **The fill's head is every row the next wave already carries**:
  `WaveFill::head_rows = decode rows + Scheduler::held_creep_rows`. Every
  admission is priced as an increment over that head (the gate's charge), and
  `buy_kv_ground(cost, wave_tier)` guards `wave_tier = tier(head) + increment`
  — the whole wave — in the gap. A parked turn resuming guards the least
  forward.
* **The least wave is bought into the budget.** `publish_tier_budget` asks for
  `(least + margin) − gap` (`least_wave_purchase_regions`), bounded by how far
  the zone stands above the hold the gate defends, so a zone at its hold buys
  nothing and the wave runs its decodes alone. The least wave is the head plus
  one least chunk, or the head alone while a group is held (a held group takes
  no new member).
* **The margin is fixed** at four regions (`TIER_MARGIN_REGIONS`), covering only
  what moves between build and placement — the persistence thread's arena
  creation and the placement's rounding. The doubling ratchet, its cap and its
  decay are deleted: they were a second loop on the same budget as the purchase,
  and both of their measured pathologies (§4.11.4 run AN's churn at the cap;
  run 11's zero budget) came from that. What the cap's "final for started
  prefills" did is kept as a streak: `TIER_REFUSALS_BEFORE_FAIL` (8) consecutive
  refusals — each one a re-formed wave — fail the started prefills so their
  ground comes back; a placed forward ends the streak.

## 5. Plan

| phase | deliverable | gate | status |
|---|---|---|---|
| 1 | wave/admission trace + report | records join; both sides visible | ✅ |
| 2 | fat pipe: queue-depth-bounded pool | queue depth p50 ≥ 8 candidates | ✅ p50 = 12 |
| 3 | re-baseline with a full queue | identify the real constraint | ✅ it is the feeder |
| 4 | learned controller on the admission budget | climbs without ratcheting | ❌ falsified — §4.5 |
| 5 | calibrated model (`scheduler/sim.rs`) | reproduces the measured points and §4.5 | ✅, retired with the controller in §4.11 |
| 6 | residency climb on the producer's open-conversation count | weight zone holds while width grows | ❌ falsified — §4.6 |
| 7 | account for the regions the class line did not | arenas + span tenants = `live`, no remainder | ✅ §4.6 — no leak; three span tenants |
| 8 | re-fit the model with a **per-sequence span-region cost** | reproduces the measured displacement | ✅ §4.7 |
| 9 | controller on width vs. weight residency | holds the weight zone while width grows | ❌ falsified — §4.8 |
| 10 | show the lever has authority | vary it; measure the tenants respond | ✅ §4.9 — stores, `regions = 9 × stores` |
| 11 | throttle at admission on the store price | weight zone holds while the queue stays fat | ❌ §4.10 — the queue already holds the stores |
| 12 | park idle recurrent stores to host RAM; claim on admission | stores track in-flight width, not backlog | ✅ §4.11 — 5,120 → 640 MiB, span refusals 140 → 0 |
| 13 | decode-first fill, decodes gated by the real allocators only | no fill stops at one decode while decodes are eligible | ✅ §4.11.1, §4.11.4 |
| 14 | bound the co-batched wave in tokens through `prefill_width_cap(dtype, head_rows)` | no tier refusal on a wave the fill composed | ✅ §4.11.2 |
| 15 | producer paces on `decode_starved` + backlog; controller, estimate and wait cap deleted | open conversations never exceed what the engine steps | ✅ §4.11.3, §4.11.4 |
| 16 | demote a live idle sequence's block table to warm; re-elevate on admission | idle ingest K/V leaves the device | open — §4.11.5 |
| 17 | chunked dialogue prefill: a wave carries `[offset, offset+advance)` of a turn, priced per chunk | no turn fails as too wide; long turns ingest beside decodes | ✅ built — §4.11.4 run AH |
| 18 | resume lands parked: a restored snapshot costs no device store until admission | stores = in-flight width at pool open, never open conversations | ✅ §4.11.4 run AJ — 19–23 stores for 18–21 decodes |
| 19 | producer holds on queue length; scheduler offers continuations before first turns | chains reach their next turn; open conversations ≈ carried + a wave | ✅ §4.11.4 run AK — 25–59 open, holds biting |
| 20 | hold and purchase cap on `min(extent, effective)`; tier budget leaves the weight side what it is owed under the hold | zone never bought to its floor by the tier | ✅ §4.11.4 run AL — zone at the hold, 0 refusals, first completions |
| 21 | producer bounds open conversations to carried + 2× queue slack | idle K/V does not pin the zone under the hold | ✅ §4.11.4 run AM — 22 dirs / 26 min, 0 failures |
| 22 | tier margin decays; staged-log reads copy instead of flushing | no multi-second drain stalls | ✅ §4.11.4 run AN — 16 dirs / 8 min, drain avg 0 in steady state; churn remained (2/wave) |
| 23 | weight-side growth capped at the frontier gap less slack and the recent-max tier | concessions per wave → 0 in the pool phase | ✅ §4.11.4 runs AO–AZ — 0.6/wave with the last tier, 0 relief purchases once relief stopped buying |
| 24 | one wave width, following the measured expert hit rate; decodes ≤ width−1 by promotion; one prefill row per wave | prefill and decode on every wave; width never runs away | ✅ §4.11.4 runs AW–BH |
| 25 | producer paces on the engine's published `wave_width` and report cadence; empty queue lifts a held worker to the full mark | queue one wave deep, open ≤ 1.5 waves beyond carried, no stale-report holds | ✅ §4.11.4 runs AX–BI |
| 26 | relief does not concede; KV pressure does not close admission; `room` counts running prefills only | KV side grows by claims, never bought by the setpoint | ✅ §4.11.4 runs AZ–BC |
| 27 | per-model, per-card rows for ceiling / knee / floor, derived from the gate on each machine | the same code runs the 3090 and the PRO 5000 without retuning | open — §4.11.4 last paragraph |
| 28 | delete the byte-budget admission path, the AIMD budget and the wave trace built to fit them | one admission path, on every machine; nothing computed that nothing reads | ✅ §4.11.6 |

Phase 8 was planned as a *reclamation* term. There is nothing to reclaim, so the
missing term is the one §4.6.3 names: how many regions each additional concurrent
sequence takes, and what that costs in weight residency. It is a **measured**
quantity per model, never a constant — a non-DeltaNet model has no span tenants
at all and pays nothing.

Phases 2 and 3 are done and their result is §4.4: the feeder admits 0.8 of 14.4
offered and prefill never widens past one. That is the input distribution the
simulator needs, and it did not exist before — every earlier measurement
described an engine that was never given work.

**Batched creates are deliberately still open.** A directory mints three
conversations through `new_conversation_with_projection` under the engine mutex,
where calibration uses `new_conversations_with_projection_batch`. It is a real
cost, but the trace now shows it is not the binding one, so it is sequenced after
the controller rather than before it.

## 6. For review

### 6.1 Generality — the constraint the law must satisfy

The engine serves models with no VRAM pressure at all: a dense checkpoint that
sits fully resident has no expert cache, never concedes ground, and wants the
feeder to admit everything the queue offers. **A law tuned on a streaming MoE on
a 16 GB card must not throttle it.**

So every term has to be derived from a measured quantity that goes to zero when
the pressure is absent, and the law must reduce to *admit everything that fits*
in that limit. Concretely:

* `W_min` cannot be a constant. It has to be expressed against **observed
  concession behaviour** — a weight zone that never concedes has no floor worth
  defending, and the constraint is then vacuous by construction rather than by a
  special case.
* The transient model must be **fitted per model** from `wave` records, not
  carried as a coefficient. A dense model's FFN and attention peaks scale
  differently from an MoE's, and a fixed `per_seq` cannot express either.
* The controller's default posture must be **permissive**, tightening only on
  evidence — a stall, a refusal, a measured fall in tok/s. Starting conservative
  and relaxing is what produced the 0.8-of-14.4 result in §4.4.
* The **per-sequence span cost is read from `span_tenant_counts`**, never assumed.
  §4.6.2's three tenants exist only on DeltaNet-lineage models; elsewhere the
  counts are zero, the cost term vanishes, and width is free of this constraint
  by measurement rather than by a model check. This is the constraint's best
  case: the quantity that makes the law necessary is the same quantity that
  switches it off.

The test in `repo_scan` (`the_queue_watermarks_name_no_property_of_the_card`)
pins this for the producer. The feeder needs its own equivalent.

### 6.2 Open questions

1. **Is `W_min` the right primary constraint?** It makes the feeder's objective
   "protect expert residency" rather than "maximise width", which inverts the
   current design. §2.1 supports it but only on one card — and §6.1 says it must
   vanish on a model that never concedes.
2. **Should the producer's high-water mark be tokens or candidates?** Tokens
   match the feeder's unit; candidates are what the pool can actually count
   without pricing each item.
3. **How much simulator fidelity is worth buying?** A model that reproduces
   width and tok/s may be enough to rank control laws without reproducing the
   partition's fragmentation behaviour.
4. ~~**Speculative decode is on in the batched test and absent from the daemon's
   traces.**~~ **Resolved by §4.6.2's attribution.** The `verify-stash` tenant
   peaks at 1 region across a whole ingest against `deltanet-state`'s 576, so
   speculation is genuinely not running in the daemon — it is not a tracing gap.
   `speculative_decode_is_refused_for_a_model_carrying_recurrent_state` says why:
   the 35B carries recurrent state, and a rejected draft cannot be truncated out
   of a DeltaNet running sum. §2.1's batched-test numbers are therefore **not**
   an achievable target for this model in the daemon, and the comparison in §2.2
   overstates the gap by whatever speculation was worth there.
