# Behavioural test catalogue — the VRAM partition

Companion to `docs/qwen38_layer_streaming.md` and
`docs/archived/elastic_vram_partition.md`. Those describe implementations; this
describes what the partition must **do**, in terms a test can assert without a
GPU, a checkpoint, or a forward pass.

## Why this exists

The partition is the mechanism that lets a 27B model run on a 16 GB card. It
decides, continuously, how one address reservation is split between model
weights and KV cache, and it does so against a workload it cannot predict. Every
defect found in it during development shared three properties:

- **Silent.** Nothing errors. The engine runs, produces correct tokens, and is
  slow — a 10× decode loss read as "streaming is expensive" for weeks.
- **Emergent.** No single function was wrong. The ratchet (§C) was an
  interaction between an exact shrink path and a hedged grow path, each
  defensible alone.
- **Invisible to unit tests.** Every existing test passed throughout, because
  they assert that a function returns the right number, not that a *sequence* of
  decisions converges anywhere good.

So the assertions here are on **trajectories**, not calls. A scenario is a
workload — a sequence of forwards with a given shape — and the assertion is a
property of where the partition ends up and what it did on the way.

## Why it runs without a device

The geometry is already pure (`layout_span`, `ceiling_regions`, `claimable`,
`blocked`, `tier_fits`, `kv_grow_step`, `span_from`), and so is the whole weight
side (`weight_zone.rs`, whose header says so explicitly: *"Keeping the policy out
means the whole module tests without a GPU, a model, or a routing trace."*). The
dense zone's planner (`zone::plan_zone`), its eviction order (`order`), and its
residency (`residency::LayerResidency`) are pure already.

What is not pure is the KV ledger's *policy* — `spare_regions` and the free
list — which sits on `RegionPool` behind a CUDA `Reservation`. Extracting that
ledger is a precondition for this catalogue, and it is the same extraction
`weight_zone.rs` already performed for the mirror side.

The payoff is not only speed. A GPU-backed test needs the process-global pool and
therefore the serial lock, so hundreds of them cannot run in parallel and each
one perturbs the next. A pure ledger makes each scenario an independent value.

## What a scenario is

```text
given   a span, a model shape, and a tier price
when    a sequence of forwards runs, each with a cohort size and a KV demand
then    the partition satisfies the invariants at every step
and     its final state satisfies the quality property under test
```

Two classes of assertion, and the distinction is load-bearing:

- **Invariants (§A)** must hold after *every* step of *every* scenario. They are
  checked by the harness itself, not restated per test. A violation is
  corruption or a crash in production.
- **Quality properties (§B–§H)** are what each scenario is actually about. A
  violation is a slowdown, which is exactly the class that has been escaping.

---

## Family A — Span integrity

Checked automatically after every step of every scenario. These are the
properties that make the reservation a reservation.

| id | invariant |
|----|-----------|
| A1 | Every region lies within `[region_base, weight_floor)`. |
| A2 | Regions are disjoint and their count is `total`. |
| A3 | `live + free_below_ceiling + blocked + unclaimed == total`. |
| A4 | The weight zone lies within `[weight_floor, span_end)` and never overlaps a region. |
| A5 | The transient tier's footprint lies between the arena frontier and `weight_floor`, and overlaps no live region (`tier_fits`). |
| A6 | The dense block is inside the span and never moves after `freeze_dense`. |
| A7 | Nothing is allocated outside the span. The only host allocations are the pinned tiers, and their total never exceeds the measured pinnable budget. |
| A8 | `weight_floor` moves only between forwards — never while a wave generation is open. |
| A9 | The weight zone never retracts below its floor: the pinned head plus one streaming cell (dense), or `MIN_ELASTIC_RESERVE` (MoE). |

A7 is the one worth stating even though it reads as tautology: the whole
reservation exists because a driver allocation outside it can be demoted to host
RAM by WDDM, which cost a measured 17× on decode with nothing reporting it.

---

## Family B — The boundary moves

**B1 — A claim that runs the KV side out is served.** Given a full KV side and a
weight zone above its floor, a region claim triggers a purchase and succeeds.
*Design:* this is what makes the grow direction safe to be aggressive; it is the
mechanism the whole of §C rests on.

**B2 — A purchase takes exactly the shortfall.** Not a step, not a fraction — a
claim that is short by `n` regions concedes `n`. A counter-driven version of this
once conceded 4,436 units of demand against a 28-region shortfall and evicted
1,598 experts.

**B3 — The floor is refused mid-wave, and nothing moves.** A concession attempted
while a generation is open leaves the zone, the floor and the residency exactly
as they were. Half-application is the failure mode (`boundary_half_application`).

**B4 — Concede then reclaim returns to the same address.** A zone that gives up
`k` layers and takes them back has the *same* layers at the *same* addresses,
because residency is a prefix of the protection order at every size.

**B5 — A refused purchase is a refusal, not a partial move.** If the weight side
cannot pay, the KV side is told so and the boundary has not moved.

---

## Family C — No ratchet

The defect this family exists for: the shrink path reads the present exactly
(admission evicts weights on contact) while the grow path once consulted a
forecast, so demand that rose and fell left the zone permanently small.

**C1 — Rise then fall restores the zone.** A workload that goes 1 → 20 → 1
contexts ends with the weight zone within one growth step of where it started.
*Design:* the KV the wide cohort needed is genuinely gone; ground that is free
must be offered.

**C2 — The zone is not monotonically decreasing under a stationary workload.**
Across a long run at constant demand, the zone's size has no downward trend.

**C3 — A burst does not degrade the run after it.** Decode residency for a
1-context forward is the same whether or not a 20-context burst preceded it.

**C4 — Recovery is bounded in forwards, not in wall-clock.** The zone regains
its ground within `k` forwards of the demand falling, for a stated small `k`.
*Design:* a policy keyed on wall-clock cannot serve a workload that changes
faster than its window — measured, every boundary event in a full gate run falls
inside a 30 s span against what was a 60 s window.

**C5 — Repeated bursts do not accumulate.** Ten rise/fall cycles end where one
does.

---

## Family D — Dense residency and PCIe latency hiding

The quantitative family. `order.rs` gives the model:
`window(gap) = (gap − 1) × t_compute`, `stall(gap) = max(0, t_fetch − window(gap))`,
with `t_compute ≈ 1.4 ms` and `t_fetch ≈ 8 ms` for a ~200 MiB layer.

**D1 — A model that fits is fully resident and never streams.** Zero transfers,
zero blocking joins. The streaming machinery must cost nothing when unused.

**D2 — Total stall is within a factor of the equal-gap optimum**, at every
residency count from `pinned+1` to `N`. Equal gaps are optimal by convexity and
Jensen; this asserts the order achieves it.

**D3 — Max gap ≤ 2⌈N/k⌉ at every prefix**, exact at powers of two.

**D4 — Stall is monotone non-increasing in residency.** One more resident layer
never makes the sweep slower. This catches an order whose `k+1` prefix is worse
spread than its `k`.

**D5 — A held prefix is the worst spread, and the order must beat it.** The
contiguous-tail arrangement is the natural implementation and is measurably the
worst; this is the regression guard for reverting to it.

**D6 — Growth extends the resident prefix.** Layers taken back are the ones most
recently given up, so growth restores the spread rather than filling with
whatever faults next.

**D7 — The floating cell is never the layer under the wave.** A prefetch may not
evict the layer being computed.

---

## Family E — The MoE expert zone

**E1 — Slots are filled from the right, retracted from the left.** The mirror of
the KV side, which is what lets the boundary move at all.

**E2 — A retraction relocates the hottest doomed occupants and evicts the rest.**
Never a loss: the cold tier holds every expert.

**E3 — Retraction is a suffix of the index space.** Equal-sized slots mean no
fragmentation and no compaction.

**E4 — The zone never retracts below `MIN_ELASTIC_RESERVE`.** Derived from the
phase spans plus a first wave's KV, so a change to the wave tier moves the floor
with it.

**E5 — Expert and dense zones answer the same broker.** A model is one or the
other; both must satisfy §B identically.

---

## Family F — The transient tier

**F1 — The tier packs against the arena frontier with no gap.** A gap is room for
claims that cannot arrive, because arena creation waits for the inter-forward
window.

**F2 — A tier that does not fit buys exactly its shortfall, once.** Not per
region: a purchase costs a device-wide quiesce.

**F3 — Tier ground is claimable again the instant the tier is released.** Ground
blocked by the ceiling is not lost, and `blocked` accounts for it separately from
`free` so a report can tell "spoken for" from "gone".

**F4 — The tier never overlaps a live region**, at any frontier and any floor.

**F5 — Bump ranges within a generation are disjoint and within the domain.**

**F6 — Reset is refused while a generation is live.** The count bounds the arena;
the borrow bounds one guard.

**F7 — A domain's reset never touches another domain's ranges.**

---

## Family G — Workload shapes

Each of these runs the full invariant set plus its own property.

| id | shape | property |
|----|-------|----------|
| G1 | single context, short | zone reaches full residency; tier is one wave's price |
| G2 | single context, long (huge KV) | zone concedes smoothly; no oscillation |
| G3 | many contexts, short | KV dominates; zone at its floor but never below |
| G4 | many contexts, long | the hardest case: both sides want everything |
| G5 | ingest — demand climbs monotonically | the weight side is never handed ground the next step needs |
| G6 | idle after load | the zone takes the whole span; nothing is stranded |
| G7 | alternating wide/narrow | §C5, with the tier price changing each step |
| G8 | a cohort that never ends | steady state is reached and held |

G5 is the one with a measured failure behind it: with only occupancy and history
guards the weight side still took eight regions ninety seconds into a rebuild
that then saturated. The derivative guard is what covers it, and this scenario is
its test.

---

## Family H — Convergence and churn

**H1 — A stationary workload reaches a fixed point.** After `k` forwards the
partition stops moving entirely. Churn on the hot path is the cost this whole
design is trying to avoid.

**H2 — No take-one/give-one oscillation.** Across a long stationary run the
number of boundary moves is bounded by a small constant, not proportional to the
number of forwards.

**H3 — A negotiation never takes the whole offer.** Measured: removing the
geometric hedge provoked the KV side into buying the ground straight back, and
applied grants fell from 17 to 4.

**H4 — Convergence is monotone in the useful direction.** The zone approaches its
fixed point without overshooting past it and coming back.

**H5 — The partition is deterministic.** The same workload from the same start
produces the same trajectory, every time. Everything above depends on this, and
it is the property that makes hundreds of scenarios worth running.

---

## Cost

All of it is arithmetic over small integers. The target is the **whole catalogue
in under two seconds**, in parallel, with no device and no serial lock — which is
what makes it reasonable to run on every change rather than deliberately.

Scenarios are values, so the shapes in §G are a table and the families are
generated over it: every workload shape crossed with every model kind (MoE,
dense-resident, dense-streaming) and every span size, with the invariants of §A
checked at every step of every one.
