# Held ground, eliminated

**Status:** built 2026-10-18 (see §7 for what was measured).
**Supersedes:** the fixed `HELD_GROUND_BYTES` block of `docs/vram_span_partition.md` §1.

## 1. What held ground was, and what it cost

The reservation's left end carried a fixed 1,280 MiB block, *held ground*, for the
device buffers that outlive a forward but may be asked for inside one: launch-table
staging, the sampler's argument pack, decode slot-state slabs, attention split
partials, cuBLAS workspaces. Its address never moved, so a buffer could be taken at
any instant — inside a forward, where a span region claim is refused, or from another
thread.

It was sized from one daemon measurement (`.5`, 2026-10-17: ~970 MiB under a
conversation and an ingest — kernel scratch 436, slot-state slabs 304, penalty counts
76, gallery scan 68, launch tables 23, cold-load staging up to 64). The Flash-Next
batched gate on the 16 GB laptop peaks at **38 MiB** of it. The rest was ground no
tenant could use: on that card the expert zone is capped by the span, and 1,242 MiB is
~650 expert slots at 1.90 MiB.

Two properties made a fixed block the wrong shape:

- **Its two large users scale with the workload.** Split partials grow with context
  and wave width; slot-state slabs with the number of decoding sequences. A fixed
  block either starves a busy daemon or idles on a quiet one.
- **The kernel scratch only grew.** A launcher's partial pool reached its widest
  request and kept it for the life of the process.

## 2. The replacement: two homes, by lifetime

| Lifetime | Home | Holders |
|---|---|---|
| Within one forward | **The forward's launch scratch**, carved from the forward's own tier (§3) | the launchers' `grow_scratch` blocks: INT8 decode / glue / prefill split partials |
| Longer than a forward, or taken outside one | **Held arenas** — span regions claimed as a tenant, elastic (§4) | slot-state slabs, penalty counts, sampler packs, argmax rows, info ring, device staging scratches, cuBLAS workspaces, the stager's device slab, cold-load staging, capture rows, provenance-scan accumulators |

Nothing is fixed any more except the persistence domain (64 MiB), which is the
persistence thread's copy buffer and keeps its place for the reason it always had: a
copy stream not synchronised to the wave must not have to reason about the boundary.

## 3. Launch scratch: per forward, on the tier

A launcher that sizes its scratch from the launch shape (`grow_scratch.cuh`) now holds
its block **for one forward**. The host keeps a *scratch epoch*, bumped when a forward
opens its tier; a launcher whose block was taken in an earlier epoch takes a fresh one.
The block is carved from a **launch-scratch span** at the top of the forward's tier,
which the forward reserves like its other spans and which goes with the tier.

- **Sized by what recent forwards asked for.** The plan adds the widest launch scratch
  any of the last `RECENT_FORWARDS` forwards requested, plus an eighth — the same margin
  `grow_scratch` always grew by. A prefill's chunks grow a little each forward, so the
  span is ready for the next one; a decode's need is steady.
- **A request the span cannot hold falls through, then teaches the plan.** It is served
  from held ground (§4) for that forward — every launcher keeps working, INT8 decode
  included, which has no unsplit form — and what it asked for is recorded, so the next
  forward's span holds it. Only if held ground has no room either does the launcher get
  nothing, and then INT8 prefill and glue run unsplit (their own fallback: correct, a
  little slower for that launch).
- **Outside a forward** (no forward-wide span open on the device) the scratch is a held
  block, released when the launcher replaces it or the epoch moves.

The scope is the forward-wide span's own lifetime: it opens when that span's generation
opens (`begin_wave(.., LayerPhase::Forward)` — a forward, a step, a draft walk) and
closes when its last generation drops, before the cursor rewinds. Every model and every
caller of the span gets it without naming it.

Graph capture is unaffected: segments are recorded every wave, so a block that moves
between forwards is re-captured with its new address (`docs/decode_graphs.md` §0.1).

## 4. Held arenas: elastic held ground

The held ground's *mechanism* — first-fit extents, release fenced on the stream it was
given back on, a per-site ledger — is kept. What changes is the ground under it: a set
of **span regions** claimed as a tenant (`SlotTenant::Held`) instead of a fixed block.

- **Claimed between forwards, lowest index first**, through the arena window like every
  span tenant. Lowest *at the moment of the claim*, which mid-prefill is high in the
  span; *moved down when idle*, below, is what brings them back low.
- **Contiguous runs** for a block wider than a region: the pool hands out `k`
  consecutive free regions (`claim_region_run`) — the provenance scan's accumulators are
  up to 96 MiB.
- **Provisioned ahead of demand.** Between forwards (`wave_driver::provision_held`, at
  the head of every forward and step), the held arenas keep a *headroom* of free bytes:
  the most any of the last 16 windows took, plus one region. A take inside a forward — a
  slot-state promotion at a 32-token boundary, a sampler pack growing — finds ground
  already there.
- **A take never claims.** A take arrives from arbitrary depth — a slot-state slab under
  the sequence-state lock, a staging scratch under the device's — and a region claim
  sweeps empty arenas and may buy ground from the weight side, each of which takes locks
  of its own. So a take that finds no room goes to the CUDA pool, reported exactly as a
  full held ground was, and is folded into the next provisioning: its bytes into the
  headroom, and its width into an arena of at least that width.
- **Given back when idle.** An arena with nothing live or behind a fence, past the
  headroom, goes back to the span — highest first, so the arenas that stay are the low
  ones — and the weight side grows into it at the next negotiation. An idle engine
  (`Scheduler::settle_device`) forgets the growth record first, so its arenas go back to
  what is held plus one region.
- **Moved down when idle.** Packing lowers the KV beneath an arena claimed high, and an
  arena kept as headroom never falls idle long enough to be given back — so it stood as
  the frontier, with the free ground under it unreachable to the weight side (the gate
  held 99 free regions below a held arena at region 116, the expert zone 1.1 GiB short).
  Every provisioning ends by moving each arena nothing holds onto a free run below it
  (`SpanHeldGround::lower`, `SpanClaims::claim_run_below`): claim the lower run, drop the
  arena. The claims window opens only once `run_free_below` has found such a run, since
  opening it hands back a standing tier.
- **Holders give back too.** The decode slot-state slabs were never freed: a class kept
  every slab it ever carved. A slab whose slots have all come back is now retired, once
  its class has a slab's worth of free slots elsewhere — the spare a 32-token boundary's
  promotions land in. The penalty-count table already shrinks after a run of narrow
  dispatches.

The arenas do not move live buffers: a held buffer's holder keeps its raw address
(recorded into graphs, baked into kernel tables), so relocating one would need every
holder repointed. An *empty* arena holds no address, so it is the arena that moves:
claimed lowest-first, moved down whenever it is empty and lower ground is free, and
released when it is past the headroom — which is what keeps them off the frontier. The
KV and recurrent compactions pack around them.

## 5. What the report shows

The span report breaks the fixed blocks into the held bootstrap (with what the held
ground uses of its whole capacity, its peak, and how many arenas carry the rest), the
persist staging, the dense weights and their padding to the region grid, followed by
the per-site ledger. Under the live KV, a `held arenas` line gives their count, bytes
and top region, marked when that top is the frontier — beside the span tenants
(recurrent state, the QSA index), which are marked the same way. The tier's line is
followed by the launch-scratch reserve each forward's span is priced for.

## 6. Invariants

- A launch-scratch carve never crosses the tier's top: it is a span of the tier,
  bounds-checked like the phase spans (invariant 7).
- A held-arena region is a span region: it counts in `live`, it is below
  `weight_floor` by construction, and the floor guard refuses to cut it.
- No claim is made inside a forward, and none from a take: only `provision` and
  `settle` claim, between forwards, through the claims window; a take finds room or
  goes to the pool.
- An arena moves or goes back only with nothing live in it and nothing behind a fence.

## 7. Measured

Flash-Next batched gate, RTX 4090 Mobile (16 GB), 2026-10-19. "Before" is the gate run
just before this change (held ground fixed at 1,280 MiB), on the same build otherwise.

| | before | after |
|---|---|---|
| fixed blocks | 6,784 MiB | 5,536 MiB |
| held ground in use, peak | 38 MiB of 1,280 | bootstrap 32 MiB + 1–2 arenas (16–32 MiB) |
| launch scratch | held, grow-only | 3.7 MiB per forward, carved from its span |
| expert zone at ×1 | 5,743 MiB (3,016 slots) | 6,991 MiB (3,671 slots) |
| expert hit at ×1 | 42.8% | 50.3% |

| row | decode t/s before → after | bulk t/s before → after |
|---|---|---|
| BF16 ×1 | 15.2 → 16.7–17.2 | 154.0 → 146.9–155.9 |
| BF16 ×4 | 48.3 → 52.2–53.7 | 554.8 → 565.6–586.8 |
| BF16 ×8 | 66.3 → 73.3–74.7 | 446.8 → 606.6–622.3 |
| C5 ×8 | 63.3 → 67.3–69.3 | 437.6 → 581.0–592.9 |
| C10 ×8 | 60.8 → 62.0 | 439.9 → 575.9–602.3 |

Ranges are the two gate runs after the change (with and without forward-scoped
scratch). No holder was refused to the CUDA pool in either.

On main `50c65b683` (one created stream per device), with idle arenas moving down, the
same gate's decode rates are BF16 ×1 17.4 / 20.4 (first and repeat row), ×4 55.1, ×8
77.8, C5 ×8 70.7, C10 ×8 63.7 t/s, and every ×1 and ×2 row ends with the expert zone at
its 6,991 MiB maximum. Without the move, the same tree left a held arena standing at the
frontier from the first ×4 row on, and those rows ended 5,820–6,750 MiB.
