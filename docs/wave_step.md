# The wave step: where a residual lives between forwards

A forward's residual stream — the `[rows, width]` hidden state every layer adds
into — normally lives and dies with the forward, on the forward span of the
transient tier. Two things make it outlive one: the **co-batched step**, whose
decode rows cross the layer stack in up to three forwards, and the **creep**,
whose prompt rows cross it over many steps (`docs/continuous_fair_waves.md`).
This document is where each of those residuals lives, for how long, and what it
costs.

## 1. Four lifetimes

| Lifetime | Holds | Home |
|---|---|---|
| Layer | one layer phase's intermediates | the attention / FFN span, reset per phase |
| Forward | the residual a forward works in, its tables, its head | the forward span, one generation per forward |
| Step | the full-sweep rows' residual between a step's segments | the forward span, **one generation per step** (§2) |
| Creep | a prompt's residual between the steps of its sweep | the creep arena (§3) |

Nothing is held for the worst case. The step's buffers are priced per step from
the step's own rows; the creep arena is claimed when a creep forms, sized to its
rows, and released when it reaches the head. A step with no creep, and a
process with no creep in flight, hold nothing for either.

## 2. The step

A co-batched step with a creep in flight is up to three segment forwards:

```
seg1  [0, cursor)        full-sweep rows (decode | verify | glue)
seg2  [cursor, win_end)  full-sweep rows + the creep
seg3  [win_end, N)       full-sweep rows
```

The full-sweep rows' residual passes seg1 → seg2 → seg3. It is a step-lifetime
buffer, so it lives on the transient tier, in a forward span that is opened
**once for the step** instead of once per segment.

### 2.1 Admit every segment, then place one tier

A forward admits its KV before its tier is placed, because a standing tier caps
region claims at its base and an admit refused by it spins the engine
(`end_wave_transient`). A step holds its forward span across all three
segments, so the tier stands from the first segment to the last — and every
segment's admit must therefore run **before** the step places it.

So a step runs in two passes:

1. **Prepare.** Each segment's forward runs its prologue — hand back the
   previous tier, move the boundary in the gap it is legal in, admit the
   segment's KV over its layer range, and claim the write chunk each decode
   row's token lands in (`ensure_decode_writes`, the allocation a whole
   forward's header build makes) — and returns its per-phase price instead of
   placing a tier (`SweepMode::Prepare`). Every claim a step makes is made
   here, so the header build inside the step finds nothing to allocate.
2. **Run.** The step places one tier priced at the per-phase **maximum** of the
   attention and FFN spans (they reset per layer phase, so segments share them)
   and the **sum** of the forward spans (the step's forward generation never
   resets between segments), opens that forward generation, and runs each
   segment's sweep inside it (`SweepMode::InStep`). A segment in a step does no
   prologue, no admit and no placement, and carves its forward-span buffers
   from the step's generation.

While the step's generation is live no arena is created and the weight boundary
does not move: `enter_arena_window` and `wave_is_live` both read the live
generation, exactly as they do for a forward's held logits.

A placement the partition refuses fails the step before any segment has run.
The KV the prepare pass admitted stays claimed; the retry's admit finds it
there and truncates each entry to the length it writes, so nothing is admitted
twice.

Every shape the step depends on — the creep's arena against the rows riding,
its row width and dtype against the model's residual row — is checked before
the prepare pass. A segment's sweep commits its recurrent and KV state, so a
refusal after it would leave that state ahead of a residual nobody wrote.

### 2.2 The full-sweep residual, double-buffered

A segment's residual is the buffer it works in on the forward span: a fresh
segment's embedding, a resumed one's assembled input. Each segment's price
carries exactly one such buffer for its rows (`WaveBuffer::ForwardResidual`,
`WaveBuffer::HyperResidual`), so the step's sum covers every buffer the step
holds.

- seg1 embeds into its buffer and stops short: that buffer **is** its output.
- seg2's buffer is assembled from seg1's full-sweep rows and the creep's
  committed half (§3), then swept in place.
- seg3's buffer is assembled from seg2's full-sweep rows, then swept in place.

Each assembly reads one buffer and writes another, so a segment never reads
what it is writing, and a failed segment leaves its inputs as they were.

## 3. The creep arena

A creep's residual crosses steps, so it cannot be on the tier: the tier is
re-placed every step and moves with the weight floor. It is held the way every
other per-sequence tenant of the span is — the recurrent state, the QSA index
pages, the provenance gallery — in **fixed-stride slot arenas**
(`tenant_arena`, `SlotTenant::CreepResidual`), claimed between forwards when the
creep forms and given back slot by slot when the creep reaches the head, fails,
or is dropped. The claim comes first in its step, after the previous forward's
tier is handed back (`end_wave_transient`) — that tier stands past its guards
until the next forward's prologue, and a claim under it is capped at its base.
A creep whose group re-formed at layer 0 gives back an arena that no longer
fits its rows and claims a fresh one.

A slot holds a **block** of whole rows — as many as fit a region — so a row
never crosses a slot boundary (two slots of different arenas have another
tenant's ground between them, and every address in the span is mapped). Each
block's tensor is anchored to its slot, so the slot cannot go back to its arena
while any view of the block is alive. The waste is less than one block.

### 3.0 Compaction

A creep's arena stands for the whole of its sweep, which beside a busy decode is
many seconds — long enough for the KV side to free the ground under it and leave
its blocks holding the frontier. So it compacts as every slot tenant does
(`compact_creeps`, from the scheduler's `compact_span_tenants`):
`plan_slot_moves` plans moves only for the arenas holding the frontier, with
their destinations claimed; each moved block is copied on the primary stream —
behind every launch that read or wrote it — and the block is repointed at its
new slot (`relocate_tensor`); a destination nothing moved onto goes back. It
runs between forwards on the scheduler thread, which is the creep's only holder,
so no step holds a view of a block while it moves; and planning takes the arena
window, which a forward refuses, so a pass asked for mid-forward fails rather
than racing it.

### 3.1 Two halves, flipped on commit

The arena holds **two** halves of the creep's rows. A window reads the
committed half and writes the other; the step flips them only once the window's
segment has succeeded. A window that fails mid-sweep — a refused tier, a device
OOM — leaves the committed half untouched, so its retry resumes from exactly
the residual the failed attempt started from. (A resumed forward used to add
into the caller's residual in place, so a failure part-way left it advanced by
the layers that ran.)

A paused creep keeps its arena as it stands. A member that leaves mid-sweep
takes its rows out by a copy of the survivors' rows into the uncommitted half,
which then becomes the committed one.

### 3.2 A creep without a step

A sweep paused between forwards with nothing beside it — a single prompt run a
layer range at a time — holds its residual the same way: an arena of two
halves, read and written by the standalone forward (`ResidualIo`).

## 4. Row order

A forward packs its rows in **internal** order
`[decode | single-token prefills | multi-token prefills | glue]`; callers name
rows in **caller** order `[decode | prefill (caller order) | glue]`. The two
differ only by where the single-token prefills sit, so a mapping between them is
a handful of runs (`row_map`). Every copy into or out of a residual — assembling
a segment's input, writing the creep's rows to the arena, taking a member's rows
out — is computed by composing the source's and the destination's maps into
`(source row, destination row, length)` copies. No buffer is ever reordered in
place and no intermediate is built to reorder through.

## 5. What this replaced

The residual between forwards used to be a session-held pool of
`WindowResiduals` buffers on the reservation's held ground, widened at load by
a runtime reserve priced at the widest forward the scheduler can send — four
buffers of `prefill cap × 1.25 + 1,024` rows. On Flash-Next that was 1,280 MiB
of held ground on every card, whether or not anything ever crept. On the
16 GB laptop it took the expert zone from 5,726 to 3,694 MiB and left BF16 ×8's
prefill tier 128 MiB short. The reserve, the pool, the caller↔internal reorder
buffers and the segment splice buffers are gone; the step and the arena hold
exactly the rows in flight, for exactly as long as they are in flight.
