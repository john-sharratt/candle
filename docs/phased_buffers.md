# Phased buffers: a transient tier placed by a typed program

The transient tier is the device ground one forward's intermediates live on — a
layer's activations, the expert GEMMs' gathered rows, the DeltaNet scan's
transients, the head's logits. It is bought per forward from the span between
the KV frontier and the weight floor, and on a card whose experts already want
the whole span it is bought **from the expert zone**: the zone concedes ground
for the forward and grows back empty after it, and decode pays for every expert
it dropped (`docs/wave_feeder.md`, `crate::prefill_width`).

So the tier's size is decode throughput. This document is how it is made small:
by placing it from one typed description of what a forward allocates, in which
buffers that are never live at once share ground and the compiler refuses a
program that would read a buffer after its ground was reused.

## 1. Where the tier stood

Three bump arenas, one per `LayerPhase`, laid side by side
(`candle-nn/src/kv_cache/chunked/bump_arena.rs`, `begin_wave`):

```
| Attention span | FFN span | Forward span |      tier = the sum of the three
```

* **Attention** — attention norm through `o_proj`, or a DeltaNet mixer. Reset
  every layer.
* **FFN** — pre-mix, router, gather, gate/up/down GEMMs, combine. Reset every
  layer.
* **Forward** — what crosses every layer: the residual stream, the
  ragged-prefill tables, the token ids, the DeltaNet decode table, the head.

Each phase is a bump cursor: every buffer it carves stays live until the phase
resets, so a phase costs the **sum** of its buffers (`WavePlan::phase_bytes`).
Two kinds of slack follow.

**The layer phases never coexist, yet both were reserved.** A layer's
attention generation drops before its FFN generation opens; whatever crosses
from one to the other is on the forward span. On Qwen3.8-Flash-Next at 3,072
rows the attention phase (the DeltaNet chain) prices ~1.2 GB and the FFN ~1.0 GB
— ~2.2 GB of the ~2.4 GB tier, for spans never live together.

**Inside a phase, dead buffers keep their bytes.** The census of a DeltaNet
layer at 2,100 rows (`wave_plan.rs`,
`the_pre_mix_and_live_delta_net_match_the_census`):

```
pre-mix norm 86 MB → gate raw 86 → mixed 21.5 → operand 6
→ stacked projection 138 → its split 138
→ conv 86 → scan out 51.6, u 51.6, w 51.6, kq 25.8, decay 0.4
→ norm-gate 51.6 → w_out operand 14.5 → w_out 21.5           summed: 831 MB
```

The pre-mix's buffers are dead once `mixed` is written, the projection once it
is split, the scan's transients once the scan has run. The live set peaks during
the scan — split, conv, scan out, `u`, `w`, `kq` — at ~404 MB, 49% of the sum.

A lifetime-assigned layout was built once, measured at ~1.75× on a MoE layer,
and removed (`WavePlan::phase_bytes`): every buffer declared the step it died on
by hand, with nothing tying the declaration to the code that dropped it. That is
the failure this design is shaped around.

## 2. What the compiler can and cannot prove

**It can prove the aliasing discipline.** A buffer handle cannot be used after
the scope that owns its bytes has ended; two live handles never share bytes; a
value that crosses phases must come from a scope that outlives them. The borrow
checker does this, at no run-time cost, if every buffer borrows the region it
lives in. `Generation::alloc` already holds a range to its guard's lifetime, so
a wave intermediate cannot be named after its phase drops — the property the
shared layer region rests on.

**It cannot prove capacity.** Sizes are run-time values — rows, sequences,
context depth. So capacity is checked **when the tier is built, before the first
kernel**: the builder places every buffer or refuses. A forward then never runs
out of span part way through; a refusal is a narrower width for admission, not a
fault.

**It does not see the device.** A scope ending on the host is not the device
finishing with its bytes. Reuse is safe because every reader on the forward
path is ordered on the one stream that writes the next occupant. A reader on
another stream must hold a buffer typed for it — released only behind an event
recorded after its last use — so that guarantee cannot be broken by accident.

## 3. The design

### 3.1 Types

```rust
/// The forward's reservation: placed once, before the first layer.
pub struct Tier<'t> { base: DevicePtr, len: usize, .. }

/// What crosses phases — residual, prefill tables, token ids, the head.
/// Carved first; lives for the whole forward ('f outlives every phase).
pub struct CarryOver<'f> { .. }

/// The one region every layer phase works in, in turn.
pub struct PhaseRegion<'t> { .. }

/// A phase holds the region by `&mut`, so two cannot be open at once, and
/// every buffer carved in it borrows the phase, so none outlives it.
pub struct Phase<'r, K: PhaseKind> { region: &'r mut PhaseRegion<'r>, mark: usize, .. }

/// A typed device buffer whose bytes are borrowed from the scope that carved it.
pub struct Buf<'s, T: Element> { ptr: DevicePtr, rows: usize, cols: usize, .. }
```

Phases are exclusive by construction: opening the FFN phase needs the region
`&mut`, which the attention phase holds until it is dropped — and it cannot be
dropped while a `Buf<'attn, _>` is alive.

### 3.2 Liveness inside a phase: nested scopes

```rust
attn.scope(|s| {
    let mixed = s.carve::<F32>(rows, hidden);
    s.scope(|t| {
        let norm = t.carve::<F32>(rows, streams * hidden);
        let gate = t.carve::<F32>(rows, streams * hidden);
        premix(&norm, &gate, &mut mixed.write());
    });                                 // norm, gate: bytes reused below
    let proj = s.carve::<F32>(rows, conv + value + 2 * heads);
    s.scope(|t| {
        let (u, w, kq) = (t.carve(..), t.carve(..), t.carve(..));
        scan(&proj, &u, &w, &kq, ..);
    });                                 // scan transients: reused below
});
```

A scope is a stack mark: its carves are released when it returns. An op takes
`&` inputs and `&mut` outputs, so it cannot write the buffer it reads. Chains
that alternate use a ping-pong pair:

```rust
pub struct PingPong<'s, T> { a: Buf<'s, T>, b: Buf<'s, T>, front: bool }
impl<'s, T> PingPong<'s, T> {
    /// Reads the front, writes the back, swaps.
    pub fn step(&mut self, f: impl FnOnce(&Buf<'s, T>, &mut Buf<'s, T>));
}
```

### 3.3 One program, two placers

Each chain is written **once**, generic over who places it:

```rust
pub trait Placer {
    type Buf<T: Element>;
    fn carve<T: Element>(&mut self, rows: usize, cols: usize) -> Self::Buf<T>;
    fn scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R;
}

/// Arithmetic only: walks the program, keeping the stack's high-water mark.
pub struct Sizing { cursor: usize, peak: usize }
/// Places real buffers in a reserved region.
pub struct Device<'r> { region: &'r mut PhaseRegion<'r>, .. }
```

* **Admission** asks `PhasedBuffers::total_size(geometry, width)`: the program
  run with `Sizing`. Pure arithmetic, cheap enough for the rate model's
  bisection, and exactly what the forward will place, because it is the same
  code. It replaces `WaveBuffer`, the hand-kept inventory the census caught
  1.8× and 2× wrong.
* **The build**, between admission and the first kernel, runs it with `Device`:
  everything is placed, or the build refuses.
* **Placement is deterministic per shape** — the same width gives the same
  addresses — which is what a captured graph segment needs to replay.

## 4. Stages

Each stage is measured against the last before the next is built.

1. **The shared layer region.** *Built.* The attention and FFN spans share one
   region, the larger of the two, with the forward span beside it
   (`wave_plan::tier_footprint`, `phase_offsets`); `begin_wave` refuses to open
   a layer phase while the other is live. Expected: the tier ~41% smaller on
   Flash-Next at the same width.
2. **The `Placer` program for Flash-Next's chains** (`qwen4exp`: the DeltaNet
   and attention mixers, the MoE FFN, the head), with `Sizing` pricing them in
   place of their `WaveBuffer` entries and nested scopes releasing what each
   step leaves dead. The census becomes the test that the program matches the
   device's peak.
3. **The other families**, one at a time — the generic path, `qwen35`,
   `latent_moe` — until `WaveBuffer` is gone.
4. **The layout copies** (hot-path invariant 2): the DeltaNet split and the
   pre-mix's raw gate become strided reads, removing their buffers outright.
5. **The prefill width table** (`candle-conversation/src/prefill_width.rs`)
   re-derived against the tier per row that results.

## 5. Stage 1 in detail

```
| Attention span ┐          | Forward span |      tier = max(attention, FFN) + forward
| FFN span ──────┘          |
```

* **Priced** by `WavePlan::tier_bytes`, through `tier_footprint`, rounded to a
  whole region as the placement rounds it.
* **Laid out** by `begin_wave` from `phase_offsets`: both layer arenas rebased
  at the tier's base, the forward arena past the larger of them.
* **Kept exclusive** by `begin_wave`, which refuses a layer phase while the
  other has a live generation — the run-time half of what §3.1's types make a
  compile error. Every production site opens them in sequence: the layer loops
  drop the attention guard (at a brace or explicitly) before the FFN's opens,
  and the PLE block, the embedding staging, the draft walks and the head passes
  open the attention phase alone.
* **Safe across the reset** because a range borrows its guard: the attention
  output the residual add reads cannot be named once the attention guard drops,
  so nothing reads it after the FFN starts carving over it. The single stream
  orders the FFN's first write after the attention phase's last read.
