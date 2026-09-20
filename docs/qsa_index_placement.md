# QSA index placement — giving the index the KV cache's positional contract

**Status:** superseded. §1–§4 (the invariant and the three-frame problem) and
§5.1–§5.2 (absolute frame, live tail) still hold. §5.3–§5.6, §7's gates and §8's
"where it stands" describe the placement-time signed-rotation design
(`PlacedPage.roped_base`, `qsa_page_place`'s per-placement rotate) that shipped
first and was measured here — but that design has since been replaced by
rotate-on-load: pages are stored fully un-rotated and the scorer rotates each
key at its own absolute position when it loads it, so a placement carries no
rotation and no `roped_base` state at all. See `docs/progressive_yarn.md` §7 for
the current design and its gates (`tests/qsa_score_rot_harness.rs`). §9's
"where the build departed from the plan" is history, kept for why the signed
rotation existed before it was removed again.

The QSA index is the only positional structure in the engine that cannot be
moved. Its rows carry a rotation baked at the position they were *created* at,
and its page layout is addressed by accumulation rather than by placement — so a
page borrowed into a projection scores at the wrong relative distance, and a span
of injected K/V that carries no rows shifts every page behind it.

The KV cache solved exactly this problem and its solution is the one to copy: a
stored artifact holds no absolute position, a *placement* supplies one, and the
read applies the difference. This document states the missing invariant, shows
the three frames that currently disagree, and specifies the change.

---

## 1. The invariant

> **An index page's rows are position-free. The position is the placement, and
> the scorer works in one frame: absolute slot position.**

Two properties follow, and neither holds today:

- **Relocation.** A page sealed from one conversation and injected at any offset
  in another scores exactly as if those tokens had been forwarded at that offset.
- **Holes.** A span of K/V that carries no index rows is unindexed and nothing
  else. It does not move any other page.

---

## 2. What the code does now

Three frames, and they disagree.

### 2.1 Queries rope at the absolute slot position

`select_layer` builds one position per query row from the sequence's own offset:

```rust
// indexer.rs — the row positions handed to project_queries
for (span, &off) in spans.iter().zip(offsets) {
    positions.extend(off..off + span.len);
}
```

and `offsets[i]` is `session.sequence_offset(s)` (`wave.rs`) — the absolute
position of the sequence in its slot, injected prefix included. So the query side
is absolute. This is the frame everything else must agree with.

### 2.2 Live-tail keys rope at the tail's own ordinal

`append_wave` writes the rope position into the append job as the block's ordinal
in the **live tail**:

```rust
jobs.push(((n_blocks + i) * ratio) as i64);   // indexer.rs, the append job
```

and `flush_open_block` passes `self.n_blocks` the same way. `n_blocks` counts
only rows the tail itself appended — `push_page` never touches it — so a slot
holding 3,000 tokens of injected prefix ropes its first decoded block at position
0 while the query for that same token ropes at 3,000.

`IndexCache::live_blocks`'s own doc states this plainly: *"Blocks the live tail
has completed — the position the next pooled block key ropes at."*

### 2.3 Sealed pages carry their ingest conversation's rotation

A page is closed out of a live cache, so its rows were roped at that
conversation's ordinals. The seal then writes `first_pos` as a literal zero:

```rust
// wave.rs — seal_positional_tail_span
page: IndexPage::new(p.keys.clone(), 0, p.last_cells),
```

Every live construction of an `IndexPage` passes `0`. The field is documented as
"absolute token position this page's first row begins at" and no live caller sets
it, so the rows arrive at a projection still rotated for wherever they used to
be, and nothing records where that was.

### 2.4 Placement is derived by accumulation

Because no page states its position, `IndexCache::block_start` reconstructs one
from the widths of everything pushed before it:

```rust
self.page_tokens[p] + (block - self.page_rows[p]) * ratio
```

This is why `push_gap` exists: K/V injected without rows does not merely leave a
span unindexed, it slides every later page's implied start earlier by exactly its
width. `push_gap` keeps the accumulator honest by advancing the token span
without advancing the row span. It is a correct patch for an addressing scheme
that should not need it.

### 2.5 Scope

QSA is the identity below `selected_width(top_k, ratio)` = 2,051 visible cells —
`qsa_selection_mask` returns `None` and the layer attends densely. So none of
this is reachable in a short conversation. It degrades **retrieval quality at
depth**, in the exact regime the engine exists for, and it degrades it silently:
a wrong rotation produces a plausible score, not a fault.

The cost has not been measured. It should be, and §7 says how.

---

## 3. The pattern to copy — how the KV cache does it

The paged decode kernel stores K **unroped** and rotates at read time from a
per-chunk base carried in the slice header:

```cuda
// int8_decode_kernel.cuh
const int32_t rope_base = (int32_t)slice_rope(sl);
const int32_t rope_pos  = rope_base + (within - off);
apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
```

and the base is **recomputed on every rebuild** from where the chunk sits in
*this* projection, never from where it was written:

```rust
// gpu_chunks.rs — rebuild_decode
let mut rope_base = base_pos;
for (i, chunk) in chunks.iter().enumerate() {
    write_slice_header(..., rope_base, kvheads_ptr);
    rope_base += chunk.usage;
}
```

That is "compute once, inject anywhere" (`docs/attention_provenance.md` §2.3),
and it is the property the index lacks.

---

## 4. Why the index takes the *delta* form, not the KV form

The KV kernel can afford a full per-token rotation because it is already doing a
full attention pass. The QSA scorer cannot: `qsa_score_paged` is a bare dot
product whose whole economy is that a key `float4` is read once and reused across
`TILE_R × H` query accumulators — at the production geometry a thread reads 512 B
of key and does 2,048 FMAs against it, and the kernel is measured at 87.8% of L2
peak at depth.

A per-candidate rotation adds a `rope_dim/2`-float cos/sin row per candidate —
128 B against a 512 B key, a ~25% increase on the traffic that is already the
bottleneck — plus a partner `float4` held live, on an arm whose register budget
is already the thing capping occupancy at 66.7%.

RoPE rotations compose additively: rotating a key already roped at `p` by a
further `Δ` yields the key roped at `p + Δ`. So the index takes the **delta**
form instead — the placement rotation is a constant per page, applied **once per
placement**, not once per score. The hot kernel is untouched.

---

## 5. The design

### 5.1 One frame: absolute

Everything the scorer sees is in absolute slot position. The query already is.

### 5.2 The live tail ropes absolute

`append_wave` and `flush_open_block` add the tail's base to the rope position:

```rust
jobs.push((tail_base + (n_blocks + i) * ratio) as i64);
```

`IndexCache` gains an explicit `tail_base: usize` — the absolute position the
tail opened at, set when the last page was placed or closed. Not derived from
`page_tokens`, so a hole cannot move it.

This is a two-line fix for §2.2 and costs nothing: the tail is a live buffer at a
fixed placement and never moves.

### 5.3 A page carries the frame it was roped in

`IndexCache` stores placed pages rather than bare record pages:

```rust
struct PlacedPage {
    page: IndexPage,
    /// Absolute position this page's rows occupy in THIS cache.
    base: usize,
    /// The frame the rows are roped in. `base - roped_base` is the rotation
    /// the placed buffer applies; zero for a page closed in this cache.
    roped_base: usize,
    /// `[head_dim/4, rows, 4]`, rotated to `base`. The scorer's operand.
    placed: Tensor,
}
```

- A page **closed in this cache** has `roped_base == base`, delta 0 — the live
  path, and it costs nothing.
- A page **injected from a record** has `roped_base == 0` (see §5.4), delta
  `base` — one rotation when it is placed.

### 5.4 A sealed page is position-free

Every seal normalises its rows to the zero frame — rotate by `−frame` — before
encoding. The stored artifact then genuinely holds no position, which is what
makes it injectable anywhere, forever. Four sites do it, all through
`wave::seal_page`: `seal_positional_state`, `seal_positional_range`,
`seal_positional_tail_span` (both its closed pages and its live tail), and
`export_aux_state`.

This deliberately avoids adding a field to the aux blob: `encode_aux` is guarded
by `AUX_VERSION`, and a bump invalidates every index blob in the substrate. A
normalising rotation at seal is a `rows × head_dim` pass on a cold path (a
500-token turn at ratio 4 is 125 × 128 floats) and needs no format change.

**The rotation is therefore signed**, which the plan did not anticipate. A
placement rotates a page forward to a position; a seal rotates one back to zero,
and the tables cannot be indexed by a negative row. The kernel takes the sign
itself — `cos(-x) = cos(x)`, `sin(-x) = -sin(x)`, so one table row serves both
directions and the sign rides on the staged sine. `PlacePage::delta` is an
`isize` and `place::rotate_rows` is the seal's entry point (it un-blocks the
staging back to row-major, because a record stores rows and the scorer reads
channel blocks).

### 5.5 The placement rotation replaces `blocked()`

`IndexPage::blocked` currently builds the scorer's `[head_dim/4, rows, 4]`
staging with `reshape → transpose → contiguous`, cached in a `OnceLock` on the
page. Two problems: it is an allocate-plus-copy behind three tensor ops
(invariant 2), and the cache is keyed to the page rather than to the placement,
so a page cloned from a record and placed twice would serve a stale buffer.

Replace both with one kernel, `qsa_page_place`, that reads `[rows, head_dim]`,
applies a single constant rotation, and writes the channel-blocked layout. It
takes one cos/sin row — the delta's — not a table. The buffer it produces is
owned by `PlacedPage`, so it is keyed to the placement by construction, and
`IndexPage` goes back to being a pure record: rows, `last_cells`, nothing else.

`OnceLock<Tensor>` and the `blocked` field come off `IndexPage`.

### 5.6 The placement becomes authoritative, and `push_gap` retires

With a placement recorded per page, `block_start` reads it instead of
accumulating:

```rust
self.pages[p].base + (block - self.page_rows[p]) * ratio
```

An unindexed span is then a hole and nothing more — no later page moves, so
there is nothing to keep honest. `IndexCache::push_gap` and its zero-row stand-in
page come out, replaced by `skip_to`, which advances `tail_base` and does nothing
else. `push_positional_gap` survives as the trait method — the caller still has
to say how far the slot moved — but it now only moves where the next page opens.

`PagedIndex::new`'s contiguity bail relaxes from *"page `i` starts at token X but
the pages before it cover Y"* to *ascending and non-overlapping*: a hole is
legal, an overlap is still an error.

The paged kernel's prefix mask is unaffected. Rows stay ordered by token
position, so "wholly below this query" is still `g < cnt[r]` — a hole contributes
no rows, which the host-side `candidates_at` walk gets right for free.

### 5.7 RoPE table depth

Three sites size the indexer's rotation table from the tail alone —
`live_blocks() + 1` at `wave.rs` (the flush, the close, and the seal) and
`capacity_blocks()` at `spec.rs`. Under absolute roping they must span the
absolute block extent. The wave's own append path already does: it takes
`index_rope_for(chunked_max_blocks())`, whose table spans `max_blocks ·
CHUNK_SIZE` positions. The other four become explicit rather than
accidentally-sufficient.

---

## 6. Change list

| Where | Change |
|---|---|
| `indexer.rs` `IndexCache` | `tail_base`; `pages: Vec<PlacedPage>`; `block_start` and `candidates_at` read `base`; `push_gap` → `skip_to`; `place_pending` |
| `indexer.rs` `append_wave` | rope position `tail_base + (n_blocks + i) · ratio` |
| `indexer.rs` `flush_open_block` | rope position `tail_base + n_blocks · ratio` |
| `indexer.rs` `push_page` | takes a placement `base`; refuses an overlap |
| `indexer.rs` `score_pages` | reads the `Placement`'s staging; refuses an unplaced cache |
| `place.rs` (new) | `PlacePage` / `Placement` (plan + run) / `place_pages` / `rotate_rows` |
| `paged_index.rs` `IndexPage` | drop `blocked` + its `OnceLock`; `first_pos` → `roped_base` |
| `paged_index.rs` `PagedIndex` | `new` takes `(page, base)`; contiguity → ascending, non-overlapping; `build_tables` and `score_reference` place their pages |
| `qsa_page_place.cu` (new) | signed constant rotation + channel-block, one pass, batched |
| `wave.rs` `seal_page` | the four seal sites normalise to the zero frame |
| `wave.rs` `index_rope_depth` | rope-table depth from the absolute extent, not the tail |
| `wave.rs` `push_positional_state` | places at `next_base` and runs the placement |
| `wave.rs` `push_positional_gap` | `skip_to` |

No change to `qsa_score_paged.cu`, `qsa_index_append.cu`, `qsa_topk.cu`, or the
aux blob format.

---

## 7. Gates

The oracle already exists. `qsa::qsa_selection_mask` is the reference
implementation — it caches raw keys and pools/norms/ropes at read time from
absolute positions, which is by construction the frame this design adopts.

1. **Placement equivalence (the gate this is for).** Build a cache by forwarding
   `N` tokens. Seal the first `M` as a page; build a second cache by *injecting*
   that page at base 0 and forwarding the remaining `N − M`. The two must produce
   **identical** selections. Today they do not.
2. **Relocation.** Same page injected at base `B` with `B` tokens of unrelated
   prefix forwarded first: scores must match a single cache that forwarded
   `B + M` tokens, over the page's rows.
3. **Hole.** Inject a page at base `B` with an unindexed span before it. Every
   page's `block_start` must be unchanged by the span's width — the assertion
   `push_gap`'s tests make, now holding without a gap.
4. **Tail frame.** A cache with pages must rope its first tail block at
   `tail_base`, not 0. Assert the job word directly.
5. **Bit-exactness of the delta.** `qsa_page_place` at delta 0 must be
   bit-identical to today's `blocked()`; at delta `Δ` it must equal
   `RopeTables::apply_at_positions` at `Δ` followed by the blocking.
6. **Measure the recovery.** Run the gate ladder before and after on a deep
   conversation with injected prefix. This is the number nobody has: how much
   retrieval the frame error was costing. Both regimes, per
   `docs/` — below 2,051 QSA is the identity and will show nothing.

---

## 8. Where it stands

Gates 1–5 are built and green. `qsa_page_place` is at the memory floor —
87–88% of peak DRAM, 95–97% warp occupancy, 40 registers a thread, measured with
`ncu` on an RTX PRO 5000 Blackwell; the row tile only separates shapes small
enough to sit in L2, and `tests/qsa_page_place_bench.rs` carries the sweep.

**The decisive gate is `a_sealed_page_selects_the_same_wherever_it_is_placed`**
(`indexer.rs`): forward the same tokens at two offsets, seal one into a
position-free page, inject it at the other, and the selections must match. It was
checked for teeth by stubbing the placement delta to zero — it is the only test
in the file that then fails.

Two gates had to be re-derived rather than re-toleranced, and both are worth
knowing:

- **The kernel is not bit-identical to the host rope, and cannot be.** The
  archive compiles `--use_fast_math`, so nvcc contracts `lo·cos − hi·sin` into an
  FMA while the host rounds the product out first. Measured worst case 6e-8; the
  bound is absolute rather than relative, because an output near zero is a
  cancellation and its relative error says nothing.
- **The composition and round-trip identities are bounded by the TABLES.**
  `cos a · cos b − sin a · sin b = cos(a+b)` and `cos²Δ + sin²Δ = 1` hold in
  exact arithmetic; in `f32` tables they do not, and at position 710 with
  `inv_freq = 1` the stored angle alone carries ~4e-5 radians. Both gates read
  the table rows and derive their own floor, so neither is a tolerance somebody
  picked to make a test pass.

Gate 6 — what the frame error was actually costing retrieval — has **not** been
measured. It needs the gate ladder run before and after on a deep conversation
with injected prefix, in both regimes, since below 2,051 QSA is the identity and
will show nothing.

## 9. Where the build departed from the plan

- **The rotation is signed** (§5.4). The plan assumed placement only ever
  rotates forward; normalising at seal rotates back, and the tables cannot be
  indexed by a negative row.
- **`push_positional_gap` survives**, renamed in behaviour rather than deleted.
  The caller still has to say how far the slot advanced past an unindexed piece,
  because that is where the *next* page is placed from. What it no longer does is
  invent a zero-row page to keep an accumulator honest.
- **The placement's staging is one allocation, sliced.** Per-page allocation
  measured 1.21 ms of `Tensor::empty` against 0.013 ms of kernel on a 1,536-job
  placement — 99% of the work in the allocator. The descriptor table already
  gives each job its own base pointer, so separate allocations bought nothing.
- **`PagedIndex` stays.** The plan flagged it as reachable only from its own
  tests and left the question open. It is the oracle harness `qsa_score_paged` is
  gated against (`tests/qsa_paged_index_tests.rs`), so it was updated to the
  placement model rather than removed — including `score_reference`, which now
  places its pages with the *host* rope, so the oracle and the kernel compare the
  same tensor by different implementations.

## 10. What this does not fix

- **Pooling across a placement boundary.** A page's last row is short because a
  turn boundary is not a block boundary. That is already handled (`last_cells`,
  and `close_page`'s deliberate short block) and is orthogonal.
- **Whether sections should carry index pages at all.** A section injected with
  no rows becomes a legal hole under this design rather than a corruption — which
  makes the question answerable on retrieval-quality grounds instead of being
  forced by an addressing bug.
