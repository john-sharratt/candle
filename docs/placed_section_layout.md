# Placed-section layout: a reserved position range, append-in-front, per layer

**Status:** design, not built. Written 2026-11-27 from the sealed-fresh
investigation of the files collection (`docs/provenance_system_collection.md`,
`docs/results/files_collection_rollout_2026-11-26.md`).

## 1. The problem

The knowledge base's files are **sealed members**: each is prefilled once,
behind the prompt up to `files_open`, and spliced into a turn's system prompt
by the projection. Its K/V is stored un-rotated and rotated at read time to
wherever the splice puts it (`rope_base` per chunk, `gpu_sync.rs`), and its
QSA index pages are likewise position-free and rotated at placement
(`qwen4exp/paged_index.rs`).

Today the placement is **sequential**: the files sit one after another after
the seed, and every chunk's position is `base_pos` plus the cumulative usage
of the chunks before it. Three things follow, each measured on BlackWorld with
the sealed-fresh harness (`POST /v1/substrate/zend/sealed-fresh`):

1. **A file's K/V is read at positions it was not computed at.** Every member
   is sealed at the positions right after `files_open` (call that *P0*). Placed
   third, it is rotated to P0 plus the first two files: its internal relative
   positions survive, its distance to the prefix it was computed behind does
   not.
2. **Sealed content placed after a file never saw it.** With the seed sealed
   once and placed after the files, the question read the seed's K/V — computed
   with no file ahead of it — and the model reported a placed file's content
   absent. A transplant of the sealed K/V into a fresh slot reproduced the
   failure; prefilling the seed live over the sealed file fixed it. Placing the
   seed before the files (`2c38f523c`) fixed production.
3. **More placed content degrades the read.** With the needle file followed by
   14 other files (52,259 tokens) and the tools in the prompt, the sealed side
   decided the file's content "may be outdated" and called `file_read`, 3 runs
   of 3; with the tools off it read the file. Swapping in a recurrent state
   that had run over the whole prompt fixed it — see §6.

Moving a placed section under the tokens already decoded is the other hazard
any re-layout has to avoid: their hidden states were formed against the old
relative positions.

## 2. Goals and non-goals

**Goals.**

- Each placed section is read at positions consistent with how it was
  computed, and its distance to the prompt before the collection does not
  depend on how many other sections are placed.
- Nothing already placed moves while a turn decodes, so every relative
  position a decoded token was formed against stays true.
- Each full-attention layer holds the sections that layer's attention is
  using, at positions in the order it came to use them.
- The prompt before the collection and everything after it keep fixed
  positions however the collection grows or shrinks.

**Non-goals.**

- The recurrent (DeltaNet) layers. They have no positions and never run over
  placed sections (`provenance_system_collection.md` §8, D8); nothing here
  changes that. The gap it leaves is §6.
- Re-ranking content whose order carries meaning. The dialogue's turns and
  the conversation's journal are laid out in the order they were born, never
  by relevance (§3.7).

## 3. The design

### 3.1 A reserved range per placed section

Every placed section of the system prompt — the **tools** section and the
**knowledge base** (`files`) — is given its own reserved range of positions
`[B, B + R)`, the same in every layer. The prompt between two ranges keeps
fixed positions, and so does everything after the last one (`system_close`,
the user turn, the decode), which starts at the top of the last range
whatever the sections hold. With the regions in schema order (the journal's
range is birth-ordered, §3.7):

```
| persona, grounding | journal range | tools range | tools_close … files_open, seed | files range | system_close | dialogue (window, then the request) →
0                    B_j       B_j+R_j B_t   B_t+R_t                              B_f    B_f+R_f
```

A section fills its range **from the top down**: the first member placed
ends at `B + R`, the next ends where that one begins, and so on. Positions in
the range no member holds are a **hole**: they carry no K/V and no index
rows. A hole is legal in attention (RoPE depends only on the positions that
are present) and in the index (`PagedIndex` already accepts a hole between
pages, and `push_positional_gap` places one).

`R` is the section's **projection budget** (§3.5) in tokens. It is not a
constant: a range far larger than what can be placed pushes the prompt before
it further from the question for nothing (§5, risk 1).

The gap is part of the prompt **as generated**, not only as projected. The
birth prefix — the system prompt the checkpoints are built from and every
member is sealed behind — is emitted with each range empty: the tokens after
a range start at its top, exactly as they do in a projected turn. A turn with
nothing placed and a turn with a full range therefore put `tools_close`, the
seed, `system_close` and the question at the same positions, and the prompt's
own K/V and the checkpoints do not depend on what is placed.

### 3.1b Members are sealed at the end of their range

A member is sealed (at ingest, and at any re-seal) at the **top** of its
section's range: its `T` tokens at `[B + R − T, B + R)`, behind the prompt up
to the range's start, across the empty range. It is therefore computed at the
distance from the prompt before it that it has when it is placed first — the
member the opening ranks strongest, the one nearest the question. Placed
there, its K/V and its index rows are what a native prefill of that prompt
would have produced, apart from the recurrence (§6). Placed lower, only its
distance to the prompt before the range changes, by the members above it.

The stored K is un-rotated and the index rows are position-free, so the
placement still rotates them wherever they go; what the sealing position
fixes is the hidden state each member's tokens formed against the prompt.

The sealing position depends on `B` and `R`, so both are part of a member's
**content address** (`persistence/content_hash.rs`): a change to a section's
budget or to the prompt before it re-seals its members rather than reading
K/V sealed at a different distance. The ingest running on 2026-11-27 seals at
the positions straight after `files_open` (the range of size 0); the first
ingest after this lands seals at the top of the range and, keyed by the new
address, replaces those members rather than mixing with them.

### 3.2 Order, growth and removal

- **The opening** places its sections in relevance order with the strongest
  at the top of the range, nearest the question (*Lost in the Middle*: a long
  context's start and end are read best; the end is what the question is
  nearest).
- **A section that comes into scope later** is placed **in front** of those
  already in scope — immediately below the lowest placed one — never between
  them and never above.
- **Nothing placed moves.** Every relative position among placed sections,
  the question and the decoded tokens holds for the life of the turn.
- **A section that falls out of scope** is removed and leaves a hole. Its
  pages stay in the pool, so the shadow scan
  (`docs/hit_driven_projection.md` §7.2) can bring it back; brought back, it
  is placed in front again, at new positions.
- **The range running out** — the front reaching `B` — is the one case that
  relays the whole collection out, and only at a reprojection boundary, where
  the slot is rebuilt anyway.

A section's position therefore records when it came into scope, not its rank.
That is the price of never moving anything.

### 3.3 Per layer

K/V is held per layer (one backing per layer) and so is the QSA index (one
`IndexCache` per full-attention layer, scored at every step). Each
full-attention layer therefore holds **its own** set of placed sections at
**its own** positions within the range:

- a layer's scope is the sections its own selection has been reading;
- its placement order is the order they came into its scope;
- a section in one layer's scope need not be in another's.

The query of a given token has one position across layers, above the range in
every one, so every placed section is visible to it in every layer that holds
it.

### 3.4 Where the scope comes from, and when it changes

QSA selection already scores every candidate block of a layer at every step
and keeps the top ones. Those selections are the hits: a section's hits in a
layer are the selected blocks of that layer that fall in the section, counted
over the steps since the last boundary.

Scope changes **at reprojection boundaries only** (the probe cadence,
`reproject_probe_tokens`), never per token:

- per-token change makes each token's context differ from the last one's, and
  that churn is where incoherence comes from;
- a boundary is already where the slot is rebuilt and the host decides;
- a speculative draft and its verify then always see the same layout, because
  the layout only changes between drafts.

A section enters a layer's scope when its hits there cross an entry threshold
and leaves when they stay below an eviction threshold for a boundary —
hysteresis, as belief selection already uses, so a section near the line does
not flap.

### 3.5 Three budgets per section

Each placed section carries three budgets, in tokens, nested from the widest
to the narrowest. They are the stages the knowledge base already runs
(`docs/hit_driven_projection.md` §7.2, `docs/provenance_system_collection.md`
§4); what is new is that each is declared per section and the narrowest one
is a position range.

| Budget | What it bounds | Who reduces into it | Today |
|---|---|---|---|
| **Index scan** (`scan`) | The members whose sampled-layer index rows are on the device and scored by the shadow scan each boundary. No K/V moves for them. | Provenance: the gallery scan and the working set's momentum choose whole files. | The pool: `PoolSize::for_card` (2M / 500K / 256K tokens, 72 / 24 / 16 GB), shared across the pool's kinds. |
| **Residency** (`resident`) | The members whose K/V is hot — on the device, in the KV-regions tenancy — whether placed or not, so the fill can place one at a boundary without a PCIe or redo-log wait. | The index: real and shadow hits rank the scanned members; the strongest stay hot. | Not a budget. A pooled member's K/V stays wherever it is (hot, warm, cold) until the fill places it (§7.2 step 3), and a cold one is placed a boundary late. |
| **Projection** (`project`) | The tokens placed in the section's range: `R`. The positions the section takes, and the K/V and index rows the selection attends. | The index again: the fill places chunks by `base·(1+gain·h)` under the budget. | The fill's `readings` cap, a share of the room (22.5%, ≈ 52K at a 233K room); tools by count (`budget: {min: 1, max: 5}`). |

`scan ≥ resident ≥ project`, and each stage reduces into the next:
provenance proposes the scan set from the unbounded corpus, the index
reduces it to what is kept hot, and the index places the projection from the
hot set. A member leaving the projection stays resident while it still ranks,
so it can come back in front (§3.2) at the next boundary at device speed; one
leaving the resident set keeps its index rows while it stays scanned, so the
shadow scan can still raise it.

**Per section.** The tools section and the knowledge base each declare their
own triple in `projection.yaml`, beside the section:

- **Knowledge base:** `scan` is the files' part of the pool, `resident` a
  card-sized budget between it and `project`, `project` the range.
- **Tools:** the catalog is small enough that `scan` is the whole catalog.
  `project` moves from a count of signatures to tokens, so the tools range is
  a fixed size and the belief selection fills it by score until the next
  signature does not fit. Its minimum (one signature) stays.

The projection budgets are fixed for the dialect: they are positions, so a
card does not change them, and the same turn lays out identically on all three
machines (the prescribed model is the same everywhere). `scan` and
`resident` are device memory and are sized per card, as the pool is today.
The room the dialogue's fill has is the slot less the system prompt **with its
ranges at full size**, so the dialogue's share never depends on what the
sections hold.

### 3.6 Two orders: by relevance and by birth

Every piece of provenance the projection places falls under one of two
rules. Both give it a fixed range of positions during a request; they differ
in how the range fills and when anything leaves.

| Rule | Order | Fills | Changes | Leaves |
|---|---|---|---|---|
| **By relevance** (§3.1–§3.5) | When it came into scope | Top down, each new member in front | At any reprojection boundary, mid-request included | At a boundary, leaving a hole |
| **By birth** (§3.7) | When it was written | Bottom up, each new piece after the newest | Appends only, during a request | Only between requests, the oldest first |

What every piece of provenance is, today and under this design:

| Provenance | Where it sits | Today | Rule |
|---|---|---|---|
| Tools | System prompt, `tools` collection | Belief selection, 1–5 signatures, re-chosen at boundaries | Relevance; projection in tokens (§3.5) |
| Knowledge base: readings, folders, guides, journal notes | System prompt, `files` collection (`provenance_system_collection.md` §4, D19) | Pool → fill under the kind shares | Relevance |
| The seed | System prompt, before the `files` range | Sealed once, before the files (§6.2b) | Neither: fixed prompt below the range |
| The conversation's journal | System prompt, `journal` collection (pinned), after the grounding and before the tools | A chain of sealed sections; a new entry joins at the next round's head, mid-request included; a fold or a forgotten entry reseals everything after it (`docs/zend_journal.md` §4) | **Birth** |
| The dialogue's turns, with its inherited lineage | After `system_close` | Floors, then rounds by score; re-chosen at every boundary, mid-reply included, and the slot renumbered under the reply (`docs/hit_driven_projection.md` §7.2 steps 4–5, §7.3) | **Birth** |

Journal *notes* are files of the knowledge base, chosen by relevance among
every journal the workspace holds. Notes from many journals do not share one
order of birth, and a note entering mid-request older than one already placed
would have to be inserted between placed members, which moves them. They
stay under relevance, with their recency weight (`half_life_days: 30`) in
their score.

### 3.7 Birth-ordered regions: a window, append at the end, prune between requests

A **request** is the user's turn and every tool round it takes, until a reply
ends without a call (`docs/zend_journal.md` §6). The rule for the journal and
the dialogue is stated in requests, not in boundaries:

- **The window is the budget.** Each birth-ordered region has a window in
  tokens. At a request's start it holds the newest pieces that fit, in birth
  order; older pieces have dropped out of the far end. For the dialogue the
  inherited lineage is its floor, ahead of the window, and never drops (as
  `select_conversation` keeps it today).
- **During a request nothing leaves and nothing moves.** New pieces — the
  request's turns as they seal, the journal's new entries — are appended after
  the newest at the next positions. No turn is shed, no round is re-chosen by
  score, the slot is not renumbered under the reply, and the journal is not
  folded. Every relative position a decoded token was formed against holds
  until the request ends.
- **Growth burns the reserve first, then a YaRN rung.** A request that grows
  past its window takes positions from the native context held in reserve for
  it (a small request ends there, on rung 1). One that passes even that — a
  read-heavy request, a long agent loop — runs on past `L0` on the rung its
  reach needs (`docs/progressive_yarn.md` §8), as one exchange whose floors
  pass the slot does today. The rung changes as the reach crosses it; that is
  the one change of rotation a request can see, and only an extreme one sees
  it.
- **Between requests the region is pruned and re-seated.** Once a request
  ends, and before the next is prefilled, the oldest pieces drop until the
  region fits its window again, and the region is re-seated at the bottom of
  its range. Keys and index rows are stored un-rotated, so re-seating rewrites
  nothing (`hit_driven_projection.md` §4). The next request prefills against a
  stable layout, on rung 1 again. The journal's folds and forgotten entries,
  which reseal everything after them, run here too.
- **A turn is read where it was written.** Within a request every turn keeps
  the position it was computed at, so the dialogue's own K/V is read exactly
  as computed. A prune between requests shifts the survivors down together,
  which keeps their positions relative to one another.

The regions and their ranges:

- **The journal** has its own range in the system prompt, between the
  grounding and the tools range, so its growth never moves what follows it:
  its window (`pin_tokens`, 1,024, bounds what is pinned of it) plus a
  request's reserve of entries. It fills from the bottom, as its chain is
  sealed: each entry is sealed as the continuation of the ones before it at
  the next position, so the range holds the chain exactly as it was computed.
  An entry a request writes past the range's reserve waits for the next
  request's prune; the turns it was written from are in the dialogue until
  then.
- **The dialogue** is the last region and has no range above it: it grows
  into the slot. Its window is its share of the room; the rest of the slot
  above the ranges and the window is the request's reserve.

This replaces, for the dialogue, the per-boundary fill of its rounds by score
and the renumbering under a reply in progress (`hit_driven_projection.md`
§7.2 steps 4–5, §7.3), and makes `reserve_tokens` a request's reserve rather
than one cadence's append (§4). Older turns are reached through the journal:
every entry carries the turns it was written from (`journal_read`,
`zend_journal.md` §4.1). Those sections of `hit_driven_projection.md` are
amended when this is built.

### 3.8 Models without a QSA index

A model whose attention reads every position has no index rows, no selection
and no shadow scan, so it has no index stage to reduce through:

- there is no `scan` budget; provenance's ranking is the only one;
- `resident` and `project` are filled from provenance's ranking directly,
  strongest first, `project` sized so that the whole prompt fits the model's
  native context (the flexbox sizing that `docs/hit_driven_projection.md`
  §7.2 already applies to a model that does not select);
- the ranges, the gap in the generated prompt, sealing at the top of a range
  and the front insertion apply unchanged: they concern positions, which
  every model has. Per-layer scope (§3.3) does not apply — without a
  per-layer selection there is no per-layer evidence — so every layer holds
  the same members.
- the birth-ordered regions (§3.7) apply unchanged: the window, appending
  during a request, the reserve, a rung past it, and pruning between
  requests.

## 4. Engine changes

### 4.1 Explicit positions for placed windows

Today each chunk's `rope_base` is `base_pos` plus the cumulative usage of the
chunks before it in its layer's backing (`gpu_sync.rs` `rope_bases`,
`gpu_chunks/guard.rs` `rebuild_decode`). The kernels already read one
`rope_base` per chunk from the slot state; what changes is how the host
assigns it.

- A placed window carries the position it was placed at; its `rope_base` is
  that, not the running sum.
- The slot keeps a **position frontier** apart from its token count: the next
  position a live chunk takes. Live chunks — `system_close`, the user turn,
  the decode — take positions from the frontier, which starts at `B + R`.
- `sequence_offset` (tokens held) and the frontier (next position) are two
  numbers. Every caller that today reads one as the other is found and told
  which it means; a query's position is the frontier, never the token count.

### 4.2 The index window

Each layer's `IndexCache` places injected pages at a position, and the page
list must ascend and not overlap (`PagedIndex::new`). Placing in front means
**inserting at the start** of the list, which keeps it ascending; a hole is
already legal. The candidate walk ("rows wholly below this query") reads
positions off the pages, so a query above the range sees every placed page in
every layer. Per-layer scope means each layer's window holds its own pages.

### 4.3 Per-layer descriptor tables

Each layer's backing is already separate, so a layer whose scope differs from
another's simply holds different chunks. The work is in the projection
assembler, which today injects the same pieces into every layer: it injects a
layer's own scope into that layer, and records per layer what it placed.

### 4.4 The hits stay on the device

The selection tables are on the device. Counting a section's hits, comparing
them to thresholds and choosing what to place must not read them back
(`AGENTS.md` hot-path invariants 3 and 4): a device buffer accumulates hits
per (layer, section) across a boundary's steps, and the boundary's decision
reads that summary through the mapped ring the MoE dispatch already uses for
the same purpose — off the critical path, waited on by nothing on the device.

### 4.5 Position budget

The top of the last range plus the turn must stay within the positions the
model reads well — its trained length and the YaRN schedule
(`rope_schedule::FactoredRope` selects a rung by position). Every range
pushes everything after it up by its full `R` whatever it holds, so the sum
of the projection budgets comes out of the slot's room (§3.5) and the slot
stays on rung 1 (`L0 − reserve_tokens`).

### 4.7 The assembler places ranges

The projection's assembler emits a section's range as a positional gap of
`R` minus what it places, ahead of the placed members, and the members top
down from the range's end. The same assembler emits the birth prefix with
every range empty (§3.1). The selection hands it members and their token
counts; it no longer decides positions by concatenation.

### 4.6 The drafter

A speculative draft attends the same layout as the turn it drafts for. The
layout changes only at boundaries (§3.4), between drafts, so a draft never
spans a change.

## 5. Risks

1. **The gap distances the prompt from the question.** Long-distance
   attention is weaker; instructions and persona above the range could lose
   grip. Measured before anything else (§7 stage 1).
2. **Positions no longer encode relevance.** A section brought in late sits
   furthest from the question. The opening's order is by relevance, but a long
   turn's later arrivals are not.
3. **Per-layer scope means a section may be read by some layers and not
   others.** That is what per-layer selection already does inside attention;
   holding it in the layout makes it permanent between boundaries.
4. **Re-entry moves a section.** A section evicted and brought back is placed
   at new positions — a move, but only of content no decoded token was
   attending when it left.
5. **The range running out forces a relayout**, which moves everything once.
   Sized to the budget, it should be rare; it is counted and logged.

## 6. What this does not fix

The 52K-after-the-needle failure turned on the recurrent state: the sealed
K/V and the sealed index, under a recurrence that had run over the whole
prompt, read the file; under production's recurrence, which never runs over
placed sections, it did not. No RoPE layout reaches the DeltaNet layers. That
gap — placed content the recurrence has never read — needs its own design:
running the recurrence over placed sections at a projection, or recomputing
the few tokens whose state deviates most (CacheBlend-style selective
recompute), are the candidates.

## 7. Staged plan

Each stage has a gate measured on the sealed-fresh harness, the needle cases
of §1, and a multi-file case (one question about the fifth of ten placed
files; one question needing two files).

0. **Order alone.** On today's sequential layout, place the most relevant file
   last against first. *Gate:* whether position moves the 52K-after-needle
   answer at all.
1. **Explicit positions** (§4.1, §4.2). *Gate:* unit tests asserting the
   exact `rope_base` of every chunk and the exact page bases of every window
   for a layout with a range, a hole and a front insertion; the gate models
   unchanged when every range is empty.
2. **The ranges, static, shared across layers** (§3.1, §3.1b, §3.5, §4.7).
   Both sections' ranges sized by their projection budgets, the gap in the
   birth prefix, members sealed at the top of their range under the new
   content address, the tools selected by tokens. *Gate:* no loss on the
   needle cases and on persona/instruction adherence against today's layout;
   the sealed-fresh harness with the needle placed first matches a fresh
   prefill token for token up to the recurrence's divergence.
3. **Append in front at boundaries, shared across layers, and the residency
   budget** (§3.2, §3.4, §3.5). *Gate:* the multi-file case improves or
   holds; nothing placed moves within a turn (asserted from the per-boundary
   placement record); a member re-entering from the resident set is placed at
   the boundary it is chosen, not one later.
3b. **Birth-ordered regions** (§3.7): the journal's range, the dialogue's
   window, appending only within a request, the reserve and the rung past it,
   pruning and re-seating between requests. *Gate:* asserted from the
   per-boundary placement record that no dialogue turn or journal entry
   leaves or moves within a request, on a multi-round agent request; a
   request that passes the reserve runs on the next rung and the following
   request starts on rung 1; the needle and multi-file cases hold.
4. **Per-layer scope** (§3.3, §4.3). *Gate:* per-layer placement records; the
   needle and multi-file cases hold or improve against stage 3.
5. **On the device** (§4.4). *Gate:* no readback on the per-token path
   (`AGENTS.md` invariants 3–4) and decode throughput within noise of stage 4.

## 8. Alternatives considered

- **Overlapping positions.** Every member at P0, so each is read exactly where
  it was sealed. Many keys at each distance is a pattern the model never saw,
  and order is lost; parallel-context work (Parallel Context Windows,
  Block-Attention, APE) finds it degrades as windows multiply without
  fine-tuning. QSA selection would keep collisions rare for focused queries
  (as InfLLM's fixed-distance memory does) but not for cross-file ones, and
  the index window refuses overlapping pages today.
- **Reordering every step by attention.** Attention Sorting does this between
  passes. Within a decode it moves content under decoded tokens every step;
  append-in-front keeps its benefit — what is attended stays in scope, what is
  not leaves — without moving anything placed.
- **Selective recompute** (CacheBlend). Recomputes the tokens whose sealed K/V
  deviates most from a full prefill. Complementary rather than an
  alternative: it is the most promising lever on §6.

The references in this section are to the literature as known when this was
written; their specifics are to be checked before any is relied on.
