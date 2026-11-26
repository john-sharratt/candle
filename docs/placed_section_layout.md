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
- Turns of the dialogue. Their order carries meaning; this is for the
  knowledge base's members.

## 3. The design

### 3.1 A reserved range

The collection is given a reserved range of positions `[B, B + R)` in every
layer. Everything before it — the system prompt, the seed — keeps the
positions it has today, below `B`. Everything after it — `system_close`, the
user turn, the decode — starts at `B + R`, whatever the collection holds.

The collection fills the range **from the top down**: the first section placed
ends at `B + R`, the next ends where that one begins, and so on. Positions in
the range no section holds are a **hole**: they carry no K/V and no index
rows. A hole is legal in attention (RoPE depends only on the positions that
are present) and in the index (`PagedIndex` already accepts a hole between
pages).

`R` is the collection's budget — the fill's caps — not a constant: a range
far larger than what can ever be placed pushes the prompt before it further
from the question for nothing (§5.5).

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

`B + R` plus the turn must stay within the positions the model reads well —
its trained length and the YaRN schedule (`rope_schedule::FactoredRope`
selects a rung by position). The range pushes everything after it up by `R`
minus what the collection holds, so `R` is sized to the budget the fill can
actually place.

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
1. **The range, static, shared across layers.** The collection at the top of
   a reserved range, everything else as today. *Gate:* no loss on the needle
   cases and on persona/instruction adherence against today's layout.
2. **Explicit positions** (§4.1, §4.2). *Gate:* unit tests asserting the
   exact `rope_base` of every chunk and the exact page bases of every window
   for a layout with a range, a hole and a front insertion; the gate models
   unchanged when the range is empty.
3. **Append in front at boundaries, shared across layers** (§3.2, §3.4).
   *Gate:* the multi-file case improves or holds; nothing placed moves within
   a turn (asserted from the per-boundary placement record).
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
