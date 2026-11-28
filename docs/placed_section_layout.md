# Placed-section layout: a reserved position range, ordered by relevance, per layer

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

### 3.1c Every item of the system prompt takes a fixed slot

A range fixes what follows it only if everything before it is fixed too, and
most of the system prompt is not: a dial picks one of a tree node's options, a
gated section emits or does not (`depends_on`, `depends_on_absent`,
`depends_on_configured`, `depends_on_unconfigured`), and the catalog summary
emits with the notes the turn adds only when the selection is partial. Each of
those moved `B` for every range after it, by up to a thousand positions
between the tools dial's settings alone — and the ingest sealed members
against yet another prefix, every bare section and each tree's default branch,
which no projection ever produces.

So **every item takes the positions of its largest form, always**: what it
emits and a hole of the rest. The slots are

| Item | Slot |
|---|---|
| a section, gated or not | its tokens, or its `min_tokens` when that is more — a gated one that does not emit is a hole of it |
| a section-tree node | its longest option, glue included, or its `min_tokens` when that is more |
| a section tree | the sum of its nodes' slots |
| a collection with a range | the range (§3.1) |
| a collection's catalog summary | its declared `summary_range`, the summary and its notes at the front |

Every dynamic section with text that can change declares a `min_tokens`, its
margin: room to edit its text without moving anything after it. A dialect
marker (`<tools>`, `</tools>`, `no_think`) is fixed text and needs none: its
slot is its own tokens, a hole of them when it does not emit,
so a prompt change inside the margin leaves every sealed section and member
where it was. Each tree node is padded to its own slot, inside the tree as
after it: a variant seals behind the earlier nodes each in its slot
(`tree_prefix`), so changing one option moves neither the nodes below it nor
anything after the tree. zend sizes each at about twice its longest form today
— persona 576 (273), reasoning stance 128 (61), effort 160 (~75), length 96
(~35), role 4096 (the coder's ~2400), `driven` 384 (~150), grounding 768
(426), the no-tools grounding 96 (~40), the journal intro 576 (~300), the tool
overview 768 (444). A section whose text
outgrows its minimum is warned of at a conversation's creation: everything
after it has moved, and what was sealed behind its old slot is stale.

The summary's slot stands where the summary is read: just before the first
template gated on its collection (`<tools>`), or before the collection when
none is. A collection with no range takes what its members take, and moves
what follows it, so every collection either schema fills has one; the only
unranged ones are npcd's `mood` and `task`, which the daemon never fills and
which therefore take nothing. npcd's ranges and margins are measured in the
model's own tokens against the live mind by
`engine::tests_slot_sizes::the_live_minds_items_fit_their_slots`, and
`engine::tests_slots` holds what follows its collections still across
characters, templates, stances and dials.

Every walk of the system prompt lays the slots down the same way: the
projection (`project/emit.rs`), the creation walk that seals the system
sections and builds the birth prefix and its branch checkpoints
(`conversation/sequence_construction.rs`), and the prefix a collection's
members are sealed behind (`projection/owned_prefix.rs`). Each puts the
content it has always used — the projection what it selected, the creation
walk and the ingest every section and each tree's default branch — and a hole
for what that content leaves of its slot, so a range starts at the same `B`
in all of them, whatever the dials, the tools mode or the journal hold.

### 3.2 Order, growth and removal

**The range is held in relevance order, always**: the strongest section at
the top, beside the question (*Lost in the Middle*: a long context's start and
end are read best; the end is what the question is nearest), each weaker one
further from it. Nothing is exempt — the mandatory seed files are ordered like
every other section, and sink back from the question as they stop being read.

- **The first token reads the best order there is.** A new question is read
  twice (`docs/hit_driven_projection.md` §7.4). The first reading prefills it
  against the range as it stands, the mandatory files among the placed
  sections, and the index's hits on each section are **the range's first
  step**: their rates are taken whole into the files set's hit momentum
  (`Substrate::files_read_first`). The opening projection then orders the
  range by that momentum before the second prefill (`files_splice`,
  `FilesSet::order_score`, `RangeScope::reorder`). So the second prefill and
  the first decoded token already read the order the first decode step would
  give, with the section the question read most beside it, and the first
  steps move only what the decode reads differently. A section the index has
  not read this question is ordered by its scan momentum (the provenance
  scan's fold, §3.4), taken off its 0–1000 band into 0–1, so nearly every
  section the index reads ranks above it. A pinned knowledge base,
  such as a test's or a calibration's frame, is ordered the same way. The pin
  says what is placed and its sections never leave, but the index rates them
  and the scan still folds their scores (`FilesSet::observe`), so it too
  opens with the question's strongest beside it.
- **The decode reorders it every step.** Each section's **hit momentum** is
  an exponential average of its hit rate (§3.4), held on the
  files set beside the range (`FilesSet::hits`, `files/scope/momentum.rs`). A
  section's first rate this question is taken whole, from the first reading
  or else from the first step that rates it, so the order locks on at once.
  After that each step moves it a quarter of the way toward its rate
  (`MOMENTUM_KEEP` 0.75). A new question starts it afresh
  (`FilesSet::clear_withheld`), so the last question's reading does not hold
  the order for this one. After every step the range is ordered by it
  (`files_step`), so a section the decode locks onto is drawn to the top
  within a few steps and stays there while it is read, and one it stops
  reading sinks back over several. A section passes the one above it only
  when its momentum beats that one's by more than `REORDER_SHARE` (5%) of the
  range's attention — the placed sections' momenta summed — so neighbours a
  hair apart do not trade places step after step: the mildly read sections'
  shares, 3–6% each, swing by two hundredths a step, and at a 2% margin the
  range was laid out again on nearly every step. Any
  change is laid onto the slot at the next wave (§4.4); positions are applied
  at read time, so moving a sealed section costs no prefill.
- **The nucleus of the range's attention is locked to its positions.** The
  placed sections' momenta are taken as shares of their sum — the range's
  attention, normalized to 1 (`files/scope/nucleus.rs`). The nucleus at `p`
  is the fewest strongest sections that together hold `p` of it: one section
  when the decode reads one hard, several when it reads several. A section
  locks when it enters the nucleus at `P_LOCK` (0.5), at the share it holds
  then, and stays locked while within the one at `P_UNLOCK` (0.7), so one on
  the edge does not flip in and out, and while it keeps `KEEP_OF_LOCKED`
  (half) of the share it locked at: with only the wider nucleus to hold it, a
  section that locked beside the question held that seat at 5% of the
  range's attention while one at 19% sat below it (BlackWorld, 2026-11-28). While the range is read, on the mean, no more than the selection reads
  the whole slot, nothing is locked: a share of almost nothing would lock the
  largest of it. A share is a proportion of what the range is read with,
  whatever the prompt's length, the selection's budget or the model, so the
  two proportions hold on every one. A locked section keeps the positions it
  stands at, so its K/V — and the relative distances the decode is reading
  it across — do not move under the reading (`RangeScope::reorder_locked`).
  A step reorders around the locks it starts with and only then locks again,
  so a section the decode has just begun to read hard climbs first and locks
  where it arrives. Everything else moves around it: the
  unlocked sections, in relevance order, fill the gaps between the locked
  ones from the top down, the strongest that fit into each and a section too
  big for one going on to the next, and the rest lie packed below the last
  locked one. A section that leaves beside a locked one leaves a hole that
  nothing slides into, so the range fragments while sections are locked; a
  strongly read newcomer fills the gap nearest the question it fits and
  locks there itself. An entry never cuts or moves a locked section. A
  section unlocks once its momentum falls under the bar, and the next
  reorder closes the range up around what is still locked. Without the lock,
  per-step rates swing widely and a step moves a section a quarter of the way
  to its rate, so the margin rarely held: the range was reordered, and the
  slot rebuilt, on nearly every step (BlackWorld, 2026-11-28). The
  proportions are measured: a range there held one to five sections read
  hard, together a third to three fifths of its attention, a band of five to
  ten read mildly, and a long tail; a bar on the band itself hovered in the
  mild band and flipped it in and out of the lock, and the band's scale moved
  with the selection's (16–25) within one run. The opening is not locked: it
  is ordered whole by the first reading.
- **A section that comes into scope** enters at its rank by its momentum, or
  its step rate when it has none yet.
- **A section that falls out of scope** is removed; the next reorder closes
  its hole. Its pages stay in the pool, so the shadow scan
  (`docs/hit_driven_projection.md` §7.2) can bring it back at its rank.
- **A full range makes room from its front**, the end furthest from the
  question: an entry pushes out the weakest sections there, and only those
  weaker than itself — a mandatory section, or one at least as strong, stops
  it (`RangeScope::enter_over`).

The cost is the one the earlier append-only order avoided: a reorder moves
placed sections under a reply, so a decoded token's distance to a section can
change between steps. The margin keeps that to the moves relevance calls for.
The momentum's rate and margin are the formula's two knobs and are tuned by
measurement; the behaviour — strongest beside the question, reacting from the
first step — is the design.

### 3.3 Per layer

**The scope is shared across layers, decided by the sampled layers.** The
shadow scan holds index rows for unplaced sections only at the sampled layers
(`hits.layers`, model layers 19, 31 and 43 on Flash-Next;
`hit_driven_projection.md` §6.1), so those are the only layers that count
hits for a section not yet placed. A section's hit rate (§3.4) is taken
**over the sampled layers together**, and one scope holds for every layer. Per-layer
scope, below, is built only if the shared scope's measurements call
for it, at the cost of index rows and a shadow scan at every full-attention
layer.

K/V is held per layer (one backing per layer) and so is the QSA index (one
`IndexCache` per full-attention layer, scored at every step). Under per-layer
scope each full-attention layer holds **its own** set of placed sections at
**its own** positions within the range:

- a layer's scope is the sections its own selection has been reading;
- its placement order is the order they came into its scope;
- a section in one layer's scope need not be in another's.

The query of a given token has one position across layers, above the range in
every one, so every placed section is visible to it in every layer that holds
it.

### 3.4 Order: the index's hit rate, decided per decode step

The one signal that orders a relevance-ordered range under a decode is a
section's **hit rate**: the share of the section's blocks the QSA index
selected — or, for a section not placed, would have selected above the
query's cut (the shadow scan) — per query, over the sampled layers, averaged
step to step into its momentum (§3.2). Without a QSA index it is the section's
provenance score (§3.8). At a request's start, the question's first reading
is the first step. Its hits rate the sections that reading placed, and the
shadow scan rates the resident ones, so the opening is ordered by the same
signal every later step uses. Only a section neither rated falls back to the
files set's scan momentum.

- **A rate, not a raw count, and no kernel change.** The device already
  counts hits per declared round (`qsa_hits.cu`, `qsa_shadow.cu`: one
  `atomicAdd` per selected block, and a shadow hit only above the query's
  cut). A section is one round. The host divides its hits by its blocks and
  the queries counted, so a long section does not out-vote a short one, and
  sections are bounded at `PAGE_LINES` anyway (§3.5c). A peak score per
  section would need a running maximum in both kernels; the rate needs
  none.
- **Per decode step.** At each step a section that is resident (§3.5) and out
  of scope **enters** when its rate crosses the entry bar, at its rank
  (§3.2), and the range is ordered by momentum. A section in scope
  **leaves** when its rate has stayed below the eviction bar for **64
  consecutive steps** — one reprojection cadence — and leaves a hole.
- **The bars are shares of the range's attention, so they hold
  everywhere.** A section's rate is the share of its blocks the queries
  selected, on the 0–1000 band (`section_rates`); its blocks divide out its
  length. That band moves with the prompt's length, the selection's budget
  and the model — on BlackWorld the selection itself read 16–25 on it within
  one run — so no bar is set on it. Every decision is taken on shares of the
  range's: what locks (above), the reorder margin, and the step's bars
  (`Bars::knowledge_base`, `FilesSet::step_bars`). A resident section enters
  at a rate that would hold the share of the weakest locked section — it
  would be among what the range is read by — or an even share when none is
  locked; its shadow rate is out of the shadow scan's queries, so it says how
  strongly it would be read if placed. A placed section counts a step toward
  leaving under a quarter of an even share, `EVICT_OF_EVEN`, and leaves after
  64 such steps.
- **The step's counts stay beside the slot.** Which steps a placed section
  has spent under the bar is the scheduler's (`files/scope/watch.rs`, held
  on the slot's projection state): a step that changes no layout publishes
  no substrate version. A section that leaves is evicted on the files set
  (`FilesSet::evict`), stays out until the next question even where its
  momentum would choose it. A resident section (§3.5) out of scope whose
  shadow rate reaches the entry bar enters at its rank (`FilesSet::admit`); a
  section an entry pushes out of a full range's front has left as an evicted
  one has, and stays out until the next question too, so two sections over
  the bar never trade places step after step. Any change is laid onto
  the slot at the next wave (§4.4,
  `scheduler/section_step.rs`). Placed sections are declared to the model
  as hit rounds of their own (`RoundKey::of_section`), so a step's read
  rates them; every step's read is summed into the slot's unread hits, so
  the boundary still folds them into its momentum.
- **The order is relevance.** At the opening the section the question's
  first reading read most is at the top, nearest the question. After that
  every step orders the range by the same hit momentum (§3.2), so a section's
  position records its rank, not when it entered.
- **A step's decision applies from the next step.** The hits of step *t*
  decide the layout step *t* + 1 reads, so a step never reads a layout its own
  hits changed. With speculative decoding the step is the verified step: a
  draft and its verify read the layout fixed when the draft began, and the
  verify's accepted tokens' hits decide the next one (§4.6).
- **The hits of sections not in scope come from the shadow scan**, which
  scores resident, unplaced sections' index rows against the step's queries
  (`hit_driven_projection.md` §6.3), now every step for the resident set
  rather than once per boundary for the pool. The pool beyond residency is
  still scanned at boundaries; a section there enters residency first (§3.5).

The decision is the host's, every step, and waits on nothing (§4.4).

### 3.5 Three budgets per section

Each placed section carries three budgets, in tokens, nested from the widest
to the narrowest. They are the stages the knowledge base already runs
(`docs/hit_driven_projection.md` §7.2, `docs/provenance_system_collection.md`
§4); what is new is that each is declared per section and the narrowest one
is a position range.

| Budget | What it bounds | Who reduces into it | Today |
|---|---|---|---|
| **Index scan** (`scan`) | The sections whose sampled-layer index rows are on the device and scored by the shadow scan. No K/V moves for them. | Provenance, by section (§3.5c): the gallery scan scores each section of a file as its own case. | The pool: `PoolSize::for_card` (2M / 500K / 256K tokens, 72 / 24 / 16 GB), shared across the pool's kinds, choosing whole files. |
| **Residency** (`resident`) | The sections whose K/V is hot — on the device, in the KV-regions tenancy — whether placed or not, so one can enter the range at the next step without a PCIe or redo-log wait. | Provenance and the index together: the section's provenance score and its hit rate from the shadow scan, fused (§3.5c). | Not a budget. A pooled member's K/V stays wherever it is (hot, warm, cold) until the fill places it (§7.2 step 3), and a cold one is placed a boundary late. |
| **Projection** (`project`) | The tokens placed in the section's range: `R`. The positions the sections take, and the K/V and index rows the selection attends. | The index alone: its hit rate per step (§3.4). | The fill's `readings` cap, a share of the room (22.5%, ≈ 52K at a 233K room), by `base·(1+gain·h)`; tools by count (`budget: {min: 1, max: 5}`). |

`scan ≥ resident ≥ project`, and each stage reduces into the next:
provenance proposes the scan set from the unbounded corpus, the index
reduces it to what is kept hot, and the index places the projection from the
hot set. A member leaving the projection stays resident while it still ranks,
so it can come back in front (§3.2) at the next step at device speed; one
leaving the resident set keeps its index rows while it stays scanned, so the
shadow scan can still raise it.

**The knowledge base's residency as built** (`files/scope/resident.rs`,
`scheduler/kb_residency.rs`). At each boundary, beside the pool's refresh,
the slot chooses its resident sections: twice the range's positions, less
what is placed, from the files set's sections that may enter during the
request (none an eviction sent out, none of a withheld file), strongest by
momentum, those the shadow scan read since the last boundary first. Each
resident section's index pages are held in the slot's pool and declared as
pooled rounds under the section's round (one per page), so the shadow scan
rates it at every step; its K/V is lifted hot when the decode floor holds
after the lift, and it is protected from relief eviction while it stays
resident. On a model the shadow scan rates sections for, a section whose
page this process does not hold could never enter at a step, so it is not
resident; it enters only by a boundary's fill.

**Per section.** The tools section and the knowledge base each declare their
own triple in `projection.yaml`, beside the section:

- **Knowledge base:** `scan` is the files' part of the pool, `resident` a
  card-sized budget between it and `project`, `project` the range.
- **Tools:** the catalog is small enough that `scan` is the whole catalog.
  `project` moves from a count of signatures to tokens, so the tools range is
  a fixed size and the belief selection fills it by score until the next
  signature does not fit. Its minimum (one signature) stays.
- **A range with a member bound** (`range: { tokens: N, most: M }`) holds at
  most `M` members, however much room is left: the fill stops at the first
  that does not fit or at the `M`th, and the belief budget is `M`. A
  collection whose selection is one locked choice — npcd's talk template, one
  per turn — is `most: 1`, so a range sized for its longest template never
  admits a second short one beside it. Without `most` a ranged collection
  ignores its count, as the tools do.

The projection budgets are fixed for the dialect: they are positions, so a
card does not change them, and the same turn lays out identically on all three
machines (the prescribed model is the same everywhere). `scan` and
`resident` are device memory and are sized per card, as the pool is today.
The room the dialogue's fill has is the slot less the system prompt **with its
ranges at full size**, so the dialogue's share never depends on what the
sections hold.

**No kind caps in the range.** Inside the knowledge base's range, the index's hit
rate alone decides what enters (§3.4): readings, folders, guides and
journal notes compete on one band, and the working set's kind shares
(`working_set.shares`) no longer cap them there. The kinds keep their
per-kind minimum counts in the scan pool (`PoolSize::per_kind`), so guides
and journal notes are always scanned however many readings rank, and a
section of any kind can enter when the question reaches it.

### 3.5b Every section stands alone; some are mandatory

**No file is sealed behind another; a file's sections are sealed behind its
own earlier sections.** A knowledge-base section is sealed behind the prompt
up to its range, across the empty range (§3.1b), and behind the sections
before it in its own file — section *k* behind sections 1…*k*−1, its
recurrent state continued from theirs, as `provenance_system_collection.md`
§5 seals chunks today — and behind nothing else. A file's chain ends at the
top of the range: section *k*'s last token sits at `B + R`, its file's
earlier sections immediately below it.

- **Within a file, cross-reasoning is kept.** A section of a large file is
  read as the continuation of the file, so what a function means in light of
  the types above it is in its K/V. Losing that is judged worse than what
  the next point costs.
- **The order of placement can break a file's relative positions.** Sections
  enter the range by hit rate (§3.4), not by line order, so two
  sections of one file can be placed in an order or at a distance other than
  the one they were sealed at. This is accepted and measured: §7's tests
  includes a question that needs two sections of one file placed out of line
  order.
- **Between files there is no cross-reasoning at ingest.** It is built where
  the conversation reads them together — in the turns, and above all in the
  thinking blocks, which are decoded over the placed sections and run through
  every layer, the recurrent ones included.

This removes the **seed chain** (`provenance_system_collection.md` §6): the
seed's files are no longer prefilled as a chain with the recurrence run over
them and snapshotted. They are sections like any other, sealed alone, and
the recurrent state a conversation starts from is the prompt checkpoint's,
which has read no knowledge-base content (§6).

What the seed was for is kept as **mandatory sections**, the concept the tools
collection already has (`SectionCollection::mandatory`,
`projection/project/collection.rs`: `file_read` is emitted on every
projection, outside the selection's budget):

- a mandatory section is resident and in scope from the opening, in every
  layer, and never leaves;
- it holds positions in its range like any other — inside `R`, outside the
  competition — and enters at the opening in rate order with the rest (§3.4);
- the seed's files — each repository's root listing, README, ARCHITECTURE,
  AGENTS, CLAUDE (`zend/src/files/seed.rs`) — are the knowledge base's
  mandatory files, ingested and cut into sections as every file is (§3.5c),
  and named at startup by workspace path (`files/mandatory.rs`, the
  substrate's `mandatory_files`); a path the branches hold in several
  versions places the version most branches hold. Every free knowledge base
  the seed dial leaves on places them whole, outside every kind's cap; a
  pinned one (a calibration's frames) holds its pin alone. The `seed` section
  tree and its branch checkpoints go;
- the range's budget is checked at load to hold the mandatory files with
  room for the rest — at most half of it — and refuses to start otherwise.

A section's address is the prompt's hash with the range (§3.1b), the hash of
its file's earlier sections, and its own text. A change to a file re-seals
the sections from the first changed one to the file's end; the sections
before it, and every other file, are untouched.

### 3.5c Sections of a file, and reducing by section

**The cut is `file_read`'s.** A large file is broken where `file_read` already
pages it: runs of whole parts, at most `PAGE_LINES` lines each
(`zend-tools/src/tools/file/render.rs` `windows`, the parts from
`doc_units`). Each run is one section, and its text is the parts and closing
line a live `file_read` of that range returns, byte for byte (`render.rs`
`range_read`, which `lines_read` opens with its one-line note ahead of). So a
placed section carries what an edit needs, exactly as a read does:

- each part a fenced block naming the workspace path, the part's anchor (the
  part hash `file_edit` addresses, `part=r7Lm`), its lines and its scope;
- every line behind its number relative to its part (`+1`, `+2`, …), the
  numbers `file_edit` takes;
- the closing line naming the file, the lines this section holds of how many,
  and the range to read next — or that it is the end of the file.

The model therefore edits from a placed section with the same references as
from a read, and a section and a later read of the same range agree.

**A cut is marked.** Every section but a file's last ends with `...\n`, after
its closing line, so a section placed without the one after it reads as
continuing, never as the end of the file. The last section ends at the file's
end. The marker is sealed with the section; the assembler adds no gap piece
of its own (`files/render.rs` `GAP` goes), since each section already says
where it sits.

**The knowledge base's notation changes with it.** Each section is the
heading `## The content from {path}:`, a blank line, the section's text, and
a blank line it ends on (`files/render.rs` `section_text`); a folder is one
section, its listing in a fence. The outer fence of Qwen-Agent's snippet
goes: the parts are fenced blocks of their own, and a fence inside a fence is
not well formed. With it go the file's heading member and the shared gap and
closing pieces: a section is placed alone, so it carries its own heading.
Every seam is a section's closing `\n\n` meeting the next heading's `##`,
which the vocabulary never merges; `tests/knowledge_pieces_real_vocab.rs`
checks it on every cached Qwen vocabulary, sections in and out of line order.

**Provenance and the index both reduce by section.** Today provenance chooses
whole files and the index places a leading run of a file too large for its
cap (`provenance_system_collection.md` D22). Both now work on sections:

- **provenance scores each section as its own candidate**
  (`belief_members.rs`): each section is a candidate named by its
  `SectionKey` — the digest of its file's content key and its lines — its
  member stream one window and one case, so the groups' `cases: file` (one
  case a candidate) now pools a section, not a file. A section's score is its
  own signatures' agreement and its own calibrations'. A large file no longer
  gathers every one of its sections' votes into one score: each section
  competes on what it holds, and a large file stops swamping the scan;
- **the scan set and residency are chosen by section**, by provenance score,
  and residency fuses in the section's hit rate from the shadow scan (§3.5);
- **the projection is chosen by section** on the index's hit rate alone
  (§3.4).

The calibration's votes and the working set's momentum move to the section
key accordingly: the files set holds each section's momentum, kind and file
(`files/set.rs`), the fill places sections strongest first, each that fits
(`files/fill.rs`), a pin names sections, and a file the turn read withholds
all its sections.

### 3.5d Calibration per section

Calibration (`zend/src/calibrate/`, `provenance_system_collection.md` §7)
writes questions a unit answers and seals them, with the model's thinking
about each, as signatures that vote for the unit. It moves to the section,
as provenance and the projection do (§3.5c):

- **A section's questions vote for the section.** `CALIBRATES_KEY` names the
  section's key, not the file's, so a large file's questions no longer pool
  into one file-level vote.
- **The prompts are still chosen by file type.** The profile is matched by
  path regex as today (`calibrate/profile.rs`), and each of its prompts keeps
  its own count rule (`per_tokens`, `min`, `max`), applied to the section's
  tokens instead of the file's. The rules are retuned so a profile averages
  **about 5 questions per section**: half of today's density — each
  per-section prompt's `per_tokens` doubled and its `min` and `max` halved
  (rounded up for `min`). Measured on
  candle, battle-cities and yenbot after the ingest's ignores (8,464 files,
  17,024 sections, 2.01 per file), today's rules ask 163,825–194,481
  questions — about 10 per section, for one-section and many-section files
  alike — so half is 4.8–5.7 per section, by tokens per byte (4.0–3.0).
- **The draft sees the section alone.** The draft that writes a section's
  questions has that section, and nothing else of its file, in the knowledge
  base's range: its questions are about what the section holds. A draft
  never rereads a large file per section.
- **File-level prompts are asked once per file.** A profile's prompts about
  the file as a whole (`purpose`; `architecture` for documents), marked
  `per: file` in `projection.yaml`, are asked in the file's first section's
  draft, counted by the file's tokens, and their questions vote for that
  first section. A vote counts once per token per case and the vote law
  carries no weights, so a question cannot vote for every section at a split
  weight without a kernel change; given to every section whole, a file-level
  question would vote for all of a large file's sections at once — the
  swamping this design removes. The first section holds the file's opening,
  where its purpose is stated. A guide's read of a file
  (`guidance/reads.rs`) votes for the first section the same way. The
  per-section count of the retune excludes them.
- **Asked once, behind the empty range.** With the seed chain gone there is
  no seed branch to ask behind, and the questions are no longer split
  between the empty and the target knowledge base: each is asked once,
  thinking on, behind the knowledge base's range with nothing placed — the
  frame a live query is in before it has found the section, which is what
  provenance must match to pull it in — and sealed there.
- **Half the decode.** A question costs one thinking decode of `think` (64)
  tokens and its share of the seal's prefill, so half the questions is half
  the calibration's time.
- **Corpus hygiene comes first.** At a count per section, a large generated
  file is a large share of the budget: `quantized_llama_tokenizer.json` alone
  is 1.25M lines, 6,255 sections, about 31K questions — a third of the
  corpus's. Generated and vendored text is kept out of the ingest: the
  tokenizer JSON, the tracked virtualenv (`candle-nn/tests/.venv/`), the
  generated CUDA bindings (`third_party/cudarc/src/*/sys/`) and lockfiles
  (`Cargo.lock`) join the `ignore` list of `motherd/motherd.yaml` and the
  machine configs.

### 3.5e A file is sealed behind its repository's listing

**A section sealed with nothing before it reads as the only one of its
kind.** Each file is sealed once, behind the prompt up to `files_open`, so
two files of the same shape — two `Cargo.toml`s, two `README.md`s — are each
computed as if no other existed. Their headings name their files, but the
lines below carry nothing that tells one file's from the other's: on
BlackWorld (2026-11-28) a question about candle's `serde` version was answered
with yenbot's whenever yenbot's `Cargo.toml` sat nearer the question, and with
candle's when candle's did, while a fresh prefill of the same prompt — the
same tokens, the same recurrent state, the same C4 quantization — found
candle's wherever it sat. What a fresh prefill gives the later file is the
earlier one in its attention; interpretability work on how models bind
entities finds those bindings carried by an ordering — *the first, the
second* — that a separately computed section has no way to take
(Feng & Steinhardt, ICLR 2024; Dai et al., EMNLP 2024).

**So a file's members are sealed behind its repository's root listing, then
the listing of the folder it is in** (`files/lead.rs`,
`MembersPrefix::led_by`): the listings' sections stand just ahead of the
file's first member, inside the range, after the collection's own lead
(§3.5f), the range's hole above them short by their positions, and every
section of the file is computed as a part of the repository and the folder
the listings name — its bearings — read after their trees rather than as the
first thing read. The listings' K/V is the one the knowledge base already
holds — reused as a seed chain reuses its sections, its recurrent state run
on from its own — so the lead costs no prefill. It is part of every member's
address, so a file sealed behind it is another member than one sealed behind
nothing.

- **The listings first.** The repository layer (`repo_map`) is ingested ahead
  of the files in every pass, so a file finds its listings sealed. A listing
  this process holds no version of is left out of the lead; a file at the
  repository's root is led by the root listing alone, and a listing is led by
  no listing itself.
- **The version on the file's branch.** A listing has a version per branch; a
  file is led by the first that shares a branch with it, in the file's branch
  order.
- **Placed, the lead is not.** The listing stands in the range only when the
  fill places it, wherever relevance puts it; positions are applied at read
  time, so a section sealed behind it reads the same wherever it lies.

### 3.5f Every collection declares a lead that is never placed

A tool's signature, a journal's first section and a repository's root listing
have no listing to stand behind, and each was sealed as the only one of its
kind. **So every collection of the system prompt declares a lead in
`projection.yaml`** (`lead:` beside `sections:`): records of the collection's
own kind — a signature of a tool that does not exist, a journal entry 0, a
file of a repository that is not in the workspace — sealed once by the
creation walk at the collection's place, as members are, and placed ahead of
every member each member is sealed behind (`SectionCollection::lead`).

- **Never placed.** A lead is a section of the schema but no member of its
  collection: it is never selected, never offered, never emitted. It shapes
  the K/V of what is sealed behind it and nothing else, so no tool can be
  called that is not there and no journal reads an entry nobody wrote.
- **Every seal path.** The creation walk seals a collection's members behind
  it (`insert_section_collection_with_progress`), a submitted section — a
  journal's — straight after it (`Sequence::submit_section`), and a knowledge-
  base file behind it and then its listings (`Sequence::members_prefix`). In a
  range the hole above shrinks by its positions, so the range keeps its size.
- **Part of the address.** The lead is in every member's prefix chain, so
  changing it re-seals the collection.

### 3.6 Two orders: by relevance and by birth

Every piece of provenance the projection places falls under one of two
rules. Both give it a fixed range of positions during a request; they differ
in how the range fills and when anything leaves.

| Rule | Order | Fills | Changes | Leaves |
|---|---|---|---|---|
| **By relevance** (§3.1–§3.5) | Its rank by momentum, strongest at the top | Strongest at the top, reordered every decode step | At any step, mid-request included | At a step, the next reorder closing its hole |
| **By birth** (§3.7) | When it was written | Bottom up, each new piece after the newest | Appends only, during a request | Only between requests, the oldest first |

What every piece of provenance is, today and under this design:

| Provenance | Where it sits | Today | Rule |
|---|---|---|---|
| Tools | System prompt, `tools` collection | Belief selection, 1–5 signatures, re-chosen at boundaries | Relevance; projection in tokens (§3.5) |
| Knowledge base: readings, folders, guides, journal notes | System prompt, `files` collection (`provenance_system_collection.md` §4, D19) | Pool → fill under the kind shares | Relevance |
| The seed | System prompt, before the `files` range | A chain with the recurrence run over it, sealed once, before the files (`provenance_system_collection.md` §6, §6.2b) | Relevance: mandatory sections of the knowledge base (§3.5b) |
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
  region fits its window again — for the dialogue, whole requests, the user's
  turn with its tool rounds and its reply, so no exchange is left half-shown
  and each is still reached through its journal entry — and the region is re-seated at the bottom of
  its range. Keys and index rows are stored un-rotated, so re-seating rewrites
  nothing (`hit_driven_projection.md` §4). The next request prefills against a
  stable layout, on rung 1 again. The journal's folds and forgotten entries,
  which reseal everything after them, run here too: a request's tool rounds
  carry the journal `Changes::Growing` from its growing end — the current
  request's steps, or its close (`conversation/carried.rs`,
  `session/journaling.rs` `growing_end`) — so an entry joins the chain's end
  and the steps and open items change mid-request, while a reconcile that
  would take a held section away, or seal one again ahead of that end (an
  edited entry), changes nothing until the next request's first round.
- **A turn is read where it was written.** Within a request every turn keeps
  the position it was computed at, so the dialogue's own K/V is read exactly
  as computed. A prune between requests shifts the survivors down together,
  which keeps their positions relative to one another.

The regions and their ranges:

- **The journal** has its own range in the system prompt, between the
  grounding and the tools range, so its growth never moves what follows it:
  8K on a model with a native context of 128K or more, 2K below (§3.7b), of
  which `pin_tokens` (1,024) is pinned. It fills from the bottom, as its chain is
  sealed: each entry is sealed as the continuation of the ones before it at
  the next position, so the range holds the chain exactly as it was computed.
  A full journal range is a window: an entry a request writes past it drops
  the oldest entry that is not pinned (`pin_tokens` pins the newest), and the
  entries above it slide down by its length, so the newest is always placed.
  The dropped entry's turns are reached with `journal_read`, as an entry
  folded away is.
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

### 3.7b The sizes

The first build's budgets, on Flash-Next's trained window (`L0` =
262,144 positions):

| Region | Order | Projection (positions) | Residency | Scan |
|---|---|---:|---:|---|
| Journal | Birth | 8K (8,192) on a model whose native context is 128K or more — Flash-Next, Qwen3.6-35B; 2K (2,048) on a smaller one | — (the conversation's own sections, always hot) | — |
| Tools | Relevance | 8K (8,192) | the catalog | the catalog |
| Knowledge base | Relevance | 112K (114,688), the mandatory sections inside it | 2× the range: 224K, on every card | the pool, `PoolSize::for_card` |
| Dialogue window | Birth | 96K (98,304) of past turns at a request's start | — | — |
| Request reserve | Birth | the rest: `L0` less the ranges, the window and the fixed prompt — ≈ 27K on Flash-Next with a ~6K fixed prompt | — | — |

- **The reserve is the request's.** There is no separate `reserve_tokens`
  margin: what the ranges, the window and the fixed prompt leave is the room a
  request grows into on rung 1 (§3.7), about four 200-line reads. Past it a
  request runs on the next rung.
- **The journal's range follows the native context.** 8K holds a long
  conversation's folded history and a request's notes on a model that reads
  128K or more natively; a smaller native context cannot spare it and gets
  2K. The range is chosen from the engine's reported trained window at load,
  never configured per machine. `pin_tokens` (1,024) still bounds what of the
  journal every selecting query attends whole; the rest of the range is
  selected by score.
- **Residency is the same on every card.** It is twice the knowledge base's
  range; on the 16 and 24 GB cards the expert zone gives up the difference.
- **Tools: the mandatory ones in the range, the rest by score.** `file_read`
  and the collection's other mandatory members hold positions in the tools
  range outside the competition, as the knowledge base's mandatory sections
  do; the belief selection fills the rest by score until the next signature
  does not fit. Its `min: 1` stays; its `max` is the range.
- **The mandatory set must fit.** The seed's files were about 37K tokens as
  plain text; in `file_read`'s format (§3.5c) — every part fenced with its
  anchor, every line behind its relative number — they measured 51,691 on
  2026-11-28, past half of the 96K the range was first given, and the load
  check (§3.5b), which refuses a mandatory set that leaves less than half the
  range to the competition, refused to start. The range is 112K so the seed
  fits with room to grow; the reserve gave up the 16K.

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
- **Every kernel rotates by the slices' positions.** Decode rotates its query
  at the write slice's `rope + len`; prefill rotates each sealed key it reads
  at its slice's `rope_base − offset + cell`, and each query and fresh key at
  its offset from the header's `fresh_pos` — the slot's tokens plus its holes,
  written by the host
  (`SlotHeaderHost::fresh_pos`) — so a token prefilled past a hole and the
  same token decoded see the context at the same distances. Neither costs
  the kernel a lookup: the slice's base is read by the bind its decode
  already does (`WarpPalette::pos_base`). Holes have no cells, so no kernel's
  softmax ever sees one. The float fallback rotates at the positions the
  slot's runs give (`ChunkedKvBacking::position_runs`, `token_position`).
- **Prefill addresses by token, decode by position.** The prefill kernel's
  causal bound, key count and position map are in tokens, so the QSA page
  table a prefill row reads names each page by its first token
  (`qwen4exp/indexer/token_pages.rs`); a decode row's table keeps positions,
  which its kernel resolves through the slices' RoPE bases. The index's
  coverage is checked against the frontier, never the token count.

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

### 4.4 The per-step decision: the host's, from what the device publishes

No kernel is added or changed. Every piece the per-step decision reads and
writes exists:

- **The hits are published every wave.** `qwen4exp/hits.rs` counts each
  sampled layer's selected blocks per declared round and, after the wave's
  last sampled layer, publishes the counters into a mapped block. The host's
  `SeqHits::take` reads the newest finished publish — no copy, no wait — and
  takes the difference from its last read. Read every step, it is the step's
  hits. The host already waits for each wave's token ids before building the
  next (the sanctioned embedding readback, `AGENTS.md` invariant 3), and the
  publish precedes sampling, so step *t*'s hits are always there when step
  *t* + 1 is built: a one-step lag and no added wait.
- **The shadow scan runs on the step it is asked.** `take` asks the next wave
  to scan; the scan kernel takes a mask of the kept-query slots written since
  the last scan, so asking every step scores the newest queries alone. Its
  rows are the resident set's, declared by `SeqHits::redeclare`.
- **Changing the layout is host table work.** A chunk's position is written
  by the host into its descriptor entry (`gpu_chunks/guard.rs`
  `rebuild_decode` / `extend_decode`); a front insertion re-serialises the
  layer's table rather than appending to it. A launch still reading the
  previous table holds its own pinned copy (`ChunkPin`), so a change between
  waves cannot reach the one in flight. Index pages are placed at their
  positions through the index cache, holes and all (`skip_to`). The decode
  path captures no CUDA graph, so nothing is invalidated by a new table.
- **Placement is in place.** A change is a splice of the slot's tables, not a
  reprojection's choice of the slot. As built (`scheduler/section_relayout.rs`)
  a step's change is laid out at the next wave from the segments the slot was
  last assembled from, the range rewritten from the files set's scope
  (`files_relayout`): no probe, no scan, no projection. The rebuild is the
  reprojection's own (`reproject_rebuild`): the turn's tail captured, the
  pieces before the range kept, the range spliced in where it changed, the
  rest re-walked from the pieces' own K/V; its projection event names the
  files the range now places. A hole is a piece of the layout as a section
  is — named by its size, kept, put in, taken out or moved where it stands
  (`piece_identity`) — so the gap a locked section leaves, and the section
  that later fills it, are edits like any other. A hole holds no chunk of
  its own: it is the lead of the first chunk after it, so once the edits
  stand the slot's leads are laid from the placed pieces
  (`projection_assembler::apply::lay_leads`), never carried through a
  splice. A re-layout declined below the decode floor
  stays owed and is asked again at the next drain. A slot with no assembly
  on record is reprojected whole. A resident section holds its K/V and its index rows on
  the device (§3.5), so an entry moves no bytes across PCIe.

The cost is host time on a step whose layout changes: re-serialising the
chunk tables of each full-attention layer, placing the section's index pages
and redeclaring the hit map. Hysteresis and the 64-step eviction window keep
such steps rare; their time is measured against today's per-step wave build
by §7's throughput test.

### 4.8 What is reused

| Design piece | Existing code it builds on | The change |
|---|---|---|
| Explicit key positions, holes, ranges, slides | Per-chunk `rope_base` in the descriptor table (`gpu_chunks/guard.rs`) | A chunk carries its position instead of the running sum of the usage before it |
| Query positions apart from the token count | Per-row absolute positions passed into the wave (`qwen4exp/wave/layers.rs` `dec_off` / `pre_off`) | Fed from the position frontier; every caller that reads the token count as a position is found and told which it means |
| Index holes and placement | Recorded page positions, `skip_to`, `push_positional_gap` (`qwen4exp/wave/positional.rs`) | None in the engine; the assembler places at range positions |
| Sealing at a range's top | `seal_members`, `prepare_section_ingest` | The seal's prefill starts at the frontier `B + R − T` |
| Sections in `file_read`'s format | `zend-tools/src/tools/file/render.rs` (`windows` and the page body), which the ingest already drives | Replaces `files::cut_lines` and the plain fence of `files/render.rs`; adds the `...` marker |
| Provenance by section | The gallery's signatures per member stream, the scan's candidates | Each section a candidate under its `SectionKey`; `cases: file` pools one candidate, now a section |
| Calibration by section | `calibrate/count.rs`, profiles, prompts, the seal | Counts from the section's tokens; votes by a section scan id; one frame; the draft sees the section |
| Mandatory sections | The tools collection's mandatory members (`SectionCollection::mandatory`) as the model | A mandatory set in the files splice; the seed chain removed |
| Residency | `elevate_to_hot`, the tier migration, the region pool | Promotes ranked sections, every layer's index rows with them, without placing them |
| Hit rates per step | `qwen4exp/hits.rs`, `qsa_hits.cu`, `qsa_shadow.cu`, the mapped hit summary | A section is a round; `take` every step; the rate computed on the host |
| Birth-ordered dialogue and journal | The fill, reprojection, `held_turn`; the YaRN rungs chosen by reach | No re-choice or renumbering within a request; whole requests pruned between them |

No kernel source changes.

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
layout changes per verified step (§3.4): a draft and its verify read the
layout fixed when the draft began, and only the accepted tokens' hits
decide the next one, so a draft never spans a change and a rejected token
never changed the layout.

## 5. Risks

1. **The gap distances the prompt from the question.** Long-distance
   attention is weaker; instructions and persona above the range could lose
   grip. Measured by §7's tests against today's answers.
2. **Positions no longer encode relevance.** A section brought in late sits
   furthest from the question. The opening's order is by relevance, but a long
   turn's later arrivals are not.
3. **Per-layer scope means a section may be read by some layers and not
   others.** That is what per-layer selection already does inside attention;
   holding it in the layout makes it permanent between boundaries.
4. **Re-entry moves a section.** A section evicted and brought back is placed
   at new positions — a move, but only of content no decoded token was
   attending when it left.
5. **A full range slides** (§3.2), which shortens every placed section's
   distance to the question at once. Sized to the budget, it should be rare;
   it is counted and logged.
6. **The position frontier touches every caller of the token count.** The
   slot's token count has been its position everywhere — truncation, the
   prefix kept on a rebuild, reprojection, KV-region planning. Each is found
   and told which it means; one missed reads a wrong position silently.

## 6. What this does not fix

The 52K-after-the-needle failure turned on the recurrent state: the sealed
K/V and the sealed index, under a recurrence that had run over the whole
prompt, read the file; under production's recurrence, which never runs over
placed sections, it did not. No RoPE layout reaches the DeltaNet layers.

With the seed chain gone (§3.5b) the recurrence reads no knowledge-base
content at all: the conversation starts from the prompt checkpoint, and what
the recurrent layers know of the placed sections is what the turns and the
thinking decode over them. That is the design's bet — cross-reasoning is built
in the thinking, not at ingest — and §7's needle tests measure it. If it is not
enough, the candidates are running the recurrence over placed sections at a
projection, or recomputing the few tokens whose state deviates most
(CacheBlend-style selective recompute).

## 7. Build, wipe, test

**One build, no kernel changes.** Everything in §3 and §4 lands together,
on the reuse of §4.8: explicit positions and the position frontier; every
range and the gap in the birth prefix; members sealed at the top of their
range under the new content address; sections in `file_read`'s format with
the `...` marker, each file's sections a chain; the mandatory set in place of
the seed chain; provenance, calibration and the projection by section; the
hit rate read and decided on every step by the host, the range ordered by
momentum and a full range making room from its front; the residency budget;
the birth-ordered
journal and dialogue. Scope
is shared across layers (§3.3); per-layer scope is the one part left out, and
is built only if the tests below call for it.

Every building block is unit-tested as it is written (`AGENTS.md`), against
exact values: the `rope_base` of every chunk and the page bases of every
index window for a layout with ranges, a hole, a front insertion and a full
range's slide; a section's rendered bytes against `file_read`'s for the same
range; the per-section question counts; the prune of whole requests. The
gate models pass unchanged with every range empty.

**Wipe and ingest narrow.** The substrate's members, calibrations and
signatures are all sealed under the old layout and address, so none is
carried:

1. the build green — `fast-test`, the gates — committed and pushed;
2. zend stopped on BlackWorld through keep;
3. zend's substrate **deleted**;
4. `max_depth: 1` set in BlackWorld's machine config, so the first ingest is
   the repositories' top levels and finishes fast;
5. the build deployed and zend restarted; the ingest and its calibration
   waited out to settled;
6. the tests below run. The full corpus is ingested only once they pass.

**The tests**, on the new system:

- **Sealed-fresh needle cases.** The needle placed first matches a fresh
  prefill token for token up to the recurrence's divergence; the needle with
  50K of other files after it, tools on, reads the file — the open failure
  of §1.
- **Multi-file and same-file cases.** A question about the fifth of ten
  placed files; a question needing two files; a question needing two
  sections of one file placed out of line order (§3.5b).
- **An edit from a placed section.** An edit made from a placed section by
  its part anchor and relative line numbers lands where it should, checked
  against the file.
- **Order and throughput.** The placement record shows the range in
  relevance order at every step — the section a question is about drawn to
  the top within a few steps, the others sinking back — and no dialogue turn
  or journal entry leaves within a request; no readback on the per-token path
  (`AGENTS.md` invariants 3–4); decode throughput within noise of today's.

## 8. Alternatives considered

- **Overlapping positions.** Every member at P0, so each is read exactly where
  it was sealed. Many keys at each distance is a pattern the model never saw,
  and order is lost; parallel-context work (Parallel Context Windows,
  Block-Attention, APE) finds it degrades as windows multiply without
  fine-tuning. QSA selection would keep collisions rare for focused queries
  (as InfLLM's fixed-distance memory does) but not for cross-file ones, and
  the index window refuses overlapping pages today.
- **Append-in-front: never moving anything placed.** Each new section entered
  below the lowest placed one and kept its place, so no decoded token's
  distance to a section ever changed — but a section's place recorded when it
  came into scope, not its rank, and the seed held the top whatever was read.
  The range now reorders every step by momentum, with a margin against
  flapping (§3.2), as Attention Sorting reorders between passes.
- **Selective recompute** (CacheBlend). Recomputes the tokens whose sealed K/V
  deviates most from a full prefill. Complementary rather than an
  alternative: it is the most promising lever on §6.

The references in this section are to the literature as known when this was
written; their specifics are to be checked before any is relied on.
