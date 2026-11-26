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
- **A full range is a window.** Once the front reaches `B`, an entering
  section still enters in front, and the placed run slides up by its length
  to make room: what passes the range's top — the sections that entered
  earliest — leaves. The run keeps its order and every placed section keeps
  its position relative to the others; only their distance to the question
  shortens by the entry's length. This is the one move a relevance range
  makes under a reply, and it is the same move a sliding window makes: the
  newest in front, the oldest out of the far end. Holes inside the run slide
  with it and are closed between requests, when the range is compacted.

A section's position therefore records when it came into scope, not its rank.
That is the price of never moving anything until the range is full.

### 3.3 Per layer

**The scope is shared across layers, decided by the sampled layers.** The
shadow scan holds index rows for unplaced sections only at the sampled layers
(`hits.layers`, model layers 19, 31 and 43 on Flash-Next;
`hit_driven_projection.md` §6.1), so those are the only layers with a peak
for a section not yet placed. A section's peak is the **largest of its peaks
across the sampled layers**, and one scope holds for every layer. Per-layer
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

### 3.4 Order: peak index attention, decided per decode step

The one signal that orders a relevance-ordered range is a section's **peak
index attention**: at a decode step, in a layer, the highest score the QSA
indexer gives any block of the section. Without a QSA index it is the
section's provenance score (§3.8). Nothing else orders the range — not the
pool's momentum, not provenance strength, not a hit count.

- **Peak, not count.** A count of selected blocks grows with a section's
  length, so a long section out-votes a short one that answers better. The
  peak is one number per section whatever its length, so sections of every
  size compete on what they hold.
- **Per decode step.** At each step, after the sampled layers' selection, a
  section that is resident (§3.5) and out of scope **enters** when its peak
  crosses the entry bar: it is placed in front of the sections already in
  scope (§3.2). A section in scope **leaves** when its peak has stayed below
  the eviction bar for **64 consecutive steps** — one reprojection cadence —
  and leaves a hole.
- **The bars are on the normalized band.** A peak is normalized per query to
  the 0–1000 band the tools' belief selection uses: 0 at the query's cut (the
  lowest score its real selection kept, which the selection kernel already
  records per row, `hit_driven_projection.md` §6.3), 1000 at the best score
  it kept, a peak below the cut negative and one above the best clamped to
  1000. The band is scale-free across layers, depths and queries, and "would
  have displaced a block the model read" is any peak above 0. The bars are a
  pair on it, entry above eviction — hysteresis, so a section near
  the line does not flap. They are declared per section in `projection.yaml`
  beside its budgets, as the tools declare `min_score` / `evict_score`
  (800 / 600), and measured on the needle and multi-file cases before they
  are fixed.
- **Entering never moves anything.** A step's entries take positions below the
  lowest placed section; every section already placed keeps its positions,
  so every relative position a decoded token was formed against holds. The
  next token sees one more section in front, at positions no earlier token
  attended. That is why the order is append-in-front: per-step decisions with
  no re-layout.
- **The order is the order of entry.** At the opening, the sections the
  question's prefill selects enter in descending peak, strongest first, so
  the strongest is at the top, nearest the question. After it, a section's
  position records the step it entered.
- **A step's decision applies from the next step.** The scores of step *t*
  decide the layout step *t* + 1 reads, so a step never reads a layout its own
  scores changed. With speculative decoding the step is the verified step: a
  draft and its verify read the layout fixed when the draft began, and the
  verify's accepted tokens' scores decide the next one (§4.6).
- **The peaks of sections not in scope come from the shadow scan**, which
  scores resident, unplaced sections' index rows against the step's queries
  (`hit_driven_projection.md` §6.3), now every step for the resident set
  rather than once per boundary for the pool. The pool beyond residency is
  still scanned at boundaries; a section there enters residency first (§3.5).

Every per-step decision runs on the device (§4.4): the host is not on the
per-token path.

### 3.5 Three budgets per section

Each placed section carries three budgets, in tokens, nested from the widest
to the narrowest. They are the stages the knowledge base already runs
(`docs/hit_driven_projection.md` §7.2, `docs/provenance_system_collection.md`
§4); what is new is that each is declared per section and the narrowest one
is a position range.

| Budget | What it bounds | Who reduces into it | Today |
|---|---|---|---|
| **Index scan** (`scan`) | The sections whose sampled-layer index rows are on the device and scored by the shadow scan. No K/V moves for them. | Provenance, by section (§3.5c): the gallery scan scores each section of a file as its own case. | The pool: `PoolSize::for_card` (2M / 500K / 256K tokens, 72 / 24 / 16 GB), shared across the pool's kinds, choosing whole files. |
| **Residency** (`resident`) | The sections whose K/V is hot — on the device, in the KV-regions tenancy — whether placed or not, so one can enter the range at the next step without a PCIe or redo-log wait. | Provenance and the index together: the section's provenance score and its peak index attention from the shadow scan, fused (§3.5c). | Not a budget. A pooled member's K/V stays wherever it is (hot, warm, cold) until the fill places it (§7.2 step 3), and a cold one is placed a boundary late. |
| **Projection** (`project`) | The tokens placed in the section's range: `R`. The positions the sections take, and the K/V and index rows the selection attends. | The index alone: peak index attention per step (§3.4). | The fill's `readings` cap, a share of the room (22.5%, ≈ 52K at a 233K room), by `base·(1+gain·h)`; tools by count (`budget: {min: 1, max: 5}`). |

`scan ≥ resident ≥ project`, and each stage reduces into the next:
provenance proposes the scan set from the unbounded corpus, the index
reduces it to what is kept hot, and the index places the projection from the
hot set. A member leaving the projection stays resident while it still ranks,
so it can come back in front (§3.2) at the next step at device speed; one
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

**No kind caps in the range.** Inside the knowledge base's range, peak index
attention alone decides what enters (§3.4): readings, folders, guides and
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
  enter the range by peak index attention (§3.4), not by line order, so two
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
  competition — and enters at the opening in peak order with the rest (§3.4);
- the seed's files — each repository's root listing, README, ARCHITECTURE,
  AGENTS, CLAUDE (`branch_ingest/prime.rs`) — are the knowledge base's
  mandatory sections, cut into sections as every file is (§3.5c); the seed
  dial includes or omits the mandatory set;
- the range's budget is checked at load to hold the mandatory sections with
  room for the rest, and refuses to build otherwise.

A section's address is the prompt's hash with the range (§3.1b), the hash of
its file's earlier sections, and its own text. A change to a file re-seals
the sections from the first changed one to the file's end; the sections
before it, and every other file, are untouched.

### 3.5c Sections of a file, and reducing by section

**The cut is `file_read`'s.** A large file is broken where `file_read` already
pages it: runs of whole parts, at most `PAGE_LINES` lines each
(`zend-tools/src/tools/file/render.rs` `windows`, the parts from
`doc_units`). Each run is one section, and its text is the body a live
`file_read` of that range returns, byte for byte (`render.rs` §1: the ingest
already drives the same tool call). So a placed section carries what an edit
needs, exactly as a read does:

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
heading `## The content from {path}:` and the section's text. The outer fence
of Qwen-Agent's snippet goes: the parts are fenced blocks of their own, and a
fence inside a fence is not well formed. Seams are still checked against the
tokenizer as `files::cut_lines` checks them, now at the section's
boundaries.

**Provenance and the index both reduce by section.** Today provenance chooses
whole files and the index places a leading run of a file too large for its
cap (`provenance_system_collection.md` D22). Both now work on sections:

- **provenance scores each section as its own case** (`cases: section` in
  place of `cases: file`): the gallery already holds a member's signatures
  under its `ChunkKey` (§4 of that doc), so a section's score is its own
  signatures' agreement. A large file no longer gathers every one of its
  chunks' votes into one score: each section competes on what it holds, and a
  large file stops swamping the scan;
- **the scan set and residency are chosen by section**, by provenance score,
  and residency fuses in the section's peak index attention from the shadow
  scan (§3.5);
- **the projection is chosen by section** on peak index attention alone
  (§3.4).

The calibration's votes and the working set's momentum move to the section
key accordingly.

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
  **about 5 questions per section**: half of today's density. Measured on
  candle, battle-cities and yenbot after the ingest's ignores (8,464 files,
  17,024 sections, 2.01 per file), today's rules ask 163,825–194,481
  questions — about 10 per section, for one-section and many-section files
  alike — so half is 4.8–5.7 per section, by tokens per byte (4.0–3.0).
- **The draft sees the section alone.** The draft that writes a section's
  questions has that section, and nothing else of its file, in the knowledge
  base's range: its questions are about what the section holds. A draft
  never rereads a large file per section.
- **File-level prompts are asked once per file.** A profile's prompts about
  the file as a whole (`purpose`; `architecture` for documents) are asked in
  one draft per file, from the file's first section, and each such question
  votes for every section of the file, its weight split evenly across them.
  The per-section count of §3.5d's retune excludes them.
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
  which reseal everything after them, run here too.
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
| Knowledge base | Relevance | 96K (98,304), the mandatory sections inside it | 2× the range: 192K, on every card | the pool, `PoolSize::for_card` |
| Dialogue window | Birth | 96K (98,304) of past turns at a request's start | — | — |
| Request reserve | Birth | the rest: `L0` less the ranges, the window and the fixed prompt — ≈ 43K on Flash-Next with a ~6K fixed prompt | — | — |

- **The reserve is the request's.** There is no separate `reserve_tokens`
  margin: what the ranges, the window and the fixed prompt leave is the room a
  request grows into on rung 1 (§3.7), about seven 200-line reads. Past it a
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
- **The mandatory set must fit.** The seed's files were about 37K tokens; as
  mandatory sections they take that much of the knowledge base's 96K, and
  the load check (§3.5b) refuses a mandatory set that leaves less than half
  the range to the competition.

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
layout changes per verified step (§3.4): a draft and its verify read the
layout fixed when the draft began, and only the accepted tokens' scores
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
5. **The range running out forces a relayout**, which moves everything once.
   Sized to the budget, it should be rare; it is counted and logged.

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

**One build.** Everything in §3 and §4 lands together, the per-step decision
on the device included: explicit positions and the position frontier; every
range and the gap in the birth prefix; members sealed at the top of their
range under the new content address; sections in `file_read`'s format with
the `...` marker, each file's sections a chain; the mandatory set in place of
the seed chain; provenance, calibration and the projection by section; peak
index attention per step with append-in-front and the full range as a
window; the residency budget; the birth-ordered journal and dialogue. Scope
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
- **Stability and throughput.** The placement record shows nothing placed
  moves within a request until a range is full, and no dialogue turn or
  journal entry leaves within one; no readback on the per-token path
  (`AGENTS.md` invariants 3–4); decode throughput within noise of today's.

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
