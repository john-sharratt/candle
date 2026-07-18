# Immutable Summary Forest

Status: authoritative design for the per-timeline summary structure. Supersedes the
AVL-balanced binary tree + `dirty`-sweep described in `infinite_conversations.md` §7.

## Motivation

The previous structure was a self-balancing **binary AVL** tree of summary turns. AVL
keeps the tree balanced by *rotating* — and a rotation rewrites an internal node's
children. Once an internal `SummaryOfSummaries` (SoS) node's children change, the
compressed content it was sealed with (and the Q-fingerprint captured from it) no
longer match what is actually beneath it. The `dirty` bit existed solely to track
that staleness so an async sweep could regenerate the node.

That sweep was never finished — it cleared the flag without re-compressing — and the
clear was not even persisted, so every rotated SoS read back `dirty` forever. The
deeper problem is structural: **mutating internal nodes is the only reason `dirty`
exists.** Remove mutation and the entire machinery (rotations, `dirty`, the sweep,
the dirty-set, regeneration) disappears.

## Core idea: append-only, immutable, derivable

A node's parent is determined by **arrival order and position**, never by rebalancing.
Once a node is created its children are fixed → its content is fixed → its
Q-fingerprint is fixed. Nodes are **immutable**. The structure is an append-only
**forest** (a Merkle-Mountain-Range), not a single rooted tree.

Consequences:

- **No `dirty` flag.** Whether the persisted shape is correct is *derivable* from the
  leaf count, so it is never stored. Reconciliation on load recomputes the canonical
  shape and rebuilds anything that doesn't match.
- **No rotations, no balance bookkeeping, no regeneration.** `mark_dirty`,
  `clear_summary_dirty`, `dirty_summary_set`, `sweep_one_dirty`, and the AVL rotation
  family are deleted.
- **Crash-safe by construction.** A half-written summary simply fails to reconcile and
  is rebuilt; there is no torn mutable state to recover.

## The structure: 8-ary Merkle Mountain Range

Fan-out is **8** (one tunable constant, `MERGE_FANOUT`). Eight children per SoS cut the
internal-node count to ≈`(N−1)/7` for `N` leaves (versus `N−1` for binary) — a ~7×
reduction in the heaviest part of the store, since each SoS is a full substrate turn
(KV chunks + tokens + signatures + a `TreeMetadata` record). A wide fan-out also means a
summary only appears once there is real history to compress: a short conversation stays
a flat run of leaves and is read verbatim, and roll-up starts only past the eighth turn.

### Node kinds (unchanged)

- `Normal` — a user/assistant exchange turn. Content sub-leaf; not part of the forest
  spine.
- `SummaryOfTurns` (SoT) — a leaf: one summary turn over exactly one **exchange**
  (`children` = that exchange's `Normal` turns, in order). Level 1.
- `SummaryOfSummaries` (SoS) — an internal node over exactly `MERGE_FANOUT` summary
  children (SoT or SoS), all of the same level `h`; the SoS is level `h + 1`.

`tree_height` carries the level. SoT = 1; an SoS over level-`h` children = `h + 1`.

### Peaks (the orphan set)

A **peak** is a summary node no other node lists as a child. The peak set is a
complete, non-overlapping, contiguous cover of turns `0..N`, coarsest (oldest) on the
left. It is derivable as "summary nodes not referenced as anyone's child", and it is
maintained incrementally on merge. The peak set is the entry point for attention (see
*Window*).

### Append + merge rule

Leaves are appended on the right in chronological order. After appending a level-1 SoT
leaf as the rightmost peak:

```
while the last MERGE_FANOUT peaks all have equal level h:
    replace them with one new SoS of level h+1 whose children are those peaks
```

This is the base-8 carry. The number of peaks equals the base-8 digit sum of `N`
(each level holds 0..7 peaks); an eighth peak at any level appears only transiently
at the right edge and is immediately carried up. Worked example (peak levels as `N`
grows):

```
N=1  [1]           N=8  [2]           N=16 [2,2]
N=2  [1,1]         N=9  [2,1]         N=63 [2×7, 1×7]
N=7  [1×7]         N=15 [2,1×7]       N=64 [3]        (= 8^2: single peak)
```

At `N = 8^k` the forest collapses to a single peak summarising everything; between
powers it fans into up to `7·log₈N` peaks. The peaks are therefore **coarse entry
points**, not a fixed-resolution window — the BDP provenance scan drills *into* a peak
(down its immutable children) to recover detail where it is relevant. This is the same
provenance-selected attention as before, rooted at peaks instead of one global root.

The immutability boundary is precise: a node's **content** never changes, but its
**peak membership** does — a node leaves the peak set the instant a later merge gives
it a parent.

## Canonical shape + reconciliation

Given the SoT leaves in chronological order, the append+merge rule is deterministic, so
the full set of internal nodes — and each one's children — is a pure function of the
leaf sequence. That canonical shape is what "dirty" used to approximate; here it is
computed directly.

**On load** (after the substrate has replayed its records), reconcile each timeline:

1. Enumerate the persisted SoT leaves in chronological order.
2. Simulate the append+merge to obtain the canonical internal nodes, each identified
   structurally by the child node indices it must cover (resolved bottom-up — a parent's
   children are the already-resolved lower nodes).
3. For each canonical internal node, check for a persisted SoS whose `children` exactly
   equals the resolved child set. Match → reconciled, reuse its index going up. No match
   (missing, wrong children, or empty content) → the node does not reconcile.
4. Enqueue every non-reconciling node, **lowest-buildable first** (all its children must
   already exist), on the **low-priority reconcile queue**.

Reconciliation builds bottom-up across passes and converges when the persisted forest
matches the canonical shape. On the very first load against a substrate written by the
old AVL code, none of the rotated SoS nodes match the canonical 8-ary shape, so they
are simply rebuilt through the low-priority queue — the migration is free, with no
migration code.

## Two queues

The summariser drains two per-timeline queues with strict priority:

- **High priority — pending normals.** New `Normal` turns awaiting their first SoT leaf
  and the merges that follow. This is the live frontier; it must never be blocked by
  backfill.
- **Low priority — reconcile rebuilds.** Internal nodes that did not reconcile on load.
  Drained only when the high-priority queue is empty, so a reload never stalls new
  turns behind rebuilding old summaries.

Each summariser pass: drain pending normals fully (append + merge), then, if nothing is
pending, build at most one reconcile node. The periodic tick keeps draining the
low-priority queue in the background until the forest is whole.

## Window of attention

The attention window is produced by the existing score-density selection
(`select_dense`) over the **forest** rather than a single rooted tree:

- The selector scores every node (Normal sub-leaves, SoT leaves, SoS internals),
  greedily fills the token budget by score, eliminates redundant ancestors, and fills
  coverage gaps — all over the peak forest. The orphan peaks are the largest covering
  nodes; provenance scores pull in the relevant ones and the gap-fill guarantees the
  whole timeline is covered at *some* granularity.
- The **recent-raw tail** is the existing recency anchor (`RecencyConfig`): the newest
  `hard_anchor` **exchanges** are hard-anchored (`+∞`) at full fidelity, so recent
  context is verbatim while older history is the coarse peak cover. No separate
  mechanism is needed.

  The tail is measured in *exchanges*, not turns (`recency::anchor_start`). Counting
  turns cuts a tool round-trip in half at the boundary — anchoring a
  `<tool_response>` whose call falls outside the window, or a call whose result does.
  The boundary is derived once per selection as a position threshold, so scoring stays
  O(1) per node. SoT leaves need no equivalent: a leaf covers exactly one exchange, so
  the newest `hard_anchor` leaves *are* the newest `hard_anchor` exchanges.

  Anchoring the **turns** — not merely the SoT leaves that summarise them — is what
  makes this work, and is load-bearing: an SoT covers exactly one `Normal`, so if the
  newest turns are not anchored they score `0.0`, lose the greedy fit, and the
  coverage-gap step covers each one with the very summary that compresses it. The
  window then contains *no verbatim conversation at all* — only summaries. With the
  turns anchored they win the fit and the redundant-ancestor step drops their now-covered
  SoT parents, which is what yields "recent verbatim, older coarse".

`build_summary_tree_in_memory` and `select_dense` are made forest-aware: traversal and
coverage iterate over all peaks instead of descending from one root.

## Exchanges — what a leaf covers

A leaf covers one **exchange**, not one turn. A tool round-trip spans several
`Normal` turns: the model answers with `<think>` + a `<tool_call>`, the tool's
output arrives as the next turn's user half (`<tool_response>`), and only then does
the assistant answer. A leaf over half of that breaks three ways — the summary over
the call turn has no answer among its children so the compressor **invents** one (an
observed leaf claimed "15:30 UTC" when the tool returned 12:21); the summary over the
response turn has no question so its scope becomes raw JSON; and selecting one without
the other injects a tool response nothing requested.

An exchange is the maximal run of `Normal` turns joined by `TurnCoupling` records
(`summary_tree::exchange`). The record names only the call turn, and the response is
the next **`Normal`** turn — precise, not loose: summary turns share the index space
and the summariser is async, so a leaf can be recorded between a call and its
response, and raw `from + 1` would name the call turn's own summary. Everything works
in positions over the `Normal` subsequence (`exchange::over_normals`).

The record is written by the caller, not at either turn's seal, in the one window
where the round-trip is certain: after the tools have returned real output and before
the response turn is submitted. It is **authoritative** — a coupling exists iff the
round-trip happened; capture mode (calls emitted, never run) and malformed calls emit
none, so nothing is guessed from the decode.

The coupling's role is purely **grouping** — it tells the derivation which turns form
one exchange. It deliberately does *not* gate sealing, because it is written *after*
the call turn has already sealed (the tool has to run first). A summariser that sealed
a leaf the instant a turn had no coupling would freeze the call turn's leaf before its
coupling could ever land — half an exchange, with a fabricated answer.

The **closing** decision is instead a *frontier* test (`exchange::is_settled`): an
exchange is sealable once a later `Normal` turn exists beyond it. The timing guarantee
makes this race-free — the coupling is written before the response turn is submitted,
and the response precedes any later turn, so by the time **any** turn exists past an
exchange, that exchange's couplings are all already in the log; the run is both
complete and fully grouped. The newest exchange (the frontier) is left unsealed until
the next turn arrives — correct, since the live tail is anchored verbatim by recency
and needs no summary yet. Nothing is ever sealed early and then repaired: leaves are
appended only for settled exchanges, so there is no reconcile step for leaf membership.

A leaf's scope is its exchange's **head** question; the other members' user halves are
tool output, not questions. Selection expands any hit to the whole exchange
(`SummaryTree::exchange_unit`), so a scan hitting either half pulls in both — charged
and admitted all-or-nothing, at **both** admission points (the greedy fit and the
refill pass). Half an exchange is worse than none: an exchange that cannot fit is
covered by its summary via the gap-fill instead. Admitting it at only one point leaves
the other free to refill a tool response whose call was rejected as too large,
reintroducing the dangling half by the back door.

The MMR math is untouched — the base-8 carry counts *leaves* and is indifferent to
what each covers. A substrate written before couplings existed has none, so every turn
is its own exchange and the shape is exactly as before.

## The two halves: scope × content

A summary node is a compressed **exchange**, not a bare body. It keeps both halves of a real
turn — a user half (the *scope*: what this node is about) and an assistant half (the
*content*: the compressed answer). Keeping both is what makes a summary re-injectable: it
lands role-coherent, a question in user-role position and its answer in assistant-role
position, and it gives retrieval a scope to match against. A single-body summary node has
neither, and a window built from such nodes carries no user-role content at all — so the
model, asked what was discussed, reaches for whatever user-role text it *can* see (in
practice the `repo_map` layer's ``Repository index — `…`:`` headers) and reports that.

The halves are produced by different means, and that split is the core rule:

| half | produced by | why |
|------|-------------|-----|
| user (scope) | **always derived** from the children's scopes (`summary_tree::scope`, `Scope`) | a decode always speaks *as the assistant*; asking it for the question half asks it to invent a question that was never asked |
| assistant (content) | `Content::Decode` (model) or `Content::Structural` (deterministic) | an answer is exactly what a decode is good at — except where the structure is fully determined by the input |

`Scope` and `Content` are orthogonal; a layer picks each independently. There is deliberately
no user-half *prompt*: nothing would ever run it.

### Scope derivation (`Scope`)

A child's scope is its user-half tokens: for a `Normal` turn that is **the real question the
user asked**, and for a summary node it is the scope this same derivation produced one level
down. So every node's scope is grounded in real user text at every height.

- **`union`** (default) — deduplicate the children's scopes, keep the newest, and *count* the
  elided remainder (`…; (+4 earlier)`) rather than dropping it silently.
- **`line_spans`** — parse the line references the children carry and coalesce them per file:
  reads of `1-40` and `41-93` roll up to `1-93`, not a list of both (adjacent counts as
  overlapping; disjoint ranges stay apart, since merging them would claim a read that never
  happened). It parses the excerpt header a `code_reading` turn actually carries
  (``Source excerpt — `path` lines 47-93:``) *and* the compact `path:a-b` form this
  derivation itself emits, so a roll-up can re-parse its children and merge again one level
  up. Above `h=2` the ranges stop earning their tokens and only the paths are kept.

Every variant is **monotonically coarsening**: a node's scope is never larger than the naive
union of its children's, and never grows with height (the entry budget halves per level —
`MERGE_FANOUT >> (h−1)`, never below 1). A plain union would grow the scope toward the root
until it was the whole conversation — the same failure the structural roll-up exists to avoid
on the content side.

A `SummaryOfTurns` leaf has exactly one child, so it keeps that turn's question verbatim: the
leaf is a faithful `(question, compressed answer)` pair.

## Structural content (directory trees)

The `repo_map` layer's summaries are directory *skeletons* — directory paths and their files.
The model cannot be trusted with them: a faithful *merge* of children just unions their trees,
so the skeleton grows toward the root instead of compressing; and given the thin input a
summary node carries, a decode fabricates placeholders (`/path/to/repo`), emits shell
snippets, and rambles (a model ceiling no prompt fixes — the *input* is degenerate).

For a directory tree the structure is **fully determined by the children**, so the model is
not needed at all. `repo_map` sets `content: structural` (`Content::Structural`) on **both**
its levels, and each is built deterministically with **no model decode**
(`scheduler::seal_structural_turn` → `summary_tree::structural`):

- a **leaf** (`leaf_skeleton`) strips the `(N lines, …)` size annotations off its one scan
  turn, keeping the full skeleton — a leaf is the base detail, so files are kept;
- a **summary-of-summaries** (`structural_rollup`) reconstructs every directory path from the
  children's sealed skeletons (`assistant_text_of`), robust to both the full-path-header form
  `zend/src/code_read/` and the nested-indent form `zend/` → `  - examples/`, then
  **truncates each path to a depth set by the node's tree height** — `h=2` keeps three
  segments, `h=3` two, `h≥4` only the top-level directory — and deduplicates. Each step up
  drops a segment, so a SoS is provably coarser than its children; files are dropped (they
  live in the leaves at full fidelity), so a roll-up names *which directories exist*. The
  `height` rides the `ProbeRequest` from the summariser.

A structural level derives **both** halves from the children's skeletons, and a `scope:` key
here is rejected rather than silently ignored. For a directory tree the skeleton is the
authoritative statement of what the node covers, whereas the children's scopes are not: a
`repo_map` scan turn's user half is prose (``Repository index — `candle-nn/src`:``), so a
directory derivation over it would parse the prose as a path. The scope is therefore the
distinct top-level directories of the skeleton.

Both halves are then encoded and sealed through the shared `seal_compression_turn` — the same
think-strip, marker-framing, and role-coherent re-prefill the decode path uses. Skipping the
decode removes the cost *and* the fabrication surface: for structure the model is not
consulted at all. Other layers (`code_reading`, the analyses) summarise code and artifacts
rather than paths, so their content stays `decode`.

## What is deleted

- `TreeMetadataPayload.dirty`, `TreeNodeMeta.dirty`, `Node.dirty`.
- `SummaryTree`: `mark_dirty`, `is_balanced`, `insert_leaf_rightmost`,
  `avl_insert_rightmost`, `rebalance`, `rotate_left`/`rotate_right`,
  `child_left_height`/`child_right_height`, `refresh_height`, `set_summary_children`,
  the single `root` pointer.
- `Substrate`: `dirty_summary_set`, `mark_summary_dirty`, `clear_summary_dirty`,
  `pop_oldest_dirty`, `dirty_summary_len`, `tree_root`/`set_tree_root`/`tree_root_of`
  (replaced by the derived peak set).
- `summariser`: `sweep_one_dirty`, `perform_avl_insert_rightmost`,
  `recursive_avl_insert`, `seal_leaf_and_avl_insert`'s AVL specifics,
  `commit_tree_to_substrate`'s dirty logic.

## What replaces it

- `SummaryTree`: `peaks()`, `append_leaf` + the `MERGE_FANOUT`-wide carry-merge,
  forest-aware `post_order`/coverage, `MERGE_FANOUT`.
- `Substrate`: derived `peaks_of(timeline)`, low-priority reconcile queue
  (`push_reconcile`/`pop_reconcile`/`reconcile_len`), `canonical_forest` derivation.
- `summariser`: `absorb_pending_turns` (append + carry-merge), `reconcile_pass`
  (one node per pass, low priority).

## Invariants (asserted in tests)

1. After `N` appends the peak levels equal the base-`MERGE_FANOUT` (base-8) digits of
   `N` (count per level).
2. Every SoS has exactly `MERGE_FANOUT` children, all of equal level, and is level
   `child_level + 1`.
3. No node's children ever change after creation (immutability).
4. The persisted forest, after reconcile, exactly equals the canonical shape for `N`.
5. The peak set is a contiguous, non-overlapping cover of `0..N`.
