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

## The structure: ternary Merkle Mountain Range

Fan-out is **3** (one tunable constant, `MERGE_FANOUT`). Three children per SoS halves
the internal-node count versus binary (≈`(N−1)/2` internals for `N` leaves instead of
`N−1`), which halves the heaviest part of the store — each SoS is a full substrate
turn (KV chunks + tokens + signatures + a `TreeMetadata` record).

### Node kinds (unchanged)

- `Normal` — a user/assistant exchange turn. Content sub-leaf; not part of the forest
  spine.
- `SummaryOfTurns` (SoT) — a leaf: one summary turn over exactly one `Normal` turn
  (`children = [normal_idx]`). Level 1.
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
while the last 3 peaks all have equal level h:
    replace them with one new SoS of level h+1 whose children are those 3 peaks
```

This is the base-3 carry. The number of peaks equals the base-3 digit sum of `N`
(each level holds 0, 1, or 2 peaks); a third peak at any level appears only transiently
at the right edge and is immediately carried up. Worked example (peak levels as `N`
grows):

```
N=1 [1]            N=4 [2,1]          N=7 [2,2,1]
N=2 [1,1]          N=5 [2,1,1]        N=8 [2,2,1,1]
N=3 [2]            N=6 [2,2]          N=9 [3]        (= 3^2: single peak)
```

At `N = 3^k` the forest collapses to a single peak summarising everything; between
powers it fans into up to `2·log₃N` peaks. The peaks are therefore **coarse entry
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
old AVL code, none of the rotated SoS nodes match the canonical ternary shape, so they
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
  leaves/turns are hard-anchored at full fidelity, so recent context is verbatim while
  older history is the coarse peak cover. No separate mechanism is needed.

`build_summary_tree_in_memory` and `select_dense` are made forest-aware: traversal and
coverage iterate over all peaks instead of descending from one root.

## Structural roll-up (directory trees)

The `repo_map` layer's summaries are directory *skeletons* — directory paths and their
files. A summary-of-summaries over them cannot trust the model: given the thin scope its
children carry, the user-half decode fabricates placeholders (`/path/to/repo`), emits shell
snippets, and rambles (a model ceiling no prompt fixes — the *input* is degenerate); and a
faithful *merge* just unions the child trees, so the skeleton grows toward the root instead
of compressing.

For a directory tree, though, the structure is **fully determined by the children**, so the
model is not needed at all. `repo_map` sets `summary.summaries.mode: structural`
(`SummaryMode::Structural`), and a structural SoS is built deterministically with **no model
decode** (`scheduler::seal_structural_turn` → `summary_tree::structural::structural_rollup`):

1. reconstruct every directory path from the children's sealed skeletons
   (`assistant_text_of`), robust to both the full-path-header form `zend/src/code_read/` and
   the nested-indent form `zend/` → `  - examples/`;
2. **truncate each path to a depth set by the node's tree height** — `h=2` keeps three path
   segments, `h=3` two, `h≥4` only the top-level directory — then deduplicate. Each step up
   drops a segment, so a SoS is provably coarser than its children; files are dropped (they
   live in the leaves at full fidelity). The `height` rides the `ProbeRequest` from the
   summariser.
3. **derive** the scope (the user-half) as the distinct top-level directories.

The two halves are encoded and sealed through the shared `seal_compression_turn` path (the
same role-coherent re-prefill the model-decode path uses). Skipping the decode removes the
cost *and* the fabrication surface. This is the tools categorize→assign split taken to its
conclusion: for structure the model is not consulted at all. The leaf (`turns`) level is
always single-pass; only `repo_map`'s `summaries` is `structural`. Other layers
(`code_reading`, analyses) summarise code definitions/artifacts, not paths, so they stay
single-pass.

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

- `SummaryTree`: `peaks()`, `append_leaf` + `merge_full_peaks` (ternary), forest-aware
  `post_order`/coverage, `MERGE_FANOUT`.
- `Substrate`: derived `peaks_of(timeline)`, low-priority reconcile queue
  (`push_reconcile`/`pop_reconcile`/`reconcile_len`), `canonical_forest` derivation.
- `summariser`: ternary `absorb_pending_turns` (append + carry-merge), `reconcile_pass`
  (one node per pass, low priority).

## Invariants (asserted in tests)

1. After `N` appends the peak levels equal the base-3 digits of `N` (count per level).
2. Every SoS has exactly `MERGE_FANOUT` children, all of equal level, and is level
   `child_level + 1`.
3. No node's children ever change after creation (immutability).
4. The persisted forest, after reconcile, exactly equals the canonical shape for `N`.
5. The peak set is a contiguous, non-overlapping cover of `0..N`.
