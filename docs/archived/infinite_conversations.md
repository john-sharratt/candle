# Infinite Conversations — Unbounded Turn History with Provenance-Selected Recall

> **Status — Design v1, ready to build.** Specifies the full conversation
> mechanics for a substrate-backed, never-ending conversation timeline:
> continuous summarisation via probe forward passes, a self-balancing
> binary summary tree persisted as ordinary substrate turns with type
> tagging, an async summariser thread, projection via score-density
> selection over the tree, and a three-tier test harness (synthetic
> algorithm tests / cached-substrate fixtures / debug_id-resumable
> grow-conversation runs) culminating in the unbounded-window recall
> stress test. Open questions are tracked in §12 — nothing structural
> is deferred. The implementation plan in §11 is phased so every
> feature ships in a self-contained commit.

---

## 1. Abstract

This document specifies how the substrate runs an **infinite conversation**:
a single conversation timeline that grows without bound, never ends, keeps
the model coherent across arbitrary depth, and answers every generation
step in constant wall-clock under bounded resources.

It is the conversation-layer counterpart to two earlier designs:

- `kv_tier_migration.md` — gives the KV cache its three-tier storage
  substrate (VRAM ↔ RAM ↔ NVMe), making conversation state durable.
- `attention_provenance.md` — gives the system a Q-vector fingerprint
  index and a flat CPU-side scan (XOR + popcount over depth-banded
  sign-bits) that surfaces relevant content from arbitrary-scale
  corpora in 3–10 ms.

Those two designs solved *storage* and *retrieval* over a flat list of
turns. This document specifies the **organisation of that flat list into
a self-balancing tree of summary nodes**, an async pipeline that builds
that tree by running model-driven probes, and a score-density
projection algorithm that fills the layer's token budget by picking
the highest-scoring subset of tree nodes (under the existing
provenance scorer's verdict) and eliminating redundant ancestors.

The core technical claim:

> **The model's own attention, replayed against frozen summaries it
> previously produced, is the navigation mechanism for an unbounded
> conversation tree.** Every summary node's Q-vector fingerprint is the
> cognitive state of the model attending to its children at probe time;
> at decode time, the current probe Q chooses descent paths by Hamming
> agreement with those stored Qs. No clustering algorithm, no embedding
> model, no separately-trained retriever — the same forward pass that
> generated the summary is what later finds it.

---

## 2. Goal & Scope

### In scope

- A conversation can grow to **arbitrary depth** (millions of turns) on a
  single workstation, with bounded VRAM and bounded per-turn latency.
- The model stays **coherent at depth**: relevant prior context, whether
  5 turns or 5 000 turns ago, is recovered and made attendable at decode
  time.
- Operation is **continuous** — no "session end", no offline reindex
  pass, no defrag pause. Every cost is amortised into the ongoing
  per-turn budget.
- A **test harness** can create or resume a named conversation by
  `debug_id` and grow it by N turns per invocation, so quality and
  performance metrics can be measured at arbitrary depth without
  re-running the full preceding history each time.

### Out of scope (this document)

- The kernel-level decode-attention path
  (`candle-nn::kv_cache`, `candle-kernels/paged-decode/`).
- The three-tier storage substrate itself (`kv_tier_migration.md`).
- The Q-vector fingerprint format, the BDP scan algorithm, and the depth-
  band aggregation (`attention_provenance.md`). This document specifies
  how that index is **populated** and **queried** by the conversation
  runtime; the index format itself is unchanged.

---

## 3. The Asymmetries That Make This Tractable

Three structural asymmetries collapse what would be an intractable problem
into a clean design surface.

### 3.1 Capture is bilateral; scan is unilateral

Q-vector capture (the provenance fingerprint) happens in **both** phases
of every turn — prefill and decode. The `extract_prov_after_step` pass
in the scheduler stamps Q sign-bits onto every just-produced K/V chunk,
unconditionally, on every token.

What is asymmetric is the **scan**:

- The expensive BDP scan over the full historical corpus fires **at
  projection time** — once per assistant turn — to decide which ancient
  context to swap into the slot for that turn's prefill + decode.
- The user-prefill phase does *not* run its own scan; it rides on the
  projection the previous turn already produced, topped up by the
  right-anchor of recent turns.
- This bounds the global scan rate to **one scan per assistant turn**,
  regardless of token count or conversation depth.

### 3.2 Summary as forward continuation, not re-encoding

A summary node is **not** built by re-prefilling the children's text and
running a fresh forward pass. It is built by:

1. Injecting the children's existing K/V chunks at the head of a probe
   slot (zero-copy, metadata only).
2. Decoding a short summary continuation against that prefix.
3. Sealing the decoded tokens as a fresh substrate turn — whose Q
   sign-bits, captured by the normal `extract_prov_after_step` path, are
   *literally the cognitive state of the model attending to the
   children* at every layer.

The summary's stored Q-fingerprint is therefore a faithful provenance
proxy for everything in its subtree. A later probe Q that agrees with
the summary's Q is — by construction — the same kind of probe Q that
would agree with the high-relevance leaves under that summary.

### 3.3 Continuous, never-ending

There is no "wrap up" event. Every per-turn step pays the same amortised
cost for summarisation, indexing, eviction, and persistence. The
foreground tempo (user-turn arrival rate) is decoupled from the
background tempo (summariser throughput) by a `pending` queue (§7), so
the foreground never stalls on the background and the background can
take as long as it needs without blocking generation.

---

## 4. Architecture Overview

```
                FOREGROUND THREAD                  BACKGROUND THREAD
                (scheduler, sync)                  (summariser, async)
                ═════════════════                  ═══════════════════

   user turn ──►┌──────────────────┐
                │ ① Projection (§8)│
                │   ▸ score-density │◄── reads tree shape via Conversation
                │   ▸ selection     │    handle (RwLock-protected read)
                │   ▸ fits layer    │
                │     window        │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │ ② Prefill +      │
                │   decode the     │
                │   assistant turn │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │ ③ Seal turn      │
                │   ▸ enqueue to    │── push ──┐
                │     pending       │           │
                │   ▸ fire summary │           ▼
                │     trigger       │   ┌──────────────────┐
                └──────────────────┘   │ ④ Dequeue one    │
                                       │   pending turn    │
                                       └────────┬─────────┘
                                                │
                                       ┌────────▼─────────┐
                                       │ ⑤ Summary probe  │
                                       │   (§6): inject    │
                                       │   children K/V    │
                                       │   + structured    │
                                       │   decode          │
                                       └────────┬─────────┘
                                                │
                                       ┌────────▼─────────┐
                                       │ ⑥ Atomic write   │
                                       │   ▸ extend open-  │
                                       │     cluster, OR   │
                                       │   ▸ freeze leaf + │
                                       │     AVL rotate    │
                                       │   ▸ mark rotated  │
                                       │     ancestors     │
                                       │     dirty         │
                                       │   ▸ persist as    │
                                       │     a normal turn │
                                       │     with type tag │
                                       └────────┬─────────┘
                                                │
                                       ┌────────▼─────────┐
                                       │ ⑦ Dirty-node     │
                                       │   sweep: regen    │
                                       │   one ancestor    │
                                       │   per pass        │
                                       └──────────────────┘
```

Two threads, one queue, one substrate. The atomic boundary at step ⑥ is
a single write that moves a turn from `pending` into the tree (as a leaf
or absorbed into the rightmost-open-cluster) and persists the change.
Projection (step ①) sees either before-or-after, never partial — RwLock
semantics on the `Conversation` handle ensure this.

---

## 5. The Conversation Summary Tree

### 5.1 Shape

A **balanced binary tree** (AVL-style; height invariant ≤ 1) of
summary nodes, with Normal turns hanging off the binary leaves as
content sub-leaves:

```
   ┌─────────────────────┐
   │ SummaryOfSummaries  │  ◄── kind = TurnKind::SummaryOfSummaries
   │ (internal node)     │      Two summary children.
   └─────────────────────┘      Produced by the §6 probe over those
                                two child summary nodes.

   ┌─────────────────────┐
   │ SummaryOfTurns      │  ◄── kind = TurnKind::SummaryOfTurns
   │ (binary-tree leaf)  │      N Normal-turn children (the contiguous
   └─────────────────────┘      run it covers).
                                Produced by the §6 probe over those
                                Normal turns.  Compresses [t_a..t_b]
                                into one ~20-token summary turn.

   ┌─────────────────────┐
   │ Normal              │  ◄── kind = TurnKind::Normal
   │ (content sub-leaf)  │      The user's / assistant's actual
   └─────────────────────┘      conversational content.  Children of
                                a SummaryOfTurns leaf in the tree,
                                but the AVL balance invariant ignores
                                them — only the binary structure of
                                summary nodes is balanced.
```

**Structural distinction**: the **binary AVL property** applies to
`SummaryOfSummaries → SummaryOfTurns` only.  Normal turns are content
under the `SummaryOfTurns` leaves; their count is variable per leaf
(determined by where the §6 probe ruled a topic-boundary), and they do
not participate in the balance invariant.  For the score-density
selection (§8), however, **every turn is a scoreable node** — Normal
turns, SummaryOfTurns leaves, and SummaryOfSummaries internals all
appear in the scan corpus and can be selected independently.

Every node — internal, binary-leaf, or content sub-leaf — is persisted
as an ordinary turn in the substrate, tagged by `kind`. There is **no
parallel storage system** for summary nodes; the BDP scan, the
persistence tier flow, the cold-load path all treat them as turns.

### 5.2 Geometry

```
                 ┌─────────────────────────────┐
                 │     SummaryOfSummaries      │  level 3 (root)
                 │     (root summary)          │
                 └──────────────┬──────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                            │
        ┌─────────▼─────────┐        ┌─────────▼─────────┐
        │ SummaryOfSummaries│        │ SummaryOfSummaries│   level 2
        └─────────┬─────────┘        └─────────┬─────────┘   (internal)
                  │                              │
              ┌───┴───┐                      ┌───┴────────┐
              │       │                      │            │
        ┌─────▼─────┐ │                ┌─────▼──────┐    │
        │ Summary   │ │                │ Summary    │    │   level 1
        │ OfTurns   │ │                │ OfTurns    │    │   (binary
        └─────┬─────┘ │                └─────┬──────┘    │    leaves)
              │       │                      │            │
              │       ▼                      │            ▼
              │  ┌─────────┐                 │       ┌───────────┐
              │  │ Sum     │                 │       │ OPEN      │
              │  │ OfTurns │                 │       │ CLUSTER   │
              │  └────┬────┘                 │       │ (in flux) │
              │       │                      │       └──────┬────┘
              │   t_5 t_6 t_7 t_8        t_9 t_10           │   level 0
              │                                              │   (Normal
            t_1 t_2 t_3 t_4                          t_11 t_12 t_13 ...  sub-
                                                              │   leaves)
                                                              │
                                                              ▼
                                                       ┌──────────────┐
                                                       │ pending      │
                                                       │ queue        │
                                                       │ (turns not   │
                                                       │  yet absorbed│
                                                       │  by the      │
                                                       │  summariser) │
                                                       └──────────────┘
```

- **Internal nodes** (level ≥ 2) are `SummaryOfSummaries`, each with
  exactly two summary children.  The AVL balance invariant operates
  here.
- **Binary leaves** (level 1) are `SummaryOfTurns`, each covering a
  contiguous run of Normal turns.  The AVL balance invariant treats
  these as the leaves of its tree.
- **Content sub-leaves** (level 0) are `Normal` turns — the actual
  user/assistant content.  Not in the AVL structure; their count per
  SummaryOfTurns parent is variable.
- The **rightmost binary leaf** is the *open cluster* — a
  SummaryOfTurns in flux, still absorbing newly-arrived Normal turns
  as new sub-leaves.  Its summary content rewrites every time a new
  Normal turn is absorbed.
- **Pending** turns sit outside the tree (no parent yet) until the
  summariser thread processes them and attaches them to the open
  cluster or splits off a new one.

### 5.3 Persistence

Every tree node is a normal substrate turn. The substrate's existing
record layout — `record_turn`, `set_section_full`, persistence tier flow,
the redo log — handles all of them. The only new field is the kind tag:

```rust
enum TurnKind {
    Normal,                 // ordinary user/assistant content
    SummaryOfTurns,         // leaf in the summary tree
    SummaryOfSummaries,     // internal node in the summary tree
}
```

Each turn entry on the substrate carries:

| Field                    | Per kind         | Source                            |
|--------------------------|------------------|-----------------------------------|
| `kind: TurnKind`         | all              | New: this design                  |
| `children: Vec<TurnKey>` | summary nodes    | Tree structure                    |
|                          | — SumOfSummaries: exactly 2 (binary)                 |
|                          | — SumOfTurns:    N Normal-turn children (variable)   |
|                          | — Normal:         empty                              |
| `tree_height: u8`        | summary nodes    | AVL bookkeeping (binary structure) |
| Token IDs                | all              | Existing: the turn's content      |
| Sealed K/V chunks        | all              | Existing: the turn's GPU/CPU/disk K/V |
| Q sign-bits              | all              | Existing: ProvenanceFile via `SigEntry` |
| `dirty: bool`            | summary nodes    | New: marks "summary needs regeneration" |

The redo-log records for summary turns are byte-identical in format to
Normal-turn records; only the `kind` discriminator + the children/height
metadata differ. Nothing in the cold-load path needs special-casing.

### 5.4 The open cluster

The rightmost leaf is privileged: it is allowed to *mutate*. Specifically:

- When a new `Normal` turn arrives in `pending` and the probe verdict is
  **coherent**, the open cluster's summary is **rewritten in place**:
  the old summary turn record is superseded by a new one (a new
  `TurnKey`, replacing the old in the leaf slot). The old summary's
  bytes are eligible for eviction.
- When the probe verdict is **boundary**, the open cluster is **frozen**:
  no further mutations. A new open cluster is appended to the right,
  containing only the new turn. The frozen leaf is inserted into the
  AVL — which may trigger rotations.

The open cluster is the only place the tree changes shape during steady-
state; every other update is an O(log N) AVL rotation.

### 5.5 AVL rotations + dirty propagation

When a frozen leaf gets inserted into the AVL, standard rotations
restore the height invariant. A rotation changes a node's children, so
its summary is no longer faithful — it gets marked **dirty**:

```
   Before insert + left rotation:                After:

           A                                          B
          ╱╲                                         ╱╲
         X  B                                       A  Z
            ╱╲           ──────────►              ╱╲
           Y  Z         (left rotation)          X  Y
                                                       
          (A: dirty after Y moves under it; B: dirty after children swap)
```

Every node whose children changed during a rotation is marked dirty.
Their **regeneration** is deferred: the §7 dirty-node sweep regenerates
one dirty node per background pass, amortised against future turns.

Dirty internal nodes are still scoreable — their stored Q-fingerprint is
slightly stale (it summarises the *previous* children) but it still
encodes related content. The walker (§8) scores them at face value;
worst case is a few turns of suboptimal navigation until the sweep
catches up. The architecture absorbs the lag because the walker is
Q-driven — a stale summary that no Q wants to descend into costs
nothing.

### 5.6 Restart reload

On daemon restart, the substrate reconstructs from the redo log
(`kv_tier_migration.md::§5.7`). Each turn comes back tagged with its
`kind`. The reconstruction pass walks the turn graph and:

1. Builds the in-memory tree structure from `children` + `tree_height`
   fields.
2. For each summary node: checks that its children all exist as substrate
   turns. **If any child is missing**, the summary node is enqueued for
   regeneration in the §7 pending queue. The summary stays in place
   structurally (with its stale Q-fingerprint) until the queue drains.
3. If a `pending`-state turn is found (a `Normal` turn whose
   `SummaryOfTurns` parent doesn't reference it yet), it is enqueued to
   the summariser.

Restart never fails on a partially-built tree — it always converges to a
consistent state by re-queueing what's missing.

---

## 6. The Summary Probe

A single probe template covers both `SummaryOfTurns` (over Normal turns)
and `SummaryOfSummaries` (over two child summary nodes). The slot
construction is identical except for what's injected at the head.

### 6.1 Probe slot recipe

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  Probe slot                                                    ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║                                                                ║
   ║  ┌──────────────────────────────────────────────────────────┐ ║
   ║  │ ① Synthetic system section:                              │ ║
   ║  │    "You are a summariser.  Read the turns above and      │ ║
   ║  │     decide whether they form a single coherent topic.    │ ║
   ║  │     Output JSON only:                                     │ ║
   ║  │       {\"coherent\": true,  \"summary\": \"...\"}         │ ║
   ║  │       {\"coherent\": false, \"split_at\": N}              │ ║
   ║  │     where N is the index of the first turn of the new    │ ║
   ║  │     topic."                                               │ ║
   ║  │                                                            │ ║
   ║  │  (precomputed K/V, pinned in the substrate as a normal   │ ║
   ║  │   section — cached forever, no per-probe cost.)          │ ║
   ║  └──────────────────────────────────────────────────────────┘ ║
   ║  ┌──────────────────────────────────────────────────────────┐ ║
   ║  │ ② inject_sealed_at_tail:                                 │ ║
   ║  │      child K/V chunks pulled from substrate              │ ║
   ║  │   For SummaryOfTurns:    Normal turns t_a .. t_b         │ ║
   ║  │   For SummaryOfSummaries: two child summary turns        │ ║
   ║  │                                                            │ ║
   ║  │  Zero-copy metadata clone — no DMA, no forward pass.     │ ║
   ║  └──────────────────────────────────────────────────────────┘ ║
   ║  ┌──────────────────────────────────────────────────────────┐ ║
   ║  │ ③ User-turn prefill:                                      │ ║
   ║  │      "Summarise the above turns."                         │ ║
   ║  │                                                            │ ║
   ║  │  Short prefill (~5 tokens).                               │ ║
   ║  └──────────────────────────────────────────────────────────┘ ║
   ║  ┌──────────────────────────────────────────────────────────┐ ║
   ║  │ ④ Assistant-turn opening + decode:                       │ ║
   ║  │      Prefill: "{"                                         │ ║
   ║  │      Decode:  rest of the JSON (~30 tokens with grammar  │ ║
   ║  │               constraint).                                │ ║
   ║  │                                                            │ ║
   ║  │  The decoded tokens attend, at every layer, to every     │ ║
   ║  │  child K/V chunk injected at step ②.  Their captured Q   │ ║
   ║  │  sign-bits are the summary's fingerprint.                │ ║
   ║  └──────────────────────────────────────────────────────────┘ ║
   ║                                                                ║
   ╚═══════════════════════════════════════════════════════════════╝
                                ▼
              ┌──────────────────────────────────────┐
              │ Seal the decoded turn.               │
              │   ▸ kind = SummaryOfTurns | OfSums  │
              │   ▸ children = injected child keys   │
              │   ▸ Q sign-bits land in              │
              │     ProvenanceFile via standard       │
              │     extract_prov_after_step path     │
              │   ▸ persists via existing tier flow  │
              └──────────────────────────────────────┘
```

### 6.2 Structured output

The probe emits exactly one of two JSON shapes:

```json
{ "coherent": true,  "summary": "<one-line digest of the turns>" }
{ "coherent": false, "split_at": <turn index> }
```

The summariser thread parses the JSON. On `coherent: true`, the digest
text *is* the summary turn's content; on `coherent: false`, the digest
text is discarded and the open cluster is split at `split_at`. In both
cases, the **Q sign-bits captured during decode** are what gets indexed
— whether they belong to a kept summary or a discarded probe.

### 6.3 Why the Q-fingerprint is the right signal

The fingerprint is captured at the 3 depth bands (lower / mid / upper
~15% / 50% / 85% of layers per `attention_provenance.md::§1.2`). At
each band, the summary token's Q is — at that band — a non-linear
attention pool over the children's K/V at the same band. The band
semantics propagate without mixing as we climb the tree:

```
   Walking down from root:
   
     probe Q  ─── agree band by band ───►   summary Q (root)
                                                 │
                                            agree band by band
                                                 │
                                                 ▼
                                            summary Q (level 2)
                                                 │
                                            agree band by band
                                                 │
                                                 ▼
                                            summary Q (leaf)
                                                 │
                                            agree band by band
                                                 │
                                                 ▼
                                              Normal turn Q

   Each step: agreement(probe Q, node Q) per depth, combined via
   the ScoreFormula::aggregate the projection already uses for turns.
```

This is the same Q-against-Q Hamming agreement (`signature.rs::agreement`,
`scan.rs::BdpScanner::scan`) that the existing BDP scan uses for turn
scoring — no new scoring kernel is needed. Summary nodes are just
additional entries in the scan corpus.

---

## 7. The Async Summariser Thread

### 7.1 Lifecycle

One summariser thread per workspace. Spawned alongside the persistence
thread at engine start; mirrors its idiom (trigger + tick loop).

```
   ┌──────────────────────────────────────────────────────────────┐
   │  loop {                                                       │
   │      select! {                                                │
   │          tick (every 250ms) ─► run_pass()                    │
   │          trigger             ─► run_pass()                    │
   │          shutdown            ─► drain + exit                  │
   │      }                                                         │
   │  }                                                             │
   │                                                                │
   │  run_pass() {                                                  │
   │      while let Some(turn) = pending.pop_front() {              │
   │          probe_result = run_summary_probe(turn);               │
   │          atomic_substrate_write(turn, probe_result);           │
   │      }                                                         │
   │      if let Some(dirty_node) = dirty_set.pop_oldest() {        │
   │          regen = regen_summary_probe(dirty_node);              │
   │          atomic_substrate_write_regen(dirty_node, regen);      │
   │      }                                                         │
   │  }                                                             │
   └──────────────────────────────────────────────────────────────┘
```

The thread submits probe RPCs to the same batched scheduler the
foreground uses. Probes share GPU time with foreground turns —
amortised by the scheduler's batching. The summariser thread is just
an *issuer*, not a separate inference engine.

### 7.2 The atomic substrate write

```
   ╔══════════════════════════════════════════════════════════════╗
   ║  atomic_substrate_write(turn, probe_result) {                 ║
   ║                                                                ║
   ║      take Conversation.write() — RwLock acquired               ║
   ║                                                                ║
   ║      if probe_result.coherent {                                ║
   ║          ▸ supersede open cluster's summary turn               ║
   ║            (new TurnKey, old turn evictable)                   ║
   ║          ▸ children += turn                                    ║
   ║      } else {                                                  ║
   ║          ▸ freeze open cluster as a fixed leaf                 ║
   ║          ▸ AVL insert the frozen leaf                          ║
   ║          ▸ rotations: mark rotated ancestors as dirty          ║
   ║          ▸ create a new open cluster from `turn`              ║
   ║      }                                                         ║
   ║                                                                ║
   ║      persist the changed substrate state via the existing      ║
   ║      record_turn path — including dirty-bit updates            ║
   ║                                                                ║
   ║      release RwLock                                            ║
   ║  }                                                             ║
   ╚══════════════════════════════════════════════════════════════╝
```

The write lock is brief — milliseconds. Foreground projection reads
(step ①) take the read side of the same RwLock; they see either
before-state or after-state.

### 7.3 Dirty-node sweep

A separate small data structure holds the set of dirty summary nodes.
After every pending-turn processing, if dirty nodes exist, the thread
processes **at most one** dirty regeneration per pass — amortised
against foreground turn cadence. Regeneration:

1. Pop the oldest dirty node.
2. Run the §6 probe with the node's two child summary K/Vs injected.
3. Atomic-write the new summary; the old version is superseded.
4. Mark the node's parent dirty (cascade), but do **not** chase the
   cascade — let the next sweep pass pick it up.

The cascade is bounded: in the worst case, an insertion's rotations
mark O(log N) ancestors dirty; the sweep takes O(log N) future passes
to clean them. Foreground sees stale-but-coherent summaries throughout.

### 7.4 Backpressure

If the foreground turn rate exceeds the background's summarisation
rate, `pending` grows. The system absorbs this through the slot
composition described in §8.3: pending turns are included in the slot
**verbatim** as their own slot region, displacing score-density-
selected tree content as their region grows. Generation does not
stall — see §9 for the steady / pressured / saturated regimes.

---

## 8. Projection — Score-Density Selection

### 8.1 The free-scan insight

The existing `BdpScanner::scan` (called from
`Scheduler::reproject_view`, `scheduler/mod.rs:3346`) already
scans the **entire substrate turn corpus** against the live probe Q
every reprojection — and summary nodes *are* substrate turns (per
§5.3). So **every node in the tree gets a Q-agreement score for free,
as a side-effect of the reprojection scan we already run**.

This makes a recursive top-down tree walk unnecessary. We have, at
projection time:

```
   agree_score : TurnKey → f32     (every node in the tree, scored)
```

…and we just need to pick the top-density subset that fits the budget
and covers the conversation history.

### 8.2 Effective score: provenance + recency-decay competition

A turn's *effective* score is the **max** of two independent signals:

```
   effective_score(node) = max(provenance_score(node), recency_score(node))
```

where:

- **`provenance_score`** is `ScoreFormula::aggregate(per_depth_scores)`
  using the layer's existing `depth_weights`. The aggregate used is
  **`top_k_mean`** (the robust "most-relevant cluster of tokens" signal
  from `PerDepthScores`) — robust against single-token outliers, while
  still letting a strong matching span dominate. See §12 for the
  rationale.
- **`recency_score`** is a hard anchor for the last 3 leaves, then an
  exponentially-decaying score for older right-edge leaves:

```
   recency_score(node) =
       +∞           if node ∈ last 3 right-edge leaves    (hard anchor)
       d^(k - 3)    if node is the kᵗʰ-most-recent leaf, k > 3
       0            otherwise

   where 0 < d < 1 is the decay rate (default d = 0.8).
```

The hard anchor for the last 3 leaves guarantees local conversational
coherence regardless of Q. After that, recency *competes* with
provenance: an old turn with a strong Q match wins over a moderately
recent turn with no match; a moderately recent turn with no Q signal
still beats an ancient turn with no signal. The right edge bleeds
gracefully into the rest of the tree instead of having a hard cut-off.

```
   Score plot for the right-edge leaves
   (provenance score = some flat noisy line; recency score decays)

         score
           ▲
       +∞ ─┤■  ■  ■
           │            d⁰  d¹  d²   d³  d⁴  d⁵  ...
           │            ┃   ┃   ┃    ┃   ┃   ┃        ← recency
           │            ▼   ▼   ▼    ▼   ▼   ▼
           │             ●        ●           ●  ●    ← provenance
           │                                            (noise + matches)
           │
           └─────────────────────────────────────────────► leaf index
              t_n  t_n−1  …    │     wherever provenance lights up,
              ▲ ▲ ▲             │     it wins from there.
              │ │ └─ last 3 hard anchors
```

### 8.3 Slot composition

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  Slot — total layer.window tokens                              ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║                                                                ║
   ║   ◄── older content                              recent ───►   ║
   ║                                                                ║
   ║  ┌───────────────────────────────────────┬──────────────────┐ ║
   ║  │ Selected nodes by effective_score      │ Pending          │ ║
   ║  │   ▸ Last 3 leaves (hard anchor)        │ (turns not yet   │ ║
   ║  │   ▸ Top-density provenance hits        │  absorbed into   │ ║
   ║  │     (leaves + summaries, any depth)    │  the tree)       │ ║
   ║  │   ▸ Decay-vs-walk competition on the   │                  │ ║
   ║  │     right edge                         │                  │ ║
   ║  │   ▸ Redundant ancestors dropped (§8.5) │                  │ ║
   ║  │   ▸ Default coverage by root summary   │                  │ ║
   ║  │                                          │                  │ ║
   ║  │       ◄────── elastic ──────►            │  ◄── grows under │ ║
   ║  │                                          │      backpressure │ ║
   ║  └───────────────────────────────────────┴──────────────────┘ ║
   ║                                                                ║
   ╚═══════════════════════════════════════════════════════════════╝
```

Pending turns are mandatory — they're recent and have no summary yet.
Everything else is selected by score-density.

### 8.4 Selection algorithm

```
   ╔════════════════════════════════════════════════════════════════╗
   ║  select_dense(probe_Q, tree, scores, budget):                  ║
   ║                                                                 ║
   ║      // Step 1 — every node has a score (from existing scan).  ║
   ║      effective : NodeKey → f32                                  ║
   ║      for node in tree.all_nodes():                              ║
   ║          effective[node] = max(                                 ║
   ║              ScoreFormula::aggregate(scores[node]),             ║
   ║              recency_score(node)                                ║
   ║          )                                                       ║
   ║                                                                 ║
   ║      // Step 2 — greedy fit by score, highest first.            ║
   ║      ranked := tree.all_nodes() sorted by effective desc        ║
   ║      selected := {}                                             ║
   ║      used := 0                                                  ║
   ║      for node in ranked:                                        ║
   ║          if used + node.tokens ≤ budget:                       ║
   ║              selected += { node }                               ║
   ║              used     += node.tokens                            ║
   ║                                                                 ║
   ║      // Step 3 — eliminate redundant ancestors, bottom-up.     ║
   ║      // A node is redundant when its full subtree is already   ║
   ║      // covered by other set members below it.                 ║
   ║      eliminate_redundant(tree, selected, used)                 ║
   ║                                                                 ║
   ║      // Step 4 — fill coverage gaps, largest gap first.        ║
   ║      // Greedy fit may have left whole conversation ranges     ║
   ║      // uncovered (no ancestor in `selected`).  Add the        ║
   ║      // smallest node covering each gap, in priority order     ║
   ║      // (largest gap first), until budget is exhausted.        ║
   ║      gaps := uncovered_ranges(selected, tree)                  ║
   ║      gap_queue := priority_queue(gaps, key = gap.leaf_count)   ║
   ║      while gap_queue.non_empty():                              ║
   ║          gap     := gap_queue.pop_largest()                    ║
   ║          covering := smallest_node_covering(gap, tree)         ║
   ║          if used + covering.tokens ≤ budget:                   ║
   ║              selected += { covering }                           ║
   ║              used     += covering.tokens                        ║
   ║          else:                                                  ║
   ║              break   // can't afford even this; later gaps      ║
   ║                      // can't be bigger.  (Strict by tree      ║
   ║                      // structure: covering node size grows    ║
   ║                      // monotonically with gap depth.)         ║
   ║                                                                 ║
   ║      // Step 5 — multi-pass refill until convergence.          ║
   ║      // Each pass: add the highest-score non-selected nodes    ║
   ║      // that fit, then re-eliminate redundancy (which may      ║
   ║      // free budget for the next pass).  Converges when an     ║
   ║      // entire pass adds nothing.                              ║
   ║      loop:                                                      ║
   ║          added_any := false                                     ║
   ║          for node in ranked:                                    ║
   ║              if node ∉ selected and                             ║
   ║                 used + node.tokens ≤ budget:                   ║
   ║                  selected += { node }                           ║
   ║                  used     += node.tokens                        ║
   ║                  added_any := true                              ║
   ║          eliminate_redundant(tree, selected, used)             ║
   ║          if not added_any: break                                ║
   ║                                                                 ║
   ║      return selected sorted chronologically                    ║
   ║                                                                 ║
   ║                                                                 ║
   ║  eliminate_redundant(tree, selected, used):                    ║
   ║      // post_order walks Normal sub-leaves, then SummaryOfTurns ║
   ║      // binary leaves, then SummaryOfSummaries internals.       ║
   ║      // Only nodes with children can be redundant.              ║
   ║      for node in tree.post_order():                            ║
   ║          if node ∈ selected and node.has_children():           ║
   ║              if all(covered(c, selected) for c in node.children): ║
   ║                  selected -= { node }                           ║
   ║                  used     -= node.tokens                        ║
   ║                                                                 ║
   ║  covered(node, selected):                                       ║
   ║      // Holds for Normal turns (no children, terminal), for     ║
   ║      // SummaryOfTurns (N Normal children), and for             ║
   ║      // SummaryOfSummaries (2 summary children).                ║
   ║      if node ∈ selected:           return true                  ║
   ║      if not node.has_children():   return false                ║
   ║      return all(covered(c, selected) for c in node.children)   ║
   ╚════════════════════════════════════════════════════════════════╝
```

### 8.4.1 Why this ordering converges

The step order (greedy → eliminate → cover → refill+eliminate-loop) is
not arbitrary:

1. **Eliminating before covering** ensures we don't fill a gap with a
   node that's about to be redundancy-dropped anyway.
2. **Filling largest gap first** prioritises restoring the model's
   broad-strokes view of the conversation over micro-gap completeness;
   if budget runs out, the missed gaps are small.
3. **Multi-pass refill with re-elimination** lets the algorithm settle
   into a stable state: each pass replaces a redundant ancestor with
   one or two more relevant children's siblings, until no node can fit
   without violating budget or redundancy. In practice this converges
   in 2–3 passes (each pass is O(N log N) sort access + O(N)
   elimination, so total cost stays well under the BDP scan's ms-range
   ceiling).

### 8.5 The redundancy rule in pictures

```
   Before redundancy elimination:
   selected = { L1, L2, S(L1+L2), Σ }   where Σ is root, S is internal

                        Σ  ■  (selected — but its whole subtree
                       ╱ ╲      is already covered below)
                      ╱   ╲
                  S  ■     ●
                  ╱ ╲     ╱ ╲
                ╱   ╲    ●   ●
              L1     L2
              ■      ■

   After bottom-up elimination:
   (S covers L1+L2 redundantly; Σ covers the rest only via S, but
    other branches under Σ aren't selected, so Σ stays.)

                        Σ  ■
                       ╱ ╲
                      ╱   ╲
                  S          ●
                  ╱ ╲       ╱ ╲
                ╱   ╲     ●     ●
              L1     L2
              ■      ■
                                          ▲
                                    Σ stays because its right
                                    subtree isn't covered.

   If we then add a leaf under Σ's right subtree that covers it,
   Σ becomes redundant too and gets dropped:

                        Σ
                       ╱ ╲
                      ╱   ╲
                  S          R  ■  (newly added)
                  ╱ ╲       ╱ ╲
                ╱   ╲     ●     ●
              L1     L2
              ■      ■

   And step 4 drops Σ. Step 5 then uses the freed Σ budget to
   pull in another high-score node somewhere.
```

The rule **`if both subtrees are covered, drop the ancestor`** has the
elegant property that the conversation is always covered at *some*
level — but with the *finest* level the score-density argued for.

### 8.6 Default coverage by root

If the greedy fit at step 2 selects nothing in some range AND step 3
can't fit a covering summary for budget reasons, the **root summary**
remains in `selected` and provides the default broad-strokes view of
that range. Root is dropped by step 4 only when *every other range* is
also covered specifically.

In practice the root is almost always selected at step 2 (its
provenance score is generally high — it summarises everything — and
its token cost is one summary turn ≈ 20 tokens), and gets dropped at
step 4 only in the rare case where the conversation is so well-covered
by specific selections that root adds nothing.

### 8.7 Why this composition is "fair"

1. The only inputs are `provenance_score` (the model's own
   attention-driven Q match) and `recency_score` (a hard floor for the
   last 3 + a decaying tie-breaker after). Neither involves any
   clustering algorithm.
2. Every node selected at step 2 was *individually justified by the
   model's Q*. No node is in the set because its parent argued for it.
3. The redundancy rule is content-preserving: dropping a redundant
   parent doesn't lose any coverage, because all the parent's content
   is already covered more specifically.
4. The right-edge decay never *prevents* a high-Q ancient turn from
   winning; it just provides a fallback when nothing ancient is
   especially relevant.

### 8.8 Where this plugs in

`Builder::project()`'s twelve-step pipeline (`project.rs:40–70`)
selects turns to fit the layer's `window` at step 9 via two-phase
flexbox (`reconcile.rs:131`). The score-density selection
**replaces step 9's turn-selection** for any layer whose timeline has
a summary tree. The rest of the pipeline (system sections at step 8,
flexbox across other groups, etc.) is unchanged.

### 8.9 Cost analysis

Per projection:

| Cost                                 | Order              |
|--------------------------------------|--------------------|
| BDP scan of probe Q against corpus   | O(N) bit-popcount  |
|   *(already runs every reprojection)*|                    |
| Sort all nodes by score              | O(N log N)         |
| Greedy fit                           | O(N) scan          |
| Coverage gap analysis                | O(N) tree walk     |
| Redundancy elimination (post-order)  | O(N) tree walk     |
| Refill pass                          | O(N)               |
| Turn-K/V injection into slot         | O(K) metadata      |

N is the total turn count: Normal turns + SummaryOfTurns binary leaves
+ SummaryOfSummaries internals. With an average cluster size c (Normal
turns per binary leaf), `N ≈ N_normal · (1 + 2/c)` — Normal turns
dominate, summaries add a small constant factor on top.  K is the size
of the final selected set (bounded by `layer.window /
avg_summary_tokens`). For `N_normal = 1 M` and c ≈ 10, `N ≈ 1.2 M`;
the dominant cost remains the BDP scan at ~10 ms (which counts all of
N), with the score-density math adding under 1 ms.

---

## 9. Backpressure Semantics

The slot has two top-level regions (per §8.3): **score-density
selection** (the tree-walked content — root + leaves + summaries
picked by `effective_score`, with the last 3 leaves hard-anchored
inside this region) and **pending** (turns not yet absorbed into the
tree). Pending is mandatory; selection is elastic. The three regimes
below show how that elasticity behaves.

### 9.1 Steady state

```
   ┌────────────────────────────────────────────────────────────┐
   │  Turn rate: 1 / 30s    Probe latency: 200ms                 │
   │  pending.len() < 2 at all times                             │
   │                                                              │
   │  Slot composition:                                           │
   │  ┌────────────────────────────────────────────────┬─────┐  │
   │  │ Score-density selection                         │ Pn  │  │
   │  │ (root summary + many leaves + interior          │ 0-2 │  │
   │  │  summaries + last 3 hard-anchored leaves)       │turns│  │
   │  └────────────────────────────────────────────────┴─────┘  │
   │                                                              │
   │  Selection has near-full budget; tree walk explores deeply. │
   └────────────────────────────────────────────────────────────┘
```

### 9.2 Under backpressure

```
   ┌────────────────────────────────────────────────────────────┐
   │  Turn rate: 1 / 1s     Probe latency: 200ms                 │
   │  pending.len() grows to 8+                                  │
   │                                                              │
   │  Slot composition:                                           │
   │  ┌──────────────────────────────┬──────────────────────┐   │
   │  │ Score-density selection       │ Pending (8+ turns)   │   │
   │  │ (shrunken — fewer interior    │                      │   │
   │  │  summaries, fewer leaves      │                      │   │
   │  │  beyond the hard anchor)      │                      │   │
   │  └──────────────────────────────┴──────────────────────┘   │
   │                                                              │
   │  Selection's least-relevant nodes drop out first             │
   │  (lowest effective_score among the previously-fitting set). │
   │  The hard-anchor 3 leaves and the root summary remain.      │
   └────────────────────────────────────────────────────────────┘
```

### 9.3 At saturation

```
   ┌────────────────────────────────────────────────────────────┐
   │  pending.len() consumes >50% of budget                      │
   │                                                              │
   │  Slot composition:                                           │
   │  ┌────────┬───────────────────────────────────────────┐    │
   │  │ root   │ Pending (everything else)                  │    │
   │  │ summary│                                            │    │
   │  └────────┴───────────────────────────────────────────┘    │
   │                                                              │
   │  Selection collapses to just the root (default coverage      │
   │  per §8.6).  The model sees one root summary + the recent   │
   │  pending tail.  Quality degrades gracefully but inference   │
   │  does not stall.                                             │
   └────────────────────────────────────────────────────────────┘
```

Quality recovers automatically once the foreground tempo drops and the
background catches up — pending drains, selection re-expands.

---

## 10. Test Harness

### 10.1 Three-tier test pyramid

Testing the infinite-conversation system has to bridge a five-orders-of-
magnitude latency gap: the *algorithms* (AVL rotations, redundancy
elimination, coverage gap-filling) run in microseconds, but a full
end-to-end turn through the production model is seconds. If every
algorithm change requires a full growth run we burn hours per
iteration. So the harness is a three-tier pyramid, fastest-first:

```
   ╔═════════════════════════════════════════════════════════════╗
   ║                                                              ║
   ║                ┌─────────────────────────────┐               ║
   ║                │ Tier 3 (slow, real)         │               ║
   ║                │ Grow-conversation harness   │               ║
   ║                │ Real model, real probes,    │               ║
   ║                │ debug_id-resumable          │               ║
   ║                │ ~minutes per growth pass    │               ║
   ║                │ 1–10 tests in CI            │               ║
   ║                └─────────────────────────────┘               ║
   ║              ┌─────────────────────────────────┐             ║
   ║              │ Tier 2 (medium, real-ish)       │             ║
   ║              │ Substrate fixture replay        │             ║
   ║              │ Cached-signature redo-log       │             ║
   ║              │ Reloaded by production code     │             ║
   ║              │ ~seconds per test               │             ║
   ║              │ 10–100 tests                    │             ║
   ║              └─────────────────────────────────┘             ║
   ║          ┌─────────────────────────────────────────┐         ║
   ║          │ Tier 1 (fast, synthetic)                │         ║
   ║          │ Pure algorithm unit tests               │         ║
   ║          │ No model, no GPU, no substrate I/O      │         ║
   ║          │ ~microseconds per test                  │         ║
   ║          │ 100–1 000 tests                          │         ║
   ║          └─────────────────────────────────────────┘         ║
   ║                                                              ║
   ╚═════════════════════════════════════════════════════════════╝
```

Each tier has its own fixture format and its own assertion vocabulary.
The same bug surfaces fastest at the lowest tier it's reachable from —
an AVL rotation bug shows up in Tier 1; a probe-quality bug needs
Tier 3.

### 10.2 Tier 1 — Algorithm unit tests (synthetic, no model)

The tree algorithms — `insert`, `rotate_left`, `rotate_right`,
`mark_dirty`, `eliminate_redundant`, `covered`, `uncovered_ranges`,
`select_dense` — are pure data-structure code. They take a tree shape
and a score map as input and return a tree shape (and possibly a
selected set) as output. No model, no GPU, no substrate I/O.

```
   ╔════════════════════════════════════════════════════════════════╗
   ║  Tier 1 fixture format                                         ║
   ╠════════════════════════════════════════════════════════════════╣
   ║                                                                 ║
   ║  struct SyntheticTreeFixture {                                  ║
   ║      // Tree shape, declared as parent/child node IDs           ║
   ║      nodes:    Vec<(NodeId, NodeKind, Option<NodeId>,           ║
   ║                     Option<NodeId>)>,                            ║
   ║      // Score map, declared per node                             ║
   ║      scores:   HashMap<NodeId, f32>,                            ║
   ║      // Token cost per node                                      ║
   ║      tokens:   HashMap<NodeId, u32>,                            ║
   ║                                                                 ║
   ║      // Expected outputs (what the tested algorithm should      ║
   ║      // produce given the inputs)                               ║
   ║      expected_avl_balanced: bool,                               ║
   ║      expected_select_set:   Vec<NodeId>,                        ║
   ║      expected_coverage:     CoverageReport,                      ║
   ║  }                                                               ║
   ║                                                                 ║
   ╚════════════════════════════════════════════════════════════════╝
```

These fixtures live as Rust `const` literals or `yaml!` blocks inline
in the test file — small, hand-crafted, exhaustive of edge cases:

```
   src/tests/tree/
   ├── avl_rotations.rs         — left-left, left-right, right-left,
   │                              right-right rotation cases
   ├── insert.rs                — empty tree, single leaf, fill-and-split
   ├── redundancy.rs            — minimal redundant, deep redundant,
   │                              redundancy chained up to root
   ├── coverage.rs              — single gap, multi-gap, gap at edge,
   │                              gap spanning whole subtree
   ├── select_dense.rs          — score-driven selection, budget edges,
   │                              recency-decay vs provenance tie-break
   └── pathological.rs          — 1 leaf, 1 000 leaves single cluster,
                                  100 leaves alternating, deeply skewed
```

Critically: **Tier 1 has no notion of K/V chunks, no notion of the
substrate's persistence layer, no notion of probes**. Tree nodes are
abstract; scores are arbitrary `f32` values; the test asserts the
algorithm produces the right output. This is where AVL invariants get
fuzz-tested with `proptest` over 10 000 random insertion sequences,
and where `select_dense` gets driven against pathological score
distributions to confirm the redundancy + coverage rules behave.

### 10.3 Tier 2 — Substrate fixtures with cached signatures

Tier 1 tells us the algorithms are right. Tier 2 tells us the
algorithms produce the right outputs when fed **real Q-signature
distributions** from a real model. To get those signatures without
paying the per-test cost of running the model, we cache them.

A **substrate fixture** is a directory:

```
   tests/fixtures/conv-fixture-N/
   ├── .substrate/
   │   ├── substrate.log        — real redo log, produced by Tier 3
   │   ├── manifest.json
   │   └── provenance/
   │       └── *.bin            — Q sign-bits, real model output
   ├── manifest.yaml            — fixture metadata (see below)
   └── README.md                — what this fixture is for
```

The fixture is a frozen artefact built **once** by a Tier 3 build run
(see §10.4), then committed to the test tree. Tier 2 tests load it
read-only:

```rust
let fixture = SubstrateFixture::load("tests/fixtures/conv-fixture-N");
let engine = ConversationEngine::open(fixture.workspace_path())?;
let tree   = engine.conversation().read().summary_tree();
assert!(tree.is_balanced());
assert_eq!(select_dense(&probe_q, &tree, budget),
           fixture.expected_selection());
```

The `manifest.yaml` accompanying each fixture declares:

```yaml
# Fixture metadata
debug_id: small-coherent-50
schema_version: 1
created_by: tier-3 growth at <git-sha>
model: Qwen3-30B-A3B-Q4
n_turns_normal: 50
n_leaves: 12
n_internals: 11
tree_depth: 4

# Planted facts for recall tests
plants:
  - turn:  3
    fact: "The password is rosebud"
    probe: "what was the password?"
  - turn: 27
    fact: "We chose typescript for the frontend"
    probe: "what frontend language?"

# Pre-recorded probe Q vectors for deterministic scoring tests
probes:
  recall_password:
    q_blob: provenance/probe_recall_password.bin
    expected_top:
      - turn 3
      - leaf containing turn 3
  topic_summary:
    q_blob: provenance/probe_topic_summary.bin
    expected_top:
      - root summary
      - leaf 5

# Expected algorithm outputs at this fixture state
expected:
  avl_balanced: true
  no_dirty_nodes: true
  coverage_complete: true
```

This format lets us pin behaviour over time. When we change
`select_dense`, the existing fixtures continue to assert deterministic
output; any regression surfaces immediately as a fixture diff.

#### Fixture lifecycle

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║                                                                ║
   ║                       (Tier 3, slow)                           ║
   ║         ┌──────────────────────────────────────────┐           ║
   ║         │  grow-conversation harness                │           ║
   ║         │  cargo test grow_fixture_N --release      │           ║
   ║         │     • real model, real probes              │           ║
   ║         │     • produces a substrate dir            │           ║
   ║         │     • emits manifest.yaml                  │           ║
   ║         └────────────────┬─────────────────────────┘           ║
   ║                          │                                      ║
   ║                          ▼                                      ║
   ║         ┌──────────────────────────────────────────┐           ║
   ║         │ tests/fixtures/conv-fixture-N/            │           ║
   ║         │ (committed to git, frozen)                │           ║
   ║         └────────────────┬─────────────────────────┘           ║
   ║                          │                                      ║
   ║                          ▼                                      ║
   ║                  (Tier 2, fast)                                 ║
   ║         ┌──────────────────────────────────────────┐           ║
   ║         │  algorithm-against-real-data tests         │           ║
   ║         │  cargo test fixture_replay                │           ║
   ║         │     • load fixture                         │           ║
   ║         │     • run select_dense, AVL ops, etc.     │           ║
   ║         │     • compare to manifest's `expected:`   │           ║
   ║         └──────────────────────────────────────────┘           ║
   ║                                                                ║
   ╚═══════════════════════════════════════════════════════════════╝
```

This is the same pattern that worked for the section-quantize fix:
record a substrate once, expand into it for testing many times.

#### Fixture set (v1)

| Fixture                   | Turns | Purpose                              |
|---------------------------|-------|--------------------------------------|
| `coherent-50`             | 50    | Single-topic, expects flat tree      |
| `two-topics-100`          | 100   | One topic shift mid-stream           |
| `alternating-200`         | 200   | Alternating topics, many boundaries  |
| `drift-300`               | 300   | Gradual topic drift                  |
| `deep-1000`               | 1 000 | Depth stress; tree height ≥ 8        |
| `cold-load-resume-50`     | 50    | Built across two daemon restarts     |
| `dirty-mid-rotate-100`    | 100   | Snapshotted mid-rotation (crash sim) |

### 10.4 Tier 3 — Grow-conversation harness (real, slow)

The end-to-end harness, run with the production model. This is what
the user described in the original §10:

```
   ╔════════════════════════════════════════════════════════════════╗
   ║  grow_conversation(workspace, debug_id, script, n_turns):      ║
   ║                                                                 ║
   ║      engine := load_engine(workspace)                          ║
   ║                                                                 ║
   ║      conv := engine.find_or_create(debug_id) {                 ║
   ║          system_prompt: script.system_prompt,                  ║
   ║          persona:       script.persona,                        ║
   ║          seed:          script.seed,                           ║
   ║      }                                                          ║
   ║                                                                 ║
   ║      for i in 0..n_turns:                                      ║
   ║          user_msg := script.user_message(conv.turn_count, i)   ║
   ║          plants   := script.plants_at(conv.turn_count)         ║
   ║          if plants.non_empty():                                ║
   ║              user_msg = plant_into(user_msg, plants)           ║
   ║          response := conv.send_turn(user_msg)                  ║
   ║          script.observe(i, response)                           ║
   ║                                                                 ║
   ║      engine.wait_summariser_drain()                            ║
   ║                                                                 ║
   ║      // Recall pass: walk through planted facts                ║
   ║      for plant in script.all_plants():                         ║
   ║          probe_msg := plant.probe_text                          ║
   ║          response  := conv.send_turn(probe_msg)                ║
   ║          script.observe_recall(plant, response)                 ║
   ║                                                                 ║
   ║      conv.close()                                              ║
   ║                                                                 ║
   ║      script.assert_properties(conv.turn_count(),               ║
   ║                               conv.tree_shape(),               ║
   ║                               recall_results)                   ║
   ╚════════════════════════════════════════════════════════════════╝
```

#### Growth scripts

The `script` is a small DSL describing the conversation pattern.
Initially Rust-coded fixtures; if useful, a YAML format later. Three
v1 scripts, used to build the seven Tier-2 fixtures listed in §10.3:

| Script             | Pattern                                  | Builds fixtures               |
|--------------------|------------------------------------------|-------------------------------|
| `coherent_lecture` | Long single-topic discussion             | `coherent-50`, `deep-1000`,   |
|                    |                                          | `cold-load-resume-50`         |
| `topic_walk`       | Series of 5-turn topic bursts, distinct  | `two-topics-100`,             |
|                    |                                          | `alternating-200`,            |
|                    |                                          | `dirty-mid-rotate-100`        |
| `drift`            | Slowly-evolving subject; no boundaries   | `drift-300`                   |

Each script:
- Owns the system prompt + persona.
- Generates user-turn text deterministically (seeded RNG over a
  template).
- Plants `(turn_idx, fact, probe_text)` triples at known positions.
- Reports assertions after grow + recall.

#### CI scheduling

Tier 3 runs are `#[ignore]`-d by default — they're hours-long. CI runs
them in two modes:

1. **Per-PR sanity**: run `grow_conversation_smoke` (50 turns,
   `coherent_lecture`) on every PR. ~5 min budget; catches gross
   regressions.
2. **Nightly depth**: grow the four "deep" fixtures by 50 turns each
   from where last night left off; assert at each new depth.
   Persistent workspace shared across nights so depth accumulates.

#### Resume semantics

The harness uses `debug_id` as the substrate-side resume key.
Per-conversation:

```rust
struct ConversationMetadata {
    debug_id: Option<String>,    // new — substrate-side
    created_at: SystemTime,
    // ...existing fields
}
```

Workspace + `debug_id` is the resume tuple. The harness:

```
   1. Open workspace at WORKSPACE_DIR (env var or default path)
   2. engine.lookup_by_debug_id(debug_id):
       - Some(conv) → resume; use existing system prompt + persona
       - None       → create with script's system prompt + persona
   3. Grow by n_turns; persist as usual
   4. (Optional) take a snapshot for Tier 2 fixture creation
```

A nightly job that grows by 50 each night accumulates a real
1 000-turn conversation in 20 nights. By month-end we have a real
50 000-turn conversation if we want one.

### 10.5 The fact-plant + recall protocol

The single most load-bearing property test is **does the tree-walk
recover relevant ancient turns?** The protocol:

```
   t=3:    user: "Important: the password is 'rosebud'.  Anyway, ..."
   t=5..N: filler turns about other topics
   t=N+1:  user: "what was the password?"
           model: "rosebud" → recall PASS
                   else      → recall FAIL
```

Recall scoring:

```
   score(response, fact) := substring-match(extract_answer(response),
                                            fact.canonical_form)
```

The harness records `(plant_depth, recall_pass)` per fixture-run and
emits a recall-curve CSV:

```
   plant_depth  ,  recall_pass  ,  selected_top_node
   3            ,  true         ,  leaf-3
   27           ,  true         ,  leaf-15-summary
   142          ,  false        ,  root-summary
   ...
```

A plant at depth 142 that surfaces only `root-summary` and produces a
recall MISS is the diagnostic signal: the tree walk lost the leaf
between summarisation passes, OR the model lost the fact in the
summary. Both are actionable.

### 10.6 Property checks (used by all three tiers)

A small `properties::` module providing:

| Function                          | Tier(s) used  |
|-----------------------------------|---------------|
| `check_avl_invariants(tree)`      | 1, 2, 3       |
| `check_no_dirty(tree)`            | 1, 2, 3       |
| `check_coverage(tree, leaves)`    | 1, 2, 3       |
| `check_children_exist(tree, sub)` | 2, 3          |
| `recall_score(plants, conv)`      | 3 only        |
| `latency_p50_p99(timing_log)`     | 3 only        |
| `vram_peak(workspace)`            | 3 only        |

Each returns a structured report (not a panic), so the harness can
emit detailed diagnostics on failure.

### 10.7 Layout in the codebase

```
   candle-conversation/
   ├── src/
   │   └── conversation/
   │       └── tree.rs            — tree datatype, AVL ops, redundancy,
   │                                  coverage (Tier 1 surface)
   ├── tests/
   │   ├── tree_algorithm.rs      — Tier 1 (`cargo test --lib tree_*`)
   │   ├── tree_proptest.rs        — Tier 1 fuzz (`proptest`)
   │   ├── fixture_replay.rs       — Tier 2 (`cargo test fixture_*`)
   │   └── fixtures/              — Tier 2 fixture artefacts
   │       ├── coherent-50/
   │       ├── two-topics-100/
   │       └── ...
   zend/
   └── tests/
       ├── infinite_conversation_smoke.rs    — Tier 3 PR-gated
       └── infinite_conversation_deep.rs     — Tier 3 nightly
                                                (`#[ignore]`-d)
```

The Tier 2 fixtures live in-repo as git-committed artefacts. They're
small (typical fixture is < 10 MB — a few KB of manifest + a few MB
of substrate.log + Q sign-bits). When a fixture goes stale (algorithm
change invalidates it), we re-run the Tier 3 builder for that fixture
and commit the new artefact. The manifest's `created_by: git-sha`
field makes drift detectable.

### 10.8 The unbounded-window recall stress test

The single load-bearing claim of this design is: **the model can hold a
conversation many times larger than its context window, and still recall
arbitrary content from anywhere in that history**. This is the test that
validates that claim end-to-end. Tier 3, real model, nightly-scheduled.

#### 10.8.1 Build phase: overflow the window deliberately

```
   ╔════════════════════════════════════════════════════════════════╗
   ║                                                                 ║
   ║   T_window  = layer.window                  e.g. 8 000 tokens   ║
   ║   T_target  = N × T_window                  10× / 100× / …      ║
   ║                                                                 ║
   ║   Build phase grows the conversation to T_target total tokens.  ║
   ║   At T_target = 10 × T_window the conversation cannot possibly  ║
   ║   fit in the model's window — every projection has to elide ≥   ║
   ║   90 % of history.  At 100× / 1 000× the ratio gets brutal.    ║
   ║                                                                 ║
   ║                  ┌────────────────────────────────────────────┐ ║
   ║                  │  T_window  ◄──── one window worth          │ ║
   ║                  └────────────────────────────────────────────┘ ║
   ║   ┌────────────────────────────────────────────────────────────┐║
   ║   │  T_target (10×)                                            │║
   ║   └────────────────────────────────────────────────────────────┘║
   ║                                                                 ║
   ║   The infinite-conversation system has to compress T_target     ║
   ║   into T_window via the tree, every turn, without losing the    ║
   ║   model's ability to find planted content from any depth.       ║
   ║                                                                 ║
   ╚════════════════════════════════════════════════════════════════╝
```

#### 10.8.2 Plant distribution

Facts are planted at **strategically chosen depths**, picked to stress
the tree walk's retrieval at each depth scale:

```
   plant table (T_target = 10 × T_window scenario):

   plant id  │ turn  │ depth band             │ fact pattern
   ──────────┼───────┼───────────────────────┼─────────────────────
   P-near    │  N-5  │ inside hard-anchor 3   │ "color is mauve"
   P-recent  │  N-20 │ recent leaves (decay)  │ "ship date is May 13"
   P-mid     │  N/2  │ middle of timeline     │ "the budget is 50k"
   P-old     │  N/10 │ first 10% of timeline  │ "we chose Postgres"
   P-deep    │  3    │ near turn-zero         │ "password is rosebud"
   P-topic-A │  N/3  │ topic-A cluster        │ "Alice's favourite tea"
   P-topic-B │ 2N/3  │ topic-B cluster        │ "Bob's favourite tea"
   ──────────┴───────┴───────────────────────┴─────────────────────
```

Each plant is a `(turn_idx, fact_text, recall_probe_text)` tuple. The
script embeds `fact_text` into a normal user turn at `turn_idx`, mixed
with filler:

```
   User turn at index `turn_idx`:
   "<filler topic-coherent prose>
    Important: the password is 'rosebud'.
    <more filler>"
```

The probe text is what the test will later ask:
```
   "What did I tell you the password was?"
```

#### 10.8.3 Validation phase: probe + selection inspection

After build, the harness submits each plant's `recall_probe_text` as
a fresh user turn, and asserts **two** things:

1. **The selected set contains the planted turn.**
   This is the algorithm-level assertion: the score-density selection
   actually pulled the right leaf into the slot.
2. **The model's response contains the planted fact.**
   This is the end-to-end assertion: given the planted leaf in its
   slot, the model retrieves the fact and produces it in its answer.

Both failures are diagnostic:

| 1 passes, 2 passes | Recall works end-to-end. ✓                       |
|--------------------|-------------------------------------------------|
| 1 passes, 2 fails  | Selection is right, but the model isn't using   |
|                    | the retrieved content. → model-quality bug      |
|                    | (quantize too aggressive, prompt structure off) |
| 1 fails, 2 passes  | Selection is wrong, but the model recalls the   |
|                    | fact anyway. → either the plant leaked through  |
|                    | another path, or the selection isn't tracked    |
|                    | accurately by the test instrumentation          |
| 1 fails, 2 fails   | Selection is wrong AND model can't recall.      |
|                    | → primary failure mode; the score-density       |
|                    | algorithm or the tree structure missed it       |

This decomposition is why test instrumentation matters: passing
end-to-end with a wrong selection is a *worse* outcome than failing
both, because it masks the bug.

#### 10.8.4 Selection-set instrumentation

The harness needs to know which tree nodes were selected for each
turn's slot. Implementation:

```rust
pub struct TurnResponse {
    pub text: String,
    pub usage: TurnUsage,
    pub selection: SelectionDiagnostics,   // new
}

pub struct SelectionDiagnostics {
    /// TurnKeys of every node included in the slot, in chronological
    /// order — anchor + selected interior + pending.
    pub selected_nodes: Vec<TurnKey>,
    /// Per-node origin: how the node entered the slot.
    pub origin: Vec<SelectionOrigin>,
    /// Effective score that won the node a slot (provenance vs decay).
    pub effective_scores: HashMap<TurnKey, f32>,
}

pub enum SelectionOrigin {
    Pending,                  // not yet absorbed into the tree
    HardAnchor,               // one of the last 3 leaves
    RecencyDecay,             // recency_score won
    ProvenanceScore,          // provenance_score won
    CoverageFill,             // step-3 gap fill added it
    Refill,                   // step-5 refill loop added it
}
```

Populated on every assistant-turn submit, returned in the `TurnResponse`.
Always populated (the diagnostics struct is small, ~K-of-bytes per
turn); no feature gate needed.

#### 10.8.5 Per-probe assertion

For each plant:

```
   ╔════════════════════════════════════════════════════════════════╗
   ║  assert_plant_recall(plant, response):                         ║
   ║                                                                 ║
   ║      // Algorithm-level                                         ║
   ║      planted_or_ancestor :=                                     ║
   ║          response.selection.selected_nodes.iter()              ║
   ║              .any(|key| covers(key, plant.turn_key))           ║
   ║      assert!(planted_or_ancestor,                              ║
   ║              "plant turn not covered by any selected node")    ║
   ║                                                                 ║
   ║      // Soft form: ideally the LEAF (not an ancestor summary)  ║
   ║      // makes it in.  A summary covering the plant means the   ║
   ║      // detail is implicit, not explicit.                      ║
   ║      planted_directly :=                                       ║
   ║          response.selection.selected_nodes.contains(            ║
   ║              &plant.turn_key)                                   ║
   ║      // not asserted hard — recorded as a quality signal        ║
   ║                                                                 ║
   ║      // End-to-end                                              ║
   ║      assert!(response.text.contains_canonical(plant.fact_text),║
   ║              "model's response does not contain planted fact") ║
   ║                                                                 ║
   ║      record_for_csv(                                            ║
   ║          plant.id,                                              ║
   ║          plant.depth_band,                                      ║
   ║          planted_directly,                                      ║
   ║          covered_by: outermost_covering_node(response.selection,║
   ║                                              plant.turn_key),  ║
   ║          recall_pass: true                                      ║
   ║      )                                                          ║
   ╚════════════════════════════════════════════════════════════════╝
```

The recorded CSV per nightly run:

```
   run_id, plant_id, depth_band, planted_directly, covered_by, recall_pass
   2026-06-05, P-deep, 0.3%, true,  leaf-3,             true
   2026-06-05, P-old,  10%,  true,  leaf-100,           true
   2026-06-05, P-mid,  50%,  false, summary-mid-A,      true
   2026-06-05, P-near, 99.5%,true,  anchor-leaf-N-5,    true
```

`planted_directly = false` but `recall_pass = true` is a quality
signal: the model recovered the fact from a summary node, not the
leaf. This is acceptable for `P-mid` and `P-old` (we summarised the
detail and the summary still carried it). It is *not* acceptable for
`P-deep` or `P-near` (those should still be in their leaves at the
test's chosen scale). The signal calibrates the summariser's
fidelity.

#### 10.8.6 Cross-time reference

A subtler test: plant a **dependent pair** of facts.

```
   plant id  │ turn  │ fact pattern                                   │
   ──────────┼───────┼────────────────────────────────────────────────┤
   P-link-1  │ N/4   │ "Alice prefers earl grey"                      │
   P-link-2  │ 3N/4  │ "Same as Alice"  (refers to P-link-1)          │
   ──────────┴───────┴────────────────────────────────────────────────┘

   recall probe at turn N+1:
   "Who prefers earl grey?"

   The model can only answer correctly if BOTH P-link-1 and P-link-2
   are selected (or summarised together).  Tests that the score-
   density walk surfaces the linked pair even though they're at very
   different depths.
```

This is the *bridge* test from §6's probe taxonomy — the case the
score-density rule's "tunnel through summary to anchor" property is
designed to handle. If the walk only surfaces P-link-2 because that's
the nearer turn, the model produces "I don't know who Alice is"; if
both surface, it answers correctly.

#### 10.8.7 Negative test

```
   recall probe at turn N+1:
   "What's the capital of Bolivia?"

   No plant for this exists.  The model should answer the question
   from its own world knowledge OR say it doesn't have the
   information from the conversation.  It should NOT hallucinate a
   conversation-grounded answer ("we discussed this earlier — it's
   X").
```

The harness flags hallucination via heuristic ("we discussed", "as
I mentioned", "earlier we said"); manual review required when
flagged.

#### 10.8.8 Scaling tiers

The same recall-stress script runs at four target scales.  The scales
map directly onto the Tier-3 cadences §10.4 defined for the generic
grow-conversation harness — Smoke ↔ Per-PR sanity, Cruise ↔ Nightly
depth — with Stress and Marathon added for longer-horizon validation:

| Scale       | T_target    | Reach                    | CI cadence    |
|-------------|-------------|--------------------------|---------------|
| Smoke       | 2 × window  | ~few hundred turns       | per PR        |
| Cruise      | 10× window  | thousands of turns       | nightly       |
| Stress      | 100× window | tens of thousands turns  | weekly (manual) |
| Marathon    | 1 000× win  | hundreds of K turns      | quarterly     |

The smoke variant catches gross regressions every PR. Cruise validates
realistic-scale unbounded behaviour. Stress and Marathon are the
long-tail validation that the asymptotic claims hold. All four reuse
the same `debug_id`-resumable substrate so depth accumulates across
runs rather than rebuilding from scratch.

#### 10.8.9 What this test would have caught earlier

If we'd had this test during the section-quantize bisect, the failure
mode "model produces JSON-fragment gibberish when sections are
quantized" would have shown up as a recall-curve cliff at T_target =
2× window: anchor-region plants still passed (recent turns escape the
broken K), but anything requiring the model to read a tree-walked
summary failed. The CSV pattern (`planted_directly = false` and
`recall_pass = false` for every depth-band beyond the anchor) would
have isolated the problem to "tree-walked content is unreadable" in
hours, not days.

This is the lasting value of the harness: future quality regressions
surface as recall-curve shape changes, not as ad-hoc complaints from
the user.

### 10.9 Iteration model

The harness is designed so the **inner development loop** is fast and
the **outer validation loop** is real:

```
   inner loop:                            outer loop:
   ┌──────────────────────┐                ┌────────────────────┐
   │ change algorithm     │                │ change algorithm   │
   │      ▼               │                │      ▼             │
   │ Tier 1 + Tier 2      │                │ + Tier 3 nightly   │
   │ ~seconds total       │                │ ~hours per run     │
   │      ▼               │                │      ▼             │
   │ red? fix.            │                │ red? regenerate    │
   │ green? continue.     │                │       fixture; fix.│
   └──────────────────────┘                └────────────────────┘
       run per save                            run per night
```

Algorithm correctness is bisected at sub-second granularity; real-
model quality is validated overnight.

---

## 11. Implementation Plan

Phased. Each phase ships in a self-contained commit. No phase is
allowed to leave `TODO`s, stubs, or half-built features.

### Phase 1 — Substrate types + tree state

Add `TurnKind`, `children`, `tree_height`, `dirty` to substrate's turn
entry. Extend the redo-log record format to carry the new fields (the
existing record is generic enough that this is additive). Add
`debug_id` to conversation metadata. Re-load logic must round-trip the
new fields byte-perfect.

### Phase 2 — Summary probe

Add the synthetic "summariser" system prompt as a pinned substrate
section. Implement `run_summary_probe` as a scheduler RPC that takes a
list of children `TurnKey`s and a slot configuration, runs the probe,
parses the JSON output, returns `(coherent, summary_text | split_at,
sealed_turn_key)`.

### Phase 3 — Summariser thread

Spawn the thread alongside the persistence thread at engine start. Wire
the trigger to fire after every assistant-turn seal. Implement the
atomic substrate write logic, including the AVL insertion + rotation
path. Implement the dirty-node sweep.

### Phase 4 — Restart reload

Extend the substrate reconstruction pass to populate the tree from
turn records, detect missing summary children, and enqueue regeneration
into the summariser. Test with crash-mid-rotation scenarios.

### Phase 5 — Score-density selection

Implement `select_dense` per §8.4 and wire it into `Builder::project()`'s
step 9 (turn selection) for any layer whose timeline has a summary
tree. Reuse the existing `BdpScanner::scan` results — every tree node
is already scored by the standing scan, so no extra scoring kernel is
needed. Implement `recency_score` (hard anchor for last 3, exponential
decay thereafter) and `covered(node, selected)` for the redundancy
rule.

### Phase 6 — Backpressure + observability

Surface `pending.len()` and `dirty_set.len()` as substrate-side metrics.
Wire the slot composition's elastic boundaries (anchor shrink under
backpressure). Add structured tracing at every boundary so the harness
can assert tree health.

### Phase 7 — Tier 1 algorithm harness

Implement the tree datatype + AVL ops + redundancy / coverage / selection
algorithms in `candle-conversation/src/conversation/tree.rs`, with
unit tests in `candle-conversation/tests/tree_algorithm.rs` and
`proptest`-driven fuzz tests in `tree_proptest.rs`. No model, no
substrate I/O — pure data-structure code. Coverage: AVL invariants,
redundancy correctness, coverage completeness, `select_dense` over
pathological score distributions.

### Phase 8 — Tier 2 substrate fixtures

Implement the `SubstrateFixture` loader (`tests/fixture_replay.rs`),
the manifest schema, and the property checks that compare loaded
fixtures against their declared `expected:` outputs. Initial fixture
set built by hand: `coherent-50` and `two-topics-100`. These exist as
committed artefacts in `candle-conversation/tests/fixtures/`.

### Phase 9 — Tier 3 grow-conversation + debug_id

Implement `grow_conversation` in `zend/tests/infinite_conversation_smoke.rs`
(PR-gated) and `infinite_conversation_deep.rs` (nightly, `#[ignore]`-d).
Implement the substrate-side `debug_id` field and the engine-side
`lookup_by_debug_id`. Add the basic fact-plant + recall protocol from
§10.5. Add the `SelectionDiagnostics` struct on `TurnResponse` (§10.8.4)
so tests can inspect which nodes the score-density selection put in
each slot.

### Phase 10 — Unbounded-window recall stress test (§10.8)

The load-bearing test. Implements the seven-plant distribution
(P-near / P-recent / P-mid / P-old / P-deep / P-topic-A / P-topic-B),
the dual algorithm + end-to-end assertion, the cross-time bridge plant,
the negative-test heuristic, and the per-run CSV emission. Wired at four
scale tiers: smoke (per PR), cruise (nightly), stress (weekly), marathon
(quarterly).

### Phase 11 — Continuous-growth CI

Add the nightly CI job that grows the deep-fixture set + cruise stress
test by 50 turns per night against a persistent workspace, asserting
recall properties at each new depth. Publishes recall-curve CSV per
night so depth-vs-recall regression is visible commit-over-commit.

---

## 12. Open Questions

All design-time questions are resolved. Tracked here for visibility.

### Architecture-level (resolved)

- **Resolved** — Q vs K: Q-against-Q via Hamming agreement on sign-bits
  (§6.3).
- **Resolved** — Summary persistence: a tagged normal turn in the
  substrate, no parallel storage system (§5.3).
- **Resolved** — Boundary detection: combined into the summary probe's
  JSON output (§6.2).
- **Resolved** — Rotation regeneration: lazy, one ancestor per
  background pass (§5.5).
- **Resolved** — Budget plumbing: layer's `window`; score-density
  selection is the new step-9 selector (§8.8).
- **Resolved** — Selection algorithm: score-density with redundancy
  elimination, *not* recursive frontier walk. The existing scan
  already produces per-node scores; we use them directly (§8).

### Parameter-level (resolved)

- **N1 — Score aggregate** — `top_k_mean` from `PerDepthScores`.
  Reasoning: `max` is noisy on short summary turns (one outlier token
  agreement dominates); `mean` dilutes across turn length; `top_k_mean`
  is the robust median between them, letting a strong matching span
  dominate without single-token noise. v1 default; subject to
  harness-measured revision.
- **N2 — Right-anchor + decay** — last **3 leaves** are hard-anchored
  (effective score `+∞`); leaves 4..K score `d^(k-3)` with `d = 0.8`
  default decay. The decay competes with provenance score via
  `effective_score = max(prov, recency)` per §8.2; no hard cut-off,
  the right edge bleeds gracefully into the rest of the tree.
- **N3 — Probe latency cap** — none. Probes are fast (~ms range, batched
  with foreground turns by the scheduler). If GPU contention spikes
  enough to make probes slow, that surfaces as backpressure on
  `pending` (§9), which is already the correct response.
- **N4 — Cascade depth cap** — none. Dirty propagation may cascade to
  root on every insert; the sweep amortises one regen per foreground
  turn, so even worst-case O(log N) cascade catches up in O(log N)
  turns. The architecture absorbs the lag without quality regression
  (stale internal summaries still navigate sanely).
- **N5 — Probe RPC** — reuse `SchedulerRequest::SubmitTurn` with an
  extra metadata field carrying `TurnKind::SummaryOfTurns |
  SummaryOfSummaries`. No new RPC variant. Sampling/decode policy
  matches user-turn defaults; the structured-JSON output is enforced
  by the system prompt + grammar constraint at sampling time, same as
  any structured-output use.

### Score-density opens (resolved)

- **N6 — Coverage-gap minimum unit** — no minimum. Fill gaps
  largest-first until the budget refuses the next covering node. In
  practice score-density gaps are structural (whole topic ranges the
  current Q ignores) rather than micro-gaps, and summary nodes are
  cheap (~20 tokens), so a hard threshold buys nothing. The step-4
  break is the natural stopping rule: by tree structure, covering-node
  size grows monotonically with gap depth, so once one covering node
  is unaffordable, the rest are too.
- **N7 — Step-5 refill termination** — multi-pass until convergence.
  Each pass adds nodes that fit, then re-eliminates redundancy (which
  may free budget for the next pass). Converges when an entire pass
  adds nothing. Empirically 2–3 passes; the cost is bounded by the
  ranked sort (O(N log N)) and stays well under the BDP-scan budget.
