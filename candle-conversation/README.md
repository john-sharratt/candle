# candle-conversation

Turn-based conversation engine for the unbounded-context inference stack: a persistent, provenance-retrieved
substrate projected into a fixed context window every turn, backed by a mandatory on-disk redo log.

## What it does

`candle-conversation` sits above `candle-transformers` / `candle-nn` / `candle-core` / `candle-kernels` and is
the engine `zend` (and any other product built on this fork) constructs conversations on. A single **scheduler
thread** owns every GPU resource — model weights, KV arenas, the batched inference session — and caller threads
never touch the GPU directly: they hold a lightweight `Sequence` handle, submit work as `SchedulerRequest`s over
a channel, and get a `TurnHandle` back to stream or block on the reply. Four subsystems cooperate to make that
turn cheap regardless of how much history exists: the substrate, the projection builder, provenance retrieval,
and mandatory persistence.

```text
 caller thread                       scheduler thread (owns the GPU)
 ┌────────────┐   SchedulerRequest   ┌───────────────────────────────────┐
 │ Sequence   │ ───────────────────► │ Substrate (turns, sections, hot/   │
 │ (timeline) │                      │  warm/cold residence)              │
 └────────────┘ ◄─────────────────── │ Builder::project(...)              │
      TurnHandle (stream/block)      │   → BDP provenance scan            │
                                     │   → budget reconciliation          │
                                     │ SubstratePersistence (redo log)     │
                                     └───────────────────────────────────┘
```

### The substrate: timelines, turns, sections, collections

The **substrate** (`src/substrate.rs`, type `Substrate`) is a workspace-shared, unbounded store of every turn and
system-prompt section that has ever existed in a workspace, plus the single source of truth for *where* each
turn's K/V bytes currently live: hot (GPU VRAM), warm (CPU RAM), or cold (the redo log on NVMe). Storage is keyed
by `(GroupId, TurnIndex)`, with insertion order tracked separately so selection rules that care about recency can
walk turns in append order. A turn can be resident in more than one tier simultaneously — a warm→hot promotion
keeps its warm copy so the next eviction is free.

A **timeline** (`TimelineId`, a microsecond-timestamp-derived id minted by `TimelineAllocator`) is one
conversation's turn sequence within the substrate — a `Sequence` corresponds to exactly one timeline, but many
timelines (one live dialogue, one per ingested file, one per repo-map cluster, …) share the same substrate and
the same GPU.

A **turn** (`turn::Turn`/`TurnId`, and `projection::TurnIndex`/`TurnKey` for the group-scoped projection view) is
one user/assistant exchange; its persisted K/V shape is exactly `[user_msg][user_end][assistant_start][response]`
— the cross-turn `user_start`/`assistant_end` boundary markers are *not* baked into the seal, they're re-emitted
live by the scheduler's projection assembler at every projection so their K vectors reflect the actual runtime
causal prefix rather than a stale one.

A **section** (`SectionId`) is a piece of the system prompt — static framing text or a schema-declared dynamic
item — pinned into the substrate once and re-injected by every projection that selects it. A **collection**
(`SectionCollection`) is a schema-declared group of sections that ingest and are selected together; the tool
catalog is the canonical example — one section per tool, with the collection's `top_k` selection rule picking
which tools actually surface in a given turn's prompt. `SectionTree` extends this to branch-selected content
(mood, identity, response-style variants), sealing every option/variant combination up front so switching a
selector at projection time is free rather than requiring a re-prefill.

### Projection: turning an unbounded substrate into a fixed window

The **projection builder** (`src/projection/`, type `Builder`) is a pure structural reconciler — it owns no
content, no tokenizer, and no scoring mechanism. A YAML `Schema` declares, once, a set of ordered **layers**
(each with its own system prompt, window size, and score threshold), each layer's ordered **groups** (each with
a `SelectionRule` and a budget), and the shared system-prompt's sections/collections/section-trees. Calling
`Builder::project(ProjectionTarget { layer, group, timeline }, &resolver)` runs a 12-step pipeline:

1. mask everything outside the target's visibility
2. score every visible turn via the resolver, then apply group score thresholds
3. run each group's selection rule under unbounded budget, then aggregate turn scores into a group score
4. apply layer score thresholds and filter empty groups / empty layers
5. emit the target layer's own system-prompt sections
6. reconcile the remaining turn budget across layers, then groups, CSS-flexbox style (priority-weighted shares
   clamped to min/max, released budget redistributed across iterations)
7. re-run selection per group under its allocated budget, repeating until convergence or `MAX_ITERATIONS`
8. emit sections in declaration order, then layers in declaration order with groups sorted by *ascending* score
   — the highest-relevance content lands closest to the attention sink

Selection rules are a closed set — `AlwaysVisible`, `TopK`, `Single`, and the composite `Conversation { recent,
historical_top_k }` used by the live dialogue group — and score formulas (`Max`, `Sum`, `Mean`, `TopKMean`,
`Count`) turn a group's turn scores into the single number layers are ranked by.

### Provenance-driven selection

Everything dynamic in a projection — which turns exist, how many tokens they cost, how relevant they are *right
now* — flows through the `ContentResolver` trait, which `Substrate` implements. Relevance scores come from
**provenance retrieval** (`src/provenance/`): at turn-seal time, each real token's decode-time `sign(Q)` bits
(from the R16 KV band) are folded via `fold_provenance` into a locked 1536-bit `WideQSig`, stored per turn on the
substrate. Every reprojection extracts the live decode window's own folded signature as a probe and scans it
against a tag-scoped gallery of past turns' signatures (`score_provenance_late_fusion`): per query token, per
folded layer-group, a Hamming/XNOR-popcount agreement is computed against every gallery candidate, and a
z-score×margin-weighted vote — restricted to the top 25% of query tokens by vote magnitude (the "needle gate",
discarding diffuse haystack signal) — produces each candidate's belief score. `ToolBelief` (`belief.rs`)
accumulates these scores turn-to-turn with a leaky decay, and `SectionSelector`/`SectionPolicy` (`selection.rs`)
apply hysteresis (separate admit/evict thresholds) so selection doesn't flap turn-to-turn. The gallery itself is
scanned CPU-side (`scan.rs` / `packed.rs`) or, for larger corpora, on the GPU via a paged VRAM-resident arena
(`gallery_arena/`) that keeps signatures in fixed 6 KiB pages inside 16 MiB slabs, avoiding a pinned-memory
re-upload every wave.

### The scheduler: waves and admission

The scheduler (`src/scheduler/`) runs a **continuous fair-wave loop**, not an alternating prefill/decode phase
machine: a decode cursor sweeps every layer once per wave, driving foreground token generation forward, while a
prefill/section-ingest cursor creeps through the same layers at a throttled rate governed by a per-layer
`decode_priority` (Low / Normal / High ⇒ roughly 1 / 16 / 64 decode tokens per completed prefill step). Wherever
the two cursors land on the same layer in the same wave, their tokens co-batch through that layer's MoE expert
GEMM, so a large prefill never starves decode of its working set and decode never goes cold behind a big ingest.

`SchedulerRequest` is the full caller-facing surface: `NewSequence` / `ResumeSequence` / `FreeSequence` for slot
lifecycle, `SubmitTurn` (the full turn lifecycle — project, prefill, decode to EOS/max-tokens, seal) with an
optional `ReprojectionPolicy` for continuous mid-decode reprojection, `IngestSection` / `RestoreSection` for
one-shot system-prompt section prefill, and diagnostic/maintenance variants (`PrimingProjection`,
`ProbeWideSigs`, `ReconstructSubstrate`, `Shutdown`, …). Prefill **admission** is governed by an AIMD controller
(`shrink_admit_window` / `grow_admit_window`) with an evidence-based escape hatch (`evidence_admit_grow`) that
reopens the admission window after enough consecutive OOM-free ticks even under a card whose steady-state VRAM
reading always looks "pressured". Per-wave phase timings (`Decode` / `Prefill` / `Section` / `Projection` /
`Sealing` / `Eviction` / `Allocation` / `Blocked` / `Idle` / `Sync`) accumulate into a process-global ring buffer
(`phase_ring.rs`) that `zend`'s HTTP layer reads directly for the live performance dashboard.

### Persistence: the mandatory redo log

There is **no in-memory-only mode**. `ConversationEngine::new` unconditionally opens a `SubstratePersistence` at
`<workspace>/.substrate/` and replays it into a `Substrate` in one walker pass before serving any turn. The store
is an append-only, content-addressed, **segmented** redo log — many sealed `seg-*.log` files plus one
`seg-*.active`, rotated at a size target — that is a complete, self-contained image of the substrate: every K/V
chunk, token blob, projection-event log, wide-Q signature, label, and turn-metadata record needed to reconstruct
state lives in the log, so a fresh process can rebuild the entire conversation history (and its provenance
gallery) from nothing but this directory. Recovery uses a `HeaderIndex` hash-chain fast path with a forward-walk
fallback for any un-indexed tail. A background `PersistenceThread` drives hot→warm migration
(`migrate_group_hot_to_warm`) and warm→cold writes, and triggers compaction once a log's dead-byte ratio crosses
50% (on logs at least 64 MiB) by rewriting only the live records into a fresh segment.

## Key modules / layout

| Path | Role |
|---|---|
| `src/lib.rs` | Crate root; re-exports the public surface |
| `src/engine.rs` | `ConversationEngine` — spawns the scheduler thread, owns the workspace `Conversation` handle, the persistence thread, the summariser thread |
| `src/conversation.rs` | `Sequence` — the caller-side conversation handle: turn submission, system-prompt section ingestion into the substrate |
| `src/substrate.rs` | `Substrate` — turn/section store + hot/warm/cold KV residence tracking (`SequenceResidence`, `hot_lru`/`warm_lru`) |
| `src/projection/` | `Schema`, `Builder`, the 12-step projection pipeline, budget reconciliation (`reconcile.rs`), score formulas (`score.rs`), selection rules (`selection.rs`) |
| `src/scheduler/` | `Scheduler`, `SchedulerRequest`, the fair-wave loop, admission control, `phase_ring.rs` telemetry ring |
| `src/provenance/` | `WideQSig`, the BDP scan (`scan.rs`, `packed.rs`, `gpu.rs`), the paged VRAM gallery (`gallery_arena/`), belief accumulation (`belief.rs`), hysteresis selection (`selection.rs`) |
| `src/persistence/` | `SubstratePersistence`, the segmented redo log, compaction, recovery, hot↔warm↔cold migration (`elevate.rs`, `thread.rs`) |
| `src/prompts/` | Compile-time-embedded (`include_str!`) system prompts — treated as code because they're baked into a pinned KV cache |
| `src/tree/` | `ConversationTree` — the in-memory per-`Sequence` turn history (system prompt + paired exchanges + temporal markers) |
| `src/summary_tree/` | The immutable, append-only 8-ary Merkle-Mountain-Range summary forest built over turns |
| `src/stencil/` | Constrained decoding (tool-call JSON shape, `<think>` steering trees) |
| `src/models/` | Chat dialect definitions (role markers, glue tokens) |
| `src/narrator/` | Text-to-structured-input conversion for the tree-generation pipeline |
| `src/normalization/` | Per-scope provenance score normalization — asymmetric-EWMA hit levels (`hit_level.rs`) cached per `ScopeKey` (`cache.rs`, `scope.rs`) |
| `src/turn_layout.rs`, `turn.rs` | Turn boundary/token-range bookkeeping and the `Role`/`Turn`/`TurnOptions` API |

## Key types & entry points

- `ConversationEngine::new(model, tokenizer, config)` — the entry point. Builds the batched inference session,
  opens persistence, spawns the persistence/summariser/scheduler threads.
- `ConversationEngine::new_conversation` / `new_conversation_with_projection[_progress]` — mint a fresh
  `TimelineId` and return a `Sequence`.
- `Sequence::submit_turn` / `submit_turn_with_options` → `TurnHandle` — submit a user turn, stream or block on
  the response; `Sequence::insert_section` / `insert_section_collection` ingest system-prompt content into the
  substrate.
- `projection::Builder::from_yaml_with_vars` / `Builder::project(ProjectionTarget, &resolver)` — construct a
  schema and run one projection; `ProjectionTarget { layer, group, timeline }` selects which layer's window and
  framing apply.
- `substrate::Substrate` / `projection::Conversation` (`Arc<RwLock<Substrate>>` + persistence) — the
  workspace-shared handle every `Sequence` clones.
- `provenance::WideQSig`, `fold_provenance`, `score_provenance_late_fusion`, `GalleryArena` — the retrieval
  fingerprint and scan.
- `persistence::SubstratePersistence::open_in_with_substrate` — open (or create) `.substrate/` and fully replay
  it into a `Substrate`.

## How it is used

`zend` (the Zen Code daemon) is the primary consumer: it constructs one `ConversationEngine`, ingests its
workspace into several projection layers (`repo_map`, `code_reading`, live dialogue), and drives turns through
`Sequence` per HTTP request. Battle Cities NPCs are expected to use the same engine with a different `Schema`
(per-agent layers instead of per-file/per-directory ones).

The crate is **CUDA-only** — there is no non-CUDA Cargo feature; the substrate, background quantizer, cold-load
bridge, and scheduler all assume a CUDA device. Feature flags:

- `hub` — enables `hf-hub` model download (required by the `chat`, `tree_gen`, `ruler_stream` examples).
- `profile` — zero-cost scoped timing of the projection/reproject hot path (`scheduler::profile`), forwarding to
  `candle-transformers/profile` for per-op breakdowns.
- `kv-zero-check` — debug-only scan for all-zero (dead) K/V tokens at every gather/seal/inject boundary.
- `context-dump` — trace-level dump of every slot's full token stream at turn completion.
- `test-helpers` — exposes normally-hidden test-introspection methods (warm pool state, tiered store).

Examples live under `examples/`: `chat` (interactive CLI), `tree_gen` (offline conversation-tree corpus
generation), `ruler_stream`, `substrate_inspect` (redo-log/manifest dump tool, no `hub` needed),
`gpu_belief_parity` (BDP scan CPU/GPU parity harness).

## Related docs

- `docs/conversation_builder.md` — the projection engine's design: schema, selection rules, budget reconciliation
  algorithm, masking semantics.
- `docs/attention_provenance.md` — the research paper motivating provenance-selected attention; the *shipped*
  mechanism (binary `sign(Q)` Hamming/XNOR-popcount voting over `WideQSig`) is a leaner, production-hardened
  descendant of the float INT8 dot-product design this paper describes.
- `docs/immutable_summary_forest.md` — the append-only Merkle-Mountain-Range summary tree (`src/summary_tree/`).
- `docs/continuous_fair_waves.md` — the current scheduler design: decoupled decode/prefill cursors sharing
  per-layer MoE batches.
- `docs/kv_tier_migration.md` — the three-tier hot/warm/cold KV migration design; two divergences from it remain
  (warm tier is pageable, not pinned; hot→warm runs on the primary CUDA stream).
- `docs/provenance_score_normalization.md` — normalizing BDP scan scores across candidates of differing
  "loudness"; implemented in `src/normalization/`.
- `docs/paged_gallery_arena.md` — the VRAM-resident provenance gallery (`src/provenance/gallery_arena/`) that
  removed the ~10 ms/scan pinned-upload cost.
